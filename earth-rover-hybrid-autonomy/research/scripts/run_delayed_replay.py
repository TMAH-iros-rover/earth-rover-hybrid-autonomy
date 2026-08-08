#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from statistics import fmean

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from earth_rover.control.command_filter import CommandFilter
from earth_rover.control.hybrid_controller import HybridReactiveController
from earth_rover.core.types import CandidateDirection, ControlCommand, FrameData, RoverData
from earth_rover.navigation.gps_utils import bearing_deg, normalize_angle_deg
from earth_rover.navigation.heading_filter import HeadingFilter
from earth_rover.navigation.waypoint_manager import WaypointManager
from earth_rover.perception.dummy_traversability import DummyTraversabilityModel
from earth_rover.planning.candidate_planner import CandidateDirectionPlanner
from earth_rover.safety.emergency_stop import EmergencyStopMonitor
from earth_rover.safety.recovery import RecoveryController
from earth_rover.safety.stuck_detector import StuckDetector
from earth_rover.utils.config import load_config
from earth_rover.utils.math_utils import safe_float
from earth_rover.utils.status import format_urban_status


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay an Urban run with delayed sensor/frame packets.")
    parser.add_argument("run_dir")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--override", default="configs/urban_latency_2s.yaml")
    parser.add_argument("--delay-sec", type=float, default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_config(ROOT / args.config, ROOT / args.override)
    run_dir = Path(args.run_dir)
    rows = load_timeline(run_dir)
    if not rows:
        raise SystemExit(f"No timeline rows found in {run_dir / 'timeline.csv'}")

    latency_cfg = config.get("latency", {})
    delay_sec = float(
        args.delay_sec
        if args.delay_sec is not None
        else latency_cfg.get("sensor_delay_sec", latency_cfg.get("assumed_latency_sec", 2.0))
    )
    frame_delay_sec = float(latency_cfg.get("frame_delay_sec", delay_sec))
    data_delay_sec = float(latency_cfg.get("data_delay_sec", delay_sec))

    output_dir = Path(args.output_dir) if args.output_dir else Path("logs") / time.strftime("replay_%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    perception = DummyTraversabilityModel(config)
    waypoint_manager = WaypointManager(extract_checkpoints(rows), config["urban"]["waypoint_switch_radius_m"])
    heading_filter = HeadingFilter(alpha=0.5)
    planner = CandidateDirectionPlanner(config)
    controller = HybridReactiveController(config)
    command_filter = CommandFilter(config)
    emergency_monitor = EmergencyStopMonitor(config)
    stuck_detector = StuckDetector(config)
    recovery = RecoveryController(config)

    status_cfg = config.get("status_output", {})
    status_enabled = bool(status_cfg.get("enabled", True))
    status_interval_sec = float(status_cfg.get("interval_sec", 1.0))
    last_status_print = 0.0
    last_command = ControlCommand(0.0, 0.0)
    last_replay_time = replay_time(rows[0])
    metrics = ReplayMetrics(delay_sec=delay_sec)

    commands_path = output_dir / "replay_commands.csv"
    with commands_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "replay_time",
                "frame_source_time",
                "data_source_time",
                "target_checkpoint_sequence",
                "distance_to_checkpoint_m",
                "heading_error_deg",
                "raw_linear",
                "raw_angular",
                "safe_linear",
                "safe_angular",
                "mode",
                "candidate_name",
                "emergency_reason",
                "stuck",
                "recovery_state",
                "data_age_sec",
                "frame_age_sec",
            ],
        )
        writer.writeheader()

        for row in rows:
            now = replay_time(row)
            dt = max(1e-3, now - last_replay_time)
            last_replay_time = now
            frame_row = row_at_or_before(rows, now - frame_delay_sec)
            data_row = row_at_or_before(rows, now - data_delay_sec)
            if frame_row is None or data_row is None:
                metrics.num_invalid_packets += 1
                continue

            frame = frame_from_row(run_dir, frame_row)
            data = data_from_row(data_row)
            if frame is None:
                metrics.num_invalid_packets += 1
                continue

            gps_valid = data.latitude is not None and data.longitude is not None and data.orientation is not None
            waypoint_state = waypoint_manager.update(data.latitude, data.longitude)
            target = waypoint_state["target"]
            target_sequence = target.get("sequence") if isinstance(target, dict) else None
            target_bearing = None
            heading_error_deg = 0.0
            if target is not None and gps_valid:
                target_lat = target.get("latitude", target.get("lat"))
                target_lon = target.get("longitude", target.get("lon"))
                filtered_heading = heading_filter.update_deg(data.orientation)
                target_bearing = bearing_deg(data.latitude, data.longitude, target_lat, target_lon)
                heading_error_deg = normalize_angle_deg(target_bearing - filtered_heading)
            heading_error_rad = math.radians(heading_error_deg)

            perception_result = perception.infer(frame)
            stuck, _stuck_debug = stuck_detector.update(data, last_command)
            if stuck:
                recovery.trigger()
            recovery_command = recovery.update(now)
            perception_valid = all(
                math.isfinite(value)
                for value in [
                    perception_result.left_free_score,
                    perception_result.center_free_score,
                    perception_result.right_free_score,
                    perception_result.obstacle_confidence,
                    perception_result.traversability_confidence,
                ]
            )
            data_age = max(0.0, now - replay_time(data_row))
            frame_age = max(0.0, now - replay_time(frame_row))
            emergency, emergency_reason = emergency_monitor.update(
                now=now,
                last_frame_time=frame.timestamp,
                last_data_time=data.timestamp,
                sdk_failure_count=0,
                loop_delay_sec=dt,
                gps_valid=gps_valid,
                perception_valid=perception_valid,
            )

            if target is None:
                candidate = CandidateDirection("STOP", 0.0, 0.0, 0.0, 0.0, 0.0)
                raw = ControlCommand(0.0, 0.0, mode="MISSION_COMPLETE")
            else:
                candidate = planner.select(heading_error_rad, perception_result, force_stop=emergency)
                raw = controller.compute(heading_error_rad, candidate, perception_result, emergency, recovery_command, dt)

            command = command_filter.apply(raw, dt, frame_is_stale=False, data_is_stale=False)
            if waypoint_state["reached"]:
                waypoint_manager.mark_current_reported()

            writer.writerow(
                {
                    "replay_time": now,
                    "frame_source_time": replay_time(frame_row),
                    "data_source_time": replay_time(data_row),
                    "target_checkpoint_sequence": target_sequence,
                    "distance_to_checkpoint_m": waypoint_state.get("distance_m"),
                    "heading_error_deg": heading_error_deg,
                    "raw_linear": raw.linear,
                    "raw_angular": raw.angular,
                    "safe_linear": command.linear,
                    "safe_angular": command.angular,
                    "mode": command.mode,
                    "candidate_name": candidate.name,
                    "emergency_reason": emergency_reason,
                    "stuck": stuck,
                    "recovery_state": recovery.state,
                    "data_age_sec": data_age,
                    "frame_age_sec": frame_age,
                }
            )
            metrics.update(heading_error_deg, command, emergency, stuck, recovery.state, data_age, frame_age)
            last_command = command

            if status_enabled and now - last_status_print >= status_interval_sec:
                last_status_print = now
                print(
                    format_urban_status(
                        {
                            "mode": command.mode,
                            "target_checkpoint_sequence": target_sequence,
                            "distance_to_checkpoint_m": waypoint_state.get("distance_m"),
                            "target_bearing_deg": target_bearing,
                            "current_heading_deg": data.orientation,
                            "heading_error_deg": heading_error_deg,
                            "raw_linear": raw.linear,
                            "raw_angular": raw.angular,
                            "safe_linear": command.linear,
                            "safe_angular": command.angular,
                            "gps_signal": data.gps_signal,
                            "signal_level": data.signal_level,
                            "data_age_sec": data_age,
                            "frame_age_sec": frame_age,
                            "stuck_state": stuck,
                            "recovery_state": recovery.state,
                            "log_dir": output_dir,
                        }
                    ),
                    flush=True,
                )

    summary = metrics.summary()
    summary["source_run_dir"] = str(run_dir)
    summary["output_dir"] = str(output_dir)
    with (output_dir / "replay_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"delayed replay complete: {output_dir}")
    return 0


class ReplayMetrics:
    def __init__(self, delay_sec: float):
        self.delay_sec = delay_sec
        self.heading_errors = []
        self.angular_commands = []
        self.data_ages = []
        self.frame_ages = []
        self.num_steps = 0
        self.num_emergency_stops = 0
        self.num_stuck_events = 0
        self.num_recovery_events = 0
        self.num_invalid_packets = 0

    def update(
        self,
        heading_error_deg: float,
        command: ControlCommand,
        emergency: bool,
        stuck: bool,
        recovery_state: str,
        data_age: float,
        frame_age: float,
    ) -> None:
        self.num_steps += 1
        self.heading_errors.append(abs(float(heading_error_deg)))
        self.angular_commands.append(abs(float(command.angular)))
        self.data_ages.append(data_age)
        self.frame_ages.append(frame_age)
        if emergency or command.mode == "EMERGENCY_STOP":
            self.num_emergency_stops += 1
        if stuck:
            self.num_stuck_events += 1
        if recovery_state not in {"IDLE", "RETRY"}:
            self.num_recovery_events += 1

    def summary(self) -> dict:
        return {
            "delay_sec": self.delay_sec,
            "num_steps": self.num_steps,
            "mean_data_age_sec": mean_or_zero(self.data_ages),
            "mean_frame_age_sec": mean_or_zero(self.frame_ages),
            "mean_abs_heading_error_deg": mean_or_zero(self.heading_errors),
            "max_abs_heading_error_deg": max(self.heading_errors) if self.heading_errors else 0.0,
            "mean_abs_angular_command": mean_or_zero(self.angular_commands),
            "num_emergency_stops": self.num_emergency_stops,
            "num_stuck_events": self.num_stuck_events,
            "num_recovery_events": self.num_recovery_events,
            "num_invalid_packets": self.num_invalid_packets,
        }


def load_timeline(run_dir: Path) -> list[dict]:
    with (run_dir / "timeline.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return sorted(rows, key=replay_time)


def replay_time(row: dict) -> float:
    return safe_float(row.get("timestamp"), 0.0)


def row_at_or_before(rows: list[dict], target_time: float) -> dict | None:
    selected = None
    for row in rows:
        if replay_time(row) <= target_time:
            selected = row
        else:
            break
    return selected


def frame_from_row(run_dir: Path, row: dict) -> FrameData | None:
    rel_path = row.get("front_frame_path")
    if not rel_path:
        return None
    image = cv2.imread(str(run_dir / rel_path))
    if image is None:
        return None
    return FrameData(
        timestamp=safe_float(row.get("frame_timestamp"), replay_time(row)),
        image=image,
        source="delayed_replay",
        sdk_timestamp=safe_float(row.get("frame_sdk_timestamp")),
    )


def data_from_row(row: dict) -> RoverData:
    return RoverData(
        timestamp=safe_float(row.get("data_timestamp"), replay_time(row)),
        latitude=safe_float(row.get("latitude")),
        longitude=safe_float(row.get("longitude")),
        orientation=safe_float(row.get("orientation")),
        speed=safe_float(row.get("speed")),
        rpms=parse_rpms(row.get("rpms")),
        battery=safe_float(row.get("battery")),
        signal_level=safe_float(row.get("signal_level")),
        gps_signal=safe_float(row.get("gps_signal")),
        raw=row,
        sdk_timestamp=safe_float(row.get("data_sdk_timestamp")),
    )


def parse_rpms(value: str | None) -> list[float] | None:
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list):
        return None
    return [float(item) for item in parsed if safe_float(item) is not None]


def extract_checkpoints(rows: list[dict]) -> list[dict]:
    checkpoints = {}
    for row in rows:
        raw = row.get("target_checkpoint")
        if not raw:
            continue
        try:
            checkpoint = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(checkpoint, dict):
            continue
        sequence = checkpoint.get("sequence", len(checkpoints) + 1)
        checkpoints[str(sequence)] = checkpoint
    return list(checkpoints.values())


def mean_or_zero(values: list[float]) -> float:
    return fmean(values) if values else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
