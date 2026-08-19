#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from earth_rover.control.command_filter import CommandFilter
from earth_rover.control.hybrid_controller import HybridReactiveController
from earth_rover.control.latency_compensator import LatencyCompensator
from earth_rover.core.types import CandidateDirection, ControlCommand
from earth_rover.logger import RunLogger
from earth_rover.navigation.gps_utils import bearing_deg, normalize_angle_deg
from earth_rover.navigation.heading_filter import HeadingFilter
from earth_rover.navigation.waypoint_manager import WaypointManager
from earth_rover.perception.dummy_traversability import DummyTraversabilityModel
from earth_rover.planning.candidate_planner import CandidateDirectionPlanner
from earth_rover.safety.emergency_stop import EmergencyStopMonitor
from earth_rover.safety.recovery import RecoveryController
from earth_rover.safety.stuck_detector import StuckDetector
from earth_rover.sdk_client import EarthRoverSDKClient
from earth_rover.utils.config import load_config
from earth_rover.utils.status import format_urban_status
from earth_rover.utils.timing import sleep_to_maintain_loop_hz


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--override", default="research/configs/urban.yaml")
    args = parser.parse_args()

    config = load_config(ROOT / args.config, ROOT / args.override)
    sdk_cfg = config["sdk"]
    sdk = EarthRoverSDKClient(sdk_cfg["base_url"], float(sdk_cfg["request_timeout_sec"]))
    logger = RunLogger(config)
    perception = DummyTraversabilityModel(config)
    logger.log_event("MISSION_START", {})
    try:
        sdk.start_mission()
    except Exception as exc:
        logger.log_event("MISSION_START_FAILED", {"error": str(exc)})

    try:
        checkpoint_state = sdk.get_checkpoint_state()
        checkpoints = checkpoint_state["checkpoints"]
        latest_scanned_checkpoint = int(checkpoint_state["latest_scanned_checkpoint"] or 0)
    except Exception as exc:
        checkpoints = []
        latest_scanned_checkpoint = 0
        logger.log_event("CHECKPOINT_FETCH_FAILED", {"error": str(exc)})
    waypoint_manager = WaypointManager(
        checkpoints,
        config["urban"]["waypoint_switch_radius_m"],
        latest_scanned_checkpoint=latest_scanned_checkpoint,
    )
    heading_filter = HeadingFilter(alpha=0.5)
    candidate_planner = CandidateDirectionPlanner(config)
    latency_comp = LatencyCompensator(config)
    controller = HybridReactiveController(config)
    command_filter = CommandFilter(config)
    emergency_monitor = EmergencyStopMonitor(config)
    stuck_detector = StuckDetector(config)
    recovery = RecoveryController(config)

    last_command = ControlCommand(0.0, 0.0)
    last_loop = time.monotonic()
    last_frame_time = None
    last_data_time = None
    sdk_failure_count = 0
    status_cfg = config.get("status_output", {})
    status_enabled = bool(status_cfg.get("enabled", True))
    status_interval_sec = float(status_cfg.get("interval_sec", 1.0))
    last_status_print = 0.0
    recovery_logged_for_attempt = 0

    try:
        while True:
            loop_start = time.monotonic()
            wall_now = time.time()
            dt = max(1e-3, loop_start - last_loop)
            last_loop = loop_start

            try:
                frame = sdk.get_front_frame()
                data = sdk.get_data()
                last_frame_time = frame.timestamp
                last_data_time = data.timestamp
                sdk_failure_count = 0
            except Exception as exc:
                sdk_failure_count += 1
                raw_cmd = ControlCommand(0.0, 0.0, mode="SDK_FAILURE_STOP")
                cmd = command_filter.apply(raw_cmd, dt, True, True)
                try:
                    sdk.send_control(cmd)
                except Exception:
                    pass
                logger.log_event("SDK_FAILURE", {"error": str(exc), "count": sdk_failure_count})
                sleep_to_maintain_loop_hz(config["project"]["loop_hz"], loop_start)
                continue

            perception_result = perception.infer(frame)
            gps_valid = data.latitude is not None and data.longitude is not None and data.orientation is not None
            waypoint_state = waypoint_manager.update(data.latitude, data.longitude)
            target = waypoint_state["target"]
            heading_error_deg = 0.0
            target_bearing = None
            target_sequence = target.get("sequence") if isinstance(target, dict) else None
            signal_low = data.signal_level is not None and data.signal_level < 0.2

            if target is not None and gps_valid:
                target_lat = target.get("latitude", target.get("lat"))
                target_lon = target.get("longitude", target.get("lon"))
                filtered_heading = heading_filter.update_deg(data.orientation)
                stuck_for_prediction = False
                predicted_heading = latency_comp.predict_heading_deg(
                    filtered_heading,
                    last_command.angular,
                    is_stuck=stuck_for_prediction,
                    signal_low=signal_low,
                )
                target_bearing = bearing_deg(data.latitude, data.longitude, target_lat, target_lon)
                heading_error_deg = normalize_angle_deg(target_bearing - predicted_heading)
            heading_error_rad = math.radians(heading_error_deg)

            stuck, stuck_debug = stuck_detector.update(data, last_command)
            if stuck:
                recovery.trigger()
                logger.log_event("STUCK_DETECTED", stuck_debug)
                if recovery.attempts > recovery_logged_for_attempt:
                    recovery_logged_for_attempt = recovery.attempts
                    rear_frame = None
                    try:
                        rear_frame = sdk.get_rear_frame()
                    except Exception as exc:
                        logger.log_event("RECOVERY_REAR_FRAME_FAILED", {"error": str(exc), "attempt": recovery.attempts})
                    logger.log_recovery_event(
                        {
                            "recovery_id": recovery.attempts,
                            "start_timestamp": time.time(),
                            "reason": "stuck",
                            "mode_before": last_command.mode,
                            "target_checkpoint_sequence": target_sequence,
                            "distance_to_checkpoint_m": waypoint_state.get("distance_m"),
                            "data_timestamp": data.timestamp,
                            "data_sdk_timestamp": data.sdk_timestamp,
                            "linear_before": last_command.linear,
                            "angular_before": last_command.angular,
                            "recovery_action": "reverse_then_rotate",
                            "result": "pending",
                            "stuck_debug": stuck_debug,
                        },
                        front_frame=frame,
                        rear_frame=rear_frame,
                    )
            recovery_command = recovery.update(loop_start)

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
            emergency, emergency_reason = emergency_monitor.update(
                now=wall_now,
                last_frame_time=last_frame_time,
                last_data_time=last_data_time,
                sdk_failure_count=sdk_failure_count,
                loop_delay_sec=dt,
                gps_valid=gps_valid,
                perception_valid=perception_valid,
            )

            if target is None:
                candidate = CandidateDirection("STOP", 0.0, 0.0, 0.0, 0.0, 0.0)
                raw_cmd = ControlCommand(0.0, 0.0, mode="MISSION_COMPLETE")
            else:
                candidate = candidate_planner.select(heading_error_rad, perception_result, force_stop=emergency)
                raw_cmd = controller.compute(
                    heading_error_rad=heading_error_rad,
                    candidate=candidate,
                    perception=perception_result,
                    emergency_stop=emergency,
                    recovery_command=recovery_command,
                    dt=dt,
                )

            frame_is_stale = wall_now - last_frame_time > config["safety"]["frame_timeout_sec"]
            data_is_stale = wall_now - last_data_time > config["safety"]["data_timeout_sec"]
            cmd = command_filter.apply(raw_cmd, dt, frame_is_stale, data_is_stale)
            sdk.send_control(cmd)

            if status_enabled and wall_now - last_status_print >= status_interval_sec:
                last_status_print = wall_now
                print(
                    format_urban_status(
                        {
                            "mode": cmd.mode,
                            "target_checkpoint_sequence": target_sequence,
                            "distance_to_checkpoint_m": waypoint_state.get("distance_m"),
                            "target_bearing_deg": target_bearing,
                            "current_heading_deg": data.orientation,
                            "heading_error_deg": heading_error_deg,
                            "raw_linear": raw_cmd.linear,
                            "raw_angular": raw_cmd.angular,
                            "safe_linear": cmd.linear,
                            "safe_angular": cmd.angular,
                            "gps_signal": data.gps_signal,
                            "signal_level": data.signal_level,
                            "data_age_sec": wall_now - data.timestamp,
                            "frame_age_sec": wall_now - frame.timestamp,
                            "stuck_state": stuck,
                            "recovery_state": recovery.state,
                            "log_dir": logger.root,
                        }
                    ),
                    flush=True,
                )

            if waypoint_state["reached"]:
                try:
                    sdk.report_checkpoint()
                    waypoint_manager.mark_current_reported()
                    logger.log_event("CHECKPOINT_REACHED", waypoint_state)
                except Exception as exc:
                    logger.log_event("CHECKPOINT_REPORT_FAILED", {"error": str(exc), "state": waypoint_state})

            logger.log_step(
                frame=frame,
                data=data,
                perception=perception_result,
                candidate=candidate,
                raw_command=raw_cmd,
                command=cmd,
                extra_debug={
                    "heading_error_deg": heading_error_deg,
                    "target_checkpoint": target,
                    "distance_to_checkpoint": waypoint_state.get("distance_m"),
                    "emergency_reason": emergency_reason,
                    "stuck": stuck,
                    "recovery_state": recovery.state,
                },
            )
            last_command = cmd
            sleep_to_maintain_loop_hz(config["project"]["loop_hz"], loop_start)
    except KeyboardInterrupt:
        logger.log_event("MISSION_INTERRUPTED", {})
        sdk.send_control(ControlCommand(0.0, 0.0, mode="USER_STOP"))
    finally:
        try:
            sdk.end_mission()
        except Exception:
            pass
        logger.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
