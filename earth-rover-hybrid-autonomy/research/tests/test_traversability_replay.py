from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from earth_rover.traversability_replay import (
    LogOnlyControlSink,
    OfflineTraversabilityPipeline,
    ReplayObservation,
    run_offline_replay,
)


def config(frame_timeout: float = 1.0) -> dict:
    return {
        "project": {"loop_hz": 10},
        "control": {
            "linear_min": 0.0,
            "linear_max": 0.22,
            "angular_min": -0.55,
            "angular_max": 0.55,
            "base_linear": 0.18,
            "heading_kp": 0.60,
            "heading_kd": 0.0,
            "local_goal_kp": 0.35,
            "slow_heading_error_deg": 30,
            "rotate_in_place_error_deg": 90,
            "rotate_angular": 0.55,
            "command_smoothing_alpha": 0.0,
            "max_linear_delta_per_sec": 10.0,
            "max_angular_delta_per_sec": 10.0,
        },
        "perception": {"obstacle_stop_threshold": 0.75},
        "safety": {
            "enable_emergency_stop": True,
            "frame_timeout_sec": frame_timeout,
            "data_timeout_sec": frame_timeout,
            "command_loop_delay_limit_sec": 1.0,
            "sdk_failure_limit": 3,
        },
        "traversability_adapter": {
            "roi_top": 0.25,
            "near_top": 0.70,
            "sector_boundaries": [0.0, 0.33, 0.67, 1.0],
            "near_obstacle_stop_ratio": 0.45,
        },
        "goal_aware_planner": {
            "candidate_heading_offsets_deg": {"LEFT": 35, "CENTER": 0, "RIGHT": -35},
            "hard_obstacle_ratio": 0.55,
        },
    }


def observation(
    source_timestamp: float = 10.0,
    frame_timestamp: float = 10.0,
    sensor_timestamp: float = 10.0,
) -> ReplayObservation:
    return ReplayObservation(
        source_timestamp=source_timestamp,
        frame_timestamp=frame_timestamp,
        sensor_timestamp=sensor_timestamp,
        source_mask=np.ones((60, 90), dtype=np.uint8),
        confidence=np.ones((60, 90), dtype=np.float32),
        inference_latency_ms=12.5,
        heading_error_deg=0.0,
        goal_input_mode="fixed_heading_error",
        goal_input_valid=True,
        gps_valid=False,
        ride_id="42",
        frame_id=7,
    )


def test_replay_never_transmits_and_clamps_controller_output() -> None:
    result = OfflineTraversabilityPipeline(config()).process(observation())

    assert result.record["command_transmitted"] is False
    assert result.record["gps_valid"] is False
    assert result.record["goal_input_mode"] == "fixed_heading_error"
    assert 0.0 <= result.expected_command.linear <= 0.22
    assert -0.55 <= result.expected_command.angular <= 0.55


def test_stale_frame_or_sensor_forces_safety_stop() -> None:
    result = OfflineTraversabilityPipeline(config(frame_timeout=1.0)).process(
        observation(source_timestamp=12.0, frame_timestamp=10.0, sensor_timestamp=10.0)
    )

    assert result.expected_command.linear == 0.0
    assert result.expected_command.angular == 0.0
    assert result.record["safety_state"] == "STOP"
    assert result.record["safety_reason"] in {"FRAME_TIMEOUT", "STALE_INPUT", "STALE_DATA_STOP"}


def test_near_obstacle_stop_overrides_previous_smoothed_command() -> None:
    pipeline = OfflineTraversabilityPipeline(config())
    moving = pipeline.process(observation())
    blocked_mask = np.ones((60, 90), dtype=np.uint8)
    blocked_mask[42:, 30:60] = 3
    item = observation(10.1, 10.1, 10.1)
    blocked = ReplayObservation(**{**item.__dict__, "source_mask": blocked_mask})

    stopped = pipeline.process(blocked)

    assert moving.expected_command.linear > 0.0
    assert stopped.expected_command.linear == 0.0
    assert stopped.expected_command.angular == 0.0
    assert stopped.record["safety_state"] == "STOP"
    assert "NEAR_CENTER_OBSTACLE" in stopped.record["safety_reason"]


def test_invalid_goal_input_forces_stop_without_claiming_gps() -> None:
    item = observation()
    invalid = ReplayObservation(**{**item.__dict__, "goal_input_valid": False})

    result = OfflineTraversabilityPipeline(config()).process(invalid)

    assert result.expected_command.linear == 0.0
    assert result.record["safety_reason"] == "INVALID_GOAL_INPUT"
    assert result.record["gps_valid"] is False


def test_log_only_sink_writes_required_csv_and_jsonl_schema(tmp_path: Path) -> None:
    output = tmp_path / "replay"
    count = run_offline_replay(
        [observation(), observation(10.1, 10.1, 10.1)],
        OfflineTraversabilityPipeline(config()),
        LogOnlyControlSink(output),
    )

    assert count == 2
    rows = list(csv.DictReader((output / "replay_steps.csv").open(encoding="utf-8")))
    records = [
        json.loads(line) for line in (output / "replay_steps.jsonl").read_text().splitlines()
    ]
    assert len(rows) == len(records) == 2
    assert set(LogOnlyControlSink.FIELDNAMES) == set(rows[0])
    assert all(record["command_transmitted"] is False for record in records)
    assert records[0]["candidate_scores"]
