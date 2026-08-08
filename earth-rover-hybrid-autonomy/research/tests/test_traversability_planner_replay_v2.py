from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from earth_rover.traversability_replay import OfflineTraversabilityPipeline, ReplayObservation
from research.training.run_traversability_planner_replay_v2 import parse_args
from research.training.traversability_planner_replay_v2 import (
    compose_planner_review_frame,
    delayed_replay_pairs,
)
from research.training.traversability_video_review_v2 import (
    H264VideoWriter,
    ReviewFrame,
    ReviewSegment,
)


def config() -> dict:
    return {
        "project": {"loop_hz": 10},
        "control": {
            "linear_min": 0.0,
            "linear_max": 0.22,
            "angular_min": -0.55,
            "angular_max": 0.55,
            "base_linear": 0.18,
            "heading_kp": 0.6,
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
            "frame_timeout_sec": 3.0,
            "data_timeout_sec": 3.0,
            "command_loop_delay_limit_sec": 1.0,
            "sdk_failure_limit": 3,
        },
        "traversability_adapter": {
            "roi_top": 0.35,
            "near_top": 0.72,
            "sector_boundaries": [0.0, 0.34, 0.66, 1.0],
        },
        "goal_aware_planner": {
            "candidate_heading_offsets_deg": {"LEFT": 35, "CENTER": 0, "RIGHT": -35},
        },
    }


def segment() -> ReviewSegment:
    frames = tuple(
        ReviewFrame(
            dataset="output_rides_0",
            ride_id="7",
            frame_id=index,
            timestamp=1000.0 + index * 0.1,
            playlist_reference="ride_7/recordings/front_uid_s_1000_video.m3u8",
            segment_reference="ride_7/recordings/front_1000.ts",
            timeline_section_id=0,
        )
        for index in range(51)
    )
    return ReviewSegment("output_rides_0", "7", 1000.0, 1005.0, 5.0, frames)


def test_delayed_pairing_uses_latest_available_frame() -> None:
    immediate = delayed_replay_pairs(segment(), 0.0, 1.0)
    delayed = delayed_replay_pairs(segment(), 2.0, 1.0)

    assert len(immediate) == len(delayed) == 10
    assert immediate[0].source_timestamp == immediate[0].observed_frame.timestamp
    assert delayed[0].source_timestamp - delayed[0].observed_frame.timestamp == pytest.approx(2.0)
    assert all(
        pair.observed_frame.timestamp <= pair.source_timestamp - 2.0 + 1e-9
        for pair in delayed
    )


def test_synthetic_pipeline_frame_renders_three_aspect_preserved_panels() -> None:
    frame = np.full((90, 160, 3), 100, dtype=np.uint8)
    mask = np.ones((90, 160), dtype=np.uint8)
    mask[:, :45] = 3
    confidence = np.full((90, 160), 0.9, dtype=np.float32)
    result = OfflineTraversabilityPipeline(config()).process(
        ReplayObservation(
            source_timestamp=1002.0,
            frame_timestamp=1000.0,
            sensor_timestamp=1000.0,
            source_mask=mask,
            confidence=confidence,
            inference_latency_ms=11.0,
            heading_error_deg=20.0,
            goal_input_mode="fixed_heading_error",
            goal_input_valid=True,
            ride_id="7",
            frame_id=1,
        )
    )

    rendered = compose_planner_review_frame(
        frame,
        mask,
        confidence,
        result,
        "output_rides_0",
        "7",
        1,
        "v2:synthetic",
        panel_width=320,
        low_confidence_threshold=0.5,
    )

    assert rendered.dtype == np.uint8
    assert rendered.shape == (292, 960, 3)
    assert result.record["command_transmitted"] is False


def test_h264_writer_produces_quicktime_compatible_video(tmp_path: Path) -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg and ffprobe are unavailable")
    encoders = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        check=True,
    )
    if "libx264" not in encoders.stdout:
        pytest.skip("ffmpeg does not provide libx264")
    path = tmp_path / "review.mp4"
    writer = H264VideoWriter(path, 10.0, (320, 180))
    for _ in range(3):
        writer.write(np.zeros((180, 320, 3), dtype=np.uint8))

    report = writer.close()

    assert report["codec"] == "h264"
    assert report["pixel_format"] == "yuv420p"
    assert report["frame_rate"] == "10/1"
    capture = cv2.VideoCapture(str(path))
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    assert ok and frame is not None


def test_cli_requires_explicit_goal_heading_and_supports_short_replay() -> None:
    args = parse_args(
        [
            "--checkpoint",
            "checkpoint.pt",
            "--training-config",
            "config.yaml",
            "--output-dir",
            "output",
            "--datasets",
            "1",
            "--duration-seconds",
            "5",
            "--latency-sec",
            "2",
            "--goal-heading-error-deg",
            "-15",
        ]
    )

    assert args.datasets == ["1"]
    assert args.duration_seconds == 5.0
    assert args.latency_sec == 2.0
    assert args.goal_heading_error_deg == -15.0
