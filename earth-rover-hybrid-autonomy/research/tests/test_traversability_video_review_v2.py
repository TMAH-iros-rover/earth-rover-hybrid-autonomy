from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytest

from research.training.datasets.frodobots_2k_manifest import HlsSegment, HlsTimeline
from research.training.run_traversability_video_review_v2 import (
    parse_args,
    selected_dataset_indexes,
)
from research.training.traversability_video_review_v2 import (
    ReviewFrame,
    compose_three_panel_frame,
    process_dataset_review,
    select_review_segments,
)


@dataclass(frozen=True)
class SyntheticFrame:
    frame_id: int
    timestamp: float


@dataclass(frozen=True)
class SyntheticRide:
    dataset: str
    dataset_root: Path
    ride_id: str
    timeline: HlsTimeline
    frames: tuple[SyntheticFrame, ...]


def make_ride(tmp_path: Path, ride_id: str, duration: float = 4.0) -> SyntheticRide:
    dataset_root = tmp_path / "output_rides_0"
    ride_dir = dataset_root / f"ride_{ride_id}_test"
    playlist = ride_dir / "recordings/front_uid_s_1000_video.m3u8"
    start = 1000.0
    segment = HlsSegment(
        reference=f"ride_{ride_id}_test/recordings/front_20240101000000000.ts",
        start=start,
        end=start + duration,
        section_id=0,
    )
    timeline = HlsTimeline(
        playlist_reference=f"ride_{ride_id}_test/recordings/front_uid_s_1000_video.m3u8",
        segments=(segment,),
        starts=(start,),
        stats={"hls_segments": 1, "hls_sections": 1},
    )
    frames = tuple(
        SyntheticFrame(index, start + index * 0.05)
        for index in range(int(duration / 0.05))
    )
    return SyntheticRide(
        dataset=dataset_root.name,
        dataset_root=dataset_root,
        ride_id=ride_id,
        timeline=timeline,
        frames=frames,
    )


def test_segment_selection_is_deterministic_ride_balanced_and_timestamp_sampled(
    tmp_path: Path,
) -> None:
    rides = [make_ride(tmp_path, str(index)) for index in range(7)]

    first, first_skips = select_review_segments(rides, 5, 1.0, 10.0, 0.1, 0.2, 17)
    repeated, repeated_skips = select_review_segments(rides, 5, 1.0, 10.0, 0.1, 0.2, 17)

    assert first == repeated
    assert first_skips == repeated_skips
    assert len(first) == 5
    assert len({segment.ride_id for segment in first}) == 5
    assert all(len(segment.frames) == 10 for segment in first)
    assert all(
        current.timestamp > previous.timestamp
        for segment in first
        for previous, current in zip(segment.frames, segment.frames[1:])
    )


def test_short_ride_is_skipped_with_reason(tmp_path: Path) -> None:
    rides = [make_ride(tmp_path, "short", duration=0.5)]

    segments, skipped = select_review_segments(rides, 1, 1.0, 10.0, 0.0, 0.2, 17)

    assert segments == []
    assert [item["reason"] for item in skipped] == [
        "no_continuous_window",
        "insufficient_eligible_rides",
    ]


def test_three_panel_render_preserves_aspect_and_marks_low_confidence() -> None:
    frame = np.full((90, 160, 3), 120, dtype=np.uint8)
    prediction = np.ones((90, 160), dtype=np.uint8)
    confidence = np.ones((90, 160), dtype=np.float32)
    confidence[:, :80] = 0.2

    composed = compose_three_panel_frame(
        frame,
        prediction,
        confidence,
        {
            "dataset": "output_rides_0",
            "ride_id": "7",
            "frame_id": 10,
            "timestamp": 1000.5,
            "checkpoint_version": "v2:test",
            "inference_latency_ms": 4.0,
            "measured_fps": 20.0,
        },
        panel_width=320,
        low_confidence_threshold=0.5,
    )

    assert composed.shape == (258, 960, 3)
    mask_panel = composed[78:, 640:]
    assert np.all(mask_panel[:, :160] == 0)
    assert np.any(mask_panel[:, 160:] != 0)


class MockDecoder:
    def decode(self, frame: ReviewFrame) -> np.ndarray:
        image = np.zeros((72, 128, 3), dtype=np.uint8)
        image[:, :, 1] = frame.frame_id % 255
        return image


class MockPredictor:
    checkpoint_version = "v2:synthetic"

    def predict(self, frame_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        prediction = np.ones(frame_rgb.shape[:2], dtype=np.uint8)
        prediction[:, frame_rgb.shape[1] // 2 :] = 3
        confidence = np.full(frame_rgb.shape[:2], 0.8, dtype=np.float32)
        return prediction, confidence, 2.5


class SyntheticMp4Writer:
    def __init__(self, path: Path, fps: float, frame_size: tuple[int, int]) -> None:
        self.path = path
        self.frame_size = frame_size
        self.writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            frame_size,
        )
        if not self.writer.isOpened():
            pytest.skip("local OpenCV does not provide an MP4 test encoder")

    def write(self, frame_bgr: np.ndarray) -> None:
        self.writer.write(frame_bgr)

    def close(self) -> dict[str, object]:
        self.writer.release()
        return {
            "codec": "synthetic-mp4v",
            "pixel_format": "test-only",
            "width": self.frame_size[0],
            "height": self.frame_size[1],
            "size_bytes": self.path.stat().st_size,
        }


def test_mock_prediction_writes_synthetic_video_and_manifest(tmp_path: Path) -> None:
    ride = make_ride(tmp_path, "9")
    segments, skipped = select_review_segments([ride], 1, 0.5, 10.0, 0.0, 0.2, 17)
    output = tmp_path / "review"
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"synthetic")

    report = process_dataset_review(
        dataset_root=ride.dataset_root,
        segments=segments,
        skipped_rides=skipped,
        decoder=MockDecoder(),
        predictor=MockPredictor(),
        output_dir=output,
        output_fps=10.0,
        panel_width=160,
        checkpoint_path=checkpoint,
        checkpoint_sha256="abc",
        low_confidence_threshold=None,
        writer_factory=SyntheticMp4Writer,
    )

    saved = json.loads((output / "review_manifest.json").read_text(encoding="utf-8"))
    capture = cv2.VideoCapture(str(output / "traversability_review.mp4"))
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    assert report["success"]
    assert saved["processed_frame_count"] == 5
    assert saved["skipped_frame_count"] == 0
    assert saved["inference_latency_ms"]["median"] == 2.5
    assert ok and frame is not None
    assert frame.shape[1] == 480


def test_cli_dataset_selection_supports_individual_and_all() -> None:
    args = parse_args(
        [
            "--checkpoint",
            "checkpoint.pt",
            "--config",
            "config.yaml",
            "--output-dir",
            "output",
            "--datasets",
            "0",
            "2",
        ]
    )

    assert selected_dataset_indexes(args.datasets) == ("0", "2")
    assert selected_dataset_indexes(("all",)) == ("0", "1", "2")
    with pytest.raises(ValueError, match="cannot be combined"):
        selected_dataset_indexes(("all", "1"))
