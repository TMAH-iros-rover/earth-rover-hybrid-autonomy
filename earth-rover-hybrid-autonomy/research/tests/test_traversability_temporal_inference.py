from dataclasses import replace

import cv2
import numpy as np
import pytest
import torch

from research.training.datasets.frodobots_2k_dataset import ManifestSample
from research.training.datasets.traversability_dataset_v1 import (
    image_rgb_to_tensor,
    letterbox_image,
    letterbox_image_and_mask,
    restore_letterbox,
)
from research.training.run_traversability_temporal_inference import (
    IndexedSample,
    TemporalSegment,
    compose_review_frame,
    expand_selected_segments_from_raw_timestamps,
    latency_summary,
    select_temporal_segments,
    temporal_anomaly_reasons,
)


def sample(ride: str, index: int, action: str = "FORWARD", section: int = 0) -> ManifestSample:
    return ManifestSample(
        ride_id=ride,
        front_playlist_ref=f"ride_{ride}/front.m3u8",
        front_segment_ref=f"ride_{ride}/front/{1000 + index // 20}.ts",
        front_frame_id=index,
        front_timestamp=1000.0 + index * 0.05,
        matched_control_timestamp=1000.0 + index * 0.05,
        control_delta_ms=0.0,
        linear=0.5,
        angular=0.2 if action in {"LEFT", "RIGHT"} else 0.0,
        action_class=action,
        timeline_section_id=section,
    )


def temporal_samples() -> tuple[ManifestSample, ...]:
    rows = []
    for ride in range(6):
        for index in range(0, 641):
            action = "LEFT" if ride % 2 == 0 and 200 <= index < 300 else "FORWARD"
            rows.append(sample(str(ride), index, action))
    return tuple(rows)


def test_segment_selection_is_deterministic_and_excludes_approved_rides() -> None:
    rows = temporal_samples()

    first, first_samples = select_temporal_segments(rows, {"0", "1"}, 3, 30.0, 0.15, 17)
    repeated, repeated_samples = select_temporal_segments(rows, {"0", "1"}, 3, 30.0, 0.15, 17)

    assert first == repeated
    assert first_samples == repeated_samples
    assert len(first) == 3
    assert not ({segment.ride_id for segment in first} & {"0", "1"})
    assert all(29.9 <= segment.duration_seconds <= 30.0 for segment in first)
    assert all(segment.frame_count >= 599 for segment in first)


def test_raw_front_timestamps_expand_control_aligned_selection(tmp_path) -> None:
    ride = tmp_path / "ride_9_test"
    recordings = ride / "recordings"
    recordings.mkdir(parents=True)
    segment_file = recordings / "front_20240101000000000.ts"
    segment_file.write_bytes(b"placeholder")
    (recordings / "front_uid_s_1000_video.m3u8").write_text(
        "#EXTM3U\n#EXTINF:40.0,\nfront_20240101000000000.ts\n",
        encoding="utf-8",
    )
    start = 1704067200.0
    with (ride / "front_camera_timestamps_9.csv").open("w", encoding="utf-8") as handle:
        handle.write("frame_id,timestamp\n")
        for index in range(601):
            handle.write(f"{index},{start + index * 0.05:.6f}\n")
    aligned_samples = [
        IndexedSample(index, sample("9", index * 2))
        for index in range(301)
    ]
    aligned_samples = [
        IndexedSample(
            item.manifest_index,
            replace(
                item.sample,
                front_timestamp=start + item.sample.front_frame_id * 0.05,
                front_segment_ref="ride_9_test/recordings/front_20240101000000000.ts",
                front_playlist_ref="ride_9_test/recordings/front_uid_s_1000_video.m3u8",
            ),
        )
        for item in aligned_samples
    ]
    segment = TemporalSegment(
        segment_id="segment_01_ride_9",
        ride_id="9",
        start_timestamp=start,
        end_timestamp=start + 30.0,
        duration_seconds=30.0,
        start_manifest_index=0,
        end_manifest_index=300,
        frame_count=301,
        aligned_manifest_frame_count=301,
        timeline_section_id=0,
        action_distribution={"FORWARD": 301},
        selection_score=0.0,
        selection_reason="test",
    )

    segments, expanded, exclusions = expand_selected_segments_from_raw_timestamps(
        tmp_path,
        [segment],
        {segment.segment_id: aligned_samples},
        maximum_gap_seconds=0.15,
    )

    assert exclusions == []
    assert segments[0].frame_count == 601
    assert segments[0].aligned_manifest_frame_count == 301
    assert len(expanded[segment.segment_id]) == 601
    assert expanded[segment.segment_id][1].sample.action_class == "UNALIGNED"


def test_segment_selection_rejects_timestamp_gaps_and_section_boundaries() -> None:
    rows = []
    for ride in range(3):
        rows.extend(sample(str(ride), index, section=index // 300) for index in range(641))

    try:
        select_temporal_segments(tuple(rows), set(), 3, 30.0, 0.15, 17)
    except ValueError as exc:
        assert "only 0 unseen rides" in str(exc)
    else:
        raise AssertionError("section boundaries must prevent a false continuous 30-second window")


def test_temporal_preprocessing_reuses_training_letterbox_and_normalization() -> None:
    image = np.zeros((9, 16, 3), dtype=np.uint8)
    image[:, 8:] = (10, 120, 240)
    mask = np.ones((9, 16), dtype=np.uint8)

    temporal_image, _ = letterbox_image(image, 32)
    training_image, _ = letterbox_image_and_mask(image, mask, 32)

    assert np.array_equal(temporal_image, training_image)
    assert torch.equal(image_rgb_to_tensor(temporal_image), image_rgb_to_tensor(training_image))
    restored = restore_letterbox(temporal_image[:, :, 0], image.shape[:2], cv2.INTER_LINEAR)
    assert restored.shape == image.shape[:2]


def test_temporal_anomaly_flags_do_not_modify_predictions() -> None:
    previous_prediction = np.zeros((10, 10), dtype=np.uint8)
    current_prediction = np.ones((10, 10), dtype=np.uint8)
    previous = {
        "prediction": previous_prediction,
        "class_ratios": {"ON_ROAD": 0.1, "OFF_ROAD": 0.1, "OBSTACLE": 0.8},
        "mean_confidence": 0.8,
    }
    current = {
        "prediction": current_prediction,
        "class_ratios": {"ON_ROAD": 0.6, "OFF_ROAD": 0.3, "OBSTACLE": 0.1},
        "mean_confidence": 0.4,
    }
    thresholds = {
        "pixel_flicker_fraction": 0.18,
        "class_ratio_l1_change": 0.2,
        "mean_confidence_drop": 0.12,
        "low_mean_confidence": 0.45,
        "dominant_class_fraction": 0.98,
        "obstacle_fraction_drop": 0.08,
    }

    reasons, metrics = temporal_anomaly_reasons(previous, current, thresholds)

    assert set(reasons) == {
        "low_mean_confidence",
        "high_pixel_flicker",
        "large_class_ratio_change",
        "confidence_drop",
        "obstacle_disappearance_candidate",
    }
    assert metrics["pixel_flicker_fraction"] == 1.0
    assert np.array_equal(current_prediction, current["prediction"])


def test_review_frame_contains_four_aspect_preserved_panels() -> None:
    image = np.zeros((90, 160, 3), dtype=np.uint8)
    prediction = np.ones((90, 160), dtype=np.uint8)
    confidence = np.full((90, 160), 0.75, dtype=np.float32)
    record = {
        "ride_id": "ride",
        "frame_id": 4,
        "timestamp": 1000.2,
        "class_pixel_ratios": {"ON_ROAD": 1.0, "OFF_ROAD": 0.0, "OBSTACLE": 0.0},
        "mean_confidence": 0.75,
        "model_only_latency_ms": 10.0,
    }

    composed = compose_review_frame(image, prediction, confidence, record, panel_width=320)

    assert composed.shape == (360, 640, 3)


def test_latency_summary_reports_required_percentiles() -> None:
    result = latency_summary([1.0, 2.0, 3.0, 4.0])

    assert result["p50"] == 2.5
    assert result["p95"] == pytest.approx(3.85)
    assert result["max"] == 4.0
    assert result["mean"] == 2.5
