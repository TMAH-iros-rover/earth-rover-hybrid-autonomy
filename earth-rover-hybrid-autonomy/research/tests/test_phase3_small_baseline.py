from collections import Counter

import pytest
import torch

from research.training.datasets.action_labels import ACTION_NAMES
from research.training.datasets.frodobots_2k_dataset import ManifestSample
from research.training.train_small_baseline import (
    _is_consecutive,
    choose_ride_split,
    classification_metrics,
    select_split_indices,
    select_video_indices,
)


def make_sample(
    ride_id: str,
    frame_id: int,
    action: str,
    section_id: int = 0,
) -> ManifestSample:
    timestamp = 1000.0 + int(ride_id) * 1000.0 + frame_id * 0.05
    return ManifestSample(
        ride_id=ride_id,
        front_playlist_ref=f"ride_{ride_id}/front.m3u8",
        front_segment_ref=f"ride_{ride_id}/front_20240101000000000.ts",
        front_frame_id=frame_id,
        front_timestamp=timestamp,
        matched_control_timestamp=timestamp,
        control_delta_ms=0.0,
        linear=0.0,
        angular=0.0,
        action_class=action,
        timeline_section_id=section_id,
    )


def make_rides(ride_count: int, frames_per_ride: int = 300) -> tuple[ManifestSample, ...]:
    return tuple(
        make_sample(str(ride), frame, ACTION_NAMES[frame % len(ACTION_NAMES)])
        for ride in range(1, ride_count + 1)
        for frame in range(frames_per_ride)
    )


def test_ride_split_is_deterministic_disjoint_and_has_expected_sizes() -> None:
    samples = make_rides(18)

    first = choose_ride_split(samples, minimum_samples_per_ride=250, minimum_video_run=200, seed=7)
    second = choose_ride_split(samples, minimum_samples_per_ride=250, minimum_video_run=200, seed=7)

    assert first == second
    assert len(first.train) == 10
    assert len(first.validation) == 2
    assert len(first.test) == 2
    assert len(set(first.train + first.validation + first.test)) == 14


def test_ride_split_requires_fourteen_eligible_rides() -> None:
    with pytest.raises(ValueError, match="need 14 eligible rides"):
        choose_ride_split(make_rides(13), 250, 200, seed=7)


def test_bounded_selection_is_deterministic_and_capped_per_ride() -> None:
    samples = make_rides(2)

    first = select_split_indices(samples, ("1", "2"), samples_per_ride=100, seed=7)
    second = select_split_indices(samples, ("1", "2"), samples_per_ride=100, seed=7)

    assert first == second
    assert len(first) == 200
    counts = Counter(samples[index].ride_id for index in first)
    assert counts == {"1": 100, "2": 100}
    assert {samples[index].action_class for index in first} == set(ACTION_NAMES)


def test_classification_metrics_report_identity_confusion_matrix() -> None:
    targets = torch.arange(len(ACTION_NAMES), dtype=torch.long)

    metrics = classification_metrics(targets, targets.clone(), len(ACTION_NAMES))

    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["confusion_matrix"] == torch.eye(len(ACTION_NAMES), dtype=torch.long).tolist()


def test_video_selection_uses_consecutive_frames_without_crossing_section() -> None:
    first = [make_sample("1", frame, "FORWARD") for frame in range(205)]
    second = [make_sample("1", 205 + frame, "FORWARD", section_id=1) for frame in range(50)]
    samples = tuple(first + second)

    selected = select_video_indices(samples, ("1",), frames_per_ride=200)["1"]

    assert len(selected) == 200
    assert all(_is_consecutive(samples[left], samples[right]) for left, right in zip(selected, selected[1:]))
    assert {samples[index].timeline_section_id for index in selected} == {0}


def test_classification_metrics_reject_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="matching one-dimensional shapes"):
        classification_metrics(torch.tensor([0, 1]), torch.tensor([0]), len(ACTION_NAMES))
