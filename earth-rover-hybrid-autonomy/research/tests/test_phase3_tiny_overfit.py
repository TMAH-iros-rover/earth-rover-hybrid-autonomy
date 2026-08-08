import os

import pytest
import torch

from research.training.datasets.action_labels import ACTION_NAMES
from research.training.datasets.frodobots_2k_dataset import ManifestSample
from research.training.models.action_resnet18 import build_action_resnet18
from research.training.train_tiny_overfit import compute_class_weights, select_balanced_indices


def make_sample(action: str, index: int) -> ManifestSample:
    return ManifestSample(
        ride_id=str(index // 10),
        front_playlist_ref="ride/front.m3u8",
        front_segment_ref="ride/front_20240101000000000.ts",
        front_frame_id=index,
        front_timestamp=1000.0 + index * 0.05,
        matched_control_timestamp=1000.0 + index * 0.05,
        control_delta_ms=0.0,
        linear=0.0,
        angular=0.0,
        action_class=action,
        timeline_section_id=0,
    )


def test_balanced_selection_is_deterministic() -> None:
    samples = tuple(
        make_sample(action, action_index * 100 + index)
        for action_index, action in enumerate(ACTION_NAMES)
        for index in range(10)
    )

    first = select_balanced_indices(samples, sample_count=25, seed=7)
    second = select_balanced_indices(samples, sample_count=25, seed=7)

    assert first == second
    assert len(set(first)) == 25
    assert {action: sum(samples[index].action_class == action for index in first) for action in ACTION_NAMES} == {
        action: 5 for action in ACTION_NAMES
    }


def test_balanced_selection_rejects_underrepresented_action() -> None:
    samples = tuple(make_sample(action, index) for index, action in enumerate(ACTION_NAMES))

    with pytest.raises(ValueError, match="need 2"):
        select_balanced_indices(samples, sample_count=10, seed=7)


def test_class_weights_are_inverse_frequency() -> None:
    targets = torch.tensor([0, 0, 1, 2, 3, 4], dtype=torch.long)

    weights = compute_class_weights(targets, num_classes=5)

    assert weights.tolist() == pytest.approx([0.6, 1.2, 1.2, 1.2, 1.2])


def test_class_weights_require_every_action() -> None:
    with pytest.raises(ValueError, match="every action class"):
        compute_class_weights(torch.tensor([0, 1, 2, 3], dtype=torch.long), num_classes=5)


def test_resnet18_uses_requested_action_head() -> None:
    model = build_action_resnet18(num_classes=len(ACTION_NAMES), pretrained=False)

    assert model.fc.out_features == len(ACTION_NAMES)


def test_resnet18_rejects_invalid_class_count() -> None:
    with pytest.raises(ValueError, match="num_classes must be positive"):
        build_action_resnet18(num_classes=0, pretrained=False)


def test_deterministic_cublas_workspace_is_configured() -> None:
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] in {":4096:8", ":16:8"}
