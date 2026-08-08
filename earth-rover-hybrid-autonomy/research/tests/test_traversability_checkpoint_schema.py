from __future__ import annotations

from dataclasses import dataclass

import pytest

from research.training.models.traversability_segformer import validate_three_class_checkpoint


@dataclass(frozen=True)
class FakeTensor:
    shape: tuple[int, ...]


def checkpoint_payload(class_count: int = 3) -> dict[str, object]:
    return {
        "model_state_dict": {
            "decode_head.classifier.weight": FakeTensor((class_count, 256, 1, 1)),
            "decode_head.classifier.bias": FakeTensor((class_count,)),
        },
        "model_config": {
            "id2label": {0: "ON_ROAD", 1: "OFF_ROAD", 2: "OBSTACLE"},
            "label2id": {"ON_ROAD": 0, "OFF_ROAD": 1, "OBSTACLE": 2},
            "semantic_loss_ignore_index": 255,
        },
        "epoch": 4,
        "metrics": {},
        "source_checkpoint": "source",
        "source_revision": "revision",
    }


def test_schema_derives_num_labels_from_label_mapping() -> None:
    validate_three_class_checkpoint(checkpoint_payload())


def test_schema_rejects_non_three_class_head() -> None:
    with pytest.raises(ValueError, match="classifier head is not 3-class"):
        validate_three_class_checkpoint(checkpoint_payload(class_count=150))


def test_schema_rejects_conflicting_explicit_num_labels() -> None:
    checkpoint = checkpoint_payload()
    checkpoint["model_config"]["num_labels"] = 150

    with pytest.raises(ValueError, match="num_labels conflicts"):
        validate_three_class_checkpoint(checkpoint)
