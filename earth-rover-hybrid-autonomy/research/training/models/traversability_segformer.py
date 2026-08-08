from __future__ import annotations

from collections.abc import Mapping
from typing import Any


CHECKPOINT = "nvidia/segformer-b0-finetuned-ade-512-512"
REVISION = "489d5cd81a0b59fab9b7ea758d3548ebe99677da"
WEIGHT_FILE = "model.safetensors"
ID2LABEL = {0: "ON_ROAD", 1: "OFF_ROAD", 2: "OBSTACLE"}


def validate_three_class_checkpoint(checkpoint: Mapping[str, object]) -> None:
    required = {
        "model_state_dict",
        "model_config",
        "epoch",
        "metrics",
        "source_checkpoint",
        "source_revision",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(f"initial checkpoint schema is missing keys: {missing}")

    config = checkpoint["model_config"]
    if not isinstance(config, Mapping):
        raise ValueError("initial checkpoint model_config must be a mapping")
    id2label = config.get("id2label")
    if not isinstance(id2label, Mapping):
        raise ValueError("initial checkpoint must define id2label")
    try:
        normalized_id2label = {int(class_id): str(name) for class_id, name in id2label.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError("initial checkpoint has invalid id2label keys") from exc
    if normalized_id2label != ID2LABEL:
        raise ValueError("initial checkpoint must use the approved 3-class label mapping")
    label2id = config.get("label2id")
    if label2id is not None:
        if not isinstance(label2id, Mapping):
            raise ValueError("initial checkpoint label2id must be a mapping")
        try:
            normalized_label2id = {str(name): int(class_id) for name, class_id in label2id.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError("initial checkpoint has invalid label2id values") from exc
        expected_label2id = {name: class_id for class_id, name in ID2LABEL.items()}
        if normalized_label2id != expected_label2id:
            raise ValueError("initial checkpoint label2id conflicts with id2label")

    num_labels = config.get("num_labels")
    if num_labels is not None and int(num_labels) != len(ID2LABEL):
        raise ValueError("initial checkpoint num_labels conflicts with id2label")
    if int(config.get("semantic_loss_ignore_index", -1)) != 255:
        raise ValueError("initial checkpoint must use semantic_loss_ignore_index=255")

    state = checkpoint["model_state_dict"]
    if not isinstance(state, Mapping):
        raise ValueError("initial checkpoint model_state_dict must be a mapping")
    for name in ("decode_head.classifier.weight", "decode_head.classifier.bias"):
        tensor = state.get(name)
        shape = getattr(tensor, "shape", None)
        if not shape or int(shape[0]) != len(ID2LABEL):
            raise ValueError("initial checkpoint classifier head is not 3-class")


def build_traversability_segformer(
    pretrained: bool = True,
    config_dict: dict[str, object] | None = None,
) -> Any:
    try:
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
    except ImportError as exc:
        raise RuntimeError("install requirements-segmentation.txt on Dell") from exc

    if config_dict is not None:
        config = SegformerConfig.from_dict(config_dict)
    else:
        config = SegformerConfig.from_pretrained(CHECKPOINT, revision=REVISION)
        config.num_labels = 3
        config.id2label = ID2LABEL
        config.label2id = {name: class_id for class_id, name in ID2LABEL.items()}
        config.semantic_loss_ignore_index = 255
    if pretrained:
        return SegformerForSemanticSegmentation.from_pretrained(
            CHECKPOINT,
            revision=REVISION,
            config=config,
            ignore_mismatched_sizes=True,
            use_safetensors=True,
        )
    return SegformerForSemanticSegmentation(config)
