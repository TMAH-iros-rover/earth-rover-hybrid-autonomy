from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from training.sam_tp_checkpoint_format import (  # noqa: E402
    CheckpointFormat,
    CheckpointFormatError,
    load_checkpoint_state_dict,
)

_HF_SIGNATURE_STATE_DICT = {
    "no_memory_embedding": torch.zeros(1, 1, 256),
    "shared_image_embedding.positional_embedding": torch.zeros(2, 128),
    "vision_encoder.backbone.patch_embed.projection.weight": torch.zeros(96, 3, 7, 7),
    "prompt_encoder.point_embed.weight": torch.zeros(4, 256),
    "mask_decoder.iou_token.weight": torch.zeros(1, 256),
}


def test_detects_genie_format(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint_2.pt"
    torch.save(
        {"model": {"a": torch.zeros(2, 2)}, "epoch": 3, "optimizer": {}},
        checkpoint,
    )

    fmt, state_dict = load_checkpoint_state_dict(checkpoint)

    assert fmt is CheckpointFormat.GENIE_SAM_TP
    assert set(state_dict) == {"a"}


def test_detects_hf_sam2_format(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best_sam_tp.pt"
    torch.save(dict(_HF_SIGNATURE_STATE_DICT), checkpoint)

    fmt, state_dict = load_checkpoint_state_dict(checkpoint)

    assert fmt is CheckpointFormat.HF_SAM2
    assert set(state_dict) == set(_HF_SIGNATURE_STATE_DICT)


def test_rejects_bare_tensor_dict_missing_hf_signature_keys(tmp_path: Path) -> None:
    checkpoint = tmp_path / "unrelated.pt"
    torch.save({"some_unrelated_weight": torch.zeros(3, 3)}, checkpoint)

    with pytest.raises(CheckpointFormatError, match="missing expected"):
        load_checkpoint_state_dict(checkpoint)


def test_rejects_genie_shaped_model_key_with_non_tensor_values(tmp_path: Path) -> None:
    checkpoint = tmp_path / "malformed.pt"
    torch.save({"model": {"a": "not-a-tensor"}}, checkpoint)

    with pytest.raises(CheckpointFormatError, match="not tensors"):
        load_checkpoint_state_dict(checkpoint)


def test_rejects_checkpoint_with_no_recognizable_top_level_shape(tmp_path: Path) -> None:
    checkpoint = tmp_path / "garbage.pt"
    torch.save({"epoch": 3, "notes": "hello"}, checkpoint)

    with pytest.raises(CheckpointFormatError, match="does not match a recognized"):
        load_checkpoint_state_dict(checkpoint)


def test_missing_checkpoint_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_checkpoint_state_dict(tmp_path / "missing.pt")
