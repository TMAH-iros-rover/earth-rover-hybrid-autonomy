from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from training.sam_tp_checkpoint_format import CheckpointFormat, CheckpointFormatError  # noqa: E402
from training.sam_tp_hf_backend import (  # noqa: E402
    HF_SAM2_CONFIG_DIR,
    HF_SAM2_MODEL_CONFIG_ID,
    HfSam2Wrapper,
    build_sam_tp_predictor,
)
from training.sam_tp_reproduction import SamTpPredictor  # noqa: E402


def _build_real_architecture_state_dict() -> dict[str, torch.Tensor]:
    """A state dict shaped exactly like the vendored sam2_hf_tiny config.

    Random weights, but the real Sam2Model class/config -- so this exercises
    the actual strict key/shape validation path without depending on the
    ~120MB best_sam_tp.pt file being present.
    """
    from transformers import Sam2Config, Sam2Model

    with (HF_SAM2_CONFIG_DIR / "sam2_config.json").open(encoding="utf-8") as handle:
        config_dict = json.load(handle)
    model = Sam2Model(Sam2Config(**config_dict))
    return model.state_dict()


def test_vendored_config_matches_sam2_model_architecture_exactly() -> None:
    # This is the same check performed against the real best_sam_tp.pt
    # checkpoint during investigation: 0 missing / 0 unexpected / 0
    # shape-mismatched keys, strict load succeeds.
    from transformers import Sam2Config, Sam2Model

    with (HF_SAM2_CONFIG_DIR / "sam2_config.json").open(encoding="utf-8") as handle:
        config_dict = json.load(handle)
    model = Sam2Model(Sam2Config(**config_dict))
    state_dict = _build_real_architecture_state_dict()

    model.load_state_dict(state_dict, strict=True)  # must not raise
    assert len(state_dict) == 309
    assert sum(v.numel() for v in state_dict.values()) == 31_441_233


def test_hf_sam2_wrapper_loads_and_infers_on_cpu(tmp_path: Path) -> None:
    checkpoint = tmp_path / "synthetic_best_sam_tp.pt"
    torch.save(_build_real_architecture_state_dict(), checkpoint)

    wrapper = HfSam2Wrapper(checkpoint, device="cpu")
    assert wrapper.model_config_id == HF_SAM2_MODEL_CONFIG_ID

    image = np.random.default_rng(0).integers(0, 255, size=(48, 64, 3), dtype=np.uint8)
    result = wrapper.run_sam2_inference(image)

    logits = result["logits"]
    heatmap = result["heatmap"]
    assert logits.shape == (48, 64)
    assert logits.dtype == np.float32
    assert np.isfinite(logits).all()
    assert heatmap.shape == (48, 64, 3)
    assert heatmap.dtype == np.uint8

    score = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    assert bool((score >= 0.0).all() and (score <= 1.0).all())


def test_hf_sam2_wrapper_rejects_shape_mismatched_checkpoint(tmp_path: Path) -> None:
    state_dict = _build_real_architecture_state_dict()
    # Corrupt a non-signature key's shape; format detection still recognizes
    # this as hf_sam2 (all signature keys are intact), so the failure must
    # come from the backend's own strict architecture validation.
    key = "vision_encoder.neck.convs.0.weight"
    state_dict[key] = torch.zeros(1, 1)
    checkpoint = tmp_path / "corrupted_best_sam_tp.pt"
    torch.save(state_dict, checkpoint)

    with pytest.raises(CheckpointFormatError, match="shape_mismatches"):
        HfSam2Wrapper(checkpoint, device="cpu")


def test_hf_sam2_wrapper_rejects_genie_format_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint_2.pt"
    torch.save({"model": {"a": torch.zeros(2, 2)}}, checkpoint)

    with pytest.raises(CheckpointFormatError, match="not hf_sam2"):
        HfSam2Wrapper(checkpoint, device="cpu")


def test_build_sam_tp_predictor_dispatches_hf_sam2_backend(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best_sam_tp.pt"
    torch.save(_build_real_architecture_state_dict(), checkpoint)

    predictor, checkpoint_format = build_sam_tp_predictor(
        tmp_path / "unused-upstream",
        tmp_path / "unused-model-config.yaml",
        checkpoint,
        device="cpu",
    )

    assert checkpoint_format is CheckpointFormat.HF_SAM2
    assert isinstance(predictor, SamTpPredictor)
    assert predictor.upstream_root == HF_SAM2_CONFIG_DIR
    assert predictor.model_config == HF_SAM2_CONFIG_DIR / "sam2_config.json"
    assert predictor.checkpoint == checkpoint.resolve()


def test_build_sam_tp_predictor_dispatches_genie_backend(tmp_path: Path) -> None:
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    model_config = tmp_path / "model.yaml"
    model_config.write_text("model: {}\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint_2.pt"
    torch.save({"model": {"a": torch.zeros(2, 2)}}, checkpoint)

    # device stays "cuda" (the default): SamTpPredictor requires it unless a
    # custom loader is supplied, and the GENIE branch intentionally uses the
    # official loader unchanged. This only exercises dispatch/wiring, not an
    # actual load, so no GPU is needed to run this assertion.
    predictor, checkpoint_format = build_sam_tp_predictor(upstream, model_config, checkpoint)

    assert checkpoint_format is CheckpointFormat.GENIE_SAM_TP
    assert isinstance(predictor, SamTpPredictor)
    assert predictor.upstream_root == upstream.resolve()
    assert predictor.model_config == model_config.resolve()


def test_build_sam_tp_predictor_rejects_malformed_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "garbage.pt"
    torch.save({"epoch": 3}, checkpoint)

    with pytest.raises(CheckpointFormatError, match="does not match a recognized"):
        build_sam_tp_predictor(
            tmp_path / "unused-upstream",
            tmp_path / "unused-model-config.yaml",
            checkpoint,
            device="cpu",
        )
