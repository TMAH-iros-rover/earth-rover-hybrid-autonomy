from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from training.sam_tp_reproduction import SamTpPredictor
from training.sam_tp_checkpoint_format import CheckpointFormat
from training.sam_tp_hf_backend import build_sam_tp_predictor


def test_official_checkpoint_single_image_cuda_smoke() -> None:
    upstream = Path(os.environ.get("SAM_TP_UPSTREAM_ROOT", "")).expanduser()
    checkpoint = Path(os.environ.get("SAM_TP_CHECKPOINT", "")).expanduser()
    image_path = Path(os.environ.get("SAM_TP_SMOKE_IMAGE", "")).expanduser()
    if not all(
        (
            os.environ.get("SAM_TP_UPSTREAM_ROOT"),
            os.environ.get("SAM_TP_CHECKPOINT"),
            os.environ.get("SAM_TP_SMOKE_IMAGE"),
        )
    ):
        pytest.skip("SAM-TP integration paths are not configured")
    model_config = (
        upstream / "sam2/configs/sam2.1_inference_tiny/sam2.1_custom2.yaml"
    )
    if not all(path.is_file() for path in (model_config, checkpoint, image_path)):
        pytest.skip("SAM-TP integration checkpoint or input is unavailable")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    assert image_bgr is not None
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor = SamTpPredictor(
        upstream,
        model_config,
        checkpoint,
        synchronize=torch.cuda.synchronize,
    )

    prediction = predictor.predict(image_rgb)

    assert prediction.raw_logits.shape == image_rgb.shape[:2]
    assert prediction.traversability_score.shape == image_rgb.shape[:2]
    assert np.isfinite(prediction.raw_logits).all()
    assert np.isfinite(prediction.traversability_score).all()
    assert predictor.load_count == 1


def test_hf_checkpoint_single_image_cuda_smoke() -> None:
    upstream = Path(os.environ.get("SAM_TP_UPSTREAM_ROOT", "")).expanduser()
    checkpoint = Path(os.environ.get("SAM_TP_HF_CHECKPOINT", "")).expanduser()
    image_path = Path(os.environ.get("SAM_TP_SMOKE_IMAGE", "")).expanduser()
    if not all(
        (
            os.environ.get("SAM_TP_UPSTREAM_ROOT"),
            os.environ.get("SAM_TP_HF_CHECKPOINT"),
            os.environ.get("SAM_TP_SMOKE_IMAGE"),
        )
    ):
        pytest.skip("HF SAM2 integration paths are not configured")
    model_config = upstream / "sam2/configs/sam2.1_inference_tiny/sam2.1_custom2.yaml"
    if not all(path.is_file() for path in (model_config, checkpoint, image_path)):
        pytest.skip("HF SAM2 integration checkpoint or input is unavailable")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    assert image_bgr is not None
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor, checkpoint_format = build_sam_tp_predictor(
        upstream,
        model_config,
        checkpoint,
        synchronize=torch.cuda.synchronize,
    )

    first = predictor.predict(image_rgb)
    second = predictor.predict(image_rgb)

    assert checkpoint_format is CheckpointFormat.HF_SAM2
    assert first.raw_logits.shape == image_rgb.shape[:2]
    assert second.raw_logits.shape == image_rgb.shape[:2]
    assert np.isfinite(first.raw_logits).all()
    assert np.isfinite(second.traversability_score).all()
    assert predictor.load_count == 1


def test_sdk_shadow_launcher_rejects_checkpoint_sha256_mismatch(tmp_path: Path) -> None:
    """Both backends share this gate: it runs before any CUDA/backend work."""
    upstream = Path(os.environ.get("SAM_TP_UPSTREAM_ROOT", "")).expanduser()
    checkpoint = Path(os.environ.get("SAM_TP_CHECKPOINT", "")).expanduser()
    if not os.environ.get("SAM_TP_UPSTREAM_ROOT") or not os.environ.get("SAM_TP_CHECKPOINT"):
        pytest.skip("SAM-TP integration paths are not configured")
    model_config = upstream / "sam2/configs/sam2.1_inference_tiny/sam2.1_custom2.yaml"
    if not all(path.exists() for path in (upstream, model_config, checkpoint)):
        pytest.skip("SAM-TP integration checkpoint or input is unavailable")
    root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            sys.executable,
            str(root / "training/run_sam_tp_sdk_shadow.py"),
            "--upstream-root",
            str(upstream),
            "--model-config",
            str(model_config),
            "--checkpoint",
            str(checkpoint),
            "--expected-checkpoint-sha256",
            "0" * 64,
            "--output-dir",
            str(tmp_path / "shadow-output"),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode != 0
    assert "checkpoint SHA-256 differs from the explicitly approved value" in result.stderr
