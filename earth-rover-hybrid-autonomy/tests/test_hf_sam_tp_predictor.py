from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from training.hf_sam_tp_predictor import HfSamTpPredictor


class FakeInputs(dict):
    def to(self, device: str) -> "FakeInputs":
        return self


class FakeProcessor:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, images: np.ndarray, return_tensors: str) -> FakeInputs:
        self.calls += 1
        return FakeInputs(pixel_values=torch.zeros((1, 3, 4, 4)))


class FakeOutput:
    def __init__(self, low_res_hw: tuple[int, int]) -> None:
        height, width = low_res_hw
        # shape (batch=1, num_masks=1, num_multimask=1, h_low, w_low), matching
        # transformers.Sam2Model's pred_masks layout that forward_no_prompt slices
        # with [:, 0, 0] in 03_train_sam_tp.py.
        self.pred_masks = torch.full((1, 1, 1, height, width), -2.0)


class FakeModel:
    def __init__(self, low_res_hw: tuple[int, int] = (8, 10)) -> None:
        self.low_res_hw = low_res_hw
        self.eval_calls = 0
        self.forward_calls = 0

    def eval(self) -> None:
        self.eval_calls += 1

    def __call__(self, *, pixel_values, multimask_output: bool) -> FakeOutput:
        self.forward_calls += 1
        assert multimask_output is False
        return FakeOutput(self.low_res_hw)


def fake_loader_factory(model: FakeModel, processor: FakeProcessor):
    calls: list[tuple[str, str, str]] = []

    def loader(base_model_id: str, checkpoint_path: str, device: str):
        calls.append((base_model_id, checkpoint_path, device))
        return model, processor

    return loader, calls


def test_predict_loads_once_and_upsamples_low_res_logits_to_frame_size(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "best_sam_tp.pt"
    checkpoint.write_bytes(b"checkpoint")
    model = FakeModel(low_res_hw=(8, 10))
    processor = FakeProcessor()
    loader, calls = fake_loader_factory(model, processor)

    predictor = HfSamTpPredictor(
        checkpoint,
        base_model_id="facebook/sam2.1-hiera-tiny",
        device="cpu",
        loader=loader,
    )
    image = np.zeros((36, 64, 3), dtype=np.uint8)

    first = predictor.predict(image)
    second = predictor.predict(image)

    assert len(calls) == 1
    assert predictor.load_count == 1
    assert model.eval_calls == 1
    assert model.forward_calls == 2
    assert first.output_shape == (36, 64)
    assert first.traversability_score.shape == (36, 64)
    assert np.all((0.0 <= first.traversability_score) & (first.traversability_score <= 1.0))
    # logits were a constant -2.0 -> sigmoid(-2.0) for every upsampled pixel.
    assert first.traversability_score == pytest.approx(1.0 / (1.0 + np.exp(2.0)), abs=1e-4)
    assert second.device == "cpu"


def test_predict_rejects_missing_checkpoint(tmp_path: Path) -> None:
    predictor = HfSamTpPredictor(tmp_path / "missing.pt", device="cpu")
    with pytest.raises(FileNotFoundError, match="checkpoint does not exist"):
        predictor.load()


def test_predict_rejects_invalid_rgb_input(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best_sam_tp.pt"
    checkpoint.write_bytes(b"checkpoint")
    loader, _calls = fake_loader_factory(FakeModel(), FakeProcessor())
    predictor = HfSamTpPredictor(checkpoint, device="cpu", loader=loader)

    with pytest.raises(ValueError, match="HxWx3"):
        predictor.predict(np.zeros((10, 10), dtype=np.uint8))
