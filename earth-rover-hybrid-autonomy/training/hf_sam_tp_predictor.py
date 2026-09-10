from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from training.sam_tp_reproduction import (
    SamTpPrediction,
    score_to_heatmap,
    sigmoid_logits,
    validate_rgb_image,
)

DEFAULT_BASE_MODEL_ID = "facebook/sam2.1-hiera-tiny"


class HfSamTpPredictor:
    """Load a self-trained SAM-TP checkpoint through HuggingFace ``transformers``.

    ``scripts/GeNIE_ws/pre_processing_dataset_withSAM2/03_train_sam_tp.py`` fine-tunes
    ``transformers.Sam2Model`` (no point/box prompt -- the model's own
    ``not_a_point_embed``/``no_mask_embed`` act as the learned "traversable" prompt
    token) and saves ``model.state_dict()``. That checkpoint format is unrelated to
    ``training.sam_tp_reproduction.SamTpPredictor``, which instead imports a frozen
    upstream ``sam2.sam_tp`` checkout. This class loads the HF-format checkpoint
    directly and exposes the same validated ``SamTpPrediction`` interface, so it can
    be swapped in anywhere a ``Predictor`` (``.predict(image_rgb) -> SamTpPrediction``)
    is expected -- the phase1 frame processor and the local planner do not know or
    care which backend produced the traversability score map.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        base_model_id: str = DEFAULT_BASE_MODEL_ID,
        device: str = "cuda",
        loader: Callable[[str, str, str], tuple[Any, Any]] | None = None,
        synchronize: Callable[[], None] | None = None,
    ) -> None:
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.base_model_id = base_model_id
        self.device = device
        self._loader = loader
        self._synchronize = synchronize
        self._model: Any | None = None
        self._processor: Any | None = None
        self.load_time_ms: float | None = None
        self.load_count = 0

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"checkpoint does not exist: {self.checkpoint}")
        started = time.perf_counter()
        loader = self._loader or self._default_loader
        self._model, self._processor = loader(
            self.base_model_id, str(self.checkpoint), self.device
        )
        self._model.eval()
        if self._synchronize is not None:
            self._synchronize()
        self.load_time_ms = (time.perf_counter() - started) * 1000.0
        self.load_count += 1

    def _default_loader(
        self, base_model_id: str, checkpoint_path: str, device: str
    ) -> tuple[Any, Any]:
        import torch
        from transformers import Sam2Model, Sam2Processor

        processor = Sam2Processor.from_pretrained(base_model_id)
        model = Sam2Model.from_pretrained(base_model_id).to(device)
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        return model, processor

    def predict(self, image_rgb: np.ndarray) -> SamTpPrediction:
        validate_rgb_image(image_rgb)
        self.load()
        if self._synchronize is not None:
            self._synchronize()
        started = time.perf_counter()
        logits = self._forward_no_prompt(image_rgb)
        if self._synchronize is not None:
            self._synchronize()
        latency_ms = (time.perf_counter() - started) * 1000.0
        score = sigmoid_logits(logits)
        heatmap = score_to_heatmap(score)
        return SamTpPrediction(
            raw_logits=logits,
            traversability_score=score,
            heatmap=heatmap,
            input_shape=tuple(int(value) for value in image_rgb.shape),
            output_shape=tuple(int(value) for value in logits.shape),
            inference_time_ms=latency_ms,
            device=self.device,
        )

    def _forward_no_prompt(self, image_rgb: np.ndarray) -> np.ndarray:
        """Mirror 03_train_sam_tp.py's ``forward_no_prompt`` and upsample the
        low-resolution mask logits it returns back to the input frame size."""

        import torch
        import torch.nn.functional as functional

        height, width = image_rgb.shape[:2]
        inputs = self._processor(images=image_rgb, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            output = self._model(**inputs, multimask_output=False)
            low_res_logits = output.pred_masks[:, 0, 0]
            resized = functional.interpolate(
                low_res_logits.unsqueeze(1).float(),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            logits = resized[0, 0].detach().to("cpu").numpy()
        return logits.astype(np.float32)
