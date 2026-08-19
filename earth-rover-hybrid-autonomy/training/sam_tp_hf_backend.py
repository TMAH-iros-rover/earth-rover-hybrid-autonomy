from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from training.sam_tp_checkpoint_format import (
    CheckpointFormat,
    CheckpointFormatError,
    load_checkpoint_state_dict,
)
from training.sam_tp_reproduction import (
    SamTpPredictor,
    compare_state_dicts,
    score_to_heatmap,
    sigmoid_logits,
)

# See configs/sam2_hf_tiny/README.md for how these were derived and verified
# against best_sam_tp.pt (0 missing / 0 unexpected / 0 shape-mismatched keys,
# strict load succeeds).
HF_SAM2_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "sam2_hf_tiny"
HF_SAM2_MODEL_CONFIG_ID = "sam2_hf_tiny"


class HfSam2Wrapper:
    """Adapts ``transformers.Sam2Model`` to the exact ``run_sam2_inference``
    contract the official GENIE-SAMTP ``sam2.sam_tp.SAM_TP`` wrapper exposes,
    so it can be used as a drop-in ``SamTpPredictor`` loader with no changes
    to the validated ``SamTpPredictor``/``SamTpPrediction`` pipeline.

    Point prompts and mask selection intentionally mirror ``SAM_TP``:
    bottom-left/bottom-center/bottom-right foreground points, and a single
    (``multimask_output=False``) mask. Coordinates are passed through
    ``Sam2Processor`` in real image pixel space and it performs the
    (correct) proportional resize-to-1024 normalization itself.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "cuda",
        config_dir: str | Path = HF_SAM2_CONFIG_DIR,
    ) -> None:
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        checkpoint_format, state_dict = load_checkpoint_state_dict(checkpoint_path)
        if checkpoint_format is not CheckpointFormat.HF_SAM2:
            raise CheckpointFormatError(
                f"{checkpoint_path} is {checkpoint_format.value}, not hf_sam2; "
                "use the GENIE SAM_TP backend for this checkpoint instead"
            )

        import torch
        from transformers import Sam2Config, Sam2ImageProcessor, Sam2Model, Sam2Processor

        config_dir = Path(config_dir)
        with (config_dir / "sam2_config.json").open(encoding="utf-8") as handle:
            model_config = json.load(handle)
        with (config_dir / "image_processor_config.json").open(encoding="utf-8") as handle:
            processor_config = json.load(handle)

        model = Sam2Model(Sam2Config(**model_config))
        comparison = compare_state_dicts(model.state_dict(), state_dict)
        if not comparison["compatible"]:
            raise CheckpointFormatError(
                f"{checkpoint_path} does not exactly match the vendored "
                f"{config_dir.name} architecture: "
                f"missing_keys={comparison['missing_keys']} "
                f"unexpected_keys={comparison['unexpected_keys']} "
                f"shape_mismatches={comparison['shape_mismatches']}"
            )
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        model.to(device)

        self.sam2_model = model
        self.processor = Sam2Processor(image_processor=Sam2ImageProcessor(**processor_config))
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.model_config_id = HF_SAM2_MODEL_CONFIG_ID
        self._torch = torch

    def run_sam2_inference(self, input_image_np: np.ndarray) -> dict[str, np.ndarray]:
        torch = self._torch
        height, width = input_image_np.shape[:2]
        bottom_left = [0.0, float(height - 1)]
        bottom_right = [float(width - 1), float(height - 1)]
        bottom_mid = [float((width - 1) // 2), float(height - 1)]

        inputs = self.processor(
            images=input_image_np,
            input_points=[[[bottom_left, bottom_mid, bottom_right]]],
            input_labels=[[[1, 1, 1]]],
            return_tensors="pt",
        )
        inputs = {
            key: (value.to(self.device) if hasattr(value, "to") else value)
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            outputs = self.sam2_model(
                pixel_values=inputs["pixel_values"],
                input_points=inputs["input_points"],
                input_labels=inputs["input_labels"],
                multimask_output=False,
            )
            processed = self.processor.image_processor.post_process_masks(
                [outputs.pred_masks[0]],
                original_sizes=inputs["original_sizes"],
                binarize=False,
            )

        logits = processed[0].squeeze(0).squeeze(0).detach().float().cpu().numpy()
        if logits.shape != (height, width):
            raise CheckpointFormatError(
                f"HF SAM2 output shape {logits.shape} differs from input "
                f"{(height, width)}"
            )
        heatmap = score_to_heatmap(sigmoid_logits(logits))
        return {"heatmap": heatmap, "logits": logits}


def build_sam_tp_predictor(
    upstream_root: str | Path,
    model_config: str | Path,
    checkpoint: str | Path,
    device: str = "cuda",
    synchronize: Any = None,
) -> tuple[SamTpPredictor, CheckpointFormat]:
    """Select and construct the right SAM-TP backend for ``checkpoint``.

    Returns a ``SamTpPredictor`` (the existing, tested validation/timing/
    load-once wrapper) plus the detected ``CheckpointFormat`` so callers can
    surface which backend is actually in use. GENIE-format checkpoints keep
    using the official ``sam2.sam_tp.SAM_TP`` loader unchanged; HF-format
    checkpoints are routed through ``HfSam2Wrapper`` instead.
    """
    checkpoint_format, _ = load_checkpoint_state_dict(checkpoint)
    if checkpoint_format is CheckpointFormat.GENIE_SAM_TP:
        predictor = SamTpPredictor(
            upstream_root,
            model_config,
            checkpoint,
            device=device,
            synchronize=synchronize,
        )
    elif checkpoint_format is CheckpointFormat.HF_SAM2:
        predictor = SamTpPredictor(
            HF_SAM2_CONFIG_DIR,
            HF_SAM2_CONFIG_DIR / "sam2_config.json",
            checkpoint,
            device=device,
            loader=lambda _model_config, checkpoint_path: HfSam2Wrapper(
                checkpoint_path, device=device
            ),
            synchronize=synchronize,
        )
    else:  # pragma: no cover - exhaustive by construction
        raise AssertionError(f"unhandled checkpoint format: {checkpoint_format}")
    return predictor, checkpoint_format
