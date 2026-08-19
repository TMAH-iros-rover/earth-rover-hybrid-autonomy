from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any


class CheckpointFormat(Enum):
    """The two SAM-TP checkpoint formats supported by the live pipeline."""

    GENIE_SAM_TP = "genie_sam_tp"
    HF_SAM2 = "hf_sam2"


class CheckpointFormatError(ValueError):
    """A checkpoint file did not match a recognized SAM-TP format."""


# A handful of keys unique to the Hugging Face transformers.Sam2Model export
# of best_sam_tp.pt. Requiring all of them (rather than a loose heuristic
# like "top level is a bare tensor dict") keeps detection strict: an
# unrelated or truncated bare state dict is rejected instead of silently
# guessed as hf_sam2.
_HF_SAM2_SIGNATURE_KEYS = (
    "no_memory_embedding",
    "shared_image_embedding.positional_embedding",
    "vision_encoder.backbone.patch_embed.projection.weight",
    "prompt_encoder.point_embed.weight",
    "mask_decoder.iou_token.weight",
)


def load_checkpoint_state_dict(path: str | Path) -> tuple[CheckpointFormat, dict[str, Any]]:
    """Load a SAM-TP checkpoint and strictly classify its format.

    Returns the detected format and the flat tensor state dict, already
    unwrapped from GENIE's ``{"model": state_dict, ...}`` container when
    present. Raises ``CheckpointFormatError`` for anything that is not
    unambiguously one of the two recognized formats -- this never falls
    back to a best-effort guess.
    """
    import torch

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {resolved}")
    checkpoint = torch.load(resolved, map_location="cpu", weights_only=True)

    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), dict):
        state_dict = checkpoint["model"]
        if not state_dict or not all(hasattr(value, "shape") for value in state_dict.values()):
            raise CheckpointFormatError(
                f"{resolved} has a GENIE-shaped 'model' key but its values are "
                "not tensors"
            )
        return CheckpointFormat.GENIE_SAM_TP, state_dict

    if isinstance(checkpoint, dict) and checkpoint and all(
        hasattr(value, "shape") for value in checkpoint.values()
    ):
        missing_signature = [key for key in _HF_SAM2_SIGNATURE_KEYS if key not in checkpoint]
        if missing_signature:
            raise CheckpointFormatError(
                f"{resolved} is a bare tensor state dict but is missing expected "
                f"Hugging Face SAM2 keys {missing_signature}; refusing to guess "
                f"a backend. Top-level key sample: {sorted(checkpoint)[:5]}"
            )
        return CheckpointFormat.HF_SAM2, checkpoint

    top_level = sorted(checkpoint.keys()) if isinstance(checkpoint, dict) else None
    raise CheckpointFormatError(
        f"{resolved} does not match a recognized SAM-TP checkpoint format "
        "(expected a GENIE {'model': state_dict, ...} container or a bare "
        f"Hugging Face SAM2 state dict); top-level type="
        f"{type(checkpoint).__name__} keys={top_level}"
    )
