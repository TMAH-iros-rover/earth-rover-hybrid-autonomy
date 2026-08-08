from __future__ import annotations

from dataclasses import dataclass
from typing import Any


ACTION_NAMES = ("STOP", "FORWARD", "LEFT", "RIGHT", "REVERSE")


@dataclass(frozen=True)
class ActionThresholds:
    stop_linear_abs: float = 0.05
    stop_angular_abs: float = 0.05
    reverse_linear: float = -0.10
    turn_angular: float = 0.25


def classify_action(linear: float, angular: float, thresholds: ActionThresholds | None = None) -> str:
    thresholds = thresholds or ActionThresholds()
    if linear < thresholds.reverse_linear:
        return "REVERSE"
    if abs(linear) < thresholds.stop_linear_abs and abs(angular) < thresholds.stop_angular_abs:
        return "STOP"
    if angular > thresholds.turn_angular:
        return "LEFT"
    if angular < -thresholds.turn_angular:
        return "RIGHT"
    return "FORWARD"


def action_to_linear_angular(action: Any) -> tuple[float, float] | None:
    """Best-effort conversion for dataset action values.

    The Berkeley/FrodoBots datasets may expose actions as arrays, tuples, or
    dict-like values depending on whether they are read from Zarr, WebDataset,
    or HF's parquet viewer. This keeps exploration code format-tolerant.
    """
    if action is None:
        return None
    if isinstance(action, dict):
        linear = _first_present(action, ("linear", "lin", "throttle", "speed", "v"))
        angular = _first_present(action, ("angular", "ang", "steering", "turn", "omega", "w"))
        if linear is None or angular is None:
            return None
        return float(linear), float(angular)
    if hasattr(action, "tolist"):
        action = action.tolist()
    if isinstance(action, (list, tuple)) and len(action) >= 2:
        return float(action[0]), float(action[1])
    return None


def _first_present(values: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in values:
            return values[name]
    return None
