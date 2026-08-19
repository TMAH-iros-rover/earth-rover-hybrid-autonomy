from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from earth_rover.planning.motion_primitive_planner import MotionPrimitivePlanner
from training.sam_tp_event_recorder import ReplayCaptureFrame


@dataclass
class ReplayClock:
    value: float = 0.0

    def __call__(self) -> float:
        return self.value


def evaluate_replay_frames(
    frames: list[ReplayCaptureFrame],
    predictors: dict[str, Any],
    planner_config: dict[str, Any],
    *,
    nominal_fps: float = 4.0,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, np.ndarray]]]:
    """Run both backends on identical frames and preserve real sequence gaps."""

    if nominal_fps <= 0.0 or not math.isfinite(nominal_fps):
        raise ValueError("nominal_fps must be finite and positive")
    if set(predictors) != {"baseline", "candidate"}:
        raise ValueError("predictors must contain baseline and candidate")
    clocks = {name: ReplayClock() for name in predictors}
    planners = {
        name: MotionPrimitivePlanner(planner_config, monotonic=clocks[name])
        for name in predictors
    }
    rows: list[dict[str, Any]] = []
    score_maps: dict[int, dict[str, np.ndarray]] = {}
    previous_index: int | None = None
    for frame in sorted(frames, key=lambda item: item.frame_index):
        if previous_index is not None and frame.frame_index != previous_index + 1:
            for planner in planners.values():
                planner.reset()
        timestamp = frame.frame_index / nominal_fps
        navigation = _mapping(frame.record.get("navigation"))
        target_heading_deg = _finite(navigation.get("heading_error_deg"))
        target_heading_rad = (
            None if target_heading_deg is None else math.radians(target_heading_deg)
        )
        target_sequence = _integer(navigation.get("target_sequence"))
        image_rgb = cv2.cvtColor(frame.source_bgr, cv2.COLOR_BGR2RGB)
        model_rows: dict[str, dict[str, Any]] = {}
        model_scores: dict[str, np.ndarray] = {}
        for name, predictor in predictors.items():
            clocks[name].value = timestamp
            prediction = predictor.predict(image_rgb)
            score = np.asarray(prediction.traversability_score, dtype=np.float32)
            if score.shape != frame.source_bgr.shape[:2] or not np.isfinite(score).all():
                raise ValueError(
                    f"{name} score shape/values invalid at frame {frame.frame_index}"
                )
            plan = planners[name].plan(
                score,
                np.ones_like(score, dtype=bool),
                target_heading_error_rad=target_heading_rad,
                checkpoint_sequence=target_sequence,
                timestamp=timestamp,
                navigation=navigation,
            )
            model_rows[name] = {
                "inference_time_ms": float(prediction.inference_time_ms),
                "score_mean": float(np.mean(score)),
                "score_std": float(np.std(score)),
                "planner": plan.to_status(include_candidates=False),
            }
            model_scores[name] = score.copy()
        baseline_score = model_scores["baseline"]
        candidate_score = model_scores["candidate"]
        baseline_mask = baseline_score >= 0.5
        candidate_mask = candidate_score >= 0.5
        union = int(np.count_nonzero(baseline_mask | candidate_mask))
        intersection = int(np.count_nonzero(baseline_mask & candidate_mask))
        rows.append(
            {
                "frame_index": frame.frame_index,
                "contiguous_with_previous": (
                    previous_index is not None and frame.frame_index == previous_index + 1
                ),
                "target_heading_error_deg": target_heading_deg,
                "target_sequence": target_sequence,
                "capture_event_reasons": frame.record.get("capture_event_reasons", []),
                "baseline": model_rows["baseline"],
                "candidate": model_rows["candidate"],
                "score_mae": float(np.mean(np.abs(baseline_score - candidate_score))),
                "mask_iou_at_0_5": 1.0 if union == 0 else intersection / union,
            }
        )
        score_maps[frame.frame_index] = model_scores
        previous_index = frame.frame_index
    return rows, score_maps


def summarize_ab_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    headings = {
        name: [_selected_heading(row, name) for row in rows]
        for name in ("baseline", "candidate")
    }
    paired_deltas = [
        abs(_angle_delta(candidate, baseline))
        for baseline, candidate in zip(headings["baseline"], headings["candidate"])
        if baseline is not None and candidate is not None
    ]
    summary: dict[str, Any] = {
        "frame_count": len(rows),
        "paired_valid_heading_count": len(paired_deltas),
        "selected_heading_disagreement_fraction": _mean(
            [delta > 1e-6 for delta in paired_deltas]
        ),
        "selected_heading_abs_delta_deg_mean": _mean(paired_deltas),
        "selected_heading_abs_delta_deg_max": max(paired_deltas, default=None),
        "score_mae_mean": _mean([row.get("score_mae") for row in rows]),
        "mask_iou_at_0_5_mean": _mean(
            [row.get("mask_iou_at_0_5") for row in rows]
        ),
    }
    for name in ("baseline", "candidate"):
        valid = [value for value in headings[name] if value is not None]
        transitions: list[float] = []
        sign_flips = 0
        for index in range(1, len(rows)):
            if not rows[index].get("contiguous_with_previous"):
                continue
            before = headings[name][index - 1]
            after = headings[name][index]
            if before is None or after is None:
                continue
            delta = abs(_angle_delta(after, before))
            transitions.append(delta)
            if before * after < 0.0 and abs(before) >= 5.0 and abs(after) >= 5.0:
                sign_flips += 1
        summary[name] = {
            "valid_heading_count": len(valid),
            "unsafe_or_stop_frame_count": sum(
                _switch_stop_required(row, name) or headings[name][index] is None
                for index, row in enumerate(rows)
            ),
            "inference_time_ms_mean": _mean(
                [_mapping(row.get(name)).get("inference_time_ms") for row in rows]
            ),
            "contiguous_heading_change_count": sum(delta > 1e-6 for delta in transitions),
            "contiguous_heading_jump_over_20_deg_count": sum(
                delta >= 20.0 for delta in transitions
            ),
            "contiguous_left_right_flip_count": sign_flips,
            "contiguous_heading_delta_deg_max": max(transitions, default=None),
        }
    return summary


def render_ab_overlay(
    source_bgr: np.ndarray,
    scores: dict[str, np.ndarray],
    row: dict[str, Any],
) -> np.ndarray:
    panels = [source_bgr]
    for name in ("baseline", "candidate"):
        heatmap = cv2.applyColorMap(
            np.rint(np.clip(scores[name], 0.0, 1.0) * 255.0).astype(np.uint8),
            cv2.COLORMAP_JET,
        )
        panel = cv2.addWeighted(source_bgr, 0.45, heatmap, 0.55, 0.0)
        heading = _selected_heading(row, name)
        label = f"{name}: STOP" if heading is None else f"{name}: {heading:+.0f} deg"
        cv2.putText(
            panel,
            label,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        panels.append(panel)
    return np.concatenate(panels, axis=1)


def _selected_heading(row: dict[str, Any], name: str) -> float | None:
    planner = _mapping(_mapping(row.get(name)).get("planner"))
    return _finite(planner.get("selected_candidate_heading_deg"))


def _switch_stop_required(row: dict[str, Any], name: str) -> bool:
    planner = _mapping(_mapping(row.get(name)).get("planner"))
    return planner.get("switch_stop_required") is True


def _angle_delta(after: float, before: float) -> float:
    return (after - before + 180.0) % 360.0 - 180.0


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _finite(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _integer(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[Any]) -> float | None:
    finite = [_finite(value) for value in values]
    filtered = [value for value in finite if value is not None]
    return None if not filtered else float(np.mean(filtered))
