from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from earth_rover.core.types import CandidateTrajectory, TraversabilityOutput
from earth_rover.perception.sam_tp_adapter import SamTpOutputAdapter
from training.sam_tp_reproduction import SamTpPrediction


class Predictor(Protocol):
    def predict(self, image_rgb: np.ndarray) -> SamTpPrediction: ...


@dataclass(frozen=True)
class Phase1FrameResult:
    """Validated SAM-TP output paired with fixed rover-frame candidates."""

    traversability: TraversabilityOutput
    trajectories: tuple[CandidateTrajectory, ...]
    prediction: SamTpPrediction
    image_path: ImageSpacePathProposal


@dataclass(frozen=True)
class ImageSpacePathProposal:
    """Image-space local path constrained to connected high-score pixels.

    ``target_heading_error_rad`` is projected into image space only to express
    left/right goal preference. Until camera calibration is available this is
    not a metric projection; live consumers must use conservative bounded
    commands and retain independent stop/watchdog gates.
    """

    valid: bool
    points_uv: np.ndarray
    mean_score: float
    minimum_score: float
    reason: str
    target_heading_error_rad: float | None = None
    selected_heading_rad: float | None = None
    heading_residual_rad: float | None = None
    target_uv: tuple[int, int] | None = None
    path_length_px: float = 0.0
    goal_alignment_weight: float = 0.0
    smoothing_method: str = "NONE"
    smoothing_applied: bool = False
    smoothing_iterations: int = 0


class SamTpPhase1FrameProcessor:
    """Run the common RGB frame boundary used by replay and SDK shadow mode."""

    def __init__(
        self,
        predictor: Predictor,
        trajectories: tuple[CandidateTrajectory, ...],
        model_version: str,
        adapter: SamTpOutputAdapter | None = None,
        minimum_path_score: float = 0.55,
        corridor_half_width_ratio: float = 0.018,
        path_replan_interval_frames: int = 3,
        heading_replan_threshold_deg: float = 12.0,
    ) -> None:
        if not trajectories:
            raise ValueError("trajectories must not be empty")
        self.predictor = predictor
        self.trajectories = trajectories
        self.model_version = model_version
        self.adapter = adapter or SamTpOutputAdapter()
        if not 0.0 <= minimum_path_score <= 1.0:
            raise ValueError("minimum_path_score must be in [0, 1]")
        if not 0.0 < corridor_half_width_ratio < 0.25:
            raise ValueError("corridor_half_width_ratio must be in (0, 0.25)")
        if (
            isinstance(path_replan_interval_frames, bool)
            or not isinstance(path_replan_interval_frames, int)
            or path_replan_interval_frames <= 0
        ):
            raise ValueError("path_replan_interval_frames must be a positive integer")
        if (
            not math.isfinite(heading_replan_threshold_deg)
            or heading_replan_threshold_deg < 0.0
        ):
            raise ValueError("heading_replan_threshold_deg must be finite and non-negative")
        self.minimum_path_score = minimum_path_score
        self.corridor_half_width_ratio = corridor_half_width_ratio
        self.path_replan_interval_frames = path_replan_interval_frames
        self.heading_replan_threshold_rad = math.radians(heading_replan_threshold_deg)
        self._previous_image_path_points: np.ndarray | None = None
        self._previous_image_path: ImageSpacePathProposal | None = None
        self._previous_target_heading_error_rad: float | None = None
        self._frames_since_path_replan = 0
        self._consecutive_invalid_paths = 0
        self._maximum_temporal_gap_frames = 5

    def process(
        self,
        image_rgb: np.ndarray,
        frame_timestamp: float,
        target_heading_error_rad: float | None = None,
    ) -> Phase1FrameResult:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("image_rgb must have shape HxWx3")
        if image_rgb.dtype != np.uint8:
            raise ValueError("image_rgb must use uint8 pixels")
        prediction = self.predictor.predict(image_rgb)
        traversability = self.adapter.adapt(
            prediction,
            image_rgb.shape[:2],
            frame_timestamp,
            self.model_version,
        )
        image_path = self._hold_previous_path_if_still_safe(
            traversability.score_map,
            traversability.valid_mask,
            target_heading_error_rad,
        )
        if image_path is None:
            image_path = propose_image_space_path(
                traversability.score_map,
                traversability.valid_mask,
                self.minimum_path_score,
                self.corridor_half_width_ratio,
                target_heading_error_rad=target_heading_error_rad,
                previous_points_uv=self._previous_image_path_points,
            )
        if image_path.valid:
            if image_path.reason.startswith("TEMPORAL_HOLD_"):
                self._frames_since_path_replan += 1
            else:
                self._previous_image_path_points = image_path.points_uv.copy()
                self._previous_image_path = image_path
                self._previous_target_heading_error_rad = target_heading_error_rad
                self._frames_since_path_replan = 0
            self._consecutive_invalid_paths = 0
        else:
            self._consecutive_invalid_paths += 1
            if self._consecutive_invalid_paths > self._maximum_temporal_gap_frames:
                self._previous_image_path_points = None
                self._previous_image_path = None
                self._previous_target_heading_error_rad = None
                self._frames_since_path_replan = 0
        return Phase1FrameResult(
            traversability=traversability,
            trajectories=self.trajectories,
            prediction=prediction,
            image_path=image_path,
        )

    def _hold_previous_path_if_still_safe(
        self,
        score_map: np.ndarray,
        valid_mask: np.ndarray,
        target_heading_error_rad: float | None,
    ) -> ImageSpacePathProposal | None:
        previous = self._previous_image_path
        if previous is None or not previous.valid:
            return None
        if self._frames_since_path_replan >= self.path_replan_interval_frames - 1:
            return None
        if _heading_changed_too_much(
            self._previous_target_heading_error_rad,
            target_heading_error_rad,
            self.heading_replan_threshold_rad,
        ):
            return None
        points = np.asarray(previous.points_uv, dtype=np.int32)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
            return None
        score = np.asarray(score_map, dtype=np.float32)
        valid = np.asarray(valid_mask, dtype=bool)
        if score.ndim != 2 or valid.shape != score.shape:
            return None
        if (
            np.any(points[:, 0] < 0)
            or np.any(points[:, 0] >= score.shape[1])
            or np.any(points[:, 1] < 0)
            or np.any(points[:, 1] >= score.shape[0])
        ):
            return None
        hold_threshold = max(0.30, self.minimum_path_score - 0.20)
        safe = valid & (score >= hold_threshold)
        if not _path_inside_safe_corridor(points, safe):
            return None
        sampled_scores = score[points[:, 1], points[:, 0]]
        if float(sampled_scores.mean()) < self.minimum_path_score * 0.85:
            return None
        deltas = np.diff(points.astype(np.float64), axis=0)
        selected_heading = previous.selected_heading_rad
        heading_residual = (
            None
            if target_heading_error_rad is None or selected_heading is None
            else target_heading_error_rad - selected_heading
        )
        held_points = points.copy()
        held_points.setflags(write=False)
        return ImageSpacePathProposal(
            valid=True,
            points_uv=held_points,
            mean_score=float(sampled_scores.mean()),
            minimum_score=float(sampled_scores.min()),
            reason=f"TEMPORAL_HOLD_{previous.reason}",
            target_heading_error_rad=target_heading_error_rad,
            selected_heading_rad=selected_heading,
            heading_residual_rad=heading_residual,
            target_uv=previous.target_uv,
            path_length_px=float(np.linalg.norm(deltas, axis=1).sum()),
            goal_alignment_weight=previous.goal_alignment_weight,
            smoothing_method="TEMPORAL_PATH_HOLD",
            smoothing_applied=True,
            smoothing_iterations=0,
        )


def propose_image_space_path(
    score_map: np.ndarray,
    valid_mask: np.ndarray,
    minimum_score: float,
    corridor_half_width_ratio: float,
    target_heading_error_rad: float | None = None,
    maximum_visual_heading_deg: float = 55.0,
    goal_alignment_weight: float = 1.35,
    previous_points_uv: np.ndarray | None = None,
) -> ImageSpacePathProposal:
    """Find a bottom-to-top image path without crossing low-score pixels.

    Pixel displacement has no calibrated metric relationship to rover
    curvature, so the live Mission1 controller treats it only as a bounded
    experimental left/right heading signal.
    """

    score = np.asarray(score_map, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    if score.ndim != 2 or valid.shape != score.shape:
        raise ValueError("score_map and valid_mask must be matching 2D arrays")
    if not np.isfinite(score).all() or score.size == 0:
        raise ValueError("score_map must be finite and non-empty")
    if not 0.0 <= minimum_score <= 1.0:
        raise ValueError("minimum_score must be in [0, 1]")
    if not 0.0 < corridor_half_width_ratio < 0.25:
        raise ValueError("corridor_half_width_ratio must be in (0, 0.25)")
    if not math.isfinite(maximum_visual_heading_deg) or maximum_visual_heading_deg <= 0.0:
        raise ValueError("maximum_visual_heading_deg must be finite and positive")
    if not math.isfinite(goal_alignment_weight) or goal_alignment_weight < 0.0:
        raise ValueError("goal_alignment_weight must be finite and non-negative")
    if target_heading_error_rad is not None:
        target_heading_error_rad = float(target_heading_error_rad)
        if not math.isfinite(target_heading_error_rad):
            raise ValueError("target_heading_error_rad must be finite when provided")

    height, width = score.shape
    half_width = max(1, round(width * corridor_half_width_ratio))
    start_y = min(height - 1, max(0, round(height * 0.92)))
    # The uncalibrated live camera's ground plane begins near mid-frame.  Do
    # not force an image-space path into sky/buildings above that boundary.
    end_y = min(start_y, max(0, round(height * 0.52)))
    row_step = max(1, height // 50)
    rows = list(range(start_y, end_y - 1, -row_step))
    if rows[-1] != end_y:
        rows.append(end_y)
    x_step = max(1, width // 96)
    xs = np.arange(0, width, x_step, dtype=np.int32)
    if xs[-1] != width - 1:
        xs = np.append(xs, width - 1)
    maximum_lateral_pixels = max(x_step, round(width * 0.055))
    maximum_index_change = max(1, maximum_lateral_pixels // x_step)
    center_x = (width - 1) / 2.0
    maximum_visual_heading_rad = math.radians(maximum_visual_heading_deg)
    if target_heading_error_rad is None:
        target_x = center_x
    else:
        normalized_heading = float(
            np.clip(
                target_heading_error_rad / maximum_visual_heading_rad,
                -1.0,
                1.0,
            )
        )
        # Positive rover heading error means left, which is smaller image u.
        target_x = center_x - normalized_heading * width * 0.42

    thresholds = _path_threshold_attempts(minimum_score)
    half_width_attempts = tuple(
        dict.fromkeys((half_width, max(1, half_width // 2), 1))
    )
    search_result = None
    for allow_partial in (False, True):
        for threshold in thresholds:
            for candidate_half_width in half_width_attempts:
                search_result = _search_connected_image_path(
                    score,
                    valid,
                    threshold,
                    candidate_half_width,
                    rows,
                    xs,
                    center_x,
                    target_x,
                    maximum_lateral_pixels,
                    maximum_index_change,
                    goal_alignment_weight,
                    allow_partial=allow_partial,
                )
                if search_result is not None:
                    break
            if search_result is not None:
                break
        if search_result is not None:
            break

    if search_result is None:
        return _invalid_image_path(
            "NO_CONNECTED_TRAVERSABLE_PATH",
            target_heading_error_rad,
            (int(round(target_x)), end_y),
            goal_alignment_weight,
        )
    points, corridor_safe, used_threshold, used_half_width, used_partial = search_result
    points, smoothing_applied = smooth_image_path_spline(
        points,
        corridor_safe,
        iterations=5,
    )
    points, temporal_smoothing_applied = smooth_image_path_temporally(
        points,
        previous_points_uv,
        corridor_safe,
        current_weight=0.45,
    )
    centerline = np.zeros(score.shape, dtype=np.uint8)
    cv2.polylines(centerline, [points], False, 1, 1, cv2.LINE_8)
    if np.any((centerline == 1) & ~corridor_safe):
        return _invalid_image_path(
            "NO_CONNECTED_TRAVERSABLE_PATH",
            target_heading_error_rad,
            (int(round(target_x)), end_y),
            goal_alignment_weight,
        )
    sampled_scores = score[points[:, 1], points[:, 0]]
    deltas = np.diff(points.astype(np.float64), axis=0)
    path_length_px = float(np.linalg.norm(deltas, axis=1).sum())
    selected_ratio = float(
        np.clip((center_x - float(points[-1, 0])) / (width * 0.42), -1.0, 1.0)
    )
    selected_heading = selected_ratio * maximum_visual_heading_rad
    heading_residual = (
        None
        if target_heading_error_rad is None
        else target_heading_error_rad - selected_heading
    )
    points.setflags(write=False)
    relaxed = used_threshold < minimum_score or used_half_width < half_width
    partial = used_partial
    return ImageSpacePathProposal(
        valid=True,
        points_uv=points,
        mean_score=float(sampled_scores.mean()),
        minimum_score=float(sampled_scores.min()),
        reason=_image_path_reason(
            goal_aligned=target_heading_error_rad is not None,
            relaxed=relaxed,
            partial=partial,
        ),
        target_heading_error_rad=target_heading_error_rad,
        selected_heading_rad=selected_heading,
        heading_residual_rad=heading_residual,
        target_uv=(int(round(target_x)), end_y),
        path_length_px=path_length_px,
        goal_alignment_weight=goal_alignment_weight,
        smoothing_method=(
            "CONSTRAINED_CUBIC_B_SPLINE_TEMPORAL_EMA"
            if temporal_smoothing_applied
            else "CONSTRAINED_CUBIC_B_SPLINE"
        ),
        smoothing_applied=smoothing_applied or temporal_smoothing_applied,
        smoothing_iterations=5,
    )


def _image_path_reason(*, goal_aligned: bool, relaxed: bool, partial: bool) -> str:
    prefix = ""
    if partial:
        prefix += "PARTIAL_"
    if relaxed:
        prefix += "RELAXED_"
    if goal_aligned:
        return prefix + "GPS_HEADING_ALIGNED_TRAVERSABLE_PATH"
    if not prefix:
        return "CONNECTED_HIGH_TRAVERSABILITY_IMAGE_PATH"
    return prefix + "CONNECTED_TRAVERSABILITY_IMAGE_PATH"


def _heading_changed_too_much(
    previous: float | None,
    current: float | None,
    threshold_rad: float,
) -> bool:
    if previous is None and current is None:
        return False
    if previous is None or current is None:
        return True
    return abs(math.atan2(math.sin(current - previous), math.cos(current - previous))) > threshold_rad


def _path_threshold_attempts(minimum_score: float) -> tuple[float, ...]:
    attempts = []
    for threshold in (
        minimum_score,
        minimum_score - 0.10,
        minimum_score - 0.20,
        0.30,
    ):
        clipped = max(0.0, min(float(threshold), float(minimum_score)))
        if clipped not in attempts:
            attempts.append(clipped)
    return tuple(attempts)


def _search_connected_image_path(
    score: np.ndarray,
    valid: np.ndarray,
    threshold: float,
    half_width: int,
    rows: list[int],
    xs: np.ndarray,
    center_x: float,
    target_x: float,
    maximum_lateral_pixels: int,
    maximum_index_change: int,
    goal_alignment_weight: float,
    allow_partial: bool = False,
) -> tuple[np.ndarray, np.ndarray, float, int, bool] | None:
    safe = (valid & (score >= threshold)).astype(np.uint8)
    corridor_safe = cv2.erode(
        safe,
        np.ones((1, half_width * 2 + 1), dtype=np.uint8),
        iterations=1,
    ).astype(bool)
    scores = np.full(xs.shape, -np.inf, dtype=np.float64)
    start_allowed = corridor_safe[rows[0], xs]
    center_penalty = 0.08 * np.abs(xs - center_x) / max(score.shape[1] / 2.0, 1.0)
    scores[start_allowed] = (
        score[rows[0], xs[start_allowed]] - center_penalty[start_allowed]
    )
    if not np.isfinite(scores).any():
        return None
    parents: list[np.ndarray] = []
    best_scores = scores.copy()
    best_row_index = 0
    best_parents: list[np.ndarray] = []
    for row_index, row in enumerate(rows[1:], start=1):
        next_scores = np.full(xs.shape, -np.inf, dtype=np.float64)
        parent = np.full(xs.shape, -1, dtype=np.int32)
        progress = row_index / max(len(rows) - 1, 1)
        desired_x = center_x + progress * (target_x - center_x)
        for index in np.flatnonzero(corridor_safe[row, xs]):
            left = max(0, index - maximum_index_change)
            right = min(len(xs), index + maximum_index_change + 1)
            previous = scores[left:right]
            finite = np.isfinite(previous)
            if not finite.any():
                continue
            candidate_indexes = np.arange(left, right)[finite]
            candidate_scores = previous[finite] - (
                0.12
                * np.abs(xs[candidate_indexes] - xs[index])
                / maximum_lateral_pixels
            )
            best_offset = int(np.argmax(candidate_scores))
            parent[index] = int(candidate_indexes[best_offset])
            next_scores[index] = (
                float(candidate_scores[best_offset]) + float(score[row, xs[index]])
                - goal_alignment_weight
                * progress
                * abs(float(xs[index]) - desired_x)
                / max(score.shape[1] / 2.0, 1.0)
            )
        parents.append(parent)
        scores = next_scores
        if np.isfinite(scores).any():
            best_scores = scores.copy()
            best_row_index = row_index
            best_parents = list(parents)
    if not np.isfinite(scores).any():
        minimum_partial_rows = max(3, math.ceil(len(rows) * 0.35))
        if not allow_partial or best_row_index + 1 < minimum_partial_rows:
            return None
        scores = best_scores
        parents = best_parents
        rows = rows[: best_row_index + 1]
        used_partial = True
    else:
        used_partial = False
    current = int(np.argmax(scores))
    indexes = [current]
    for parent in reversed(parents):
        current = int(parent[current])
        if current < 0:
            return None
        indexes.append(current)
    indexes.reverse()
    points = np.column_stack(
        (
            xs[np.asarray(indexes, dtype=np.int32)],
            np.asarray(rows, dtype=np.int32),
        )
    ).astype(np.int32)
    return points, corridor_safe, float(threshold), int(half_width), used_partial


def _invalid_image_path(
    reason: str,
    target_heading_error_rad: float | None = None,
    target_uv: tuple[int, int] | None = None,
    goal_alignment_weight: float = 0.0,
) -> ImageSpacePathProposal:
    points = np.empty((0, 2), dtype=np.int32)
    points.setflags(write=False)
    return ImageSpacePathProposal(
        valid=False,
        points_uv=points,
        mean_score=0.0,
        minimum_score=0.0,
        reason=reason,
        target_heading_error_rad=target_heading_error_rad,
        target_uv=target_uv,
        goal_alignment_weight=goal_alignment_weight,
    )


def smooth_image_path_spline(
    points_uv: np.ndarray,
    safe_corridor: np.ndarray,
    iterations: int = 3,
) -> tuple[np.ndarray, bool]:
    """Apply a bounded cubic B-spline filter without leaving safe pixels.

    Only image ``u`` is smoothed; the monotonic row samples remain unchanged.
    Repeated ``[1, 4, 6, 4, 1] / 16`` filtering is a small uniform cubic
    B-spline approximation.  A short line search reduces smoothing if a curve
    would cut across non-traversable pixels.  The original valid path is the
    deterministic fallback, keeping runtime linear in the number of points.
    """

    points = np.asarray(points_uv)
    safe = np.asarray(safe_corridor, dtype=bool)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
        raise ValueError("points_uv must be a non-empty Nx2 array")
    if safe.ndim != 2 or safe.size == 0:
        raise ValueError("safe_corridor must be a non-empty 2D array")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 0:
        raise ValueError("iterations must be a non-negative integer")

    original = points.astype(np.float64, copy=True)
    if iterations == 0 or len(points) < 5:
        result = points.astype(np.int32, copy=True)
        result.setflags(write=False)
        return result, False

    smoothed_x = original[:, 0].copy()
    kernel = np.asarray([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float64) / 16.0
    for _ in range(iterations):
        padded = np.pad(smoothed_x, (2, 2), mode="edge")
        filtered = np.convolve(padded, kernel, mode="valid")
        filtered[0] = original[0, 0]
        filtered[-1] = original[-1, 0]
        smoothed_x = filtered

    maximum_x = safe.shape[1] - 1
    for strength in (1.0, 0.75, 0.5, 0.25):
        candidate = original.copy()
        candidate[:, 0] = np.clip(
            original[:, 0] + strength * (smoothed_x - original[:, 0]),
            0,
            maximum_x,
        )
        rounded = np.rint(candidate).astype(np.int32)
        if _path_inside_safe_corridor(rounded, safe):
            rounded.setflags(write=False)
            changed = not np.array_equal(rounded, points.astype(np.int32))
            return rounded, changed

    # A global smoothing strength can fail when only one short section hugs a
    # safe-corridor boundary. Smooth each interior control point locally so a
    # single constrained bend does not force the entire spline to fall back.
    partial = points.astype(np.int32, copy=True)
    changed = False
    target_x = np.rint(smoothed_x).astype(np.int32)
    for index in range(1, len(partial) - 1):
        original_x = int(partial[index, 0])
        delta = int(target_x[index]) - original_x
        for strength in (1.0, 0.75, 0.5, 0.25):
            candidate_x = int(round(original_x + strength * delta))
            if candidate_x == original_x:
                continue
            candidate = partial.copy()
            candidate[index, 0] = np.clip(candidate_x, 0, maximum_x)
            if _path_inside_safe_corridor(candidate, safe):
                partial = candidate
                changed = True
                break
    partial.setflags(write=False)
    return partial, changed


def smooth_image_path_temporally(
    current_points_uv: np.ndarray,
    previous_points_uv: np.ndarray | None,
    safe_corridor: np.ndarray,
    current_weight: float = 0.45,
) -> tuple[np.ndarray, bool]:
    """Reduce frame-to-frame path flicker while retaining current safe pixels."""

    current = np.asarray(current_points_uv, dtype=np.int32)
    safe = np.asarray(safe_corridor, dtype=bool)
    if not 0.0 < current_weight <= 1.0:
        raise ValueError("current_weight must be in (0, 1]")
    if previous_points_uv is None:
        result = current.copy()
        result.setflags(write=False)
        return result, False
    previous = np.asarray(previous_points_uv, dtype=np.int32)
    if previous.shape != current.shape or not np.array_equal(previous[:, 1], current[:, 1]):
        result = current.copy()
        result.setflags(write=False)
        return result, False

    for weight in (current_weight, 0.60, 0.75, 0.90):
        candidate = current.copy()
        candidate[1:-1, 0] = np.rint(
            weight * current[1:-1, 0]
            + (1.0 - weight) * previous[1:-1, 0]
        ).astype(np.int32)
        if _path_inside_safe_corridor(candidate, safe):
            candidate.setflags(write=False)
            return candidate, not np.array_equal(candidate, current)

    result = current.copy()
    result.setflags(write=False)
    return result, False


def _path_inside_safe_corridor(points_uv: np.ndarray, safe: np.ndarray) -> bool:
    xs = points_uv[:, 0]
    ys = points_uv[:, 1]
    if (
        np.any(xs < 0)
        or np.any(xs >= safe.shape[1])
        or np.any(ys < 0)
        or np.any(ys >= safe.shape[0])
        or np.any(~safe[ys, xs])
    ):
        return False
    centerline = np.zeros(safe.shape, dtype=np.uint8)
    cv2.polylines(centerline, [points_uv], False, 1, 1, cv2.LINE_8)
    return not np.any((centerline == 1) & ~safe)


def draw_image_path_rgb(
    image_rgb: np.ndarray,
    proposal: ImageSpacePathProposal,
    corridor_half_width_ratio: float,
) -> np.ndarray:
    """Draw a display-only path on an RGB frame without changing its geometry."""

    output = image_rgb.copy()
    if proposal.target_uv is not None:
        start = (image_rgb.shape[1] // 2, min(image_rgb.shape[0] - 1, round(image_rgb.shape[0] * 0.92)))
        target_behind = (
            proposal.target_heading_error_rad is not None
            and abs(proposal.target_heading_error_rad) > math.pi / 2.0
        )
        if target_behind:
            # target_heading_error_rad follows the codebase-wide
            # positive_clockwise_right convention (see heading_convention in
            # motion_primitive_planner.py), so a positive error means the
            # goal is to the right, not the left.
            left = proposal.target_heading_error_rad < 0.0
            arrow_y = min(image_rgb.shape[0] - 8, round(image_rgb.shape[0] * 0.86))
            arrow_end = (
                max(8, round(image_rgb.shape[1] * 0.08)) if left
                else min(image_rgb.shape[1] - 8, round(image_rgb.shape[1] * 0.92)),
                arrow_y,
            )
            cv2.arrowedLine(
                output,
                (image_rgb.shape[1] // 2, arrow_y),
                arrow_end,
                (40, 230, 230),
                3,
                cv2.LINE_AA,
                tipLength=0.12,
            )
            cv2.putText(
                output,
                f"GOAL BEHIND {'LEFT' if left else 'RIGHT'}",
                (8, max(18, arrow_y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (40, 230, 230),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.line(output, start, proposal.target_uv, (40, 230, 230), 2, cv2.LINE_AA)
            cv2.circle(output, proposal.target_uv, 5, (40, 230, 230), -1, cv2.LINE_AA)
    if not proposal.valid:
        return output
    half_width = max(1, round(image_rgb.shape[1] * corridor_half_width_ratio))
    overlay = output.copy()
    cv2.polylines(
        overlay,
        [proposal.points_uv],
        False,
        (40, 210, 80),
        half_width * 2,
        cv2.LINE_AA,
    )
    output = cv2.addWeighted(output, 0.62, overlay, 0.38, 0.0)
    cv2.polylines(
        output,
        [proposal.points_uv],
        False,
        (255, 255, 255),
        max(2, half_width // 3),
        cv2.LINE_AA,
    )
    return output


def render_trajectory_geometry_rgb(
    trajectories: tuple[CandidateTrajectory, ...],
    width: int,
    height: int,
) -> np.ndarray:
    """Render rover-frame geometry only; this is not a camera projection."""

    if width <= 0 or height <= 0:
        raise ValueError("render dimensions must be positive")
    if not trajectories:
        raise ValueError("trajectories must not be empty")
    canvas = np.full((height, width, 3), 18, dtype=np.uint8)
    all_points = np.concatenate(
        [
            trajectory.points_xy
            for trajectory in trajectories
        ]
        + [
            trajectory.left_boundary_xy
            for trajectory in trajectories
        ]
        + [
            trajectory.right_boundary_xy
            for trajectory in trajectories
        ],
        axis=0,
    )
    maximum_x = max(float(all_points[:, 0].max()), 0.1)
    maximum_abs_y = max(float(np.abs(all_points[:, 1]).max()), 0.1)
    margin = 24

    def pixel(points: np.ndarray) -> np.ndarray:
        x_pixels = margin + points[:, 0] / maximum_x * (width - margin * 2)
        y_pixels = height / 2.0 - points[:, 1] / maximum_abs_y * (
            height / 2.0 - margin
        )
        return np.rint(np.column_stack((x_pixels, y_pixels))).astype(np.int32)

    cv2.line(canvas, (margin, height // 2), (width - margin, height // 2), (80, 80, 80), 1)
    for trajectory in trajectories:
        color = (60, 220, 60) if trajectory.curvature == 0.0 else (80, 180, 245)
        cv2.polylines(canvas, [pixel(trajectory.left_boundary_xy)], False, (70, 70, 70), 1)
        cv2.polylines(canvas, [pixel(trajectory.right_boundary_xy)], False, (70, 70, 70), 1)
        centerline = pixel(trajectory.points_xy)
        cv2.polylines(canvas, [centerline], False, color, 2)
        endpoint = tuple(int(value) for value in centerline[-1])
        cv2.circle(canvas, endpoint, 3, color, -1)
    cv2.putText(
        canvas,
        "ROVER FRAME: +x forward, +y left",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (225, 225, 225),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "GEOMETRY ONLY - NOT CAMERA PROJECTED",
        (10, height - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (80, 180, 245),
        1,
        cv2.LINE_AA,
    )
    return canvas
