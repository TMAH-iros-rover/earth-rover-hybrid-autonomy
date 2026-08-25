from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from earth_rover.core.types import CandidateTrajectory
from earth_rover.perception.camera_calibration import CalibrationError, CameraCalibration, validate_for_live_use
from earth_rover.perception.camera_projection import project_trajectory
from earth_rover.planning.trajectory_sampler import (
    DEFAULT_CURVATURES,
    ConstantCurvatureTrajectorySampler,
)
from training.sam_tp_phase1_review import ImageSpacePathProposal


def normalize_angle_deg(value: float) -> float:
    """Return an angle in [-180, 180)."""

    return (float(value) + 180.0) % 360.0 - 180.0


def normalize_angle_rad(value: float) -> float:
    return math.atan2(math.sin(float(value)), math.cos(float(value)))


def metric_terminal_heading_deg(curvature: float, horizon_m: float) -> float:
    """Convert a rover-frame constant-curvature terminal heading to the
    controller's positive-clockwise-right convention.

    The rover frame uses standard math convention: positive curvature turns
    left, and the terminal heading at arc length ``horizon_m`` is
    ``curvature * horizon_m`` radians, positive being counter-clockwise
    (left). The planner/controller convention used everywhere else in this
    module is the opposite: positive heading means clockwise/right. The
    conversion is therefore a sign flip, kept explicit here rather than
    folded silently into a projection step.
    """

    return -math.degrees(float(curvature) * float(horizon_m))


@dataclass(frozen=True)
class MotionPrimitivePlannerConfig:
    mode: str = "motion_primitives"
    candidate_headings_deg: tuple[float, ...] = (-45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0)
    traversability_weight: float = 1.0
    goal_heading_weight: float = 0.7
    continuity_weight: float = 0.45
    curvature_weight: float = 0.10
    near_field_risk_weight: float = 1.5
    # near_field_risk_weight only applies once near_low drops below
    # near_field_stop_threshold -- above that, two candidates score
    # identically (zero risk penalty) regardless of whether one is at 0.61
    # and the other at 0.99. That let goal_heading_weight fully decide
    # between two "safe by threshold" candidates, including picking one
    # that's markedly closer to an obstacle just because it's better
    # goal-aligned -- observed as the rover trying to drive back toward a
    # wall instead of the open road once it was already close to one. This
    # adds a small continuous penalty on (1 - near_low) with no threshold,
    # so a candidate that's merely "safe enough" still loses some ground to
    # a candidate that's clearly safer, even before near_field_risk_weight
    # engages.
    near_field_soft_risk_weight: float = 0.5
    path_score_threshold: float = 0.45
    near_field_stop_threshold: float = 0.25
    near_field_percentile: float = 10.0
    full_path_percentile: float = 15.0
    candidate_score_ema_alpha: float = 0.35
    min_candidate_commit_sec: float = 0.8
    transient_invalid_grace_sec: float = 0.8
    max_plan_age_sec: float = 1.5
    switch_score_margin: float = 0.08
    switch_confirm_count: int = 2
    unsafe_switch_confirm_count: int = 3
    max_candidate_switch_deg: float = 15.0
    corridor_half_width_ratio: float = 0.025
    maximum_visual_heading_deg: float = 55.0
    debug_candidate_scores: bool = True
    # geometry_mode selects how candidate corridors are produced.
    # "image_heuristic" is the legacy/replay mode: fixed image-space curves
    # drawn from candidate_headings_deg with no relationship to real rover
    # geometry or camera calibration. "metric_projected" samples real
    # rover-frame constant-curvature footprints (below) and projects them
    # through a validated CameraCalibration; live profiles must use this
    # mode, and it never silently falls back to image_heuristic.
    geometry_mode: str = "image_heuristic"
    curvatures: tuple[float, ...] = DEFAULT_CURVATURES
    horizon_m: float = 2.0
    sample_interval_m: float = 0.1
    rover_width_m: float = 0.4
    safety_margin_m: float = 0.1
    # A candidate whose projected corridor covers less than this fraction of
    # horizon_m (measured over the longest contiguous visible run -- see
    # camera_projection.project_trajectory) is rejected outright rather than
    # scored on a near-useless sliver.
    min_projected_coverage_ratio: float = 0.5
    # The closest fraction of a candidate's *visible* projected run used for
    # near-field safety statistics, matching the image_heuristic mode's
    # near_count = ceil(len(points) * 0.35) convention.
    near_field_horizon_fraction: float = 0.35

    # side_sector_* configure an independent LEFT/RIGHT perception check over
    # the raw score map, deliberately not derived from the candidate curves:
    # when an obstacle fills the shared near-field start region every
    # candidate passes through, every candidate's own near-field stats are
    # contaminated identically and can't tell left from right. These sectors
    # sample fixed image regions instead, so they stay informative exactly
    # when the candidate-based near-field check can't be.
    side_sector_enabled: bool = False
    side_sector_top_ratio: float = 0.45
    side_sector_bottom_exclude_ratio: float = 0.08
    side_sector_left_width_ratio: float = 0.30
    side_sector_right_width_ratio: float = 0.30
    side_sector_low_percentile: float = 20.0
    side_sector_traversable_score_threshold: float = 0.55
    side_sector_min_traversable_ratio: float = 0.55
    side_sector_min_valid_pixel_ratio: float = 0.4
    side_sector_stop_score: float = 0.5
    side_sector_margin: float = 0.12

    @classmethod
    def from_dict(cls, config: dict[str, Any] | None) -> "MotionPrimitivePlannerConfig":
        values = dict(config or {})
        if "candidate_headings_deg" in values:
            values["candidate_headings_deg"] = tuple(float(v) for v in values["candidate_headings_deg"])
        if "curvatures" in values:
            values["curvatures"] = tuple(float(v) for v in values["curvatures"])
        return cls(**{key: values[key] for key in cls.__dataclass_fields__ if key in values})

    def validate(self) -> None:
        if self.mode not in {"connected_path", "motion_primitives", "gps_only"}:
            raise ValueError("planner.mode must be connected_path, motion_primitives, or gps_only")
        if len(self.candidate_headings_deg) < 7:
            raise ValueError("planner.candidate_headings_deg must contain at least 7 candidates")
        if any(not math.isfinite(v) for v in self.candidate_headings_deg):
            raise ValueError("candidate headings must be finite")
        if not 0.0 <= self.path_score_threshold <= 1.0:
            raise ValueError("planner.path_score_threshold must be in [0, 1]")
        if not 0.0 <= self.near_field_stop_threshold <= 1.0:
            raise ValueError("planner.near_field_stop_threshold must be in [0, 1]")
        if not 0.0 < self.candidate_score_ema_alpha <= 1.0:
            raise ValueError("planner.candidate_score_ema_alpha must be in (0, 1]")
        if self.switch_confirm_count < 1 or self.unsafe_switch_confirm_count < 1:
            raise ValueError("planner switch confirmation counts must be >= 1")
        for name in (
            "min_candidate_commit_sec",
            "transient_invalid_grace_sec",
            "max_plan_age_sec",
            "corridor_half_width_ratio",
            "maximum_visual_heading_deg",
            "max_candidate_switch_deg",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"planner.{name} must be finite and positive")
        if not math.isfinite(self.near_field_soft_risk_weight) or self.near_field_soft_risk_weight < 0.0:
            raise ValueError("planner.near_field_soft_risk_weight must be finite and non-negative")
        if self.geometry_mode not in {"image_heuristic", "metric_projected"}:
            raise ValueError(
                "planner.geometry_mode must be image_heuristic or metric_projected"
            )
        if self.geometry_mode == "metric_projected":
            if len(self.curvatures) < 3:
                raise ValueError(
                    "planner.curvatures must contain at least 3 values in metric_projected mode"
                )
            if any(not math.isfinite(v) for v in self.curvatures):
                raise ValueError("planner.curvatures must be finite")
            for name in ("horizon_m", "sample_interval_m", "rover_width_m"):
                value = float(getattr(self, name))
                if not math.isfinite(value) or value <= 0.0:
                    raise ValueError(
                        f"planner.{name} must be finite and positive in metric_projected mode"
                    )
            if not math.isfinite(self.safety_margin_m) or self.safety_margin_m < 0.0:
                raise ValueError(
                    "planner.safety_margin_m must be finite and non-negative in metric_projected mode"
                )
            if not 0.0 < self.min_projected_coverage_ratio <= 1.0:
                raise ValueError(
                    "planner.min_projected_coverage_ratio must be in (0, 1] in metric_projected mode"
                )
            if not 0.0 < self.near_field_horizon_fraction <= 1.0:
                raise ValueError(
                    "planner.near_field_horizon_fraction must be in (0, 1] in metric_projected mode"
                )
        if not isinstance(self.side_sector_enabled, bool):
            raise ValueError("planner.side_sector_enabled must be boolean")
        if self.side_sector_enabled:
            if not 0.0 <= self.side_sector_top_ratio < 1.0:
                raise ValueError("planner.side_sector_top_ratio must be in [0, 1)")
            if not 0.0 <= self.side_sector_bottom_exclude_ratio < 1.0:
                raise ValueError("planner.side_sector_bottom_exclude_ratio must be in [0, 1)")
            if self.side_sector_top_ratio + self.side_sector_bottom_exclude_ratio >= 1.0:
                raise ValueError(
                    "planner.side_sector_top_ratio + side_sector_bottom_exclude_ratio must be < 1"
                )
            if not 0.0 < self.side_sector_left_width_ratio <= 1.0:
                raise ValueError("planner.side_sector_left_width_ratio must be in (0, 1]")
            if not 0.0 < self.side_sector_right_width_ratio <= 1.0:
                raise ValueError("planner.side_sector_right_width_ratio must be in (0, 1]")
            if not 0.0 <= self.side_sector_low_percentile <= 100.0:
                raise ValueError("planner.side_sector_low_percentile must be in [0, 100]")
            for name in (
                "side_sector_traversable_score_threshold",
                "side_sector_min_traversable_ratio",
                "side_sector_min_valid_pixel_ratio",
                "side_sector_stop_score",
            ):
                value = float(getattr(self, name))
                if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                    raise ValueError(f"planner.{name} must be in [0, 1]")
            if not math.isfinite(self.side_sector_margin) or self.side_sector_margin < 0.0:
                raise ValueError("planner.side_sector_margin must be finite and non-negative")


@dataclass(frozen=True)
class CandidateScore:
    index: int
    heading_deg: float
    traversability_weighted_mean: float
    traversability_low_percentile: float
    near_field_mean: float
    near_field_low_percentile: float
    goal_heading_error_deg: float
    goal_penalty: float
    continuity_delta_deg: float
    continuity_penalty: float
    curvature_magnitude: float
    curvature_penalty: float
    near_field_risk_penalty: float
    near_field_soft_penalty: float
    final_score_raw: float
    final_score: float
    hard_rejected: bool
    reject_reason: str | None
    points_uv: np.ndarray = field(repr=False)
    left_boundary_uv: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2), dtype=np.int32), repr=False
    )
    right_boundary_uv: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2), dtype=np.int32), repr=False
    )
    footprint_pixel_count: int = 0
    projected_coverage_ratio: float = 1.0

    def to_status(self) -> dict[str, Any]:
        endpoint_x_offset = selected_candidate_endpoint_x_offset_px(self.points_uv)
        return {
            "index": self.index,
            "heading_deg": self.heading_deg,
            "heading_convention": "positive_clockwise_right",
            "image_direction": image_direction_from_x_offset(endpoint_x_offset),
            "endpoint_x_offset_px": endpoint_x_offset,
            "traversability": self.traversability_weighted_mean,
            "traversability_low_percentile": self.traversability_low_percentile,
            "near_field": self.near_field_low_percentile,
            "near_field_mean": self.near_field_mean,
            "goal_heading_error_deg": self.goal_heading_error_deg,
            "goal_penalty": self.goal_penalty,
            "continuity_penalty": self.continuity_penalty,
            "curvature_penalty": self.curvature_penalty,
            "near_field_risk_penalty": self.near_field_risk_penalty,
            "near_field_soft_penalty": self.near_field_soft_penalty,
            "final_score": self.final_score,
            "hard_rejected": self.hard_rejected,
            "reject_reason": self.reject_reason,
            "footprint_pixel_count": self.footprint_pixel_count,
            "projected_coverage_ratio": self.projected_coverage_ratio,
            "centerline_uv": self.points_uv.astype(int, copy=False).tolist(),
            "left_boundary_uv": self.left_boundary_uv.astype(int, copy=False).tolist(),
            "right_boundary_uv": self.right_boundary_uv.astype(int, copy=False).tolist(),
        }


@dataclass(frozen=True)
class SideSectorScore:
    side: str
    mean: float
    low_percentile: float
    traversable_ratio: float
    valid_pixel_ratio: float
    pixel_count: int
    composite: float
    viable: bool

    def to_status(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "mean": self.mean,
            "low_percentile": self.low_percentile,
            "traversable_ratio": self.traversable_ratio,
            "valid_pixel_ratio": self.valid_pixel_ratio,
            "pixel_count": self.pixel_count,
            "composite": self.composite,
            "viable": self.viable,
        }


@dataclass(frozen=True)
class SideSectorDecision:
    left: SideSectorScore
    right: SideSectorScore
    chosen: str | None
    status: str
    margin: float
    reason: str

    def to_status(self) -> dict[str, Any]:
        return {
            "left": self.left.to_status(),
            "right": self.right.to_status(),
            "chosen": self.chosen,
            "status": self.status,
            "margin": self.margin,
            "reason": self.reason,
        }


def _score_side_sector(
    side: str,
    region_score: np.ndarray,
    region_valid: np.ndarray,
    config: MotionPrimitivePlannerConfig,
) -> SideSectorScore:
    pixel_count = int(region_score.size)
    valid_pixel_ratio = float(np.mean(region_valid)) if pixel_count else 0.0
    if pixel_count == 0 or valid_pixel_ratio < config.side_sector_min_valid_pixel_ratio:
        return SideSectorScore(
            side=side,
            mean=0.0,
            low_percentile=0.0,
            traversable_ratio=0.0,
            valid_pixel_ratio=valid_pixel_ratio,
            pixel_count=pixel_count,
            composite=0.0,
            viable=False,
        )
    sampled = region_score[region_valid]
    mean = float(np.mean(sampled))
    low_percentile = float(np.percentile(sampled, config.side_sector_low_percentile))
    traversable_ratio = float(np.mean(sampled >= config.side_sector_traversable_score_threshold))
    composite = 0.5 * low_percentile + 0.5 * traversable_ratio
    viable = (
        low_percentile >= config.side_sector_stop_score
        and traversable_ratio >= config.side_sector_min_traversable_ratio
    )
    return SideSectorScore(
        side=side,
        mean=mean,
        low_percentile=low_percentile,
        traversable_ratio=traversable_ratio,
        valid_pixel_ratio=valid_pixel_ratio,
        pixel_count=pixel_count,
        composite=composite,
        viable=viable,
    )


def evaluate_side_sectors(
    score: np.ndarray,
    valid: np.ndarray,
    config: MotionPrimitivePlannerConfig,
) -> SideSectorDecision:
    """Score independent LEFT/RIGHT image regions for a recovery rotation.

    Unlike the forward candidate curves (which all share the same
    bottom-center start region and so score identically contaminated when a
    wide near obstacle blocks that shared region), these sectors sample fixed
    left/right image bands directly, excluding sky/horizon rows and the
    rover body/bumper rows at the very bottom.
    """

    height, width = int(score.shape[0]), int(score.shape[1])
    row_start = int(round(height * config.side_sector_top_ratio))
    row_end = int(round(height * (1.0 - config.side_sector_bottom_exclude_ratio)))
    row_end = max(row_start + 1, min(height, row_end))
    row_start = min(row_start, row_end - 1)
    left_col_end = max(1, min(width, int(round(width * config.side_sector_left_width_ratio))))
    right_col_start = max(0, min(width - 1, width - int(round(width * config.side_sector_right_width_ratio))))

    left = _score_side_sector(
        "LEFT",
        score[row_start:row_end, 0:left_col_end],
        valid[row_start:row_end, 0:left_col_end],
        config,
    )
    right = _score_side_sector(
        "RIGHT",
        score[row_start:row_end, right_col_start:width],
        valid[row_start:row_end, right_col_start:width],
        config,
    )

    margin = float(config.side_sector_margin)
    diff = right.composite - left.composite
    if not left.viable and not right.viable:
        chosen, status, reason = None, "BOTH_UNSAFE", "neither side sector clears the safety thresholds"
    elif abs(diff) < margin:
        chosen, status, reason = (
            None,
            "AMBIGUOUS",
            f"side sector composite margin {diff:.3f} is below the required {margin:.3f}",
        )
    elif diff > 0.0:
        if right.viable:
            chosen, status, reason = "RIGHT", "RIGHT_CLEAR", "right side sector is clearly safer than left"
        else:
            chosen, status, reason = None, "AMBIGUOUS", "right side scores higher but is not independently viable"
    else:
        if left.viable:
            chosen, status, reason = "LEFT", "LEFT_CLEAR", "left side sector is clearly safer than right"
        else:
            chosen, status, reason = None, "AMBIGUOUS", "left side scores higher but is not independently viable"

    return SideSectorDecision(
        left=left,
        right=right,
        chosen=chosen,
        status=status,
        margin=diff,
        reason=reason,
    )


@dataclass(frozen=True)
class MotionPrimitivePlan:
    mode: str
    image_path: ImageSpacePathProposal
    selected_candidate: CandidateScore | None
    candidate_scores: tuple[CandidateScore, ...]
    near_field_safe: bool
    near_field_score: float
    trajectory_valid: bool
    trajectory_quality: float
    planner_confidence: float
    using_held_plan: bool
    plan_age_sec: float | None
    candidate_switched: bool
    switch_reason: str | None
    switch_stop_required: bool
    selected_candidate_since: float | None
    last_valid_candidate_time: float | None
    switch_pending_index: int | None
    switch_confirm_count: int
    geometry_mode: str = "image_heuristic"
    camera_projection_applied: bool = False
    image_path_metric_calibrated: bool = False
    calibration_id: str | None = None
    calibration_sha256_prefix: str | None = None
    side_sector: SideSectorDecision | None = None

    @property
    def path_valid(self) -> bool:
        return self.near_field_safe and (self.trajectory_valid or self.using_held_plan)

    def to_status(self, *, include_candidates: bool = True) -> dict[str, Any]:
        selected_age = (
            None
            if self.selected_candidate_since is None
            else max(0.0, time.monotonic() - self.selected_candidate_since)
        )
        status = {
            "mode": self.mode,
            "heading_convention": "positive_clockwise_right",
            "selected_candidate_index": (
                self.selected_candidate.index if self.selected_candidate is not None else None
            ),
            "selected_candidate_heading_deg": (
                self.selected_candidate.heading_deg if self.selected_candidate is not None else None
            ),
            "selected_candidate_score": (
                self.selected_candidate.final_score if self.selected_candidate is not None else None
            ),
            "selected_candidate_image_direction": (
                image_direction_from_x_offset(
                    selected_candidate_endpoint_x_offset_px(self.selected_candidate.points_uv)
                )
                if self.selected_candidate is not None
                else None
            ),
            "selected_endpoint_x_offset_px": (
                selected_candidate_endpoint_x_offset_px(self.selected_candidate.points_uv)
                if self.selected_candidate is not None
                else None
            ),
            "selected_candidate_age_sec": selected_age,
            "candidate_switched": self.candidate_switched,
            "switch_reason": self.switch_reason,
            "switch_stop_required": self.switch_stop_required,
            "plan_age_sec": self.plan_age_sec,
            "planner_confidence": self.planner_confidence,
            "near_field_safe": self.near_field_safe,
            "near_field_score": self.near_field_score,
            "trajectory_valid": self.trajectory_valid,
            "trajectory_quality": self.trajectory_quality,
            "using_held_plan": self.using_held_plan,
            "candidate_switch_pending": self.switch_pending_index,
            "candidate_switch_confirm_count": self.switch_confirm_count,
            "geometry_mode": self.geometry_mode,
            "camera_projection_applied": self.camera_projection_applied,
            "image_path_metric_calibrated": self.image_path_metric_calibrated,
            "calibration_id": self.calibration_id,
            "calibration_sha256_prefix": self.calibration_sha256_prefix,
            "side_sector": self.side_sector.to_status() if self.side_sector is not None else None,
        }
        if include_candidates:
            status["candidate_scores"] = [candidate.to_status() for candidate in self.candidate_scores]
        return status


class MotionPrimitivePlanner:
    """GPS-primary local planner that uses SAM-TP as a candidate cost evaluator.

    The planner evaluates a fixed set of image-space motion primitives instead
    of searching for a connected high-score pixel topology each frame.  The
    chosen primitive is temporally committed and only switches when a better
    candidate is confirmed, reducing left/right oscillation.
    """

    def __init__(
        self,
        config: MotionPrimitivePlannerConfig | dict[str, Any] | None = None,
        *,
        monotonic: Any = time.monotonic,
    ) -> None:
        self.config = (
            config
            if isinstance(config, MotionPrimitivePlannerConfig)
            else MotionPrimitivePlannerConfig.from_dict(config)
        )
        self.config.validate()
        self.monotonic = monotonic
        self._selected_index: int | None = None
        self._selected_since: float | None = None
        self._last_valid_candidate_time: float | None = None
        self._last_plan: MotionPrimitivePlan | None = None
        self._score_ema: dict[int, float] = {}
        self._switch_pending_index: int | None = None
        self._switch_confirm_count = 0
        self._unsafe_switch_pending = False
        self._previous_target_heading_deg: float | None = None
        self._previous_checkpoint_sequence: int | None = None
        self._metric_trajectories: tuple[CandidateTrajectory, ...] = ()
        self._metric_heading_by_index: tuple[float, ...] = ()
        if self.config.geometry_mode == "metric_projected":
            self._metric_trajectories = ConstantCurvatureTrajectorySampler(
                self.config.curvatures,
                horizon_m=self.config.horizon_m,
                sample_interval_m=self.config.sample_interval_m,
                rover_width_m=self.config.rover_width_m,
                safety_margin_m=self.config.safety_margin_m,
            ).sample()
            self._metric_heading_by_index = tuple(
                metric_terminal_heading_deg(trajectory.curvature, trajectory.horizon_m)
                for trajectory in self._metric_trajectories
            )

    def reset(self) -> None:
        self._selected_index = None
        self._selected_since = None
        self._last_valid_candidate_time = None
        self._last_plan = None
        self._score_ema.clear()
        self._switch_pending_index = None
        self._switch_confirm_count = 0
        self._unsafe_switch_pending = False
        self._previous_target_heading_deg = None
        self._previous_checkpoint_sequence = None

    def plan(
        self,
        score_map: np.ndarray,
        valid_mask: np.ndarray,
        *,
        target_heading_error_rad: float | None,
        checkpoint_sequence: int | None = None,
        timestamp: float | None = None,
        navigation: dict[str, Any] | None = None,
        calibration: CameraCalibration | None = None,
    ) -> MotionPrimitivePlan:
        score = np.asarray(score_map, dtype=np.float32)
        valid = np.asarray(valid_mask, dtype=bool)
        if score.ndim != 2 or valid.shape != score.shape:
            raise ValueError("score_map and valid_mask must be matching 2D arrays")
        if score.size == 0 or not np.isfinite(score).all():
            raise ValueError("score_map must be finite and non-empty")
        side_sector = (
            evaluate_side_sectors(score, valid, self.config)
            if self.config.side_sector_enabled
            else None
        )
        now = self.monotonic()
        checkpoint_changed = (
            checkpoint_sequence is not None
            and self._previous_checkpoint_sequence is not None
            and checkpoint_sequence != self._previous_checkpoint_sequence
        )
        if checkpoint_changed:
            self.reset()
        if checkpoint_sequence is not None:
            self._previous_checkpoint_sequence = checkpoint_sequence
        target_heading_deg = (
            0.0
            if target_heading_error_rad is None or not math.isfinite(target_heading_error_rad)
            else math.degrees(normalize_angle_rad(target_heading_error_rad))
        )
        large_target_change = False
        if self._previous_target_heading_deg is not None:
            large_target_change = (
                abs(normalize_angle_deg(target_heading_deg - self._previous_target_heading_deg))
                >= 35.0
            )
        self._previous_target_heading_deg = target_heading_deg

        if self.config.mode == "gps_only":
            return self._gps_only_plan(
                score,
                valid,
                target_heading_deg=target_heading_deg,
                target_heading_error_rad=target_heading_error_rad,
                now=now,
                side_sector=side_sector,
            )

        metric_status = {
            "geometry_mode": self.config.geometry_mode,
            "camera_projection_applied": False,
            "image_path_metric_calibrated": False,
            "calibration_id": None,
            "calibration_sha256_prefix": None,
        }
        if self.config.geometry_mode == "metric_projected":
            candidates, metric_status = self._score_candidates_metric(
                score,
                valid,
                target_heading_deg,
                calibration,
                ignore_continuity=large_target_change,
            )
        else:
            candidates = self._score_candidates(
                score,
                valid,
                target_heading_deg,
                ignore_continuity=large_target_change,
            )
        for candidate in candidates:
            previous = self._score_ema.get(candidate.index, candidate.final_score_raw)
            self._score_ema[candidate.index] = (
                self.config.candidate_score_ema_alpha * candidate.final_score_raw
                + (1.0 - self.config.candidate_score_ema_alpha) * previous
            )
        candidates = tuple(
            CandidateScore(
                **{
                    **candidate.__dict__,
                    "final_score": float(self._score_ema.get(candidate.index, candidate.final_score_raw)),
                }
            )
            for candidate in candidates
        )
        safe_candidates = [candidate for candidate in candidates if not candidate.hard_rejected]
        near_field_score = max((candidate.near_field_low_percentile for candidate in candidates), default=0.0)
        all_near_unsafe = bool(candidates) and all(candidate.hard_rejected for candidate in candidates)
        near_field_safe = not all_near_unsafe and near_field_score >= self.config.near_field_stop_threshold

        selected, switched, switch_reason, switch_stop_required = self._select_candidate(
            safe_candidates,
            now=now,
            checkpoint_changed=checkpoint_changed,
        )
        using_held = False
        plan_age = None
        if selected is None:
            held = self._last_plan
            if (
                not switch_stop_required
                and held is not None
                and held.selected_candidate is not None
                and near_field_safe
                and self._last_valid_candidate_time is not None
                and now - self._last_valid_candidate_time <= self.config.transient_invalid_grace_sec
            ):
                selected = held.selected_candidate
                using_held = True
                plan_age = now - self._last_valid_candidate_time
                switch_reason = "transient_invalid_hold"
        if selected is not None and not using_held:
            self._last_valid_candidate_time = now
            plan_age = 0.0
        elif plan_age is None and self._last_valid_candidate_time is not None:
            plan_age = now - self._last_valid_candidate_time

        last_valid_age = (
            None
            if self._last_valid_candidate_time is None
            else now - self._last_valid_candidate_time
        )
        trajectory_quality = (
            max(0.0, min(1.0, selected.traversability_weighted_mean))
            if selected is not None
            else 0.0
        )
        trajectory_valid = (
            selected is not None
            and not selected.hard_rejected
            and selected.traversability_weighted_mean >= self.config.path_score_threshold
        )
        if (
            selected is not None
            and not trajectory_valid
            and self._last_plan is not None
            and self._last_plan.selected_candidate is not None
            and near_field_safe
            and last_valid_age is not None
            and last_valid_age <= self.config.transient_invalid_grace_sec
        ):
            selected = self._last_plan.selected_candidate
            using_held = True
            plan_age = last_valid_age
            switch_reason = "transient_low_quality_hold"
            trajectory_quality = self._last_plan.trajectory_quality
            trajectory_valid = True
        planner_confidence = self._confidence(
            selected,
            near_field_safe=near_field_safe,
            trajectory_valid=trajectory_valid,
            plan_age_sec=plan_age,
            using_held=using_held,
        )
        image_path = self._image_path_for_candidate(
            selected,
            valid=near_field_safe and selected is not None,
            reason=self._reason(
                selected,
                near_field_safe=near_field_safe,
                trajectory_valid=trajectory_valid,
                using_held=using_held,
            ),
            target_heading_error_rad=target_heading_error_rad,
            mean_score=trajectory_quality,
        )
        plan = MotionPrimitivePlan(
            mode=self.config.mode,
            image_path=image_path,
            selected_candidate=selected,
            candidate_scores=candidates,
            near_field_safe=near_field_safe,
            near_field_score=float(near_field_score),
            trajectory_valid=trajectory_valid,
            trajectory_quality=trajectory_quality,
            planner_confidence=planner_confidence,
            using_held_plan=using_held,
            plan_age_sec=plan_age,
            candidate_switched=switched,
            switch_reason=switch_reason,
            switch_stop_required=switch_stop_required,
            selected_candidate_since=self._selected_since,
            last_valid_candidate_time=self._last_valid_candidate_time,
            switch_pending_index=self._switch_pending_index,
            switch_confirm_count=self._switch_confirm_count,
            side_sector=side_sector,
            **metric_status,
        )
        self._last_plan = plan
        return plan

    def _score_candidates(
        self,
        score: np.ndarray,
        valid: np.ndarray,
        target_heading_deg: float,
        *,
        ignore_continuity: bool = False,
    ) -> tuple[CandidateScore, ...]:
        candidates: list[CandidateScore] = []
        maximum_heading = max(1.0, float(self.config.maximum_visual_heading_deg))
        for index, heading_deg in enumerate(self.config.candidate_headings_deg):
            points = primitive_curve_points(
                score.shape,
                heading_deg,
                maximum_visual_heading_deg=self.config.maximum_visual_heading_deg,
            )
            mask = corridor_mask(score.shape, points, self.config.corridor_half_width_ratio)
            sampled = score[mask & valid]
            center_sampled = score[points[:, 1], points[:, 0]]
            if sampled.size == 0 or center_sampled.size == 0:
                candidates.append(
                    CandidateScore(
                        index=index,
                        heading_deg=float(heading_deg),
                        traversability_weighted_mean=0.0,
                        traversability_low_percentile=0.0,
                        near_field_mean=0.0,
                        near_field_low_percentile=0.0,
                        goal_heading_error_deg=abs(normalize_angle_deg(heading_deg - target_heading_deg)),
                        goal_penalty=1.0,
                        continuity_delta_deg=0.0,
                        continuity_penalty=0.0,
                        curvature_magnitude=abs(float(heading_deg)) / maximum_heading,
                        curvature_penalty=abs(float(heading_deg)) / maximum_heading,
                        near_field_risk_penalty=1.0,
                        near_field_soft_penalty=1.0,
                        final_score_raw=-10.0,
                        final_score=-10.0,
                        hard_rejected=True,
                        reject_reason="NO_VALID_CORRIDOR_SAMPLES",
                        points_uv=points,
                    )
                )
                continue
            weights = primitive_sample_weights(points[:, 1], score.shape[0])
            weighted_mean = float(np.average(center_sampled, weights=weights))
            low_percentile = float(np.percentile(sampled, self.config.full_path_percentile))
            near_count = max(2, int(math.ceil(len(points) * 0.35)))
            near_scores = score[points[:near_count, 1], points[:near_count, 0]]
            near_mean = float(np.mean(near_scores))
            near_low = float(np.percentile(near_scores, self.config.near_field_percentile))
            goal_error = abs(normalize_angle_deg(float(heading_deg) - target_heading_deg))
            goal_penalty = min(1.0, goal_error / maximum_heading)
            if self._selected_index is None or ignore_continuity:
                continuity_delta = 0.0
            else:
                continuity_delta = abs(
                    normalize_angle_deg(
                        float(heading_deg)
                        - float(self.config.candidate_headings_deg[self._selected_index])
                    )
                )
            continuity_penalty = min(1.0, continuity_delta / maximum_heading)
            curvature_mag = min(1.0, abs(float(heading_deg)) / maximum_heading)
            near_risk = max(0.0, self.config.near_field_stop_threshold - near_low)
            # Unlike near_risk, this applies at every near_low value, not
            # just below near_field_stop_threshold -- see
            # near_field_soft_risk_weight's docstring for why.
            near_soft_penalty = max(0.0, min(1.0, 1.0 - near_low))
            hard_rejected = near_low < self.config.near_field_stop_threshold
            final_raw = (
                self.config.traversability_weight * (0.70 * weighted_mean + 0.30 * low_percentile)
                - self.config.goal_heading_weight * goal_penalty
                - self.config.continuity_weight * continuity_penalty
                - self.config.curvature_weight * curvature_mag
                - self.config.near_field_risk_weight * near_risk
                - self.config.near_field_soft_risk_weight * near_soft_penalty
            )
            candidates.append(
                CandidateScore(
                    index=index,
                    heading_deg=float(heading_deg),
                    traversability_weighted_mean=weighted_mean,
                    traversability_low_percentile=low_percentile,
                    near_field_mean=near_mean,
                    near_field_low_percentile=near_low,
                    goal_heading_error_deg=goal_error,
                    goal_penalty=goal_penalty,
                    continuity_delta_deg=continuity_delta,
                    continuity_penalty=continuity_penalty,
                    curvature_magnitude=curvature_mag,
                    curvature_penalty=curvature_mag,
                    near_field_risk_penalty=near_risk,
                    near_field_soft_penalty=near_soft_penalty,
                    final_score_raw=final_raw,
                    final_score=final_raw,
                    hard_rejected=hard_rejected,
                    reject_reason="NEAR_FIELD_UNSAFE" if hard_rejected else None,
                    points_uv=points,
                )
            )
        return tuple(candidates)

    def _check_calibration(
        self,
        calibration: CameraCalibration | None,
        frame_shape: tuple[int, int],
    ) -> str | None:
        if calibration is None:
            return "NO_VALID_CALIBRATION"
        try:
            validate_for_live_use(calibration, frame_shape)
        except CalibrationError as exc:
            return exc.reason
        return None

    def _score_candidates_metric(
        self,
        score: np.ndarray,
        valid: np.ndarray,
        target_heading_deg: float,
        calibration: CameraCalibration | None,
        *,
        ignore_continuity: bool = False,
    ) -> tuple[tuple[CandidateScore, ...], dict[str, Any]]:
        metric_status = {
            "geometry_mode": self.config.geometry_mode,
            "camera_projection_applied": False,
            "image_path_metric_calibrated": False,
            "calibration_id": None,
            "calibration_sha256_prefix": None,
        }
        calibration_error = self._check_calibration(calibration, score.shape)
        if calibration_error is not None:
            candidates = tuple(
                self._rejected_metric_candidate(
                    index, trajectory, target_heading_deg, calibration_error
                )
                for index, trajectory in enumerate(self._metric_trajectories)
            )
            return candidates, metric_status
        assert calibration is not None
        metric_status["camera_projection_applied"] = True
        metric_status["image_path_metric_calibrated"] = True
        metric_status["calibration_id"] = calibration.calibration_id
        metric_status["calibration_sha256_prefix"] = calibration.sha256_prefix

        candidates: list[CandidateScore] = []
        maximum_heading = max(1.0, float(self.config.maximum_visual_heading_deg))
        for index, trajectory in enumerate(self._metric_trajectories):
            heading_deg = self._metric_heading_by_index[index]
            footprint = project_trajectory(
                trajectory,
                calibration,
                min_projected_coverage_ratio=self.config.min_projected_coverage_ratio,
                near_field_horizon_fraction=self.config.near_field_horizon_fraction,
            )
            if not footprint.valid:
                candidates.append(
                    self._rejected_metric_candidate(
                        index, trajectory, target_heading_deg, footprint.reason
                    )
                )
                continue
            footprint_scores = score[footprint.footprint_mask & valid]
            near_scores = score[footprint.near_field_mask & valid]
            if footprint_scores.size == 0 or near_scores.size == 0:
                candidates.append(
                    self._rejected_metric_candidate(
                        index,
                        trajectory,
                        target_heading_deg,
                        "NO_VALID_CORRIDOR_SAMPLES",
                    )
                )
                continue
            weighted_mean = float(np.mean(footprint_scores))
            low_percentile = float(np.percentile(footprint_scores, self.config.full_path_percentile))
            near_mean = float(np.mean(near_scores))
            near_low = float(np.percentile(near_scores, self.config.near_field_percentile))
            goal_error = abs(normalize_angle_deg(heading_deg - target_heading_deg))
            goal_penalty = min(1.0, goal_error / maximum_heading)
            if self._selected_index is None or ignore_continuity or self._selected_index >= len(
                self._metric_heading_by_index
            ):
                continuity_delta = 0.0
            else:
                continuity_delta = abs(
                    normalize_angle_deg(
                        heading_deg - self._metric_heading_by_index[self._selected_index]
                    )
                )
            continuity_penalty = min(1.0, continuity_delta / maximum_heading)
            curvature_mag = min(1.0, abs(heading_deg) / maximum_heading)
            near_risk = max(0.0, self.config.near_field_stop_threshold - near_low)
            near_soft_penalty = max(0.0, min(1.0, 1.0 - near_low))
            near_field_unsafe = near_low < self.config.near_field_stop_threshold
            footprint_unsafe = low_percentile < self.config.path_score_threshold
            hard_rejected = near_field_unsafe or footprint_unsafe
            final_raw = (
                self.config.traversability_weight * (0.70 * weighted_mean + 0.30 * low_percentile)
                - self.config.goal_heading_weight * goal_penalty
                - self.config.continuity_weight * continuity_penalty
                - self.config.curvature_weight * curvature_mag
                - self.config.near_field_risk_weight * near_risk
                - self.config.near_field_soft_risk_weight * near_soft_penalty
            )
            candidates.append(
                CandidateScore(
                    index=index,
                    heading_deg=heading_deg,
                    traversability_weighted_mean=weighted_mean,
                    traversability_low_percentile=low_percentile,
                    near_field_mean=near_mean,
                    near_field_low_percentile=near_low,
                    goal_heading_error_deg=goal_error,
                    goal_penalty=goal_penalty,
                    continuity_delta_deg=continuity_delta,
                    continuity_penalty=continuity_penalty,
                    curvature_magnitude=curvature_mag,
                    curvature_penalty=curvature_mag,
                    near_field_risk_penalty=near_risk,
                    near_field_soft_penalty=near_soft_penalty,
                    final_score_raw=final_raw,
                    final_score=final_raw,
                    hard_rejected=hard_rejected,
                    reject_reason=(
                        "NEAR_FIELD_UNSAFE"
                        if near_field_unsafe
                        else "FOOTPRINT_UNSAFE" if footprint_unsafe else None
                    ),
                    points_uv=footprint.centerline_uv,
                    left_boundary_uv=footprint.left_uv,
                    right_boundary_uv=footprint.right_uv,
                    footprint_pixel_count=footprint.footprint_pixel_count,
                    projected_coverage_ratio=footprint.coverage_ratio,
                )
            )
        return tuple(candidates), metric_status

    def _rejected_metric_candidate(
        self,
        index: int,
        trajectory: CandidateTrajectory,
        target_heading_deg: float,
        reason: str,
    ) -> CandidateScore:
        heading_deg = metric_terminal_heading_deg(trajectory.curvature, trajectory.horizon_m)
        maximum_heading = max(1.0, float(self.config.maximum_visual_heading_deg))
        goal_error = abs(normalize_angle_deg(heading_deg - target_heading_deg))
        curvature_mag = min(1.0, abs(heading_deg) / maximum_heading)
        return CandidateScore(
            index=index,
            heading_deg=heading_deg,
            traversability_weighted_mean=0.0,
            traversability_low_percentile=0.0,
            near_field_mean=0.0,
            near_field_low_percentile=0.0,
            goal_heading_error_deg=goal_error,
            goal_penalty=1.0,
            continuity_delta_deg=0.0,
            continuity_penalty=0.0,
            curvature_magnitude=curvature_mag,
            curvature_penalty=curvature_mag,
            near_field_risk_penalty=1.0,
            near_field_soft_penalty=1.0,
            final_score_raw=-10.0,
            final_score=-10.0,
            hard_rejected=True,
            reject_reason=reason,
            points_uv=np.zeros((0, 2), dtype=np.int32),
            footprint_pixel_count=0,
            projected_coverage_ratio=0.0,
        )

    def _select_candidate(
        self,
        safe_candidates: list[CandidateScore],
        *,
        now: float,
        checkpoint_changed: bool,
    ) -> tuple[CandidateScore | None, bool, str | None, bool]:
        if not safe_candidates:
            self._unsafe_switch_pending = True
            return None, False, "all_candidates_hard_rejected", True
        best = max(safe_candidates, key=lambda item: item.final_score)
        current = next(
            (candidate for candidate in safe_candidates if candidate.index == self._selected_index),
            None,
        )
        if self._selected_index is None:
            self._commit(best.index, now)
            return best, False, None, False
        if current is None:
            self._unsafe_switch_pending = True
            previous_heading = self._candidate_heading_for_index(self._selected_index)
            bounded = self._bounded_switch_candidate(
                safe_candidates,
                previous_heading,
                best.heading_deg,
            )
            if bounded is None:
                self._switch_pending_index = None
                self._switch_confirm_count = 0
                return None, False, "stop_no_adjacent_safe_candidate", True
            if self._switch_pending_index == bounded.index:
                self._switch_confirm_count += 1
            else:
                self._switch_pending_index = bounded.index
                self._switch_confirm_count = 1
            if self._switch_confirm_count < self.config.unsafe_switch_confirm_count:
                return None, False, "stop_unsafe_candidate_switch_pending", True
            self._commit(bounded.index, now)
            return bounded, True, "stop_confirmed_bounded_switch", False
        if self._unsafe_switch_pending:
            self._unsafe_switch_pending = False
            self._switch_pending_index = None
            self._switch_confirm_count = 0
            return current, False, "unsafe_candidate_recovered", False
        immediate = checkpoint_changed
        if current is not None and self._last_valid_candidate_time is not None:
            if now - self._last_valid_candidate_time > self.config.max_plan_age_sec:
                immediate = True
        if immediate:
            changed = self._selected_index != best.index
            self._commit(best.index, now)
            return best, changed, "checkpoint_reset" if changed else None, False
        assert current is not None
        if self._selected_since is not None and now - self._selected_since < self.config.min_candidate_commit_sec:
            self._switch_pending_index = None
            self._switch_confirm_count = 0
            return current, False, None, False
        if best.index == current.index:
            self._switch_pending_index = None
            self._switch_confirm_count = 0
            return current, False, None, False
        if best.final_score < current.final_score + self.config.switch_score_margin:
            self._switch_pending_index = None
            self._switch_confirm_count = 0
            return current, False, None, False
        bounded = self._bounded_switch_candidate(
            safe_candidates,
            current.heading_deg,
            best.heading_deg,
        ) or current
        if (
            bounded.index == current.index
            or bounded.final_score < current.final_score + self.config.switch_score_margin
        ):
            self._switch_pending_index = None
            self._switch_confirm_count = 0
            return current, False, None, False
        best = bounded
        if self._switch_pending_index == best.index:
            self._switch_confirm_count += 1
        else:
            self._switch_pending_index = best.index
            self._switch_confirm_count = 1
        if self._switch_confirm_count >= self.config.switch_confirm_count:
            self._commit(best.index, now)
            return best, True, "score_margin_confirmed", False
        return current, False, "switch_pending", False

    def _bounded_switch_candidate(
        self,
        safe_candidates: list[CandidateScore],
        current_heading_deg: float,
        desired_heading_deg: float,
    ) -> CandidateScore | None:
        delta = normalize_angle_deg(desired_heading_deg - current_heading_deg)
        if abs(delta) <= self.config.max_candidate_switch_deg + 1e-6:
            return next(
                (
                    candidate
                    for candidate in safe_candidates
                    if candidate.heading_deg == desired_heading_deg
                ),
                None,
            )
        direction = 1.0 if delta > 0.0 else -1.0
        bounded = [
            candidate
            for candidate in safe_candidates
            if 0.0
            < direction
            * normalize_angle_deg(candidate.heading_deg - current_heading_deg)
            <= self.config.max_candidate_switch_deg + 1e-6
        ]
        if not bounded:
            return None
        return max(
            bounded,
            key=lambda candidate: direction
            * normalize_angle_deg(candidate.heading_deg - current_heading_deg),
        )

    def _candidate_heading_for_index(self, index: int) -> float:
        if self.config.geometry_mode == "metric_projected":
            return float(self._metric_heading_by_index[index])
        return float(self.config.candidate_headings_deg[index])

    def _commit(self, index: int, now: float) -> None:
        if self._selected_index != index:
            self._selected_since = now
        elif self._selected_since is None:
            self._selected_since = now
        self._selected_index = index
        self._switch_pending_index = None
        self._switch_confirm_count = 0
        self._unsafe_switch_pending = False

    def _gps_only_plan(
        self,
        score: np.ndarray,
        valid: np.ndarray,
        *,
        target_heading_deg: float,
        target_heading_error_rad: float | None,
        now: float,
        side_sector: SideSectorDecision | None = None,
    ) -> MotionPrimitivePlan:
        clipped_heading = float(
            np.clip(
                target_heading_deg,
                -self.config.maximum_visual_heading_deg,
                self.config.maximum_visual_heading_deg,
            )
        )
        points = primitive_curve_points(score.shape, clipped_heading)
        near_count = max(2, int(math.ceil(len(points) * 0.35)))
        near_scores = score[points[:near_count, 1], points[:near_count, 0]]
        near_low = float(np.percentile(near_scores, self.config.near_field_percentile))
        near_safe = near_low >= self.config.near_field_stop_threshold
        candidate = CandidateScore(
            index=0,
            heading_deg=clipped_heading,
            traversability_weighted_mean=float(np.mean(score[points[:, 1], points[:, 0]])),
            traversability_low_percentile=float(np.percentile(score[points[:, 1], points[:, 0]], self.config.full_path_percentile)),
            near_field_mean=float(np.mean(near_scores)),
            near_field_low_percentile=near_low,
            goal_heading_error_deg=0.0,
            goal_penalty=0.0,
            continuity_delta_deg=0.0,
            continuity_penalty=0.0,
            curvature_magnitude=abs(clipped_heading) / max(1.0, self.config.maximum_visual_heading_deg),
            curvature_penalty=abs(clipped_heading) / max(1.0, self.config.maximum_visual_heading_deg),
            near_field_risk_penalty=max(0.0, self.config.near_field_stop_threshold - near_low),
            near_field_soft_penalty=max(0.0, min(1.0, 1.0 - near_low)),
            final_score_raw=1.0,
            final_score=1.0,
            hard_rejected=not near_safe,
            reject_reason=None if near_safe else "GPS_ONLY_NEAR_FIELD_UNSAFE",
            points_uv=points,
        )
        self._selected_index = 0
        if self._selected_since is None:
            self._selected_since = now
        if near_safe:
            self._last_valid_candidate_time = now
        image_path = self._image_path_for_candidate(
            candidate if near_safe else None,
            valid=near_safe,
            reason="GPS_ONLY_SELECTED" if near_safe else "GPS_ONLY_NEAR_FIELD_UNSAFE",
            target_heading_error_rad=target_heading_error_rad,
            mean_score=candidate.traversability_weighted_mean,
        )
        plan = MotionPrimitivePlan(
            mode="gps_only",
            image_path=image_path,
            selected_candidate=candidate if near_safe else None,
            candidate_scores=(candidate,),
            near_field_safe=near_safe,
            near_field_score=near_low,
            trajectory_valid=near_safe,
            trajectory_quality=candidate.traversability_weighted_mean if near_safe else 0.0,
            planner_confidence=0.65 if near_safe else 0.0,
            using_held_plan=False,
            plan_age_sec=0.0 if near_safe else None,
            candidate_switched=False,
            switch_reason=None,
            switch_stop_required=not near_safe,
            selected_candidate_since=self._selected_since,
            last_valid_candidate_time=self._last_valid_candidate_time,
            switch_pending_index=None,
            switch_confirm_count=0,
            side_sector=side_sector,
        )
        self._last_plan = plan
        return plan

    def _image_path_for_candidate(
        self,
        candidate: CandidateScore | None,
        *,
        valid: bool,
        reason: str,
        target_heading_error_rad: float | None,
        mean_score: float,
    ) -> ImageSpacePathProposal:
        if candidate is None:
            points = np.zeros((0, 2), dtype=np.int32)
            return ImageSpacePathProposal(
                valid=False,
                points_uv=points,
                mean_score=0.0,
                minimum_score=0.0,
                reason=reason,
                target_heading_error_rad=target_heading_error_rad,
            )
        points = candidate.points_uv.copy()
        points.setflags(write=False)
        deltas = np.diff(points.astype(np.float64), axis=0)
        selected_heading_rad = math.radians(candidate.heading_deg)
        return ImageSpacePathProposal(
            valid=bool(valid),
            points_uv=points,
            mean_score=float(mean_score),
            minimum_score=float(candidate.traversability_low_percentile),
            reason=reason,
            target_heading_error_rad=target_heading_error_rad,
            selected_heading_rad=selected_heading_rad,
            heading_residual_rad=(
                None
                if target_heading_error_rad is None
                else normalize_angle_rad(target_heading_error_rad - selected_heading_rad)
            ),
            target_uv=(int(points[-1, 0]), int(points[-1, 1])),
            path_length_px=float(np.linalg.norm(deltas, axis=1).sum()) if len(points) > 1 else 0.0,
            goal_alignment_weight=self.config.goal_heading_weight,
            smoothing_method="MOTION_PRIMITIVE_TIME_HYSTERESIS",
            smoothing_applied=True,
            smoothing_iterations=0,
        )

    def _reason(
        self,
        selected: CandidateScore | None,
        *,
        near_field_safe: bool,
        trajectory_valid: bool,
        using_held: bool,
    ) -> str:
        if not near_field_safe:
            return "MOTION_PRIMITIVE_NEAR_FIELD_UNSAFE"
        if selected is None:
            return "MOTION_PRIMITIVE_NO_SAFE_CANDIDATE"
        if using_held:
            return "MOTION_PRIMITIVE_HELD_TRANSIENT_INVALID"
        if not trajectory_valid:
            return "MOTION_PRIMITIVE_LOW_QUALITY"
        return "MOTION_PRIMITIVE_SELECTED"

    def _confidence(
        self,
        selected: CandidateScore | None,
        *,
        near_field_safe: bool,
        trajectory_valid: bool,
        plan_age_sec: float | None,
        using_held: bool,
    ) -> float:
        if selected is None or not near_field_safe:
            return 0.0
        score_factor = max(0.0, min(1.0, selected.traversability_weighted_mean))
        near_factor = max(0.0, min(1.0, selected.near_field_low_percentile))
        quality_factor = 1.0 if trajectory_valid else 0.55
        if plan_age_sec is None:
            freshness = 1.0
        else:
            freshness = max(0.0, 1.0 - plan_age_sec / max(1e-6, self.config.max_plan_age_sec))
        held_factor = 0.65 if using_held else 1.0
        return float(max(0.0, min(1.0, score_factor * 0.45 + near_factor * 0.35 + freshness * 0.20)) * quality_factor * held_factor)


def primitive_curve_points(
    shape: tuple[int, int],
    heading_deg: float,
    *,
    maximum_visual_heading_deg: float = 55.0,
) -> np.ndarray:
    height, width = int(shape[0]), int(shape[1])
    start_y = min(height - 1, max(0, round(height * 0.92)))
    end_y = min(start_y, max(0, round(height * 0.52)))
    count = max(12, min(48, height // 5))
    ys = np.linspace(start_y, end_y, count)
    progress = np.linspace(0.0, 1.0, count)
    center_x = (width - 1) / 2.0
    normalized = float(np.clip(float(heading_deg) / max(1.0, maximum_visual_heading_deg), -1.0, 1.0))
    # Mission1 uses compass/SDK-compatible heading offsets:
    # positive heading is clockwise/right, which is larger image x.
    lateral = normalized * width * 0.42
    xs = center_x + lateral * (progress**1.45)
    points = np.stack(
        (
            np.clip(np.rint(xs), 0, width - 1),
            np.clip(np.rint(ys), 0, height - 1),
        ),
        axis=1,
    ).astype(np.int32)
    # Keep monotonic row samples and remove duplicates introduced by rounding.
    _, unique_indices = np.unique(points[:, 1] * width + points[:, 0], return_index=True)
    points = points[np.sort(unique_indices)]
    return points


def selected_candidate_endpoint_x_offset_px(points_uv: np.ndarray) -> int | None:
    if points_uv.size == 0:
        return None
    width_center = (int(np.max(points_uv[:, 0])) + int(np.min(points_uv[:, 0]))) / 2.0
    # The primitive starts at the camera center; use it as the local origin
    # instead of the min/max of the curved points when available.
    start_x = float(points_uv[0, 0])
    return int(round(float(points_uv[-1, 0]) - start_x if math.isfinite(start_x) else width_center))


def image_direction_from_x_offset(offset_px: int | float | None, *, deadband_px: float = 1.0) -> str | None:
    if offset_px is None:
        return None
    value = float(offset_px)
    if value > deadband_px:
        return "RIGHT"
    if value < -deadband_px:
        return "LEFT"
    return "CENTER"


def corridor_mask(
    shape: tuple[int, int],
    points_uv: np.ndarray,
    half_width_ratio: float,
) -> np.ndarray:
    height, width = int(shape[0]), int(shape[1])
    half_width = max(1, int(round(width * float(half_width_ratio))))
    mask = np.zeros((height, width), dtype=np.uint8)
    if len(points_uv) == 1:
        cv2.circle(mask, tuple(int(v) for v in points_uv[0]), half_width, 1, -1)
    elif len(points_uv) > 1:
        cv2.polylines(mask, [points_uv.astype(np.int32)], False, 1, half_width * 2 + 1, cv2.LINE_AA)
    return mask.astype(bool)


def primitive_sample_weights(rows: np.ndarray, height: int) -> np.ndarray:
    # Bottom/near field should dominate the decision because it is closest to
    # the rover footprint.
    near = np.clip(rows.astype(np.float64) / max(1.0, float(height - 1)), 0.0, 1.0)
    weights = 0.65 + 0.70 * near
    return weights.astype(np.float64)
