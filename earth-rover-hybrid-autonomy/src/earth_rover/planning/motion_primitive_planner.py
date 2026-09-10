from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from training.sam_tp_phase1_review import ImageSpacePathProposal


def normalize_angle_deg(value: float) -> float:
    """Return an angle in [-180, 180)."""

    return (float(value) + 180.0) % 360.0 - 180.0


def normalize_angle_rad(value: float) -> float:
    return math.atan2(math.sin(float(value)), math.cos(float(value)))


@dataclass(frozen=True)
class MotionPrimitivePlannerConfig:
    mode: str = "motion_primitives"
    candidate_headings_deg: tuple[float, ...] = (-45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0)
    traversability_weight: float = 1.0
    goal_heading_weight: float = 0.7
    continuity_weight: float = 0.45
    curvature_weight: float = 0.10
    near_field_risk_weight: float = 1.5
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
    corridor_half_width_ratio: float = 0.025
    maximum_visual_heading_deg: float = 55.0
    debug_candidate_scores: bool = True

    # GeNIE-paper-style path generation (Sec III-D, Algorithm 1): sample a fan
    # of candidate paths, keep the top-K by traversability, cluster them
    # (silhouette-selected k-means) to fuse near-duplicates, merge close
    # clusters, then pick the cluster whose heading is angularly closest to
    # the GPS goal direction. Only used when mode == "genie_cluster"; the
    # temporal hold/confirm gating below still applies so the live controller
    # does not oscillate between angularly-close clusters frame to frame --
    # the paper itself does not specify live-control temporal behavior.
    genie_n_candidates: int = 25
    genie_top_k: int = 8
    genie_k_max: int = 6
    genie_waypoint_count: int = 10
    genie_curvature_jitter: float = 0.2
    genie_merge_threshold_ratio: float = 0.05
    genie_switch_heading_deadband_deg: float = 6.0

    @classmethod
    def from_dict(cls, config: dict[str, Any] | None) -> "MotionPrimitivePlannerConfig":
        values = dict(config or {})
        if "candidate_headings_deg" in values:
            values["candidate_headings_deg"] = tuple(float(v) for v in values["candidate_headings_deg"])
        return cls(**{key: values[key] for key in cls.__dataclass_fields__ if key in values})

    def validate(self) -> None:
        if self.mode not in {"connected_path", "motion_primitives", "gps_only", "genie_cluster"}:
            raise ValueError(
                "planner.mode must be connected_path, motion_primitives, gps_only, or genie_cluster"
            )
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
        if self.switch_confirm_count < 1:
            raise ValueError("planner.switch_confirm_count must be >= 1")
        for name in (
            "min_candidate_commit_sec",
            "transient_invalid_grace_sec",
            "max_plan_age_sec",
            "corridor_half_width_ratio",
            "maximum_visual_heading_deg",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"planner.{name} must be finite and positive")
        if self.mode == "genie_cluster":
            if self.genie_n_candidates < 5:
                raise ValueError("planner.genie_n_candidates must be >= 5")
            if self.genie_top_k < 1:
                raise ValueError("planner.genie_top_k must be >= 1")
            if self.genie_k_max < 2:
                raise ValueError("planner.genie_k_max must be >= 2")
            if self.genie_waypoint_count < 3:
                raise ValueError("planner.genie_waypoint_count must be >= 3")
            if self.genie_curvature_jitter < 0.0:
                raise ValueError("planner.genie_curvature_jitter must be >= 0")
            if not math.isfinite(self.genie_merge_threshold_ratio) or self.genie_merge_threshold_ratio <= 0.0:
                raise ValueError("planner.genie_merge_threshold_ratio must be finite and positive")
            if (
                not math.isfinite(self.genie_switch_heading_deadband_deg)
                or self.genie_switch_heading_deadband_deg < 0.0
            ):
                raise ValueError("planner.genie_switch_heading_deadband_deg must be finite and non-negative")


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
    final_score_raw: float
    final_score: float
    hard_rejected: bool
    reject_reason: str | None
    points_uv: np.ndarray = field(repr=False)

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
            "final_score": self.final_score,
            "hard_rejected": self.hard_rejected,
            "reject_reason": self.reject_reason,
        }


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
    selected_candidate_since: float | None
    last_valid_candidate_time: float | None
    switch_pending_index: int | None
    switch_confirm_count: int

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
            "plan_age_sec": self.plan_age_sec,
            "planner_confidence": self.planner_confidence,
            "near_field_safe": self.near_field_safe,
            "near_field_score": self.near_field_score,
            "trajectory_valid": self.trajectory_valid,
            "trajectory_quality": self.trajectory_quality,
            "using_held_plan": self.using_held_plan,
            "candidate_switch_pending": self.switch_pending_index,
            "candidate_switch_confirm_count": self.switch_confirm_count,
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
        self._previous_target_heading_deg: float | None = None
        self._previous_checkpoint_sequence: int | None = None
        # genie_cluster mode keeps its own committed-heading state instead of
        # reusing the fixed-candidate-index state above, because the number
        # and identity of clusters can change from frame to frame. It still
        # shares self._last_valid_candidate_time/self._last_plan with the
        # other modes for the transient-hold grace period in plan()'s tail.
        self._genie_selected_heading: float | None = None
        self._genie_selected_since: float | None = None
        self._genie_switch_pending_heading: float | None = None
        self._genie_switch_confirm_count = 0

    def reset(self) -> None:
        self._selected_index = None
        self._selected_since = None
        self._last_valid_candidate_time = None
        self._last_plan = None
        self._score_ema.clear()
        self._switch_pending_index = None
        self._switch_confirm_count = 0
        self._previous_target_heading_deg = None
        self._previous_checkpoint_sequence = None
        self._genie_selected_heading = None
        self._genie_selected_since = None
        self._genie_switch_pending_heading = None
        self._genie_switch_confirm_count = 0

    def plan(
        self,
        score_map: np.ndarray,
        valid_mask: np.ndarray,
        *,
        target_heading_error_rad: float | None,
        checkpoint_sequence: int | None = None,
        timestamp: float | None = None,
        navigation: dict[str, Any] | None = None,
    ) -> MotionPrimitivePlan:
        score = np.asarray(score_map, dtype=np.float32)
        valid = np.asarray(valid_mask, dtype=bool)
        if score.ndim != 2 or valid.shape != score.shape:
            raise ValueError("score_map and valid_mask must be matching 2D arrays")
        if score.size == 0 or not np.isfinite(score).all():
            raise ValueError("score_map must be finite and non-empty")
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
            )

        if self.config.mode == "genie_cluster":
            candidates, selected, switched, switch_reason, near_field_safe, near_field_score = (
                self._genie_cluster_plan(
                    score,
                    valid,
                    target_heading_deg=target_heading_deg,
                    now=now,
                    checkpoint_changed=checkpoint_changed,
                    large_target_change=large_target_change,
                )
            )
        else:
            candidates = self._score_candidates(score, valid, target_heading_deg)
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

            selected, switched, switch_reason = self._select_candidate(
                safe_candidates,
                now=now,
                checkpoint_changed=checkpoint_changed,
                large_target_change=large_target_change,
            )
        using_held = False
        plan_age = None
        if selected is None:
            held = self._last_plan
            if (
                held is not None
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
            selected_candidate_since=(
                self._genie_selected_since if self.config.mode == "genie_cluster" else self._selected_since
            ),
            last_valid_candidate_time=self._last_valid_candidate_time,
            switch_pending_index=(
                None if self.config.mode == "genie_cluster" else self._switch_pending_index
            ),
            switch_confirm_count=(
                self._genie_switch_confirm_count
                if self.config.mode == "genie_cluster"
                else self._switch_confirm_count
            ),
        )
        self._last_plan = plan
        return plan

    def _score_candidates(
        self,
        score: np.ndarray,
        valid: np.ndarray,
        target_heading_deg: float,
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
            if self._selected_index is None:
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
            hard_rejected = near_low < self.config.near_field_stop_threshold
            final_raw = (
                self.config.traversability_weight * (0.70 * weighted_mean + 0.30 * low_percentile)
                - self.config.goal_heading_weight * goal_penalty
                - self.config.continuity_weight * continuity_penalty
                - self.config.curvature_weight * curvature_mag
                - self.config.near_field_risk_weight * near_risk
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
                    final_score_raw=final_raw,
                    final_score=final_raw,
                    hard_rejected=hard_rejected,
                    reject_reason="NEAR_FIELD_UNSAFE" if hard_rejected else None,
                    points_uv=points,
                )
            )
        return tuple(candidates)

    def _select_candidate(
        self,
        safe_candidates: list[CandidateScore],
        *,
        now: float,
        checkpoint_changed: bool,
        large_target_change: bool,
    ) -> tuple[CandidateScore | None, bool, str | None]:
        if not safe_candidates:
            return None, False, "all_candidates_hard_rejected"
        best = max(safe_candidates, key=lambda item: item.final_score)
        current = next(
            (candidate for candidate in safe_candidates if candidate.index == self._selected_index),
            None,
        )
        immediate = checkpoint_changed or large_target_change or current is None
        if current is not None and self._last_valid_candidate_time is not None:
            if now - self._last_valid_candidate_time > self.config.max_plan_age_sec:
                immediate = True
        if immediate:
            changed = self._selected_index != best.index
            self._commit(best.index, now)
            return best, changed, "immediate_reset" if changed else None
        assert current is not None
        if self._selected_since is not None and now - self._selected_since < self.config.min_candidate_commit_sec:
            self._switch_pending_index = None
            self._switch_confirm_count = 0
            return current, False, None
        if best.index == current.index:
            self._switch_pending_index = None
            self._switch_confirm_count = 0
            return current, False, None
        if best.final_score < current.final_score + self.config.switch_score_margin:
            self._switch_pending_index = None
            self._switch_confirm_count = 0
            return current, False, None
        if self._switch_pending_index == best.index:
            self._switch_confirm_count += 1
        else:
            self._switch_pending_index = best.index
            self._switch_confirm_count = 1
        if self._switch_confirm_count >= self.config.switch_confirm_count:
            self._commit(best.index, now)
            return best, True, "score_margin_confirmed"
        return current, False, "switch_pending"

    def _commit(self, index: int, now: float) -> None:
        if self._selected_index != index:
            self._selected_since = now
        elif self._selected_since is None:
            self._selected_since = now
        self._selected_index = index
        self._switch_pending_index = None
        self._switch_confirm_count = 0

    def _gps_only_plan(
        self,
        score: np.ndarray,
        valid: np.ndarray,
        *,
        target_heading_deg: float,
        target_heading_error_rad: float | None,
        now: float,
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
            selected_candidate_since=self._selected_since,
            last_valid_candidate_time=self._last_valid_candidate_time,
            switch_pending_index=None,
            switch_confirm_count=0,
        )
        self._last_plan = plan
        return plan

    def _build_genie_candidate(
        self,
        score: np.ndarray,
        valid: np.ndarray,
        heading_deg: float,
        *,
        target_heading_deg: float,
        maximum_heading: float,
        points_xy: np.ndarray | None = None,
        curvature_exponent: float = 1.45,
    ) -> CandidateScore | None:
        shape = score.shape
        if points_xy is None:
            points_xy = genie_candidate_points(
                shape,
                heading_deg,
                maximum_visual_heading_deg=maximum_heading,
                n_waypoints=self.config.genie_waypoint_count,
                curvature_exponent=curvature_exponent,
            )
        sampled, sampled_valid, rows, _cols = sample_score_along_path(score, valid, points_xy)
        valid_sampled = sampled[sampled_valid]
        if valid_sampled.size == 0:
            return None
        weights = primitive_sample_weights(rows, shape[0])
        weighted_mean = float(np.average(sampled, weights=weights))
        low_percentile = float(np.percentile(valid_sampled, self.config.full_path_percentile))
        near_count = max(2, int(math.ceil(len(points_xy) * 0.35)))
        near_scores = sampled[:near_count]
        near_low = float(np.percentile(near_scores, self.config.near_field_percentile))
        hard_rejected = near_low < self.config.near_field_stop_threshold
        goal_error = abs(normalize_angle_deg(float(heading_deg) - target_heading_deg))
        points_int = np.stack(
            (
                np.clip(np.rint(points_xy[:, 0]), 0, shape[1] - 1),
                np.clip(np.rint(points_xy[:, 1]), 0, shape[0] - 1),
            ),
            axis=1,
        ).astype(np.int32)
        rank = 0.70 * weighted_mean + 0.30 * low_percentile
        return CandidateScore(
            index=0,
            heading_deg=float(heading_deg),
            traversability_weighted_mean=weighted_mean,
            traversability_low_percentile=low_percentile,
            near_field_mean=float(np.mean(near_scores)),
            near_field_low_percentile=near_low,
            goal_heading_error_deg=goal_error,
            goal_penalty=min(1.0, goal_error / maximum_heading),
            continuity_delta_deg=0.0,
            continuity_penalty=0.0,
            curvature_magnitude=min(1.0, abs(float(heading_deg)) / maximum_heading),
            curvature_penalty=min(1.0, abs(float(heading_deg)) / maximum_heading),
            near_field_risk_penalty=max(0.0, self.config.near_field_stop_threshold - near_low),
            final_score_raw=rank,
            final_score=rank,
            hard_rejected=hard_rejected,
            reject_reason="NEAR_FIELD_UNSAFE" if hard_rejected else None,
            points_uv=points_int,
        )

    def _genie_cluster_plan(
        self,
        score: np.ndarray,
        valid: np.ndarray,
        *,
        target_heading_deg: float,
        now: float,
        checkpoint_changed: bool,
        large_target_change: bool,
    ) -> tuple[tuple[CandidateScore, ...], CandidateScore | None, bool, str | None, bool, float]:
        """GeNIE Algorithm 1 in image-space: sample -> top-K -> cluster ->
        merge -> pick the cluster angularly closest to the GPS goal heading.

        A committed-heading hold/confirm gate (mirroring ``_select_candidate``
        but keyed on heading instead of a fixed candidate index, since cluster
        identity is not stable frame to frame) prevents oscillation between
        angularly-close clusters; the paper's Algorithm 1 itself is a
        per-frame selection with no live-control temporal behavior specified.
        """

        shape = score.shape
        width = shape[1]
        maximum_heading = max(1.0, float(self.config.maximum_visual_heading_deg))
        n_candidates = max(5, int(self.config.genie_n_candidates))
        headings = np.linspace(-maximum_heading, maximum_heading, n_candidates)
        rng = np.random.default_rng(0)

        fan: list[CandidateScore] = []
        for heading in headings:
            exponent = 1.45 + float(
                rng.uniform(-self.config.genie_curvature_jitter, self.config.genie_curvature_jitter)
            )
            candidate = self._build_genie_candidate(
                score,
                valid,
                float(heading),
                target_heading_deg=target_heading_deg,
                maximum_heading=maximum_heading,
                curvature_exponent=exponent,
            )
            if candidate is not None:
                fan.append(candidate)

        near_field_score = max((candidate.near_field_low_percentile for candidate in fan), default=0.0)
        all_unsafe = bool(fan) and all(candidate.hard_rejected for candidate in fan)
        near_field_safe = bool(fan) and not all_unsafe and near_field_score >= self.config.near_field_stop_threshold

        safe = [candidate for candidate in fan if not candidate.hard_rejected]
        top_k = sorted(safe, key=lambda candidate: -candidate.final_score)[: max(1, int(self.config.genie_top_k))]

        fresh_heading: float | None = None
        fresh_points: np.ndarray | None = None
        if top_k:
            paths = np.stack([candidate.points_uv.astype(np.float64) for candidate in top_k])
            _labels, centers = adaptive_kmeans_paths(paths, k_max=self.config.genie_k_max, seed=0)
            merge_threshold_px = max(1.0, self.config.genie_merge_threshold_ratio * width)
            merged = merge_close_path_clusters(centers, threshold_px=merge_threshold_px)
            merged_headings = [
                path_heading_from_endpoint_deg(path, width=width, maximum_visual_heading_deg=maximum_heading)
                for path in merged
            ]
            best_index = int(
                np.argmin([abs(normalize_angle_deg(h - target_heading_deg)) for h in merged_headings])
            )
            fresh_heading = merged_headings[best_index]
            fresh_points = merged[best_index]

        current_heading = self._genie_selected_heading
        immediate = checkpoint_changed or large_target_change or current_heading is None
        if (
            current_heading is not None
            and self._last_valid_candidate_time is not None
            and now - self._last_valid_candidate_time > self.config.max_plan_age_sec
        ):
            immediate = True

        selected: CandidateScore | None = None
        switched = False
        switch_reason: str | None = None

        if immediate:
            if fresh_heading is not None:
                selected = self._build_genie_candidate(
                    score,
                    valid,
                    fresh_heading,
                    target_heading_deg=target_heading_deg,
                    maximum_heading=maximum_heading,
                    points_xy=fresh_points,
                )
                changed = (
                    current_heading is None
                    or abs(normalize_angle_deg(fresh_heading - current_heading)) > 1e-6
                )
                self._genie_commit(fresh_heading, now)
                switched = changed
                switch_reason = "immediate_reset" if changed else None
            else:
                switch_reason = "all_candidates_hard_rejected"
        else:
            held = self._build_genie_candidate(
                score,
                valid,
                current_heading,
                target_heading_deg=target_heading_deg,
                maximum_heading=maximum_heading,
            )
            holding_unsafe = held is None or held.hard_rejected
            if holding_unsafe and fresh_heading is not None:
                selected = self._build_genie_candidate(
                    score,
                    valid,
                    fresh_heading,
                    target_heading_deg=target_heading_deg,
                    maximum_heading=maximum_heading,
                    points_xy=fresh_points,
                )
                self._genie_commit(fresh_heading, now)
                switched = True
                switch_reason = "held_heading_became_unsafe"
            elif holding_unsafe:
                switch_reason = "held_heading_unsafe_no_alternative"
            elif (
                self._genie_selected_since is not None
                and now - self._genie_selected_since < self.config.min_candidate_commit_sec
            ):
                selected = held
                self._genie_switch_pending_heading = None
                self._genie_switch_confirm_count = 0
            elif fresh_heading is None:
                selected = held
            else:
                heading_delta = abs(normalize_angle_deg(fresh_heading - current_heading))
                if heading_delta <= self.config.genie_switch_heading_deadband_deg:
                    selected = held
                    self._genie_switch_pending_heading = None
                    self._genie_switch_confirm_count = 0
                else:
                    if (
                        self._genie_switch_pending_heading is not None
                        and abs(normalize_angle_deg(self._genie_switch_pending_heading - fresh_heading))
                        <= self.config.genie_switch_heading_deadband_deg
                    ):
                        self._genie_switch_confirm_count += 1
                    else:
                        self._genie_switch_pending_heading = fresh_heading
                        self._genie_switch_confirm_count = 1
                    if self._genie_switch_confirm_count >= self.config.switch_confirm_count:
                        selected = self._build_genie_candidate(
                            score,
                            valid,
                            fresh_heading,
                            target_heading_deg=target_heading_deg,
                            maximum_heading=maximum_heading,
                            points_xy=fresh_points,
                        )
                        self._genie_commit(fresh_heading, now)
                        switched = True
                        switch_reason = "angular_selection_confirmed"
                    else:
                        selected = held
                        switch_reason = "switch_pending"

        return tuple(top_k), selected, switched, switch_reason, near_field_safe, float(near_field_score)

    def _genie_commit(self, heading_deg: float, now: float) -> None:
        if self._genie_selected_heading != heading_deg:
            self._genie_selected_since = now
        elif self._genie_selected_since is None:
            self._genie_selected_since = now
        self._genie_selected_heading = heading_deg
        self._genie_switch_pending_heading = None
        self._genie_switch_confirm_count = 0

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


# ---------------------------------------------------------------------------
# GeNIE-paper-style candidate generation and fusion (Sec III-D, Algorithm 1),
# ported to image-space. See scripts/06_path_planning.py in the sibling
# GeNIE_ws workspace for the original BEV/metric-space reproduction this
# mirrors -- there is no camera calibration here, so paths stay in pixels
# rather than meters.
# ---------------------------------------------------------------------------


def genie_candidate_points(
    shape: tuple[int, int],
    heading_deg: float,
    *,
    maximum_visual_heading_deg: float,
    n_waypoints: int,
    curvature_exponent: float = 1.45,
) -> np.ndarray:
    """Float (x, y) image-space waypoints for one candidate fan path.

    Every candidate uses the same fixed waypoint count and the same
    parametric row schedule (``t``), so waypoint ``i`` of any two candidates
    is directly comparable -- this correspondence is what makes the pairwise
    path distance used by clustering below meaningful.
    """

    height, width = int(shape[0]), int(shape[1])
    start_y = min(height - 1, max(0, height * 0.92))
    end_y = min(start_y, max(0, height * 0.52))
    t = np.linspace(0.0, 1.0, int(n_waypoints))
    ys = start_y + (end_y - start_y) * t
    center_x = (width - 1) / 2.0
    normalized = float(np.clip(float(heading_deg) / max(1.0, maximum_visual_heading_deg), -1.0, 1.0))
    lateral = normalized * width * 0.42
    xs = center_x + lateral * (t**curvature_exponent)
    return np.stack((xs, ys), axis=1)


def sample_score_along_path(
    score: np.ndarray,
    valid: np.ndarray,
    points_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Nearest-pixel sample of ``score``/``valid`` along float waypoints."""

    height, width = score.shape
    cols = np.clip(np.rint(points_xy[:, 0]).astype(np.int32), 0, width - 1)
    rows = np.clip(np.rint(points_xy[:, 1]).astype(np.int32), 0, height - 1)
    return score[rows, cols], valid[rows, cols], rows, cols


def path_heading_from_endpoint_deg(
    path_xy: np.ndarray,
    *,
    width: int,
    maximum_visual_heading_deg: float,
) -> float:
    """Invert ``genie_candidate_points``'s endpoint offset back to a heading.

    The curvature exponent only reshapes points before the final waypoint
    (``t < 1``); at ``t == 1`` the offset is exactly
    ``normalized * width * 0.42`` regardless of curvature, so this inversion
    is exact for a single sampled candidate and a good linear approximation
    for a cluster centroid/merge that averages several such candidates.
    """

    offset = float(path_xy[-1, 0] - path_xy[0, 0])
    normalized = float(np.clip(offset / max(1e-6, width * 0.42), -1.0, 1.0))
    return normalized * max(1.0, maximum_visual_heading_deg)


def _pairwise_path_distance(paths: np.ndarray) -> np.ndarray:
    """Mean per-waypoint Euclidean distance between every pair of paths.

    ``paths``: (n, n_waypoints, 2). Matches GeNIE Sec III-D's definition of
    path-to-path distance for clustering/merging.
    """

    diff = paths[:, None, :, :] - paths[None, :, :, :]
    return np.linalg.norm(diff, axis=-1).mean(axis=-1)


def _kmeans_paths(
    paths: np.ndarray,
    k: int,
    *,
    n_iter: int = 30,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(paths)
    centers = paths[rng.choice(n, size=k, replace=False)].copy()
    labels = np.full(n, -1)
    for _ in range(n_iter):
        distances = np.linalg.norm(paths[:, None, :, :] - centers[None, :, :, :], axis=-1).mean(axis=-1)
        new_labels = distances.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for cluster in range(k):
            members = labels == cluster
            if members.any():
                centers[cluster] = paths[members].mean(axis=0)
    return labels, centers


def _silhouette_score_paths(paths: np.ndarray, labels: np.ndarray) -> float:
    distance = _pairwise_path_distance(paths)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return -1.0
    n = len(paths)
    silhouette = np.zeros(n)
    for i in range(n):
        same = labels == labels[i]
        same[i] = False
        a = distance[i, same].mean() if same.any() else 0.0
        b = min(distance[i, labels == cluster].mean() for cluster in unique_labels if cluster != labels[i])
        silhouette[i] = 0.0 if max(a, b) == 0 else (b - a) / max(a, b)
    return float(silhouette.mean())


def adaptive_kmeans_paths(
    paths: np.ndarray,
    *,
    k_max: int = 6,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """k-means over waypoint sequences with k chosen by best silhouette score.

    Mirrors GeNIE's "the number of clusters k is determined by optimizing the
    silhouette loss" (Sec III-D). Fewer than 3 paths skip clustering entirely.
    """

    n = len(paths)
    k_max = min(k_max, n - 1)
    if n < 3 or k_max < 2:
        return np.zeros(n, dtype=int), paths.mean(axis=0, keepdims=True)

    best_score = -2.0
    best_labels = np.zeros(n, dtype=int)
    best_centers = paths.mean(axis=0, keepdims=True)
    for k in range(2, k_max + 1):
        labels, centers = _kmeans_paths(paths, k, seed=seed)
        if len(np.unique(labels)) < 2:
            continue
        score = _silhouette_score_paths(paths, labels)
        if score > best_score:
            best_score, best_labels, best_centers = score, labels, centers
    return best_labels, best_centers


def merge_close_path_clusters(centers: np.ndarray, *, threshold_px: float) -> list[np.ndarray]:
    """Union-find merge of cluster centroids within ``threshold_px`` of each other.

    GeNIE prefers under-merging (treating two genuinely different paths as
    one is a collision risk) over over-merging (a duplicate cluster is just
    redundant), so this threshold should stay conservative.
    """

    k = len(centers)
    parent = list(range(k))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_a] = root_b

    for i in range(k):
        for j in range(i + 1, k):
            distance = np.linalg.norm(centers[i] - centers[j], axis=-1).mean()
            if distance <= threshold_px:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(k):
        groups.setdefault(find(i), []).append(i)
    return [centers[indices].mean(axis=0) for indices in groups.values()]
