from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from earth_rover.perception.camera_projection import project_trajectory
from earth_rover.planning.motion_primitive_planner import (
    MotionPrimitivePlanner,
    MotionPrimitivePlannerConfig,
    evaluate_side_sectors,
    image_direction_from_x_offset,
    metric_terminal_heading_deg,
    normalize_angle_deg,
    primitive_curve_points,
    selected_candidate_endpoint_x_offset_px,
)

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from test_camera_projection import _make_calibration  # noqa: E402


class Clock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


def planner(clock: Clock, **overrides) -> MotionPrimitivePlanner:
    values = {
        "candidate_score_ema_alpha": 1.0,
        "min_candidate_commit_sec": 0.8,
        "switch_score_margin": 0.08,
        "switch_confirm_count": 2,
        "transient_invalid_grace_sec": 0.8,
        "max_plan_age_sec": 1.5,
    } | overrides
    config = MotionPrimitivePlannerConfig(**values)
    return MotionPrimitivePlanner(config, monotonic=clock)


def score_map_for_heading(
    heading_deg: float,
    *,
    base: float = 0.35,
    path_score: float = 0.90,
    shape: tuple[int, int] = (120, 160),
) -> tuple[np.ndarray, np.ndarray]:
    score = np.full(shape, base, dtype=np.float32)
    valid = np.ones(shape, dtype=bool)
    points = primitive_curve_points(shape, heading_deg)
    for x, y in points:
        score[max(0, y - 2) : min(shape[0], y + 3), max(0, x - 3) : min(shape[1], x + 4)] = path_score
    return score, valid


def block_near_field(score: np.ndarray, heading_deg: float, value: float = 0.05) -> None:
    points = primitive_curve_points(score.shape, heading_deg)
    near_count = max(2, int(math.ceil(len(points) * 0.35)))
    # Do not block the shared rover-origin footprint; this represents the
    # straight-ahead corridor becoming unsafe slightly farther out while an
    # adjacent primitive can still be selected.
    for x, y in points[max(2, near_count // 2) : near_count]:
        score[max(0, y - 3) : min(score.shape[0], y + 4), max(0, x - 4) : min(score.shape[1], x + 5)] = value


def test_selects_straight_when_gps_and_score_prefer_straight() -> None:
    clock = Clock()
    local = planner(clock)
    score, valid = score_map_for_heading(0.0)

    plan = local.plan(score, valid, target_heading_error_rad=0.0)

    assert plan.path_valid is True
    assert plan.selected_candidate is not None
    assert plan.selected_candidate.heading_deg == pytest.approx(0.0)
    assert plan.image_path.reason == "MOTION_PRIMITIVE_SELECTED"


def test_candidate_geometry_uses_positive_clockwise_right_convention() -> None:
    shape = (120, 160)

    right = primitive_curve_points(shape, 30.0)
    left = primitive_curve_points(shape, -30.0)
    straight = primitive_curve_points(shape, 0.0)

    right_offset = selected_candidate_endpoint_x_offset_px(right)
    left_offset = selected_candidate_endpoint_x_offset_px(left)
    straight_offset = selected_candidate_endpoint_x_offset_px(straight)

    assert right_offset > 0
    assert left_offset < 0
    assert abs(straight_offset) <= 1
    assert image_direction_from_x_offset(right_offset) == "RIGHT"
    assert image_direction_from_x_offset(left_offset) == "LEFT"
    assert image_direction_from_x_offset(straight_offset) == "CENTER"


def test_positive_gps_error_selects_right_image_candidate() -> None:
    clock = Clock()
    local = planner(clock)
    score, valid = score_map_for_heading(30.0)

    plan = local.plan(score, valid, target_heading_error_rad=math.radians(30.0))

    assert plan.selected_candidate is not None
    assert plan.selected_candidate.heading_deg > 0.0
    assert plan.to_status()["heading_convention"] == "positive_clockwise_right"
    assert plan.to_status()["selected_candidate_image_direction"] == "RIGHT"


def test_negative_gps_error_selects_left_image_candidate() -> None:
    clock = Clock()
    local = planner(clock)
    score, valid = score_map_for_heading(-30.0)

    plan = local.plan(score, valid, target_heading_error_rad=math.radians(-30.0))

    assert plan.selected_candidate is not None
    assert plan.selected_candidate.heading_deg < 0.0
    assert plan.to_status()["selected_candidate_image_direction"] == "LEFT"


def test_uses_adjacent_safe_candidate_when_straight_near_field_is_dangerous() -> None:
    clock = Clock()
    local = planner(clock)
    score, valid = score_map_for_heading(15.0, base=0.55, path_score=0.90)
    block_near_field(score, 0.0)

    plan = local.plan(score, valid, target_heading_error_rad=0.0)

    assert plan.near_field_safe is True
    assert plan.selected_candidate is not None
    assert abs(plan.selected_candidate.heading_deg) in {15.0, 30.0, 45.0}


def test_near_field_soft_penalty_is_continuous_above_the_hard_threshold() -> None:
    # near_field_risk_penalty (the hard-threshold term) is 0 for any
    # candidate above near_field_stop_threshold, so two "safe by threshold"
    # candidates score identically on safety with the hard term alone.
    # near_field_soft_penalty must differ even when both are "safe".
    clock = Clock()
    local = planner(clock, near_field_stop_threshold=0.5)
    score, valid = score_map_for_heading(0.0, base=0.9, path_score=0.9)
    block_near_field(score, 0.0, value=0.51)  # just above the hard threshold

    plan = local.plan(score, valid, target_heading_error_rad=0.0)

    straight = next(c for c in plan.candidate_scores if c.heading_deg == 0.0)
    wide = next(c for c in plan.candidate_scores if c.heading_deg == -45.0)
    assert straight.near_field_risk_penalty == 0.0  # "safe" by the hard threshold
    assert wide.near_field_risk_penalty == 0.0
    assert straight.near_field_soft_penalty == pytest.approx(
        1.0 - straight.near_field_low_percentile
    )
    # Still meaningfully less safe than the wide, clear candidate even
    # though the hard term can't see any difference between them.
    assert straight.near_field_soft_penalty > wide.near_field_soft_penalty


def test_near_field_soft_penalty_can_prefer_a_safer_candidate_over_goal_alignment() -> None:
    # The actual bug report: the rover needed to turn back toward the open
    # road but kept driving toward a wall instead, because the wall-facing
    # candidate was still "safe by threshold" and better goal-aligned, and
    # nothing before this penalty existed to weigh "how much safer" once a
    # candidate cleared the hard threshold.
    clock = Clock()
    score, valid = score_map_for_heading(0.0, base=0.9, path_score=0.9)
    block_near_field(score, 0.0, value=0.51)  # straight: technically safe, but close to a wall

    without_soft_term = planner(
        clock, near_field_stop_threshold=0.5, near_field_soft_risk_weight=0.0
    )
    plan_without = without_soft_term.plan(
        score.copy(), valid, target_heading_error_rad=math.radians(5.0)
    )
    assert plan_without.selected_candidate.heading_deg == pytest.approx(0.0)

    with_soft_term = planner(
        clock, near_field_stop_threshold=0.5, near_field_soft_risk_weight=1.5
    )
    plan_with = with_soft_term.plan(
        score.copy(), valid, target_heading_error_rad=math.radians(5.0)
    )
    assert plan_with.selected_candidate.heading_deg != 0.0


def test_small_score_gain_keeps_committed_candidate() -> None:
    clock = Clock()
    local = planner(clock)
    straight, valid = score_map_for_heading(0.0, path_score=0.90)
    first = local.plan(straight, valid, target_heading_error_rad=0.0)
    clock.value += 1.0
    left, valid = score_map_for_heading(15.0, base=0.50, path_score=0.91)
    second = local.plan(left, valid, target_heading_error_rad=math.radians(15.0))

    assert first.selected_candidate.heading_deg == pytest.approx(0.0)
    assert second.selected_candidate.heading_deg == pytest.approx(0.0)
    assert second.candidate_switched is False


def test_switch_requires_margin_and_confirmation_count() -> None:
    clock = Clock()
    local = planner(clock)
    straight, valid = score_map_for_heading(0.0, path_score=0.90)
    local.plan(straight, valid, target_heading_error_rad=0.0)
    clock.value += 1.0
    left, valid = score_map_for_heading(30.0, base=0.58, path_score=1.0)
    pending = local.plan(left, valid, target_heading_error_rad=math.radians(30.0))
    clock.value += 0.2
    switched = local.plan(left, valid, target_heading_error_rad=math.radians(30.0))

    assert pending.selected_candidate.heading_deg == pytest.approx(0.0)
    assert pending.switch_reason == "switch_pending"
    assert switched.selected_candidate.heading_deg == pytest.approx(15.0)
    assert switched.candidate_switched is True


def test_large_target_change_does_not_keep_wall_side_candidate_by_continuity() -> None:
    clock = Clock()
    local = planner(
        clock,
        candidate_score_ema_alpha=1.0,
        min_candidate_commit_sec=0.01,
        switch_score_margin=0.0,
        switch_confirm_count=1,
    )
    score, valid = score_map_for_heading(45.0, base=0.70, path_score=0.98)

    first = local.plan(score, valid, target_heading_error_rad=math.radians(45.0))
    clock.value += 1.0
    second = local.plan(score, valid, target_heading_error_rad=0.0)

    assert first.selected_candidate is not None
    assert first.selected_candidate.heading_deg == pytest.approx(45.0)
    assert second.selected_candidate is not None
    assert second.selected_candidate.heading_deg == pytest.approx(30.0)
    assert all(candidate.continuity_penalty == 0.0 for candidate in second.candidate_scores)


def test_unsafe_active_candidate_stops_before_confirmed_adjacent_switch() -> None:
    clock = Clock()
    local = planner(clock, unsafe_switch_confirm_count=3)
    score, valid = score_map_for_heading(0.0)
    initial = local.plan(score, valid, target_heading_error_rad=0.0)
    candidates = {
        candidate.heading_deg: candidate for candidate in initial.candidate_scores
    }
    adjacent = replace(candidates[15.0], final_score=1.0, hard_rejected=False)
    farther = replace(candidates[45.0], final_score=2.0, hard_rejected=False)

    clock.value += 1.0
    first = local._select_candidate(
        [adjacent, farther], now=clock.value, checkpoint_changed=False
    )
    second = local._select_candidate(
        [adjacent, farther], now=clock.value + 0.2, checkpoint_changed=False
    )
    third = local._select_candidate(
        [adjacent, farther], now=clock.value + 0.4, checkpoint_changed=False
    )

    assert first[0] is None and first[3] is True
    assert second[0] is None and second[3] is True
    assert third[0] is not None
    assert third[0].heading_deg == pytest.approx(15.0)
    assert third[1] is True
    assert third[3] is False


def test_one_frame_unsafe_candidate_recovers_without_switching() -> None:
    clock = Clock()
    local = planner(clock, unsafe_switch_confirm_count=3)
    score, valid = score_map_for_heading(0.0)
    initial = local.plan(score, valid, target_heading_error_rad=0.0)
    candidates = {
        candidate.heading_deg: candidate for candidate in initial.candidate_scores
    }
    adjacent = replace(candidates[15.0], final_score=1.0, hard_rejected=False)

    clock.value += 1.0
    pending = local._select_candidate(
        [adjacent], now=clock.value, checkpoint_changed=False
    )
    recovered = local._select_candidate(
        [candidates[0.0], adjacent],
        now=clock.value + 0.2,
        checkpoint_changed=False,
    )

    assert pending[0] is None and pending[3] is True
    assert recovered[0] is not None
    assert recovered[0].heading_deg == pytest.approx(0.0)
    assert recovered[1] is False
    assert recovered[3] is False


def test_unsafe_candidate_never_jumps_to_nonadjacent_safe_corridor() -> None:
    clock = Clock()
    local = planner(clock, unsafe_switch_confirm_count=2)
    score, valid = score_map_for_heading(0.0)
    initial = local.plan(score, valid, target_heading_error_rad=0.0)
    far = replace(initial.candidate_scores[0], final_score=2.0, hard_rejected=False)

    clock.value += 1.0
    outcomes = [
        local._select_candidate([far], now=clock.value + i * 0.2, checkpoint_changed=False)
        for i in range(4)
    ]

    assert all(selected is None for selected, _, _, _ in outcomes)
    assert all(stop_required is True for _, _, _, stop_required in outcomes)
    assert all(reason == "stop_no_adjacent_safe_candidate" for _, _, reason, _ in outcomes)


def test_holds_last_plan_for_one_transient_invalid_frame() -> None:
    clock = Clock()
    local = planner(clock)
    score, valid = score_map_for_heading(0.0)
    local.plan(score, valid, target_heading_error_rad=0.0)
    clock.value += 0.3
    invalid_score = np.full_like(score, 0.30)
    invalid_score[: int(invalid_score.shape[0] * 0.55), :] = 0.05
    held = local.plan(invalid_score, valid, target_heading_error_rad=0.0)

    assert held.using_held_plan is True
    assert held.path_valid is True
    assert held.image_path.reason == "MOTION_PRIMITIVE_HELD_TRANSIENT_INVALID"


def test_hold_expires_after_grace_period() -> None:
    clock = Clock()
    local = planner(clock, transient_invalid_grace_sec=0.4)
    score, valid = score_map_for_heading(0.0)
    local.plan(score, valid, target_heading_error_rad=0.0)
    clock.value += 0.6
    invalid_score = np.zeros_like(score)
    expired = local.plan(invalid_score, valid, target_heading_error_rad=0.0)

    assert expired.using_held_plan is False
    assert expired.path_valid is False


def test_all_near_field_unsafe_returns_blocked_plan() -> None:
    clock = Clock()
    local = planner(clock)
    score = np.zeros((120, 160), dtype=np.float32)
    valid = np.ones_like(score, dtype=bool)

    plan = local.plan(score, valid, target_heading_error_rad=0.0)

    assert plan.near_field_safe is False
    assert plan.path_valid is False
    assert plan.image_path.reason == "MOTION_PRIMITIVE_NEAR_FIELD_UNSAFE"


def test_candidate_score_ema_smooths_raw_score_changes() -> None:
    clock = Clock()
    local = planner(clock, candidate_score_ema_alpha=0.5)
    score, valid = score_map_for_heading(0.0, path_score=1.0)
    first = local.plan(score, valid, target_heading_error_rad=0.0)
    clock.value += 0.2
    lower, valid = score_map_for_heading(0.0, path_score=0.5)
    second = local.plan(lower, valid, target_heading_error_rad=0.0)

    first_score = first.selected_candidate.final_score
    second_score = second.selected_candidate.final_score
    assert second_score < first_score
    assert second_score > 0.0


def test_angle_wraparound_uses_short_direction() -> None:
    assert normalize_angle_deg(-179.0 - 179.0) == pytest.approx(2.0)
    assert normalize_angle_deg(179.0 - -179.0) == pytest.approx(-2.0)


def metric_planner(clock: Clock, **overrides) -> MotionPrimitivePlanner:
    values = {
        "geometry_mode": "metric_projected",
        "curvatures": (-0.3, -0.15, 0.0, 0.15, 0.3),
        "horizon_m": 2.0,
        "sample_interval_m": 0.1,
        "rover_width_m": 0.4,
        "safety_margin_m": 0.1,
        "min_projected_coverage_ratio": 0.3,
        "near_field_horizon_fraction": 0.35,
        "near_field_stop_threshold": 0.3,
        "path_score_threshold": 0.3,
        "candidate_score_ema_alpha": 1.0,
        "min_candidate_commit_sec": 0.8,
        "switch_score_margin": 0.08,
        "switch_confirm_count": 2,
        "unsafe_switch_confirm_count": 3,
        "max_candidate_switch_deg": 20.0,
        "transient_invalid_grace_sec": 0.8,
        "max_plan_age_sec": 1.5,
    } | overrides
    config = MotionPrimitivePlannerConfig(**values)
    return MotionPrimitivePlanner(config, monotonic=clock)


def test_metric_terminal_heading_deg_sign_convention() -> None:
    # Rover-frame positive curvature turns left; the controller convention
    # is positive-clockwise-right, so left must be a negative heading_deg.
    assert metric_terminal_heading_deg(curvature=0.3, horizon_m=2.0) < 0.0
    assert metric_terminal_heading_deg(curvature=-0.3, horizon_m=2.0) > 0.0
    assert metric_terminal_heading_deg(curvature=0.0, horizon_m=2.0) == pytest.approx(0.0)
    # Magnitude is exactly the arc's terminal heading in degrees.
    assert metric_terminal_heading_deg(curvature=0.5, horizon_m=2.0) == pytest.approx(
        -math.degrees(0.5 * 2.0)
    )


def test_metric_mode_without_calibration_hard_rejects_every_candidate_and_never_falls_back() -> None:
    clock = Clock()
    local = metric_planner(clock)
    score = np.full((480, 640), 0.9, dtype=np.float32)
    valid = np.ones((480, 640), dtype=bool)

    plan = local.plan(score, valid, target_heading_error_rad=0.0, calibration=None)

    assert plan.near_field_safe is False
    assert plan.path_valid is False
    assert plan.camera_projection_applied is False
    assert plan.image_path_metric_calibrated is False
    assert plan.calibration_id is None
    assert plan.candidate_scores
    assert all(candidate.hard_rejected for candidate in plan.candidate_scores)
    assert all(
        candidate.reject_reason == "NO_VALID_CALIBRATION" for candidate in plan.candidate_scores
    )
    # No silent fallback to the image-space heuristic curves: rejected
    # metric candidates carry no drawn points at all.
    assert all(candidate.points_uv.size == 0 for candidate in plan.candidate_scores)


def test_metric_mode_with_invalid_calibration_resolution_hard_rejects_with_reason() -> None:
    clock = Clock()
    local = metric_planner(clock)
    calibration = _make_calibration(image_width=640, image_height=480)
    score = np.full((240, 320), 0.9, dtype=np.float32)  # mismatched resolution
    valid = np.ones((240, 320), dtype=bool)

    plan = local.plan(score, valid, target_heading_error_rad=0.0, calibration=calibration)

    assert plan.camera_projection_applied is False
    assert all(
        candidate.reject_reason == "CALIBRATION_RESOLUTION_MISMATCH"
        for candidate in plan.candidate_scores
    )


def test_metric_mode_with_valid_calibration_projects_and_selects_a_candidate() -> None:
    clock = Clock()
    local = metric_planner(clock)
    calibration = _make_calibration(image_width=640, image_height=480)
    score = np.full((480, 640), 0.95, dtype=np.float32)
    valid = np.ones((480, 640), dtype=bool)

    plan = local.plan(score, valid, target_heading_error_rad=0.0, calibration=calibration)

    assert plan.camera_projection_applied is True
    assert plan.image_path_metric_calibrated is True
    assert plan.calibration_id == calibration.calibration_id
    assert plan.calibration_sha256_prefix == calibration.sha256_prefix
    assert plan.selected_candidate is not None
    assert plan.selected_candidate.footprint_pixel_count > 0
    assert plan.selected_candidate.heading_deg == pytest.approx(0.0, abs=1.0)


def test_metric_mode_obstacle_in_footprint_hard_rejects_that_candidate() -> None:
    clock = Clock()
    local = metric_planner(clock)
    calibration = _make_calibration(image_width=640, image_height=480)
    score = np.full((480, 640), 0.95, dtype=np.float32)
    valid = np.ones((480, 640), dtype=bool)
    # Block a wide vertical strip through the middle of the image, hitting
    # the near-field footprint of the straight (curvature=0.0) candidate.
    score[:, 280:360] = 0.0

    plan = local.plan(score, valid, target_heading_error_rad=0.0, calibration=calibration)

    straight = next(c for c in plan.candidate_scores if c.heading_deg == pytest.approx(0.0, abs=1.0))
    assert straight.hard_rejected is True
    assert straight.reject_reason == "NEAR_FIELD_UNSAFE"


def test_metric_mode_far_footprint_obstacle_hard_rejects_candidate() -> None:
    clock = Clock()
    local = metric_planner(clock)
    calibration = _make_calibration(image_width=640, image_height=480)
    score = np.full((480, 640), 0.95, dtype=np.float32)
    valid = np.ones((480, 640), dtype=bool)
    straight_index = next(
        index
        for index, heading in enumerate(local._metric_heading_by_index)
        if abs(heading) < 1.0
    )
    footprint = project_trajectory(
        local._metric_trajectories[straight_index],
        calibration,
        min_projected_coverage_ratio=local.config.min_projected_coverage_ratio,
        near_field_horizon_fraction=local.config.near_field_horizon_fraction,
    )
    far_mask = footprint.footprint_mask & ~footprint.near_field_mask
    score[far_mask] = 0.0

    plan = local.plan(score, valid, target_heading_error_rad=0.0, calibration=calibration)

    straight = plan.candidate_scores[straight_index]
    assert straight.near_field_low_percentile >= local.config.near_field_stop_threshold
    assert straight.hard_rejected is True
    assert straight.reject_reason == "FOOTPRINT_UNSAFE"


def test_metric_unsafe_switch_uses_metric_previous_heading() -> None:
    clock = Clock()
    local = metric_planner(
        clock,
        curvatures=(0.15, 0.0, -0.15),
        max_candidate_switch_deg=20.0,
        unsafe_switch_confirm_count=2,
    )
    calibration = _make_calibration(image_width=640, image_height=480)
    score = np.full((480, 640), 0.95, dtype=np.float32)
    valid = np.ones((480, 640), dtype=bool)
    initial = local.plan(score, valid, target_heading_error_rad=0.0, calibration=calibration)
    assert initial.selected_candidate is not None
    assert initial.selected_candidate.heading_deg == pytest.approx(0.0)
    adjacent = next(
        candidate
        for candidate in initial.candidate_scores
        if candidate.heading_deg > 0.0
    )

    clock.value += 1.0
    first = local._select_candidate(
        [adjacent], now=clock.value, checkpoint_changed=False
    )
    second = local._select_candidate(
        [adjacent], now=clock.value + 0.2, checkpoint_changed=False
    )

    assert first[0] is None and first[3] is True
    assert second[0] is not None
    assert second[0].heading_deg == pytest.approx(adjacent.heading_deg)
    assert second[3] is False


def test_metric_mode_stop_before_switch_and_bounded_switch_still_enforced() -> None:
    # _select_candidate is shared, untouched code between geometry modes;
    # this exercises it end-to-end through .plan() with real metric
    # candidates instead of only the heuristic-mode path.
    clock = Clock()
    local = metric_planner(clock, max_candidate_switch_deg=20.0, unsafe_switch_confirm_count=2)
    calibration = _make_calibration(image_width=640, image_height=480)
    score = np.full((480, 640), 0.95, dtype=np.float32)
    valid = np.ones((480, 640), dtype=bool)

    first = local.plan(score, valid, target_heading_error_rad=0.0, calibration=calibration)
    assert first.selected_candidate is not None
    assert first.selected_candidate.heading_deg == pytest.approx(0.0, abs=1.0)

    # Make the currently-selected (straight) candidate unsafe; the far
    # (+0.3 curvature) candidate now scores best but is not adjacent to the
    # committed candidate within max_candidate_switch_deg, so the planner
    # must stop rather than jump straight to it.
    clock.value += 1.0
    unsafe_score = score.copy()
    unsafe_score[:, 280:360] = 0.0
    second = local.plan(unsafe_score, valid, target_heading_error_rad=0.0, calibration=calibration)
    assert second.switch_stop_required is True
    assert second.selected_candidate is None


def side_blocked_scene(
    *,
    shape: tuple[int, int] = (120, 160),
    left_score: float = 0.05,
    right_score: float = 0.95,
    center_score: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """A scene with a wall on the left, an obstacle ahead, and open ground on
    the right -- the scenario this feature exists to recover from."""

    height, width = shape
    score = np.full(shape, center_score, dtype=np.float32)
    left_end = int(width * 0.30)
    right_start = width - int(width * 0.30)
    score[:, :left_end] = left_score
    score[:, right_start:] = right_score
    valid = np.ones(shape, dtype=bool)
    return score, valid


def test_repro_center_and_left_blocked_right_open_hard_rejects_every_candidate() -> None:
    # Documents the reported bug before the fix: an obstacle directly ahead
    # plus a wall on the left contaminates every candidate's shared
    # near-field start region, including the widest (+/-30 deg) candidates,
    # even though the right side of the frame is genuinely open.
    clock = Clock()
    local = planner(
        clock,
        candidate_headings_deg=(-30.0, -20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0, 30.0),
        maximum_visual_heading_deg=30.0,
        near_field_stop_threshold=0.65,
        side_sector_enabled=True,
    )
    score, valid = side_blocked_scene()

    plan = local.plan(score, valid, target_heading_error_rad=0.0)

    assert len(plan.candidate_scores) == 9
    assert all(candidate.hard_rejected for candidate in plan.candidate_scores)
    assert all(candidate.reject_reason == "NEAR_FIELD_UNSAFE" for candidate in plan.candidate_scores)
    assert plan.selected_candidate is None
    assert plan.near_field_safe is False
    assert plan.switch_reason == "all_candidates_hard_rejected"
    assert plan.switch_stop_required is True
    # The gap this feature closes: independent side-sector evidence already
    # knows the right is clearly the safer side, even though every candidate
    # curve above was rejected identically.
    assert plan.side_sector is not None
    assert plan.side_sector.chosen == "RIGHT"


def test_side_sector_prefers_right_when_left_is_wall_and_right_is_open() -> None:
    config = MotionPrimitivePlannerConfig()
    score, valid = side_blocked_scene(left_score=0.05, right_score=0.95, center_score=0.5)

    decision = evaluate_side_sectors(score, valid, config)

    assert decision.chosen == "RIGHT"
    assert decision.status == "RIGHT_CLEAR"
    assert decision.left.viable is False
    assert decision.right.viable is True


def test_side_sector_prefers_left_when_right_is_wall_and_left_is_open() -> None:
    config = MotionPrimitivePlannerConfig()
    score, valid = side_blocked_scene(left_score=0.95, right_score=0.05, center_score=0.5)

    decision = evaluate_side_sectors(score, valid, config)

    assert decision.chosen == "LEFT"
    assert decision.status == "LEFT_CLEAR"
    assert decision.left.viable is True
    assert decision.right.viable is False


def test_side_sector_returns_ambiguous_when_margin_too_small() -> None:
    config = MotionPrimitivePlannerConfig(side_sector_margin=0.12)
    # Both sides comfortably viable, but nearly identical -- not a clear pick.
    score, valid = side_blocked_scene(left_score=0.75, right_score=0.80, center_score=0.5)

    decision = evaluate_side_sectors(score, valid, config)

    assert decision.chosen is None
    assert decision.status == "AMBIGUOUS"
    assert decision.left.viable is True
    assert decision.right.viable is True


def test_side_sector_returns_both_unsafe_when_neither_side_is_viable() -> None:
    config = MotionPrimitivePlannerConfig()
    score, valid = side_blocked_scene(left_score=0.05, right_score=0.10, center_score=0.05)

    decision = evaluate_side_sectors(score, valid, config)

    assert decision.chosen is None
    assert decision.status == "BOTH_UNSAFE"
    assert decision.left.viable is False
    assert decision.right.viable is False


def test_side_sector_excludes_sky_and_bumper_rows() -> None:
    config = MotionPrimitivePlannerConfig()
    height, width = 120, 160
    score_a, valid = side_blocked_scene(shape=(height, width), left_score=0.2, right_score=0.9, center_score=0.5)
    score_b = score_a.copy()
    sky_rows = int(round(height * config.side_sector_top_ratio))
    bumper_start = int(round(height * (1.0 - config.side_sector_bottom_exclude_ratio)))
    # Change only the excluded sky/bumper rows -- the decision must not move.
    score_b[:sky_rows, :] = 0.0
    score_b[bumper_start:, :] = 0.0

    decision_a = evaluate_side_sectors(score_a, valid, config)
    decision_b = evaluate_side_sectors(score_b, valid, config)

    assert decision_a.left.mean == pytest.approx(decision_b.left.mean)
    assert decision_a.right.mean == pytest.approx(decision_b.right.mean)
    assert decision_a.chosen == decision_b.chosen == "RIGHT"


def test_plan_to_status_exposes_side_sector_fields() -> None:
    clock = Clock()
    local = planner(
        clock,
        candidate_headings_deg=(-30.0, -20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0, 30.0),
        maximum_visual_heading_deg=30.0,
        near_field_stop_threshold=0.65,
        side_sector_enabled=True,
    )
    score, valid = side_blocked_scene()

    plan = local.plan(score, valid, target_heading_error_rad=0.0)
    status = plan.to_status()

    side_sector = status["side_sector"]
    assert side_sector is not None
    assert set(side_sector.keys()) >= {"left", "right", "chosen", "status", "margin", "reason"}
    assert side_sector["chosen"] == "RIGHT"
    assert side_sector["right"]["viable"] is True
