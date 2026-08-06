from __future__ import annotations

import math

import numpy as np
import pytest

from earth_rover.planning.motion_primitive_planner import (
    MotionPrimitivePlanner,
    MotionPrimitivePlannerConfig,
    adaptive_kmeans_paths,
    genie_candidate_points,
    image_direction_from_x_offset,
    merge_close_path_clusters,
    normalize_angle_deg,
    path_heading_from_endpoint_deg,
    primitive_curve_points,
    selected_candidate_endpoint_x_offset_px,
)


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
    assert switched.selected_candidate.heading_deg == pytest.approx(30.0)
    assert switched.candidate_switched is True


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


# --------------------------------------------------------------------------
# genie_cluster mode: GeNIE Algorithm 1 (sample -> top-K -> cluster -> merge
# -> angular selection) ported to image space.
# --------------------------------------------------------------------------


def genie_planner(clock: Clock, **overrides) -> MotionPrimitivePlanner:
    values = {
        "mode": "genie_cluster",
        "min_candidate_commit_sec": 0.1,
        "switch_confirm_count": 2,
        "genie_switch_heading_deadband_deg": 6.0,
        "transient_invalid_grace_sec": 0.8,
        "max_plan_age_sec": 1.5,
    } | overrides
    config = MotionPrimitivePlannerConfig(**values)
    return MotionPrimitivePlanner(config, monotonic=clock)


def test_path_heading_from_endpoint_deg_inverts_genie_candidate_points() -> None:
    shape = (120, 160)
    for heading in (-40.0, -10.0, 0.0, 17.5, 45.0):
        points = genie_candidate_points(
            shape, heading, maximum_visual_heading_deg=55.0, n_waypoints=10
        )
        recovered = path_heading_from_endpoint_deg(
            points, width=shape[1], maximum_visual_heading_deg=55.0
        )
        assert recovered == pytest.approx(heading, abs=1e-6)


def test_adaptive_kmeans_paths_separates_two_distinct_directions() -> None:
    shape = (120, 160)
    left_paths = np.stack(
        [genie_candidate_points(shape, -30.0, maximum_visual_heading_deg=55.0, n_waypoints=8)] * 3
    )
    right_paths = np.stack(
        [genie_candidate_points(shape, 30.0, maximum_visual_heading_deg=55.0, n_waypoints=8)] * 3
    )
    paths = np.concatenate([left_paths, right_paths], axis=0)

    labels, centers = adaptive_kmeans_paths(paths, k_max=4, seed=0)

    assert len(np.unique(labels)) == 2
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]
    assert len(centers) == 2


def test_merge_close_path_clusters_unions_within_threshold() -> None:
    shape = (120, 160)
    near_zero = genie_candidate_points(shape, 0.0, maximum_visual_heading_deg=55.0, n_waypoints=8)
    near_two = genie_candidate_points(shape, 2.0, maximum_visual_heading_deg=55.0, n_waypoints=8)
    far_right = genie_candidate_points(shape, 45.0, maximum_visual_heading_deg=55.0, n_waypoints=8)
    centers = np.stack([near_zero, near_two, far_right])

    merged = merge_close_path_clusters(centers, threshold_px=5.0)

    assert len(merged) == 2


def test_genie_cluster_selects_straight_when_goal_and_score_prefer_straight() -> None:
    clock = Clock()
    local = genie_planner(clock)
    score, valid = score_map_for_heading(0.0)

    plan = local.plan(score, valid, target_heading_error_rad=0.0)

    assert plan.path_valid is True
    assert plan.selected_candidate is not None
    assert plan.selected_candidate.heading_deg == pytest.approx(0.0, abs=3.0)
    assert plan.image_path.reason == "MOTION_PRIMITIVE_SELECTED"


def test_genie_cluster_switches_toward_goal_only_after_confirmation() -> None:
    clock = Clock()
    local = genie_planner(clock)
    straight, valid = score_map_for_heading(0.0)
    local.plan(straight, valid, target_heading_error_rad=0.0)

    clock.value += 1.0
    right, valid = score_map_for_heading(30.0)
    pending = local.plan(right, valid, target_heading_error_rad=math.radians(30.0))
    assert pending.selected_candidate.heading_deg == pytest.approx(0.0, abs=3.0)
    assert pending.switch_reason == "switch_pending"

    clock.value += 0.5
    switched = local.plan(right, valid, target_heading_error_rad=math.radians(30.0))
    assert switched.candidate_switched is True
    assert switched.selected_candidate.heading_deg > 15.0


def test_genie_cluster_all_near_field_unsafe_returns_blocked_plan() -> None:
    clock = Clock()
    local = genie_planner(clock)
    score = np.zeros((120, 160), dtype=np.float32)
    valid = np.ones_like(score, dtype=bool)

    plan = local.plan(score, valid, target_heading_error_rad=0.0)

    assert plan.near_field_safe is False
    assert plan.path_valid is False
    assert plan.image_path.reason == "MOTION_PRIMITIVE_NEAR_FIELD_UNSAFE"


def test_genie_cluster_config_rejects_invalid_tuning() -> None:
    with pytest.raises(ValueError, match="genie_top_k"):
        MotionPrimitivePlannerConfig(mode="genie_cluster", genie_top_k=0).validate()
    with pytest.raises(ValueError, match="genie_n_candidates"):
        MotionPrimitivePlannerConfig(mode="genie_cluster", genie_n_candidates=1).validate()
