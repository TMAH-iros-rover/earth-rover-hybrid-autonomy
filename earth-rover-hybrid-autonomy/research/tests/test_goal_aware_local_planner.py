from __future__ import annotations

import math

from earth_rover.core.types import LocalTraversability
from earth_rover.perception.traversability_adapter import TraversabilityAdapter
from earth_rover.planning.goal_aware_local_planner import GoalAwareLocalPlanner

import numpy as np


def config() -> dict:
    return {
        "traversability_adapter": {
            "roi_top": 0.25,
            "near_top": 0.70,
            "sector_boundaries": [0.0, 0.33, 0.67, 1.0],
            "near_obstacle_stop_ratio": 0.45,
        },
        "goal_aware_planner": {
            "candidate_heading_offsets_deg": {"LEFT": 35, "CENTER": 0, "RIGHT": -35},
            "traversability_weight": 1.0,
            "gps_alignment_weight": 0.8,
            "obstacle_weight": 1.2,
            "uncertainty_weight": 0.35,
            "direction_change_penalty": 0.02,
            "hysteresis_margin": 0.10,
            "hard_obstacle_ratio": 0.55,
            "minimum_safe_score": 0.08,
        },
    }


def result(mask: np.ndarray, confidence: float = 1.0) -> LocalTraversability:
    return TraversabilityAdapter(config()).adapt(
        mask,
        np.full(mask.shape, confidence, dtype=np.float32),
    )


def test_gps_left_is_preferred_when_all_directions_are_safe() -> None:
    plan = GoalAwareLocalPlanner(config()).select(
        result(np.ones((60, 90), dtype=np.uint8)),
        heading_error_rad=math.radians(35),
    )

    assert plan.selected_direction == "LEFT"
    assert plan.stop_requested is False


def test_unsafe_right_cannot_override_safe_camera_choice() -> None:
    mask = np.ones((60, 90), dtype=np.uint8)
    mask[:, 60:] = 3

    plan = GoalAwareLocalPlanner(config()).select(
        result(mask),
        heading_error_rad=math.radians(-35),
    )

    assert plan.selected_direction != "RIGHT"
    assert plan.candidate_scores["RIGHT"] == -1.0


def test_left_obstacle_selects_center_or_right_without_global_stop() -> None:
    mask = np.ones((60, 90), dtype=np.uint8)
    mask[:, :30] = 3

    plan = GoalAwareLocalPlanner(config()).select(result(mask), heading_error_rad=0.0)

    assert plan.selected_direction in {"CENTER", "RIGHT"}
    assert plan.stop_requested is False


def test_low_confidence_reduces_speed_or_stops() -> None:
    planner = GoalAwareLocalPlanner(config())
    high = planner.select(result(np.ones((60, 90), dtype=np.uint8), 1.0), 0.0)
    low = planner.select(result(np.ones((60, 90), dtype=np.uint8), 0.5), 0.0)

    assert low.stop_requested or low.speed_target < high.speed_target


def test_hysteresis_prevents_small_direction_oscillation() -> None:
    planner = GoalAwareLocalPlanner(config())
    safe = result(np.ones((60, 90), dtype=np.uint8))
    first = planner.select(safe, math.radians(35))
    second = planner.select(safe, math.radians(-2), previous_direction=first.selected_direction)

    assert first.selected_direction == "LEFT"
    assert second.selected_direction == "LEFT"
    assert "HYSTERESIS_HOLD" in second.reason
