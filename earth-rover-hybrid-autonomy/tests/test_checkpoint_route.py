from __future__ import annotations

import math

import pytest

from earth_rover.navigation.checkpoint_route import CheckpointRoutePlanner


CHECKPOINTS = [
    {"sequence": 3, "latitude": 37.0, "longitude": 127.0},
    {"sequence": 1, "latitude": 37.0001, "longitude": 127.0},
    {"sequence": 2, "latitude": 37.0001, "longitude": 127.0001},
]


def test_builds_current_to_remaining_checkpoint_polyline_in_sequence_order() -> None:
    planner = CheckpointRoutePlanner(CHECKPOINTS, switch_radius_m=2.0)

    state = planner.update(37.00005, 127.0, heading_deg=0.0)

    assert state.route_polyline == (
        (37.00005, 127.0),
        (37.0001, 127.0),
        (37.0001, 127.0001),
        (37.0, 127.0),
    )
    assert state.target_sequence == 1
    assert state.reason == "TRACKING"
    assert state.gps_valid is True
    assert state.heading_valid is True


def test_computes_wrapped_heading_error_for_current_target() -> None:
    checkpoints = [{"sequence": 1, "latitude": 1.0, "longitude": 0.0}]
    planner = CheckpointRoutePlanner(checkpoints, switch_radius_m=1.0)

    state = planner.update(0.0, 0.0, heading_deg=350.0)

    assert state.target_bearing_deg == pytest.approx(0.0, abs=0.1)
    assert state.heading_error_rad == pytest.approx(math.radians(10.0), abs=1e-6)


def test_heading_error_filter_deadband_and_large_change_reset() -> None:
    checkpoints = [{"sequence": 1, "latitude": 1.0, "longitude": 0.0}]
    planner = CheckpointRoutePlanner(
        checkpoints,
        switch_radius_m=1.0,
        heading_filter_alpha=0.5,
        target_heading_deadband_deg=6.0,
        large_heading_change_deg=35.0,
    )

    first = planner.update(0.0, 0.0, heading_deg=0.0)
    deadband = planner.update(0.0, 0.0, heading_deg=4.0)
    large = planner.update(0.0, 0.0, heading_deg=80.0)

    assert first.heading_error_rad == pytest.approx(0.0)
    assert deadband.heading_error_rad == pytest.approx(0.0)
    assert large.heading_error_rad == pytest.approx(math.radians(-80.0), abs=1e-6)


def test_reached_checkpoint_waits_for_successful_report_before_advancing() -> None:
    planner = CheckpointRoutePlanner(CHECKPOINTS, switch_radius_m=2.0)

    reached = planner.update(37.0001, 127.0, heading_deg=90.0)
    assert reached.reached is True
    assert reached.target_sequence == 1
    assert reached.reason == "CHECKPOINT_REACHED_PENDING_REPORT"
    assert reached.heading_error_rad is None
    assert reached.current_heading_deg == 90.0

    still_waiting = planner.update(37.0001, 127.0, heading_deg=90.0)
    assert still_waiting.target_sequence == 1

    planner.mark_current_reported()
    next_target = planner.update(37.0001, 127.0, heading_deg=90.0)
    assert next_target.target_sequence == 2


def test_invalid_gps_and_heading_are_explicit_safe_states() -> None:
    planner = CheckpointRoutePlanner(CHECKPOINTS, switch_radius_m=2.0)

    bad_gps = planner.update(None, 127.0, heading_deg=0.0)
    assert bad_gps.gps_valid is False
    assert bad_gps.heading_error_rad is None
    assert bad_gps.reason == "INVALID_GPS"

    bad_heading = planner.update(37.00005, 127.0, heading_deg=float("nan"))
    assert bad_heading.gps_valid is True
    assert bad_heading.heading_valid is False
    assert bad_heading.target_bearing_deg is not None
    assert bad_heading.heading_error_rad is None
    assert bad_heading.reason == "INVALID_HEADING"


def test_route_finishes_only_after_last_checkpoint_is_reported() -> None:
    planner = CheckpointRoutePlanner(
        CHECKPOINTS,
        switch_radius_m=2.0,
        latest_scanned_checkpoint=2,
    )
    reached = planner.update(37.0, 127.0, heading_deg=0.0)
    assert reached.target_sequence == 3
    assert reached.reached is True

    planner.mark_current_reported()
    complete = planner.update(37.0, 127.0, heading_deg=0.0)
    assert complete.finished is True
    assert complete.reason == "MISSION_COMPLETE"
    assert complete.target_checkpoint is None


@pytest.mark.parametrize("radius", [0.0, -1.0, float("nan")])
def test_switch_radius_must_be_positive(radius: float) -> None:
    with pytest.raises(ValueError, match="switch_radius_m"):
        CheckpointRoutePlanner(CHECKPOINTS, switch_radius_m=radius)
