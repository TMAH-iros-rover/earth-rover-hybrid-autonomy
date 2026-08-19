from earth_rover.navigation.waypoint_manager import WaypointManager


CHECKPOINTS = [
    {"sequence": 2, "latitude": "37.0002", "longitude": "127.0"},
    {"sequence": 1, "latitude": "37.0001", "longitude": "127.0"},
]


def test_starts_after_latest_scanned_checkpoint_and_sorts_by_sequence():
    manager = WaypointManager(CHECKPOINTS, switch_radius_m=15.0, latest_scanned_checkpoint=1)

    assert manager.current_target()["sequence"] == 2


def test_reached_does_not_advance_until_report_success():
    manager = WaypointManager(CHECKPOINTS, switch_radius_m=15.0, latest_scanned_checkpoint=0)

    state = manager.update(37.0001, 127.0)

    assert state["reached"] is True
    assert manager.current_target()["sequence"] == 1

    manager.mark_current_reported()

    assert manager.current_target()["sequence"] == 2


def test_reached_latches_when_gps_drifts_back_outside_radius():
    manager = WaypointManager(CHECKPOINTS, switch_radius_m=15.0, latest_scanned_checkpoint=0)

    assert manager.update(37.0001, 127.0)["reached"] is True
    drifted = manager.update(37.001, 127.0)

    assert drifted["reached"] is True
    assert manager.current_target()["sequence"] == 1


def test_finishes_after_last_reported():
    manager = WaypointManager(CHECKPOINTS, switch_radius_m=15.0, latest_scanned_checkpoint=1)

    assert manager.update(37.0002, 127.0)["reached"] is True
    manager.mark_current_reported()

    assert manager.current_target() is None
