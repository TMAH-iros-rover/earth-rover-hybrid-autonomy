from __future__ import annotations

import pytest

from earth_rover.autonomy.mission1_controller import (
    Mission1Autonomy,
    Mission1ControlConfig,
    mission1_command_to_sdk_command,
    mission1_to_sdk_angular,
)
from earth_rover.core.types import ControlCommand


class FakeSdk:
    def __init__(self, active: bool = False) -> None:
        self.active = active
        self.commands = []
        self.reports = 0

    def get_mission_status(self):
        return {"mission_active": self.active}

    def send_control(self, command):
        self.commands.append(command)
        return True

    def report_checkpoint_details(self):
        self.reports += 1
        return {"next_checkpoint_sequence": 2}


class FailingControlSdk(FakeSdk):
    def send_control(self, command):
        self.commands.append(command)
        raise RuntimeError("control 503")


class FakeSam:
    def __init__(self, payload):
        self.payload = payload

    def get(self):
        return dict(self.payload)


class Clock:
    def __init__(self, value=100.0):
        self.value = value

    def __call__(self):
        return self.value


def valid_sam(**overrides):
    payload = {
        "ready": True,
        "state": "CLEAR",
        "published_timestamp": 100.0,
        "path_valid": True,
        "path_reason": "GPS_HEADING_ALIGNED_TRAVERSABLE_PATH",
        "path_mean_score": 0.80,
        "local_path_selected_heading_deg": 20.0,
        "navigation": {
            "target_sequence": 1,
            "gps_valid": True,
            "heading_valid": True,
            "reached": False,
            "finished": False,
        },
    }
    payload.update(overrides)
    return payload


def controller(sdk, sam, clock, live=True):
    config = {
        "control": {
            "linear_min": 0.0,
            "linear_max": 0.06,
            "angular_min": -0.22,
            "angular_max": 0.22,
            "command_smoothing_alpha": 0.0,
            "max_linear_delta_per_sec": 10.0,
            "max_angular_delta_per_sec": 10.0,
        }
    }
    return Mission1Autonomy(
        sdk,
        sam,
        Mission1ControlConfig(),
        config,
        live_control_enabled=live,
        clock=clock,
        monotonic=clock,
    )


def test_waits_for_dashboard_start_mission_without_command():
    sdk = FakeSdk(active=False)
    clock = Clock()
    autonomy = controller(sdk, FakeSam(valid_sam()), clock)

    status = autonomy.tick()

    assert status["state"] == "WAITING_FOR_START_MISSION"
    assert sdk.commands == []


def test_waits_for_rtm_bridge_without_sending_a_control_request():
    sdk = FakeSdk(active=True)
    sdk.get_mission_status = lambda: {
        "mission_active": True,
        "control_bridge_ready": False,
    }
    clock = Clock()
    sam = FakeSam(valid_sam())
    autonomy = controller(sdk, sam, clock)

    status = autonomy.tick()

    assert status["state"] == "WAITING_FOR_CONTROL_BRIDGE"
    assert status["command_transmitted"] is False
    assert sdk.commands == []


def test_active_mission_tracks_valid_local_path():
    sdk = FakeSdk(active=True)
    clock = Clock()
    autonomy = controller(sdk, FakeSam(valid_sam()), clock)
    clock.value += 0.2

    status = autonomy.tick()

    assert status["state"] == "DRIVING"
    assert status["command_transmitted"] is True
    assert 0.0 < sdk.commands[-1].linear <= 0.06
    assert status["angular"] == pytest.approx(0.139626, rel=1e-4)
    assert sdk.commands[-1].angular == pytest.approx(-0.139626, rel=1e-4)


def test_positive_internal_angular_maps_to_negative_sdk_angular():
    assert mission1_to_sdk_angular(0.25) == pytest.approx(-0.25)


def test_negative_internal_angular_maps_to_positive_sdk_angular():
    assert mission1_to_sdk_angular(-0.25) == pytest.approx(0.25)


def test_zero_internal_angular_remains_zero_for_sdk():
    assert mission1_to_sdk_angular(0.0) == pytest.approx(0.0)


def test_mission1_command_to_sdk_command_preserves_linear_lamp_and_clamp_range():
    sdk_command = mission1_command_to_sdk_command(
        ControlCommand(0.1, 0.22, lamp=1, mode="TEST")
    )

    assert sdk_command.linear == pytest.approx(0.1)
    assert sdk_command.angular == pytest.approx(-0.22)
    assert -1.0 <= sdk_command.angular <= 1.0
    assert sdk_command.lamp == 1
    assert sdk_command.mode == "TEST"


def test_controller_positive_local_path_sends_negative_sdk_angular_for_physical_right():
    sdk = FakeSdk(active=True)
    clock = Clock()
    autonomy = controller(
        sdk,
        FakeSam(valid_sam(local_path_selected_heading_deg=20.0)),
        clock,
    )
    clock.value += 0.2

    status = autonomy.tick()

    assert status["state"] == "DRIVING"
    assert status["angular"] > 0.0
    assert sdk.commands[-1].angular < 0.0
    assert status["controller_debug"]["angular_convention"] == "mission1_internal_positive_clockwise_right"
    assert status["controller_debug"]["filtered_angular"] > 0.0
    assert status["controller_debug"]["internal_angular"] > 0.0
    assert status["controller_debug"]["sdk_angular"] < 0.0
    assert status["controller_debug"]["internal_angular_convention"] == "positive_right"
    assert status["controller_debug"]["sdk_angular_convention"] == "negative_right_positive_left"


def test_controller_negative_local_path_sends_positive_sdk_angular_for_physical_left():
    sdk = FakeSdk(active=True)
    clock = Clock()
    autonomy = controller(
        sdk,
        FakeSam(valid_sam(local_path_selected_heading_deg=-20.0)),
        clock,
    )
    clock.value += 0.2

    status = autonomy.tick()

    assert status["state"] == "DRIVING"
    assert status["angular"] < 0.0
    assert sdk.commands[-1].angular > 0.0
    assert status["controller_debug"]["filtered_angular"] < 0.0
    assert status["controller_debug"]["sdk_angular"] > 0.0


def test_control_send_failure_enters_cooldown_without_command_spam():
    sdk = FailingControlSdk(active=True)
    clock = Clock()
    autonomy = controller(sdk, FakeSam(valid_sam()), clock)
    clock.value += 0.2

    first = autonomy.tick()
    clock.value += 0.2
    second = autonomy.tick()

    assert first["state"] == "ERROR_STOP"
    assert "control 503" in first["reason"]
    assert second["state"] == "WAITING_FOR_CONTROL_BRIDGE"
    assert len(sdk.commands) == 1


@pytest.mark.parametrize(
    "change, reason",
    [
        ({"published_timestamp": 98.0}, "stale"),
        ({"path_valid": False}, "local path invalid"),
        ({"state": "STALE_FRAME"}, "SAM-TP state"),
        ({"path_mean_score": 0.3}, "local path score"),
    ],
)
def test_invalid_perception_sends_explicit_stop(change, reason):
    sdk = FakeSdk(active=True)
    clock = Clock()
    autonomy = controller(sdk, FakeSam(valid_sam(**change)), clock)

    status = autonomy.tick()

    assert status["state"] == "SAFETY_STOP"
    assert reason in status["reason"]
    assert sdk.commands[-1].linear == 0.0
    assert sdk.commands[-1].angular == 0.0


def test_reached_checkpoint_stops_reports_once_and_waits_for_route_advance():
    sdk = FakeSdk(active=True)
    clock = Clock()
    navigation = valid_sam()["navigation"] | {"reached": True}
    autonomy = controller(
        sdk,
        FakeSam(valid_sam(navigation=navigation)),
        clock,
    )

    first = autonomy.tick()
    clock.value += 0.2
    second = autonomy.tick()

    assert first["state"] == "CHECKPOINT_REPORTED"
    assert second["state"] == "CHECKPOINT_WAIT"
    assert sdk.reports == 1
    assert all(command.linear == command.angular == 0.0 for command in sdk.commands)


def test_reached_checkpoint_reports_even_when_forward_path_is_invalid():
    sdk = FakeSdk(active=True)
    clock = Clock()
    navigation = valid_sam()["navigation"] | {"reached": True}
    autonomy = controller(
        sdk,
        FakeSam(
            valid_sam(
                navigation=navigation,
                path_valid=False,
                path_reason="NO_CONNECTED_TRAVERSABLE_PATH",
                path_mean_score=None,
                local_path_selected_heading_deg=None,
            )
        ),
        clock,
    )

    status = autonomy.tick()

    assert status["state"] == "CHECKPOINT_REPORTED"
    assert sdk.reports == 1
    assert sdk.commands[-1].linear == sdk.commands[-1].angular == 0.0


def test_goal_behind_rotates_instead_of_stopping_on_invalid_forward_path():
    sdk = FakeSdk(active=True)
    clock = Clock()
    navigation = valid_sam()["navigation"] | {"heading_error_deg": -174.0}
    autonomy = controller(
        sdk,
        FakeSam(
            valid_sam(
                navigation=navigation,
                path_valid=False,
                path_reason="NO_CONNECTED_TRAVERSABLE_PATH",
                path_mean_score=None,
                local_path_selected_heading_deg=None,
            )
        ),
        clock,
    )
    clock.value += 0.2

    status = autonomy.tick()

    assert status["state"] == "ROTATING_TO_GOAL"
    assert status["command_transmitted"] is True
    assert sdk.commands[-1].linear == 0.0
    assert status["angular"] < 0.0
    assert sdk.commands[-1].angular > 0.0
    assert status["target_sequence"] == 1


def test_invalid_forward_path_still_stops_when_goal_is_not_behind():
    sdk = FakeSdk(active=True)
    clock = Clock()
    navigation = valid_sam()["navigation"] | {"heading_error_deg": 35.0}
    autonomy = controller(
        sdk,
        FakeSam(
            valid_sam(
                navigation=navigation,
                path_valid=False,
                path_reason="NO_CONNECTED_TRAVERSABLE_PATH",
                path_mean_score=None,
                local_path_selected_heading_deg=None,
            )
        ),
        clock,
    )

    status = autonomy.tick()

    assert status["state"] == "SAFETY_STOP"
    assert sdk.commands[-1].linear == 0.0
    assert sdk.commands[-1].angular == 0.0


def test_transient_invalid_path_holds_last_valid_command_before_stopping():
    sdk = FakeSdk(active=True)
    clock = Clock()
    good = valid_sam(local_path_selected_heading_deg=10.0, path_mean_score=0.8)
    bad = valid_sam(
        path_valid=False,
        path_reason="NO_CONNECTED_TRAVERSABLE_PATH",
        path_mean_score=None,
        local_path_selected_heading_deg=None,
    )
    sam = FakeSam(good)
    autonomy = controller(sdk, sam, clock)
    autonomy.settings = Mission1ControlConfig(
        path_invalid_grace_ticks=2,
        held_path_linear_scale=0.5,
    )
    clock.value += 0.2
    driving = autonomy.tick()
    sam.payload = bad
    clock.value += 0.2
    held = autonomy.tick()
    clock.value += 0.2
    held_again = autonomy.tick()
    clock.value += 0.2
    stopped = autonomy.tick()

    assert driving["state"] == "DRIVING"
    assert held["state"] == "PATH_HOLD"
    assert held_again["state"] == "PATH_HOLD"
    assert stopped["state"] == "SAFETY_STOP"
    assert sdk.commands[-2].linear > 0.0
    assert sdk.commands[-1].linear == 0.0


def test_partial_relaxed_path_reduces_forward_speed():
    sdk = FakeSdk(active=True)
    clock = Clock()
    autonomy = controller(
        sdk,
        FakeSam(
            valid_sam(
                path_reason="PARTIAL_RELAXED_GPS_HEADING_ALIGNED_TRAVERSABLE_PATH",
                path_mean_score=0.9,
                local_path_selected_heading_deg=0.0,
            )
        ),
        clock,
    )
    clock.value += 0.2

    status = autonomy.tick()

    assert status["state"] == "DRIVING"
    assert 0.0 < sdk.commands[-1].linear < 0.12


def test_dry_run_never_transmits_even_during_active_mission():
    sdk = FakeSdk(active=True)
    clock = Clock()
    autonomy = controller(sdk, FakeSam(valid_sam()), clock, live=False)

    autonomy.tick()
    clock.value += 0.2
    status = autonomy.tick()

    assert status["state"] == "DRY_RUN"
    assert sdk.commands == []
    assert status["command_transmitted"] is False
    assert status["linear"] > 0.0
    assert status["angular"] == pytest.approx(0.139626, rel=1e-4)


def test_dry_run_reached_checkpoint_does_not_report():
    sdk = FakeSdk(active=True)
    clock = Clock()
    navigation = valid_sam()["navigation"] | {"reached": True}
    autonomy = controller(
        sdk,
        FakeSam(valid_sam(navigation=navigation)),
        clock,
        live=False,
    )

    status = autonomy.tick()

    assert status["state"] == "DRY_RUN_CHECKPOINT_REACHED"
    assert sdk.reports == 0
    assert sdk.commands == []


def test_operator_stop_latches_zero_until_resume():
    sdk = FakeSdk(active=True)
    clock = Clock()
    autonomy = controller(sdk, FakeSam(valid_sam()), clock)

    autonomy.tick()
    stopped = autonomy.operator_stop()
    clock.value += 0.2
    still_stopped = autonomy.tick()
    autonomy.operator_resume()
    clock.value += 0.2
    resumed = autonomy.tick()

    assert stopped["state"] == "OPERATOR_STOP"
    assert still_stopped["state"] == "OPERATOR_STOP"
    assert sdk.commands[-2].linear == 0.0
    assert resumed["state"] == "DRIVING"
