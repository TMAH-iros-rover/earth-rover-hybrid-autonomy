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
        "frame_index": 1,
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


def blocked_sam(candidate_scores=None, **overrides):
    planner = {
        "near_field_safe": False,
        "near_field_score": 0.02,
        "candidate_scores": (
            [
                {"heading_deg": -30.0, "near_field": 0.05},
                {"heading_deg": -10.0, "near_field": 0.02},
                {"heading_deg": 0.0, "near_field": 0.01},
                {"heading_deg": 10.0, "near_field": 0.03},
                {"heading_deg": 30.0, "near_field": 0.20},
            ]
            if candidate_scores is None
            else candidate_scores
        ),
    }
    return valid_sam(planner=planner, **overrides)


def controller(sdk, sam, clock, live=True, settings=None):
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
        settings or Mission1ControlConfig(),
        config,
        live_control_enabled=live,
        clock=clock,
        monotonic=clock,
    )


def test_target_sequence_requires_distinct_confirmed_sam_frames():
    sdk = FakeSdk(active=True)
    clock = Clock()
    payload = valid_sam(frame_index=10)
    sam = FakeSam(payload)
    autonomy = controller(
        sdk,
        sam,
        clock,
        settings=Mission1ControlConfig(target_sequence_confirm_frames=3),
    )

    first = autonomy.tick()
    repeated = autonomy.tick()
    sam.payload["frame_index"] = 11
    second = autonomy.tick()
    sam.payload["frame_index"] = 12
    confirmed = autonomy.tick()

    assert first["state"] == "STARTUP_ALIGN"
    assert repeated["state"] == "STARTUP_ALIGN"
    assert "(1/3)" in repeated["reason"]
    assert second["state"] == "STARTUP_ALIGN"
    assert "(2/3)" in second["reason"]
    assert confirmed["state"] == "DRIVING"


def test_unconfirmed_sequence_bounce_does_not_replace_current_target():
    sdk = FakeSdk(active=True)
    clock = Clock()
    sam = FakeSam(valid_sam(frame_index=1))
    autonomy = controller(
        sdk,
        sam,
        clock,
        settings=Mission1ControlConfig(target_sequence_confirm_frames=2),
    )
    autonomy.tick()
    sam.payload["frame_index"] = 2
    assert autonomy.tick()["state"] == "DRIVING"

    sam.payload = valid_sam(
        frame_index=3,
        navigation=valid_sam()["navigation"] | {"target_sequence": 2},
    )
    assert autonomy.tick()["state"] == "STARTUP_ALIGN"
    sam.payload = valid_sam(frame_index=4)
    recovered = autonomy.tick()

    assert recovered["state"] == "DRIVING"
    assert recovered["target_sequence"] == 1


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
    assert status["sdk_angular"] == pytest.approx(-status["angular"], rel=1e-4)
    assert status["sdk_linear"] == pytest.approx(sdk.commands[-1].linear, rel=1e-4)
    assert status["sdk_angular"] == pytest.approx(sdk.commands[-1].angular, rel=1e-4)
    assert status["frame_id"] == "frame-00000001"
    assert status["plan_id"] == "plan-00000001"
    assert status["command_id"] == "cmd-00000001"
    assert status["command_accepted"] is True
    assert status["command_response_latency_ms"] >= 0.0
    assert status["control_limits"] == {
        "linear_max": 0.06,
        "angular_max": 0.22,
    }


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
    assert first["command_accepted"] is False
    assert first["command_id"] == "cmd-00000001"
    assert first["sdk_linear"] == pytest.approx(sdk.commands[0].linear)
    assert first["sdk_angular"] == pytest.approx(sdk.commands[0].angular)
    assert "control 503" in first["command_error"]
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


def test_rotate_to_goal_direction_does_not_flip_when_heading_error_wraps_near_180():
    # Regression: a near-180 deg heading_error is ambiguous under GPS/compass
    # noise (e.g. -179 -> +179 is a tiny real heading change but flips the
    # raw sign). Recomputing direction from that sign every tick made the
    # rover reverse mid-rotation and get stuck oscillating instead of
    # completing the turn.
    sdk = FakeSdk(active=True)
    clock = Clock()

    def sam_payload(heading_error_deg):
        navigation = valid_sam()["navigation"] | {"heading_error_deg": heading_error_deg}
        return valid_sam(
            navigation=navigation,
            path_valid=False,
            path_reason="NO_CONNECTED_TRAVERSABLE_PATH",
            path_mean_score=None,
            local_path_selected_heading_deg=None,
        )

    sam = FakeSam(sam_payload(-179.0))
    autonomy = controller(sdk, sam, clock)
    clock.value += 0.2
    first = autonomy.tick()

    sam.payload = sam_payload(179.0)
    clock.value += 0.2
    second = autonomy.tick()

    assert first["state"] == "ROTATING_TO_GOAL"
    assert second["state"] == "ROTATING_TO_GOAL"
    assert first["angular"] < 0.0
    assert second["angular"] < 0.0


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


class SequencedSam:
    """Serves one payload per tick from a fixed list (last one repeats)."""

    def __init__(self, payloads):
        self.payloads = payloads
        self.index = 0

    def get(self):
        payload = self.payloads[min(self.index, len(self.payloads) - 1)]
        self.index += 1
        return dict(payload)


def test_rotate_to_goal_has_hysteresis_between_entry_and_exit_thresholds():
    sdk = FakeSdk(active=True)
    clock = Clock()
    base_nav = valid_sam()["navigation"]
    sam = SequencedSam(
        [
            valid_sam(navigation=base_nav | {"heading_error_deg": -174.0}),
            # Below the entry threshold (100 deg default) but above the exit
            # threshold (20 deg default): without hysteresis this would fall
            # straight back to path tracking and immediately re-enter on the
            # next noisy reading, which showed up as the rover spinning back
            # and forth instead of settling.
            valid_sam(navigation=base_nav | {"heading_error_deg": -60.0}),
            valid_sam(navigation=base_nav | {"heading_error_deg": -10.0}),
        ]
    )
    autonomy = controller(sdk, sam, clock)

    entered = autonomy.tick()
    clock.value += 0.2
    still_rotating = autonomy.tick()
    clock.value += 0.2
    exited = autonomy.tick()

    assert entered["state"] == "ROTATING_TO_GOAL"
    assert still_rotating["state"] == "ROTATING_TO_GOAL"
    assert exited["state"] == "DRIVING"


def test_rotate_to_goal_stays_above_live_motor_deadzone_until_exit():
    sdk = FakeSdk(active=True)
    clock = Clock()
    base_nav = valid_sam()["navigation"]
    sam = SequencedSam(
        [
            valid_sam(navigation=base_nav | {"heading_error_deg": -179.0}),
            valid_sam(navigation=base_nav | {"heading_error_deg": -85.6}),
            valid_sam(navigation=base_nav | {"heading_error_deg": -19.0}),
        ]
    )
    settings = Mission1ControlConfig(
        max_angular=0.15,
        rotate_to_goal_heading_deg=170.0,
        rotate_exit_threshold_deg=20.0,
        rotate_to_goal_angular=0.15,
        minimum_rotate_angular=0.12,
    )
    autonomy = controller(sdk, sam, clock, settings=settings)

    entered = autonomy.tick()
    clock.value += 0.2
    deadzone_case = autonomy.tick()
    clock.value += 0.2
    exited = autonomy.tick()

    assert entered["state"] == "ROTATING_TO_GOAL"
    assert deadzone_case["state"] == "ROTATING_TO_GOAL"
    assert deadzone_case["controller_debug"]["desired_angular"] == pytest.approx(-0.12)
    assert deadzone_case["angular"] == pytest.approx(-0.12)
    assert deadzone_case["sdk_angular"] == pytest.approx(0.12)
    assert sdk.commands[1].angular == pytest.approx(0.12)
    assert exited["state"] == "DRIVING"


def test_rotate_to_goal_floor_is_preserved_after_smoothing_and_slew_limit():
    sdk = FakeSdk(active=True)
    clock = Clock()
    navigation = valid_sam()["navigation"] | {"heading_error_deg": -179.0}
    settings = Mission1ControlConfig(
        max_angular=0.15,
        rotate_to_goal_heading_deg=170.0,
        rotate_exit_threshold_deg=20.0,
        rotate_to_goal_angular=0.15,
        minimum_rotate_angular=0.12,
    )
    autonomy = Mission1Autonomy(
        sdk,
        FakeSam(valid_sam(navigation=navigation)),
        settings,
        {
            "control": {
                "linear_min": 0.0,
                "linear_max": 0.12,
                "angular_min": -0.15,
                "angular_max": 0.15,
                "command_smoothing_alpha": 0.95,
                "max_linear_delta_per_sec": 0.01,
                "max_angular_delta_per_sec": 0.01,
            }
        },
        live_control_enabled=True,
        clock=clock,
        monotonic=clock,
    )

    status = autonomy.tick()

    assert status["controller_debug"]["desired_angular"] == pytest.approx(-0.15)
    assert status["controller_debug"]["filtered_angular"] == pytest.approx(-0.12)
    assert status["sdk_angular"] == pytest.approx(0.12)
    assert sdk.commands[-1].angular == pytest.approx(0.12)


def test_rotate_to_goal_stops_before_reversing_previous_steering_sign():
    sdk = FakeSdk(active=True)
    clock = Clock()
    base_nav = valid_sam()["navigation"]
    sam = SequencedSam(
        [
            valid_sam(
                navigation=base_nav | {"heading_error_deg": 0.0},
                local_path_selected_heading_deg=30.0,
            ),
            valid_sam(navigation=base_nav | {"heading_error_deg": -179.0}),
            valid_sam(navigation=base_nav | {"heading_error_deg": -179.0}),
        ]
    )
    settings = Mission1ControlConfig(
        max_angular=0.15,
        rotate_to_goal_heading_deg=170.0,
        rotate_exit_threshold_deg=20.0,
        rotate_to_goal_angular=0.15,
        minimum_rotate_angular=0.12,
    )
    autonomy = Mission1Autonomy(
        sdk,
        sam,
        settings,
        {
            "control": {
                "linear_min": 0.0,
                "linear_max": 0.12,
                "angular_min": -0.15,
                "angular_max": 0.15,
                "command_smoothing_alpha": 0.25,
                "max_linear_delta_per_sec": 1.0,
                "max_angular_delta_per_sec": 1.0,
                "reverse_angular_sign_slowdown": True,
                "angular_deadband": 0.03,
            }
        },
        live_control_enabled=True,
        clock=clock,
        monotonic=clock,
    )

    clock.value += 0.2
    driving = autonomy.tick()
    clock.value += 0.2
    reversal_stop = autonomy.tick()
    clock.value += 0.2
    rotating = autonomy.tick()

    assert driving["angular"] > 0.0
    assert reversal_stop["state"] == "ROTATING_TO_GOAL"
    assert reversal_stop["angular"] == 0.0
    assert rotating["angular"] == pytest.approx(-0.12)
    assert rotating["sdk_angular"] == pytest.approx(0.12)


@pytest.mark.parametrize("minimum", [0.16, float("inf"), float("nan")])
def test_minimum_rotate_angular_must_fit_rotation_command_envelope(minimum):
    settings = Mission1ControlConfig(
        max_angular=0.15,
        rotate_to_goal_angular=0.15,
        minimum_rotate_angular=minimum,
    )

    with pytest.raises(ValueError, match="minimum_rotate_angular|positive control"):
        settings.validate()


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


def test_planner_switch_stop_bypasses_path_hold_and_search_rotate():
    sdk = FakeSdk(active=True)
    clock = Clock()
    good = valid_sam(local_path_selected_heading_deg=0.0)
    sam = FakeSam(good)
    autonomy = controller(sdk, sam, clock)
    clock.value += 0.2
    assert autonomy.tick()["state"] == "DRIVING"

    sam.payload = valid_sam(
        planner={
            "switch_stop_required": True,
            "switch_reason": "stop_unsafe_candidate_switch_pending",
            "near_field_safe": True,
        },
        local_path_selected_heading_deg=None,
    )
    clock.value += 0.2
    stopped = autonomy.tick()

    assert stopped["state"] == "SAFETY_STOP"
    assert sdk.commands[-1].linear == 0.0
    assert sdk.commands[-1].angular == 0.0


def test_planner_switch_stop_requires_distinct_safe_frames_before_restart():
    sdk = FakeSdk(active=True)
    clock = Clock()
    good = valid_sam(local_path_selected_heading_deg=0.0, frame_index=10)
    stopped_payload = valid_sam(
        frame_index=11,
        planner={
            "switch_stop_required": True,
            "switch_reason": "stop_unsafe_candidate_switch_pending",
            "near_field_safe": True,
        },
        local_path_selected_heading_deg=None,
    )
    sam = FakeSam(good)
    autonomy = controller(
        sdk,
        sam,
        clock,
        settings=Mission1ControlConfig(
            path_recovery_confirm_frames=3,
            maximum_sam_age_sec=10.0,
        ),
    )

    clock.value += 0.2
    assert autonomy.tick()["state"] == "DRIVING"
    sam.payload = stopped_payload
    clock.value += 0.2
    assert autonomy.tick()["state"] == "SAFETY_STOP"

    sam.payload = valid_sam(local_path_selected_heading_deg=0.0, frame_index=12)
    clock.value += 0.2
    first = autonomy.tick()
    clock.value += 0.2
    duplicate = autonomy.tick()
    sam.payload = valid_sam(local_path_selected_heading_deg=0.0, frame_index=13)
    clock.value += 0.2
    second = autonomy.tick()
    sam.payload = valid_sam(local_path_selected_heading_deg=0.0, frame_index=14)
    clock.value += 0.2
    recovered = autonomy.tick()

    assert first["state"] == "SAFETY_STOP"
    assert duplicate["reason"] == "confirming safe path recovery (1/3)"
    assert second["reason"] == "confirming safe path recovery (2/3)"
    assert recovered["state"] == "DRIVING"
    assert sdk.commands[-1].linear > 0.0


def test_live_config_can_disable_unsafe_search_rotation():
    sdk = FakeSdk(active=True)
    clock = Clock()
    settings = Mission1ControlConfig(enable_search_rotate=False)
    autonomy = controller(sdk, LiveSam(blocked_sam(), clock), clock, settings=settings)
    clock.value += 0.2

    status = autonomy.tick()

    assert status["state"] == "SAFETY_STOP"
    assert status["linear"] == 0.0
    assert status["angular"] == 0.0


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
    assert sdk.commands[-1].linear >= autonomy.settings.minimum_linear


def test_live_drive_command_never_falls_below_motor_deadzone_floor():
    sdk = FakeSdk(active=True)
    clock = Clock()
    settings = Mission1ControlConfig(
        minimum_linear=0.10,
        base_linear=0.20,
        max_linear=0.30,
    )
    filter_config = {
        "control": {
            "linear_min": 0.0,
            "linear_max": 0.30,
            "angular_min": -0.30,
            "angular_max": 0.30,
            "command_smoothing_alpha": 0.0,
            "max_linear_delta_per_sec": 0.10,
            "max_angular_delta_per_sec": 1.0,
        }
    }
    autonomy = Mission1Autonomy(
        sdk,
        FakeSam(valid_sam(local_path_selected_heading_deg=30.0)),
        settings,
        filter_config,
        live_control_enabled=True,
        clock=clock,
        monotonic=clock,
    )
    clock.value += 0.2

    status = autonomy.tick()

    assert status["state"] == "DRIVING"
    assert sdk.commands[-1].linear == pytest.approx(0.10)
    assert status["linear"] == pytest.approx(0.10)


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


class TimeoutThenConfirmedSdk(FakeSdk):
    """First report POST raises, but the SDK already advanced server-side."""

    def __init__(self, active: bool = True) -> None:
        super().__init__(active=active)
        self.report_attempts = 0

    def get_mission_status(self):
        status = {"mission_active": self.active}
        if self.report_attempts >= 1:
            status["latest_scanned_checkpoint"] = 1
        return status

    def report_checkpoint_details(self):
        self.report_attempts += 1
        raise RuntimeError("POST /checkpoint-reached failed: Read timed out")


class AlwaysTimeoutSdk(FakeSdk):
    def report_checkpoint_details(self):
        self.reports += 1
        raise RuntimeError("POST /checkpoint-reached failed: Read timed out")


class LiveSam:
    """Like FakeSam, but keeps published_timestamp in sync with the clock so
    advancing past the checkpoint cooldown doesn't also trip the staleness
    check unrelated to what these tests exercise."""

    def __init__(self, base_payload, clock):
        self.base_payload = base_payload
        self.clock = clock

    def get(self):
        payload = dict(self.base_payload)
        payload["published_timestamp"] = self.clock.value
        return payload


def test_search_rotate_when_all_candidates_blocked_turns_toward_clearer_side():
    sdk = FakeSdk(active=True)
    clock = Clock()
    autonomy = controller(sdk, LiveSam(blocked_sam(), clock), clock)
    clock.value += 0.2

    status = autonomy.tick()

    assert status["state"] == "SEARCH_ROTATE"
    assert status["linear"] == 0.0
    assert status["angular"] > 0.0  # right side (near_field 0.20) is clearer
    assert sdk.commands[-1].linear == 0.0
    assert sdk.commands[-1].angular < 0.0  # sdk-level convention: right


def test_search_rotate_turns_left_when_left_side_is_clearer():
    sdk = FakeSdk(active=True)
    clock = Clock()
    payload = blocked_sam(
        candidate_scores=[
            {"heading_deg": -30.0, "near_field": 0.30},
            {"heading_deg": -10.0, "near_field": 0.10},
            {"heading_deg": 10.0, "near_field": 0.02},
            {"heading_deg": 30.0, "near_field": 0.01},
        ]
    )
    autonomy = controller(sdk, LiveSam(payload, clock), clock)
    clock.value += 0.2

    status = autonomy.tick()

    assert status["state"] == "SEARCH_ROTATE"
    assert status["angular"] < 0.0


def test_search_rotate_gives_up_after_timeout_without_a_clear_path():
    sdk = FakeSdk(active=True)
    clock = Clock()
    autonomy = controller(sdk, LiveSam(blocked_sam(), clock), clock)
    autonomy.settings = Mission1ControlConfig(search_rotate_timeout_sec=1.0)

    clock.value += 0.2
    first = autonomy.tick()
    clock.value += 1.5
    second = autonomy.tick()

    assert first["state"] == "SEARCH_ROTATE"
    assert second["state"] == "SAFETY_STOP"


def test_search_rotate_resumes_driving_once_a_clear_path_appears():
    sdk = FakeSdk(active=True)
    clock = Clock()
    sam = LiveSam(blocked_sam(), clock)
    autonomy = controller(sdk, sam, clock)
    clock.value += 0.2
    blocked = autonomy.tick()

    sam.base_payload = valid_sam()
    clock.value += 0.2
    resumed = autonomy.tick()

    assert blocked["state"] == "SEARCH_ROTATE"
    assert resumed["state"] == "DRIVING"
    assert autonomy._searching_for_path is False


def test_checkpoint_report_timeout_is_confirmed_via_mission_status_without_duplicate_post():
    sdk = TimeoutThenConfirmedSdk()
    clock = Clock()
    navigation = valid_sam()["navigation"] | {"reached": True}
    autonomy = controller(sdk, LiveSam(valid_sam(navigation=navigation), clock), clock)

    first = autonomy.tick()
    clock.value += autonomy.settings.checkpoint_report_cooldown_sec + 0.1
    second = autonomy.tick()

    assert first["state"] == "CHECKPOINT_WAIT"
    assert "report failed" in first["reason"]
    assert second["state"] == "CHECKPOINT_REPORTED"
    assert "already accepted" in second["reason"]
    assert sdk.report_attempts == 1


def test_checkpoint_report_timeout_retries_after_cooldown_when_not_confirmed():
    sdk = AlwaysTimeoutSdk(active=True)
    clock = Clock()
    navigation = valid_sam()["navigation"] | {"reached": True}
    autonomy = controller(sdk, LiveSam(valid_sam(navigation=navigation), clock), clock)

    first = autonomy.tick()
    clock.value += autonomy.settings.checkpoint_report_cooldown_sec + 0.1
    second = autonomy.tick()

    assert first["state"] == "CHECKPOINT_WAIT"
    assert second["state"] == "CHECKPOINT_WAIT"
    assert sdk.reports == 2


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


def test_metric_mode_without_calibration_sends_zero_and_never_reaches_path_hold():
    sdk = FakeSdk(active=True)
    clock = Clock()
    sam = FakeSam(
        valid_sam(
            planner={"geometry_mode": "metric_projected", "near_field_safe": True},
            camera_projection_applied=False,
            image_path_metric_calibrated=False,
        )
    )
    autonomy = controller(sdk, sam, clock)
    clock.value += 0.2

    status = autonomy.tick()

    assert status["state"] == "SAFETY_STOP"
    assert "metric" in status["reason"]
    assert status["linear"] == 0.0
    assert status["angular"] == 0.0
    assert sdk.commands[-1].linear == 0.0
    assert sdk.commands[-1].angular == 0.0


def test_live_metric_requirement_rejects_image_heuristic_shadow_status():
    sdk = FakeSdk(active=True)
    clock = Clock()
    settings = Mission1ControlConfig(require_metric_projection=True)
    autonomy = controller(
        sdk,
        FakeSam(valid_sam(planner={"geometry_mode": "image_heuristic"})),
        clock,
        settings=settings,
    )
    clock.value += 0.2

    status = autonomy.tick()

    assert status["state"] == "SAFETY_STOP"
    assert "metric camera projection is required" in status["reason"]
    assert sdk.commands[-1].linear == 0.0
    assert sdk.commands[-1].angular == 0.0


def test_metric_mode_without_calibration_never_reaches_search_rotate_even_when_blocked():
    sdk = FakeSdk(active=True)
    clock = Clock()
    payload = valid_sam(
        planner={
            "geometry_mode": "metric_projected",
            "near_field_safe": False,
            "near_field_score": 0.0,
            "candidate_scores": [
                {"heading_deg": -30.0, "near_field": 0.05},
                {"heading_deg": 30.0, "near_field": 0.20},
            ],
        },
        camera_projection_applied=False,
        image_path_metric_calibrated=False,
    )
    settings = Mission1ControlConfig(enable_search_rotate=True)
    autonomy = controller(sdk, FakeSam(payload), clock, settings=settings)
    clock.value += 0.2

    status = autonomy.tick()

    assert status["state"] == "SAFETY_STOP"
    assert "metric" in status["reason"]
    assert status["linear"] == 0.0
    assert status["angular"] == 0.0


_DRIVABLE_METRIC_PLANNER = {
    "near_field_safe": True,
    "trajectory_valid": True,
    "trajectory_quality": 0.9,
    "planner_confidence": 0.9,
    "plan_age_sec": 0.0,
    "using_held_plan": False,
}


def test_metric_mode_with_valid_calibration_flags_does_not_trigger_calibration_gate():
    sdk = FakeSdk(active=True)
    clock = Clock()
    sam = FakeSam(
        valid_sam(
            planner={**_DRIVABLE_METRIC_PLANNER, "geometry_mode": "metric_projected"},
            camera_projection_applied=True,
            image_path_metric_calibrated=True,
        )
    )
    autonomy = controller(sdk, sam, clock)
    clock.value += 0.2

    status = autonomy.tick()

    assert status["state"] == "DRIVING"

    assert sdk.commands[-1].linear > 0.0


def test_metric_mode_accepts_calibration_flags_from_real_planner_status_shape():
    sdk = FakeSdk(active=True)
    clock = Clock()
    sam = FakeSam(
        valid_sam(
            planner={
                **_DRIVABLE_METRIC_PLANNER,
                "geometry_mode": "metric_projected",
                "camera_projection_applied": True,
                "image_path_metric_calibrated": True,
                "calibration_id": "mission1_front_camera_1024x576_2026_08_14",
            },
        )
    )
    autonomy = controller(sdk, sam, clock)
    clock.value += 0.2

    status = autonomy.tick()

    assert status["state"] == "DRIVING"
    assert sdk.commands[-1].linear > 0.0


def test_image_heuristic_mode_is_unaffected_by_calibration_gate():
    sdk = FakeSdk(active=True)
    clock = Clock()
    sam = FakeSam(
        valid_sam(
            planner={**_DRIVABLE_METRIC_PLANNER, "geometry_mode": "image_heuristic"}
        )
    )
    autonomy = controller(sdk, sam, clock)
    clock.value += 0.2

    status = autonomy.tick()

    assert status["state"] == "DRIVING"


def _side_sector(
    *,
    chosen="RIGHT",
    status="RIGHT_CLEAR",
    left_viable=False,
    right_viable=True,
    left_composite=0.05,
    right_composite=0.9,
    margin=0.85,
    reason="right side sector is clearly safer than left",
):
    return {
        "left": {
            "side": "LEFT",
            "mean": left_composite,
            "low_percentile": left_composite,
            "traversable_ratio": 1.0 if left_viable else 0.0,
            "valid_pixel_ratio": 1.0,
            "pixel_count": 100,
            "composite": left_composite,
            "viable": left_viable,
        },
        "right": {
            "side": "RIGHT",
            "mean": right_composite,
            "low_percentile": right_composite,
            "traversable_ratio": 1.0 if right_viable else 0.0,
            "valid_pixel_ratio": 1.0,
            "pixel_count": 100,
            "composite": right_composite,
            "viable": right_viable,
        },
        "chosen": chosen,
        "status": status,
        "margin": margin,
        "reason": reason,
    }


_UNSET = object()


def _telemetry(timestamp, *, stationary=True):
    return {
        "local_timestamp": timestamp,
        "sdk_timestamp": timestamp,
        "speed": 0.0 if stationary else 0.5,
        "rpms": [0.0, 0.0] if stationary else [5.0, 5.0],
    }


def _resolve_telemetry(telemetry, telemetry_timestamp, stationary):
    if telemetry is not _UNSET:
        return telemetry
    if telemetry_timestamp is None:
        return None
    return _telemetry(telemetry_timestamp, stationary=stationary)


def blocked_sam_with_side_sector(
    *,
    side_sector=None,
    frame_index=1,
    telemetry_timestamp=1.0,
    stationary=True,
    telemetry=_UNSET,
    telemetry_valid=True,
    telemetry_age_sec=0.1,
    **overrides,
):
    planner = {
        "near_field_safe": False,
        "switch_stop_required": True,
        "switch_reason": "all_candidates_hard_rejected",
        "side_sector": side_sector if side_sector is not None else _side_sector(),
    }
    return valid_sam(
        planner=planner,
        local_path_selected_heading_deg=None,
        frame_index=frame_index,
        telemetry=_resolve_telemetry(telemetry, telemetry_timestamp, stationary),
        telemetry_valid=telemetry_valid,
        telemetry_age_sec=telemetry_age_sec,
        **overrides,
    )


def safe_sam_with_side_sector(
    *,
    side_sector=None,
    frame_index=1,
    heading_deg=0.0,
    telemetry_timestamp=1.0,
    stationary=True,
    telemetry=_UNSET,
    telemetry_valid=True,
    telemetry_age_sec=0.1,
    **overrides,
):
    planner = {
        "near_field_safe": True,
        "switch_stop_required": False,
        "side_sector": side_sector if side_sector is not None else _side_sector(),
        "trajectory_valid": True,
        "trajectory_quality": 0.9,
        "planner_confidence": 0.8,
        "plan_age_sec": 0.0,
        "using_held_plan": False,
        "selected_candidate_index": 4,
    }
    return valid_sam(
        planner=planner,
        local_path_selected_heading_deg=heading_deg,
        path_mean_score=0.9,
        frame_index=frame_index,
        telemetry=_resolve_telemetry(telemetry, telemetry_timestamp, stationary),
        telemetry_valid=telemetry_valid,
        telemetry_age_sec=telemetry_age_sec,
        **overrides,
    )


def stop_turn_go_settings(**overrides):
    values = {
        "enable_stop_turn_go": True,
        "minimum_linear": 0.06,
        "base_linear": 0.06,
        "max_linear": 0.06,
        "max_angular": 0.15,
        "minimum_rotate_angular": 0.12,
        "stop_turn_rotate_angular": 0.12,
        "stop_turn_heading_threshold_deg": 7.5,
        "stop_turn_confirm_frames": 2,
        "stop_turn_stationary_confirm_samples": 1,
        "stop_turn_rotate_pulse_sec": 0.3,
        "stop_turn_settle_sec": 0.5,
        "stop_turn_drive_burst_sec": 0.6,
    }
    values.update(overrides)
    return Mission1ControlConfig(**values)


def test_stop_turn_go_stops_then_pulse_rotates_without_linear_motion():
    sdk = FakeSdk(active=True)
    clock = Clock()
    sam = LiveSam(
        safe_sam_with_side_sector(frame_index=1, heading_deg=20.0), clock
    )
    autonomy = controller(sdk, sam, clock, settings=stop_turn_go_settings())

    stopped = autonomy.tick()
    sam.base_payload = safe_sam_with_side_sector(
        frame_index=2, heading_deg=20.0, telemetry_timestamp=2.0
    )
    clock.value += 0.2
    first_confirmation = autonomy.tick()
    sam.base_payload = safe_sam_with_side_sector(
        frame_index=3, heading_deg=20.0, telemetry_timestamp=3.0
    )
    clock.value += 0.2
    second_confirmation = autonomy.tick()
    clock.value += 0.2
    pulse = autonomy.tick()

    assert stopped["state"] == "STG_STOP_CONFIRM"
    assert first_confirmation["state"] == second_confirmation["state"] == "STG_ALIGN_CONFIRM"
    assert pulse["state"] == "STG_ROTATE_RIGHT"
    assert pulse["linear"] == 0.0
    assert pulse["angular"] == 0.12
    assert sdk.commands[-1].linear == 0.0
    assert sdk.commands[-1].angular == -0.12


def test_stop_turn_go_motion_gate_aborts_unresponsive_rotate_actuator():
    sdk = FakeSdk(active=True)
    clock = Clock()
    sam = LiveSam(
        safe_sam_with_side_sector(frame_index=1, heading_deg=20.0), clock
    )
    autonomy = controller(
        sdk,
        sam,
        clock,
        settings=stop_turn_go_settings(
            stop_turn_require_motion_response=True,
            stop_turn_motion_response_timeout_sec=0.6,
        ),
    )

    autonomy.tick()
    for frame_index in (2, 3):
        sam.base_payload = safe_sam_with_side_sector(
            frame_index=frame_index,
            heading_deg=20.0,
            telemetry_timestamp=float(frame_index),
        )
        clock.value += 0.2
        autonomy.tick()
    clock.value += 0.2
    assert autonomy.tick()["state"] == "STG_ROTATE_RIGHT"
    clock.value += 0.3
    still_waiting = autonomy.tick()
    clock.value += 0.3
    settling = autonomy.tick()
    clock.value += 0.4
    aborted = autonomy.tick()

    assert still_waiting["state"] == settling["state"] == "STG_ROTATE_SETTLE"
    assert still_waiting["angular"] == settling["angular"] == 0.0
    assert settling["stop_turn_go"]["motion_observed"] is False
    assert aborted["state"] == "STG_SAFETY_STOP"
    assert "actuator did not respond" in aborted["reason"]
    assert aborted["linear"] == aborted["angular"] == 0.0


def test_stop_turn_go_motion_gate_accepts_fresh_rpm_response():
    sdk = FakeSdk(active=True)
    clock = Clock()
    sam = LiveSam(
        safe_sam_with_side_sector(frame_index=1, heading_deg=20.0), clock
    )
    autonomy = controller(
        sdk,
        sam,
        clock,
        settings=stop_turn_go_settings(
            stop_turn_require_motion_response=True,
            stop_turn_motion_response_timeout_sec=0.8,
        ),
    )

    autonomy.tick()
    for frame_index in (2, 3):
        sam.base_payload = safe_sam_with_side_sector(
            frame_index=frame_index,
            heading_deg=20.0,
            telemetry_timestamp=float(frame_index),
        )
        clock.value += 0.2
        autonomy.tick()
    clock.value += 0.2
    assert autonomy.tick()["state"] == "STG_ROTATE_RIGHT"

    sam.base_payload = safe_sam_with_side_sector(
        frame_index=4,
        heading_deg=20.0,
        telemetry_timestamp=4.0,
        stationary=False,
    )
    clock.value += 0.3
    settled = autonomy.tick()

    assert settled["state"] == "STG_ROTATE_SETTLE"
    assert settled["stop_turn_go"]["motion_observed"] is True
    assert settled["stop_turn_go"]["max_abs_rpm"] == pytest.approx(5.0)


def test_stop_turn_go_requires_distinct_frames_before_driving_straight():
    sdk = FakeSdk(active=True)
    clock = Clock()
    sam = LiveSam(safe_sam_with_side_sector(frame_index=10, heading_deg=0.0), clock)
    autonomy = controller(sdk, sam, clock, settings=stop_turn_go_settings())

    first = autonomy.tick()
    clock.value += 0.2
    repeated = autonomy.tick()
    sam.base_payload = safe_sam_with_side_sector(
        frame_index=11, heading_deg=0.0, telemetry_timestamp=11.0
    )
    clock.value += 0.2
    confirmed = autonomy.tick()
    sam.base_payload = safe_sam_with_side_sector(
        frame_index=12, heading_deg=0.0, telemetry_timestamp=12.0
    )
    clock.value += 0.2
    straight_confirmed = autonomy.tick()
    clock.value += 0.2
    driving = autonomy.tick()

    assert first["state"] == "STG_STOP_CONFIRM"
    assert repeated["state"] == confirmed["state"] == "STG_ALIGN_CONFIRM"
    assert straight_confirmed["state"] == "STG_STRAIGHT_CONFIRM"
    assert driving["state"] == "STG_DRIVE_STRAIGHT"
    assert driving["linear"] == 0.06
    assert driving["angular"] == 0.0


def test_stop_turn_go_global_heading_blocks_opposite_local_straight_drive():
    sdk = FakeSdk(active=True)
    clock = Clock()
    navigation = valid_sam()["navigation"] | {"heading_error_deg": 170.0}
    sam = LiveSam(
        safe_sam_with_side_sector(
            frame_index=1,
            heading_deg=0.0,
            navigation=navigation,
        ),
        clock,
    )
    autonomy = controller(sdk, sam, clock, settings=stop_turn_go_settings())

    states = []
    for frame_index in range(1, 5):
        sam.base_payload = safe_sam_with_side_sector(
            frame_index=frame_index,
            heading_deg=0.0,
            telemetry_timestamp=float(frame_index),
            navigation=navigation,
        )
        clock.value += 0.2
        states.append(autonomy.tick())

    assert [status["state"] for status in states] == [
        "STG_STOP_CONFIRM",
        "STG_ALIGN_CONFIRM",
        "STG_ALIGN_CONFIRM",
        "STG_ROTATE_RIGHT",
    ]
    assert all(status["linear"] == 0.0 for status in states)
    assert states[-1]["angular"] > 0.0
    assert all(command.linear == 0.0 for command in sdk.commands)


def test_stop_turn_go_uses_only_viable_side_when_shortest_global_turn_is_blocked():
    sdk = FakeSdk(active=True)
    clock = Clock()
    navigation = valid_sam()["navigation"] | {"heading_error_deg": 170.0}
    left_only = _side_sector(
        chosen="LEFT",
        status="LEFT_CLEAR",
        left_viable=True,
        right_viable=False,
        left_composite=0.9,
        right_composite=0.05,
        margin=0.85,
    )
    sam = LiveSam(
        safe_sam_with_side_sector(
            side_sector=left_only,
            heading_deg=0.0,
            navigation=navigation,
        ),
        clock,
    )
    autonomy = controller(sdk, sam, clock, settings=stop_turn_go_settings())

    states = []
    for frame_index in range(1, 5):
        sam.base_payload = safe_sam_with_side_sector(
            side_sector=left_only,
            frame_index=frame_index,
            heading_deg=0.0,
            telemetry_timestamp=float(frame_index),
            navigation=navigation,
        )
        clock.value += 0.2
        states.append(autonomy.tick())

    assert states[-1]["state"] == "STG_ROTATE_LEFT"
    assert states[-1]["angular"] < 0.0
    assert all(status["linear"] == 0.0 for status in states)
    assert all(command.linear == 0.0 for command in sdk.commands)


def test_stop_turn_go_rejects_global_turn_without_one_viable_side():
    sdk = FakeSdk(active=True)
    clock = Clock()
    navigation = valid_sam()["navigation"] | {"heading_error_deg": 170.0}
    ambiguous = _side_sector(
        chosen=None,
        status="AMBIGUOUS",
        left_viable=False,
        right_viable=False,
    )
    sam = LiveSam(
        safe_sam_with_side_sector(
            side_sector=ambiguous,
            heading_deg=0.0,
            navigation=navigation,
        ),
        clock,
    )
    autonomy = controller(sdk, sam, clock, settings=stop_turn_go_settings())

    status = autonomy.tick()

    assert status["state"] == "SAFETY_STOP"
    assert status["linear"] == status["angular"] == 0.0
    assert "side-sector evidence is AMBIGUOUS" in status["reason"]
    assert sdk.commands[-1].linear == sdk.commands[-1].angular == 0.0


def test_stop_turn_go_rotation_settle_requires_time_and_fresh_frame():
    sdk = FakeSdk(active=True)
    clock = Clock()
    sam = LiveSam(safe_sam_with_side_sector(frame_index=1, heading_deg=20.0), clock)
    autonomy = controller(sdk, sam, clock, settings=stop_turn_go_settings())

    autonomy.tick()
    for frame_index in (2, 3):
        sam.base_payload = safe_sam_with_side_sector(
            frame_index=frame_index,
            heading_deg=20.0,
            telemetry_timestamp=float(frame_index),
        )
        clock.value += 0.2
        autonomy.tick()
    clock.value += 0.2
    assert autonomy.tick()["state"] == "STG_ROTATE_RIGHT"
    clock.value += 0.3
    deadline = autonomy.tick()
    clock.value += 0.5
    repeated = autonomy.tick()

    assert deadline["state"] == repeated["state"] == "STG_ROTATE_SETTLE"
    assert deadline["angular"] == repeated["angular"] == 0.0
    assert repeated["stop_turn_go"]["phase"] == "ROTATE_SETTLE"

    sam.base_payload = safe_sam_with_side_sector(
        frame_index=4, heading_deg=0.0, telemetry_timestamp=4.0
    )
    clock.value += 0.2
    fresh = autonomy.tick()
    assert fresh["state"] == "STG_ROTATE_SETTLE"
    clock.value += 0.1
    next_phase = autonomy.tick()
    assert next_phase["state"] == "STG_STRAIGHT_CONFIRM"


def test_stop_turn_go_aborts_drive_immediately_when_heading_leaves_straight_band():
    sdk = FakeSdk(active=True)
    clock = Clock()
    sam = LiveSam(safe_sam_with_side_sector(frame_index=1, heading_deg=0.0), clock)
    autonomy = controller(sdk, sam, clock, settings=stop_turn_go_settings())

    autonomy.tick()
    sam.base_payload = safe_sam_with_side_sector(
        frame_index=2, heading_deg=0.0, telemetry_timestamp=2.0
    )
    clock.value += 0.2
    autonomy.tick()
    sam.base_payload = safe_sam_with_side_sector(
        frame_index=3, heading_deg=0.0, telemetry_timestamp=3.0
    )
    clock.value += 0.2
    autonomy.tick()
    clock.value += 0.2
    assert autonomy.tick()["state"] == "STG_DRIVE_STRAIGHT"

    sam.base_payload = safe_sam_with_side_sector(
        frame_index=4, heading_deg=-20.0, telemetry_timestamp=4.0
    )
    clock.value += 0.1
    stopped = autonomy.tick()

    assert stopped["state"] == "STG_STOP_CONFIRM"
    assert stopped["linear"] == stopped["angular"] == 0.0


def test_stop_turn_go_drive_settle_waits_for_physical_stop_and_fresh_frame():
    sdk = FakeSdk(active=True)
    clock = Clock()
    settings = stop_turn_go_settings(stop_turn_stationary_confirm_samples=2)
    sam = LiveSam(safe_sam_with_side_sector(frame_index=1, heading_deg=0.0), clock)
    autonomy = controller(sdk, sam, clock, settings=settings)

    autonomy.tick()
    sam.base_payload = safe_sam_with_side_sector(
        frame_index=2, heading_deg=0.0, telemetry_timestamp=2.0
    )
    clock.value += 0.2
    autonomy.tick()
    sam.base_payload = safe_sam_with_side_sector(
        frame_index=3, heading_deg=0.0, telemetry_timestamp=3.0
    )
    clock.value += 0.2
    autonomy.tick()
    sam.base_payload = safe_sam_with_side_sector(
        frame_index=4, heading_deg=0.0, telemetry_timestamp=4.0
    )
    clock.value += 0.2
    autonomy.tick()
    clock.value += 0.2
    autonomy.tick()
    clock.value += 0.6
    assert autonomy.tick()["state"] == "STG_DRIVE_SETTLE"

    sam.base_payload = safe_sam_with_side_sector(
        frame_index=5, heading_deg=0.0, telemetry_timestamp=5.0, stationary=False
    )
    clock.value += 0.5
    moving = autonomy.tick()
    assert moving["state"] == "STG_DRIVE_SETTLE"
    assert moving["stop_turn_go"]["stationary_count"] == 0

    for frame_index in (6, 7):
        sam.base_payload = safe_sam_with_side_sector(
            frame_index=frame_index,
            heading_deg=0.0,
            telemetry_timestamp=float(frame_index),
        )
        clock.value += 0.2
        settled = autonomy.tick()

    assert settled["linear"] == settled["angular"] == 0.0
    clock.value += 0.1
    next_phase = autonomy.tick()
    assert next_phase["state"] == "STG_STRAIGHT_CONFIRM"


def test_stop_turn_go_never_sends_linear_and_angular_together():
    sdk = FakeSdk(active=True)
    clock = Clock()
    sam = LiveSam(safe_sam_with_side_sector(frame_index=1, heading_deg=20.0), clock)
    autonomy = controller(sdk, sam, clock, settings=stop_turn_go_settings())

    autonomy.tick()
    for frame_index in (2, 3):
        sam.base_payload = safe_sam_with_side_sector(
            frame_index=frame_index,
            heading_deg=20.0,
            telemetry_timestamp=float(frame_index),
        )
        clock.value += 0.2
        autonomy.tick()
    clock.value += 0.2
    autonomy.tick()
    clock.value += 0.3
    autonomy.tick()
    sam.base_payload = safe_sam_with_side_sector(
        frame_index=4, heading_deg=0.0, telemetry_timestamp=4.0
    )
    clock.value += 0.5
    autonomy.tick()
    sam.base_payload = safe_sam_with_side_sector(
        frame_index=5, heading_deg=0.0, telemetry_timestamp=5.0
    )
    clock.value += 0.2
    autonomy.tick()
    clock.value += 0.2
    autonomy.tick()

    assert sdk.commands
    assert all(not (command.linear != 0.0 and command.angular != 0.0) for command in sdk.commands)


def _tick_until(autonomy, clock, predicate, *, dt=0.2, max_ticks=30):
    """Advance ticks until predicate(status) is true; returns that status.

    Timing-robust alternative to hand-computing exact elapsed/pulse_sec/
    settle_sec arithmetic in a test -- asserts on state transitions actually
    happening, not on precisely replicating the implementation's internal
    per-tick elapsed-time bookkeeping.
    """

    status = None
    for _ in range(max_ticks):
        clock.value += dt
        status = autonomy.tick()
        if predicate(status):
            return status
    raise AssertionError(f"predicate not satisfied within {max_ticks} ticks; last status={status}")


def rotate_escape_settings(**overrides):
    # A single stationary telemetry sample and a single tick suffice by
    # default, so most tests only need two ticks to reach the first pulse:
    # one STOP_CONFIRM tick (which sees the fresh stationary sample and
    # transitions internally) and one ROTATE_PULSE tick.
    values = {
        "enable_rotate_escape": True,
        "rotate_escape_stop_confirm_ticks": 1,
        "rotate_escape_stationary_confirm_samples": 1,
        "rotate_escape_direction_confirm_frames": 1,
        "rotate_escape_pulse_sec": 0.3,
        # Existing recovery tests exercise state transitions, not actuator
        # feedback. Zero makes their stationary fixture count as observed;
        # dedicated tests below cover the production no-response gate.
        "rotate_escape_motion_rpm_threshold": 0.0,
        "rotate_escape_settle_sec": 0.3,
        "max_angular": 0.15,
    } | overrides
    return Mission1ControlConfig(**values)


def _confirm_direction(
    autonomy, sam, clock, *, frame_index=2, telemetry_timestamp=2.0, dt=0.2
):
    payload = dict(sam.base_payload)
    payload["frame_index"] = frame_index
    payload["telemetry"] = _telemetry(telemetry_timestamp)
    sam.base_payload = payload
    clock.value += dt
    status = autonomy.tick()
    assert status["state"] == "DIRECTION_CONFIRM"
    assert status["recovery"]["direction_confirm_count"] >= 1
    return status


def test_rotate_escape_selects_right_when_left_wall_right_open():
    sdk = FakeSdk(active=True)
    clock = Clock()
    settings = rotate_escape_settings()
    sam = LiveSam(blocked_sam_with_side_sector(side_sector=_side_sector(chosen="RIGHT")), clock)
    autonomy = controller(sdk, sam, clock, settings=settings)

    clock.value += 0.2
    stop_confirm = autonomy.tick()
    assert stop_confirm["state"] == "STOP_CONFIRM"
    assert stop_confirm["linear"] == 0.0
    assert stop_confirm["angular"] == 0.0

    _confirm_direction(autonomy, sam, clock)
    clock.value += 0.2
    pulse = autonomy.tick()
    assert pulse["state"] == "ROTATE_PULSE_RIGHT"
    assert pulse["linear"] == 0.0
    assert pulse["angular"] > 0.0
    assert sdk.commands[-1].linear == 0.0
    assert sdk.commands[-1].angular < 0.0  # sdk convention: negative is right


def test_rotate_escape_selects_left_when_right_wall_left_open():
    sdk = FakeSdk(active=True)
    clock = Clock()
    settings = rotate_escape_settings()
    side_sector = _side_sector(
        chosen="LEFT",
        status="LEFT_CLEAR",
        left_viable=True,
        right_viable=False,
        left_composite=0.9,
        right_composite=0.05,
        reason="left side sector is clearly safer than right",
    )
    sam = LiveSam(blocked_sam_with_side_sector(side_sector=side_sector), clock)
    autonomy = controller(sdk, sam, clock, settings=settings)

    clock.value += 0.2
    autonomy.tick()
    _confirm_direction(autonomy, sam, clock)
    clock.value += 0.2
    pulse = autonomy.tick()

    assert pulse["state"] == "ROTATE_PULSE_LEFT"
    assert pulse["linear"] == 0.0
    assert pulse["angular"] < 0.0
    assert sdk.commands[-1].angular > 0.0  # sdk convention: positive is left


def test_rotate_escape_stops_when_side_sectors_ambiguous():
    sdk = FakeSdk(active=True)
    clock = Clock()
    side_sector = _side_sector(
        chosen=None,
        status="AMBIGUOUS",
        left_viable=True,
        right_viable=True,
        left_composite=0.60,
        right_composite=0.65,
        margin=0.05,
        reason="side sector composite margin 0.050 is below the required 0.120",
    )
    settings = rotate_escape_settings()
    autonomy = controller(sdk, LiveSam(blocked_sam_with_side_sector(side_sector=side_sector), clock), clock, settings=settings)
    clock.value += 0.2

    status = autonomy.tick()

    assert status["state"] == "SAFETY_STOP"
    assert status["linear"] == 0.0
    assert status["angular"] == 0.0


def test_rotate_escape_stops_when_both_side_sectors_unsafe():
    sdk = FakeSdk(active=True)
    clock = Clock()
    side_sector = _side_sector(
        chosen=None,
        status="BOTH_UNSAFE",
        left_viable=False,
        right_viable=False,
        left_composite=0.05,
        right_composite=0.10,
        reason="neither side sector clears the safety thresholds",
    )
    settings = rotate_escape_settings()
    autonomy = controller(sdk, LiveSam(blocked_sam_with_side_sector(side_sector=side_sector), clock), clock, settings=settings)
    clock.value += 0.2

    status = autonomy.tick()

    assert status["state"] == "SAFETY_STOP"
    assert status["linear"] == 0.0
    assert status["angular"] == 0.0


def test_rotate_escape_ignores_goal_heading_and_picks_open_side():
    sdk = FakeSdk(active=True)
    clock = Clock()
    settings = rotate_escape_settings()
    navigation = valid_sam()["navigation"] | {"heading_error_deg": -15.0}
    payload = blocked_sam_with_side_sector(
        side_sector=_side_sector(chosen="RIGHT", status="RIGHT_CLEAR"),
        navigation=navigation,
    )
    sam = LiveSam(payload, clock)
    autonomy = controller(sdk, sam, clock, settings=settings)

    clock.value += 0.2
    autonomy.tick()
    _confirm_direction(autonomy, sam, clock)
    clock.value += 0.2
    status = autonomy.tick()

    assert status["state"] == "ROTATE_PULSE_RIGHT"
    assert status["angular"] > 0.0


def test_rotate_escape_sdk_angular_sign_matches_existing_convention():
    sdk = FakeSdk(active=True)
    clock = Clock()
    settings = rotate_escape_settings()
    sam = LiveSam(blocked_sam_with_side_sector(side_sector=_side_sector(chosen="RIGHT")), clock)
    autonomy = controller(sdk, sam, clock, settings=settings)

    clock.value += 0.2
    autonomy.tick()
    _confirm_direction(autonomy, sam, clock)
    clock.value += 0.2
    status = autonomy.tick()

    assert status["state"] == "ROTATE_PULSE_RIGHT"
    assert status["angular"] > 0.0
    assert sdk.commands[-1].angular == pytest.approx(mission1_to_sdk_angular(status["angular"]))
    assert sdk.commands[-1].angular < 0.0


def test_rotate_escape_declines_without_side_sector_and_legacy_search_rotate_still_works():
    # Regression guard for the design decision to leave SEARCH_ROTATE
    # untouched: with no "side_sector" key in the planner status (as in the
    # pre-existing blocked_sam() fixture), ROTATE_ESCAPE must fully defer and
    # the legacy mechanism must behave exactly as before. enable_rotate_escape
    # is also left at its (now fail-closed) False default here on purpose.
    sdk = FakeSdk(active=True)
    clock = Clock()
    autonomy = controller(sdk, LiveSam(blocked_sam(), clock), clock)
    clock.value += 0.2

    status = autonomy.tick()

    assert status["state"] == "SEARCH_ROTATE"
    assert status["linear"] == 0.0
    assert status["angular"] > 0.0


# --- B: real telemetry-confirmed stationary state before rotating --------


def test_stop_confirm_blocks_pulse_while_telemetry_shows_motion():
    sdk = FakeSdk(active=True)
    clock = Clock()
    settings = rotate_escape_settings(rotate_escape_stop_timeout_sec=100.0)
    sam = LiveSam(
        blocked_sam_with_side_sector(side_sector=_side_sector(chosen="RIGHT"), stationary=False),
        clock,
    )
    autonomy = controller(sdk, sam, clock, settings=settings)

    for _ in range(10):
        clock.value += 0.2
        status = autonomy.tick()
        assert status["state"] == "STOP_CONFIRM"
        assert status["linear"] == 0.0
        assert status["angular"] == 0.0


def test_stop_confirm_repeated_telemetry_sample_does_not_advance_stationary_count():
    sdk = FakeSdk(active=True)
    clock = Clock()
    settings = rotate_escape_settings(
        rotate_escape_stationary_confirm_samples=3, rotate_escape_stop_timeout_sec=100.0
    )
    # Same telemetry timestamp reused across ticks -- polling the same
    # sample repeatedly must not look like 3 distinct confirmations.
    payload = blocked_sam_with_side_sector(side_sector=_side_sector(chosen="RIGHT"), telemetry_timestamp=5.0)
    sam = LiveSam(payload, clock)
    autonomy = controller(sdk, sam, clock, settings=settings)

    counts = []
    for _ in range(4):
        clock.value += 0.2
        status = autonomy.tick()
        assert status["state"] == "STOP_CONFIRM"
        counts.append(status["recovery"]["stationary_confirm_count"])

    assert counts == [1, 1, 1, 1]


def test_stop_confirm_requires_n_distinct_stationary_samples_before_pulse():
    sdk = FakeSdk(active=True)
    clock = Clock()
    settings = rotate_escape_settings(
        rotate_escape_stationary_confirm_samples=3, rotate_escape_stop_timeout_sec=100.0
    )
    sam = LiveSam(
        blocked_sam_with_side_sector(side_sector=_side_sector(chosen="RIGHT"), telemetry_timestamp=1.0), clock
    )
    autonomy = controller(sdk, sam, clock, settings=settings)

    clock.value += 0.2
    first = autonomy.tick()
    sam.base_payload = blocked_sam_with_side_sector(
        side_sector=_side_sector(chosen="RIGHT"), telemetry_timestamp=2.0
    )
    clock.value += 0.2
    second = autonomy.tick()
    sam.base_payload = blocked_sam_with_side_sector(
        side_sector=_side_sector(chosen="RIGHT"), telemetry_timestamp=3.0
    )
    clock.value += 0.2
    third = autonomy.tick()

    sam.base_payload = blocked_sam_with_side_sector(
        side_sector=_side_sector(chosen="RIGHT"), telemetry_timestamp=4.0
    )
    clock.value += 0.2
    fourth = autonomy.tick()

    assert first["state"] == second["state"] == third["state"] == "STOP_CONFIRM"
    assert first["recovery"]["stationary_confirm_count"] == 1
    assert second["recovery"]["stationary_confirm_count"] == 2
    assert third["recovery"]["stationary_confirm_count"] == 3
    assert fourth["state"] == "DIRECTION_CONFIRM"
    assert fourth["angular"] == 0.0


def test_stop_confirm_fails_closed_on_missing_stale_and_nonfinite_telemetry():
    sdk = FakeSdk(active=True)
    clock = Clock()
    settings = rotate_escape_settings(rotate_escape_stop_timeout_sec=100.0)

    for payload in (
        blocked_sam_with_side_sector(side_sector=_side_sector(chosen="RIGHT"), telemetry_timestamp=None),
        blocked_sam_with_side_sector(side_sector=_side_sector(chosen="RIGHT"), telemetry_valid=False),
        blocked_sam_with_side_sector(side_sector=_side_sector(chosen="RIGHT"), telemetry_age_sec=999.0),
        blocked_sam_with_side_sector(
            side_sector=_side_sector(chosen="RIGHT"),
            telemetry={"local_timestamp": 1.0, "sdk_timestamp": 1.0, "speed": float("nan"), "rpms": [0.0]},
        ),
    ):
        sdk = FakeSdk(active=True)
        clock = Clock()
        autonomy = controller(sdk, LiveSam(payload, clock), clock, settings=settings)
        clock.value += 0.2
        status = autonomy.tick()
        assert status["state"] == "STOP_CONFIRM"
        assert status["recovery"]["stationary_confirm_count"] == 0


def test_stop_confirm_times_out_to_safety_stop_without_stationary_telemetry():
    sdk = FakeSdk(active=True)
    clock = Clock()
    settings = rotate_escape_settings(rotate_escape_stop_timeout_sec=0.3)
    sam = LiveSam(
        blocked_sam_with_side_sector(side_sector=_side_sector(chosen="RIGHT"), stationary=False), clock
    )
    autonomy = controller(sdk, sam, clock, settings=settings)

    clock.value += 0.2
    first = autonomy.tick()
    clock.value += 0.5
    second = autonomy.tick()

    assert first["state"] == "STOP_CONFIRM"
    assert second["state"] == "SAFETY_STOP"
    assert second["linear"] == 0.0
    assert second["angular"] == 0.0


# --- Pulse/settle helpers --------------------------------------------------


def _advance_to_first_pulse(autonomy, sam, clock):
    """Confirm stop and one distinct post-stop direction frame."""

    clock.value += 0.2
    stop_confirm = autonomy.tick()
    current_frame = sam.base_payload.get("frame_index")
    assert isinstance(current_frame, int)
    _confirm_direction(
        autonomy,
        sam,
        clock,
        frame_index=current_frame + 1,
        telemetry_timestamp=float(current_frame + 1),
    )
    clock.value += 0.2
    pulse = autonomy.tick()
    assert pulse["state"] in {"ROTATE_PULSE_LEFT", "ROTATE_PULSE_RIGHT"}
    return stop_confirm, pulse


# --- C: fail-closed frame identity -----------------------------------------


def test_settle_repeated_frame_index_does_not_start_next_pulse():
    sdk = FakeSdk(active=True)
    clock = Clock()
    settings = rotate_escape_settings(rotate_escape_pulse_sec=0.2, rotate_escape_settle_sec=0.2)
    payload = blocked_sam_with_side_sector(side_sector=_side_sector(chosen="RIGHT"), frame_index=1)
    sam = LiveSam(payload, clock)
    autonomy = controller(sdk, sam, clock, settings=settings)

    _advance_to_first_pulse(autonomy, sam, clock)
    settle = _tick_until(autonomy, clock, lambda s: s["state"] == "ROTATE_SETTLE", dt=0.3)
    assert settle["recovery"]["pulse_count"] == 1

    # Same frame_index the whole time (no new SAM frame) -- settle must keep
    # waiting, not silently start a second pulse.
    for _ in range(5):
        clock.value += 0.2
        status = autonomy.tick()
        assert status["state"] == "ROTATE_SETTLE"
        assert status["angular"] == 0.0
        assert status["recovery"]["pulse_count"] == 1


def test_second_pulse_starts_only_on_new_blocked_frame():
    sdk = FakeSdk(active=True)
    clock = Clock()
    settings = rotate_escape_settings(rotate_escape_pulse_sec=0.2, rotate_escape_settle_sec=0.2)
    sam = LiveSam(
        blocked_sam_with_side_sector(side_sector=_side_sector(chosen="RIGHT"), frame_index=1), clock
    )
    autonomy = controller(sdk, sam, clock, settings=settings)

    _advance_to_first_pulse(autonomy, sam, clock)
    _tick_until(autonomy, clock, lambda s: s["state"] == "ROTATE_SETTLE", dt=0.3)

    sam.base_payload = blocked_sam_with_side_sector(
        side_sector=_side_sector(chosen="RIGHT"), frame_index=3, telemetry_timestamp=3.0
    )
    clock.value += 0.2
    settle_sees_new_frame = autonomy.tick()
    clock.value += 0.2
    second_pulse = autonomy.tick()

    assert settle_sees_new_frame["state"] == "ROTATE_SETTLE"
    assert second_pulse["state"] == "ROTATE_PULSE_RIGHT"
    assert second_pulse["recovery"]["pulse_count"] == 2


def test_missing_frame_index_fails_closed_during_post_rotate_replan():
    sdk = FakeSdk(active=True)
    clock = Clock()
    settings = rotate_escape_settings(rotate_escape_pulse_sec=0.2, rotate_escape_settle_sec=0.2)
    sam = LiveSam(
        blocked_sam_with_side_sector(side_sector=_side_sector(chosen="RIGHT"), frame_index=1), clock
    )
    autonomy = controller(sdk, sam, clock, settings=settings)

    _advance_to_first_pulse(autonomy, sam, clock)
    _tick_until(autonomy, clock, lambda s: s["state"] == "ROTATE_SETTLE", dt=0.3)

    # A real, distinct safe frame gets us into POST_ROTATE_REPLAN...
    sam.base_payload = safe_sam_with_side_sector(frame_index=3, telemetry_timestamp=3.0)
    clock.value += 0.2
    autonomy.tick()  # settle detects the safe path -> POST_ROTATE_REPLAN next tick

    # ...but once there, frame_index=None is malformed live input and must
    # fail closed before it can satisfy the post-rotate confirmation count.
    sam.base_payload = safe_sam_with_side_sector(frame_index=None, telemetry_timestamp=3.0)
    clock.value += 0.2
    status = autonomy.tick()

    assert status["state"] == "SAFETY_STOP"
    assert status["reason"] == "SAM-TP frame index is invalid"
    assert status["linear"] == status["angular"] == 0.0


def test_missing_frame_index_never_confirms_target_sequence():
    sdk = FakeSdk(active=True)
    clock = Clock()
    sam = FakeSam(valid_sam(frame_index=None))
    autonomy = controller(
        sdk,
        sam,
        clock,
        settings=Mission1ControlConfig(target_sequence_confirm_frames=3),
    )

    for _ in range(3):
        status = autonomy.tick()

    assert status["state"] == "SAFETY_STOP"
    assert status["reason"] == "SAM-TP frame index is invalid"
    assert status["linear"] == status["angular"] == 0.0


def test_frame_index_regression_during_settle_aborts():
    sdk = FakeSdk(active=True)
    clock = Clock()
    settings = rotate_escape_settings(rotate_escape_pulse_sec=0.2, rotate_escape_settle_sec=0.2)
    sam = LiveSam(
        blocked_sam_with_side_sector(side_sector=_side_sector(chosen="RIGHT"), frame_index=10), clock
    )
    autonomy = controller(sdk, sam, clock, settings=settings)

    _advance_to_first_pulse(autonomy, sam, clock)
    _tick_until(autonomy, clock, lambda s: s["state"] == "ROTATE_SETTLE", dt=0.3)

    sam.base_payload = blocked_sam_with_side_sector(
        side_sector=_side_sector(chosen="RIGHT"), frame_index=3, telemetry_timestamp=2.0
    )
    clock.value += 0.3
    status = autonomy.tick()

    assert status["state"] == "SAFETY_STOP"
    assert "regressed" in status["reason"]


def test_frame_index_regression_during_post_rotate_replan_aborts():
    sdk = FakeSdk(active=True)
    clock = Clock()
    settings = rotate_escape_settings(rotate_escape_pulse_sec=0.2, rotate_escape_settle_sec=0.2)
    sam = LiveSam(
        blocked_sam_with_side_sector(side_sector=_side_sector(chosen="RIGHT"), frame_index=1), clock
    )
    autonomy = controller(sdk, sam, clock, settings=settings)

    _advance_to_first_pulse(autonomy, sam, clock)
    _tick_until(autonomy, clock, lambda s: s["state"] == "ROTATE_SETTLE", dt=0.3)

    sam.base_payload = safe_sam_with_side_sector(frame_index=5, telemetry_timestamp=2.0)
    clock.value += 0.3
    autonomy.tick()  # settle detects the safe path -> POST_ROTATE_REPLAN next tick
    clock.value += 0.2
    first_post = autonomy.tick()  # frame 5 confirmed (count=1, last_frame_index=5)
    assert first_post["state"] == "POST_ROTATE_REPLAN"

    sam.base_payload = safe_sam_with_side_sector(frame_index=2, telemetry_timestamp=3.0)
    clock.value += 0.2
    status = autonomy.tick()

    assert status["state"] == "SAFETY_STOP"
    assert "regressed" in status["reason"]


def test_legacy_path_recovery_latch_repeated_frame_does_not_advance():
    # The pre-existing (non-ROTATE_ESCAPE) path-recovery latch must require
    # distinct source observations rather than controller-loop ticks.
    sdk = FakeSdk(active=True)
    clock = Clock()
    good = valid_sam(local_path_selected_heading_deg=0.0, frame_index=10)
    stopped_payload = valid_sam(
        frame_index=11,
        planner={
            "switch_stop_required": True,
            "switch_reason": "stop_unsafe_candidate_switch_pending",
            "near_field_safe": True,
        },
        local_path_selected_heading_deg=None,
    )
    sam = FakeSam(good)
    autonomy = controller(
        sdk, sam, clock, settings=Mission1ControlConfig(path_recovery_confirm_frames=3, maximum_sam_age_sec=10.0)
    )
    clock.value += 0.2
    assert autonomy.tick()["state"] == "DRIVING"
    sam.payload = stopped_payload
    clock.value += 0.2
    assert autonomy.tick()["state"] == "SAFETY_STOP"

    # The same frame on every subsequent tick must never satisfy the
    # 3-distinct-frame confirmation just by ticking.
    repeated_frame = valid_sam(local_path_selected_heading_deg=0.0, frame_index=12)
    for _ in range(5):
        sam.payload = repeated_frame
        clock.value += 0.2
        status = autonomy.tick()
        assert status["state"] == "SAFETY_STOP"
        assert "confirming safe path recovery (1/3)" in status["reason"]


# --- D: safe-path check before side-sector-degradation check ---------------


def test_settle_completes_recovery_when_safe_path_appears_even_if_side_chosen_ambiguous():
    sdk = FakeSdk(active=True)
    clock = Clock()
    settings = rotate_escape_settings(rotate_escape_pulse_sec=0.2, rotate_escape_settle_sec=0.2)
    sam = LiveSam(
        blocked_sam_with_side_sector(side_sector=_side_sector(chosen="RIGHT"), frame_index=1), clock
    )
    autonomy = controller(sdk, sam, clock, settings=settings)

    _advance_to_first_pulse(autonomy, sam, clock)
    _tick_until(autonomy, clock, lambda s: s["state"] == "ROTATE_SETTLE", dt=0.3)

    # The open area moved to the center: a full safe path now exists, but
    # the side sector itself reports AMBIGUOUS (neither side clearly best).
    ambiguous_but_safe = safe_sam_with_side_sector(
        side_sector=_side_sector(
            chosen=None, status="AMBIGUOUS", left_viable=True, right_viable=True,
            left_composite=0.6, right_composite=0.62, margin=0.02,
        ),
        frame_index=3,
        telemetry_timestamp=3.0,
    )
    sam.base_payload = ambiguous_but_safe
    clock.value += 0.2
    settle_result = autonomy.tick()

    assert settle_result["state"] == "ROTATE_SETTLE"
    assert settle_result["angular"] == 0.0
    assert settle_result["recovery"]["maneuver_phase"] == "ROTATE_SETTLE"

    clock.value += 0.2
    next_tick = autonomy.tick()
    assert next_tick["state"] == "POST_ROTATE_REPLAN"
    assert next_tick["angular"] == 0.0


def test_settle_aborts_when_chosen_side_itself_becomes_unsafe_without_safe_path():
    sdk = FakeSdk(active=True)
    clock = Clock()
    settings = rotate_escape_settings(rotate_escape_pulse_sec=0.2, rotate_escape_settle_sec=0.2)
    sam = LiveSam(
        blocked_sam_with_side_sector(side_sector=_side_sector(chosen="RIGHT"), frame_index=1), clock
    )
    autonomy = controller(sdk, sam, clock, settings=settings)

    _advance_to_first_pulse(autonomy, sam, clock)
    _tick_until(autonomy, clock, lambda s: s["state"] == "ROTATE_SETTLE", dt=0.3)

    sam.base_payload = blocked_sam_with_side_sector(
        side_sector=_side_sector(
            chosen=None, status="AMBIGUOUS", left_viable=False, right_viable=False,
            left_composite=0.10, right_composite=0.12,
        ),
        frame_index=3,
        telemetry_timestamp=3.0,
    )
    clock.value += 0.2
    status = autonomy.tick()

    assert status["state"] == "SAFETY_STOP"
    assert status["angular"] == 0.0


def test_settle_aborts_when_opposite_side_becomes_clearly_chosen_without_safe_path():
    sdk = FakeSdk(active=True)
    clock = Clock()
    settings = rotate_escape_settings(rotate_escape_pulse_sec=0.2, rotate_escape_settle_sec=0.2)
    sam = LiveSam(
        blocked_sam_with_side_sector(side_sector=_side_sector(chosen="RIGHT"), frame_index=1), clock
    )
    autonomy = controller(sdk, sam, clock, settings=settings)

    _advance_to_first_pulse(autonomy, sam, clock)
    _tick_until(autonomy, clock, lambda s: s["state"] == "ROTATE_SETTLE", dt=0.3)

    # RIGHT (the direction already in progress) is still independently
    # viable -- this must hit the "opposite clearly chosen" branch, not the
    # "chosen side itself unsafe" branch (covered by a separate test).
    sam.base_payload = blocked_sam_with_side_sector(
        side_sector=_side_sector(
            chosen="LEFT", status="LEFT_CLEAR", left_viable=True, right_viable=True,
            left_composite=0.95, right_composite=0.55, margin=0.12,
        ),
        frame_index=3,
        telemetry_timestamp=3.0,
    )
    clock.value += 0.3
    status = autonomy.tick()

    assert status["state"] == "SAFETY_STOP"
    assert status["angular"] == 0.0
    assert "flipped" in status["reason"]


# --- E: pulse/settle structure ---------------------------------------------


def test_pulse_then_settle_sends_exact_zero_command():
    sdk = FakeSdk(active=True)
    clock = Clock()
    settings = rotate_escape_settings(rotate_escape_pulse_sec=0.2, rotate_escape_settle_sec=0.3)
    sam = LiveSam(
        blocked_sam_with_side_sector(side_sector=_side_sector(chosen="RIGHT"), frame_index=1), clock
    )
    autonomy = controller(sdk, sam, clock, settings=settings)

    _, pulse = _advance_to_first_pulse(autonomy, sam, clock)
    assert pulse["state"] == "ROTATE_PULSE_RIGHT"
    assert pulse["angular"] != 0.0

    settle = _tick_until(autonomy, clock, lambda s: s["state"] == "ROTATE_SETTLE", dt=0.3)
    assert settle["linear"] == 0.0
    assert settle["angular"] == 0.0
    assert sdk.commands[-1].linear == 0.0
    assert sdk.commands[-1].angular == 0.0


def test_max_pulses_exceeded_leads_to_safety_stop():
    sdk = FakeSdk(active=True)
    clock = Clock()
    settings = rotate_escape_settings(
        rotate_escape_pulse_sec=0.1, rotate_escape_settle_sec=0.1, rotate_escape_max_pulses=2,
        rotate_escape_max_total_sec=100.0,
    )
    frame_index = 1
    sam = LiveSam(
        blocked_sam_with_side_sector(side_sector=_side_sector(chosen="RIGHT"), frame_index=frame_index), clock
    )
    autonomy = controller(sdk, sam, clock, settings=settings)

    clock.value += 0.2
    autonomy.tick()  # STOP_CONFIRM
    final_state = None
    for _ in range(20):
        frame_index += 1
        sam.base_payload = blocked_sam_with_side_sector(
            side_sector=_side_sector(chosen="RIGHT"), frame_index=frame_index, telemetry_timestamp=float(frame_index)
        )
        clock.value += 0.2
        status = autonomy.tick()
        final_state = status["state"]
        if final_state == "SAFETY_STOP":
            break

    assert final_state == "SAFETY_STOP"
    assert "max pulse" in status["reason"]


def test_rotate_escape_exceeding_max_total_time_leads_to_safety_stop():
    sdk = FakeSdk(active=True)
    clock = Clock()
    settings = rotate_escape_settings(rotate_escape_max_total_sec=1.0)
    sam = LiveSam(
        blocked_sam_with_side_sector(side_sector=_side_sector(chosen="RIGHT"), frame_index=1), clock
    )
    autonomy = controller(sdk, sam, clock, settings=settings)

    _, first = _advance_to_first_pulse(autonomy, sam, clock)
    clock.value += 1.0
    second = autonomy.tick()

    assert first["state"] == "ROTATE_PULSE_RIGHT"
    assert second["state"] == "SAFETY_STOP"
    assert second["angular"] == 0.0
    assert "max total" in second["reason"]


def test_pulse_angular_respects_slew_limit_and_stop_is_exact_zero():
    sdk = FakeSdk(active=True)
    clock = Clock()
    settings = rotate_escape_settings(rotate_escape_pulse_sec=1.0, max_angular=0.15)
    sam = LiveSam(
        blocked_sam_with_side_sector(side_sector=_side_sector(chosen="RIGHT"), frame_index=1), clock
    )
    filter_config = {
        "control": {
            "linear_min": 0.0,
            "linear_max": 0.15,
            "angular_min": -0.15,
            "angular_max": 0.15,
            "command_smoothing_alpha": 0.0,
            "max_linear_delta_per_sec": 10.0,
            "max_angular_delta_per_sec": 0.3,  # slow slew: 0.06 per 0.2s tick
        }
    }
    autonomy = Mission1Autonomy(
        sdk, sam, settings, filter_config, live_control_enabled=True, clock=clock, monotonic=clock
    )

    _, first_pulse = _advance_to_first_pulse(autonomy, sam, clock)

    assert first_pulse["state"] == "ROTATE_PULSE_RIGHT"
    # Slew-limited: shouldn't jump straight to the full 0.15 in one 0.2s tick.
    assert 0.0 < first_pulse["angular"] < 0.15
    assert first_pulse["linear"] == 0.0


def test_rotate_escape_aborts_first_pulse_when_actuator_does_not_respond():
    sdk = FakeSdk(active=True)
    clock = Clock()
    settings = rotate_escape_settings(
        rotate_escape_pulse_sec=0.3,
        rotate_escape_motion_response_timeout_sec=0.6,
        rotate_escape_motion_rpm_threshold=1.0,
        rotate_escape_motion_heading_delta_deg=1.5,
        rotate_escape_max_total_sec=10.0,
    )
    sam = LiveSam(
        blocked_sam_with_side_sector(
            side_sector=_side_sector(chosen="RIGHT"), frame_index=1
        ),
        clock,
    )
    autonomy = controller(sdk, sam, clock, settings=settings)

    _, first_pulse = _advance_to_first_pulse(autonomy, sam, clock)
    clock.value += 0.3
    still_waiting = autonomy.tick()
    sam.base_payload = blocked_sam_with_side_sector(
        side_sector=_side_sector(chosen="RIGHT"),
        frame_index=3,
        telemetry_timestamp=3.0,
    )
    clock.value += 0.3
    settling = autonomy.tick()
    clock.value += 0.4
    failed = autonomy.tick()

    assert first_pulse["state"] == "ROTATE_PULSE_RIGHT"
    assert still_waiting["state"] == settling["state"] == "ROTATE_SETTLE"
    assert still_waiting["angular"] == settling["angular"] == 0.0
    assert settling["recovery"]["pulse_motion_observed"] is False
    assert failed["state"] == "SAFETY_STOP"
    assert failed["linear"] == failed["angular"] == 0.0
    assert "actuator did not respond" in failed["reason"]
    assert "max_rpm=0.00" in failed["reason"]


def test_rotate_escape_accepts_fresh_rotation_rpm_as_motion_response():
    sdk = FakeSdk(active=True)
    clock = Clock()
    settings = rotate_escape_settings(
        rotate_escape_pulse_sec=0.3,
        rotate_escape_motion_response_timeout_sec=0.8,
        rotate_escape_motion_rpm_threshold=1.0,
        rotate_escape_motion_heading_delta_deg=1.5,
    )
    sam = LiveSam(
        blocked_sam_with_side_sector(
            side_sector=_side_sector(chosen="RIGHT"), frame_index=1
        ),
        clock,
    )
    autonomy = controller(sdk, sam, clock, settings=settings)

    _advance_to_first_pulse(autonomy, sam, clock)
    moving_telemetry = _telemetry(3.0)
    moving_telemetry["rpms"] = [2.0, -2.0, 2.0, -2.0]
    sam.base_payload = blocked_sam_with_side_sector(
        side_sector=_side_sector(chosen="RIGHT"),
        frame_index=3,
        telemetry=moving_telemetry,
    )
    clock.value += 0.3
    settled = autonomy.tick()

    assert settled["state"] == "ROTATE_SETTLE"
    assert settled["angular"] == 0.0
    assert settled["recovery"]["pulse_motion_observed"] is True
    assert settled["recovery"]["pulse_max_abs_rpm"] == 2.0


# --- F: cooldown blocks all re-entry directions -----------------------------


def test_cooldown_blocks_same_direction_reentry_after_abort():
    sdk = FakeSdk(active=True)
    clock = Clock()
    settings = rotate_escape_settings(rotate_escape_max_total_sec=0.3, rotate_escape_cooldown_sec=5.0)
    sam = LiveSam(
        blocked_sam_with_side_sector(side_sector=_side_sector(chosen="RIGHT"), frame_index=1), clock
    )
    autonomy = controller(sdk, sam, clock, settings=settings)

    clock.value += 0.2
    autonomy.tick()
    clock.value += 0.2
    autonomy.tick()
    clock.value += 0.5
    timed_out = autonomy.tick()
    assert timed_out["state"] == "SAFETY_STOP"

    sam.base_payload = blocked_sam_with_side_sector(
        side_sector=_side_sector(chosen="RIGHT"), frame_index=10, telemetry_timestamp=10.0
    )
    clock.value += 0.2
    status = autonomy.tick()

    assert status["state"] == "RECOVERY_COOLDOWN"
    assert status["linear"] == 0.0
    assert status["angular"] == 0.0
    assert status["recovery"]["cooldown_remaining_sec"] > 0.0


def test_cooldown_blocks_opposite_direction_reentry_after_abort():
    sdk = FakeSdk(active=True)
    clock = Clock()
    settings = rotate_escape_settings(rotate_escape_max_total_sec=0.3, rotate_escape_cooldown_sec=5.0)
    sam = LiveSam(
        blocked_sam_with_side_sector(side_sector=_side_sector(chosen="RIGHT"), frame_index=1), clock
    )
    autonomy = controller(sdk, sam, clock, settings=settings)

    clock.value += 0.2
    autonomy.tick()
    clock.value += 0.2
    autonomy.tick()
    clock.value += 0.5
    timed_out = autonomy.tick()
    assert timed_out["state"] == "SAFETY_STOP"

    sam.base_payload = blocked_sam_with_side_sector(
        side_sector=_side_sector(
            chosen="LEFT", status="LEFT_CLEAR", left_viable=True, right_viable=False,
            left_composite=0.9, right_composite=0.05,
        ),
        frame_index=10,
        telemetry_timestamp=10.0,
    )
    clock.value += 0.2
    status = autonomy.tick()

    assert status["state"] == "RECOVERY_COOLDOWN"
    assert status["angular"] == 0.0


def test_cooldown_allows_normal_driving_once_a_safe_path_is_confirmed():
    sdk = FakeSdk(active=True)
    clock = Clock()
    settings = rotate_escape_settings(rotate_escape_max_total_sec=0.3, rotate_escape_cooldown_sec=5.0)
    sam = LiveSam(
        blocked_sam_with_side_sector(side_sector=_side_sector(chosen="RIGHT"), frame_index=1), clock
    )
    autonomy = controller(sdk, sam, clock, settings=settings)

    clock.value += 0.2
    autonomy.tick()
    clock.value += 0.2
    autonomy.tick()
    clock.value += 0.5
    assert autonomy.tick()["state"] == "SAFETY_STOP"

    # Still inside the cooldown window, but the path is now genuinely safe --
    # cooldown must only gate *recovery* re-entry, not ordinary driving. The
    # legacy path_stop_latched gate (separate from cooldown) also needs a
    # real, distinct frame_index to release now that it's fail-closed on
    # missing frame_index too.
    sam.base_payload = valid_sam(local_path_selected_heading_deg=0.0, frame_index=100)
    clock.value += 0.2
    status = autonomy.tick()

    assert status["state"] == "DRIVING"


def test_cooldown_expires_and_allows_new_recovery():
    sdk = FakeSdk(active=True)
    clock = Clock()
    settings = rotate_escape_settings(rotate_escape_max_total_sec=0.3, rotate_escape_cooldown_sec=1.0)
    sam = LiveSam(
        blocked_sam_with_side_sector(side_sector=_side_sector(chosen="RIGHT"), frame_index=1), clock
    )
    autonomy = controller(sdk, sam, clock, settings=settings)

    clock.value += 0.2
    autonomy.tick()
    clock.value += 0.2
    autonomy.tick()
    clock.value += 0.5
    assert autonomy.tick()["state"] == "SAFETY_STOP"

    clock.value += 2.0  # cooldown (1.0s) has expired
    sam.base_payload = blocked_sam_with_side_sector(
        side_sector=_side_sector(chosen="RIGHT"), frame_index=20, telemetry_timestamp=20.0
    )
    status = autonomy.tick()

    assert status["state"] == "STOP_CONFIRM"


# --- H: strengthened POST_ROTATE_REPLAN contract + full status metadata ----


def test_post_rotate_replan_requires_three_distinct_safe_frames_before_driving():
    sdk = FakeSdk(active=True)
    clock = Clock()
    settings = rotate_escape_settings(
        rotate_escape_pulse_sec=0.2,
        rotate_escape_settle_sec=0.2,
        minimum_linear=0.12,
        base_linear=0.12,
        max_linear=0.12,
    )
    sam = LiveSam(
        blocked_sam_with_side_sector(side_sector=_side_sector(chosen="RIGHT"), frame_index=1), clock
    )
    autonomy = controller(sdk, sam, clock, settings=settings)

    _advance_to_first_pulse(autonomy, sam, clock)
    _tick_until(autonomy, clock, lambda s: s["state"] == "ROTATE_SETTLE", dt=0.3)

    sam.base_payload = safe_sam_with_side_sector(frame_index=3, telemetry_timestamp=3.0)
    clock.value += 0.2
    autonomy.tick()  # settle detects safe path -> POST_ROTATE_REPLAN next tick

    for fi in (4, 5, 6):
        sam.base_payload = safe_sam_with_side_sector(frame_index=fi, telemetry_timestamp=float(fi))
        clock.value += 0.2
        status = autonomy.tick()
        assert status["linear"] == 0.0

    sam.base_payload = safe_sam_with_side_sector(frame_index=7, telemetry_timestamp=7.0)
    clock.value += 0.2
    driving = autonomy.tick()

    assert driving["state"] == "DRIVING"
    assert driving["linear"] >= 0.12


def test_post_rotate_replan_aborts_if_path_stops_being_safe():
    sdk = FakeSdk(active=True)
    clock = Clock()
    settings = rotate_escape_settings(rotate_escape_pulse_sec=0.2, rotate_escape_settle_sec=0.2)
    sam = LiveSam(
        blocked_sam_with_side_sector(side_sector=_side_sector(chosen="RIGHT"), frame_index=1), clock
    )
    autonomy = controller(sdk, sam, clock, settings=settings)

    _advance_to_first_pulse(autonomy, sam, clock)
    _tick_until(autonomy, clock, lambda s: s["state"] == "ROTATE_SETTLE", dt=0.3)

    sam.base_payload = safe_sam_with_side_sector(frame_index=3, telemetry_timestamp=3.0)
    clock.value += 0.2
    autonomy.tick()

    sam.base_payload = blocked_sam_with_side_sector(
        side_sector=_side_sector(chosen="RIGHT"), frame_index=4, telemetry_timestamp=4.0
    )
    clock.value += 0.2
    status = autonomy.tick()

    assert status["state"] == "SAFETY_STOP"
    assert "lost the safe path" in status["reason"]


def test_rotate_escape_status_exposes_recovery_metadata_across_phases():
    sdk = FakeSdk(active=True)
    clock = Clock()
    settings = rotate_escape_settings(rotate_escape_pulse_sec=0.2, rotate_escape_settle_sec=0.2)
    sam = LiveSam(
        blocked_sam_with_side_sector(side_sector=_side_sector(chosen="RIGHT"), frame_index=1), clock
    )
    autonomy = controller(sdk, sam, clock, settings=settings)

    clock.value += 0.2
    stop_confirm = autonomy.tick()
    assert stop_confirm["recovery"]["maneuver_type"] == "ROTATE_THEN_REPLAN"
    assert stop_confirm["recovery"]["maneuver_phase"] == "STOP_CONFIRM"
    assert stop_confirm["recovery"]["maneuver_direction"] == "RIGHT"
    assert stop_confirm["recovery"]["side_sector"]["chosen"] == "RIGHT"

    _confirm_direction(autonomy, sam, clock)
    clock.value += 0.3
    pulse = autonomy.tick()
    assert pulse["recovery"]["maneuver_phase"] == "ROTATE_PULSE"
    assert pulse["recovery"]["recovery_elapsed_sec"] is not None
    assert pulse["recovery"]["rotation_reason"]
    assert pulse["recovery"]["pulse_count"] == 1
    assert pulse["recovery"]["pulse_count_max"] == settings.rotate_escape_max_pulses

    settle = _tick_until(autonomy, clock, lambda s: s["state"] == "ROTATE_SETTLE", dt=0.3)
    assert settle["recovery"]["maneuver_phase"] == "ROTATE_SETTLE"

    sam.base_payload = safe_sam_with_side_sector(frame_index=3, telemetry_timestamp=3.0)
    clock.value += 0.3
    autonomy.tick()

    sam.base_payload = safe_sam_with_side_sector(frame_index=4, telemetry_timestamp=4.0)
    clock.value += 0.2
    post = autonomy.tick()

    assert post["recovery"]["maneuver_phase"] == "POST_ROTATE_REPLAN"
    assert post["recovery"]["safe_frame_confirm_count"] == 1
    assert post["recovery"]["safe_frame_confirm_required"] == settings.rotate_escape_safe_frame_confirm_count


def test_first_pulse_requires_two_distinct_post_stop_direction_frames():
    sdk = FakeSdk(active=True)
    clock = Clock()
    settings = rotate_escape_settings(rotate_escape_direction_confirm_frames=2)
    sam = LiveSam(blocked_sam_with_side_sector(frame_index=10), clock)
    autonomy = controller(sdk, sam, clock, settings=settings)

    clock.value += 0.2
    assert autonomy.tick()["state"] == "STOP_CONFIRM"

    sam.base_payload = blocked_sam_with_side_sector(frame_index=11, telemetry_timestamp=11.0)
    clock.value += 0.2
    first = autonomy.tick()
    clock.value += 0.2
    repeated = autonomy.tick()

    sam.base_payload = blocked_sam_with_side_sector(frame_index=12, telemetry_timestamp=12.0)
    clock.value += 0.2
    second = autonomy.tick()
    clock.value += 0.2
    pulse = autonomy.tick()

    assert first["state"] == repeated["state"] == second["state"] == "DIRECTION_CONFIRM"
    assert first["recovery"]["direction_confirm_count"] == 1
    assert repeated["recovery"]["direction_confirm_count"] == 1
    assert second["recovery"]["direction_confirm_count"] == 2
    assert pulse["state"] == "ROTATE_PULSE_RIGHT"


def test_pulse_deadline_tick_sends_zero_instead_of_one_more_rotate_command():
    sdk = FakeSdk(active=True)
    clock = Clock()
    settings = rotate_escape_settings(rotate_escape_pulse_sec=0.5)
    sam = LiveSam(blocked_sam_with_side_sector(frame_index=1), clock)
    autonomy = controller(sdk, sam, clock, settings=settings)
    _, pulse = _advance_to_first_pulse(autonomy, sam, clock)
    assert pulse["angular"] > 0.0

    clock.value += 0.5
    deadline = autonomy.tick()

    assert deadline["state"] == "ROTATE_SETTLE"
    assert deadline["linear"] == 0.0
    assert deadline["angular"] == 0.0
    assert sdk.commands[-1].linear == 0.0
    assert sdk.commands[-1].angular == 0.0


@pytest.mark.parametrize(
    ("planner_override", "sam_override"),
    [
        ({"trajectory_valid": False}, {}),
        ({"using_held_plan": True}, {}),
        ({"near_field_safe": None}, {}),
        ({"planner_confidence": None}, {}),
        ({"plan_age_sec": None}, {}),
        ({"selected_candidate_index": None}, {}),
        ({}, {"path_valid": False}),
    ],
)
def test_post_rotate_strict_gate_rejects_incomplete_or_held_paths(
    planner_override, sam_override
):
    sdk = FakeSdk(active=True)
    clock = Clock()
    autonomy = controller(sdk, FakeSam(valid_sam()), clock)
    payload = safe_sam_with_side_sector()
    payload["planner"] = payload["planner"] | planner_override
    payload.update(sam_override)

    assert autonomy._validate_post_rotate_path(payload) is not None
