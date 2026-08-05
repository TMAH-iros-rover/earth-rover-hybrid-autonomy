from __future__ import annotations

import time

import pytest

from scripts.teleop_dashboard import (
    LEFT_KEYS,
    UP_KEYS,
    SharedState,
    control_snapshot,
    handle_key,
)


def test_teleop_starts_disarmed_and_ignores_motion_keys() -> None:
    state = SharedState(10)

    assert handle_key(state, ord("w"))

    assert not state.armed
    assert state.linear == 0.0
    assert state.angular == 0.0


def test_arm_then_motion_uses_bounded_low_speed_without_accumulating() -> None:
    state = SharedState(10)

    handle_key(state, ord("e"))
    handle_key(state, ord("w"))
    handle_key(state, ord("w"))
    handle_key(state, ord("a"))

    assert state.armed
    assert state.linear == pytest.approx(0.20)
    assert state.angular == pytest.approx(0.35)
    assert state.last_key == "A/left"
    assert state.key_event_count == 4


def test_extended_arrow_keys_control_motion_when_armed() -> None:
    state = SharedState(10)
    handle_key(state, ord("e"))

    handle_key(state, next(iter(UP_KEYS)))
    handle_key(state, next(iter(LEFT_KEYS)))

    assert state.linear == pytest.approx(0.20)
    assert state.angular == pytest.approx(0.35)


def test_deadman_expires_linear_and_angular_independently() -> None:
    state = SharedState(10, deadman_timeout=0.5)
    state.armed = True
    state.linear = 0.15
    state.angular = 0.25
    state.last_linear_key_time = 10.0
    state.last_angular_key_time = 10.4

    should_send, linear, angular, _lamp = control_snapshot(state, 10.6)

    assert should_send
    assert linear == 0.0
    assert angular == pytest.approx(0.25)


def test_space_disarms_and_requests_repeated_stop_commands() -> None:
    state = SharedState(10)
    handle_key(state, ord("e"))
    handle_key(state, ord("s"))
    handle_key(state, ord("d"))

    handle_key(state, ord(" "))

    assert not state.armed
    assert state.linear == 0.0
    assert state.angular == 0.0
    snapshots = [control_snapshot(state, time.monotonic()) for _ in range(4)]
    assert [item[0] for item in snapshots] == [True, True, True, False]
    assert all(item[1:3] == (0.0, 0.0) for item in snapshots)


def test_q_disarms_before_requesting_exit() -> None:
    state = SharedState(10)
    handle_key(state, ord("e"))
    handle_key(state, ord("w"))

    assert not handle_key(state, ord("q"))
    assert not state.armed
    assert state.stop_burst_remaining == 3


def test_speed_adjustment_cannot_exceed_conservative_teleop_limits() -> None:
    state = SharedState(10)

    for _ in range(20):
        handle_key(state, ord("+"))

    assert state.drive_speed == pytest.approx(0.25)
    assert state.turn_speed == pytest.approx(0.40)
