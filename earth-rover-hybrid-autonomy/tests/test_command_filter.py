from earth_rover.control.command_filter import CommandFilter
from earth_rover.core.types import ControlCommand


def config(alpha=0.0, max_linear=100.0, max_angular=100.0, reverse=False):
    return {
        "control": {
            "linear_min": -0.25,
            "linear_max": 0.35,
            "angular_min": -0.70,
            "angular_max": 0.70,
            "command_smoothing_alpha": alpha,
            "max_linear_delta_per_sec": max_linear,
            "max_angular_delta_per_sec": max_angular,
            "reverse_angular_sign_slowdown": reverse,
            "angular_deadband": 0.03,
        }
    }


def test_clamps_linear_and_angular():
    filt = CommandFilter(config())
    cmd = filt.apply(ControlCommand(10.0, -10.0), dt=1.0, frame_is_stale=False, data_is_stale=False)
    assert cmd.linear == 0.35
    assert cmd.angular == -0.70


def test_stale_frame_stops():
    filt = CommandFilter(config())
    cmd = filt.apply(ControlCommand(0.2, 0.2), dt=1.0, frame_is_stale=True, data_is_stale=False)
    assert cmd.linear == 0.0
    assert cmd.angular == 0.0
    assert cmd.mode == "STALE_DATA_STOP"


def test_smoothing_is_applied():
    filt = CommandFilter(config(alpha=0.5))
    cmd = filt.apply(ControlCommand(0.2, 0.4), dt=1.0, frame_is_stale=False, data_is_stale=False)
    assert cmd.linear == 0.1
    assert cmd.angular == 0.2


def test_rate_limit_is_applied():
    filt = CommandFilter(config(alpha=0.0, max_linear=0.1, max_angular=0.2))
    cmd = filt.apply(ControlCommand(0.35, 0.7), dt=1.0, frame_is_stale=False, data_is_stale=False)
    assert cmd.linear == 0.1
    assert cmd.angular == 0.2


def test_angular_sign_reversal_passes_through_zero_when_enabled():
    filt = CommandFilter(config(reverse=True))
    first = filt.apply(ControlCommand(0.1, 0.4), dt=1.0, frame_is_stale=False, data_is_stale=False)
    second = filt.apply(ControlCommand(0.1, -0.4), dt=1.0, frame_is_stale=False, data_is_stale=False)

    assert first.angular == 0.4
    assert second.angular == 0.0


def test_stale_stop_bypasses_sign_reversal_smoothing():
    filt = CommandFilter(config(reverse=True))
    filt.apply(ControlCommand(0.1, 0.4), dt=1.0, frame_is_stale=False, data_is_stale=False)
    stopped = filt.apply(ControlCommand(0.1, -0.4), dt=1.0, frame_is_stale=True, data_is_stale=False)

    assert stopped.linear == 0.0
    assert stopped.angular == 0.0
