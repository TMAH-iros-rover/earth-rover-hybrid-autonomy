from earth_rover.core.types import ControlCommand, RoverData
from earth_rover.safety.stuck_detector import StuckDetector


def rover(timestamp, speed, rpms=None):
    return RoverData(
        timestamp=timestamp,
        latitude=37.0,
        longitude=127.0,
        orientation=0.0,
        speed=speed,
        rpms=rpms if rpms is not None else [0.0, 0.0],
        battery=None,
        signal_level=None,
        gps_signal=None,
        raw={},
    )


def config():
    return {"stuck": {"enabled": True, "stuck_time_sec": 3.0, "min_speed_for_not_stuck": 0.03, "low_rpm_threshold": 1.0}}


def test_forward_low_speed_low_rpm_over_time_is_stuck():
    detector = StuckDetector(config())
    command = ControlCommand(0.2, 0.0)
    stuck, _ = detector.update(rover(0.0, 0.0), command)
    assert stuck is False
    stuck, _ = detector.update(rover(3.1, 0.0), command)
    assert stuck is True


def test_speed_sufficient_is_not_stuck():
    detector = StuckDetector(config())
    stuck, _ = detector.update(rover(5.0, 0.1), ControlCommand(0.2, 0.0))
    assert stuck is False


def test_reverse_or_stop_command_is_not_stuck():
    detector = StuckDetector(config())
    assert detector.update(rover(5.0, 0.0), ControlCommand(-0.1, 0.0))[0] is False
    assert detector.update(rover(6.0, 0.0), ControlCommand(0.0, 0.0))[0] is False

