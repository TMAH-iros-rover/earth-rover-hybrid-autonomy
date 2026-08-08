import math

from earth_rover.control.hybrid_controller import HybridReactiveController
from earth_rover.core.types import CandidateDirection, ControlCommand, PerceptionResult


def config():
    return {
        "control": {
            "base_linear": 0.22,
            "heading_kp": 0.70,
            "heading_kd": 0.05,
            "local_goal_kp": 0.35,
            "slow_heading_error_deg": 35,
            "rotate_in_place_error_deg": 90,
            "rotate_angular": 0.55,
        },
        "perception": {"obstacle_stop_threshold": 0.75},
    }


def candidate(name="CENTER"):
    return CandidateDirection(name, 0.0, 0.9, 0.1, 0.0, 1.0)


def perception(obstacle=0.1):
    return PerceptionResult(0.8, 0.9, 0.8, obstacle, 0.8, {})


def test_emergency_stop_zeroes_command():
    controller = HybridReactiveController(config())
    cmd = controller.compute(0.0, candidate(), perception(), True, None, 0.2)
    assert cmd.linear == 0.0
    assert cmd.angular == 0.0
    assert cmd.mode == "EMERGENCY_STOP"


def test_recovery_command_has_priority():
    controller = HybridReactiveController(config())
    recovery = ControlCommand(-0.18, 0.0, mode="RECOVERY_REVERSE")
    cmd = controller.compute(0.0, candidate(), perception(), False, recovery, 0.2)
    assert cmd is recovery


def test_high_obstacle_confidence_stops():
    controller = HybridReactiveController(config())
    cmd = controller.compute(0.0, candidate(), perception(0.9), False, None, 0.2)
    assert cmd.linear == 0.0
    assert cmd.angular == 0.0
    assert cmd.mode == "OBSTACLE_STOP"


def test_large_heading_error_rotates_in_place():
    controller = HybridReactiveController(config())
    cmd = controller.compute(math.radians(91), candidate(), perception(), False, None, 0.2)
    assert cmd.linear == 0.0
    assert cmd.angular > 0.0
    assert cmd.mode == "ROTATE_IN_PLACE"


def test_normal_mode_moves_forward():
    controller = HybridReactiveController(config())
    cmd = controller.compute(0.0, candidate(), perception(), False, None, 0.2)
    assert cmd.linear > 0.0
    assert cmd.mode == "NORMAL_DRIVE"

