import math

from earth_rover.navigation.gps_utils import (
    bearing_deg,
    haversine_distance_m,
    normalize_angle_deg,
    normalize_angle_rad,
)


def test_same_position_distance_is_zero():
    assert haversine_distance_m(37.0, 127.0, 37.0, 127.0) < 1e-6


def test_north_bearing_is_zero_degrees():
    assert bearing_deg(0.0, 0.0, 1.0, 0.0) == pytest_approx(0.0, abs=0.5)


def test_east_bearing_is_ninety_degrees():
    assert bearing_deg(0.0, 0.0, 0.0, 1.0) == pytest_approx(90.0, abs=0.5)


def test_angle_normalization():
    assert normalize_angle_deg(181.0) == pytest_approx(-179.0)
    assert normalize_angle_deg(-181.0) == pytest_approx(179.0)
    assert normalize_angle_deg(90.0) == pytest_approx(90.0)
    assert normalize_angle_deg(270.0) == pytest_approx(-90.0)
    assert normalize_angle_deg(10.0 - 350.0) == pytest_approx(20.0)
    assert normalize_angle_deg(350.0 - 10.0) == pytest_approx(-20.0)
    assert normalize_angle_rad(3.5) == pytest_approx(3.5 - 2 * math.pi)


def test_compass_heading_error_physical_direction_examples():
    # heading_error = target_bearing - current_heading, normalized.
    # With compass angles, positive is clockwise, i.e. physical right.
    assert normalize_angle_deg(90.0 - 0.0) == pytest_approx(90.0)
    assert normalize_angle_deg(270.0 - 0.0) == pytest_approx(-90.0)
    assert normalize_angle_deg(0.0 - 90.0) == pytest_approx(-90.0)
    assert normalize_angle_deg(10.0 - 350.0) == pytest_approx(20.0)
    assert normalize_angle_deg(350.0 - 10.0) == pytest_approx(-20.0)


def pytest_approx(*args, **kwargs):
    import pytest

    return pytest.approx(*args, **kwargs)
