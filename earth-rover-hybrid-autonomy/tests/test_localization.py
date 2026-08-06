from __future__ import annotations

import random
import statistics

import pytest

from earth_rover.navigation.checkpoint_route import CheckpointRoutePlanner
from earth_rover.navigation.gps_utils import (
    haversine_distance_m,
    latlon_to_local_xy,
    local_xy_to_latlon,
)
from earth_rover.navigation.localization import GpsHeadingEkf, GpsHeadingEkfConfig


class Clock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


def ekf(clock: Clock, **overrides) -> GpsHeadingEkf:
    return GpsHeadingEkf(overrides, monotonic=clock)


def test_local_xy_projection_round_trips_at_low_and_high_latitude() -> None:
    for origin_lat in (0.0, 37.5, 60.0):
        origin_lon = 127.0
        lat, lon = origin_lat + 0.0001, origin_lon + 0.0002
        x, y = latlon_to_local_xy(lat, lon, origin_lat, origin_lon)
        back_lat, back_lon = local_xy_to_latlon(x, y, origin_lat, origin_lon)
        assert back_lat == pytest.approx(lat, abs=1e-9)
        assert back_lon == pytest.approx(lon, abs=1e-9)


def test_stationary_jitter_is_reduced_relative_to_raw_gps() -> None:
    # The actual bug report: GPS jitters noticeably even standing still.
    random.seed(0)
    clock = Clock()
    filt = ekf(clock)
    true_lat, true_lon = 37.0, 127.0

    raw_lats = []
    fused_lats = []
    for _ in range(40):
        clock.value += 0.5
        noisy_lat = true_lat + random.uniform(-0.00004, 0.00004)  # ~roughly meters-scale
        noisy_lon = true_lon + random.uniform(-0.00004, 0.00004)
        filt.observe_gps_heading(noisy_lat, noisy_lon, 90.0 + random.uniform(-5, 5))
        raw_lats.append(noisy_lat)
        fused_lats.append(filt.current_estimate()[0])

    raw_std = statistics.pstdev(raw_lats)
    fused_std = statistics.pstdev(fused_lats[10:])  # drop the initial convergence tail
    assert fused_std < raw_std * 0.5


def test_step_change_in_true_position_converges_within_tolerance() -> None:
    # Guards against the filter being tuned so conservatively it never
    # tracks a real move -- the flip side of the jitter-reduction test.
    random.seed(2)
    clock = Clock()
    filt = ekf(clock, origin_lock_fix_count=3, max_linear_speed_mps=1.0)
    start_lat, start_lon = 37.0, 127.0
    target_lat, target_lon = 37.0002, 127.0  # ~22m away
    noise = lambda: random.uniform(-0.00002, 0.00002)

    for _ in range(5):
        clock.value += 0.3
        filt.observe_gps_heading(start_lat + noise(), start_lon + noise(), 0.0)
    assert filt.is_locked

    for _ in range(40):
        clock.value += 0.3
        filt.observe_gps_heading(target_lat + noise(), target_lon + noise(), 0.0)

    lat, lon, _ = filt.current_estimate()
    assert haversine_distance_m(lat, lon, target_lat, target_lon) < 5.0


def test_heading_updates_wrap_correctly_across_0_360_boundary() -> None:
    clock = Clock()
    filt = ekf(clock, origin_lock_fix_count=1)
    headings = [358.0, 359.0, 1.0, 2.0, 359.0, 1.0]
    for heading_deg in headings:
        clock.value += 0.5
        filt.observe_gps_heading(37.0, 127.0, heading_deg)
        _, _, fused_heading = filt.current_estimate()
        # Never jumps to the "long way around" (e.g. ~180 deg away).
        delta = min(
            abs(fused_heading - heading_deg), 360.0 - abs(fused_heading - heading_deg)
        )
        assert delta < 10.0


def test_current_estimate_is_a_cheap_read_that_does_not_refuse() -> None:
    clock = Clock()
    filt = ekf(clock, origin_lock_fix_count=1)
    clock.value += 0.5
    filt.observe_gps_heading(37.0, 127.0, 90.0)
    first = filt.current_estimate()
    for _ in range(5):
        assert filt.current_estimate() == first


def test_origin_locks_to_the_average_of_the_first_n_fixes_not_a_single_one() -> None:
    clock = Clock()
    filt = ekf(clock, origin_lock_fix_count=3)
    fixes = [(37.0, 127.0), (37.0002, 127.0), (37.0004, 127.0)]

    assert filt.origin is None
    assert filt.current_estimate() == (None, None, None)

    for lat, lon in fixes[:-1]:
        clock.value += 0.5
        filt.observe_gps_heading(lat, lon, 90.0)
        assert filt.origin is None  # not locked until the Nth fix

    clock.value += 0.5
    filt.observe_gps_heading(*fixes[-1], 90.0)
    assert filt.is_locked
    assert filt.origin == pytest.approx(
        (sum(f[0] for f in fixes) / 3, sum(f[1] for f in fixes) / 3)
    )


def test_gyro_bias_converges_toward_a_constant_simulated_bias() -> None:
    import math

    clock = Clock()
    filt = ekf(clock, origin_lock_fix_count=1, gyro_disagreement_window=5000)
    true_bias_deg_s = 3.0
    # True heading never actually changes; the gyro consistently reports a
    # nonzero rate -- exactly what a real MEMS gyro's zero-rate bias looks
    # like. The filter should learn to discount it rather than let it drift
    # the fused heading away from what GPS/compass keeps confirming.
    for _ in range(1000):
        clock.value += 0.2
        filt.observe_gyro([0.0, 0.0, true_bias_deg_s])
        filt.observe_gps_heading(37.0, 127.0, 90.0)

    true_bias_rad_s = math.radians(true_bias_deg_s)
    assert abs(filt.gyro_bias_rad_s - true_bias_rad_s) < abs(true_bias_rad_s) * 0.35
    _, _, fused_heading = filt.current_estimate()
    assert fused_heading == pytest.approx(90.0, abs=2.0)


def test_gyro_fusion_disables_itself_when_it_disagrees_with_measured_heading() -> None:
    # Safety net for the case where gyro_yaw_axis_index/sign/scale turn out
    # to be wrong on the real hardware: a consistently wrong-signed gyro
    # must not get to keep fighting real motion for the rest of the run.
    random.seed(1)
    clock = Clock()
    filt = ekf(clock, origin_lock_fix_count=1, gyro_disagreement_window=10)
    heading = 0.0
    assert filt.gyro_trusted

    disabled_at = None
    for i in range(30):
        clock.value += 0.5
        gyro_val = 20.0 + random.uniform(-3, 3)  # gyro says "turning right"
        filt.observe_gyro([0.0, 0.0, gyro_val])
        heading -= 5.0 + random.uniform(-3, 3)  # truth is turning left
        filt.observe_gps_heading(37.0, 127.0, heading % 360.0)
        if not filt.gyro_trusted and disabled_at is None:
            disabled_at = i

    assert disabled_at is not None


def test_invalid_and_missing_inputs_do_not_raise_or_corrupt_state() -> None:
    clock = Clock()
    filt = ekf(clock, origin_lock_fix_count=2)

    filt.observe_gps_heading(None, None, None)
    filt.observe_gps_heading(float("nan"), 127.0, 90.0)
    filt.observe_gyro(None)
    filt.observe_gyro([[float("nan"), 0.0, 0.0]])
    assert not filt.is_locked
    assert filt.current_estimate() == (None, None, None)

    clock.value += 0.5
    filt.observe_gps_heading(37.0, 127.0, 90.0)
    clock.value += 0.5
    filt.observe_gps_heading(37.0, 127.0, None)  # missing heading, valid GPS
    assert filt.is_locked
    lat, lon, heading = filt.current_estimate()
    assert lat == pytest.approx(37.0)
    assert lon == pytest.approx(127.0)
    assert heading == pytest.approx(0.0)  # no heading ever observed yet


def test_config_from_dict_defaults_are_sane_and_validate() -> None:
    config = GpsHeadingEkfConfig.from_dict({})
    config.validate()
    assert config.enabled is True
    assert config.origin_lock_fix_count == 3


@pytest.mark.parametrize(
    "overrides",
    [
        {"origin_lock_fix_count": 0},
        {"max_linear_speed_mps": 0.0},
        {"default_gps_accuracy_m": -1.0},
        {"gps_signal_accuracy_floor_m": 20.0, "gps_signal_accuracy_ceiling_m": 5.0},
        {"gyro_yaw_sign": 0.5},
        {"gyro_yaw_axis_index": 3},
        {"gyro_disagreement_window": 1},
        {"gyro_disagreement_threshold": 1.5},
    ],
)
def test_config_validate_rejects_bad_values(overrides: dict) -> None:
    config = GpsHeadingEkfConfig.from_dict(overrides)
    with pytest.raises(ValueError):
        config.validate()


def test_checkpoint_reached_transitions_monotonically_with_fused_input() -> None:
    # Integration regression: raw GPS jitter right at switch_radius_m can
    # make CheckpointRoutePlanner's `reached` flap true/false/true. Feeding
    # it EKF-fused (smoothed) positions during a realistic-speed approach
    # should settle false -> true exactly once.
    random.seed(3)
    checkpoints = [{"sequence": 1, "latitude": 37.0002, "longitude": 127.0}]
    route = CheckpointRoutePlanner(checkpoints, switch_radius_m=5.0)

    clock = Clock()
    filt = ekf(clock, origin_lock_fix_count=3, default_gps_accuracy_m=3.0, max_linear_speed_mps=1.0)
    start_lat, start_lon = 37.0, 127.0
    target_lat, target_lon = 37.0002, 127.0
    noise = lambda: random.uniform(-0.00002, 0.00002)

    reached_flags = []
    for i in range(90):
        frac = min(1.0, i / 60.0)
        true_lat = start_lat + (target_lat - start_lat) * frac
        clock.value += 0.3
        filt.observe_gps_heading(true_lat + noise(), start_lon + noise(), 0.0)
        lat, lon, heading = filt.current_estimate()
        if lat is None:
            continue
        reached_flags.append(route.update(lat, lon, heading).reached)

    assert True in reached_flags
    seen_true = False
    for flag in reached_flags:
        if flag:
            seen_true = True
        elif seen_true:
            pytest.fail("reached flapped back to False after first becoming True")
