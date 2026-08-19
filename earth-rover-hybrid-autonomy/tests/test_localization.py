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
    # Guards against the filter being tuned so conservatively it never tracks
    # real, physically gradual motion -- the flip side of outlier rejection.
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

    for step in range(1, 81):
        clock.value += 0.3
        fraction = step / 80.0
        filt.observe_gps_heading(
            start_lat + (target_lat - start_lat) * fraction + noise(),
            target_lon + noise(),
            0.0,
        )

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
    fixes = [(37.0, 127.0), (37.00001, 127.0), (37.00002, 127.0)]

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


def test_origin_does_not_lock_across_distant_multipath_clusters() -> None:
    clock = Clock()
    filt = ekf(clock, origin_lock_fix_count=3, origin_lock_max_spread_m=5.0)
    cluster_a = (30.482479095458984, 114.30262756347656)
    cluster_b = (30.482717514038086, 114.30302429199219)  # about 48 m away

    for fix in (cluster_a, cluster_b, cluster_a, cluster_b):
        clock.value += 0.5
        filt.observe_gps_heading(*fix, 177.0)

    assert filt.is_locked is False
    assert filt.position_valid is False
    assert filt.status()["position_status_reason"] == "ORIGIN_FIX_SPREAD_RESET"

    for _ in range(2):
        clock.value += 0.5
        filt.observe_gps_heading(*cluster_b, 177.0)

    assert filt.is_locked is True
    assert filt.position_valid is True
    assert haversine_distance_m(*filt.current_estimate()[:2], *cluster_b) < 0.5


def test_reset_discards_previous_session_origin_and_reacquires() -> None:
    clock = Clock()
    filt = ekf(clock, origin_lock_fix_count=3)
    first_session = (30.4826126, 114.3026733)
    mission_session = (30.4824720, 114.3026428)

    for _ in range(3):
        clock.value += 0.5
        filt.observe_gps_heading(*first_session, 167.0)
    assert filt.is_locked is True
    assert filt.position_valid is True

    filt.reset()
    assert filt.is_locked is False
    assert filt.position_valid is False
    assert filt.heading_valid is False
    assert filt.current_estimate() == (None, None, None)
    assert filt.status()["position_status_reason"] == "UNINITIALIZED"

    for _ in range(3):
        clock.value += 0.5
        filt.observe_gps_heading(*mission_session, 181.0)

    lat, lon, heading = filt.current_estimate()
    assert filt.is_locked is True
    assert filt.position_valid is True
    assert filt.heading_valid is True
    assert haversine_distance_m(lat, lon, *mission_session) < 0.5
    assert heading == pytest.approx(181.0)


def test_live_stationary_gps_cluster_jump_is_rejected_until_normal_fixes_recover() -> None:
    clock = Clock()
    filt = ekf(
        clock,
        origin_lock_fix_count=1,
        position_outlier_reject_m=8.0,
        position_recovery_streak=3,
    )
    good = (30.482479095458984, 114.30262756347656)
    jumped = (30.482717514038086, 114.30302429199219)
    filt.observe_gps_heading(*good, 177.0)
    accepted = filt.current_estimate()[:2]

    clock.value += 0.5
    filt.observe_gps_heading(*jumped, 4.0)

    assert filt.position_valid is False
    assert filt.current_estimate()[:2] == pytest.approx(accepted)
    assert filt.status()["position_status_reason"] == "POSITION_OUTLIER_REJECTED"
    assert filt.status()["last_position_innovation_m"] > 40.0

    for expected_count in (1, 2):
        clock.value += 0.5
        filt.observe_gps_heading(*good, 177.0)
        assert filt.position_valid is False
        assert filt.status()["position_recovery_count"] == expected_count

    clock.value += 0.5
    filt.observe_gps_heading(*good, 177.0)
    assert filt.position_valid is True
    assert filt.status()["position_status_reason"] == "OK"


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


def test_heading_outlier_burst_is_held_out_and_does_not_flip_fused_heading() -> None:
    # Reproduces a live capture: the SDK's pre-fused `orientation` reading
    # flipped ~171 deg every 10-35s while the rover sat perfectly still
    # (compass/magnetometer fault). Before the outlier gate, Kalman gain
    # alone pulled the fused heading to follow every flip within one
    # update. Outliers must be held out regardless of how long they persist.
    clock = Clock()
    filt = ekf(clock, origin_lock_fix_count=1, heading_recovery_streak=3)
    filt.observe_gps_heading(37.0, 127.0, 176.0)

    for _ in range(10):
        clock.value += 0.5
        filt.observe_gps_heading(37.0, 127.0, 5.0)  # bogus ~171 deg flip
        _, _, heading = filt.current_estimate()
        assert abs(heading - 176.0) < 5.0

    assert filt.gyro_trusted  # the bad reading must not poison gyro trust


def test_wrapped_52_degree_live_jump_is_rejected_with_deployment_threshold() -> None:
    clock = Clock()
    filt = ekf(
        clock,
        origin_lock_fix_count=1,
        heading_outlier_reject_deg=35.0,
    )
    filt.observe_gps_heading(37.0, 127.0, 310.0)
    clock.value += 0.5
    filt.observe_gps_heading(37.0, 127.0, 2.0)

    assert filt.heading_valid is False
    assert filt.current_estimate()[2] == pytest.approx(310.0)
    assert filt.status()["heading_status_reason"] == "HEADING_OUTLIER_REJECTED"


def test_stable_heading_outlier_reanchors_when_gyro_is_untrusted() -> None:
    random.seed(4)
    clock = Clock()
    filt = ekf(
        clock,
        origin_lock_fix_count=1,
        gyro_disagreement_window=10,
        heading_recovery_streak=3,
    )
    heading = 0.0
    for _ in range(15):
        clock.value += 0.5
        filt.observe_gyro([0.0, 0.0, 20.0 + random.uniform(-3, 3)])  # says "turning right"
        heading -= 5.0 + random.uniform(-3, 3)  # truth is turning left
        filt.observe_gps_heading(37.0, 127.0, heading % 360.0)
    assert not filt.gyro_trusted  # sanity check: this run should have disabled it

    for _ in range(3):
        clock.value += 0.5
        filt.observe_gps_heading(37.0, 127.0, 176.0)

    _, _, fused_heading = filt.current_estimate()
    assert fused_heading == pytest.approx(176.0)
    assert filt.heading_valid is True
    assert (
        filt.status()["heading_status_reason"]
        == "HEADING_REACQUIRED_GYRO_UNTRUSTED"
    )
    assert filt.consume_heading_reanchor() is True
    assert filt.consume_heading_reanchor() is False
    assert filt.status()["heading_outlier_streak"] == 0


def test_live_161_to_355_heading_recovers_only_after_stable_outlier_streak() -> None:
    clock = Clock()
    filt = ekf(
        clock,
        origin_lock_fix_count=1,
        heading_outlier_reject_deg=35.0,
        heading_recovery_streak=3,
        heading_recovery_max_delta_deg=8.0,
    )
    filt.observe_gps_heading(37.0, 127.0, 161.0)
    filt._gyro_trusted = False

    for expected_count, heading in ((1, 355.0), (2, 357.0)):
        clock.value += 0.5
        filt.observe_gps_heading(37.0, 127.0, heading)
        assert filt.heading_valid is False
        assert filt.current_estimate()[2] == pytest.approx(161.0)
        assert filt.status()["heading_recovery_count"] == expected_count

    clock.value += 0.5
    filt.observe_gps_heading(37.0, 127.0, 356.0)
    assert filt.heading_valid is True
    assert filt.current_estimate()[2] == pytest.approx(356.0)
    assert (
        filt.status()["heading_status_reason"]
        == "HEADING_REACQUIRED_GYRO_UNTRUSTED"
    )


def test_untrusted_heading_does_not_reanchor_from_inconsistent_outliers() -> None:
    clock = Clock()
    filt = ekf(
        clock,
        origin_lock_fix_count=1,
        heading_outlier_reject_deg=35.0,
        heading_recovery_streak=3,
        heading_recovery_max_delta_deg=8.0,
    )
    filt.observe_gps_heading(37.0, 127.0, 161.0)
    filt._gyro_trusted = False

    for heading in (355.0, 330.0, 5.0, 350.0):
        clock.value += 0.5
        filt.observe_gps_heading(37.0, 127.0, heading)
        assert filt.heading_valid is False
        assert filt.status()["heading_recovery_count"] == 1

    assert filt.current_estimate()[2] == pytest.approx(161.0)


def test_persistent_heading_outlier_has_no_hard_cap_force_accept() -> None:
    clock = Clock()
    filt = ekf(clock, origin_lock_fix_count=1, heading_recovery_streak=3)
    filt.observe_gps_heading(37.0, 127.0, 176.0)

    for _ in range(20):
        clock.value += 0.5
        filt.observe_gps_heading(37.0, 127.0, 5.0)

    _, _, heading = filt.current_estimate()
    assert abs(heading - 176.0) < 5.0
    assert filt.heading_valid is False
    assert filt.gyro_trusted  # never challenged -- still nominally trusted


def test_heading_recovers_only_after_consecutive_accepted_measurements() -> None:
    clock = Clock()
    filt = ekf(clock, origin_lock_fix_count=1, heading_recovery_streak=3)
    filt.observe_gps_heading(37.0, 127.0, 176.0)
    clock.value += 0.5
    filt.observe_gps_heading(37.0, 127.0, 5.0)
    assert filt.heading_valid is False

    for expected_count in (1, 2):
        clock.value += 0.5
        filt.observe_gps_heading(37.0, 127.0, 176.0)
        assert filt.heading_valid is False
        assert filt.status()["heading_recovery_count"] == expected_count
    clock.value += 0.5
    filt.observe_gps_heading(37.0, 127.0, 176.0)
    assert filt.heading_valid is True
    assert filt.status()["heading_status_reason"] == "OK"


def test_heading_does_not_recover_from_mutually_inconsistent_measurements() -> None:
    clock = Clock()
    filt = ekf(
        clock,
        origin_lock_fix_count=1,
        heading_outlier_reject_deg=60.0,
        heading_recovery_streak=3,
        heading_recovery_max_delta_deg=8.0,
    )
    filt.observe_gps_heading(37.0, 127.0, 0.0)
    clock.value += 0.5
    filt.observe_gps_heading(37.0, 127.0, 100.0)
    assert filt.heading_valid is False

    for heading in (20.0, 35.0, 10.0, 30.0):
        clock.value += 0.5
        filt.observe_gps_heading(37.0, 127.0, heading)
        assert filt.heading_valid is False
        assert filt.status()["heading_recovery_count"] == 1


def test_gyro_corroborated_large_heading_change_is_accepted_immediately() -> None:
    # The outlier gate must not block a real fast turn the gyro is
    # actively tracking -- only readings the gyro doesn't corroborate.
    clock = Clock()
    filt = ekf(clock, origin_lock_fix_count=1)
    filt.observe_gps_heading(37.0, 127.0, 0.0)

    for _ in range(4):
        clock.value += 0.5
        filt.observe_gyro([0.0, 0.0, 90.0])  # gyro honestly reports the turn
        filt.observe_gps_heading(37.0, 127.0, 90.0)

    _, _, heading = filt.current_estimate()
    assert heading == pytest.approx(90.0, abs=10.0)
    assert filt._heading_outlier_streak == 0


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
        {"heading_outlier_reject_deg": 0.0},
        {"heading_recovery_streak": 0},
        {"heading_recovery_max_delta_deg": 0.0},
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
