from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from earth_rover.navigation.gps_utils import (
    latlon_to_local_xy,
    local_xy_to_latlon,
    normalize_angle_rad,
)
from earth_rover.utils.math_utils import clamp, safe_float

logger = logging.getLogger(__name__)

_STATE_SIZE = 4  # [x_m, y_m, heading_rad, gyro_bias_rad_s]

# Last-resort multiple of heading_outlier_confirm_streak before a
# persistently-rejected heading reading is force-accepted even while gyro
# is nominally trusted but never corroborates it either way (e.g. gyro
# feed silently stops). Not exposed as a config knob -- it only matters
# for this one degraded edge case, not something a deployment should need
# to independently tune.
_HEADING_OUTLIER_HARD_CAP_MULTIPLIER = 3


@dataclass(frozen=True)
class GpsHeadingEkfConfig:
    """Tunables for :class:`GpsHeadingEkf`.

    Every "physical units" knob here is a human estimate, not a value
    derived from the rover's normalized [-1, 1] control command range
    (``control.linear_max``/``mission1_autonomy.max_angular`` etc. are
    command duty-cycle, not m/s or rad/s -- there is no confirmed
    conversion between the two anywhere in this codebase).
    """

    enabled: bool = True

    # Local ENU origin lock.
    origin_lock_fix_count: int = 3

    # Position process noise: how far the rover could plausibly have moved
    # since the last fusion, absent any other information.
    max_linear_speed_mps: float = 0.5
    process_noise_position_scale: float = 1.0

    # GPS measurement trust.
    default_gps_accuracy_m: float = 3.0
    use_gps_signal_for_accuracy: bool = False
    gps_signal_accuracy_floor_m: float = 1.0
    gps_signal_accuracy_ceiling_m: float = 15.0
    gps_signal_min: float = 0.0
    gps_signal_max: float = 100.0
    gps_signal_higher_is_better: bool = True

    # Heading measurement (the SDK's pre-fused `orientation` scalar).
    heading_measurement_std_deg: float = 8.0

    # Heading process noise. Reuse navigation.max_heading_rate_deg_per_sec
    # (already tuned, already physical-units) rather than a second
    # independently-tuned bound -- the wiring code is responsible for
    # forwarding that value in here so the two stay in sync.
    max_heading_rate_deg_per_sec: float = 120.0

    # Gyro yaw axis: which raw [x, y, z, timestamp] column is yaw, its sign
    # relative to this codebase's positive-clockwise-right convention, and
    # its scale to rad/s. All three are unverified assumptions (the SDK
    # documents no units for `gyros`) -- defaults assume raw units are
    # already deg/s. See the gyro-disagreement safety net below.
    gyro_yaw_axis_index: int = 2
    gyro_yaw_sign: float = 1.0
    gyro_scale_rad_per_sec_per_unit: float = math.radians(1.0)
    gyro_bias_process_noise: float = 1.0e-5

    # Safety net: if the gyro-only predicted heading change disagrees with
    # the independently-measured heading change over this many consecutive
    # heading updates (correlation below the threshold), gyro fusion is
    # permanently disabled for the rest of this process's life and the
    # filter falls back to GPS+heading-only (a wrong sign/scale assumption
    # degrades to "no worse than before," not to actively fighting real
    # motion).
    gyro_disagreement_window: int = 20
    gyro_disagreement_threshold: float = 0.5

    # Heading-measurement outlier gate: a live run showed the SDK's
    # pre-fused `orientation` reading flip by ~170 deg every 10-35s while
    # the rover sat perfectly still (compass/magnetometer fault, not GPS
    # position noise -- see localization.py module docs). Kalman gain alone
    # doesn't reject this: an 8 deg measurement std still pulls the state
    # most of the way to a 170 deg-away reading in one update, and it fed
    # straight into the gyro-disagreement correlation above, which then
    # mistook the bad *measurement* for a bad *gyro* and disabled gyro
    # fusion to compensate -- the opposite of what was needed, since the
    # gyro (real motion evidence) was the one telling the truth. A heading
    # measurement whose innovation exceeds this many degrees is now held
    # out of both the Kalman update and the gyro-disagreement bookkeeping
    # instead of being fused.
    heading_outlier_reject_deg: float = 60.0
    # A rejected reading isn't discarded forever once gyro *can't* vouch
    # for it either way (untrusted/disabled): after this many consecutive
    # rejections, the most recent reading is fused anyway (P has grown
    # unmeasured that whole time, so the correction snaps straight to it).
    # While gyro is trusted this count alone is deliberately NOT enough to
    # force-accept -- see the comment in _update_heading for why a live
    # run showed that force-accepting on a strike count anyway snaps onto
    # whatever the rejected reading happens to be at that moment, which is
    # exactly as likely to be the fault as the truth.
    heading_outlier_confirm_streak: int = 20

    @classmethod
    def from_dict(cls, config: dict[str, Any] | None) -> "GpsHeadingEkfConfig":
        values = dict(config or {})
        return cls(**{key: values[key] for key in cls.__dataclass_fields__ if key in values})

    def validate(self) -> None:
        if (
            isinstance(self.origin_lock_fix_count, bool)
            or not isinstance(self.origin_lock_fix_count, int)
            or self.origin_lock_fix_count < 1
        ):
            raise ValueError("origin_lock_fix_count must be a positive integer")
        positive = {
            "max_linear_speed_mps": self.max_linear_speed_mps,
            "process_noise_position_scale": self.process_noise_position_scale,
            "default_gps_accuracy_m": self.default_gps_accuracy_m,
            "gps_signal_accuracy_floor_m": self.gps_signal_accuracy_floor_m,
            "gps_signal_accuracy_ceiling_m": self.gps_signal_accuracy_ceiling_m,
            "heading_measurement_std_deg": self.heading_measurement_std_deg,
            "max_heading_rate_deg_per_sec": self.max_heading_rate_deg_per_sec,
            "gyro_bias_process_noise": self.gyro_bias_process_noise,
            "gyro_scale_rad_per_sec_per_unit": self.gyro_scale_rad_per_sec_per_unit,
            "heading_outlier_reject_deg": self.heading_outlier_reject_deg,
        }
        if any(not math.isfinite(value) or value <= 0.0 for value in positive.values()):
            raise ValueError(
                "GpsHeadingEkfConfig positive settings must be finite and positive"
            )
        if self.gps_signal_accuracy_floor_m > self.gps_signal_accuracy_ceiling_m:
            raise ValueError(
                "gps_signal_accuracy_floor_m must be <= gps_signal_accuracy_ceiling_m"
            )
        if self.gps_signal_min >= self.gps_signal_max:
            raise ValueError("gps_signal_min must be < gps_signal_max")
        if self.gyro_yaw_sign not in (-1.0, 1.0):
            raise ValueError("gyro_yaw_sign must be 1.0 or -1.0")
        if self.gyro_yaw_axis_index not in (0, 1, 2):
            raise ValueError("gyro_yaw_axis_index must be 0, 1, or 2")
        if (
            isinstance(self.gyro_disagreement_window, bool)
            or not isinstance(self.gyro_disagreement_window, int)
            or self.gyro_disagreement_window < 2
        ):
            raise ValueError("gyro_disagreement_window must be an integer >= 2")
        if not math.isfinite(self.gyro_disagreement_threshold) or not (
            -1.0 <= self.gyro_disagreement_threshold <= 1.0
        ):
            raise ValueError("gyro_disagreement_threshold must be in [-1, 1]")
        if (
            isinstance(self.heading_outlier_confirm_streak, bool)
            or not isinstance(self.heading_outlier_confirm_streak, int)
            or self.heading_outlier_confirm_streak < 1
        ):
            raise ValueError("heading_outlier_confirm_streak must be a positive integer")


class GpsHeadingEkf:
    """Fuses noisy GPS + a pre-fused heading reading (+ optionally gyro yaw
    rate) into a stable position/heading estimate for checkpoint navigation.

    This sits *before* :class:`~earth_rover.navigation.checkpoint_route.
    CheckpointRoutePlanner`, which is unchanged -- callers fuse telemetry
    here first via :meth:`observe_gps_heading`, then pass
    :meth:`current_estimate` into ``CheckpointRoutePlanner.update()`` instead
    of raw telemetry. ``CheckpointRoutePlanner``'s own heading-rate outlier
    rejector (``_sanitize_heading``) stays in place as defense in depth; it
    should rarely trigger once this filter is active since both are sized
    from the same ``max_heading_rate_deg_per_sec`` bound.

    Position (x, y) is a process-noise-only random walk between GPS fixes --
    it is *not* dead-reckoned from gyro or accelerometer, since there is no
    trustworthy speed source to integrate and accelerometer double
    integration drifts unboundedly without an independent correction this
    rover doesn't have. Heading, when a gyro sample is available and trusted,
    is propagated with the yaw-rate reading between measurements and
    corrected by the absolute heading measurement on every fusion -- the
    standard "gyro-propagated heading + absolute correction" pattern. With
    no gyro coupling into (x, y), the process model here has no nonlinear
    term to linearize (only the heading measurement's angle-wrap is
    nonlinear) -- this is a linear Kalman filter with a wrapped residual,
    not a true EKF, until gyro-coupled heading propagation is extended to
    also drive position (not implemented; accelerometer/velocity-based
    position propagation is deliberately out of scope, see module plan).
    The class is still named for forward compatibility with that extension.
    """

    def __init__(
        self,
        config: GpsHeadingEkfConfig | dict[str, Any] | None = None,
        *,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = (
            config
            if isinstance(config, GpsHeadingEkfConfig)
            else GpsHeadingEkfConfig.from_dict(config)
        )
        self.config.validate()
        self._monotonic = monotonic

        self._origin: tuple[float, float] | None = None
        self._origin_fixes: list[tuple[float, float]] = []
        self._state: np.ndarray | None = None
        self._P: np.ndarray | None = None
        self._last_fusion_monotonic: float | None = None

        self._latest_yaw_rate_rad_s = 0.0
        self._gyro_trusted = True
        self._gyro_delta_accum = 0.0
        self._heading_at_last_measurement: float | None = None
        self._heading_outlier_streak = 0
        self._disagreement_pairs: deque[tuple[float, float]] = deque(
            maxlen=self.config.gyro_disagreement_window
        )

    @property
    def gyro_trusted(self) -> bool:
        return self._gyro_trusted

    @property
    def is_locked(self) -> bool:
        return self._state is not None

    @property
    def origin(self) -> tuple[float, float] | None:
        """The locked local-ENU origin (lat, lon), or ``None`` before lock."""

        return self._origin

    @property
    def gyro_bias_rad_s(self) -> float:
        """Current estimated gyro yaw-rate bias, 0.0 before origin lock."""

        return float(self._state[3]) if self._state is not None else 0.0

    def observe_gyro(self, raw_samples: Any) -> None:
        """Record the latest yaw-rate sample(s) for the next predict step.

        ``raw_samples`` may be a single ``[x, y, z, ...]`` row, a list of
        such rows (the SDK bundles several per telemetry poll), or ``None``.
        Multiple rows are averaged into one representative rate rather than
        integrated per-sample, since the SDK's own per-sample timestamps are
        not a trusted clock (see ``sdk_clock_offset_hours`` handling
        elsewhere in this codebase) -- this loses some time resolution
        within a poll but avoids trusting a second unverified clock source.
        """

        rate = self._extract_yaw_rate(raw_samples)
        if rate is not None:
            self._latest_yaw_rate_rad_s = rate

    def _extract_yaw_rate(self, raw_samples: Any) -> float | None:
        if raw_samples is None:
            return None
        rows: list[Any]
        if raw_samples and isinstance(raw_samples[0], (list, tuple)):
            rows = list(raw_samples)
        else:
            rows = [raw_samples]
        axis = self.config.gyro_yaw_axis_index
        values: list[float] = []
        for row in rows:
            if not isinstance(row, (list, tuple)) or len(row) <= axis:
                continue
            value = safe_float(row[axis])
            if value is not None:
                values.append(value)
        if not values:
            return None
        raw_mean = sum(values) / len(values)
        return self.config.gyro_yaw_sign * raw_mean * self.config.gyro_scale_rad_per_sec_per_unit

    def observe_gps_heading(
        self,
        latitude: object,
        longitude: object,
        heading_deg: object,
        gps_signal: object = None,
    ) -> None:
        """Fuse one GPS(+heading) reading. Call once per fresh telemetry fetch.

        Must not be called more than once for the same telemetry sample --
        re-fusing a stale reading makes the filter overconfident in it and
        more resistant to genuinely new fixes, the opposite of the goal.
        """

        lat = safe_float(latitude)
        lon = safe_float(longitude)
        if lat is None or lon is None or not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
            return
        heading_deg_value = safe_float(heading_deg)
        now = self._monotonic()

        if self._origin is None:
            self._origin_fixes.append((lat, lon))
            self._last_fusion_monotonic = now
            if len(self._origin_fixes) < self.config.origin_lock_fix_count:
                return
            origin_lat = sum(fix[0] for fix in self._origin_fixes) / len(self._origin_fixes)
            origin_lon = sum(fix[1] for fix in self._origin_fixes) / len(self._origin_fixes)
            self._origin = (origin_lat, origin_lon)
            x0, y0 = latlon_to_local_xy(lat, lon, origin_lat, origin_lon)
            heading0 = math.radians(heading_deg_value) if heading_deg_value is not None else 0.0
            self._state = np.array([x0, y0, heading0, 0.0], dtype=float)
            self._P = np.diag(
                [
                    self.config.default_gps_accuracy_m**2,
                    self.config.default_gps_accuracy_m**2,
                    math.radians(30.0) ** 2,
                    (math.radians(self.config.max_heading_rate_deg_per_sec) * 0.1) ** 2,
                ]
            )
            self._heading_at_last_measurement = heading0
            return

        dt = 0.0
        if self._last_fusion_monotonic is not None:
            dt = max(0.0, now - self._last_fusion_monotonic)
        self._predict(dt)

        x_meas, y_meas = latlon_to_local_xy(lat, lon, *self._origin)
        self._update_gps(x_meas, y_meas, self._gps_accuracy_m(gps_signal))
        if heading_deg_value is not None:
            self._update_heading(math.radians(heading_deg_value))
        self._last_fusion_monotonic = now

    def current_estimate(self) -> tuple[float, float, float] | tuple[None, None, None]:
        """Cheap, unconditional read of the current fused estimate.

        Safe to call every loop tick regardless of telemetry cadence; it
        does not advance the filter (call :meth:`observe_gps_heading` for
        that). Returns ``(None, None, None)`` before the origin has locked.
        """

        if self._state is None or self._origin is None:
            return (None, None, None)
        lat, lon = local_xy_to_latlon(
            float(self._state[0]), float(self._state[1]), *self._origin
        )
        heading_deg = math.degrees(float(self._state[2])) % 360.0
        return (lat, lon, heading_deg)

    def _gps_accuracy_m(self, gps_signal: object) -> float:
        if not self.config.use_gps_signal_for_accuracy:
            return self.config.default_gps_accuracy_m
        value = safe_float(gps_signal)
        if value is None:
            return self.config.default_gps_accuracy_m
        span = self.config.gps_signal_max - self.config.gps_signal_min
        normalized = clamp((value - self.config.gps_signal_min) / span, 0.0, 1.0)
        quality = normalized if self.config.gps_signal_higher_is_better else 1.0 - normalized
        floor = self.config.gps_signal_accuracy_floor_m
        ceiling = self.config.gps_signal_accuracy_ceiling_m
        return ceiling - quality * (ceiling - floor)

    def _predict(self, dt: float) -> None:
        if self._state is None or self._P is None or dt <= 0.0:
            return
        heading_delta = 0.0
        bias_jacobian = 0.0
        if self._gyro_trusted:
            yaw_rate = self._latest_yaw_rate_rad_s - float(self._state[3])
            heading_delta = yaw_rate * dt
            bias_jacobian = -dt
            self._gyro_delta_accum += heading_delta
        self._state[2] = normalize_angle_rad(float(self._state[2]) + heading_delta)

        F = np.eye(_STATE_SIZE)
        F[2, 3] = bias_jacobian
        pos_std = (
            self.config.max_linear_speed_mps * self.config.process_noise_position_scale * dt
        )
        heading_std = math.radians(self.config.max_heading_rate_deg_per_sec) * dt
        Q = np.diag(
            [
                pos_std**2,
                pos_std**2,
                heading_std**2,
                self.config.gyro_bias_process_noise * dt,
            ]
        )
        self._P = F @ self._P @ F.T + Q

    def _kalman_update(self, H: np.ndarray, R: np.ndarray, innovation: np.ndarray) -> None:
        assert self._state is not None and self._P is not None
        S = H @ self._P @ H.T + R
        K = self._P @ H.T @ np.linalg.inv(S)
        self._state = self._state + (K @ innovation)
        identity = np.eye(_STATE_SIZE)
        self._P = (identity - K @ H) @ self._P

    def _update_gps(self, x_meas: float, y_meas: float, accuracy_m: float) -> None:
        H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        R = np.diag([accuracy_m**2, accuracy_m**2])
        innovation = np.array(
            [x_meas - float(self._state[0]), y_meas - float(self._state[1])]
        )
        self._kalman_update(H, R, innovation)

    def _update_heading(self, heading_meas_rad: float) -> None:
        assert self._state is not None
        raw_innovation = normalize_angle_rad(heading_meas_rad - float(self._state[2]))
        reject_threshold = math.radians(self.config.heading_outlier_reject_deg)
        if abs(raw_innovation) > reject_threshold:
            self._heading_outlier_streak += 1
            # A live rotation test caught the flaw in a plain strike-count
            # fallback: while gyro is trusted it's still propagating state[2]
            # every _predict tick (real motion, unfrozen), so a reading that
            # STAYS rejected for the full confirm streak means gyro actively
            # disagrees with it -- force-accepting anyway (the original
            # design) snapped the filter onto a bogus ~170 deg reading mid
            # rotation, then gyro dragged that wrong anchor along in the
            # true turn's direction for another ~15s before a second
            # force-accept happened to land back near truth. A genuine
            # change gyro agrees with instead drifts state close enough on
            # its own for innovation to drop under the threshold -- no
            # force-accept needed. So: only blind-accept on a strike count
            # once gyro can't vouch either way (untrusted), and even then
            # only past the normal confirm streak; a much longer hard cap
            # is the last-resort safety valve for a gyro that's nominally
            # trusted but simply never being fed anything (never
            # challenged, never disagrees) so a real change would otherwise
            # be held out forever.
            untrusted_fallback = (
                not self._gyro_trusted
                and self._heading_outlier_streak >= self.config.heading_outlier_confirm_streak
            )
            hard_cap_reached = self._heading_outlier_streak >= (
                self.config.heading_outlier_confirm_streak
                * _HEADING_OUTLIER_HARD_CAP_MULTIPLIER
            )
            if not (untrusted_fallback or hard_cap_reached):
                logger.debug(
                    "GpsHeadingEkf: rejecting heading measurement %.1f deg "
                    "(innovation %.1f deg exceeds heading_outlier_reject_deg, "
                    "streak=%d, gyro_trusted=%s)",
                    math.degrees(heading_meas_rad),
                    math.degrees(raw_innovation),
                    self._heading_outlier_streak,
                    self._gyro_trusted,
                )
                return
        self._heading_outlier_streak = 0

        H = np.array([[0.0, 0.0, 1.0, 0.0]])
        R = np.array([[math.radians(self.config.heading_measurement_std_deg) ** 2]])
        innovation = np.array([raw_innovation])
        measured_delta = None
        if self._heading_at_last_measurement is not None:
            measured_delta = normalize_angle_rad(
                heading_meas_rad - self._heading_at_last_measurement
            )
        self._kalman_update(H, R, innovation)
        self._state[2] = normalize_angle_rad(float(self._state[2]))

        if measured_delta is not None and self._gyro_trusted:
            self._disagreement_pairs.append((self._gyro_delta_accum, measured_delta))
            self._maybe_disable_gyro()
        self._gyro_delta_accum = 0.0
        self._heading_at_last_measurement = heading_meas_rad

    def _maybe_disable_gyro(self) -> None:
        if len(self._disagreement_pairs) < self.config.gyro_disagreement_window:
            return
        gyro_deltas = np.array([pair[0] for pair in self._disagreement_pairs])
        measured_deltas = np.array([pair[1] for pair in self._disagreement_pairs])
        if np.std(gyro_deltas) < 1e-9 or np.std(measured_deltas) < 1e-9:
            return
        correlation = float(np.corrcoef(gyro_deltas, measured_deltas)[0, 1])
        if not math.isfinite(correlation):
            return
        if correlation < self.config.gyro_disagreement_threshold:
            self._gyro_trusted = False
            logger.warning(
                "GpsHeadingEkf: gyro yaw-rate disagrees with measured heading "
                "change over the last %d observations (correlation=%.2f); "
                "disabling gyro fusion for the rest of this run. Check "
                "gyro_yaw_axis_index/gyro_yaw_sign/gyro_scale_rad_per_sec_per_unit.",
                self.config.gyro_disagreement_window,
                correlation,
            )
