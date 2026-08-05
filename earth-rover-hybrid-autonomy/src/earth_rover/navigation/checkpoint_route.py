from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable

from earth_rover.navigation.gps_utils import bearing_deg, normalize_angle_deg
from earth_rover.navigation.waypoint_manager import WaypointManager
from earth_rover.utils.math_utils import safe_float


@dataclass(frozen=True)
class GlobalRouteState:
    """Read-only GPS guidance for the current checkpoint route.

    ``route_polyline`` starts at the rover's current GPS position and contains
    every remaining checkpoint in mission sequence order.  It is a global
    guidance polyline, not a claim that every point on a straight segment is
    locally traversable.  SAM-TP is responsible for selecting a safe local
    deviation while preserving progress toward ``target_bearing_deg``.
    """

    route_polyline: tuple[tuple[float, float], ...]
    target_checkpoint: dict[str, Any] | None
    target_sequence: int | None
    distance_to_target_m: float | None
    target_bearing_deg: float | None
    current_heading_deg: float | None
    heading_error_rad: float | None
    gps_valid: bool
    heading_valid: bool
    reached: bool
    finished: bool
    reason: str


class CheckpointRoutePlanner:
    """Turn rover GPS and ordered mission checkpoints into global guidance.

    Reaching the switch radius does not advance the route.  The caller must
    first report the checkpoint successfully and then call
    :meth:`mark_current_reported`, matching the SDK mission contract.
    """

    def __init__(
        self,
        checkpoints: list[dict[str, Any]],
        switch_radius_m: float,
        latest_scanned_checkpoint: int = 0,
        heading_filter_alpha: float = 1.0,
        target_heading_deadband_deg: float = 0.0,
        large_heading_change_deg: float = 180.0,
        max_heading_rate_deg_per_sec: float | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        radius = _finite(switch_radius_m)
        if radius is None or radius <= 0.0:
            raise ValueError("switch_radius_m must be finite and positive")
        if not math.isfinite(heading_filter_alpha) or not 0.0 < heading_filter_alpha <= 1.0:
            raise ValueError("heading_filter_alpha must be in (0, 1]")
        if not math.isfinite(target_heading_deadband_deg) or target_heading_deadband_deg < 0.0:
            raise ValueError("target_heading_deadband_deg must be finite and non-negative")
        if not math.isfinite(large_heading_change_deg) or large_heading_change_deg <= 0.0:
            raise ValueError("large_heading_change_deg must be finite and positive")
        if max_heading_rate_deg_per_sec is not None and (
            not math.isfinite(max_heading_rate_deg_per_sec) or max_heading_rate_deg_per_sec <= 0.0
        ):
            raise ValueError("max_heading_rate_deg_per_sec must be finite and positive")
        self._waypoints = WaypointManager(
            checkpoints,
            radius,
            latest_scanned_checkpoint=latest_scanned_checkpoint,
        )
        self._heading_filter_alpha = float(heading_filter_alpha)
        self._target_heading_deadband_deg = float(target_heading_deadband_deg)
        self._large_heading_change_deg = float(large_heading_change_deg)
        self._max_heading_rate_deg_per_sec = (
            None
            if max_heading_rate_deg_per_sec is None
            else float(max_heading_rate_deg_per_sec)
        )
        self._monotonic = monotonic
        self._filtered_heading_error_deg: float | None = None
        self._filtered_target_sequence: int | None = None
        self._last_accepted_heading_deg: float | None = None
        self._last_accepted_heading_monotonic: float | None = None

    def update(
        self,
        latitude: object,
        longitude: object,
        heading_deg: object,
    ) -> GlobalRouteState:
        lat = _latitude(latitude)
        lon = _longitude(longitude)
        heading = self._sanitize_heading(_heading(heading_deg))
        target = self._waypoints.current_target()

        if target is None:
            current = ((lat, lon),) if lat is not None and lon is not None else ()
            return GlobalRouteState(
                route_polyline=current,
                target_checkpoint=None,
                target_sequence=None,
                distance_to_target_m=None,
                target_bearing_deg=None,
                current_heading_deg=heading,
                heading_error_rad=None,
                gps_valid=lat is not None and lon is not None,
                heading_valid=heading is not None,
                reached=False,
                finished=True,
                reason="MISSION_COMPLETE",
            )

        target_sequence = _sequence(target)
        if lat is None or lon is None:
            return GlobalRouteState(
                route_polyline=self._remaining_checkpoint_coordinates(),
                target_checkpoint=dict(target),
                target_sequence=target_sequence,
                distance_to_target_m=None,
                target_bearing_deg=None,
                current_heading_deg=heading,
                heading_error_rad=None,
                gps_valid=False,
                heading_valid=heading is not None,
                reached=False,
                finished=False,
                reason="INVALID_GPS",
            )

        waypoint_state = self._waypoints.update(lat, lon)
        target_lat, target_lon = _checkpoint_coordinate(target)
        route = ((lat, lon),) + self._remaining_checkpoint_coordinates()
        if target_lat is None or target_lon is None:
            return GlobalRouteState(
                route_polyline=route,
                target_checkpoint=dict(target),
                target_sequence=target_sequence,
                distance_to_target_m=None,
                target_bearing_deg=None,
                current_heading_deg=heading,
                heading_error_rad=None,
                gps_valid=True,
                heading_valid=heading is not None,
                reached=False,
                finished=False,
                reason="INVALID_CHECKPOINT",
            )

        target_bearing = bearing_deg(lat, lon, target_lat, target_lon)
        if waypoint_state["reached"]:
            # Bearing from sub-radius GPS points is dominated by GPS jitter.
            # Stop/report the checkpoint before generating guidance to the
            # next target instead of projecting a misleading camera heading.
            heading_error = None
            reason = "CHECKPOINT_REACHED_PENDING_REPORT"
        elif heading is None:
            heading_error = None
            reason = "INVALID_HEADING"
        else:
            filtered_error_deg = self._filter_heading_error(
                normalize_angle_deg(target_bearing - heading),
                target_sequence,
            )
            heading_error = math.radians(filtered_error_deg)
            reason = "TRACKING"

        return GlobalRouteState(
            route_polyline=route,
            target_checkpoint=dict(target),
            target_sequence=target_sequence,
            distance_to_target_m=waypoint_state["distance_m"],
            target_bearing_deg=target_bearing,
            current_heading_deg=heading,
            heading_error_rad=heading_error,
            gps_valid=True,
            heading_valid=heading is not None,
            reached=bool(waypoint_state["reached"]),
            finished=False,
            reason=reason,
        )

    def mark_current_reported(self) -> None:
        self._waypoints.mark_current_reported()
        self._filtered_heading_error_deg = None
        self._filtered_target_sequence = None

    def _sanitize_heading(self, heading: float | None) -> float | None:
        """Reject a heading reading that implies an impossible turn rate.

        A live run showed current_heading_deg jump between unrelated values
        (e.g. 297 -> 213 -> 40 -> 299 within about a second of telemetry) --
        physically impossible for this platform, but each jump was still
        >= large_heading_change_deg, so _filter_heading_error's smoothing
        (which intentionally snaps straight through on a large change, to
        track a real sharp turn quickly) let the noise straight into
        heading_error_rad. That drove the local planner's immediate_reset
        path and mission1's ROTATE_TO_GOAL to chase a bogus heading, which
        showed up as the rover spinning in place. Hold the last accepted
        heading instead of accepting a reading whose implied turn rate
        exceeds what the rover can actually do.
        """

        if heading is None or self._max_heading_rate_deg_per_sec is None:
            if heading is not None:
                self._last_accepted_heading_deg = heading
                self._last_accepted_heading_monotonic = self._monotonic()
            return heading
        now = self._monotonic()
        if (
            self._last_accepted_heading_deg is None
            or self._last_accepted_heading_monotonic is None
        ):
            self._last_accepted_heading_deg = heading
            self._last_accepted_heading_monotonic = now
            return heading
        dt = max(1e-3, now - self._last_accepted_heading_monotonic)
        implied_rate = abs(normalize_angle_deg(heading - self._last_accepted_heading_deg)) / dt
        if implied_rate > self._max_heading_rate_deg_per_sec:
            # Don't advance the reference timestamp: if this keeps being
            # reported, growing dt against the same stale reference will
            # eventually let a real (just fast) change through instead of
            # rejecting it forever.
            return self._last_accepted_heading_deg
        self._last_accepted_heading_deg = heading
        self._last_accepted_heading_monotonic = now
        return heading

    def _remaining_checkpoint_coordinates(self) -> tuple[tuple[float, float], ...]:
        coordinates: list[tuple[float, float]] = []
        for checkpoint in self._waypoints.checkpoints[self._waypoints.index :]:
            lat, lon = _checkpoint_coordinate(checkpoint)
            if lat is not None and lon is not None:
                coordinates.append((lat, lon))
        return tuple(coordinates)

    def _filter_heading_error(
        self,
        raw_error_deg: float,
        target_sequence: int | None,
    ) -> float:
        raw_error_deg = normalize_angle_deg(raw_error_deg)
        if (
            self._filtered_heading_error_deg is None
            or target_sequence != self._filtered_target_sequence
        ):
            self._filtered_heading_error_deg = raw_error_deg
            self._filtered_target_sequence = target_sequence
            return raw_error_deg
        delta = normalize_angle_deg(raw_error_deg - self._filtered_heading_error_deg)
        if abs(delta) < self._target_heading_deadband_deg:
            return self._filtered_heading_error_deg
        if abs(delta) >= self._large_heading_change_deg:
            self._filtered_heading_error_deg = raw_error_deg
            return raw_error_deg
        self._filtered_heading_error_deg = normalize_angle_deg(
            self._filtered_heading_error_deg + self._heading_filter_alpha * delta
        )
        return self._filtered_heading_error_deg


def _checkpoint_coordinate(checkpoint: dict[str, Any]) -> tuple[float | None, float | None]:
    return (
        _latitude(checkpoint.get("latitude", checkpoint.get("lat"))),
        _longitude(checkpoint.get("longitude", checkpoint.get("lon"))),
    )


def _sequence(checkpoint: dict[str, Any]) -> int | None:
    value = safe_float(checkpoint.get("sequence"))
    return int(value) if value is not None and math.isfinite(value) else None


def _latitude(value: object) -> float | None:
    parsed = _finite(value)
    return parsed if parsed is not None and -90.0 <= parsed <= 90.0 else None


def _longitude(value: object) -> float | None:
    parsed = _finite(value)
    return parsed if parsed is not None and -180.0 <= parsed <= 180.0 else None


def _heading(value: object) -> float | None:
    parsed = _finite(value)
    return parsed % 360.0 if parsed is not None else None


def _finite(value: object) -> float | None:
    parsed = safe_float(value)
    return parsed if parsed is not None and math.isfinite(parsed) else None
