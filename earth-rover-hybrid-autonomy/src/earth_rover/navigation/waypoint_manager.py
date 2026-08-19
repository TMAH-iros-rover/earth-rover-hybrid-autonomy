from __future__ import annotations

from typing import Optional

from earth_rover.navigation.gps_utils import haversine_distance_m
from earth_rover.utils.math_utils import safe_float


class WaypointManager:
    def __init__(self, checkpoints: list[dict], switch_radius_m: float, latest_scanned_checkpoint: int = 0):
        self.checkpoints = sorted(checkpoints, key=lambda item: int(safe_float(item.get("sequence"), 0) or 0))
        self.switch_radius_m = switch_radius_m
        self.index = self._index_after_sequence(latest_scanned_checkpoint)
        self._reached_latched = False

    def current_target(self) -> Optional[dict]:
        if self.index >= len(self.checkpoints):
            return None
        return self.checkpoints[self.index]

    def update(self, latitude: float, longitude: float) -> dict:
        target = self.current_target()
        if target is None:
            return {"target": None, "distance_m": None, "reached": False, "finished": True}

        target_lat = safe_float(target.get("latitude", target.get("lat")))
        target_lon = safe_float(target.get("longitude", target.get("lon")))
        if latitude is None or longitude is None or target_lat is None or target_lon is None:
            return {"target": target, "distance_m": None, "reached": False, "finished": False}

        distance = haversine_distance_m(latitude, longitude, target_lat, target_lon)
        if distance <= self.switch_radius_m:
            self._reached_latched = True
        reached = self._reached_latched
        return {
            "target": target,
            "distance_m": distance,
            "reached": reached,
            "finished": self.index >= len(self.checkpoints),
        }

    def mark_current_reported(self) -> None:
        if self.index < len(self.checkpoints):
            self.index += 1
            self._reached_latched = False

    def _index_after_sequence(self, latest_scanned_checkpoint: int) -> int:
        latest = int(latest_scanned_checkpoint or 0)
        for index, checkpoint in enumerate(self.checkpoints):
            sequence = int(safe_float(checkpoint.get("sequence"), index + 1) or index + 1)
            if sequence > latest:
                return index
        return len(self.checkpoints)
