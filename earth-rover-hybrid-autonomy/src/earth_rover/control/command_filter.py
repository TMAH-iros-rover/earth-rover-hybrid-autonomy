from __future__ import annotations

import math

from earth_rover.core.types import ControlCommand
from earth_rover.utils.math_utils import clamp


class CommandFilter:
    def __init__(self, config: dict):
        control_cfg = config.get("control", {})
        self.linear_min = float(control_cfg.get("linear_min", -0.25))
        self.linear_max = float(control_cfg.get("linear_max", 0.35))
        self.angular_min = float(control_cfg.get("angular_min", -0.70))
        self.angular_max = float(control_cfg.get("angular_max", 0.70))
        self.alpha = float(control_cfg.get("command_smoothing_alpha", 0.45))
        self.max_linear_delta_per_sec = float(control_cfg.get("max_linear_delta_per_sec", 0.25))
        self.max_angular_delta_per_sec = float(control_cfg.get("max_angular_delta_per_sec", 0.80))
        self.reverse_angular_sign_slowdown = bool(
            control_cfg.get("reverse_angular_sign_slowdown", False)
        )
        self.angular_deadband = float(control_cfg.get("angular_deadband", 0.0))
        self._previous = ControlCommand(0.0, 0.0)

    def apply(
        self,
        raw_command: ControlCommand,
        dt: float,
        frame_is_stale: bool,
        data_is_stale: bool,
    ) -> ControlCommand:
        if frame_is_stale or data_is_stale:
            self._previous = ControlCommand(0.0, 0.0, mode="STALE_DATA_STOP")
            return self._previous

        if not self._valid(raw_command):
            self._previous = ControlCommand(0.0, 0.0, mode="INVALID_COMMAND_STOP")
            return self._previous

        if (
            self.reverse_angular_sign_slowdown
            and self._previous.angular * raw_command.angular < 0.0
            and abs(self._previous.angular) > self.angular_deadband
            and abs(raw_command.angular) > self.angular_deadband
        ):
            raw_command = ControlCommand(
                raw_command.linear,
                0.0,
                lamp=raw_command.lamp,
                mode=raw_command.mode,
            )

        smoothed_linear = self.alpha * self._previous.linear + (1.0 - self.alpha) * raw_command.linear
        smoothed_angular = self.alpha * self._previous.angular + (1.0 - self.alpha) * raw_command.angular

        dt = max(0.0, dt)
        max_linear_delta = self.max_linear_delta_per_sec * dt
        max_angular_delta = self.max_angular_delta_per_sec * dt
        limited_linear = self._previous.linear + clamp(
            smoothed_linear - self._previous.linear, -max_linear_delta, max_linear_delta
        )
        limited_angular = self._previous.angular + clamp(
            smoothed_angular - self._previous.angular, -max_angular_delta, max_angular_delta
        )

        command = ControlCommand(
            linear=clamp(limited_linear, self.linear_min, self.linear_max),
            angular=clamp(limited_angular, self.angular_min, self.angular_max),
            lamp=raw_command.lamp,
            mode=raw_command.mode,
        )
        self._previous = command
        return command

    @staticmethod
    def _valid(command: ControlCommand) -> bool:
        return math.isfinite(command.linear) and math.isfinite(command.angular)
