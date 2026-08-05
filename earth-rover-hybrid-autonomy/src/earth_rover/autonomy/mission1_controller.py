from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

import requests

from earth_rover.control.command_filter import CommandFilter
from earth_rover.core.types import ControlCommand


class MissionSdk(Protocol):
    def get_mission_status(self) -> dict[str, Any]: ...
    def send_control(self, command: ControlCommand) -> bool: ...
    def report_checkpoint_details(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class Mission1ControlConfig:
    loop_hz: float = 5.0
    sam_status_url: str = "http://127.0.0.1:8001/status"
    sam_timeout_sec: float = 0.5
    maximum_sam_age_sec: float = 0.8
    base_linear: float = 0.04
    minimum_linear: float = 0.012
    max_linear: float = 0.06
    heading_kp: float = 0.40
    max_angular: float = 0.22
    stop_forward_heading_deg: float = 50.0
    rotate_to_goal_heading_deg: float = 100.0
    rotate_exit_threshold_deg: float = 20.0
    rotate_to_goal_angular: float = 0.26
    minimum_path_score: float = 0.55
    checkpoint_report_cooldown_sec: float = 2.0
    checkpoint_transition_stop_sec: float = 0.4
    control_error_cooldown_sec: float = 2.0
    transient_invalid_grace_sec: float = 0.8
    max_plan_age_sec: float = 1.5
    path_invalid_grace_ticks: int = 4
    held_path_linear_scale: float = 0.45
    partial_path_linear_scale: float = 0.60
    relaxed_path_linear_scale: float = 0.75
    confidence_slowdown_gain: float = 0.70
    curvature_slowdown_gain: float = 0.65
    angular_deadband: float = 0.0

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "Mission1ControlConfig":
        values = config.get("mission1_autonomy", {})
        return cls(**{key: values[key] for key in cls.__dataclass_fields__ if key in values})

    def validate(self) -> None:
        positive = {
            "loop_hz": self.loop_hz,
            "sam_timeout_sec": self.sam_timeout_sec,
            "maximum_sam_age_sec": self.maximum_sam_age_sec,
            "max_linear": self.max_linear,
            "heading_kp": self.heading_kp,
            "max_angular": self.max_angular,
            "stop_forward_heading_deg": self.stop_forward_heading_deg,
            "rotate_to_goal_heading_deg": self.rotate_to_goal_heading_deg,
            "rotate_exit_threshold_deg": self.rotate_exit_threshold_deg,
            "rotate_to_goal_angular": self.rotate_to_goal_angular,
            "checkpoint_report_cooldown_sec": self.checkpoint_report_cooldown_sec,
            "checkpoint_transition_stop_sec": self.checkpoint_transition_stop_sec,
            "control_error_cooldown_sec": self.control_error_cooldown_sec,
            "transient_invalid_grace_sec": self.transient_invalid_grace_sec,
            "max_plan_age_sec": self.max_plan_age_sec,
        }
        if any(not math.isfinite(value) or value <= 0.0 for value in positive.values()):
            raise ValueError("Mission1 positive control settings must be finite and positive")
        if not 0.0 <= self.minimum_linear <= self.base_linear <= self.max_linear <= 1.0:
            raise ValueError("linear settings must satisfy 0 <= minimum <= base <= max <= 1")
        if not 0.0 <= self.minimum_path_score <= 1.0:
            raise ValueError("minimum_path_score must be in [0, 1]")
        if (
            isinstance(self.path_invalid_grace_ticks, bool)
            or not isinstance(self.path_invalid_grace_ticks, int)
            or self.path_invalid_grace_ticks < 0
        ):
            raise ValueError("path_invalid_grace_ticks must be a non-negative integer")
        for name, value in {
            "held_path_linear_scale": self.held_path_linear_scale,
            "partial_path_linear_scale": self.partial_path_linear_scale,
            "relaxed_path_linear_scale": self.relaxed_path_linear_scale,
            "confidence_slowdown_gain": self.confidence_slowdown_gain,
            "curvature_slowdown_gain": self.curvature_slowdown_gain,
        }.items():
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not math.isfinite(self.angular_deadband) or self.angular_deadband < 0.0:
            raise ValueError("angular_deadband must be finite and non-negative")


class SamStatusSource:
    def __init__(self, url: str, timeout: float) -> None:
        self.url = url
        self.timeout = timeout
        self.session = requests.Session()

    def get(self) -> dict[str, Any]:
        response = self.session.get(self.url, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("SAM-TP status must be a JSON object")
        return payload


class Mission1Autonomy:
    """Fail-closed Mission1 controller driven by the latest SAM local path.

    The runtime is armed when explicitly launched, but motion is gated by the
    SDK mission state. Consequently the dashboard's Start Mission transition
    is the only event that changes this process from waiting to driving.
    """

    def __init__(
        self,
        sdk: MissionSdk,
        sam_source: Any,
        settings: Mission1ControlConfig,
        filter_config: dict[str, Any],
        *,
        live_control_enabled: bool,
        clock: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        settings.validate()
        self.sdk = sdk
        self.sam_source = sam_source
        self.settings = settings
        self.live_control_enabled = live_control_enabled
        self.clock = clock
        self.monotonic = monotonic
        self._filter_config = filter_config
        self.command_filter = CommandFilter(filter_config)
        self._previous_tick = monotonic()
        self._lock = threading.RLock()
        self._mission_was_active = False
        self._reported_sequences: set[int] = set()
        self._last_report_time = -math.inf
        self._operator_stop_latched = False
        self._control_error_cooldown_until = -math.inf
        self._last_control_error = ""
        self._consecutive_path_invalid = 0
        self._last_valid_raw_command: ControlCommand | None = None
        self._last_valid_path_time = -math.inf
        self._last_valid_path_reason = ""
        self._last_target_sequence: int | None = None
        self._checkpoint_transition_until = -math.inf
        self._last_command = ControlCommand(0.0, 0.0, mode="STARTUP_STOP")
        self.status: dict[str, Any] = {
            "service": "mission1-autonomy",
            "armed": live_control_enabled,
            "state": "WAITING_FOR_START_MISSION" if live_control_enabled else "DRY_RUN",
            "command_transmitted": False,
            "linear": 0.0,
            "angular": 0.0,
            "reason": "Start Mission has not been pressed",
            "updated_timestamp": clock(),
        }

    def tick(self) -> dict[str, Any]:
        with self._lock:
            return self._tick()

    def _tick(self) -> dict[str, Any]:
        now_mono = self.monotonic()
        dt = max(0.0, now_mono - self._previous_tick)
        self._previous_tick = now_mono
        mission = self.sdk.get_mission_status()
        active = bool(mission.get("mission_active"))
        if active and not self._mission_was_active:
            self._reported_sequences.clear()
            self._last_report_time = -math.inf
            self._operator_stop_latched = False
            self._consecutive_path_invalid = 0
            self._last_valid_raw_command = None
            self._last_valid_path_time = -math.inf
            self._last_valid_path_reason = ""
            self._last_target_sequence = None
            self._checkpoint_transition_until = -math.inf
        self._mission_was_active = active

        if not active:
            return self._stop("WAITING_FOR_START_MISSION", "Start Mission has not been pressed")
        if not bool(mission.get("control_bridge_ready", True)):
            self._last_command = ControlCommand(
                0.0, 0.0, mode="WAITING_FOR_CONTROL_BRIDGE"
            )
            return self._publish(
                "WAITING_FOR_CONTROL_BRIDGE",
                "SDK RTC/RTM control bridge is still connecting",
                False,
            )
        if self._operator_stop_latched:
            return self._stop("OPERATOR_STOP", "operator stop is latched")
        if now_mono < self._control_error_cooldown_until:
            return self._publish(
                "WAITING_FOR_CONTROL_BRIDGE",
                f"SDK control bridge is recovering after error: {self._last_control_error}",
                False,
                ControlCommand(0.0, 0.0, mode="CONTROL_BRIDGE_RECOVERY"),
            )

        sam = self.sam_source.get()
        invalid = self._validate_sam_common(sam)
        if invalid is not None:
            return self._stop("SAFETY_STOP", invalid)
        navigation = sam["navigation"]
        if bool(navigation.get("finished")):
            return self._stop("MISSION_COMPLETE", "all checkpoints completed")

        sequence = _integer(navigation.get("target_sequence"))
        if (
            sequence is not None
            and self._last_target_sequence is not None
            and sequence != self._last_target_sequence
        ):
            self._reset_local_history()
            self._checkpoint_transition_until = (
                now_mono + self.settings.checkpoint_transition_stop_sec
            )
        if sequence is not None:
            self._last_target_sequence = sequence
        if now_mono < self._checkpoint_transition_until:
            return self._publish(
                "STARTUP_ALIGN",
                f"checkpoint transition reset; target={sequence}",
                self._send_stop("CHECKPOINT_TRANSITION_STOP"),
                ControlCommand(0.0, 0.0, mode="CHECKPOINT_TRANSITION_STOP"),
                target_sequence=sequence,
            )
        if bool(navigation.get("reached")):
            stop_transmitted = self._send_stop("CHECKPOINT_STOP")
            if sequence is None:
                return self._publish(
                    "SAFETY_STOP",
                    "reached checkpoint has no sequence",
                    stop_transmitted,
                )
            if not self.live_control_enabled:
                return self._publish(
                    "DRY_RUN_CHECKPOINT_REACHED",
                    f"would report checkpoint {sequence}; live control disabled",
                    False,
                    target_sequence=sequence,
                )
            if sequence not in self._reported_sequences:
                if now_mono - self._last_report_time < self.settings.checkpoint_report_cooldown_sec:
                    return self._publish(
                        "CHECKPOINT_WAIT",
                        f"checkpoint {sequence} report cooldown",
                        stop_transmitted,
                    )
                # Apply cooldown before the network call as well, preventing a
                # rejected cloud report from being retried at control-loop rate.
                self._last_report_time = now_mono
                response = self.sdk.report_checkpoint_details()
                self._reported_sequences.add(sequence)
                next_sequence = response.get("next_checkpoint_sequence")
                return self._publish(
                    "CHECKPOINT_REPORTED",
                    f"checkpoint {sequence} accepted; next={next_sequence or 'complete'}",
                    stop_transmitted,
                    target_sequence=sequence,
                )
            return self._publish(
                "CHECKPOINT_WAIT",
                f"waiting for SAM route to advance past checkpoint {sequence}",
                stop_transmitted,
                target_sequence=sequence,
            )

        rotate = self._rotate_to_goal_command(navigation)
        if rotate is not None:
            command = self.command_filter.apply(
                rotate, dt, frame_is_stale=False, data_is_stale=False
            )
            self._last_command = command
            heading_error = _finite(navigation.get("heading_error_deg"))
            if not self.live_control_enabled:
                return self._publish(
                    "DRY_RUN_ROTATING_TO_GOAL",
                    (
                        f"would rotate toward checkpoint {sequence}; "
                        f"heading_error={heading_error:.1f} deg"
                    ),
                    False,
                    command,
                    target_sequence=sequence,
                    heading_error_deg=heading_error,
                    controller_debug=self._controller_debug(
                        rotate,
                        command,
                        heading_error_deg=heading_error,
                    ),
                )
            if not self._try_send_control(command, now_mono):
                return self._publish(
                    "ERROR_STOP",
                    f"SDK control bridge rejected rotate command: {self._last_control_error}",
                    False,
                    ControlCommand(0.0, 0.0, mode="CONTROL_SEND_FAILED"),
                    target_sequence=sequence,
                    heading_error_deg=heading_error,
                )
            return self._publish(
                "ROTATING_TO_GOAL",
                (
                    f"rotating toward checkpoint {sequence}; "
                    f"heading_error={heading_error:.1f} deg"
                ),
                True,
                command,
                target_sequence=sequence,
                heading_error_deg=heading_error,
                controller_debug=self._controller_debug(
                    rotate,
                    command,
                    heading_error_deg=heading_error,
                ),
            )

        invalid = self._validate_path(sam)
        if invalid is not None:
            held = self._held_path_command(sam, now_mono, dt)
            if held is not None:
                self._last_command = held
                if not self.live_control_enabled:
                    return self._publish(
                        "DRY_RUN_PATH_HOLD",
                        f"would hold last local path during transient invalid path: {invalid}",
                        False,
                        held,
                        target_sequence=sequence,
                    )
                if not self._try_send_control(held, now_mono):
                    return self._publish(
                        "ERROR_STOP",
                        f"SDK control bridge rejected held command: {self._last_control_error}",
                        False,
                        ControlCommand(0.0, 0.0, mode="CONTROL_SEND_FAILED"),
                        target_sequence=sequence,
                    )
                return self._publish(
                    "PATH_HOLD",
                    f"holding last local path during transient invalid path: {invalid}",
                    True,
                    held,
                    target_sequence=sequence,
                )
            return self._stop("SAFETY_STOP", invalid)
        heading_deg = float(sam["local_path_selected_heading_deg"])
        path_score = float(sam["path_mean_score"])
        raw = self._command_from_path(
            heading_deg,
            path_score,
            str(sam.get("path_reason", "")),
            sam.get("planner") if isinstance(sam.get("planner"), dict) else None,
            navigation,
        )
        self._consecutive_path_invalid = 0
        self._last_valid_raw_command = raw
        self._last_valid_path_time = now_mono
        self._last_valid_path_reason = str(sam.get("path_reason", ""))
        command = self.command_filter.apply(raw, dt, frame_is_stale=False, data_is_stale=False)
        self._last_command = command
        if not self.live_control_enabled:
            return self._publish(
                "DRY_RUN",
                f"would track checkpoint {sequence}; live control disabled",
                False,
                command,
                target_sequence=sequence,
                local_heading_deg=heading_deg,
                path_mean_score=path_score,
                controller_debug=self._controller_debug(
                    raw,
                    command,
                    local_heading_deg=heading_deg,
                    heading_error_deg=_finite(navigation.get("heading_error_deg")),
                ),
            )
        if not self._try_send_control(command, now_mono):
            return self._publish(
                "ERROR_STOP",
                f"SDK control bridge rejected drive command: {self._last_control_error}",
                False,
                ControlCommand(0.0, 0.0, mode="CONTROL_SEND_FAILED"),
                target_sequence=sequence,
                local_heading_deg=heading_deg,
                path_mean_score=path_score,
            )
        return self._publish(
            "DRIVING",
            f"tracking checkpoint {sequence}",
            True,
            command,
            target_sequence=sequence,
            local_heading_deg=heading_deg,
            path_mean_score=path_score,
            controller_debug=self._controller_debug(
                raw,
                command,
                local_heading_deg=heading_deg,
                heading_error_deg=_finite(navigation.get("heading_error_deg")),
            ),
        )

    def fail_safe(self, error: BaseException) -> dict[str, Any]:
        with self._lock:
            return self._fail_safe(error)

    def _fail_safe(self, error: BaseException) -> dict[str, Any]:
        stop_sent = False
        if self.monotonic() >= self._control_error_cooldown_until:
            stop_sent = self._send_stop("ERROR_STOP")
        return self._publish(
            "ERROR_STOP",
            f"{type(error).__name__}: {error}",
            stop_sent,
            self._last_command,
        )

    def shutdown(self) -> None:
        with self._lock:
            if self.live_control_enabled and self._mission_was_active:
                for _ in range(2):
                    try:
                        self._send_stop("SHUTDOWN_STOP")
                    except Exception:
                        pass

    def operator_stop(self) -> dict[str, Any]:
        """Latch an immediate stop until explicitly resumed or a new mission starts."""

        with self._lock:
            self._operator_stop_latched = True
            self._send_stop("OPERATOR_STOP")
            return self._publish(
                "OPERATOR_STOP",
                "operator stop is latched",
                self.live_control_enabled and self._mission_was_active,
            )

    def operator_resume(self) -> dict[str, Any]:
        """Release the stop latch; normal mission and safety gates still apply."""

        with self._lock:
            self._operator_stop_latched = False
            return self._publish(
                "RESUME_PENDING",
                "operator stop released; validating mission and SAM-TP",
                False,
            )

    def _validate_sam_common(self, sam: dict[str, Any]) -> str | None:
        if not sam.get("ready"):
            return "SAM-TP is not ready"
        if sam.get("state") != "CLEAR":
            return f"SAM-TP state is {sam.get('state')}"
        published = _finite(sam.get("published_timestamp"))
        if published is None:
            return "SAM-TP status has no valid timestamp"
        age = self.clock() - published
        if age < -0.25 or age > self.settings.maximum_sam_age_sec:
            return f"SAM-TP status is stale ({age:.2f}s)"
        navigation = sam.get("navigation")
        if not isinstance(navigation, dict):
            return "GPS navigation is unavailable"
        if not navigation.get("gps_valid") or not navigation.get("heading_valid"):
            return "GPS position or heading is invalid"
        if not navigation.get("finished") and _integer(navigation.get("target_sequence")) is None:
            return "target checkpoint sequence is invalid"
        return None

    def _validate_path(self, sam: dict[str, Any]) -> str | None:
        planner = sam.get("planner")
        if isinstance(planner, dict):
            near_field_safe = planner.get("near_field_safe")
            if near_field_safe is False:
                return (
                    "near-field unsafe: "
                    f"{planner.get('near_field_score', sam.get('near_field_score'))}"
                )
            plan_age = _finite(planner.get("plan_age_sec"))
            if plan_age is not None and plan_age > self.settings.max_plan_age_sec:
                return f"local plan stale ({plan_age:.2f}s)"
            heading = _finite(sam.get("local_path_selected_heading_deg"))
            if heading is None:
                return "local path heading is invalid"
            confidence = _finite(planner.get("planner_confidence"))
            trajectory_valid = bool(planner.get("trajectory_valid"))
            using_held = bool(planner.get("using_held_plan"))
            if not trajectory_valid and not using_held:
                quality = _finite(planner.get("trajectory_quality"))
                if quality is None or quality < self.settings.minimum_path_score * 0.55:
                    return f"trajectory quality below cautious threshold: {quality}"
            if confidence is not None and confidence <= 0.02:
                return "planner confidence is zero"
            return None
        if not sam.get("path_valid"):
            return f"local path invalid: {sam.get('path_reason', 'unknown')}"
        heading = _finite(sam.get("local_path_selected_heading_deg"))
        if heading is None:
            return "local path heading is invalid"
        path_score = _finite(sam.get("path_mean_score"))
        if path_score is None or path_score < self.settings.minimum_path_score:
            return f"local path score below {self.settings.minimum_path_score:.2f}"
        return None

    def _command_from_path(
        self,
        heading_deg: float,
        path_score: float,
        path_reason: str = "",
        planner: dict[str, Any] | None = None,
        navigation: dict[str, Any] | None = None,
    ) -> ControlCommand:
        heading_rad = math.radians(heading_deg)
        angular = _clamp(
            self.settings.heading_kp * heading_rad,
            -self.settings.max_angular,
            self.settings.max_angular,
        )
        if abs(angular) < self.settings.angular_deadband:
            angular = 0.0
        alignment = max(0.0, math.cos(heading_rad))
        trajectory_quality = path_score
        planner_confidence = 1.0
        plan_age = 0.0
        if planner is not None:
            trajectory_quality = _finite(planner.get("trajectory_quality")) or path_score
            planner_confidence = _finite(planner.get("planner_confidence")) or 0.0
            plan_age = _finite(planner.get("plan_age_sec")) or 0.0
        score_scale = _clamp(
            (trajectory_quality - self.settings.minimum_path_score * 0.55)
            / max(1e-6, 1.0 - self.settings.minimum_path_score * 0.55),
            0.15,
            1.0,
        )
        confidence_scale = _clamp(
            1.0 - self.settings.confidence_slowdown_gain * (1.0 - planner_confidence),
            0.25,
            1.0,
        )
        curvature_scale = _clamp(
            1.0
            - self.settings.curvature_slowdown_gain
            * min(1.0, abs(heading_deg) / max(1.0, self.settings.stop_forward_heading_deg)),
            0.20,
            1.0,
        )
        freshness_scale = _clamp(
            1.0 - 0.55 * (plan_age / max(1e-6, self.settings.max_plan_age_sec)),
            0.35,
            1.0,
        )
        linear = self.settings.minimum_linear + (
            self.settings.base_linear - self.settings.minimum_linear
        ) * alignment * score_scale
        linear *= confidence_scale * curvature_scale * freshness_scale
        if abs(heading_deg) >= self.settings.stop_forward_heading_deg:
            linear = 0.0
        reason_upper = path_reason.upper()
        if "PARTIAL_" in reason_upper:
            linear *= self.settings.partial_path_linear_scale
        if "RELAXED_" in reason_upper:
            linear *= self.settings.relaxed_path_linear_scale
        return ControlCommand(
            _clamp(linear, 0.0, self.settings.max_linear),
            angular,
            mode="SAM_LOCAL_PATH_TRACKING",
        )

    def _controller_debug(
        self,
        raw: ControlCommand,
        filtered: ControlCommand,
        *,
        local_heading_deg: float | None = None,
        heading_error_deg: float | None = None,
    ) -> dict[str, Any]:
        return {
            "local_path_heading_deg": local_heading_deg,
            "local_path_heading_convention": "positive_clockwise_right",
            "navigation_heading_error_deg": heading_error_deg,
            "navigation_heading_error_convention": "positive_clockwise_right",
            "desired_angular": float(raw.angular),
            "filtered_angular": float(filtered.angular),
            "angular_convention": "positive_clockwise_right",
            "sdk_angular_convention_source": (
                "earth-rovers-sdk examples/README.md: angular -1 full left, +1 full right"
            ),
            "linear_raw": float(raw.linear),
            "linear_filtered": float(filtered.linear),
        }

    def _held_path_command(
        self,
        sam: dict[str, Any],
        now_mono: float,
        dt: float,
    ) -> ControlCommand | None:
        if self._last_valid_raw_command is None:
            return None
        planner = sam.get("planner")
        if isinstance(planner, dict) and planner.get("near_field_safe") is False:
            return None
        if now_mono - self._last_valid_path_time > self.settings.transient_invalid_grace_sec:
            return None
        self._consecutive_path_invalid += 1
        if self._consecutive_path_invalid > self.settings.path_invalid_grace_ticks:
            return None
        raw = ControlCommand(
            linear=self._last_valid_raw_command.linear
            * self.settings.held_path_linear_scale,
            angular=self._last_valid_raw_command.angular,
            lamp=self._last_valid_raw_command.lamp,
            mode="HELD_SAM_LOCAL_PATH",
        )
        return self.command_filter.apply(
            raw,
            dt,
            frame_is_stale=False,
            data_is_stale=False,
        )

    def _rotate_to_goal_command(self, navigation: dict[str, Any]) -> ControlCommand | None:
        heading_error = _finite(navigation.get("heading_error_deg"))
        if heading_error is None:
            return None
        if abs(heading_error) < self.settings.rotate_to_goal_heading_deg:
            return None
        direction = 1.0 if heading_error > 0.0 else -1.0
        angular = direction * min(
            self.settings.rotate_to_goal_angular,
            self.settings.max_angular,
        )
        return ControlCommand(0.0, angular, mode="ROTATE_TO_GOAL")

    def _reset_local_history(self) -> None:
        self._consecutive_path_invalid = 0
        self._last_valid_raw_command = None
        self._last_valid_path_time = -math.inf
        self._last_valid_path_reason = ""
        self.command_filter = CommandFilter(self._filter_config)

    def _stop(self, state: str, reason: str) -> dict[str, Any]:
        transmitted = self._send_stop(state)
        stop_command = ControlCommand(0.0, 0.0, mode=state)
        self._last_command = stop_command
        return self._publish(state, reason, transmitted, stop_command)

    def _send_stop(self, mode: str) -> bool:
        command = ControlCommand(0.0, 0.0, mode=mode)
        if self.live_control_enabled and self._mission_was_active:
            if not self._try_send_control(command, self.monotonic()):
                return False
        self._last_command = command
        return self.live_control_enabled and self._mission_was_active

    def _try_send_control(self, command: ControlCommand, now_mono: float) -> bool:
        try:
            self.sdk.send_control(command)
        except Exception as exc:
            self._last_control_error = f"{type(exc).__name__}: {exc}"
            self._control_error_cooldown_until = (
                now_mono + self.settings.control_error_cooldown_sec
            )
            self._last_command = ControlCommand(0.0, 0.0, mode="CONTROL_SEND_FAILED")
            return False
        return True

    def _publish(
        self,
        state: str,
        reason: str,
        transmitted: bool,
        command: ControlCommand | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        command = command or self._last_command
        self.status = {
            "service": "mission1-autonomy",
            "armed": self.live_control_enabled,
            "state": state,
            "command_transmitted": transmitted,
            "linear": float(command.linear),
            "angular": float(command.angular),
            "reason": reason,
            "updated_timestamp": self.clock(),
            **extra,
        }
        return dict(self.status)


def _finite(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _integer(value: object) -> int | None:
    parsed = _finite(value)
    return int(parsed) if parsed is not None and parsed.is_integer() else None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
