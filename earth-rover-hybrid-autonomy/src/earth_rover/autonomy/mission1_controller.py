from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

import requests

from earth_rover.control.command_filter import CommandFilter
from earth_rover.core.types import ControlCommand


def mission1_to_sdk_angular(angular_right_positive: float) -> float:
    """Convert Mission1 right-positive angular to Earth Rover SDK angular.

    Mission1 internal planning/control uses positive angular for physical
    right/clockwise turns.  The live Earth Rover SDK command transport observed
    on the rover uses the opposite sign: negative angular turns right and
    positive angular turns left.  Keep the sign inversion at this adapter
    boundary so planner geometry, filtering, and status remain in the internal
    convention.
    """

    return -float(angular_right_positive)


def mission1_command_to_sdk_command(command: ControlCommand) -> ControlCommand:
    return ControlCommand(
        linear=float(command.linear),
        angular=mission1_to_sdk_angular(command.angular),
        lamp=int(command.lamp),
        mode=command.mode,
    )


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
    minimum_rotate_angular: float = 0.12
    # Low-rate, delayed live control mode: stop, pulse-rotate until the local
    # path is nearly straight, then drive a short straight burst and replan.
    enable_stop_turn_go: bool = False
    stop_turn_heading_threshold_deg: float = 7.5
    stop_turn_confirm_frames: int = 2
    stop_turn_stationary_confirm_samples: int = 2
    stop_turn_stop_timeout_sec: float = 3.0
    stop_turn_rotate_angular: float = 0.12
    stop_turn_rotate_pulse_sec: float = 0.3
    stop_turn_settle_sec: float = 0.5
    stop_turn_drive_burst_sec: float = 0.6
    stop_turn_max_pulses: int = 8
    stop_turn_max_total_sec: float = 12.0
    stop_turn_cooldown_sec: float = 3.0
    minimum_path_score: float = 0.55
    checkpoint_report_cooldown_sec: float = 2.0
    checkpoint_transition_stop_sec: float = 0.4
    target_sequence_confirm_frames: int = 1
    control_error_cooldown_sec: float = 2.0
    transient_invalid_grace_sec: float = 0.8
    max_plan_age_sec: float = 1.5
    path_invalid_grace_ticks: int = 4
    path_recovery_confirm_frames: int = 1
    held_path_linear_scale: float = 0.45
    partial_path_linear_scale: float = 0.60
    relaxed_path_linear_scale: float = 0.75
    confidence_slowdown_gain: float = 0.70
    curvature_slowdown_gain: float = 0.65
    angular_deadband: float = 0.0
    require_metric_projection: bool = False
    enable_search_rotate: bool = True
    search_rotate_angular: float = 0.22
    search_rotate_timeout_sec: float = 10.0
    # ROTATE_ESCAPE: a bounded stop -> rotate -> replan recovery driven by the
    # planner's independent LEFT/RIGHT side-sector evidence (see
    # motion_primitive_planner.evaluate_side_sectors). This takes over
    # whenever the planner reports side_sector evidence; the legacy
    # SEARCH_ROTATE mechanism above stays untouched as a fallback for frames
    # that don't carry that evidence (e.g. gps_only mode).
    enable_rotate_escape: bool = False
    rotate_escape_angular: float = 0.15
    # Auxiliary *minimum dwell* only -- real stop confirmation requires
    # rotate_escape_stationary_confirm_samples of fresh, valid, stationary
    # telemetry (see STOP_CONFIRM). Both must be satisfied.
    rotate_escape_stop_confirm_ticks: int = 2
    rotate_escape_stationary_speed_threshold: float = 0.03
    rotate_escape_stationary_rpm_threshold: float = 1.0
    rotate_escape_stationary_confirm_samples: int = 2
    rotate_escape_stop_timeout_sec: float = 3.0
    rotate_escape_direction_confirm_frames: int = 2
    # Deprecated: superseded by the pulse/settle fields below. Kept only so
    # existing configs that still set it continue to parse; no longer read.
    rotate_escape_max_rotate_sec: float = 6.0
    rotate_escape_pulse_sec: float = 0.5
    rotate_escape_motion_response_timeout_sec: float = 1.5
    rotate_escape_motion_rpm_threshold: float = 1.0
    rotate_escape_motion_heading_delta_deg: float = 1.5
    rotate_escape_settle_sec: float = 0.5
    rotate_escape_max_pulses: int = 6
    rotate_escape_max_total_sec: float = 8.0
    rotate_escape_safe_frame_confirm_count: int = 3
    rotate_escape_cooldown_sec: float = 5.0

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
            "minimum_rotate_angular": self.minimum_rotate_angular,
            "stop_turn_heading_threshold_deg": self.stop_turn_heading_threshold_deg,
            "stop_turn_stop_timeout_sec": self.stop_turn_stop_timeout_sec,
            "stop_turn_rotate_angular": self.stop_turn_rotate_angular,
            "stop_turn_rotate_pulse_sec": self.stop_turn_rotate_pulse_sec,
            "stop_turn_settle_sec": self.stop_turn_settle_sec,
            "stop_turn_drive_burst_sec": self.stop_turn_drive_burst_sec,
            "stop_turn_max_total_sec": self.stop_turn_max_total_sec,
            "checkpoint_report_cooldown_sec": self.checkpoint_report_cooldown_sec,
            "checkpoint_transition_stop_sec": self.checkpoint_transition_stop_sec,
            "control_error_cooldown_sec": self.control_error_cooldown_sec,
            "transient_invalid_grace_sec": self.transient_invalid_grace_sec,
            "max_plan_age_sec": self.max_plan_age_sec,
            "search_rotate_angular": self.search_rotate_angular,
            "search_rotate_timeout_sec": self.search_rotate_timeout_sec,
            "rotate_escape_angular": self.rotate_escape_angular,
            "rotate_escape_max_rotate_sec": self.rotate_escape_max_rotate_sec,
            "rotate_escape_stop_timeout_sec": self.rotate_escape_stop_timeout_sec,
            "rotate_escape_pulse_sec": self.rotate_escape_pulse_sec,
            "rotate_escape_motion_response_timeout_sec": self.rotate_escape_motion_response_timeout_sec,
            "rotate_escape_motion_heading_delta_deg": self.rotate_escape_motion_heading_delta_deg,
            "rotate_escape_settle_sec": self.rotate_escape_settle_sec,
            "rotate_escape_max_total_sec": self.rotate_escape_max_total_sec,
        }
        if any(not math.isfinite(value) or value <= 0.0 for value in positive.values()):
            raise ValueError("Mission1 positive control settings must be finite and positive")
        if not isinstance(self.enable_rotate_escape, bool):
            raise ValueError("enable_rotate_escape must be boolean")
        if not isinstance(self.enable_stop_turn_go, bool):
            raise ValueError("enable_stop_turn_go must be boolean")
        for name in (
            "stop_turn_confirm_frames",
            "stop_turn_stationary_confirm_samples",
            "stop_turn_max_pulses",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.rotate_escape_stop_confirm_ticks, bool)
            or not isinstance(self.rotate_escape_stop_confirm_ticks, int)
            or self.rotate_escape_stop_confirm_ticks < 1
        ):
            raise ValueError("rotate_escape_stop_confirm_ticks must be a positive integer")
        if (
            isinstance(self.rotate_escape_stationary_confirm_samples, bool)
            or not isinstance(self.rotate_escape_stationary_confirm_samples, int)
            or self.rotate_escape_stationary_confirm_samples < 1
        ):
            raise ValueError("rotate_escape_stationary_confirm_samples must be a positive integer")
        if (
            isinstance(self.rotate_escape_direction_confirm_frames, bool)
            or not isinstance(self.rotate_escape_direction_confirm_frames, int)
            or self.rotate_escape_direction_confirm_frames < 1
        ):
            raise ValueError("rotate_escape_direction_confirm_frames must be a positive integer")
        if (
            isinstance(self.rotate_escape_max_pulses, bool)
            or not isinstance(self.rotate_escape_max_pulses, int)
            or self.rotate_escape_max_pulses < 1
        ):
            raise ValueError("rotate_escape_max_pulses must be a positive integer")
        for name in (
            "rotate_escape_stationary_speed_threshold",
            "rotate_escape_stationary_rpm_threshold",
            "rotate_escape_motion_rpm_threshold",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.rotate_escape_motion_response_timeout_sec < self.rotate_escape_pulse_sec:
            raise ValueError(
                "rotate_escape_motion_response_timeout_sec must be at least "
                "rotate_escape_pulse_sec"
            )
        if (
            isinstance(self.rotate_escape_safe_frame_confirm_count, bool)
            or not isinstance(self.rotate_escape_safe_frame_confirm_count, int)
            or self.rotate_escape_safe_frame_confirm_count < 1
        ):
            raise ValueError("rotate_escape_safe_frame_confirm_count must be a positive integer")
        if not math.isfinite(self.rotate_escape_cooldown_sec) or self.rotate_escape_cooldown_sec < 0.0:
            raise ValueError("rotate_escape_cooldown_sec must be finite and non-negative")
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
        if (
            isinstance(self.path_recovery_confirm_frames, bool)
            or not isinstance(self.path_recovery_confirm_frames, int)
            or self.path_recovery_confirm_frames < 1
        ):
            raise ValueError("path_recovery_confirm_frames must be a positive integer")
        if (
            isinstance(self.target_sequence_confirm_frames, bool)
            or not isinstance(self.target_sequence_confirm_frames, int)
            or self.target_sequence_confirm_frames < 1
        ):
            raise ValueError("target_sequence_confirm_frames must be a positive integer")
        if not isinstance(self.enable_search_rotate, bool):
            raise ValueError("enable_search_rotate must be boolean")
        if not isinstance(self.require_metric_projection, bool):
            raise ValueError("require_metric_projection must be boolean")
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
        if not self.rotate_exit_threshold_deg < self.rotate_to_goal_heading_deg:
            raise ValueError(
                "rotate_exit_threshold_deg must be less than rotate_to_goal_heading_deg "
                "to provide a hysteresis band"
            )
        if self.minimum_rotate_angular > min(
            self.rotate_to_goal_angular, self.max_angular
        ):
            raise ValueError(
                "minimum_rotate_angular must not exceed rotate_to_goal_angular "
                "or max_angular"
            )
        if self.stop_turn_rotate_angular > self.max_angular:
            raise ValueError("stop_turn_rotate_angular must not exceed max_angular")
        if self.stop_turn_rotate_angular < self.minimum_rotate_angular:
            raise ValueError(
                "stop_turn_rotate_angular must be at least minimum_rotate_angular"
            )
        if not math.isfinite(self.stop_turn_cooldown_sec) or self.stop_turn_cooldown_sec < 0.0:
            raise ValueError("stop_turn_cooldown_sec must be finite and non-negative")


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
        if self.settings.minimum_rotate_angular > min(
            abs(self.command_filter.angular_min), self.command_filter.angular_max
        ):
            raise ValueError(
                "minimum_rotate_angular must fit both CommandFilter angular bounds"
            )
        if self.settings.stop_turn_rotate_angular > min(
            abs(self.command_filter.angular_min), self.command_filter.angular_max
        ):
            raise ValueError(
                "stop_turn_rotate_angular must fit both CommandFilter angular bounds"
            )
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
        self._path_stop_latched = False
        self._path_recovery_count = 0
        self._last_path_recovery_frame_index: int | None = None
        self._last_target_sequence: int | None = None
        self._pending_target_sequence: int | None = None
        self._pending_target_sequence_count = 0
        self._last_sequence_frame_index: int | None = None
        self._checkpoint_transition_until = -math.inf
        self._rotating_to_goal = False
        self._rotate_direction: float | None = None
        self._searching_for_path = False
        self._search_direction: float | None = None
        self._search_started_monotonic = -math.inf
        self._recovery_phase: str | None = None
        self._recovery_direction: float | None = None
        self._recovery_stop_confirm_ticks_done = 0
        self._recovery_stop_confirm_started_monotonic: float | None = None
        self._recovery_stationary_confirm_count = 0
        self._recovery_last_stationary_sample_key: tuple[Any, Any] | None = None
        self._recovery_direction_confirm_count = 0
        self._recovery_started_monotonic: float | None = None
        self._recovery_phase_started_monotonic: float | None = None
        self._recovery_pulse_count = 0
        self._recovery_pulse_motion_observed = False
        self._recovery_pulse_max_abs_rpm = 0.0
        self._recovery_pulse_max_heading_delta_deg = 0.0
        self._recovery_pulse_start_heading_deg: float | None = None
        self._recovery_pulse_last_telemetry_key: tuple[Any, Any] | None = None
        self._recovery_last_frame_index: int | None = None
        self._recovery_safe_frame_count = 0
        self._recovery_cooldown_until = -math.inf
        self._recovery_last_direction: float | None = None
        self._recovery_rotation_reason = ""
        self._stg_phase: str | None = None
        self._stg_direction: float | None = None
        self._stg_started_monotonic: float | None = None
        self._stg_phase_started_monotonic: float | None = None
        self._stg_stop_started_monotonic: float | None = None
        self._stg_stationary_count = 0
        self._stg_last_stationary_sample_key: tuple[Any, Any] | None = None
        self._stg_confirm_count = 0
        self._stg_last_frame_index: int | None = None
        self._stg_pulse_count = 0
        self._stg_cooldown_until = -math.inf
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
            self._path_stop_latched = False
            self._path_recovery_count = 0
            self._last_path_recovery_frame_index = None
            self._last_target_sequence = None
            self._pending_target_sequence = None
            self._pending_target_sequence_count = 0
            self._last_sequence_frame_index = None
            self._checkpoint_transition_until = -math.inf
            self._rotating_to_goal = False
            self._rotate_direction = None
            self._searching_for_path = False
            self._search_direction = None
            self._search_started_monotonic = -math.inf
            self._recovery_phase = None
            self._recovery_direction = None
            self._recovery_stop_confirm_ticks_done = 0
            self._recovery_stop_confirm_started_monotonic = None
            self._recovery_stationary_confirm_count = 0
            self._recovery_last_stationary_sample_key = None
            self._recovery_direction_confirm_count = 0
            self._recovery_started_monotonic = None
            self._recovery_phase_started_monotonic = None
            self._recovery_pulse_count = 0
            self._recovery_pulse_motion_observed = False
            self._recovery_pulse_max_abs_rpm = 0.0
            self._recovery_pulse_max_heading_delta_deg = 0.0
            self._recovery_pulse_start_heading_deg = None
            self._recovery_pulse_last_telemetry_key = None
            self._recovery_last_frame_index = None
            self._recovery_safe_frame_count = 0
            self._recovery_cooldown_until = -math.inf
            self._recovery_last_direction = None
            self._recovery_rotation_reason = ""
            self._reset_stop_turn_go(clear_cooldown=True)
        self._mission_was_active = active

        if not active:
            if self.settings.enable_stop_turn_go:
                self._reset_stop_turn_go(clear_cooldown=True)
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
            if self.settings.enable_stop_turn_go:
                self._reset_stop_turn_go()
            return self._stop("SAFETY_STOP", invalid)
        navigation = sam["navigation"]
        if bool(navigation.get("finished")):
            return self._stop("MISSION_COMPLETE", "all checkpoints completed")

        sequence = _integer(navigation.get("target_sequence"))
        if sequence is not None and sequence != self._last_target_sequence:
            frame_index = _integer(sam.get("frame_index"))
            is_new_observation = (
                frame_index is None or frame_index != self._last_sequence_frame_index
            )
            if sequence != self._pending_target_sequence:
                self._pending_target_sequence = sequence
                self._pending_target_sequence_count = 0
                self._last_sequence_frame_index = None
            if is_new_observation:
                self._pending_target_sequence_count += 1
                self._last_sequence_frame_index = frame_index
            if (
                self._pending_target_sequence_count
                < self.settings.target_sequence_confirm_frames
            ):
                return self._publish(
                    "STARTUP_ALIGN",
                    "confirming target checkpoint sequence "
                    f"{sequence} ({self._pending_target_sequence_count}/"
                    f"{self.settings.target_sequence_confirm_frames})",
                    self._send_stop("TARGET_SEQUENCE_CONFIRM_STOP"),
                    ControlCommand(0.0, 0.0, mode="TARGET_SEQUENCE_CONFIRM_STOP"),
                    target_sequence=sequence,
                )
            previous_sequence = self._last_target_sequence
            self._last_target_sequence = sequence
            self._pending_target_sequence = None
            self._pending_target_sequence_count = 0
            self._last_sequence_frame_index = None
            if previous_sequence is not None:
                self._reset_local_history()
                self._checkpoint_transition_until = (
                    now_mono + self.settings.checkpoint_transition_stop_sec
                )
        elif sequence == self._last_target_sequence:
            self._pending_target_sequence = None
            self._pending_target_sequence_count = 0
            self._last_sequence_frame_index = None
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
                latest_scanned = _integer(mission.get("latest_scanned_checkpoint"))
                if latest_scanned is not None and latest_scanned >= sequence:
                    # The SDK already advanced past this checkpoint, most
                    # likely because a prior report succeeded server-side
                    # after our client gave up waiting for the response.
                    # Accept it locally instead of sending a duplicate report,
                    # which the cloud rejects with 422.
                    self._reported_sequences.add(sequence)
                    return self._publish(
                        "CHECKPOINT_REPORTED",
                        f"checkpoint {sequence} already accepted by SDK; next={sequence + 1}",
                        stop_transmitted,
                        target_sequence=sequence,
                    )
                if now_mono - self._last_report_time < self.settings.checkpoint_report_cooldown_sec:
                    return self._publish(
                        "CHECKPOINT_WAIT",
                        f"checkpoint {sequence} report cooldown",
                        stop_transmitted,
                    )
                # Apply cooldown before the network call as well, preventing a
                # rejected cloud report from being retried at control-loop rate.
                self._last_report_time = now_mono
                try:
                    response = self.sdk.report_checkpoint_details()
                except Exception as exc:
                    # Don't assume the report failed server-side too: the
                    # next tick re-checks latest_scanned_checkpoint before
                    # retrying, so a slow-but-successful cloud call is picked
                    # up without sending a second, duplicate report.
                    return self._publish(
                        "CHECKPOINT_WAIT",
                        f"checkpoint {sequence} report failed, will verify before retry: "
                        f"{type(exc).__name__}: {exc}",
                        stop_transmitted,
                        target_sequence=sequence,
                    )
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

        rotate = (
            None
            if self.settings.enable_stop_turn_go
            else self._rotate_to_goal_command(navigation)
        )
        if rotate is not None:
            command = self.command_filter.apply(
                rotate, dt, frame_is_stale=False, data_is_stale=False
            )
            if command.angular * rotate.angular < 0.0:
                # Do not transmit residual steering in the wrong direction
                # while CommandFilter is stopping before a sign reversal.
                command.angular = 0.0
            elif abs(command.angular) < self.settings.minimum_rotate_angular:
                command.angular = math.copysign(
                    self.settings.minimum_rotate_angular, rotate.angular
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
        if invalid is not None or self._recovery_phase is not None:
            if invalid is not None and self.settings.enable_stop_turn_go:
                self._reset_stop_turn_go()
            planner = sam.get("planner")
            if isinstance(planner, dict) and planner.get("switch_stop_required") is True:
                self._path_stop_latched = True
            if self._path_stop_latched:
                self._path_recovery_count = 0
                self._last_path_recovery_frame_index = None
            held = (
                self._held_path_command(sam, now_mono, dt)
                if (
                    not self.settings.enable_stop_turn_go
                    and self._recovery_phase is None
                    and invalid is not None
                )
                else None
            )
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
            recovery_result = self._rotate_escape_recovery(sam, now_mono, dt, sequence, invalid)
            if recovery_result is not None:
                return recovery_result
            if invalid is not None:
                if self.settings.enable_search_rotate and invalid.startswith("near-field unsafe"):
                    search = self._search_rotate_command(sam, now_mono)
                    if search is not None:
                        command = self.command_filter.apply(
                            search, dt, frame_is_stale=False, data_is_stale=False
                        )
                        self._last_command = command
                        if not self.live_control_enabled:
                            return self._publish(
                                "DRY_RUN_SEARCH_ROTATE",
                                f"would rotate to search for a drivable direction: {invalid}",
                                False,
                                command,
                                target_sequence=sequence,
                            )
                        if not self._try_send_control(command, now_mono):
                            return self._publish(
                                "ERROR_STOP",
                                f"SDK control bridge rejected search-rotate command: "
                                f"{self._last_control_error}",
                                False,
                                ControlCommand(0.0, 0.0, mode="CONTROL_SEND_FAILED"),
                                target_sequence=sequence,
                            )
                        return self._publish(
                            "SEARCH_ROTATE",
                            f"rotating to search for a drivable direction: {invalid}",
                            True,
                            command,
                            target_sequence=sequence,
                        )
                    # Search timed out without finding a clear direction; stop
                    # and let the operator intervene instead of spinning forever.
                    self._searching_for_path = False
                    self._search_direction = None
                return self._stop("SAFETY_STOP", invalid)
            # invalid is None but _recovery_phase was set: defensive fallback
            # (shouldn't normally happen -- see _rotate_escape_recovery's
            # continuation contract) -- fall through to ordinary tick
            # handling below rather than stop with no reason.
        if self._path_stop_latched:
            frame_index = _integer(sam.get("frame_index"))
            observation = _classify_observation(frame_index, self._last_path_recovery_frame_index)
            if observation == "REGRESSED":
                return self._stop(
                    "SAFETY_STOP",
                    "frame_index regressed during path recovery confirmation "
                    f"({frame_index} < {self._last_path_recovery_frame_index})",
                )
            if observation == "NEW":
                self._path_recovery_count += 1
                self._last_path_recovery_frame_index = frame_index
            if self._path_recovery_count < self.settings.path_recovery_confirm_frames:
                return self._stop(
                    "SAFETY_STOP",
                    "confirming safe path recovery "
                    f"({self._path_recovery_count}/"
                    f"{self.settings.path_recovery_confirm_frames})",
                )
            self._path_stop_latched = False
            self._path_recovery_count = 0
            self._last_path_recovery_frame_index = None
        if self._searching_for_path:
            self._searching_for_path = False
            self._search_direction = None
        heading_deg = float(sam["local_path_selected_heading_deg"])
        path_score = float(sam["path_mean_score"])
        if self.settings.enable_stop_turn_go:
            return self._stop_turn_go_tick(
                sam,
                now_mono,
                sequence,
                heading_deg,
                path_score,
            )
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
        if command.linear > 0.0:
            command.linear = max(self.settings.minimum_linear, command.linear)
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
            self._reset_stop_turn_go()
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
        calibration_invalid = self._validate_metric_calibration(sam)
        if calibration_invalid is not None:
            return calibration_invalid
        return None

    def _validate_metric_calibration(self, sam: dict[str, Any]) -> str | None:
        """Fail closed when the local planner is running in metric mode but
        has no validated camera calibration this frame.

        This is deliberately independent of (in addition to) the planner's
        own near_field_safe gate: it names the calibration problem directly
        instead of relying on every metric-mode failure path happening to
        also report near_field_safe=False.
        """

        planner = sam.get("planner")
        geometry_mode = planner.get("geometry_mode") if isinstance(planner, dict) else None
        if self.settings.require_metric_projection and geometry_mode != "metric_projected":
            return (
                "metric camera projection is required by the live controller, "
                f"but planner geometry_mode is {geometry_mode!r}"
            )
        if geometry_mode != "metric_projected":
            return None
        camera_projection_applied = sam.get("camera_projection_applied")
        image_path_metric_calibrated = sam.get("image_path_metric_calibrated")
        if isinstance(planner, dict):
            if camera_projection_applied is None:
                camera_projection_applied = planner.get("camera_projection_applied")
            if image_path_metric_calibrated is None:
                image_path_metric_calibrated = planner.get(
                    "image_path_metric_calibrated"
                )
        if camera_projection_applied is not True:
            return "metric camera projection is not active (no valid calibration)"
        if image_path_metric_calibrated is not True:
            return "local path is not metric-calibrated (no valid calibration)"
        return None

    def _validate_path(self, sam: dict[str, Any]) -> str | None:
        planner = sam.get("planner")
        if isinstance(planner, dict):
            if planner.get("switch_stop_required") is True:
                return f"planner switch stop required: {planner.get('switch_reason')}"
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

    def _validate_post_rotate_path(self, sam: dict[str, Any]) -> str | None:
        """Require a fresh, fully valid planner trajectory after recovery."""

        common_error = self._validate_path(sam)
        if common_error is not None:
            return common_error
        planner = sam.get("planner")
        if not isinstance(planner, dict):
            return "planner status is unavailable"
        if sam.get("path_valid") is not True:
            return "local path is not explicitly valid"
        if planner.get("near_field_safe") is not True:
            return "near-field safety is not explicitly confirmed"
        if planner.get("trajectory_valid") is not True:
            return "a new valid trajectory is not confirmed"
        if planner.get("using_held_plan") is True:
            return "held trajectory is not accepted after rotate recovery"
        if planner.get("switch_stop_required") is True:
            return "planner still requires a switch stop"
        if _integer(planner.get("selected_candidate_index")) is None:
            return "planner has no selected candidate"
        if _finite(sam.get("local_path_selected_heading_deg")) is None:
            return "local path heading is invalid"
        confidence = _finite(planner.get("planner_confidence"))
        if confidence is None or confidence <= 0.02:
            return "planner confidence is missing or zero"
        plan_age = _finite(planner.get("plan_age_sec"))
        if plan_age is None or plan_age < 0.0 or plan_age > self.settings.max_plan_age_sec:
            return "local plan age is missing or stale"
        quality = _finite(planner.get("trajectory_quality"))
        if quality is None or quality < self.settings.minimum_path_score * 0.55:
            return "trajectory quality is below the cautious threshold"
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
        # minimum_linear is the measured motor deadzone floor, not merely a
        # pre-scaling baseline. Preserve explicit stops at exactly zero, but
        # never transmit a positive command too small to turn the wheels.
        if linear > 0.0:
            linear = max(self.settings.minimum_linear, linear)
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
            "internal_angular": float(filtered.angular),
            "internal_angular_convention": "positive_right",
            "sdk_angular": mission1_to_sdk_angular(filtered.angular),
            "sdk_angular_convention": "negative_right_positive_left",
            "angular_convention": "mission1_internal_positive_clockwise_right",
            "sdk_angular_convention_source": (
                "live rover test: SDK angular < 0 turns right, SDK angular > 0 turns left"
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
        if isinstance(planner, dict) and planner.get("switch_stop_required") is True:
            return None
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
        command = self.command_filter.apply(
            raw,
            dt,
            frame_is_stale=False,
            data_is_stale=False,
        )
        if command.linear > 0.0:
            command.linear = max(self.settings.minimum_linear, command.linear)
        return command

    def _stop_turn_go_tick(
        self,
        sam: dict[str, Any],
        now_mono: float,
        sequence: int | None,
        heading_deg: float,
        path_score: float,
    ) -> dict[str, Any]:
        """Execute a bounded stop-turn-straight-burst controller.

        This mode deliberately avoids simultaneous linear/angular commands.
        Every rotation is followed by a zero-command settle and a fresh SAM
        frame; every straight burst is bounded and followed by the same
        stop/replan discipline.
        """

        frame_index = _integer(sam.get("frame_index"))
        if now_mono < self._stg_cooldown_until:
            return self._dispatch_stop_turn_go(
                "STG_COOLDOWN",
                f"stop-turn-go cooldown ({self._stg_cooldown_until - now_mono:.1f}s)",
                ControlCommand(0.0, 0.0, mode="STG_COOLDOWN"),
                now_mono,
                sequence,
                heading_deg,
                path_score,
            )

        if self._stg_phase is None:
            self._stg_started_monotonic = now_mono
            self._stg_pulse_count = 0
            self._stg_last_frame_index = frame_index
            self._enter_stop_turn_stop_confirm(now_mono, heading_deg)

        if (
            self._stg_started_monotonic is not None
            and now_mono - self._stg_started_monotonic
            > self.settings.stop_turn_max_total_sec
        ):
            return self._abort_stop_turn_go(
                now_mono,
                sequence,
                heading_deg,
                path_score,
                "stop-turn-go exceeded maximum maneuver time",
            )

        if self._stg_phase == "STOP_CONFIRM":
            return self._stop_turn_stop_confirm_tick(
                sam, now_mono, sequence, heading_deg, path_score
            )
        if self._stg_phase == "ALIGN_CONFIRM":
            return self._stop_turn_align_confirm_tick(
                now_mono, sequence, heading_deg, path_score, frame_index
            )
        if self._stg_phase == "ROTATE_PULSE":
            return self._stop_turn_rotate_tick(
                now_mono, sequence, heading_deg, path_score, frame_index
            )
        if self._stg_phase == "ROTATE_SETTLE":
            return self._stop_turn_settle_tick(
                sam, now_mono, sequence, heading_deg, path_score, frame_index
            )
        if self._stg_phase == "STRAIGHT_CONFIRM":
            return self._stop_turn_straight_confirm_tick(
                now_mono, sequence, heading_deg, path_score, frame_index
            )
        if self._stg_phase == "DRIVE_BURST":
            return self._stop_turn_drive_tick(
                now_mono, sequence, heading_deg, path_score, frame_index
            )
        if self._stg_phase == "DRIVE_SETTLE":
            return self._stop_turn_drive_settle_tick(
                sam, now_mono, sequence, heading_deg, path_score, frame_index
            )
        raise AssertionError(f"unknown stop-turn-go phase {self._stg_phase!r}")

    def _enter_stop_turn_stop_confirm(
        self, now_mono: float, heading_deg: float
    ) -> None:
        self._stg_phase = "STOP_CONFIRM"
        self._stg_direction = (
            None
            if abs(heading_deg) <= self.settings.stop_turn_heading_threshold_deg
            else (1.0 if heading_deg > 0.0 else -1.0)
        )
        self._stg_stop_started_monotonic = now_mono
        self._stg_stationary_count = 0
        self._stg_last_stationary_sample_key = None
        self._stg_confirm_count = 0
        self._stg_phase_started_monotonic = now_mono

    def _stop_turn_stationary_status(
        self, sam: dict[str, Any]
    ) -> tuple[str, str]:
        telemetry = sam.get("telemetry")
        if not isinstance(telemetry, dict) or sam.get("telemetry_valid") is not True:
            return "INVALID", "telemetry unavailable"
        age = _finite(sam.get("telemetry_age_sec"))
        if age is None or age > self.settings.maximum_sam_age_sec:
            return "INVALID", "telemetry stale"
        speed = _finite(telemetry.get("speed"))
        rpms = telemetry.get("rpms")
        if speed is None or not isinstance(rpms, list) or not rpms:
            return "INVALID", "speed/rpm unavailable"
        finite_rpms = [_finite(value) for value in rpms]
        if any(value is None for value in finite_rpms):
            return "INVALID", "rpm non-finite"
        mean_abs_rpm = sum(abs(value) for value in finite_rpms) / len(finite_rpms)
        if (
            abs(speed) > self.settings.rotate_escape_stationary_speed_threshold
            or mean_abs_rpm > self.settings.rotate_escape_stationary_rpm_threshold
        ):
            return "MOVING", f"speed={speed:.3f} rpm={mean_abs_rpm:.3f}"
        sample_key = (telemetry.get("local_timestamp"), telemetry.get("sdk_timestamp"))
        if sample_key == self._stg_last_stationary_sample_key:
            return "STATIONARY_REPEAT", "duplicate telemetry sample"
        self._stg_last_stationary_sample_key = sample_key
        return "STATIONARY_NEW", "stationary"

    def _stop_turn_stop_confirm_tick(
        self,
        sam: dict[str, Any],
        now_mono: float,
        sequence: int | None,
        heading_deg: float,
        path_score: float,
    ) -> dict[str, Any]:
        assert self._stg_stop_started_monotonic is not None
        if (
            now_mono - self._stg_stop_started_monotonic
            > self.settings.stop_turn_stop_timeout_sec
        ):
            return self._abort_stop_turn_go(
                now_mono,
                sequence,
                heading_deg,
                path_score,
                "stop-turn-go could not confirm the rover was stationary",
            )
        status, detail = self._stop_turn_stationary_status(sam)
        if status == "STATIONARY_NEW":
            self._stg_stationary_count += 1
        elif status in ("MOVING", "INVALID"):
            self._stg_stationary_count = 0
        result = self._dispatch_stop_turn_go(
            "STG_STOP_CONFIRM",
            "confirming physical stop before alignment "
            f"({self._stg_stationary_count}/"
            f"{self.settings.stop_turn_stationary_confirm_samples}; {detail})",
            ControlCommand(0.0, 0.0, mode="STG_STOP_CONFIRM"),
            now_mono,
            sequence,
            heading_deg,
            path_score,
        )
        if (
            result["state"] != "ERROR_STOP"
            and self._stg_stationary_count
            >= self.settings.stop_turn_stationary_confirm_samples
        ):
            self._stg_phase = "ALIGN_CONFIRM"
            self._stg_confirm_count = 0
            self._stg_last_frame_index = _integer(sam.get("frame_index"))
            self._stg_phase_started_monotonic = now_mono
        return result

    def _stop_turn_align_confirm_tick(
        self,
        now_mono: float,
        sequence: int | None,
        heading_deg: float,
        path_score: float,
        frame_index: int | None,
    ) -> dict[str, Any]:
        observation = _classify_observation(frame_index, self._stg_last_frame_index)
        if observation == "REGRESSED":
            return self._abort_stop_turn_go(
                now_mono, sequence, heading_deg, path_score, "SAM frame regressed during alignment"
            )
        detail = "waiting for a fresh SAM frame"
        if observation == "NEW":
            self._stg_last_frame_index = frame_index
            if abs(heading_deg) <= self.settings.stop_turn_heading_threshold_deg:
                self._stg_phase = "STRAIGHT_CONFIRM"
                self._stg_confirm_count = 1
                detail = "fresh frame is straight"
            else:
                direction = 1.0 if heading_deg > 0.0 else -1.0
                if direction != self._stg_direction:
                    self._stg_direction = direction
                    self._stg_confirm_count = 1
                else:
                    self._stg_confirm_count += 1
                detail = "turn direction confirmed"
                if self._stg_confirm_count >= self.settings.stop_turn_confirm_frames:
                    self._stg_phase = "ROTATE_PULSE"
                    self._stg_phase_started_monotonic = None
        return self._dispatch_stop_turn_go(
            "STG_ALIGN_CONFIRM",
            f"confirming local alignment direction ({self._stg_confirm_count}/"
            f"{self.settings.stop_turn_confirm_frames}; {detail})",
            ControlCommand(0.0, 0.0, mode="STG_ALIGN_CONFIRM"),
            now_mono,
            sequence,
            heading_deg,
            path_score,
        )

    def _stop_turn_rotate_tick(
        self,
        now_mono: float,
        sequence: int | None,
        heading_deg: float,
        path_score: float,
        frame_index: int | None,
    ) -> dict[str, Any]:
        assert self._stg_direction is not None
        if self._stg_phase_started_monotonic is None:
            self._stg_pulse_count += 1
            if self._stg_pulse_count > self.settings.stop_turn_max_pulses:
                return self._abort_stop_turn_go(
                    now_mono,
                    sequence,
                    heading_deg,
                    path_score,
                    "stop-turn-go exceeded maximum rotate pulses",
                )
            self._stg_phase_started_monotonic = now_mono
        elapsed = now_mono - self._stg_phase_started_monotonic
        if elapsed + 1e-9 >= self.settings.stop_turn_rotate_pulse_sec:
            self._stg_phase = "ROTATE_SETTLE"
            self._stg_phase_started_monotonic = now_mono
            self._stg_stop_started_monotonic = now_mono
            self._stg_last_frame_index = frame_index
            self._stg_stationary_count = 0
            self._stg_last_stationary_sample_key = None
            return self._dispatch_stop_turn_go(
                "STG_ROTATE_SETTLE",
                "rotation pulse complete; stopping for a fresh frame",
                ControlCommand(0.0, 0.0, mode="STG_ROTATE_SETTLE"),
                now_mono,
                sequence,
                heading_deg,
                path_score,
            )
        direction_label = "RIGHT" if self._stg_direction > 0.0 else "LEFT"
        return self._dispatch_stop_turn_go(
            f"STG_ROTATE_{direction_label}",
            f"alignment pulse {self._stg_pulse_count}/"
            f"{self.settings.stop_turn_max_pulses}",
            ControlCommand(
                0.0,
                self._stg_direction * self.settings.stop_turn_rotate_angular,
                mode="STG_ROTATE_PULSE",
            ),
            now_mono,
            sequence,
            heading_deg,
            path_score,
        )

    def _stop_turn_settle_tick(
        self,
        sam: dict[str, Any],
        now_mono: float,
        sequence: int | None,
        heading_deg: float,
        path_score: float,
        frame_index: int | None,
    ) -> dict[str, Any]:
        assert self._stg_phase_started_monotonic is not None
        elapsed = now_mono - self._stg_phase_started_monotonic
        stationary_status, stationary_detail = self._stop_turn_stationary_status(sam)
        if stationary_status == "STATIONARY_NEW":
            self._stg_stationary_count += 1
        elif stationary_status in ("MOVING", "INVALID"):
            self._stg_stationary_count = 0
        result = self._dispatch_stop_turn_go(
            "STG_ROTATE_SETTLE",
            f"settling after alignment pulse ({elapsed:.2f}/"
            f"{self.settings.stop_turn_settle_sec:.2f}s; stationary "
            f"{self._stg_stationary_count}/"
            f"{self.settings.stop_turn_stationary_confirm_samples}; "
            f"{stationary_detail})",
            ControlCommand(0.0, 0.0, mode="STG_ROTATE_SETTLE"),
            now_mono,
            sequence,
            heading_deg,
            path_score,
        )
        if elapsed < self.settings.stop_turn_settle_sec:
            return result
        assert self._stg_stop_started_monotonic is not None
        if (
            now_mono - self._stg_stop_started_monotonic
            > self.settings.stop_turn_stop_timeout_sec
        ):
            return self._abort_stop_turn_go(
                now_mono,
                sequence,
                heading_deg,
                path_score,
                "stop-turn-go could not confirm a physical stop after rotation",
            )
        if (
            self._stg_stationary_count
            < self.settings.stop_turn_stationary_confirm_samples
        ):
            return result
        observation = _classify_observation(frame_index, self._stg_last_frame_index)
        if observation == "REGRESSED":
            return self._abort_stop_turn_go(
                now_mono, sequence, heading_deg, path_score, "SAM frame regressed after rotation"
            )
        if observation != "NEW":
            return result
        self._stg_last_frame_index = frame_index
        self._stg_confirm_count = 1
        if abs(heading_deg) <= self.settings.stop_turn_heading_threshold_deg:
            self._stg_phase = "STRAIGHT_CONFIRM"
        else:
            self._stg_direction = 1.0 if heading_deg > 0.0 else -1.0
            self._stg_phase = "ALIGN_CONFIRM"
        self._stg_phase_started_monotonic = now_mono
        return result

    def _stop_turn_straight_confirm_tick(
        self,
        now_mono: float,
        sequence: int | None,
        heading_deg: float,
        path_score: float,
        frame_index: int | None,
    ) -> dict[str, Any]:
        if abs(heading_deg) > self.settings.stop_turn_heading_threshold_deg:
            self._enter_stop_turn_stop_confirm(now_mono, heading_deg)
            return self._dispatch_stop_turn_go(
                "STG_STOP_CONFIRM",
                "local path left the straight band; stopping before alignment",
                ControlCommand(0.0, 0.0, mode="STG_STOP_CONFIRM"),
                now_mono,
                sequence,
                heading_deg,
                path_score,
            )
        observation = _classify_observation(frame_index, self._stg_last_frame_index)
        if observation == "REGRESSED":
            return self._abort_stop_turn_go(
                now_mono, sequence, heading_deg, path_score, "SAM frame regressed during straight confirmation"
            )
        if observation == "NEW":
            self._stg_last_frame_index = frame_index
            self._stg_confirm_count += 1
        result = self._dispatch_stop_turn_go(
            "STG_STRAIGHT_CONFIRM",
            f"confirming straight local path ({self._stg_confirm_count}/"
            f"{self.settings.stop_turn_confirm_frames})",
            ControlCommand(0.0, 0.0, mode="STG_STRAIGHT_CONFIRM"),
            now_mono,
            sequence,
            heading_deg,
            path_score,
        )
        if self._stg_confirm_count >= self.settings.stop_turn_confirm_frames:
            self._stg_phase = "DRIVE_BURST"
            self._stg_phase_started_monotonic = now_mono
        return result

    def _stop_turn_drive_tick(
        self,
        now_mono: float,
        sequence: int | None,
        heading_deg: float,
        path_score: float,
        frame_index: int | None,
    ) -> dict[str, Any]:
        assert self._stg_phase_started_monotonic is not None
        if abs(heading_deg) > self.settings.stop_turn_heading_threshold_deg:
            self._enter_stop_turn_stop_confirm(now_mono, heading_deg)
            return self._dispatch_stop_turn_go(
                "STG_STOP_CONFIRM",
                "fresh path requires steering; stopping straight burst",
                ControlCommand(0.0, 0.0, mode="STG_STOP_CONFIRM"),
                now_mono,
                sequence,
                heading_deg,
                path_score,
            )
        elapsed = now_mono - self._stg_phase_started_monotonic
        if elapsed + 1e-9 >= self.settings.stop_turn_drive_burst_sec:
            self._stg_phase = "DRIVE_SETTLE"
            self._stg_phase_started_monotonic = now_mono
            self._stg_stop_started_monotonic = now_mono
            self._stg_last_frame_index = frame_index
            self._stg_stationary_count = 0
            self._stg_last_stationary_sample_key = None
            return self._dispatch_stop_turn_go(
                "STG_DRIVE_SETTLE",
                "straight burst complete; stopping to replan",
                ControlCommand(0.0, 0.0, mode="STG_DRIVE_SETTLE"),
                now_mono,
                sequence,
                heading_deg,
                path_score,
            )
        linear = max(
            self.settings.minimum_linear,
            min(self.settings.base_linear, self.settings.max_linear),
        )
        return self._dispatch_stop_turn_go(
            "STG_DRIVE_STRAIGHT",
            f"bounded straight burst ({elapsed:.2f}/"
            f"{self.settings.stop_turn_drive_burst_sec:.2f}s)",
            ControlCommand(linear, 0.0, mode="STG_DRIVE_STRAIGHT"),
            now_mono,
            sequence,
            heading_deg,
            path_score,
        )

    def _stop_turn_drive_settle_tick(
        self,
        sam: dict[str, Any],
        now_mono: float,
        sequence: int | None,
        heading_deg: float,
        path_score: float,
        frame_index: int | None,
    ) -> dict[str, Any]:
        assert self._stg_phase_started_monotonic is not None
        elapsed = now_mono - self._stg_phase_started_monotonic
        stationary_status, stationary_detail = self._stop_turn_stationary_status(sam)
        if stationary_status == "STATIONARY_NEW":
            self._stg_stationary_count += 1
        elif stationary_status in ("MOVING", "INVALID"):
            self._stg_stationary_count = 0
        result = self._dispatch_stop_turn_go(
            "STG_DRIVE_SETTLE",
            f"settling after straight burst ({elapsed:.2f}/"
            f"{self.settings.stop_turn_settle_sec:.2f}s; stationary "
            f"{self._stg_stationary_count}/"
            f"{self.settings.stop_turn_stationary_confirm_samples}; "
            f"{stationary_detail})",
            ControlCommand(0.0, 0.0, mode="STG_DRIVE_SETTLE"),
            now_mono,
            sequence,
            heading_deg,
            path_score,
        )
        if elapsed < self.settings.stop_turn_settle_sec:
            return result
        assert self._stg_stop_started_monotonic is not None
        if (
            now_mono - self._stg_stop_started_monotonic
            > self.settings.stop_turn_stop_timeout_sec
        ):
            return self._abort_stop_turn_go(
                now_mono,
                sequence,
                heading_deg,
                path_score,
                "stop-turn-go could not confirm a physical stop after straight burst",
            )
        if (
            self._stg_stationary_count
            < self.settings.stop_turn_stationary_confirm_samples
        ):
            return result
        observation = _classify_observation(frame_index, self._stg_last_frame_index)
        if observation == "REGRESSED":
            return self._abort_stop_turn_go(
                now_mono, sequence, heading_deg, path_score, "SAM frame regressed after straight burst"
            )
        if observation != "NEW":
            return result
        self._stg_last_frame_index = frame_index
        self._stg_started_monotonic = now_mono
        self._stg_pulse_count = 0
        if abs(heading_deg) <= self.settings.stop_turn_heading_threshold_deg:
            self._stg_phase = "STRAIGHT_CONFIRM"
            self._stg_confirm_count = 1
            self._stg_phase_started_monotonic = now_mono
        else:
            self._enter_stop_turn_stop_confirm(now_mono, heading_deg)
        return result

    def _stop_turn_go_status(self) -> dict[str, Any]:
        direction = None
        if self._stg_direction is not None:
            direction = "RIGHT" if self._stg_direction > 0.0 else "LEFT"
        elapsed = None
        if self._stg_started_monotonic is not None:
            elapsed = max(0.0, self.monotonic() - self._stg_started_monotonic)
        return {
            "phase": self._stg_phase,
            "direction": direction,
            "pulse_count": self._stg_pulse_count,
            "confirm_count": self._stg_confirm_count,
            "confirm_required": self.settings.stop_turn_confirm_frames,
            "stationary_count": self._stg_stationary_count,
            "stationary_required": self.settings.stop_turn_stationary_confirm_samples,
            "elapsed_sec": elapsed,
        }

    def _dispatch_stop_turn_go(
        self,
        state: str,
        reason: str,
        command: ControlCommand,
        now_mono: float,
        sequence: int | None,
        heading_deg: float,
        path_score: float,
    ) -> dict[str, Any]:
        self._last_command = command
        published_state = state if self.live_control_enabled else f"DRY_RUN_{state}"
        if not self.live_control_enabled:
            return self._publish(
                published_state,
                reason,
                False,
                command,
                target_sequence=sequence,
                local_heading_deg=heading_deg,
                path_mean_score=path_score,
                stop_turn_go=self._stop_turn_go_status(),
            )
        if not self._try_send_control(command, now_mono):
            return self._publish(
                "ERROR_STOP",
                f"SDK control bridge rejected stop-turn-go command: {self._last_control_error}",
                False,
                ControlCommand(0.0, 0.0, mode="CONTROL_SEND_FAILED"),
                target_sequence=sequence,
                stop_turn_go=self._stop_turn_go_status(),
            )
        return self._publish(
            published_state,
            reason,
            True,
            command,
            target_sequence=sequence,
            local_heading_deg=heading_deg,
            path_mean_score=path_score,
            stop_turn_go=self._stop_turn_go_status(),
        )

    def _abort_stop_turn_go(
        self,
        now_mono: float,
        sequence: int | None,
        heading_deg: float,
        path_score: float,
        reason: str,
    ) -> dict[str, Any]:
        self._stg_cooldown_until = now_mono + self.settings.stop_turn_cooldown_sec
        self._reset_stop_turn_go()
        return self._dispatch_stop_turn_go(
            "STG_SAFETY_STOP",
            reason,
            ControlCommand(0.0, 0.0, mode="STG_SAFETY_STOP"),
            now_mono,
            sequence,
            heading_deg,
            path_score,
        )

    def _reset_stop_turn_go(self, *, clear_cooldown: bool = False) -> None:
        self._stg_phase = None
        self._stg_direction = None
        self._stg_started_monotonic = None
        self._stg_phase_started_monotonic = None
        self._stg_stop_started_monotonic = None
        self._stg_stationary_count = 0
        self._stg_last_stationary_sample_key = None
        self._stg_confirm_count = 0
        self._stg_last_frame_index = None
        self._stg_pulse_count = 0
        if clear_cooldown:
            self._stg_cooldown_until = -math.inf

    def _rotate_to_goal_command(self, navigation: dict[str, Any]) -> ControlCommand | None:
        heading_error = _finite(navigation.get("heading_error_deg"))
        if heading_error is None:
            self._rotating_to_goal = False
            self._rotate_direction = None
            return None
        magnitude = abs(heading_error)
        # Enter at rotate_to_goal_heading_deg but only exit once the error
        # drops below the lower rotate_exit_threshold_deg. Without this
        # hysteresis band, a heading_error hovering near the entry threshold
        # (or overshooting past zero from the fixed-rate turn below) flips
        # this mode on and off every tick, which shows up as the rover
        # spinning back and forth instead of settling.
        if self._rotating_to_goal:
            if magnitude < self.settings.rotate_exit_threshold_deg:
                self._rotating_to_goal = False
                self._rotate_direction = None
                return None
        else:
            if magnitude < self.settings.rotate_to_goal_heading_deg:
                return None
            self._rotating_to_goal = True
            self._rotate_direction = None
        if self._rotate_direction is None:
            # Latch the direction once, at the moment we commit to rotating,
            # and keep it for the rest of this rotate streak instead of
            # recomputing sign(heading_error) every tick. When the target is
            # nearly straight behind (~180 deg), GPS/heading noise can push
            # the raw error across the +/-180 wrap point (e.g. +179 -> -179),
            # which is a tiny real heading change but flips this sign every
            # time it happens. Recomputing direction each tick made the
            # rover reverse mid-turn and get stuck oscillating instead of
            # completing the turn.
            self._rotate_direction = 1.0 if heading_error > 0.0 else -1.0
        direction = self._rotate_direction
        # Taper the rate down as the error approaches the exit band so the
        # rotation doesn't overshoot past zero error and immediately
        # re-trigger in the opposite direction.
        span = max(
            1e-6,
            self.settings.rotate_to_goal_heading_deg - self.settings.rotate_exit_threshold_deg,
        )
        taper = _clamp((magnitude - self.settings.rotate_exit_threshold_deg) / span, 0.25, 1.0)
        angular_limit = min(
            self.settings.rotate_to_goal_angular,
            self.settings.max_angular,
        )
        angular = direction * max(
            self.settings.minimum_rotate_angular,
            angular_limit * taper,
        )
        return ControlCommand(0.0, angular, mode="ROTATE_TO_GOAL")

    def _search_rotate_command(
        self, sam: dict[str, Any], now_mono: float
    ) -> ControlCommand | None:
        """Rotate in place to bring a new view into frame when every forward
        candidate is blocked, instead of just sitting in front of an obstacle.

        The local planner only scores image-space curves drawn on the
        current camera frame, so if an obstacle fills the whole
        +/-maximum_visual_heading_deg field of view there is no candidate
        that can route around it from where the rover is currently facing --
        it has to physically turn before a clear direction even exists to
        evaluate. Returns None once the search has run for too long without
        finding one, so the caller falls back to a real SAFETY_STOP rather
        than spinning forever.
        """

        if not self._searching_for_path:
            self._searching_for_path = True
            self._search_started_monotonic = now_mono
            self._search_direction = self._pick_search_direction(sam)
        elif now_mono - self._search_started_monotonic > self.settings.search_rotate_timeout_sec:
            return None
        angular = self._search_direction * min(
            self.settings.search_rotate_angular, self.settings.max_angular
        )
        return ControlCommand(0.0, angular, mode="SEARCH_ROTATE")

    @staticmethod
    def _pick_search_direction(sam: dict[str, Any]) -> float:
        """Turn toward whichever side looked least blocked, if we know."""

        planner = sam.get("planner")
        candidates = planner.get("candidate_scores") if isinstance(planner, dict) else None
        if isinstance(candidates, list):
            left_scores = []
            right_scores = []
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                heading = _finite(candidate.get("heading_deg"))
                near_field = _finite(candidate.get("near_field"))
                if heading is None or near_field is None:
                    continue
                if heading < 0:
                    left_scores.append(near_field)
                elif heading > 0:
                    right_scores.append(near_field)
            if left_scores or right_scores:
                left_best = max(left_scores) if left_scores else -1.0
                right_best = max(right_scores) if right_scores else -1.0
                if left_best > right_best:
                    return -1.0
                if right_best > left_best:
                    return 1.0
        return 1.0

    def _rotate_escape_recovery(
        self,
        sam: dict[str, Any],
        now_mono: float,
        dt: float,
        sequence: int | None,
        invalid: str | None,
    ) -> dict[str, Any] | None:
        """Bounded stop -> rotate(pulse) -> replan recovery driven by the
        planner's independent LEFT/RIGHT side-sector evidence.

        Returns None to defer entirely to the legacy held-path/SEARCH_ROTATE/
        SAFETY_STOP chain in ``_tick`` -- either because this mechanism is
        disabled, the current frame carries no side-sector evidence (e.g.
        gps_only mode, or a hand-built status dict from before this feature
        existed), or there is simply nothing to recover from. Once a
        recovery is in progress this function always returns a real status
        dict and owns every tick until the recovery completes or aborts.
        """

        planner = sam.get("planner")
        planner_dict = planner if isinstance(planner, dict) else {}
        side_sector = planner_dict.get("side_sector")
        frame_index = _integer(sam.get("frame_index"))

        if self._recovery_phase is None:
            if (
                not self.settings.enable_rotate_escape
                or not isinstance(side_sector, dict)
                or invalid is None
                or not self._is_blocked_path_reason(planner_dict)
            ):
                return None
            if now_mono < self._recovery_cooldown_until:
                # Cooldown blocks re-entry in *either* direction, not just
                # the opposite one -- a repeated same-direction attempt
                # right after an abort is exactly the oscillation/flapping
                # pattern this gate exists to prevent.
                return self._recovery_cooldown_refusal(now_mono, sequence, side_sector)
            chosen = side_sector.get("chosen")
            if chosen not in ("LEFT", "RIGHT"):
                return self._stop(
                    "SAFETY_STOP",
                    f"side sector {side_sector.get('status')}: {side_sector.get('reason')}",
                )
            direction = 1.0 if chosen == "RIGHT" else -1.0
            self._recovery_phase = "STOP_CONFIRM"
            self._recovery_direction = direction
            self._recovery_stop_confirm_ticks_done = 0
            self._recovery_stop_confirm_started_monotonic = now_mono
            self._recovery_stationary_confirm_count = 0
            self._recovery_last_stationary_sample_key = None
            self._recovery_direction_confirm_count = 0
            self._recovery_started_monotonic = now_mono
            self._recovery_phase_started_monotonic = None
            self._recovery_pulse_count = 0
            self._recovery_pulse_motion_observed = False
            self._recovery_pulse_max_abs_rpm = 0.0
            self._recovery_pulse_max_heading_delta_deg = 0.0
            self._recovery_pulse_start_heading_deg = None
            self._recovery_pulse_last_telemetry_key = None
            self._recovery_safe_frame_count = 0
            # Seed with the frame_index observed *now*, at entry, not None --
            # otherwise the first settle-exit check would treat this same
            # pre-rotation frame as freshly-new post-rotation evidence (see
            # _classify_observation: MISSING/None never counts as new, but an
            # unseeded None "last" value makes any concrete frame_index count
            # as new on first comparison).
            self._recovery_last_frame_index = frame_index
            self._recovery_rotation_reason = str(side_sector.get("reason", ""))

        if not isinstance(side_sector, dict):
            return self._abort_recovery(now_mono, sequence, None, "side sector evidence unavailable during recovery")

        if self._recovery_phase == "STOP_CONFIRM":
            return self._recovery_stop_confirm_tick(sam, now_mono, sequence, side_sector)
        if self._recovery_phase == "DIRECTION_CONFIRM":
            return self._recovery_direction_confirm_tick(
                now_mono, sequence, side_sector, frame_index
            )
        if self._recovery_phase == "ROTATE_PULSE":
            return self._recovery_pulse_tick(
                sam, now_mono, dt, sequence, side_sector, frame_index
            )
        if self._recovery_phase == "ROTATE_SETTLE":
            return self._recovery_settle_tick(sam, now_mono, sequence, side_sector, frame_index)
        if self._recovery_phase == "POST_ROTATE_REPLAN":
            return self._recovery_post_rotate_tick(sam, now_mono, sequence, frame_index)
        raise AssertionError(f"unknown recovery phase {self._recovery_phase!r}")

    @staticmethod
    def _is_blocked_path_reason(planner: dict[str, Any]) -> bool:
        return planner.get("near_field_safe") is False or planner.get("switch_stop_required") is True

    def _recovery_direction_label(self) -> str:
        return "RIGHT" if (self._recovery_direction or 0.0) > 0.0 else "LEFT"

    def _cooldown_remaining_sec(self, now_mono: float) -> float:
        return max(0.0, self._recovery_cooldown_until - now_mono)

    def _recovery_status(self, side_sector: dict[str, Any] | None, *, phase: str) -> dict[str, Any]:
        direction = None
        if self._recovery_direction is not None:
            direction = "RIGHT" if self._recovery_direction > 0.0 else "LEFT"
        elapsed = None
        if self._recovery_phase is not None and self._recovery_started_monotonic is not None:
            elapsed = max(0.0, self.monotonic() - self._recovery_started_monotonic)
        return {
            "maneuver_type": "ROTATE_THEN_REPLAN",
            "maneuver_direction": direction,
            "maneuver_phase": phase,
            "rotation_reason": self._recovery_rotation_reason,
            "side_sector": side_sector,
            "recovery_elapsed_sec": elapsed,
            "safe_frame_confirm_count": self._recovery_safe_frame_count,
            "safe_frame_confirm_required": self.settings.rotate_escape_safe_frame_confirm_count,
            "stationary_confirm_count": self._recovery_stationary_confirm_count,
            "stationary_confirm_required": self.settings.rotate_escape_stationary_confirm_samples,
            "direction_confirm_count": self._recovery_direction_confirm_count,
            "direction_confirm_required": self.settings.rotate_escape_direction_confirm_frames,
            "pulse_count": self._recovery_pulse_count,
            "pulse_count_max": self.settings.rotate_escape_max_pulses,
            "pulse_motion_observed": self._recovery_pulse_motion_observed,
            "pulse_max_abs_rpm": self._recovery_pulse_max_abs_rpm,
            "pulse_max_heading_delta_deg": self._recovery_pulse_max_heading_delta_deg,
            "cooldown_remaining_sec": self._cooldown_remaining_sec(self.monotonic()),
        }

    @staticmethod
    def _recovery_heading_deg(sam: dict[str, Any]) -> float | None:
        localization = sam.get("localization")
        if isinstance(localization, dict):
            heading = _finite(localization.get("fused_heading_deg"))
            if heading is not None:
                return heading
        telemetry = sam.get("telemetry")
        if isinstance(telemetry, dict):
            return _finite(telemetry.get("orientation"))
        return None

    def _observe_recovery_pulse_motion(self, sam: dict[str, Any]) -> None:
        telemetry = sam.get("telemetry")
        if isinstance(telemetry, dict) and sam.get("telemetry_valid") is True:
            sample_key = (
                telemetry.get("local_timestamp"),
                telemetry.get("sdk_timestamp"),
            )
            if sample_key != self._recovery_pulse_last_telemetry_key:
                self._recovery_pulse_last_telemetry_key = sample_key
                rpms = telemetry.get("rpms")
                if isinstance(rpms, list) and rpms:
                    finite_rpms = [_finite(value) for value in rpms]
                    if all(value is not None for value in finite_rpms):
                        max_abs_rpm = max(abs(value) for value in finite_rpms)
                        self._recovery_pulse_max_abs_rpm = max(
                            self._recovery_pulse_max_abs_rpm,
                            max_abs_rpm,
                        )
                        if max_abs_rpm >= self.settings.rotate_escape_motion_rpm_threshold:
                            self._recovery_pulse_motion_observed = True

        heading = self._recovery_heading_deg(sam)
        if heading is None:
            return
        if self._recovery_pulse_start_heading_deg is None:
            self._recovery_pulse_start_heading_deg = heading
            return
        heading_delta = abs(
            (heading - self._recovery_pulse_start_heading_deg + 180.0) % 360.0
            - 180.0
        )
        self._recovery_pulse_max_heading_delta_deg = max(
            self._recovery_pulse_max_heading_delta_deg,
            heading_delta,
        )
        if heading_delta >= self.settings.rotate_escape_motion_heading_delta_deg:
            self._recovery_pulse_motion_observed = True

    def _recovery_cooldown_refusal(
        self, now_mono: float, sequence: int | None, side_sector: dict[str, Any] | None
    ) -> dict[str, Any]:
        command = ControlCommand(0.0, 0.0, mode="RECOVERY_COOLDOWN")
        transmitted = self._send_stop("RECOVERY_COOLDOWN")
        self._last_command = command
        return self._publish(
            "RECOVERY_COOLDOWN",
            "recovery cooldown active "
            f"({self._cooldown_remaining_sec(now_mono):.1f}s remaining); refusing new rotate escape",
            transmitted,
            command,
            target_sequence=sequence,
            recovery=self._recovery_status(side_sector, phase="RECOVERY_COOLDOWN"),
        )

    def _dispatch_recovery_command(
        self,
        state: str,
        reason: str,
        command: ControlCommand,
        now_mono: float,
        sequence: int | None,
        side_sector: dict[str, Any] | None,
        phase: str,
    ) -> dict[str, Any]:
        self._last_command = command
        recovery_meta = self._recovery_status(side_sector, phase=phase)
        if not self.live_control_enabled:
            return self._publish(
                f"DRY_RUN_{state}", reason, False, command, target_sequence=sequence, recovery=recovery_meta
            )
        if not self._try_send_control(command, now_mono):
            return self._publish(
                "ERROR_STOP",
                f"SDK control bridge rejected {state.lower()} command: {self._last_control_error}",
                False,
                ControlCommand(0.0, 0.0, mode="CONTROL_SEND_FAILED"),
                target_sequence=sequence,
                recovery=recovery_meta,
            )
        return self._publish(state, reason, True, command, target_sequence=sequence, recovery=recovery_meta)

    def _telemetry_stationary_status(self, sam: dict[str, Any]) -> tuple[str, str]:
        """Classify this tick's telemetry sample for STOP_CONFIRM.

        Returns (status, detail) where status is one of "STATIONARY_NEW"
        (a fresh sample confirms the rover is stationary -- counts),
        "STATIONARY_REPEAT" (same sample polled again -- doesn't count, but
        doesn't reset progress either), "MOVING" (fresh sample shows motion
        above threshold -- resets progress), or "INVALID" (missing/stale/
        non-finite telemetry -- fails closed, resets progress).
        """

        telemetry = sam.get("telemetry")
        if not isinstance(telemetry, dict):
            return "INVALID", "no telemetry"
        if sam.get("telemetry_valid") is not True:
            return "INVALID", "telemetry invalid"
        age = _finite(sam.get("telemetry_age_sec"))
        if age is None or age > self.settings.maximum_sam_age_sec:
            return "INVALID", "telemetry stale"
        speed = _finite(telemetry.get("speed"))
        rpms = telemetry.get("rpms")
        if speed is None or not isinstance(rpms, list) or not rpms:
            return "INVALID", "speed/rpm unavailable"
        finite_rpms = [_finite(value) for value in rpms]
        if any(value is None for value in finite_rpms):
            return "INVALID", "rpm non-finite"
        mean_abs_rpm = sum(abs(value) for value in finite_rpms) / len(finite_rpms)
        if (
            abs(speed) > self.settings.rotate_escape_stationary_speed_threshold
            or mean_abs_rpm > self.settings.rotate_escape_stationary_rpm_threshold
        ):
            return "MOVING", f"speed={speed:.3f} rpm={mean_abs_rpm:.3f}"
        sample_key = (telemetry.get("local_timestamp"), telemetry.get("sdk_timestamp"))
        if sample_key == self._recovery_last_stationary_sample_key:
            return "STATIONARY_REPEAT", "duplicate telemetry sample"
        self._recovery_last_stationary_sample_key = sample_key
        return "STATIONARY_NEW", "stationary"

    def _recovery_stop_confirm_tick(
        self, sam: dict[str, Any], now_mono: float, sequence: int | None, side_sector: dict[str, Any]
    ) -> dict[str, Any]:
        assert self._recovery_stop_confirm_started_monotonic is not None
        if now_mono - self._recovery_stop_confirm_started_monotonic > self.settings.rotate_escape_stop_timeout_sec:
            return self._abort_recovery(
                now_mono, sequence, side_sector, "stop confirmation timed out without confirmed-stationary telemetry"
            )
        self._recovery_stop_confirm_ticks_done += 1
        status, detail = self._telemetry_stationary_status(sam)
        if status == "STATIONARY_NEW":
            self._recovery_stationary_confirm_count += 1
        elif status in ("MOVING", "INVALID"):
            self._recovery_stationary_confirm_count = 0
        command = ControlCommand(0.0, 0.0, mode="STOP_CONFIRM")
        result = self._dispatch_recovery_command(
            "STOP_CONFIRM",
            f"confirming stop before {self._recovery_direction_label().lower()} rotate escape "
            f"(stationary {self._recovery_stationary_confirm_count}/"
            f"{self.settings.rotate_escape_stationary_confirm_samples}; {detail})",
            command,
            now_mono,
            sequence,
            side_sector,
            "STOP_CONFIRM",
        )
        if result["state"] == "ERROR_STOP":
            return result
        if (
            self._recovery_stop_confirm_ticks_done >= self.settings.rotate_escape_stop_confirm_ticks
            and self._recovery_stationary_confirm_count >= self.settings.rotate_escape_stationary_confirm_samples
        ):
            self._recovery_phase = "DIRECTION_CONFIRM"
            self._recovery_phase_started_monotonic = now_mono
            self._recovery_direction_confirm_count = 0
            self._recovery_last_frame_index = _integer(sam.get("frame_index"))
        return result

    def _recovery_direction_confirm_tick(
        self,
        now_mono: float,
        sequence: int | None,
        side_sector: dict[str, Any],
        frame_index: int | None,
    ) -> dict[str, Any]:
        assert self._recovery_started_monotonic is not None
        if now_mono - self._recovery_started_monotonic > self.settings.rotate_escape_max_total_sec:
            return self._abort_recovery(
                now_mono,
                sequence,
                side_sector,
                "rotate escape timed out while confirming a fresh side direction",
            )
        observation = _classify_observation(frame_index, self._recovery_last_frame_index)
        if observation == "REGRESSED":
            return self._abort_recovery(
                now_mono,
                sequence,
                side_sector,
                f"frame_index regressed during direction confirmation "
                f"({frame_index} < {self._recovery_last_frame_index})",
            )
        detail = "waiting for a new SAM frame"
        if observation == "NEW":
            self._recovery_last_frame_index = frame_index
            side_key = self._recovery_direction_label().lower()
            side = side_sector.get(side_key)
            if (
                isinstance(side, dict)
                and side.get("viable", False)
                and side_sector.get("chosen") == self._recovery_direction_label()
            ):
                self._recovery_direction_confirm_count += 1
                detail = "direction confirmed on fresh frame"
            else:
                self._recovery_direction_confirm_count = 0
                detail = "fresh frame did not confirm the selected direction"
        command = ControlCommand(0.0, 0.0, mode="DIRECTION_CONFIRM")
        result = self._dispatch_recovery_command(
            "DIRECTION_CONFIRM",
            f"confirming fresh {self._recovery_direction_label().lower()} side evidence "
            f"({self._recovery_direction_confirm_count}/"
            f"{self.settings.rotate_escape_direction_confirm_frames}; {detail})",
            command,
            now_mono,
            sequence,
            side_sector,
            "DIRECTION_CONFIRM",
        )
        if result["state"] == "ERROR_STOP":
            return result
        if (
            self._recovery_direction_confirm_count
            >= self.settings.rotate_escape_direction_confirm_frames
        ):
            self._recovery_phase = "ROTATE_PULSE"
            self._recovery_phase_started_monotonic = None
        return result

    def _recovery_pulse_tick(
        self,
        sam: dict[str, Any],
        now_mono: float,
        dt: float,
        sequence: int | None,
        side_sector: dict[str, Any],
        frame_index: int | None,
    ) -> dict[str, Any]:
        assert self._recovery_started_monotonic is not None
        if now_mono - self._recovery_started_monotonic > self.settings.rotate_escape_max_total_sec:
            return self._abort_recovery(
                now_mono, sequence, side_sector, "rotate escape exceeded max total recovery time"
            )
        if self._recovery_phase_started_monotonic is None:
            # Starting a new pulse: the chosen side must still be viable
            # right now, and this pulse must not exceed the pulse budget.
            side_key = self._recovery_direction_label().lower()
            side = side_sector.get(side_key)
            if (
                not isinstance(side, dict)
                or not side.get("viable", False)
                or side_sector.get("chosen") != self._recovery_direction_label()
            ):
                return self._abort_recovery(
                    now_mono,
                    sequence,
                    side_sector,
                    f"{self._recovery_direction_label().lower()} side sector no longer viable before pulse",
                )
            self._recovery_pulse_count += 1
            if self._recovery_pulse_count > self.settings.rotate_escape_max_pulses:
                return self._abort_recovery(now_mono, sequence, side_sector, "rotate escape exceeded max pulse count")
            self._recovery_phase_started_monotonic = now_mono
            self._recovery_pulse_motion_observed = False
            self._recovery_pulse_max_abs_rpm = 0.0
            self._recovery_pulse_max_heading_delta_deg = 0.0
            self._recovery_pulse_start_heading_deg = self._recovery_heading_deg(sam)
            self._recovery_pulse_last_telemetry_key = None
        self._observe_recovery_pulse_motion(sam)
        elapsed = now_mono - self._recovery_phase_started_monotonic
        if (
            elapsed + 1e-9 >= self.settings.rotate_escape_pulse_sec
            and self._recovery_pulse_motion_observed
        ):
            self._recovery_phase = "ROTATE_SETTLE"
            self._recovery_phase_started_monotonic = now_mono
            return self._recovery_settle_tick(
                sam, now_mono, sequence, side_sector, frame_index
            )
        if elapsed + 1e-9 >= self.settings.rotate_escape_motion_response_timeout_sec:
            return self._abort_recovery(
                now_mono,
                sequence,
                side_sector,
                "rotate escape actuator did not respond "
                f"(max_rpm={self._recovery_pulse_max_abs_rpm:.2f}, "
                f"heading_delta={self._recovery_pulse_max_heading_delta_deg:.2f}deg)",
            )
        angular_limit = min(self.settings.rotate_escape_angular, self.settings.max_angular)
        raw = ControlCommand(0.0, self._recovery_direction * angular_limit, mode="ROTATE_PULSE")
        command = self.command_filter.apply(raw, dt, frame_is_stale=False, data_is_stale=False)
        # linear must stay hard zero regardless of any slew/filter state.
        command.linear = 0.0
        state = "ROTATE_PULSE_RIGHT" if self._recovery_direction > 0.0 else "ROTATE_PULSE_LEFT"
        result = self._dispatch_recovery_command(
            state,
            f"pulse {self._recovery_pulse_count}/{self.settings.rotate_escape_max_pulses} "
            f"{self._recovery_direction_label().lower()} "
            f"({elapsed:.2f}/{self.settings.rotate_escape_pulse_sec:.2f}s; "
            f"motion={'yes' if self._recovery_pulse_motion_observed else 'no'})",
            command,
            now_mono,
            sequence,
            side_sector,
            "ROTATE_PULSE",
        )
        if result["state"] == "ERROR_STOP":
            return result
        return result

    def _recovery_settle_tick(
        self,
        sam: dict[str, Any],
        now_mono: float,
        sequence: int | None,
        side_sector: dict[str, Any],
        frame_index: int | None,
    ) -> dict[str, Any]:
        assert self._recovery_started_monotonic is not None
        if now_mono - self._recovery_started_monotonic > self.settings.rotate_escape_max_total_sec:
            return self._abort_recovery(
                now_mono, sequence, side_sector, "rotate escape exceeded max total recovery time"
            )
        assert self._recovery_phase_started_monotonic is not None
        elapsed = now_mono - self._recovery_phase_started_monotonic
        command = ControlCommand(0.0, 0.0, mode="ROTATE_SETTLE")
        result = self._dispatch_recovery_command(
            "ROTATE_SETTLE",
            f"settling after pulse {self._recovery_pulse_count}/{self.settings.rotate_escape_max_pulses} "
            f"({elapsed:.2f}/{self.settings.rotate_escape_settle_sec:.2f}s)",
            command,
            now_mono,
            sequence,
            side_sector,
            "ROTATE_SETTLE",
        )
        if result["state"] == "ERROR_STOP":
            return result
        if elapsed < self.settings.rotate_escape_settle_sec:
            return result
        # Settle duration alone is not enough -- a genuinely new, distinct
        # SAM frame must also have arrived before re-planning off it.
        observation = _classify_observation(frame_index, self._recovery_last_frame_index)
        if observation == "REGRESSED":
            return self._abort_recovery(
                now_mono,
                sequence,
                side_sector,
                f"frame_index regressed during settle ({frame_index} < {self._recovery_last_frame_index})",
            )
        if observation != "NEW":
            # No fresh frame yet -- keep settling; bounded by
            # rotate_escape_max_total_sec above so this can't hang forever.
            return result
        self._recovery_last_frame_index = frame_index
        # D: a complete safe forward path takes priority over side-sector
        # degradation -- if one exists now, stop rotating immediately rather
        # than judging whether the (possibly now-AMBIGUOUS) side sector is
        # still "the" chosen direction.
        if self._validate_post_rotate_path(sam) is None:
            self._recovery_phase = "POST_ROTATE_REPLAN"
            self._recovery_phase_started_monotonic = None
            self._recovery_safe_frame_count = 0
            self._recovery_last_frame_index = None
            return result
        side_key = self._recovery_direction_label().lower()
        side = side_sector.get(side_key)
        opposite_label = "LEFT" if self._recovery_direction_label() == "RIGHT" else "RIGHT"
        if not isinstance(side, dict) or not side.get("viable", False):
            return self._abort_recovery(
                now_mono, sequence, side_sector, f"{self._recovery_direction_label().lower()} side sector no longer viable"
            )
        if side_sector.get("chosen") == opposite_label:
            # A clear, positive re-selection of the other side -- abort
            # rather than silently reversing direction mid-recovery.
            return self._abort_recovery(
                now_mono, sequence, side_sector, f"side sector selection flipped to {opposite_label.lower()}"
            )
        self._recovery_phase = "ROTATE_PULSE"
        self._recovery_phase_started_monotonic = None
        return result

    def _recovery_post_rotate_tick(
        self,
        sam: dict[str, Any],
        now_mono: float,
        sequence: int | None,
        frame_index: int | None,
    ) -> dict[str, Any]:
        assert self._recovery_started_monotonic is not None
        planner = sam.get("planner")
        planner_dict = planner if isinstance(planner, dict) else {}
        side_sector = planner_dict.get("side_sector")
        if now_mono - self._recovery_started_monotonic > self.settings.rotate_escape_max_total_sec:
            return self._abort_recovery(
                now_mono, sequence, side_sector, "rotate escape exceeded max total recovery time"
            )
        # H: reuse the controller's own path-validity gate wholesale (frame
        # identity is checked separately below) instead of re-deriving a
        # subset of near_field_safe/trajectory_valid/switch_stop_required/
        # heading/plan-age/confidence checks that already live there.
        strict_path_error = self._validate_post_rotate_path(sam)
        if strict_path_error is not None:
            return self._abort_recovery(
                now_mono,
                sequence,
                side_sector,
                f"post-rotate replan lost the safe path before confirmation: {strict_path_error}",
            )
        observation = _classify_observation(frame_index, self._recovery_last_frame_index)
        if observation == "REGRESSED":
            return self._abort_recovery(
                now_mono,
                sequence,
                side_sector,
                f"frame_index regressed during post-rotate replan ({frame_index} < {self._recovery_last_frame_index})",
            )
        if observation == "NEW":
            self._recovery_last_frame_index = frame_index
            self._recovery_safe_frame_count += 1
        command = ControlCommand(0.0, 0.0, mode="POST_ROTATE_REPLAN")
        result = self._dispatch_recovery_command(
            "POST_ROTATE_REPLAN",
            "confirming safe path after rotate escape "
            f"({self._recovery_safe_frame_count}/{self.settings.rotate_escape_safe_frame_confirm_count})",
            command,
            now_mono,
            sequence,
            side_sector,
            "POST_ROTATE_REPLAN",
        )
        if result["state"] == "ERROR_STOP":
            return result
        if self._recovery_safe_frame_count >= self.settings.rotate_escape_safe_frame_confirm_count:
            if self._recovery_direction is not None:
                self._recovery_last_direction = self._recovery_direction
            self._recovery_cooldown_until = now_mono + self.settings.rotate_escape_cooldown_sec
            self._recovery_phase = None
            self._recovery_direction = None
            self._recovery_safe_frame_count = 0
            self._recovery_direction_confirm_count = 0
            self._recovery_pulse_motion_observed = False
            self._recovery_pulse_max_abs_rpm = 0.0
            self._recovery_pulse_max_heading_delta_deg = 0.0
            self._recovery_pulse_start_heading_deg = None
            self._recovery_pulse_last_telemetry_key = None
            # This mechanism's own 3-distinct-frame confirmation supersedes
            # the legacy path_stop_latched gate for this recovery -- clear it
            # too, otherwise the next tick would re-run that gate from
            # scratch (up to path_recovery_confirm_frames more stopped
            # frames) on top of the confirmation just completed here.
            self._path_stop_latched = False
            self._path_recovery_count = 0
            self._last_path_recovery_frame_index = None
            self._recovery_last_frame_index = None
        return result

    def _abort_recovery(
        self,
        now_mono: float,
        sequence: int | None,
        side_sector: dict[str, Any] | None,
        reason: str,
    ) -> dict[str, Any]:
        # Capture status (pulse count, elapsed, etc.) before resetting state,
        # so the ABORTED record reflects what actually happened.
        recovery_meta = self._recovery_status(side_sector, phase="ABORTED")
        if self._recovery_direction is not None:
            self._recovery_last_direction = self._recovery_direction
        self._recovery_cooldown_until = now_mono + self.settings.rotate_escape_cooldown_sec
        self._recovery_phase = None
        self._recovery_direction = None
        self._recovery_phase_started_monotonic = None
        self._recovery_pulse_count = 0
        self._recovery_pulse_motion_observed = False
        self._recovery_pulse_max_abs_rpm = 0.0
        self._recovery_pulse_max_heading_delta_deg = 0.0
        self._recovery_pulse_start_heading_deg = None
        self._recovery_pulse_last_telemetry_key = None
        self._recovery_stationary_confirm_count = 0
        self._recovery_last_stationary_sample_key = None
        self._recovery_direction_confirm_count = 0
        self._recovery_safe_frame_count = 0
        self._recovery_last_frame_index = None
        command = ControlCommand(0.0, 0.0, mode="SAFETY_STOP")
        transmitted = self._send_stop("SAFETY_STOP")
        self._last_command = command
        return self._publish(
            "SAFETY_STOP",
            reason,
            transmitted,
            command,
            target_sequence=sequence,
            recovery=recovery_meta,
        )

    def _reset_local_history(self) -> None:
        self._consecutive_path_invalid = 0
        self._last_valid_raw_command = None
        self._last_valid_path_time = -math.inf
        self._last_valid_path_reason = ""
        self._path_stop_latched = False
        self._path_recovery_count = 0
        self._last_path_recovery_frame_index = None
        self._rotating_to_goal = False
        self._rotate_direction = None
        self._searching_for_path = False
        self._search_direction = None
        self._search_started_monotonic = -math.inf
        self._recovery_phase = None
        self._recovery_direction = None
        self._recovery_stop_confirm_ticks_done = 0
        self._recovery_stop_confirm_started_monotonic = None
        self._recovery_stationary_confirm_count = 0
        self._recovery_last_stationary_sample_key = None
        self._recovery_direction_confirm_count = 0
        self._recovery_started_monotonic = None
        self._recovery_phase_started_monotonic = None
        self._recovery_pulse_count = 0
        self._recovery_pulse_motion_observed = False
        self._recovery_pulse_max_abs_rpm = 0.0
        self._recovery_pulse_max_heading_delta_deg = 0.0
        self._recovery_pulse_start_heading_deg = None
        self._recovery_pulse_last_telemetry_key = None
        self._recovery_last_frame_index = None
        self._recovery_safe_frame_count = 0
        self._recovery_cooldown_until = -math.inf
        self._recovery_last_direction = None
        self._recovery_rotation_reason = ""
        self._reset_stop_turn_go(clear_cooldown=True)
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
        sdk_command = mission1_command_to_sdk_command(command)
        try:
            self.sdk.send_control(sdk_command)
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
            # The value actually sent to the rover over /control, once
            # command_transmitted is true. "angular" above stays in Mission1's
            # internal positive-right convention; this is post sign-flip.
            "sdk_angular": mission1_to_sdk_angular(command.angular),
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


def _classify_observation(value: int | None, last_value: int | None) -> str:
    """Classify a candidate observation identity (e.g. frame_index) against
    the last confirmed one, for a safety confirmation counter.

    Returns "MISSING" (``value`` is None -- must never count as fresh
    evidence for a confirmation gate), "NEW", "REPEAT", or "REGRESSED"
    (``value`` dropped below ``last_value`` -- treated as invalid
    instrumentation rather than progress, not merely a repeat).
    """

    if value is None:
        return "MISSING"
    if last_value is None:
        return "NEW"
    if value < last_value:
        return "REGRESSED"
    if value == last_value:
        return "REPEAT"
    return "NEW"
