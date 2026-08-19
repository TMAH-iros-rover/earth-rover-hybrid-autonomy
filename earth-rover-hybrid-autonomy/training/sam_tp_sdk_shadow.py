from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import cv2
import numpy as np

from earth_rover.core.types import FrameData, RoverData
from earth_rover.navigation.localization import GpsHeadingEkf
from earth_rover.perception.camera_calibration import CameraCalibration
from earth_rover.navigation.checkpoint_route import (
    CheckpointRoutePlanner,
    GlobalRouteState,
)
from earth_rover.planning.trajectory_sampler import (
    DEFAULT_CURVATURES,
    ConstantCurvatureTrajectorySampler,
)
from earth_rover.planning.motion_primitive_planner import (
    MotionPrimitivePlan,
    MotionPrimitivePlanner,
)
from training.sam_tp_phase1_review import (
    SamTpPhase1FrameProcessor,
    draw_image_path_rgb,
)
from training.sam_tp_reproduction import SamTpPrediction, score_to_heatmap


class ReadOnlySdkSource(Protocol):
    def get_front_frame(self) -> FrameData: ...

    def get_data(self) -> RoverData: ...


class Predictor(Protocol):
    def predict(self, image_rgb: np.ndarray) -> SamTpPrediction: ...


_PROVISIONAL_PHASE1_TRAJECTORIES = ConstantCurvatureTrajectorySampler(
    DEFAULT_CURVATURES,
    horizon_m=2.0,
    sample_interval_m=0.1,
    rover_width_m=0.4,
    safety_margin_m=0.1,
).sample()


@dataclass(frozen=True)
class ShadowStep:
    dashboard_bgr: np.ndarray
    overlay_bgr: np.ndarray
    source_bgr: np.ndarray
    raw_logits: np.ndarray
    score_map: np.ndarray
    record: dict[str, object]


def run_shadow_step(
    sdk: ReadOnlySdkSource,
    predictor: Predictor,
    frame_index: int,
    telemetry: RoverData | None,
    fetch_telemetry: bool,
    started_monotonic: float,
    checkpoint_sha256: str,
    maximum_frame_age_sec: float,
    maximum_telemetry_age_sec: float,
    clock: Callable[[], float] = time.time,
    monotonic: Callable[[], float] = time.monotonic,
    panel_width: int = 480,
    phase1_processor: SamTpPhase1FrameProcessor | None = None,
    route_planner: CheckpointRoutePlanner | None = None,
    local_planner: MotionPrimitivePlanner | None = None,
    heading_offset_deg: float = 0.0,
    localizer: GpsHeadingEkf | None = None,
    checkpoint_metadata: dict[str, object] | None = None,
    calibration: CameraCalibration | None = None,
) -> tuple[ShadowStep, RoverData | None]:
    """Fetch one live frame, infer once, and return a read-only dashboard step.

    The source interface intentionally exposes only GET-equivalent methods.
    This function has no control, mission, checkpoint, or SDK write path.
    """

    request_started = clock()
    request_started_monotonic = monotonic()
    frame = sdk.get_front_frame()
    frame_received = clock()
    frame_received_monotonic = monotonic()
    telemetry_error = None
    if fetch_telemetry:
        try:
            telemetry = sdk.get_data()
        except Exception as exc:
            telemetry_error = f"{type(exc).__name__}: {exc}"
        else:
            # Gated on fetch_telemetry -- fusing the same stale telemetry
            # sample on every outer-loop tick would make the filter
            # overconfident in stale data and more resistant to genuinely
            # new fixes, the opposite of what it's for.
            if localizer is not None:
                raw_payload = telemetry.raw if isinstance(telemetry.raw, dict) else {}
                localizer.observe_gyro(raw_payload.get("gyros"))
                localizer.observe_gps_heading(
                    telemetry.latitude,
                    telemetry.longitude,
                    corrected_heading_deg(telemetry.orientation, heading_offset_deg),
                    telemetry.gps_signal,
                )
    if frame.image.ndim != 3 or frame.image.shape[2] != 3:
        raise ValueError("SDK front frame must be an HxWx3 BGR image")
    if frame.image.dtype != np.uint8:
        raise ValueError("SDK front frame must use uint8 pixels")

    image_bgr = np.asarray(frame.image)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    if localizer is not None and localizer.is_locked:
        fused_lat, fused_lon, fused_heading_deg = localizer.current_estimate()
        if not bool(getattr(localizer, "heading_valid", True)):
            fused_heading_deg = None
        elif route_planner is not None:
            consume_reanchor = getattr(localizer, "consume_heading_reanchor", None)
            if callable(consume_reanchor) and consume_reanchor():
                route_planner.reanchor_heading()
    elif telemetry is not None:
        fused_lat = telemetry.latitude
        fused_lon = telemetry.longitude
        fused_heading_deg = corrected_heading_deg(telemetry.orientation, heading_offset_deg)
    else:
        fused_lat = fused_lon = fused_heading_deg = None
    navigation = (
        route_planner.update(fused_lat, fused_lon, fused_heading_deg)
        if route_planner is not None and telemetry is not None
        else None
    )
    processor = phase1_processor or SamTpPhase1FrameProcessor(
        predictor,
        _PROVISIONAL_PHASE1_TRAJECTORIES,
        checkpoint_sha256[:12],
    )
    phase1 = processor.process(
        image_rgb,
        float(frame.timestamp),
        target_heading_error_rad=(
            navigation.heading_error_rad if navigation is not None else None
        ),
    )
    prediction = phase1.prediction
    traversability = phase1.traversability
    primitive_plan: MotionPrimitivePlan | None = None
    image_path = phase1.image_path
    planner_latency_sec = 0.0
    if local_planner is not None and local_planner.config.mode != "connected_path":
        planner_started_monotonic = monotonic()
        primitive_plan = local_planner.plan(
            traversability.score_map,
            traversability.valid_mask,
            target_heading_error_rad=(
                navigation.heading_error_rad if navigation is not None else None
            ),
            checkpoint_sequence=(
                navigation.target_sequence if navigation is not None else None
            ),
            timestamp=float(frame.timestamp),
            navigation=navigation_record(navigation),
            calibration=calibration,
        )
        image_path = primitive_plan.image_path
        planner_latency_sec = monotonic() - planner_started_monotonic
    finished = clock()
    finished_monotonic = monotonic()
    acquisition_latency_sec = frame_received_monotonic - request_started_monotonic
    end_to_end_latency_sec = finished_monotonic - request_started_monotonic
    if acquisition_latency_sec < 0.0 or not math.isfinite(acquisition_latency_sec):
        raise ValueError("acquisition latency is invalid")
    if end_to_end_latency_sec < 0.0 or not math.isfinite(end_to_end_latency_sec):
        raise ValueError("end-to-end latency is invalid")
    local_frame_age_sec = finished - float(frame.timestamp)
    if not math.isfinite(local_frame_age_sec):
        raise ValueError("local frame age is not finite")
    sdk_frame_age_sec = (
        finished - float(frame.sdk_timestamp)
        if frame.sdk_timestamp is not None
        else None
    )
    if sdk_frame_age_sec is not None and not math.isfinite(sdk_frame_age_sec):
        raise ValueError("SDK frame age is not finite")
    sdk_clock_offset_hours = _timezone_offset_hours(
        sdk_frame_age_sec,
        maximum_frame_age_sec,
    )
    sdk_frame_timestamp_usable = (
        sdk_frame_age_sec is not None and sdk_clock_offset_hours is None
    )
    frame_stale = (
        local_frame_age_sec < 0.0
        or local_frame_age_sec > maximum_frame_age_sec
        or (
            sdk_frame_age_sec is not None
            and sdk_frame_timestamp_usable
            and (
                sdk_frame_age_sec < -maximum_frame_age_sec
                or sdk_frame_age_sec > maximum_frame_age_sec
            )
        )
    )
    telemetry_age_sec = (
        finished - float(telemetry.timestamp) if telemetry is not None else None
    )
    if telemetry_age_sec is not None and not math.isfinite(telemetry_age_sec):
        raise ValueError("telemetry age is not finite")
    telemetry_stale = (
        telemetry_age_sec is None
        or telemetry_age_sec < 0.0
        or telemetry_age_sec > maximum_telemetry_age_sec
    )
    if frame_stale:
        shadow_state = "STALE_FRAME"
    elif telemetry_error is not None and telemetry is None:
        shadow_state = "WAITING_TELEMETRY"
    elif telemetry_stale:
        shadow_state = "STALE_TELEMETRY"
    else:
        shadow_state = "CLEAR"
    elapsed = finished_monotonic - started_monotonic
    effective_fps = (frame_index + 1) / elapsed if elapsed > 0.0 else 0.0
    record = {
        "frame_index": frame_index,
        "request_started_timestamp": request_started,
        "frame_received_timestamp": frame_received,
        "local_frame_timestamp": frame.timestamp,
        "sdk_frame_timestamp": frame.sdk_timestamp,
        "local_frame_age_sec": local_frame_age_sec,
        "sdk_frame_age_sec": sdk_frame_age_sec,
        "sdk_frame_timestamp_usable": sdk_frame_timestamp_usable,
        "sdk_clock_offset_hours": sdk_clock_offset_hours,
        "telemetry_age_sec": telemetry_age_sec,
        "telemetry_error": telemetry_error,
        "acquisition_latency_ms": acquisition_latency_sec * 1000.0,
        "inference_latency_ms": prediction.inference_time_ms,
        "planner_latency_ms": planner_latency_sec * 1000.0,
        "end_to_end_latency_ms": end_to_end_latency_sec * 1000.0,
        "effective_fps": effective_fps,
        "score_min": float(prediction.traversability_score.min()),
        "score_max": float(prediction.traversability_score.max()),
        "score_mean": float(prediction.traversability_score.mean()),
        "score_std": float(prediction.traversability_score.std()),
        "adapter_confidence": traversability.confidence,
        "candidate_trajectory_count": len(phase1.trajectories),
        "trajectory_geometry_only": not (
            primitive_plan is not None and primitive_plan.camera_projection_applied
        ),
        "camera_projection_applied": (
            primitive_plan.camera_projection_applied if primitive_plan is not None else False
        ),
        "geometry_mode": (
            primitive_plan.geometry_mode if primitive_plan is not None else "image_heuristic"
        ),
        "calibration_id": (
            primitive_plan.calibration_id if primitive_plan is not None else None
        ),
        "calibration_sha256_prefix": (
            primitive_plan.calibration_sha256_prefix if primitive_plan is not None else None
        ),
        "image_path_valid": image_path.valid,
        "image_path_reason": image_path.reason,
        "image_path_mean_score": image_path.mean_score,
        "local_path_length_px": image_path.path_length_px,
        "local_path_goal_alignment_weight": image_path.goal_alignment_weight,
        "local_path_smoothing_method": image_path.smoothing_method,
        "local_path_smoothing_applied": image_path.smoothing_applied,
        "local_path_smoothing_iterations": image_path.smoothing_iterations,
        "local_path_selected_heading_deg": _degrees_or_none(
            image_path.selected_heading_rad
        ),
        "local_path_heading_residual_deg": _degrees_or_none(
            image_path.heading_residual_rad
        ),
        "global_target_heading_error_deg": _degrees_or_none(
            image_path.target_heading_error_rad
        ),
        "planner": _planner_record(primitive_plan),
        "near_field_safe": (
            primitive_plan.near_field_safe if primitive_plan is not None else image_path.valid
        ),
        "near_field_score": (
            primitive_plan.near_field_score if primitive_plan is not None else image_path.minimum_score
        ),
        "trajectory_valid": (
            primitive_plan.trajectory_valid if primitive_plan is not None else image_path.valid
        ),
        "trajectory_quality": (
            primitive_plan.trajectory_quality if primitive_plan is not None else image_path.mean_score
        ),
        "planner_confidence": (
            primitive_plan.planner_confidence if primitive_plan is not None else image_path.mean_score
        ),
        "plan_age_sec": (
            primitive_plan.plan_age_sec if primitive_plan is not None else 0.0
        ),
        "using_held_plan": (
            primitive_plan.using_held_plan if primitive_plan is not None else False
        ),
        "image_path_metric_calibrated": (
            primitive_plan.image_path_metric_calibrated if primitive_plan is not None else False
        ),
        "image_path_experimental_control_input": True,
        "prediction_valid": not frame_stale,
        "telemetry_valid": telemetry_error is None and not telemetry_stale,
        "shadow_state": shadow_state,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint": checkpoint_metadata,
        "telemetry": telemetry_record(telemetry, fresh=fetch_telemetry),
        "navigation_heading_offset_deg": heading_offset_deg,
        "navigation": navigation_record(navigation),
        "localization": localization_record(localizer),
        "sdk_allowed_read_endpoints": [
            "/v2/front",
            "/front",
            "/data",
            "/mission-route",
        ],
        "command_transmitted": False,
    }
    dashboard = compose_shadow_dashboard(
        image_bgr,
        prediction.traversability_score,
        record,
        telemetry,
        panel_width,
        image_path,
    )
    overlay = compose_traversability_overlay(
        image_bgr,
        prediction.traversability_score,
        image_path,
    )
    return ShadowStep(
        dashboard_bgr=dashboard,
        overlay_bgr=overlay,
        source_bgr=image_bgr,
        raw_logits=prediction.raw_logits,
        score_map=prediction.traversability_score,
        record=record,
    ), telemetry


def _timezone_offset_hours(
    age_sec: float | None,
    tolerance_sec: float,
) -> int | None:
    """Recognize a whole-hour camera timestamp offset without hiding drift."""

    if age_sec is None or abs(age_sec) <= tolerance_sec:
        return None
    hours = round(age_sec / 3600.0)
    if hours == 0 or abs(hours) > 14:
        return None
    residual = age_sec - hours * 3600.0
    return hours if abs(residual) <= tolerance_sec else None


def compose_traversability_overlay(
    image_bgr: np.ndarray,
    score: np.ndarray,
    image_path=None,
) -> np.ndarray:
    """Blend SAM-TP evidence and the display-only path into the source frame."""

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("image_bgr must have shape HxWx3")
    if image_bgr.dtype != np.uint8:
        raise ValueError("image_bgr must use uint8 pixels")
    if score.shape != image_bgr.shape[:2]:
        raise ValueError("score shape must match the SDK frame")
    heatmap_bgr = cv2.cvtColor(score_to_heatmap(score), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(image_bgr, 0.58, heatmap_bgr, 0.42, 0.0)
    if image_path is not None:
        path_rgb = draw_image_path_rgb(
            cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
            image_path,
            0.018,
        )
        overlay = cv2.cvtColor(path_rgb, cv2.COLOR_RGB2BGR)
    return overlay


def telemetry_record(
    data: RoverData | None, *, fresh: bool = True
) -> dict[str, object] | None:
    if data is None:
        return None
    raw = data.raw if isinstance(data.raw, dict) else {}
    # accels/gyros/mags/vibration are only present in `raw` for logging (not
    # consumed anywhere -- units/axes undocumented, see localization.py).
    # `data` itself is reused unchanged across every outer-loop tick between
    # telemetry refetches, so without `fresh` these (comparatively large)
    # arrays would be re-serialized into every JSONL line redundantly.
    return {
        "local_timestamp": data.timestamp,
        "sdk_timestamp": data.sdk_timestamp,
        "latitude": data.latitude,
        "longitude": data.longitude,
        "orientation": data.orientation,
        "speed": data.speed,
        "rpms": data.rpms,
        "battery": data.battery,
        "signal_level": data.signal_level,
        "gps_signal": data.gps_signal,
        "accels": raw.get("accels") if fresh else None,
        "gyros": raw.get("gyros") if fresh else None,
        "mags": raw.get("mags") if fresh else None,
        "vibration": raw.get("vibration") if fresh else None,
    }


def _planner_record(plan: MotionPrimitivePlan | None) -> dict[str, object]:
    if plan is None:
        return {
            "mode": "connected_path",
            "near_field_safe": None,
            "trajectory_valid": None,
            "planner_confidence": None,
            "using_held_plan": False,
            "plan_age_sec": 0.0,
        }
    return plan.to_status(include_candidates=True)


def corrected_heading_deg(
    heading_deg: float | None,
    offset_deg: float,
) -> float | None:
    if heading_deg is None:
        return None
    heading = float(heading_deg)
    offset = float(offset_deg)
    if not math.isfinite(heading) or not math.isfinite(offset):
        return None
    return (heading + offset) % 360.0


def localization_record(localizer: GpsHeadingEkf | None) -> dict[str, object] | None:
    if localizer is None:
        return None
    lat, lon, heading_deg = localizer.current_estimate()
    record = {
        "locked": localizer.is_locked,
        "fused_latitude": lat,
        "fused_longitude": lon,
        "fused_heading_deg": heading_deg,
        "gyro_trusted": localizer.gyro_trusted,
    }
    status = getattr(localizer, "status", None)
    if callable(status):
        record.update(status())
    return record


def navigation_record(state: GlobalRouteState | None) -> dict[str, object] | None:
    if state is None:
        return None
    return {
        "route_polyline": [list(point) for point in state.route_polyline],
        "target_sequence": state.target_sequence,
        "distance_to_target_m": state.distance_to_target_m,
        "target_bearing_deg": state.target_bearing_deg,
        "target_bearing_convention": "compass_0_north_90_east_clockwise",
        "current_heading_deg": state.current_heading_deg,
        "current_heading_convention": "assumed_compass_0_north_90_east_clockwise",
        "heading_error_deg": _degrees_or_none(state.heading_error_rad),
        "heading_error_convention": "positive_clockwise_right",
        "gps_valid": state.gps_valid,
        "heading_valid": state.heading_valid,
        "reached": state.reached,
        "finished": state.finished,
        "reason": state.reason,
    }


def compose_shadow_dashboard(
    image_bgr: np.ndarray,
    score: np.ndarray,
    record: dict[str, object],
    telemetry: RoverData | None,
    panel_width: int,
    image_path=None,
) -> np.ndarray:
    if panel_width <= 0:
        raise ValueError("panel_width must be positive")
    if score.shape != image_bgr.shape[:2]:
        raise ValueError("score shape must match the SDK frame")
    height, width = image_bgr.shape[:2]
    panel_height = max(1, int(round(panel_width * height / width)))
    original = cv2.resize(
        image_bgr,
        (panel_width, panel_height),
        interpolation=cv2.INTER_AREA,
    )
    if image_path is not None:
        path_rgb = draw_image_path_rgb(
            cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
            image_path,
            0.018,
        )
        original = cv2.resize(
            cv2.cvtColor(path_rgb, cv2.COLOR_RGB2BGR),
            (panel_width, panel_height),
            interpolation=cv2.INTER_AREA,
        )
    heatmap_rgb = score_to_heatmap(score)
    heatmap_bgr = cv2.cvtColor(heatmap_rgb, cv2.COLOR_RGB2BGR)
    heatmap_bgr = cv2.resize(
        heatmap_bgr,
        (panel_width, panel_height),
        interpolation=cv2.INTER_LINEAR,
    )
    overlay = cv2.addWeighted(original, 0.55, heatmap_bgr, 0.45, 0.0)
    header_height = 64
    footer_height = 118
    canvas = np.full(
        (header_height + panel_height + footer_height, panel_width * 3, 3),
        24,
        dtype=np.uint8,
    )
    canvas[header_height : header_height + panel_height, :panel_width] = original
    canvas[
        header_height : header_height + panel_height,
        panel_width : panel_width * 2,
    ] = overlay
    canvas[
        header_height : header_height + panel_height,
        panel_width * 2 :,
    ] = heatmap_bgr
    titles = (
        "SDK FRONT + IMAGE PATH",
        "SAM-TP OVERLAY + PATH",
        "SCORE: BLUE LOW / RED HIGH",
    )
    for index, title in enumerate(titles):
        _text(canvas, title, index * panel_width + 12, 27, 0.55)
    state = str(record["shadow_state"])
    state_color = (70, 220, 70) if state == "CLEAR" else (40, 80, 240)
    cv2.putText(
        canvas,
        (
            f"READ-ONLY SHADOW | {state} | "
            f"{record.get('geometry_mode', 'image_heuristic')} | "
            f"cal={'OK' if record.get('image_path_metric_calibrated') else 'INVALID'} | "
            "command_transmitted=false"
        ),
        (12, 53),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        state_color,
        1,
        cv2.LINE_AA,
    )
    footer_y = header_height + panel_height + 28
    _text(
        canvas,
        (
            f"frame={record['frame_index']}  acquisition="
            f"{float(record['acquisition_latency_ms']):.1f}ms  inference="
            f"{float(record['inference_latency_ms']):.1f}ms  end-to-end="
            f"{float(record['end_to_end_latency_ms']):.1f}ms  effective="
            f"{float(record['effective_fps']):.2f} FPS"
        ),
        12,
        footer_y,
        0.52,
    )
    _text(
        canvas,
        (
            f"local_age={float(record['local_frame_age_sec']):.3f}s  "
            f"telemetry_age={record['telemetry_age_sec']}  "
            f"sdk_ts={record['sdk_frame_timestamp']}  "
            f"score min/mean/max={float(record['score_min']):.3f}/"
            f"{float(record['score_mean']):.3f}/{float(record['score_max']):.3f}  "
            f"path={_dashboard_path_reason(record)}"
        ),
        12,
        footer_y + 28,
        0.50,
    )
    if telemetry is None:
        telemetry_text = "telemetry: waiting"
    else:
        telemetry_text = (
            f"GPS=({telemetry.latitude}, {telemetry.longitude})  "
            f"heading={telemetry.orientation}  speed={telemetry.speed}  "
            f"battery={telemetry.battery}  signal={telemetry.signal_level}"
        )
    _text(canvas, telemetry_text, 12, footer_y + 56, 0.50)
    checkpoint_meta = record.get("checkpoint") or {}
    checkpoint_label = (
        f"checkpoint={checkpoint_meta.get('filename', '?')} "
        f"sha256={checkpoint_meta.get('sha256_prefix', str(record['checkpoint_sha256'])[:16])} "
        f"backend={checkpoint_meta.get('backend', '?')}"
        if checkpoint_meta
        else f"checkpoint={str(record['checkpoint_sha256'])[:16]}"
    )
    _text(
        canvas,
        f"{checkpoint_label}  q/esc: quit",
        12,
        footer_y + 84,
        0.48,
    )
    return canvas


def _dashboard_path_reason(record: dict[str, object]) -> str:
    planner = record.get("planner")
    if isinstance(planner, dict):
        candidates = planner.get("candidate_scores")
        if isinstance(candidates, list) and candidates:
            reject_reasons = {
                candidate.get("reject_reason")
                for candidate in candidates
                if isinstance(candidate, dict) and candidate.get("hard_rejected") is True
            }
            if len(reject_reasons) == 1:
                return str(next(iter(reject_reasons)))
    return str(record.get("image_path_reason", "UNKNOWN"))


def write_shadow_summary(
    output_dir: str | Path,
    records: list[dict[str, object]],
    failures: list[dict[str, object]],
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    checkpoint_metadata: dict[str, object] | None = None,
) -> Path:
    output = Path(output_dir)
    values = [float(item["end_to_end_latency_ms"]) for item in records]
    inference = [float(item["inference_latency_ms"]) for item in records]
    summary = {
        "success": bool(records),
        "processed_frame_count": len(records),
        "failed_frame_count": len(failures),
        "failures": failures,
        "stale_frame_count": sum(not bool(item["prediction_valid"]) for item in records),
        "stale_telemetry_count": sum(
            not bool(item["telemetry_valid"]) for item in records
        ),
        "end_to_end_latency_ms": _latency(values),
        "inference_latency_ms": _latency(inference),
        "effective_fps": records[-1]["effective_fps"] if records else 0.0,
        "checkpoint_path": str(Path(checkpoint_path).expanduser().resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint": checkpoint_metadata,
        "sdk_allowed_read_endpoints": [
            "/v2/front",
            "/front",
            "/data",
            "/mission-route",
        ],
        "sdk_write_endpoints": [],
        "command_transmitted": False,
        "live_motion_command_sent_by_process": False,
    }
    path = output / "shadow_summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _latency(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def _text(
    image: np.ndarray,
    value: str,
    x: int,
    y: int,
    scale: float,
) -> None:
    cv2.putText(
        image,
        value,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )


def _degrees_or_none(value: float | None) -> float | None:
    return math.degrees(value) if value is not None else None
