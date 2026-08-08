from __future__ import annotations

import bisect
import math
from dataclasses import dataclass

import cv2
import numpy as np

from earth_rover.traversability_replay import ReplayStepResult
from research.training.traversability_video_review_v2 import (
    DISPLAY_COLORS_RGB,
    ReviewFrame,
    ReviewSegment,
)


@dataclass(frozen=True)
class DelayedReplayPair:
    source_timestamp: float
    observed_frame: ReviewFrame


def delayed_replay_pairs(
    segment: ReviewSegment,
    latency_sec: float,
    requested_duration_sec: float,
) -> tuple[DelayedReplayPair, ...]:
    """Pair each control timestamp with the newest frame available after delay."""

    if latency_sec < 0.0 or requested_duration_sec <= 0.0:
        raise ValueError("latency must be non-negative and duration must be positive")
    if not segment.frames:
        return ()
    timestamps = [frame.timestamp for frame in segment.frames]
    output_start = timestamps[0] + latency_sec
    output_end = output_start + requested_duration_sec
    pairs: list[DelayedReplayPair] = []
    for source_frame in segment.frames:
        if source_frame.timestamp < output_start - 1e-9:
            continue
        if source_frame.timestamp >= output_end - 1e-9:
            break
        target = source_frame.timestamp - latency_sec
        index = bisect.bisect_right(timestamps, target) - 1
        if index < 0:
            continue
        pairs.append(DelayedReplayPair(source_frame.timestamp, segment.frames[index]))
    return tuple(pairs)


def compose_planner_review_frame(
    frame_rgb: np.ndarray,
    source_mask: np.ndarray,
    confidence: np.ndarray,
    result: ReplayStepResult,
    dataset_name: str,
    ride_id: str,
    frame_id: int,
    checkpoint_version: str,
    panel_width: int = 480,
    low_confidence_threshold: float | None = None,
) -> np.ndarray:
    """Render an aspect-preserving 3-panel RGB/mask/planner review frame."""

    if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
        raise ValueError("frame_rgb must be HxWx3")
    if source_mask.shape != frame_rgb.shape[:2] or confidence.shape != source_mask.shape:
        raise ValueError("mask and confidence must match frame dimensions")
    panel_height = max(1, round(panel_width * frame_rgb.shape[0] / frame_rgb.shape[1]))
    display_mask = source_mask.copy()
    if low_confidence_threshold is not None:
        display_mask[confidence < low_confidence_threshold] = 0
    color_mask = DISPLAY_COLORS_RGB[display_mask]
    overlay = np.clip(
        frame_rgb.astype(np.float32) * 0.58 + color_mask.astype(np.float32) * 0.42,
        0,
        255,
    ).astype(np.uint8)
    panels = [
        _fit_rgb(frame_rgb, panel_width, panel_height),
        _fit_rgb(overlay, panel_width, panel_height),
        _fit_rgb(color_mask, panel_width, panel_height, nearest=True),
    ]
    for panel in panels:
        _draw_sectors(panel, result)
    body = np.concatenate(panels, axis=1)
    header_height = 112
    canvas = np.zeros((header_height + panel_height, panel_width * 3, 3), dtype=np.uint8)
    canvas[header_height:] = body
    record = result.record
    lines = (
        (
            f"{dataset_name} ride={ride_id} frame={frame_id} "
            f"source={float(record['source_timestamp']):.3f} observed={float(record['frame_timestamp']):.3f}"
        ),
        (
            f"checkpoint={checkpoint_version} latency={float(record['simulated_latency_sec']):.2f}s "
            f"inference={float(record['inference_latency_ms']):.1f}ms confidence={float(record['mean_confidence']):.2f}"
        ),
        (
            f"goal={record['goal_input_mode']} heading_error={float(record['heading_error_deg']):+.1f}deg "
            f"selected={record['selected_direction']} expected=({float(record['expected_linear']):+.3f},"
            f"{float(record['expected_angular']):+.3f}) transmitted=false"
        ),
        (
            f"safety={record['safety_state']} reason={record['safety_reason'] or '-'} "
            f"planner={record['planner_reason']}"
        ),
    )
    for index, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (10, 22 + index * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
    return cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)


def _draw_sectors(panel: np.ndarray, result: ReplayStepResult) -> None:
    height, width = panel.shape[:2]
    debug = result.traversability.sector_debug
    boundaries = debug.get("sector_boundaries", (0.0, 0.34, 0.66, 1.0))
    roi_top = float(debug.get("roi_top", 0.35))
    near_top = float(debug.get("near_top", 0.72))
    for fraction in boundaries[1:3]:
        x = round(width * fraction)
        cv2.line(panel, (x, round(height * roi_top)), (x, height - 1), (255, 255, 255), 1)
    cv2.line(
        panel,
        (0, round(height * roi_top)),
        (width - 1, round(height * roi_top)),
        (180, 180, 180),
        1,
    )
    cv2.line(
        panel,
        (0, round(height * near_top)),
        (width - 1, round(height * near_top)),
        (255, 120, 255),
        1,
    )
    centers = {"LEFT": 0.17, "CENTER": 0.50, "RIGHT": 0.83}
    scores = {
        "LEFT": result.traversability.left_score,
        "CENTER": result.traversability.center_score,
        "RIGHT": result.traversability.right_score,
    }
    for name, fraction in centers.items():
        selected = name == result.plan.selected_direction
        cv2.putText(
            panel,
            f"{name[0]} {scores[name]:.2f}",
            (max(2, round(width * fraction) - 32), min(height - 5, round(height * 0.42))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255) if not selected else (40, 255, 255),
            1,
            cv2.LINE_AA,
        )
    selected_x = centers.get(result.plan.selected_direction, 0.50)
    cv2.arrowedLine(
        panel,
        (width // 2, height - 6),
        (round(width * selected_x), round(height * 0.55)),
        (40, 255, 255),
        2,
        tipLength=0.2,
    )
    goal_fraction = 0.5 - float(result.record["heading_error_deg"]) / 180.0
    goal_x = round(width * max(0.05, min(0.95, goal_fraction)))
    cv2.arrowedLine(
        panel,
        (width // 2, height - 6),
        (goal_x, round(height * 0.65)),
        (80, 180, 255),
        2,
        tipLength=0.2,
    )


def _fit_rgb(image: np.ndarray, width: int, height: int, nearest: bool = False) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized_width = max(1, round(image.shape[1] * scale))
    resized_height = max(1, round(image.shape[0] * scale))
    interpolation = cv2.INTER_NEAREST if nearest else (
        cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    )
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x = (width - resized_width) // 2
    y = (height - resized_height) // 2
    canvas[y : y + resized_height, x : x + resized_width] = resized
    return canvas
