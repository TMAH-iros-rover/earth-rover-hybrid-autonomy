from __future__ import annotations

import json
import math
import queue
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from training.sam_tp_sdk_shadow import ShadowStep


@dataclass(frozen=True)
class EventCaptureConfig:
    pre_event_frames: int = 8
    post_event_frames: int = 8
    baseline_interval_frames: int = 40
    candidate_jump_deg: float = 20.0
    fused_heading_jump_deg: float = 25.0
    target_bearing_jump_deg: float = 45.0
    score_mean_jump: float = 0.20
    queue_size: int = 32

    def validate(self) -> None:
        integer_values = {
            "pre_event_frames": self.pre_event_frames,
            "post_event_frames": self.post_event_frames,
            "baseline_interval_frames": self.baseline_interval_frames,
            "queue_size": self.queue_size,
        }
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in integer_values.values()
        ):
            raise ValueError("event capture frame counts and queue_size must be positive integers")
        thresholds = (
            self.candidate_jump_deg,
            self.fused_heading_jump_deg,
            self.target_bearing_jump_deg,
            self.score_mean_jump,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in thresholds):
            raise ValueError("event capture thresholds must be finite and positive")


@dataclass(frozen=True)
class _CapturedFrame:
    frame_index: int
    source_bgr: np.ndarray
    raw_logits: np.ndarray
    score_map: np.ndarray
    record: dict[str, object]


@dataclass(frozen=True)
class _Event:
    frame_index: int
    reasons: tuple[str, ...]
    window_start: int
    window_end: int


@dataclass(frozen=True)
class ReplayCaptureFrame:
    frame_index: int
    source_bgr: np.ndarray
    raw_logits: np.ndarray
    score_map: np.ndarray
    record: dict[str, object]


class SamTpEventRecorder:
    """Asynchronously persist replay inputs around planner/navigation hazards."""

    def __init__(self, output_dir: str | Path, config: EventCaptureConfig | None = None) -> None:
        self.config = config or EventCaptureConfig()
        self.config.validate()
        self.output_dir = Path(output_dir).expanduser().resolve() / "replay_capture"
        self.frames_dir = self.output_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=False)
        self._buffer: deque[_CapturedFrame] = deque(maxlen=self.config.pre_event_frames + 1)
        self._queue: queue.Queue[_CapturedFrame | _Event | None] = queue.Queue(
            maxsize=self.config.queue_size
        )
        self._queued_indices: set[int] = set()
        self._previous_record: dict[str, object] | None = None
        self._active_until = -1
        self._write_errors: list[str] = []
        self._dropped_items = 0
        self._saved_frames = 0
        self._event_count = 0
        self._closed = False
        self._worker = threading.Thread(target=self._write_loop, daemon=True)
        self._worker.start()

    def observe(self, step: ShadowStep) -> tuple[str, ...]:
        if self._closed:
            raise RuntimeError("event recorder is closed")
        reasons = detect_event_reasons(
            self._previous_record,
            step.record,
            candidate_jump_deg=self.config.candidate_jump_deg,
            fused_heading_jump_deg=self.config.fused_heading_jump_deg,
            target_bearing_jump_deg=self.config.target_bearing_jump_deg,
            score_mean_jump=self.config.score_mean_jump,
        )
        step.record["capture_event_reasons"] = list(reasons)
        captured = _capture_copy(step)
        self._buffer.append(captured)
        frame_index = captured.frame_index
        if reasons:
            self._active_until = max(
                self._active_until,
                frame_index + self.config.post_event_frames,
            )
            for buffered in self._buffer:
                self._enqueue_frame(buffered)
            self._enqueue(
                _Event(
                    frame_index=frame_index,
                    reasons=reasons,
                    window_start=max(0, frame_index - self.config.pre_event_frames),
                    window_end=self._active_until,
                )
            )
        if (
            frame_index <= self._active_until
            or frame_index % self.config.baseline_interval_frames == 0
        ):
            self._enqueue_frame(captured)
        self._previous_record = step.record
        return reasons

    def close(self) -> dict[str, object]:
        if not self._closed:
            self._closed = True
            self._queue.put(None)
            self._worker.join()
            summary = self.summary()
            (self.output_dir / "capture_summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        return self.summary()

    def summary(self) -> dict[str, object]:
        return {
            "enabled": True,
            "format_version": 1,
            "saved_frames": self._saved_frames,
            "event_count": self._event_count,
            "dropped_items": self._dropped_items,
            "write_errors": list(self._write_errors),
            "output_dir": str(self.output_dir),
        }

    def _enqueue_frame(self, captured: _CapturedFrame) -> None:
        if captured.frame_index in self._queued_indices:
            return
        self._queued_indices.add(captured.frame_index)
        if not self._enqueue(captured):
            self._queued_indices.discard(captured.frame_index)

    def _enqueue(self, item: _CapturedFrame | _Event) -> bool:
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            self._dropped_items += 1
            return False
        return True

    def _write_loop(self) -> None:
        frames_manifest = self.output_dir / "frames.jsonl"
        events_manifest = self.output_dir / "events.jsonl"
        with frames_manifest.open("w", encoding="utf-8") as frames_handle, events_manifest.open(
            "w", encoding="utf-8"
        ) as events_handle:
            while True:
                item = self._queue.get()
                try:
                    if item is None:
                        return
                    if isinstance(item, _CapturedFrame):
                        self._write_frame(item, frames_handle)
                    else:
                        events_handle.write(
                            json.dumps(
                                {
                                    "frame_index": item.frame_index,
                                    "reasons": list(item.reasons),
                                    "window_start": item.window_start,
                                    "window_end": item.window_end,
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
                        events_handle.flush()
                        self._event_count += 1
                except Exception as exc:  # Keep read-only shadow inference alive.
                    self._write_errors.append(f"{type(exc).__name__}: {exc}")
                finally:
                    self._queue.task_done()

    def _write_frame(self, item: _CapturedFrame, manifest_handle: Any) -> None:
        stem = f"frame_{item.frame_index:08d}"
        image_path = self.frames_dir / f"{stem}.png"
        arrays_path = self.frames_dir / f"{stem}.npz"
        record_path = self.frames_dir / f"{stem}.json"
        if not cv2.imwrite(str(image_path), item.source_bgr):
            raise OSError(f"could not write replay frame {image_path}")
        np.savez(
            arrays_path,
            raw_logits=item.raw_logits,
            traversability_score=item.score_map,
        )
        record_path.write_text(
            json.dumps(item.record, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest_handle.write(
            json.dumps(
                {
                    "frame_index": item.frame_index,
                    "image": str(image_path.relative_to(self.output_dir)),
                    "arrays": str(arrays_path.relative_to(self.output_dir)),
                    "record": str(record_path.relative_to(self.output_dir)),
                },
                sort_keys=True,
            )
            + "\n"
        )
        manifest_handle.flush()
        self._saved_frames += 1


def detect_event_reasons(
    previous: dict[str, object] | None,
    current: dict[str, object],
    *,
    candidate_jump_deg: float = 20.0,
    fused_heading_jump_deg: float = 25.0,
    target_bearing_jump_deg: float = 45.0,
    score_mean_jump: float = 0.20,
) -> tuple[str, ...]:
    reasons: list[str] = []
    state = current.get("shadow_state")
    previous_state = previous.get("shadow_state") if previous is not None else None
    if state != "CLEAR" and state != previous_state:
        reasons.append(f"shadow_state:{state}")
    planner = _mapping(current.get("planner"))
    if planner.get("switch_reason") == "immediate_reset":
        reasons.append("planner_immediate_reset")
    previous_near_safe = (
        bool(previous.get("near_field_safe", True)) if previous is not None else True
    )
    if not bool(current.get("near_field_safe", True)) and previous_near_safe:
        reasons.append("near_field_unsafe")
    if previous is not None:
        previous_planner = _mapping(previous.get("planner"))
        _append_angle_jump(
            reasons,
            "candidate_jump",
            previous_planner.get("selected_candidate_heading_deg"),
            planner.get("selected_candidate_heading_deg"),
            candidate_jump_deg,
        )
        previous_navigation = _mapping(previous.get("navigation"))
        navigation = _mapping(current.get("navigation"))
        if previous_navigation.get("target_sequence") != navigation.get("target_sequence"):
            reasons.append("target_sequence_change")
        _append_angle_jump(
            reasons,
            "target_bearing_jump",
            previous_navigation.get("target_bearing_deg"),
            navigation.get("target_bearing_deg"),
            target_bearing_jump_deg,
        )
        _append_angle_jump(
            reasons,
            "fused_heading_jump",
            _mapping(previous.get("localization")).get("fused_heading_deg"),
            _mapping(current.get("localization")).get("fused_heading_deg"),
            fused_heading_jump_deg,
        )
        previous_score = _finite(previous.get("score_mean"))
        current_score = _finite(current.get("score_mean"))
        if (
            previous_score is not None
            and current_score is not None
            and abs(current_score - previous_score) >= score_mean_jump
        ):
            reasons.append("score_mean_jump")
    return tuple(reasons)


def _capture_copy(step: ShadowStep) -> _CapturedFrame:
    return _CapturedFrame(
        frame_index=int(step.record["frame_index"]),
        source_bgr=np.asarray(step.source_bgr).copy(),
        raw_logits=np.asarray(step.raw_logits).copy(),
        score_map=np.asarray(step.score_map).copy(),
        record=json.loads(json.dumps(step.record)),
    )


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _append_angle_jump(
    reasons: list[str],
    label: str,
    previous: object,
    current: object,
    threshold: float,
) -> None:
    before = _finite(previous)
    after = _finite(current)
    if before is None or after is None:
        return
    delta = abs((after - before + 180.0) % 360.0 - 180.0)
    if delta >= threshold:
        reasons.append(label)


def _finite(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def load_replay_capture(capture_dir: str | Path) -> list[ReplayCaptureFrame]:
    """Load and validate a replay bundle without invoking either model backend."""

    root = Path(capture_dir).expanduser().resolve()
    manifest = root / "frames.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(f"replay frame manifest does not exist: {manifest}")
    frames: list[ReplayCaptureFrame] = []
    seen: set[int] = set()
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        item = json.loads(line)
        frame_index = int(item["frame_index"])
        if frame_index in seen:
            raise ValueError(f"duplicate replay frame_index {frame_index} at line {line_number}")
        seen.add(frame_index)
        image_path = _bundle_path(root, item["image"])
        arrays_path = _bundle_path(root, item["arrays"])
        record_path = _bundle_path(root, item["record"])
        source_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if source_bgr is None:
            raise ValueError(f"could not decode replay image: {image_path}")
        with np.load(arrays_path, allow_pickle=False) as arrays:
            raw_logits = np.asarray(arrays["raw_logits"]).copy()
            score_map = np.asarray(arrays["traversability_score"]).copy()
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if int(record.get("frame_index", -1)) != frame_index:
            raise ValueError(f"replay record frame_index mismatch for {frame_index}")
        expected_shape = source_bgr.shape[:2]
        if raw_logits.shape != expected_shape or score_map.shape != expected_shape:
            raise ValueError(f"replay array shape mismatch for frame {frame_index}")
        if not np.isfinite(raw_logits).all() or not np.isfinite(score_map).all():
            raise ValueError(f"replay arrays contain NaN or Inf for frame {frame_index}")
        frames.append(
            ReplayCaptureFrame(
                frame_index=frame_index,
                source_bgr=source_bgr,
                raw_logits=raw_logits,
                score_map=score_map,
                record=record,
            )
        )
    return frames


def _bundle_path(root: Path, relative: object) -> Path:
    path = (root / str(relative)).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"replay manifest path escapes bundle: {relative}")
    if not path.is_file():
        raise FileNotFoundError(f"replay artifact does not exist: {path}")
    return path
