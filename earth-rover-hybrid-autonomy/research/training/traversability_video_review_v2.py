from __future__ import annotations

import bisect
import hashlib
import json
import math
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import cv2
import numpy as np

DISPLAY_COLORS_RGB = np.array(
    [
        [0, 0, 0],
        [38, 166, 91],
        [235, 190, 35],
        [220, 50, 47],
    ],
    dtype=np.uint8,
)
CLASS_NAMES = {1: "ON_ROAD", 2: "OFF_ROAD", 3: "OBSTACLE"}


@dataclass(frozen=True)
class ReviewFrame:
    dataset: str
    ride_id: str
    frame_id: int
    timestamp: float
    playlist_reference: str
    segment_reference: str
    timeline_section_id: int


@dataclass(frozen=True)
class ReviewSegment:
    dataset: str
    ride_id: str
    start_timestamp: float
    end_timestamp: float
    requested_duration_seconds: float
    frames: tuple[ReviewFrame, ...]


class FrameDecoder(Protocol):
    def decode(self, frame: ReviewFrame) -> np.ndarray: ...


class Predictor(Protocol):
    checkpoint_version: str

    def predict(self, frame_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]: ...


class VideoWriter(Protocol):
    def write(self, frame_bgr: np.ndarray) -> None: ...

    def close(self) -> dict[str, object]: ...


class RideSourceLike(Protocol):
    dataset: str
    dataset_root: Path
    ride_id: str
    timeline: Any
    frames: tuple[Any, ...]


def select_review_segments(
    rides: list[RideSourceLike],
    ride_count: int,
    duration_seconds: float,
    output_fps: float,
    edge_margin_seconds: float,
    maximum_frame_gap_seconds: float,
    seed: int,
) -> tuple[list[ReviewSegment], list[dict[str, object]]]:
    if ride_count <= 0:
        raise ValueError("ride_count must be positive")
    if duration_seconds <= 0 or output_fps <= 0:
        raise ValueError("duration and output FPS must be positive")
    if edge_margin_seconds < 0 or maximum_frame_gap_seconds <= 0:
        raise ValueError("edge margin must be non-negative and maximum gap must be positive")

    selected: list[ReviewSegment] = []
    skipped: list[dict[str, object]] = []
    for ride in _deterministic_ride_order(rides, seed):
        segment = _select_ride_segment(
            ride,
            duration_seconds,
            output_fps,
            edge_margin_seconds,
            maximum_frame_gap_seconds,
            seed,
        )
        if segment is None:
            skipped.append(
                {
                    "dataset": ride.dataset,
                    "ride_id": ride.ride_id,
                    "reason": "no_continuous_window",
                    "required_duration_seconds": duration_seconds,
                }
            )
            continue
        selected.append(segment)
        if len(selected) == ride_count:
            break
    if len(selected) < ride_count:
        skipped.append(
            {
                "dataset": rides[0].dataset if rides else None,
                "ride_id": None,
                "reason": "insufficient_eligible_rides",
                "requested_ride_count": ride_count,
                "selected_ride_count": len(selected),
            }
        )
    return selected, skipped


def _select_ride_segment(
    ride: RideSourceLike,
    duration_seconds: float,
    output_fps: float,
    edge_margin_seconds: float,
    maximum_frame_gap_seconds: float,
    seed: int,
) -> ReviewSegment | None:
    runs = _continuous_runs(ride, maximum_frame_gap_seconds)
    ride_start = ride.frames[0].timestamp + edge_margin_seconds
    ride_end = ride.frames[-1].timestamp - edge_margin_seconds
    starts: list[tuple[tuple[Any, ...], int]] = []
    for run in runs:
        timestamps = [frame.timestamp for frame in run]
        minimum_start = max(timestamps[0], ride_start)
        maximum_start = min(timestamps[-1] - duration_seconds, ride_end - duration_seconds)
        if maximum_start < minimum_start:
            continue
        first = bisect.bisect_left(timestamps, minimum_start)
        last = bisect.bisect_right(timestamps, maximum_start)
        starts.extend((run, index) for index in range(first, last))
    if not starts:
        return None
    digest = hashlib.sha256(f"{seed}:{ride.dataset}:{ride.ride_id}:window".encode()).hexdigest()
    run, start_index = starts[int(digest[:16], 16) % len(starts)]
    start_timestamp = run[start_index].timestamp
    frames = _sample_run(run, start_timestamp, duration_seconds, output_fps, ride)
    expected = int(math.floor(duration_seconds * output_fps))
    if len(frames) < max(1, expected - 1):
        return None
    return ReviewSegment(
        dataset=ride.dataset,
        ride_id=ride.ride_id,
        start_timestamp=start_timestamp,
        end_timestamp=start_timestamp + duration_seconds,
        requested_duration_seconds=duration_seconds,
        frames=tuple(frames),
    )


def _continuous_runs(
    ride: RideSourceLike,
    maximum_frame_gap_seconds: float,
) -> list[tuple[Any, ...]]:
    runs: list[list[Any]] = []
    current: list[FrontFrame] = []
    previous_section: int | None = None
    for frame in ride.frames:
        segment = ride.timeline.find_segment(frame.timestamp)
        if segment is None:
            continue
        if current:
            previous = current[-1]
            discontinuity = (
                frame.timestamp <= previous.timestamp
                or frame.timestamp - previous.timestamp > maximum_frame_gap_seconds
                or frame.frame_id <= previous.frame_id
                or segment.section_id != previous_section
            )
            if discontinuity:
                runs.append(current)
                current = []
        current.append(frame)
        previous_section = segment.section_id
    if current:
        runs.append(current)
    return [tuple(run) for run in runs]


def _sample_run(
    run: tuple[Any, ...],
    start_timestamp: float,
    duration_seconds: float,
    output_fps: float,
    ride: RideSourceLike,
) -> list[ReviewFrame]:
    timestamps = [frame.timestamp for frame in run]
    count = int(math.floor(duration_seconds * output_fps))
    tolerance = max(0.075, 0.75 / output_fps)
    sampled: list[ReviewFrame] = []
    previous_frame_id: int | None = None
    for index in range(count):
        target = start_timestamp + index / output_fps
        position = bisect.bisect_left(timestamps, target)
        candidates = [
            candidate
            for candidate in (position - 1, position)
            if 0 <= candidate < len(run)
        ]
        if not candidates:
            continue
        nearest = min(candidates, key=lambda candidate: (abs(timestamps[candidate] - target), candidate))
        frame = run[nearest]
        if abs(frame.timestamp - target) > tolerance or frame.frame_id == previous_frame_id:
            continue
        segment = ride.timeline.find_segment(frame.timestamp)
        if segment is None:
            continue
        sampled.append(
            ReviewFrame(
                dataset=ride.dataset,
                ride_id=ride.ride_id,
                frame_id=frame.frame_id,
                timestamp=frame.timestamp,
                playlist_reference=ride.timeline.playlist_reference,
                segment_reference=segment.reference,
                timeline_section_id=segment.section_id,
            )
        )
        previous_frame_id = frame.frame_id
    return sampled


def _deterministic_ride_order(
    rides: list[RideSourceLike],
    seed: int,
) -> list[RideSourceLike]:
    return sorted(
        rides,
        key=lambda ride: hashlib.sha256(
            f"{seed}:{ride.dataset}:{ride.ride_id}".encode()
        ).hexdigest(),
    )


def compose_three_panel_frame(
    frame_rgb: np.ndarray,
    source_prediction: np.ndarray,
    confidence: np.ndarray,
    metadata: dict[str, object],
    panel_width: int,
    low_confidence_threshold: float | None = None,
) -> np.ndarray:
    if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
        raise ValueError("frame must be HxWx3 RGB")
    if source_prediction.shape != frame_rgb.shape[:2] or confidence.shape != frame_rgb.shape[:2]:
        raise ValueError("prediction and confidence must match the source frame")
    if panel_width <= 0:
        raise ValueError("panel_width must be positive")
    if low_confidence_threshold is not None and not 0.0 <= low_confidence_threshold <= 1.0:
        raise ValueError("low-confidence threshold must be between 0 and 1")
    values = set(int(value) for value in np.unique(source_prediction))
    if not values.issubset({0, 1, 2, 3}):
        raise ValueError(f"prediction contains unsupported source class IDs: {sorted(values)}")

    display_prediction = source_prediction.copy()
    low_confidence = None
    if low_confidence_threshold is not None:
        low_confidence = confidence < low_confidence_threshold
        display_prediction[low_confidence] = 0
    color_rgb = DISPLAY_COLORS_RGB[display_prediction]
    overlay_rgb = cv2.addWeighted(frame_rgb, 0.55, color_rgb, 0.45, 0.0)
    if low_confidence is not None:
        overlay_rgb[low_confidence] = (overlay_rgb[low_confidence] * 0.35).astype(np.uint8)

    panel_height = max(2, int(round(panel_width * frame_rgb.shape[0] / frame_rgb.shape[1])))
    panel_height += panel_height % 2
    original = _fit_rgb(frame_rgb, panel_width, panel_height)
    overlay = _fit_rgb(overlay_rgb, panel_width, panel_height)
    mask = _fit_rgb(color_rgb, panel_width, panel_height, nearest=True)
    body = cv2.cvtColor(np.concatenate((original, overlay, mask), axis=1), cv2.COLOR_RGB2BGR)
    header_height = 78
    canvas = np.zeros((header_height + panel_height, panel_width * 3, 3), dtype=np.uint8)
    canvas[header_height:] = body
    titles = ("ORIGINAL", "SEGMENTATION OVERLAY", "PREDICTION MASK")
    for index, title in enumerate(titles):
        cv2.putText(
            canvas,
            title,
            (index * panel_width + 10, header_height - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
    line1 = (
        f"{metadata['dataset']}  ride={metadata['ride_id']}  frame={metadata['frame_id']}  "
        f"timestamp={float(metadata['timestamp']):.3f}"
    )
    line2 = (
        f"checkpoint={metadata['checkpoint_version']}  inference={float(metadata['inference_latency_ms']):.1f}ms  "
        f"measured_fps={float(metadata['measured_fps']):.2f}"
    )
    if low_confidence_threshold is not None:
        line2 += f"  low_conf<{low_confidence_threshold:.2f}=black"
    cv2.putText(canvas, line1, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(canvas, line2, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1, cv2.LINE_AA)
    return canvas


def process_dataset_review(
    dataset_root: Path,
    segments: list[ReviewSegment],
    skipped_rides: list[dict[str, object]],
    decoder: FrameDecoder,
    predictor: Predictor,
    output_dir: Path,
    output_fps: float,
    panel_width: int,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    low_confidence_threshold: float | None,
    writer_factory: Callable[[Path, float, tuple[int, int]], VideoWriter],
    discovery: dict[str, int] | None = None,
    model_metadata: dict[str, object] | None = None,
    reported_output_dir: Path | None = None,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=False)
    video_path = output_dir / "traversability_review.mp4"
    reported_video_path = (reported_output_dir or output_dir) / video_path.name
    writer: VideoWriter | None = None
    processed = 0
    skipped_frames: list[dict[str, object]] = []
    latencies: list[float] = []
    started = time.monotonic()
    try:
        for segment in segments:
            for frame in segment.frames:
                try:
                    frame_rgb = decoder.decode(frame)
                    prediction, confidence, latency_ms = predictor.predict(frame_rgb)
                except Exception as exc:
                    skipped_frames.append(
                        {
                            "ride_id": frame.ride_id,
                            "frame_id": frame.frame_id,
                            "timestamp": frame.timestamp,
                            "reason": type(exc).__name__,
                            "detail": str(exc),
                        }
                    )
                    continue
                elapsed = time.monotonic() - started
                measured_fps = (processed + 1) / elapsed if elapsed else 0.0
                composed = compose_three_panel_frame(
                    frame_rgb,
                    prediction,
                    confidence,
                    {
                        "dataset": frame.dataset,
                        "ride_id": frame.ride_id,
                        "frame_id": frame.frame_id,
                        "timestamp": frame.timestamp,
                        "checkpoint_version": predictor.checkpoint_version,
                        "inference_latency_ms": latency_ms,
                        "measured_fps": measured_fps,
                    },
                    panel_width,
                    low_confidence_threshold,
                )
                if writer is None:
                    height, width = composed.shape[:2]
                    writer = writer_factory(video_path, output_fps, (width, height))
                writer.write(composed)
                processed += 1
                latencies.append(float(latency_ms))
    finally:
        video_info = writer.close() if writer is not None else {}
    elapsed = time.monotonic() - started
    requested_frames = sum(len(segment.frames) for segment in segments)
    report = {
        "success": processed > 0 and bool(video_info),
        "dataset_name": dataset_root.name,
        "dataset_path": str(dataset_root),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_version": predictor.checkpoint_version,
        "model": model_metadata or {},
        "selected_ride_count": len(segments),
        "selected_segments": [
            {
                **{key: value for key, value in asdict(segment).items() if key != "frames"},
                "sampled_frame_count": len(segment.frames),
                "first_frame_id": segment.frames[0].frame_id if segment.frames else None,
                "last_frame_id": segment.frames[-1].frame_id if segment.frames else None,
            }
            for segment in segments
        ],
        "skipped_rides": skipped_rides,
        "discovery": discovery or {},
        "requested_frame_count": requested_frames,
        "processed_frame_count": processed,
        "skipped_frame_count": len(skipped_frames),
        "skipped_frames": skipped_frames,
        "inference_latency_ms": latency_summary(latencies),
        "effective_fps": processed / elapsed if elapsed else 0.0,
        "output_fps": output_fps,
        "output_file_path": str(reported_video_path),
        "output_video": video_info,
        "low_confidence_visualization_threshold": low_confidence_threshold,
        "raw_prediction_modified": False,
        "temporal_smoothing_applied": False,
        "sdk_or_live_rover_commands_sent": False,
    }
    write_json(output_dir / "review_manifest.json", report)
    return report


def latency_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


class H264VideoWriter:
    def __init__(self, path: Path, fps: float, frame_size: tuple[int, int]) -> None:
        self.path = path
        self.frame_size = frame_size
        self.log_path = path.with_suffix(".ffmpeg.log")
        width, height = frame_size
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{fps:g}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ]
        self.log_handle = self.log_path.open("wb")
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=self.log_handle,
        )

    def write(self, frame_bgr: np.ndarray) -> None:
        width, height = self.frame_size
        if frame_bgr.shape != (height, width, 3) or frame_bgr.dtype != np.uint8:
            raise ValueError("video frame shape or dtype differs from writer configuration")
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg stdin is unavailable")
        self.process.stdin.write(np.ascontiguousarray(frame_bgr).tobytes())

    def close(self) -> dict[str, object]:
        if self.process.stdin is not None:
            self.process.stdin.close()
        return_code = self.process.wait()
        self.log_handle.close()
        if return_code != 0:
            detail = self.log_path.read_text(encoding="utf-8", errors="replace")
            raise OSError(f"ffmpeg H.264 encoding failed: {detail.strip()}")
        self.log_path.unlink(missing_ok=True)
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,pix_fmt,width,height,r_frame_rate",
                "-of",
                "json",
                str(self.path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        stream = json.loads(probe.stdout)["streams"][0]
        if stream.get("codec_name") != "h264" or stream.get("pix_fmt") != "yuv420p":
            raise OSError(f"output is not QuickTime-compatible H.264 yuv420p: {stream}")
        return {
            "codec": stream["codec_name"],
            "pixel_format": stream["pix_fmt"],
            "width": int(stream["width"]),
            "height": int(stream["height"]),
            "frame_rate": stream["r_frame_rate"],
            "size_bytes": self.path.stat().st_size,
        }


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


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
