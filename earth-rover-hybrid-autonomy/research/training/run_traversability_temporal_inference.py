#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import html
import importlib.metadata
import json
import os
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import cv2
import numpy as np
import torch
import torch.nn.functional as functional
import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.training.datasets.frodobots_2k_dataset import (
    FrameDecodeError,
    HlsFrameDecoder,
    ManifestSample,
    load_manifest,
)
from research.training.datasets.frodobots_2k_manifest import (
    DatasetFormatError,
    normalize_timestamp,
    parse_front_hls_playlist,
)
from research.training.datasets.traversability_dataset_v1 import (
    TRAINING_CLASS_NAMES,
    image_rgb_to_tensor,
    letterbox_image,
    load_approved_manifest,
    restore_letterbox,
    training_prediction_to_source,
)
from research.training.models.traversability_segformer import build_traversability_segformer


SOURCE_COLORS_RGB = np.array(
    [[0, 0, 0], [38, 166, 91], [43, 126, 216], [220, 50, 47]],
    dtype=np.uint8,
)


@dataclass(frozen=True)
class IndexedSample:
    manifest_index: int
    sample: ManifestSample


@dataclass(frozen=True)
class TemporalSegment:
    segment_id: str
    ride_id: str
    start_timestamp: float
    end_timestamp: float
    duration_seconds: float
    start_manifest_index: int
    end_manifest_index: int
    frame_count: int
    aligned_manifest_frame_count: int
    timeline_section_id: int
    action_distribution: dict[str, int]
    selection_score: float
    selection_reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline temporal SegFormer inference on unseen rides.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--full-manifest", required=True)
    parser.add_argument("--approved-manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    full_manifest = Path(args.full_manifest).expanduser().resolve()
    approved_manifest = Path(args.approved_manifest).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    for path in (dataset_root, full_manifest, approved_manifest, checkpoint_path, config_path):
        if not path.exists():
            raise SystemExit(f"required input does not exist: {path}")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    shutil.copy2(config_path, output / "frozen_config.yaml")
    _set_deterministic(int(config["seed"]))
    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit("CUDA is required but unavailable")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    all_samples = load_manifest(full_manifest)
    approved_samples = load_approved_manifest(approved_manifest)
    excluded_rides = {sample.ride_id for sample in approved_samples}
    segments, selected = select_temporal_segments(
        all_samples,
        excluded_rides,
        ride_count=int(config["ride_count"]),
        duration_seconds=float(config["segment_duration_seconds"]),
        maximum_gap_seconds=float(config["maximum_manifest_gap_seconds"]),
        seed=int(config["seed"]),
    )
    segments, selected, raw_timeline_exclusions = expand_selected_segments_from_raw_timestamps(
        dataset_root,
        segments,
        selected,
        maximum_gap_seconds=float(config["maximum_raw_frame_gap_seconds"]),
    )
    if {segment.ride_id for segment in segments} & excluded_rides:
        raise SystemExit("selected temporal ride overlaps approved train/validation/test rides")
    _write_json(
        output / "selected_segments.json",
        {
            "segments": [asdict(segment) for segment in segments],
            "approved_split_ride_ids": sorted(excluded_rides),
            "ride_overlap": [],
            "test_split_evaluated": False,
            "selection_source": str(full_manifest),
            "inference_frame_source": "raw front_camera_timestamps CSV mapped to HLS; control-aligned manifest used only for ride/window selection",
            "raw_timeline_exclusions": raw_timeline_exclusions,
        },
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = build_traversability_segformer(False, checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    decoder = HlsFrameDecoder(dataset_root)
    started = time.monotonic()
    result = run_temporal_inference(
        model,
        decoder,
        segments,
        selected,
        output,
        device,
        image_size=int(config["image_size"]),
        panel_width=int(config["review_panel_width"]),
        anomaly_thresholds=config["anomaly_review_heuristics"],
        maximum_anomaly_images=int(config["maximum_anomaly_images"]),
    )
    elapsed = time.monotonic() - started
    report = {
        "success": result["successful_frame_count"] > 0 and result["video_count"] == len(segments),
        "pipeline_status": "HUMAN_TEMPORAL_REVIEW_REQUIRED",
        "selected_ride_count": len(segments),
        "selected_frame_count": sum(segment.frame_count for segment in segments),
        **result,
        "total_elapsed_seconds": elapsed,
        "effective_fps": result["successful_frame_count"] / elapsed if elapsed else 0.0,
        "peak_vram_bytes": {
            "allocated": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
            "reserved": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0,
        },
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "packages": _package_versions(),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "source_checkpoint": checkpoint.get("source_checkpoint"),
        "source_revision": checkpoint.get("source_revision"),
        "full_manifest_path": str(full_manifest),
        "full_manifest_sha256": _sha256(full_manifest),
        "approved_manifest_path": str(approved_manifest),
        "approved_manifest_sha256": _sha256(approved_manifest),
        "approved_ride_overlap": [],
        "preprocessing": "training-identical aspect-ratio-preserving letterbox and ImageNet normalization",
        "class_mapping": {"0": "ON_ROAD", "1": "OFF_ROAD", "2": "OBSTACLE"},
        "confidence_threshold_applied": False,
        "temporal_smoothing_applied": False,
        "anomaly_heuristics_affect_predictions": False,
        "test_split_evaluated": False,
        "additional_training_performed": False,
        "sdk_or_live_rover_commands_sent": False,
    }
    _write_json(output / "temporal_inference_report.json", report)
    _write_review_html(output, segments)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["success"] else 1


def select_temporal_segments(
    samples: tuple[ManifestSample, ...],
    excluded_rides: set[str],
    ride_count: int,
    duration_seconds: float,
    maximum_gap_seconds: float,
    seed: int,
) -> tuple[list[TemporalSegment], dict[str, list[IndexedSample]]]:
    if not 3 <= ride_count <= 5:
        raise ValueError("ride_count must be between 3 and 5")
    if not 30.0 <= duration_seconds <= 60.0:
        raise ValueError("segment duration must be between 30 and 60 seconds")
    grouped: dict[str, list[IndexedSample]] = defaultdict(list)
    for index, sample in enumerate(samples):
        if sample.ride_id not in excluded_rides:
            grouped[sample.ride_id].append(IndexedSample(index, sample))

    candidates: list[tuple[float, str, list[IndexedSample], Counter[str]]] = []
    for ride_id, ride_samples in grouped.items():
        chronological = sorted(
            ride_samples,
            key=lambda item: (item.sample.front_timestamp, item.sample.front_frame_id, item.manifest_index),
        )
        best: tuple[float, list[IndexedSample], Counter[str]] | None = None
        for run in _continuous_runs(chronological, maximum_gap_seconds):
            for window in _duration_windows(run, duration_seconds):
                actions = Counter(item.sample.action_class for item in window)
                turn_fraction = (actions["LEFT"] + actions["RIGHT"]) / len(window)
                diversity = len(actions)
                digest = hashlib.sha256(
                    f"{seed}:{ride_id}:{window[0].manifest_index}".encode()
                ).hexdigest()
                tie = int(digest[:8], 16) / 0xFFFFFFFF
                score = 4.0 * turn_fraction + 0.1 * diversity + 1e-6 * tie
                ranked = (score, window, actions)
                if best is None or score > best[0]:
                    best = ranked
        if best is not None:
            candidates.append((best[0], ride_id, best[1], best[2]))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    if len(candidates) < ride_count:
        raise ValueError(
            f"only {len(candidates)} unseen rides contain a valid {duration_seconds:.0f}s continuous window"
        )

    segments: list[TemporalSegment] = []
    selected: dict[str, list[IndexedSample]] = {}
    for order, (score, ride_id, window, actions) in enumerate(candidates[:ride_count]):
        segment_id = f"segment_{order + 1:02d}_ride_{ride_id}"
        selected[segment_id] = window
        segments.append(
            TemporalSegment(
                segment_id=segment_id,
                ride_id=ride_id,
                start_timestamp=window[0].sample.front_timestamp,
                end_timestamp=window[-1].sample.front_timestamp,
                duration_seconds=window[-1].sample.front_timestamp - window[0].sample.front_timestamp,
                start_manifest_index=window[0].manifest_index,
                end_manifest_index=window[-1].manifest_index,
                frame_count=len(window),
                aligned_manifest_frame_count=len(window),
                timeline_section_id=window[0].sample.timeline_section_id,
                action_distribution=dict(sorted(actions.items())),
                selection_score=score,
                selection_reason="unseen ride; continuous manifest window prioritized for LEFT/RIGHT reference actions",
            )
        )
    return segments, selected


def expand_selected_segments_from_raw_timestamps(
    dataset_root: Path,
    segments: list[TemporalSegment],
    aligned: dict[str, list[IndexedSample]],
    maximum_gap_seconds: float,
) -> tuple[list[TemporalSegment], dict[str, list[IndexedSample]], list[dict[str, object]]]:
    expanded_segments: list[TemporalSegment] = []
    expanded: dict[str, list[IndexedSample]] = {}
    exclusions: list[dict[str, object]] = []
    for segment in segments:
        ride_dirs = sorted(dataset_root.glob(f"ride_{segment.ride_id}_*"))
        if len(ride_dirs) != 1:
            raise DatasetFormatError(
                f"expected exactly one raw directory for ride {segment.ride_id}, found {len(ride_dirs)}"
            )
        ride_dir = ride_dirs[0]
        front_csv = ride_dir / f"front_camera_timestamps_{segment.ride_id}.csv"
        playlists = sorted((ride_dir / "recordings").glob("*uid_s_1000*video.m3u8"))
        if not front_csv.is_file() or len(playlists) != 1:
            raise DatasetFormatError(f"raw front timeline is incomplete for ride {segment.ride_id}")
        hls = parse_front_hls_playlist(playlists[0], dataset_root)
        aligned_by_frame = {
            item.sample.front_frame_id: item for item in aligned[segment.segment_id]
        }
        raw_items: list[IndexedSample] = []
        previous_timestamp: float | None = None
        previous_frame_id: int | None = None
        with front_csv.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not {"frame_id", "timestamp"}.issubset(reader.fieldnames or []):
                raise DatasetFormatError(f"invalid front timestamp schema: {front_csv}")
            for row_index, row in enumerate(reader):
                try:
                    frame_id = int(row["frame_id"])
                    timestamp = normalize_timestamp(row["timestamp"], "seconds")
                except (TypeError, ValueError, DatasetFormatError) as exc:
                    exclusions.append(
                        {
                            "ride_id": segment.ride_id,
                            "row_index": row_index,
                            "reason": "malformed_front_row",
                            "detail": str(exc),
                        }
                    )
                    continue
                if timestamp < segment.start_timestamp - 1e-9 or timestamp > segment.end_timestamp + 1e-9:
                    continue
                if previous_timestamp is not None:
                    if timestamp <= previous_timestamp or frame_id <= int(previous_frame_id):
                        raise DatasetFormatError(f"non-monotonic raw front timeline in selected ride {segment.ride_id}")
                    if timestamp - previous_timestamp > maximum_gap_seconds:
                        raise DatasetFormatError(
                            f"raw front timeline gap exceeds {maximum_gap_seconds:.3f}s in selected ride {segment.ride_id}"
                        )
                previous_timestamp = timestamp
                previous_frame_id = frame_id
                hls_segment = hls.find_segment(timestamp)
                if hls_segment is None or hls_segment.section_id != segment.timeline_section_id:
                    exclusions.append(
                        {
                            "ride_id": segment.ride_id,
                            "row_index": row_index,
                            "frame_id": frame_id,
                            "timestamp": timestamp,
                            "reason": "outside_selected_hls_section",
                        }
                    )
                    continue
                aligned_item = aligned_by_frame.get(frame_id)
                reference = aligned_item.sample if aligned_item else None
                raw_sample = ManifestSample(
                    ride_id=segment.ride_id,
                    front_playlist_ref=hls.playlist_reference,
                    front_segment_ref=hls_segment.reference,
                    front_frame_id=frame_id,
                    front_timestamp=timestamp,
                    matched_control_timestamp=reference.matched_control_timestamp if reference else timestamp,
                    control_delta_ms=reference.control_delta_ms if reference else 0.0,
                    linear=reference.linear if reference else 0.0,
                    angular=reference.angular if reference else 0.0,
                    action_class=reference.action_class if reference else "UNALIGNED",
                    timeline_section_id=hls_segment.section_id,
                )
                raw_items.append(
                    IndexedSample(aligned_item.manifest_index if aligned_item else -1, raw_sample)
                )
        if len(raw_items) < 2:
            raise DatasetFormatError(f"no usable raw front timeline for selected segment {segment.segment_id}")
        duration = raw_items[-1].sample.front_timestamp - raw_items[0].sample.front_timestamp
        if duration < 29.9:
            raise DatasetFormatError(
                f"raw timeline for {segment.segment_id} covers only {duration:.3f}s"
            )
        expanded[segment.segment_id] = raw_items
        expanded_segments.append(
            replace(
                segment,
                start_timestamp=raw_items[0].sample.front_timestamp,
                end_timestamp=raw_items[-1].sample.front_timestamp,
                duration_seconds=duration,
                frame_count=len(raw_items),
            )
        )
    return expanded_segments, expanded, exclusions


def _continuous_runs(
    samples: list[IndexedSample],
    maximum_gap_seconds: float,
) -> list[list[IndexedSample]]:
    runs: list[list[IndexedSample]] = []
    current: list[IndexedSample] = []
    for item in samples:
        if current:
            previous = current[-1].sample
            delta = item.sample.front_timestamp - previous.front_timestamp
            discontinuity = (
                delta <= 0.0
                or delta > maximum_gap_seconds
                or item.sample.front_frame_id <= previous.front_frame_id
                or item.sample.timeline_section_id != previous.timeline_section_id
            )
            if discontinuity:
                runs.append(current)
                current = []
        current.append(item)
    if current:
        runs.append(current)
    return runs


def _duration_windows(run: list[IndexedSample], duration_seconds: float) -> list[list[IndexedSample]]:
    if len(run) < 2 or run[-1].sample.front_timestamp - run[0].sample.front_timestamp < duration_seconds:
        return []
    candidate_starts = {0, len(run) // 3, (2 * len(run)) // 3}
    turn_starts = [
        index for index, item in enumerate(run) if item.sample.action_class in {"LEFT", "RIGHT"}
    ]
    if turn_starts:
        stride = max(1, len(turn_starts) // 50)
        candidate_starts.update(turn_starts[::stride])
    timestamps = [item.sample.front_timestamp for item in run]
    windows: list[list[IndexedSample]] = []
    for start in sorted(candidate_starts):
        end_timestamp = run[start].sample.front_timestamp + duration_seconds
        end = bisect.bisect_right(timestamps, end_timestamp + 1e-9, lo=start)
        window = run[start:end]
        if window and window[-1].sample.front_timestamp - window[0].sample.front_timestamp >= duration_seconds - 0.1:
            windows.append(window)
    return windows


@torch.inference_mode()
def run_temporal_inference(
    model: torch.nn.Module,
    decoder: HlsFrameDecoder,
    segments: list[TemporalSegment],
    selected: dict[str, list[IndexedSample]],
    output: Path,
    device: torch.device,
    image_size: int,
    panel_width: int,
    anomaly_thresholds: dict[str, float],
    maximum_anomaly_images: int,
) -> dict[str, object]:
    videos_dir = output / "videos"
    anomalies_dir = output / "anomaly_candidates"
    videos_dir.mkdir()
    anomalies_dir.mkdir()
    records: list[dict[str, object]] = []
    anomalies: list[dict[str, object]] = []
    model_latencies: list[float] = []
    end_to_end_latencies: list[float] = []
    decode_failures: list[dict[str, object]] = []
    segment_reports: list[dict[str, object]] = []
    video_count = 0

    for segment in segments:
        segment_started = time.monotonic()
        model_latency_start = len(model_latencies)
        end_to_end_latency_start = len(end_to_end_latencies)
        failure_start = len(decode_failures)
        items = selected[segment.segment_id]
        frame_deltas = [
            current.sample.front_timestamp - previous.sample.front_timestamp
            for previous, current in zip(items, items[1:])
        ]
        output_fps = 1.0 / float(np.median(frame_deltas))
        writer: cv2.VideoWriter | None = None
        video_path = videos_dir / f"{segment.segment_id}.mp4"
        previous_state: dict[str, object] | None = None
        segment_successes = 0
        for sequence_index, item in enumerate(items):
            frame_started = time.monotonic()
            sample = item.sample
            try:
                frame_rgb = decoder.decode(sample)
            except FrameDecodeError as exc:
                failure = {
                    "segment_id": segment.segment_id,
                    "ride_id": sample.ride_id,
                    "manifest_index": item.manifest_index,
                    "frame_id": sample.front_frame_id,
                    "timestamp": sample.front_timestamp,
                    "error": str(exc),
                }
                decode_failures.append(failure)
                records.append({**failure, "status": "DECODE_FAILED"})
                previous_state = None
                continue

            padded, _ = letterbox_image(frame_rgb, image_size)
            tensor = image_rgb_to_tensor(padded).unsqueeze(0).to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            model_started = time.monotonic()
            logits = model(tensor).logits
            logits = functional.interpolate(
                logits,
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False,
            )
            probabilities = logits.softmax(dim=1)
            confidence_tensor, prediction_tensor = probabilities.max(dim=1)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            model_latency_ms = (time.monotonic() - model_started) * 1000.0

            prediction = prediction_tensor[0].cpu().numpy().astype(np.uint8)
            confidence = confidence_tensor[0].cpu().numpy().astype(np.float32)
            prediction = restore_letterbox(prediction, frame_rgb.shape[:2], cv2.INTER_NEAREST)
            confidence = restore_letterbox(confidence, frame_rgb.shape[:2], cv2.INTER_LINEAR)
            source_prediction = training_prediction_to_source(prediction)
            ratios = {
                name: float(np.mean(prediction == class_id))
                for class_id, name in enumerate(TRAINING_CLASS_NAMES)
            }
            current_state = {
                "prediction": prediction,
                "class_ratios": ratios,
                "mean_confidence": float(confidence.mean()),
            }
            reasons, temporal = temporal_anomaly_reasons(previous_state, current_state, anomaly_thresholds)
            record = {
                "status": "OK",
                "segment_id": segment.segment_id,
                "sequence_index": sequence_index,
                "ride_id": sample.ride_id,
                "manifest_index": item.manifest_index,
                "frame_id": sample.front_frame_id,
                "timestamp": sample.front_timestamp,
                "playlist": sample.front_playlist_ref,
                "hls_segment": sample.front_segment_ref,
                "action_reference": sample.action_class,
                "linear_reference": sample.linear,
                "angular_reference": sample.angular,
                "class_pixel_ratios": ratios,
                "mean_confidence": current_state["mean_confidence"],
                "minimum_confidence": float(confidence.min()),
                "model_only_latency_ms": model_latency_ms,
                **temporal,
                "anomaly_reasons": reasons,
            }
            composed = compose_review_frame(frame_rgb, source_prediction, confidence, record, panel_width)
            if writer is None:
                height, width = composed.shape[:2]
                writer = cv2.VideoWriter(
                    str(video_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    output_fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise OSError(f"cannot open MP4 writer: {video_path}")
            writer.write(composed)
            end_to_end_ms = (time.monotonic() - frame_started) * 1000.0
            record["end_to_end_latency_ms"] = end_to_end_ms
            records.append(record)
            model_latencies.append(model_latency_ms)
            end_to_end_latencies.append(end_to_end_ms)
            segment_successes += 1
            if reasons:
                anomaly = {
                    "segment_id": segment.segment_id,
                    "sequence_index": sequence_index,
                    "ride_id": sample.ride_id,
                    "frame_id": sample.front_frame_id,
                    "timestamp": sample.front_timestamp,
                    "reasons": reasons,
                    "mean_confidence": current_state["mean_confidence"],
                    **temporal,
                }
                if len(anomalies) < maximum_anomaly_images:
                    relative = f"anomaly_candidates/{segment.segment_id}_{sequence_index:05d}.jpg"
                    if not cv2.imwrite(str(output / relative), composed):
                        raise OSError(f"cannot write anomaly review image: {relative}")
                    anomaly["image_path"] = relative
                anomalies.append(anomaly)
            previous_state = current_state
        if writer is not None:
            writer.release()
        if segment_successes > 0 and video_path.is_file() and video_path.stat().st_size > 0:
            video_count += 1
        segment_elapsed = time.monotonic() - segment_started
        segment_reports.append(
            {
                "segment_id": segment.segment_id,
                "ride_id": segment.ride_id,
                "selected_frame_count": len(items),
                "successful_frame_count": segment_successes,
                "decode_failure_count": len(decode_failures) - failure_start,
                "output_video_fps": output_fps,
                "effective_fps": segment_successes / segment_elapsed if segment_elapsed else 0.0,
                "model_only_latency_ms": latency_summary(model_latencies[model_latency_start:]),
                "end_to_end_latency_ms": latency_summary(end_to_end_latencies[end_to_end_latency_start:]),
                "video_path": f"videos/{segment.segment_id}.mp4",
            }
        )

    _write_frame_statistics(output / "per_frame_statistics.csv", records)
    _write_json(output / "anomaly_candidates.json", anomalies)
    return {
        "successful_frame_count": sum(record["status"] == "OK" for record in records),
        "decode_failure_count": len(decode_failures),
        "decode_failures": decode_failures,
        "video_count": video_count,
        "per_segment": segment_reports,
        "anomaly_candidate_count": len(anomalies),
        "anomaly_image_count": min(len(anomalies), maximum_anomaly_images),
        "latency_ms": {
            "model_only": latency_summary(model_latencies),
            "end_to_end": latency_summary(end_to_end_latencies),
        },
        "video_paths": [f"videos/{segment.segment_id}.mp4" for segment in segments],
        "per_frame_statistics_path": "per_frame_statistics.csv",
        "anomaly_candidates_path": "anomaly_candidates.json",
    }


def temporal_anomaly_reasons(
    previous: dict[str, object] | None,
    current: dict[str, object],
    thresholds: dict[str, float],
) -> tuple[list[str], dict[str, float | None]]:
    ratios = current["class_ratios"]
    reasons: list[str] = []
    mean_confidence = float(current["mean_confidence"])
    if mean_confidence < float(thresholds["low_mean_confidence"]):
        reasons.append("low_mean_confidence")
    if max(float(value) for value in ratios.values()) >= float(thresholds["dominant_class_fraction"]):
        reasons.append("single_class_collapse_candidate")
    temporal: dict[str, float | None] = {
        "pixel_flicker_fraction": None,
        "class_ratio_l1_change": None,
        "mean_confidence_drop": None,
        "obstacle_fraction_drop": None,
    }
    if previous is None:
        return reasons, temporal
    previous_prediction = np.asarray(previous["prediction"])
    current_prediction = np.asarray(current["prediction"])
    flicker = float(np.mean(previous_prediction != current_prediction))
    ratio_change = float(
        sum(abs(float(ratios[name]) - float(previous["class_ratios"][name])) for name in TRAINING_CLASS_NAMES)
    )
    confidence_drop = float(previous["mean_confidence"]) - mean_confidence
    obstacle_drop = float(previous["class_ratios"]["OBSTACLE"]) - float(ratios["OBSTACLE"])
    temporal.update(
        {
            "pixel_flicker_fraction": flicker,
            "class_ratio_l1_change": ratio_change,
            "mean_confidence_drop": confidence_drop,
            "obstacle_fraction_drop": obstacle_drop,
        }
    )
    if flicker >= float(thresholds["pixel_flicker_fraction"]):
        reasons.append("high_pixel_flicker")
    if ratio_change >= float(thresholds["class_ratio_l1_change"]):
        reasons.append("large_class_ratio_change")
    if confidence_drop >= float(thresholds["mean_confidence_drop"]):
        reasons.append("confidence_drop")
    if obstacle_drop >= float(thresholds["obstacle_fraction_drop"]):
        reasons.append("obstacle_disappearance_candidate")
    return reasons, temporal


def compose_review_frame(
    frame_rgb: np.ndarray,
    source_prediction: np.ndarray,
    confidence: np.ndarray,
    record: dict[str, object],
    panel_width: int,
) -> np.ndarray:
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    mask_bgr = cv2.cvtColor(SOURCE_COLORS_RGB[source_prediction], cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(frame_bgr, 0.55, mask_bgr, 0.45, 0.0)
    confidence_bgr = cv2.applyColorMap(
        np.clip(confidence * 255.0, 0, 255).astype(np.uint8),
        cv2.COLORMAP_VIRIDIS,
    )
    labels = ("Original", "Prediction mask", "Prediction overlay", "Raw max-softmax confidence")
    panels = []
    for image, label in zip((frame_bgr, mask_bgr, overlay, confidence_bgr), labels):
        scale = panel_width / image.shape[1]
        resized = cv2.resize(image, (panel_width, round(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        cv2.rectangle(resized, (0, 0), (panel_width, 30), (0, 0, 0), -1)
        cv2.putText(resized, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(resized)
    top = np.hstack(panels[:2])
    bottom = np.hstack(panels[2:])
    canvas = np.vstack((top, bottom))
    ratios = record["class_pixel_ratios"]
    text_lines = (
        f"ride={record['ride_id']} frame={record['frame_id']} ts={float(record['timestamp']):.3f}",
        f"ON={ratios['ON_ROAD']:.3f} OFF={ratios['OFF_ROAD']:.3f} OBS={ratios['OBSTACLE']:.3f}",
        f"confidence={float(record['mean_confidence']):.3f} model={float(record['model_only_latency_ms']):.1f}ms",
    )
    for index, line in enumerate(text_lines):
        cv2.putText(canvas, line, (10, 55 + index * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def latency_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "max": 0.0, "mean": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def _write_frame_statistics(path: Path, records: list[dict[str, object]]) -> None:
    fieldnames = (
        "status", "segment_id", "sequence_index", "ride_id", "manifest_index", "frame_id",
        "timestamp", "playlist", "hls_segment", "action_reference", "linear_reference",
        "angular_reference", "on_road_ratio", "off_road_ratio", "obstacle_ratio",
        "mean_confidence", "minimum_confidence", "model_only_latency_ms",
        "end_to_end_latency_ms", "pixel_flicker_fraction", "class_ratio_l1_change",
        "mean_confidence_drop", "obstacle_fraction_drop", "anomaly_reasons", "error",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            ratios = record.get("class_pixel_ratios", {})
            writer.writerow(
                {
                    **{name: record.get(name, "") for name in fieldnames},
                    "on_road_ratio": ratios.get("ON_ROAD", ""),
                    "off_road_ratio": ratios.get("OFF_ROAD", ""),
                    "obstacle_ratio": ratios.get("OBSTACLE", ""),
                    "anomaly_reasons": "|".join(record.get("anomaly_reasons", [])),
                }
            )


def _write_review_html(output: Path, segments: list[TemporalSegment]) -> None:
    cards = []
    for segment in segments:
        name = html.escape(segment.segment_id)
        cards.append(
            f"<article><h2>{name}</h2><p>ride={html.escape(segment.ride_id)} "
            f"duration={segment.duration_seconds:.2f}s frames={segment.frame_count}</p>"
            f"<video controls preload='metadata' src='videos/{name}.mp4'></video></article>"
        )
    anomaly_links = []
    anomalies = json.loads((output / "anomaly_candidates.json").read_text(encoding="utf-8"))
    for anomaly in anomalies:
        relative = anomaly.get("image_path")
        if relative is None:
            continue
        escaped = html.escape(str(relative))
        reasons = html.escape(", ".join(anomaly["reasons"]))
        anomaly_links.append(
            f"<figure><a href='{escaped}'><img src='{escaped}'></a>"
            f"<figcaption>{reasons}</figcaption></figure>"
        )
    document = """<!doctype html><html><head><meta charset='utf-8'><title>Traversability Temporal Review</title>
<style>body{font-family:system-ui;margin:20px;color:#17191c}article{border-bottom:1px solid #bbb;padding:12px 0}video{width:min(100%,1280px)}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}.grid img{width:100%}@media(max-width:800px){.grid{grid-template-columns:1fr}}</style>
</head><body><h1>Traversability Temporal Review</h1><p>Offline raw argmax predictions and max-softmax confidence. No temporal smoothing or confidence threshold was applied. Anomaly flags are review heuristics only.</p>__CARDS__<h2>Anomaly candidates</h2><div class='grid'>__ANOMALIES__</div></body></html>"""
    document = document.replace("__CARDS__", "\n".join(cards)).replace("__ANOMALIES__", "\n".join(anomaly_links))
    (output / "review.html").write_text(document, encoding="utf-8")
    (output / "README.md").write_text(
        "# Traversability Temporal Review\n\nOpen `review.html` locally. Review flicker, turns, obstacle disappearance, domain shift, collapse, and confidence drops. Predictions are offline research outputs and are not rover control commands.\n",
        encoding="utf-8",
    )


def _package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in ("torch", "torchvision", "transformers", "safetensors", "Pillow", "opencv-python"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


if __name__ == "__main__":
    raise SystemExit(main())
