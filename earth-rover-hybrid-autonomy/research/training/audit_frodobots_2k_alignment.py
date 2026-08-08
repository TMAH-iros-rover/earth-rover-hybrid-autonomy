#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.training.datasets.action_labels import classify_action
from research.training.datasets.frodobots_2k_dataset import (
    FrameDecodeError,
    FrodoBotsActionDataset,
    ManifestSample,
)
from research.training.datasets.frodobots_2k_manifest import parse_front_hls_playlist


@dataclass(frozen=True)
class StripSpec:
    category: str
    name: str
    ride_id: str
    indices: tuple[int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit FrodoBots-2K temporal and semantic alignment.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir == dataset_root or dataset_root in output_dir.parents:
        raise SystemExit("output-dir must remain outside the immutable raw dataset root")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = FrodoBotsActionDataset(dataset_root, args.manifest)
    grouped = _group_indices(dataset.samples)
    hls_audit = _audit_hls_playlists(dataset_root, dataset.samples)
    eligible_rides = [ride_id for ride_id, indices in grouped.items() if len(indices) >= 3]
    anomaly_rides = [ride_id for ride_id in eligible_rides if hls_audit[ride_id]["hls_sections"] > 1]
    selected_rides = (anomaly_rides + [ride for ride in eligible_rides if ride not in anomaly_rides])[:5]
    if len(selected_rides) < 5:
        raise SystemExit(f"only {len(selected_rides)} rides have enough aligned samples")

    position_specs = _select_position_strips(dataset.samples, grouped, selected_rides)
    left_specs = _select_action_strips(dataset.samples, grouped, "LEFT", 5)
    right_specs = _select_action_strips(dataset.samples, grouped, "RIGHT", 5)
    reverse_count = sum(sample.action_class == "REVERSE" for sample in dataset.samples)
    reverse_specs = _select_action_strips(dataset.samples, grouped, "REVERSE", min(5, reverse_count))
    boundary_specs = _select_boundary_strips(dataset.samples, grouped, anomaly_rides, 5)

    decode_failures: list[dict[str, object]] = []
    decoded_by_category: dict[str, list[tuple[StripSpec, list[tuple[dict[str, object], np.ndarray]]]]] = {}
    for category, specs in (
        ("ride_positions", position_specs),
        ("left", left_specs),
        ("right", right_specs),
        ("reverse", reverse_specs),
        ("hls_boundaries", boundary_specs),
    ):
        decoded_by_category[category] = _decode_strips(dataset, specs, decode_failures)
        if decoded_by_category[category]:
            _write_strip_sheet(decoded_by_category[category], output_dir / f"{category}_strips.jpg")

    successful_centers = [
        decoded[0].indices[1]
        for category in ("ride_positions", "left", "right")
        for decoded in decoded_by_category[category]
    ]
    if len(successful_centers) < 4:
        raise SystemExit("fewer than four samples decoded; cannot inspect DataLoader batch")

    deterministic = _repeated_access_is_deterministic(dataset, successful_centers[0])
    batch = next(iter(DataLoader(Subset(dataset, successful_centers[:4]), batch_size=4, shuffle=False)))
    images = batch["image"]
    targets = batch["target"]
    first_raw = decoded_by_category["ride_positions"][0][1][1][1]
    transform_audit = _transform_audit(first_raw, images)
    monotonicity = _audit_monotonicity(dataset.samples, grouped)
    action_mismatches = _action_label_mismatches(dataset.samples)

    decoded_counts = {category: len(rows) for category, rows in decoded_by_category.items()}
    required_decode_counts = {
        "ride_positions": 15,
        "left": 5,
        "right": 5,
        "reverse": min(5, reverse_count),
    }
    automated_checks_passed = (
        all(decoded_counts[name] >= count for name, count in required_decode_counts.items())
        and deterministic
        and tuple(images.shape) == (4, 3, 224, 224)
        and images.dtype == torch.float32
        and targets.dtype == torch.int64
        and bool(torch.isfinite(images).all().item())
        and not action_mismatches
        and monotonicity["frame_id_violation_count"] == 0
        and monotonicity["timestamp_violation_count"] == 0
    )
    report = {
        "phase2_verdict": "CONDITIONAL PASS" if automated_checks_passed else "FAIL",
        "manual_review_required": [
            "Confirm LEFT strips show motion consistent with positive angular commands.",
            "Confirm RIGHT strips show motion consistent with negative angular commands.",
            "Confirm strips do not repeat a stale frame across increasing timestamps.",
        ],
        "manifest_path": str(Path(args.manifest).expanduser().resolve()),
        "manifest_sample_count": len(dataset),
        "manifest_ride_count": len(grouped),
        "selected_ride_ids": selected_rides,
        "hls_anomaly_ride_ids": anomaly_rides,
        "selected_ride_hls_quality": {ride: hls_audit[ride] for ride in selected_rides},
        "decoded_strip_counts": decoded_counts,
        "decode_failure_count": len(decode_failures),
        "decode_failures": decode_failures,
        "reverse_search": {
            "search_scope": "all rows in the supplied full manifest",
            "matching_sample_count": reverse_count,
            "visualized_strip_count": decoded_counts["reverse"],
        },
        "action_label_mismatch_count": len(action_mismatches),
        "action_label_mismatches": action_mismatches[:20],
        "monotonicity": monotonicity,
        "repeated_access_deterministic": deterministic,
        "batch": {
            "image_shape": list(images.shape),
            "image_dtype": str(images.dtype),
            "image_min": float(images.min().item()),
            "image_max": float(images.max().item()),
            "image_mean": float(images.mean().item()),
            "image_std": float(images.std().item()),
            "label_shape": list(targets.shape),
            "label_dtype": str(targets.dtype),
            "rgb_channel_order": "OpenCV BGR is converted to RGB before tensor conversion",
        },
        "transform": transform_audit,
        "artifacts": {
            category: str(output_dir / f"{category}_strips.jpg")
            for category, rows in decoded_by_category.items()
            if rows
        },
    }
    report_path = output_dir / "semantic_alignment_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if automated_checks_passed else 1


def _group_indices(samples: tuple[ManifestSample, ...]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        grouped[sample.ride_id].append(index)
    return dict(grouped)


def _audit_hls_playlists(
    dataset_root: Path,
    samples: tuple[ManifestSample, ...],
) -> dict[str, dict[str, int]]:
    playlist_by_ride: dict[str, str] = {}
    for sample in samples:
        playlist_by_ride.setdefault(sample.ride_id, sample.front_playlist_ref)
    return {
        ride_id: parse_front_hls_playlist(dataset_root / reference, dataset_root).stats
        for ride_id, reference in playlist_by_ride.items()
    }


def _select_position_strips(
    samples: tuple[ManifestSample, ...],
    grouped: dict[str, list[int]],
    ride_ids: list[str],
) -> list[StripSpec]:
    specs: list[StripSpec] = []
    for ride_id in ride_ids:
        indices = grouped[ride_id]
        for name, fraction in (("early", 0.1), ("middle", 0.5), ("late", 0.9)):
            center = _nearest_triplet_center(samples, indices, round((len(indices) - 1) * fraction), True)
            if center is not None:
                specs.append(StripSpec("ride_positions", f"{ride_id}_{name}", ride_id, center))
    return specs


def _select_action_strips(
    samples: tuple[ManifestSample, ...],
    grouped: dict[str, list[int]],
    action: str,
    count: int,
) -> list[StripSpec]:
    if count == 0:
        return []
    candidates = [index for index, sample in enumerate(samples) if sample.action_class == action]
    if not candidates:
        return []
    positions = {index: position for indices in grouped.values() for position, index in enumerate(indices)}
    used_rides: set[str] = set()
    specs: list[StripSpec] = []
    for distinct_rides_only in (True, False):
        for index in candidates:
            sample = samples[index]
            if distinct_rides_only and sample.ride_id in used_rides:
                continue
            triplet = _nearest_triplet_center(samples, grouped[sample.ride_id], positions[index], True)
            if triplet is None or samples[triplet[1]].action_class != action:
                continue
            key = (sample.ride_id, triplet)
            if any((spec.ride_id, spec.indices) == key for spec in specs):
                continue
            specs.append(StripSpec(action.lower(), f"{action.lower()}_{len(specs) + 1}", sample.ride_id, triplet))
            used_rides.add(sample.ride_id)
            if len(specs) == count:
                return specs
    return specs


def _select_boundary_strips(
    samples: tuple[ManifestSample, ...],
    grouped: dict[str, list[int]],
    ride_ids: list[str],
    count: int,
) -> list[StripSpec]:
    specs: list[StripSpec] = []
    for ride_id in ride_ids:
        indices = grouped[ride_id]
        for position in range(1, len(indices) - 1):
            previous = samples[indices[position - 1]]
            current = samples[indices[position]]
            if previous.timeline_section_id == current.timeline_section_id:
                continue
            specs.append(
                StripSpec(
                    "hls_boundaries",
                    f"{ride_id}_section_{previous.timeline_section_id}_to_{current.timeline_section_id}",
                    ride_id,
                    (indices[position - 1], indices[position], indices[position + 1]),
                )
            )
            if len(specs) == count:
                return specs
    return specs


def _nearest_triplet_center(
    samples: tuple[ManifestSample, ...],
    indices: list[int],
    target_position: int,
    require_same_section: bool,
) -> tuple[int, int, int] | None:
    positions = sorted(range(1, len(indices) - 1), key=lambda position: abs(position - target_position))
    for position in positions:
        triplet = tuple(indices[position - 1 : position + 2])
        selected = [samples[index] for index in triplet]
        if require_same_section and len({sample.timeline_section_id for sample in selected}) != 1:
            continue
        if not all(
            selected[offset + 1].front_frame_id == selected[offset].front_frame_id + 1
            and 0.0 < selected[offset + 1].front_timestamp - selected[offset].front_timestamp <= 0.075
            for offset in (0, 1)
        ):
            continue
        return triplet
    return None


def _decode_strips(
    dataset: FrodoBotsActionDataset,
    specs: list[StripSpec],
    failures: list[dict[str, object]],
) -> list[tuple[StripSpec, list[tuple[dict[str, object], np.ndarray]]]]:
    decoded: list[tuple[StripSpec, list[tuple[dict[str, object], np.ndarray]]]] = []
    for spec in specs:
        frames: list[tuple[dict[str, object], np.ndarray]] = []
        try:
            for index in spec.indices:
                frames.append(dataset.load_sample(index))
        except (FrameDecodeError, OSError) as exc:
            failures.append({"category": spec.category, "name": spec.name, "error": str(exc)})
            continue
        decoded.append((spec, frames))
    return decoded


def _write_strip_sheet(
    rows: list[tuple[StripSpec, list[tuple[dict[str, object], np.ndarray]]]],
    path: Path,
) -> None:
    cell_width, cell_height = 448, 252
    canvas = np.zeros((len(rows) * cell_height, 3 * cell_width, 3), dtype=np.uint8)
    for row, (spec, frames) in enumerate(rows):
        center = frames[1][0]
        for column, (item, frame_rgb) in enumerate(frames):
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            frame = cv2.resize(frame, (cell_width, cell_height), interpolation=cv2.INTER_AREA)
            metadata = item["metadata"]
            lines = [
                f"{spec.name} ride={spec.ride_id}",
                f"frame={metadata['front_frame_id']} ts={metadata['front_timestamp']:.3f}",
                (
                    f"CENTER linear={center['metadata']['linear']:.3f} "
                    f"angular={center['metadata']['angular']:.3f} action={center['action_class']}"
                    if column == 1
                    else "context frame"
                ),
            ]
            cv2.rectangle(frame, (0, 0), (cell_width, 82), (0, 0, 0), -1)
            for line_index, line in enumerate(lines):
                cv2.putText(
                    frame,
                    line,
                    (7, 22 + 25 * line_index),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            if column == 1:
                cv2.rectangle(frame, (2, 2), (cell_width - 3, cell_height - 3), (0, 255, 255), 3)
            canvas[
                row * cell_height : (row + 1) * cell_height,
                column * cell_width : (column + 1) * cell_width,
            ] = frame
    if not cv2.imwrite(str(path), canvas):
        raise OSError(f"failed to write strip visualization: {path}")


def _repeated_access_is_deterministic(dataset: FrodoBotsActionDataset, index: int) -> bool:
    first = dataset[index]["image"]
    second = dataset[index]["image"]
    return isinstance(first, torch.Tensor) and torch.equal(first, second)


def _audit_monotonicity(
    samples: tuple[ManifestSample, ...],
    grouped: dict[str, list[int]],
) -> dict[str, object]:
    frame_id_violations: list[dict[str, object]] = []
    timestamp_violations: list[dict[str, object]] = []
    section_boundaries: list[dict[str, object]] = []
    for ride_id, indices in grouped.items():
        for previous_index, current_index in zip(indices, indices[1:]):
            previous = samples[previous_index]
            current = samples[current_index]
            if current.front_frame_id <= previous.front_frame_id:
                frame_id_violations.append(
                    {"ride_id": ride_id, "previous": previous.front_frame_id, "current": current.front_frame_id}
                )
            if current.front_timestamp <= previous.front_timestamp:
                timestamp_violations.append(
                    {"ride_id": ride_id, "previous": previous.front_timestamp, "current": current.front_timestamp}
                )
            if current.timeline_section_id != previous.timeline_section_id:
                section_boundaries.append(
                    {
                        "ride_id": ride_id,
                        "previous_frame_id": previous.front_frame_id,
                        "current_frame_id": current.front_frame_id,
                        "previous_timestamp": previous.front_timestamp,
                        "current_timestamp": current.front_timestamp,
                        "monotonic": (
                            current.front_frame_id > previous.front_frame_id
                            and current.front_timestamp > previous.front_timestamp
                        ),
                    }
                )
    return {
        "frame_id_violation_count": len(frame_id_violations),
        "frame_id_violations": frame_id_violations[:20],
        "timestamp_violation_count": len(timestamp_violations),
        "timestamp_violations": timestamp_violations[:20],
        "hls_section_boundary_count": len(section_boundaries),
        "hls_section_boundaries": section_boundaries,
    }


def _action_label_mismatches(samples: tuple[ManifestSample, ...]) -> list[dict[str, object]]:
    mismatches: list[dict[str, object]] = []
    for index, sample in enumerate(samples):
        expected = classify_action(sample.linear, sample.angular)
        if expected != sample.action_class:
            mismatches.append(
                {
                    "manifest_index": index,
                    "ride_id": sample.ride_id,
                    "expected": expected,
                    "actual": sample.action_class,
                }
            )
    return mismatches


def _transform_audit(frame_rgb: np.ndarray, batch_images: torch.Tensor) -> dict[str, object]:
    height, width = frame_rgb.shape[:2]
    return {
        "observed_input_shape_hwc": [height, width, 3],
        "output_shape_chw": [3, 224, 224],
        "operation": "cv2.resize directly from the full RGB frame to 224x224 using INTER_AREA",
        "geometry": "aspect ratio is distorted; no center crop and no letterbox are used",
        "horizontal_scale": 224 / width,
        "vertical_scale": 224 / height,
        "lateral_information": (
            "the full left and right field of view is retained, but horizontal geometry is compressed "
            "relative to vertical geometry"
        ),
        "normalization": "RGB values are scaled to [0,1], then normalized by ImageNet mean/std",
        "batch_is_finite": bool(torch.isfinite(batch_images).all().item()),
    }


if __name__ == "__main__":
    raise SystemExit(main())
