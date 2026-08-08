#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.training.datasets.frodobots_2k_dataset import (
    FrameDecodeError,
    HlsFrameDecoder,
    ManifestSample,
)
from research.training.manual_candidate_sampling import (
    CAMERA_UID,
    CANDIDATE_FIELDS,
    deterministic_ride_order,
    discover_front_rides,
    frame_options,
    is_hash_distinct,
    load_existing_candidates,
    minimum_hash_distance,
    write_numbered_contact_sheets,
)
from research.training.traversability_expansion import difference_hash, hamming_distance
from research.training.traversability_review import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a broad manual front-camera review pool.")
    parser.add_argument("--dataset-root", action="append", required=True)
    parser.add_argument("--exclude-metadata", action="append", default=[])
    parser.add_argument("--exclude-report", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-count", type=int, default=200)
    parser.add_argument("--maximum-per-ride", type=int, default=2)
    parser.add_argument("--minimum-pair-separation-seconds", type=float, default=20.0)
    parser.add_argument("--edge-margin-seconds", type=float, default=10.0)
    parser.add_argument("--hash-distance-threshold", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.sample_count <= 0:
        raise SystemExit("sample-count must be positive")
    if args.maximum_per_ride not in {1, 2}:
        raise SystemExit("maximum-per-ride must be 1 or 2")
    if args.minimum_pair_separation_seconds < 20.0:
        raise SystemExit("minimum pair separation must be at least 20 seconds")
    if args.edge_margin_seconds <= 0:
        raise SystemExit("edge margin must be positive")

    roots = [Path(value).expanduser().resolve() for value in args.dataset_root]
    output = Path(args.output_dir).expanduser().resolve()
    if any(output == root or root in output.parents for root in roots):
        raise SystemExit("output directory must remain outside immutable dataset roots")
    if output.exists():
        raise SystemExit(f"output path already exists; choose a new path: {output}")
    temporary = output.parent / f".{output.name}.tmp"
    if temporary.exists():
        raise SystemExit(f"temporary output already exists: {temporary}")

    existing = load_existing_candidates(args.exclude_metadata, args.exclude_report)
    rides, discovery = discover_front_rides(roots)
    discovered_ride_ids = {ride.ride_id for ride in rides}
    excluded_in_sources = discovered_ride_ids & existing.ride_ids
    available = [ride for ride in rides if ride.ride_id not in existing.ride_ids]
    ordered = deterministic_ride_order(available, args.seed)
    options = {
        (ride.dataset, ride.ride_id): frame_options(
            ride,
            args.edge_margin_seconds,
            args.minimum_pair_separation_seconds,
            args.seed,
        )
        for ride in ordered
    }

    (temporary / "images").mkdir(parents=True)
    decoders = {str(root): HlsFrameDecoder(root) for root in roots}
    selected_rows: list[dict[str, str]] = []
    selected_hashes: list[int] = []
    ride_counts: Counter[str] = Counter()
    dataset_counts: Counter[str] = Counter()
    rejections: Counter[str] = Counter()

    for round_index in range(args.maximum_per_ride):
        for ride in ordered:
            if len(selected_rows) >= args.sample_count:
                break
            if ride_counts[ride.ride_id] >= args.maximum_per_ride:
                continue
            ride_options = options[(ride.dataset, ride.ride_id)]
            if round_index >= len(ride_options):
                rejections["no_frame_for_selection_round"] += 1
                continue
            frame = ride_options[round_index]
            segment = ride.timeline.find_segment(frame.timestamp)
            if segment is None:
                rejections["frame_outside_hls_coverage"] += 1
                continue
            sample = ManifestSample(
                ride_id=ride.ride_id,
                front_playlist_ref=ride.timeline.playlist_reference,
                front_segment_ref=segment.reference,
                front_frame_id=frame.frame_id,
                front_timestamp=frame.timestamp,
                matched_control_timestamp=frame.timestamp,
                control_delta_ms=0.0,
                linear=0.0,
                angular=0.0,
                action_class="STOP",
                timeline_section_id=segment.section_id,
            )
            try:
                image_rgb = decoders[str(ride.dataset_root)].decode(sample)
            except FrameDecodeError:
                rejections["decode_failure"] += 1
                continue
            candidate_id = f"manual_v2_{len(selected_rows) + 1:04d}"
            image_relative = f"images/{candidate_id}.jpg"
            image_path = temporary / image_relative
            if not cv2.imwrite(str(image_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)):
                raise OSError(f"cannot write candidate image: {image_path}")
            image_hash = difference_hash(image_path)
            if not is_hash_distinct(
                image_hash,
                [*existing.image_hashes, *selected_hashes],
                args.hash_distance_threshold,
            ):
                image_path.unlink()
                rejections["perceptual_duplicate"] += 1
                continue
            selected_hashes.append(image_hash)
            ride_counts[ride.ride_id] += 1
            dataset_counts[ride.dataset] += 1
            selected_rows.append(
                {
                    "candidate_id": candidate_id,
                    "dataset": ride.dataset,
                    "ride_id": ride.ride_id,
                    "camera_uid": CAMERA_UID,
                    "timestamp_sec": f"{frame.timestamp:.6f}",
                    "playlist_path": ride.timeline.playlist_reference,
                    "image_path": image_relative,
                }
            )
        if len(selected_rows) >= args.sample_count:
            break

    with (temporary / "candidates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_FIELDS)
        writer.writeheader()
        writer.writerows(selected_rows)
    sheets = write_numbered_contact_sheets(selected_rows, temporary)
    minimum_existing_distance = min(
        (
            hamming_distance(value, existing_hash)
            for value in selected_hashes
            for existing_hash in existing.image_hashes
        ),
        default=None,
    )
    report = {
        "success": len(selected_rows) == args.sample_count,
        "dry_run": args.dry_run,
        "requested_sample_count": args.sample_count,
        "selected_sample_count": len(selected_rows),
        "selected_ride_count": len(ride_counts),
        "ride_distribution": dict(sorted(ride_counts.items())),
        "dataset_distribution": dict(sorted(dataset_counts.items())),
        "dataset_roots": [str(root) for root in roots],
        "camera_uid": CAMERA_UID,
        "playlist_pattern": "*uid_s_1000*video.m3u8",
        "rear_camera_excluded": True,
        "existing_excluded_ride_count": len(excluded_in_sources),
        "existing_excluded_ride_ids": sorted(excluded_in_sources),
        "existing_reference_ride_count": len(existing.ride_ids),
        "discovery": discovery,
        "available_unused_ride_count": len(available),
        "selection_rejections": dict(sorted(rejections.items())),
        "maximum_per_ride": args.maximum_per_ride,
        "minimum_pair_separation_seconds": args.minimum_pair_separation_seconds,
        "edge_margin_seconds": args.edge_margin_seconds,
        "hash_distance_threshold": args.hash_distance_threshold,
        "minimum_selected_hash_distance": minimum_hash_distance(selected_hashes),
        "minimum_selected_to_existing_hash_distance": minimum_existing_distance,
        "contact_sheet_count": len(sheets),
        "contact_sheets": sheets,
        "seed": args.seed,
        "raw_images_have_overlays": False,
        "model_inference_performed": False,
        "model_training_performed": False,
        "live_rover_commands_sent": False,
    }
    write_json(temporary / "selection_report.json", report)
    (temporary / "README.md").write_text(
        "# Manual Traversability Candidates v2\n\n"
        "Review the numbered contact sheets, then use `candidates.csv` to locate untouched "
        "front-camera candidate images. Images contain no text or overlays. This is a manual "
        "selection pool, not verified annotation data.\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
