#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.training.datasets.frodobots_2k_manifest import DatasetFormatError, build_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deterministic FrodoBots-2K front-frame/control manifest without decoding video."
    )
    parser.add_argument("--dataset-root", required=True, help="Path to the immutable output_rides_0 directory.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Artifact directory outside output_rides_0. Defaults to its sibling manifests directory.",
    )
    parser.add_argument(
        "--max-rides",
        type=int,
        default=None,
        help="Process only the first N lexicographically sorted rides for a bounded dry run.",
    )
    parser.add_argument("--control-tolerance-ms", type=float, default=100.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else dataset_root.parent / "manifests" / "frodobots_2k_phase1"
    )
    try:
        report = build_manifest(
            dataset_root=dataset_root,
            output_dir=output_dir,
            max_rides=args.max_rides,
            tolerance_ms=args.control_tolerance_ms,
        )
    except (DatasetFormatError, OSError, ValueError) as exc:
        print(f"manifest build failed: {exc}", file=sys.stderr)
        return 1

    summary = {
        "processed_ride_count": report["processed_ride_count"],
        "total_front_frame_count": report["total_front_frame_count"],
        "valid_sample_count": report["valid_sample_count"],
        "rejected_sample_count": report["rejected_sample_count"],
        "rejection_reasons": report["rejection_reasons"],
        "action_class_distribution": report["action_class_distribution"],
        "control_delta_ms": report["control_delta_ms"],
        "manifest_path": report["manifest_path"],
        "manifest_sha256": report["manifest_sha256"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
