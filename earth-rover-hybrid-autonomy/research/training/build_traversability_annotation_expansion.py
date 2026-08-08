#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.training.traversability_annotation import (
    build_annotation_bundle,
    load_annotation_candidates,
    validate_annotation_dataset,
)
from research.training.traversability_expansion import (
    load_existing_annotations,
    select_expansion_candidates,
)
from research.training.traversability_review import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the additional 100-image traversability annotation bundle.")
    parser.add_argument("--source-pseudo-bundle", required=True)
    parser.add_argument("--existing-pilot", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-contract", default=str(ROOT / "research/configs/traversability_dataset_v1.yaml"))
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--minimum-separation-seconds", type=float, default=10.0)
    parser.add_argument("--maximum-per-ride", type=int, default=5)
    parser.add_argument("--hash-distance-threshold", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260718)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.sample_count != 100:
        raise SystemExit("the approved expansion stage requires exactly 100 additional samples")
    source = Path(args.source_pseudo_bundle).expanduser().resolve()
    existing_pilot = Path(args.existing_pilot).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    for path in (source, existing_pilot):
        if output == path or path in output.parents:
            raise SystemExit("output-dir must not modify a source bundle")

    candidates = load_annotation_candidates(source)
    existing = load_existing_annotations(existing_pilot)
    selected, selection_report = select_expansion_candidates(
        candidates,
        existing,
        requested_count=args.sample_count,
        minimum_separation_seconds=args.minimum_separation_seconds,
        maximum_per_ride=args.maximum_per_ride,
        hash_distance_threshold=args.hash_distance_threshold,
        seed=args.seed,
    )
    if len(selected) != args.sample_count:
        raise SystemExit(
            f"could select only {len(selected)} of {args.sample_count} samples; "
            f"inspect exclusion statistics before changing thresholds"
        )
    if selection_report["selected_existing_exact_overlap_count"] != 0:
        raise SystemExit("selected samples overlap the approved 20-image pilot")

    build_report = build_annotation_bundle(
        selected,
        output,
        args.label_contract,
        source,
        args.seed,
        args.minimum_separation_seconds,
        sample_id_prefix="trav_v1_add_",
    )
    build_report.update(
        {
            "dataset_name": "traversability_dataset_v1_annotation_100_v1",
            "existing_pilot": str(existing_pilot),
            "existing_pilot_sample_count": len(existing.sample_ids),
            "bounded_candidate_pseudo_inference_performed": True,
            "bounded_candidate_count": len(candidates),
            "pseudo_label_inference_performed": True,
            "pseudo_label_inference_scope": "bounded candidate pool only",
            "full_dataset_inference_performed": False,
            "pseudo_labels_are_ground_truth": False,
            "raw_dataset_accessed": True,
            "model_training_performed": False,
            "live_rover_commands_sent": False,
        }
    )
    validation = validate_annotation_dataset(output, require_masks=False)
    if not validation["valid"]:
        raise SystemExit(json.dumps(validation, indent=2, sort_keys=True))
    write_json(output / "build_report.json", build_report)
    write_json(output / "selection_report.json", selection_report)
    print(
        json.dumps(
            {
                "build": build_report,
                "selection": selection_report,
                "pre_annotation_validation": validation,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
