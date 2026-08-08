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
    select_annotation_candidates,
    validate_annotation_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the 20-frame traversability_dataset_v1 annotation pilot.")
    parser.add_argument("--source-pseudo-bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-contract", default=str(ROOT / "research/configs/traversability_dataset_v1.yaml"))
    parser.add_argument("--sample-count", type=int, default=20)
    parser.add_argument("--minimum-separation-seconds", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260717)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.sample_count != 20:
        raise SystemExit("the current human-review gate permits exactly 20 pilot samples")
    source = Path(args.source_pseudo_bundle).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    if output == source or source in output.parents:
        raise SystemExit("output-dir must remain outside the immutable source pseudo-label bundle")
    candidates = load_annotation_candidates(source)
    selected = select_annotation_candidates(
        candidates,
        requested_count=args.sample_count,
        minimum_separation_seconds=args.minimum_separation_seconds,
        seed=args.seed,
    )
    if len(selected) != args.sample_count:
        raise SystemExit(f"could select only {len(selected)} of {args.sample_count} requested samples")
    report = build_annotation_bundle(
        selected,
        output,
        args.label_contract,
        source,
        args.seed,
        args.minimum_separation_seconds,
    )
    validation = validate_annotation_dataset(output, require_masks=False)
    if not validation["valid"]:
        raise SystemExit(json.dumps(validation, indent=2, sort_keys=True))
    print(json.dumps({"build": report, "pre_annotation_validation": validation}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
