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
    import_cvat_masks,
    validate_annotation_dataset,
    write_annotation_review_outputs,
)
from research.training.traversability_review import write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Import a CVAT Segmentation Mask export into traversability_dataset_v1.")
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--cvat-export", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--expected-count",
        type=int,
        help="Optional exact metadata and SegmentationClass mask count safety gate.",
    )
    args = parser.parse_args()
    bundle = Path(args.bundle).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve() if args.output_dir else bundle / "reviewed_import"
    report = import_cvat_masks(
        bundle,
        args.cvat_export,
        output,
        expected_count=args.expected_count,
    )
    validation = validate_annotation_dataset(bundle, require_masks=True, masks_dir=output / "masks")
    report["validation"] = validation
    write_json(output / "import_report.json", report)
    write_json(output / "validation_report.json", validation)
    if validation["valid"]:
        write_annotation_review_outputs(bundle, output, validation)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if validation["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
