#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.training.traversability_annotation import validate_annotation_dataset
from research.training.traversability_review import write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate completed traversability_dataset_v1 image-mask pairs.")
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--masks-dir")
    parser.add_argument("--report-path")
    args = parser.parse_args()
    root = Path(args.bundle).expanduser().resolve()
    masks_dir = Path(args.masks_dir).expanduser().resolve() if args.masks_dir else root / "reviewed_import/masks"
    report_path = Path(args.report_path).expanduser().resolve() if args.report_path else root / "reviewed_import/validation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = validate_annotation_dataset(root, require_masks=True, masks_dir=masks_dir)
    write_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
