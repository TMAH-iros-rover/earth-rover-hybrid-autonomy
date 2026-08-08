#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.training.traversability_review import validate_review_bundle, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a traversability review bundle without training.")
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--verified-output")
    args = parser.parse_args()
    bundle = Path(args.bundle).expanduser().resolve()
    report = validate_review_bundle(bundle, args.verified_output)
    write_json(bundle / "review_validation_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
