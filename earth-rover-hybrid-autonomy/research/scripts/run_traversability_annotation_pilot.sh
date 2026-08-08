#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_BUNDLE="${SOURCE_BUNDLE:-$HOME/datasets/review_bundles/traversability_pilot_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/datasets/generated/traversability_dataset_v1/pilot_20}"
SAMPLE_COUNT="${SAMPLE_COUNT:-20}"
MINIMUM_SEPARATION_SECONDS="${MINIMUM_SEPARATION_SECONDS:-5}"
SEED="${SEED:-20260717}"

if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON=python
else
    echo "ERROR: Python 3 was not found." >&2
    exit 1
fi

if [[ ! -f "$SOURCE_BUNDLE/review.csv" || ! -f "$SOURCE_BUNDLE/bundle_report.json" ]]; then
    echo "ERROR: The completed pseudo-label pilot bundle was not found: $SOURCE_BUNDLE" >&2
    exit 1
fi
if [[ -d "$OUTPUT_DIR" && -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "ERROR: Output directory is not empty; preserve it and choose a new OUTPUT_DIR." >&2
    exit 1
fi

source_fingerprint() {
    "$PYTHON" - "$SOURCE_BUNDLE" <<'PY'
import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
digest = hashlib.sha256()
for path in sorted(item for item in root.rglob("*") if item.is_file()):
    stat = path.stat()
    digest.update(f"{path.relative_to(root).as_posix()}|{stat.st_size}|{stat.st_mtime_ns}\n".encode())
print(digest.hexdigest())
PY
}

cd "$ROOT_DIR"

echo "[1/4] Running focused annotation-pipeline tests"
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -m pytest \
    -p no:cacheprovider -q \
    tests/test_traversability_annotation.py \
    tests/test_traversability_review.py

echo "[2/4] Recording source pseudo-label bundle fingerprint"
before_fingerprint="$(source_fingerprint)"

echo "[3/4] Building the 20-frame CVAT annotation pilot"
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" training/build_traversability_annotation_pilot.py \
    --source-pseudo-bundle "$SOURCE_BUNDLE" \
    --output-dir "$OUTPUT_DIR" \
    --sample-count "$SAMPLE_COUNT" \
    --minimum-separation-seconds "$MINIMUM_SEPARATION_SECONDS" \
    --seed "$SEED"

echo "[4/4] Checking bundle integrity and source immutability"
after_fingerprint="$(source_fingerprint)"
if [[ "$before_fingerprint" != "$after_fingerprint" ]]; then
    echo "ERROR: Source pseudo-label bundle metadata changed." >&2
    exit 1
fi

"$PYTHON" - "$OUTPUT_DIR" <<'PY'
import csv
import json
import sys
import zipfile
from pathlib import Path

root = Path(sys.argv[1]).resolve()
report = json.loads((root / "build_report.json").read_text(encoding="utf-8"))
rows = list(csv.DictReader((root / "metadata.csv").open(newline="", encoding="utf-8")))
assert report["pipeline_status"] == "HUMAN_ANNOTATION_REQUIRED"
assert report["selected_sample_count"] == 20
assert report["processed_ride_count"] >= 2
assert report["pseudo_label_inference_performed"] is False
assert report["raw_dataset_accessed"] is False
assert report["model_training_performed"] is False
assert report["live_rover_commands_sent"] is False
assert len(rows) == 20
assert len({row["sample_id"] for row in rows}) == 20
assert len(list((root / "images").glob("*"))) == 20
assert len(list((root / "initial_masks").glob("*.png"))) == 20
assert not list((root / "masks").glob("*.png"))
with zipfile.ZipFile(root / "cvat_seed_annotations.zip") as archive:
    assert len([name for name in archive.namelist() if name.startswith("SegmentationClass/")]) == 20
    assert len([name for name in archive.namelist() if name.startswith("SegmentationObject/")]) == 20
for name in ("README.md", "label_contract.yaml", "cvat_labelmap.txt", "metadata.csv", "contact_sheet.jpg"):
    assert (root / name).is_file()
print(f"Rides: {report['processed_ride_count']}")
print(f"Frames: {report['selected_sample_count']}")
print(f"Scene categories: {report['scene_category_distribution']}")
print(f"Missing target categories: {report['target_categories_not_found']}")
print(f"Bundle: {root}")
print("Source pseudo-label bundle unchanged: PASS")
print("Training/live-rover gate: ACTIVE")
PY

du -sh "$OUTPUT_DIR"
echo "Traversability annotation pilot: PASS (20 images; annotation required)"
