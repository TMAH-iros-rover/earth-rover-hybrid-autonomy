#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_BUNDLE="${SOURCE_BUNDLE:-$HOME/datasets/review_bundles/manual_candidates_v2_selected}"
CVAT_EXPORT="${CVAT_EXPORT:-$SOURCE_BUNDLE/traversability_manual_v2_33_cvat_export.zip}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/datasets/review_bundles/traversability_manual_v2_33_imported}"
LABEL_CONTRACT="${LABEL_CONTRACT:-$ROOT_DIR/configs/traversability_dataset_v1.yaml}"
EXPECTED_COUNT=33

if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON=python
else
    echo "ERROR: Python 3 was not found." >&2
    exit 1
fi

for path in \
    "$SOURCE_BUNDLE/selected_candidates.csv" \
    "$SOURCE_BUNDLE/selection.txt" \
    "$CVAT_EXPORT" \
    "$LABEL_CONTRACT"; do
    if [[ ! -f "$path" ]]; then
        echo "ERROR: Required input is missing: $path" >&2
        exit 1
    fi
done
if [[ ! -d "$SOURCE_BUNDLE/images" ]]; then
    echo "ERROR: Source images directory is missing: $SOURCE_BUNDLE/images" >&2
    exit 1
fi
if [[ -e "$OUTPUT_DIR" ]]; then
    echo "ERROR: Output path already exists; it will not be overwritten: $OUTPUT_DIR" >&2
    exit 1
fi
case "$(realpath -m "$OUTPUT_DIR")" in
    "$ROOT_DIR"/*)
        echo "ERROR: Imported dataset artifacts must remain outside Git: $OUTPUT_DIR" >&2
        exit 1
        ;;
esac

tree_fingerprint() {
    "$PYTHON" - "$1" <<'PY'
import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
digest = hashlib.sha256()
for path in sorted(item for item in root.rglob("*") if item.is_file()):
    stat = path.stat()
    digest.update(f"{path.relative_to(root).as_posix()}|{stat.st_size}|{stat.st_mtime_ns}|".encode())
    digest.update(hashlib.sha256(path.read_bytes()).digest())
print(digest.hexdigest())
PY
}

cd "$ROOT_DIR"
before_git_status="$(git status --porcelain)"
before_source="$(tree_fingerprint "$SOURCE_BUNDLE")"
before_zip="$(sha256sum "$CVAT_EXPORT" | awk '{print $1}')"

echo "[1/4] Running focused manual-v2 import tests"
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -m pytest -p no:cacheprovider -q \
    tests/test_import_manual_traversability_v2.py \
    tests/test_select_manual_candidates_v2.py \
    tests/test_traversability_annotation.py

echo "[2/4] Importing and validating 33 SegmentationClass masks"
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" training/import_manual_traversability_v2.py \
    --source-bundle "$SOURCE_BUNDLE" \
    --cvat-export "$CVAT_EXPORT" \
    --output-dir "$OUTPUT_DIR" \
    --label-contract "$LABEL_CONTRACT" \
    --expected-count "$EXPECTED_COUNT"

echo "[3/4] Verifying output count, correspondence, and byte preservation"
"$PYTHON" - "$SOURCE_BUNDLE" "$OUTPUT_DIR" <<'PY'
import csv
import hashlib
import json
import sys
from pathlib import Path

source = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).resolve()
source_rows = list(csv.DictReader((source / "selected_candidates.csv").open(newline="", encoding="utf-8")))
metadata = list(csv.DictReader((output / "metadata.csv").open(newline="", encoding="utf-8")))
report = json.loads((output / "validation_report.json").read_text(encoding="utf-8"))
source_ids = [row["candidate_id"] for row in source_rows]
assert report["valid"] is True
assert len(source_rows) == len(metadata) == 33
assert [row["sample_id"] for row in metadata] == source_ids
assert report["image_count"] == report["segmentation_class_mask_count"] == 33
assert report["semantic_mask_source"] == "SegmentationClass"
assert report["segmentation_object_used"] is False
assert report["ignore_class_id"] == 0
assert report["allowed_class_ids"] == [0, 1, 2, 3]
assert len(list((output / "images").glob("*.jpg"))) == 33
assert len(list((output / "masks").glob("*.png"))) == 33
assert len(list((output / "overlays").glob("*.jpg"))) == 33
assert len(report["contact_sheets"]) == 2
for source_row, output_row in zip(source_rows, metadata):
    assert source_row["candidate_id"] == output_row["sample_id"]
    assert source_row["ride_id"] == output_row["ride_id"]
    assert source_row["timestamp_sec"] == output_row["timestamp_sec"]
    source_image = source / source_row["image_path"]
    output_image = output / output_row["image_path"]
    assert hashlib.sha256(source_image.read_bytes()).digest() == hashlib.sha256(output_image.read_bytes()).digest()
for relative in report["contact_sheets"]:
    assert (output / relative).is_file() and (output / relative).stat().st_size > 0
print("Images/masks/metadata: 33/33/33")
print(f"Contact sheets: {len(report['contact_sheets'])}")
print(f"Warnings: {report['warnings']}")
print("Manual-v2 correspondence and byte-preservation gate: PASS")
PY

echo "[4/4] Verifying immutable inputs and Git exclusion"
after_source="$(tree_fingerprint "$SOURCE_BUNDLE")"
after_zip="$(sha256sum "$CVAT_EXPORT" | awk '{print $1}')"
after_git_status="$(git status --porcelain)"
if [[ "$before_source" != "$after_source" || "$before_zip" != "$after_zip" ]]; then
    echo "ERROR: The source selection bundle or CVAT ZIP changed." >&2
    exit 1
fi
if [[ "$before_git_status" != "$after_git_status" ]]; then
    echo "ERROR: Git working state changed during Dell execution." >&2
    git status --short >&2
    exit 1
fi
if [[ -n "$(git ls-files -- ':(glob)**/traversability_manual_v2_33_imported/**')" ]]; then
    echo "ERROR: Generated imported masks appear in git ls-files." >&2
    exit 1
fi
du -sh "$OUTPUT_DIR"
echo "Source selection bundle and CVAT ZIP unchanged: PASS"
echo "Git exclusion: PASS"
echo "Existing approved v1 merge: NOT PERFORMED"
echo "Training/live rover operations: NOT PERFORMED"
echo "Imported manual-v2 bundle: $OUTPUT_DIR"
