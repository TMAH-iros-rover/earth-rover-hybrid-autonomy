#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUNDLE_ROOT="${BUNDLE_ROOT:-$HOME/datasets/generated/traversability_dataset_v1/pilot_20}"
CVAT_EXPORT="${CVAT_EXPORT:-$BUNDLE_ROOT/traversability_pilot_20_reviewed.zip}"
OUTPUT_DIR="${OUTPUT_DIR:-$BUNDLE_ROOT/reviewed_import}"
DATASET_ROOT="${DATASET_ROOT:-$HOME/datasets/output_rides_0}"

if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON=python
else
    echo "ERROR: Python 3 was not found." >&2
    exit 1
fi

for path in "$BUNDLE_ROOT/metadata.csv" "$BUNDLE_ROOT/label_contract.yaml" "$CVAT_EXPORT"; do
    if [[ ! -f "$path" ]]; then
        echo "ERROR: Required input is missing: $path" >&2
        exit 1
    fi
done
if [[ ! -d "$DATASET_ROOT" ]]; then
    echo "ERROR: Raw dataset root is missing: $DATASET_ROOT" >&2
    exit 1
fi
if [[ -d "$OUTPUT_DIR" && -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "ERROR: Output directory is not empty; preserve it and choose a new OUTPUT_DIR." >&2
    exit 1
fi
case "$(realpath -m "$OUTPUT_DIR")" in
    "$ROOT_DIR"/*)
        echo "ERROR: Reviewed dataset artifacts must remain outside the Git worktree." >&2
        exit 1
        ;;
esac

dataset_fingerprint() {
    "$PYTHON" - "$DATASET_ROOT" <<'PY'
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
before_git_status="$(git status --porcelain)"

echo "[1/6] Inspecting exact CLI interfaces"
"$PYTHON" training/import_cvat_traversability_masks.py --help >/dev/null
"$PYTHON" training/validate_traversability_dataset_v1.py --help >/dev/null

echo "[2/6] Running focused CVAT import and validator tests"
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -m pytest \
    -p no:cacheprovider -q \
    tests/test_traversability_annotation.py

echo "[3/6] Recording raw dataset and CVAT ZIP fingerprints"
before_dataset="$(dataset_fingerprint)"
before_zip="$(sha256sum "$CVAT_EXPORT" | awk '{print $1}')"

echo "[4/6] Importing 20 SegmentationClass masks by label name"
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" training/import_cvat_traversability_masks.py \
    --bundle "$BUNDLE_ROOT" \
    --cvat-export "$CVAT_EXPORT" \
    --output-dir "$OUTPUT_DIR" \
    --expected-count 20

echo "[5/6] Running the strict validator independently"
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" training/validate_traversability_dataset_v1.py \
    --bundle "$BUNDLE_ROOT" \
    --masks-dir "$OUTPUT_DIR/masks" \
    --report-path "$OUTPUT_DIR/validation_report.json"

echo "[6/6] Verifying immutability, reports, and Git exclusion"
after_dataset="$(dataset_fingerprint)"
after_zip="$(sha256sum "$CVAT_EXPORT" | awk '{print $1}')"
after_git_status="$(git status --porcelain)"
if [[ "$before_dataset" != "$after_dataset" ]]; then
    echo "ERROR: Raw FrodoBots dataset metadata changed." >&2
    exit 1
fi
if [[ "$before_zip" != "$after_zip" ]]; then
    echo "ERROR: Original CVAT ZIP changed." >&2
    exit 1
fi
if [[ "$before_git_status" != "$after_git_status" ]]; then
    echo "ERROR: Repository working state changed during verification." >&2
    git status --short >&2
    exit 1
fi
tracked_output="$(git ls-files -- ':(glob)**/reviewed_import/**')"
if [[ -n "$tracked_output" ]]; then
    echo "ERROR: Generated output appears in git ls-files: $tracked_output" >&2
    exit 1
fi

"$PYTHON" - "$OUTPUT_DIR" "$before_dataset" "$before_zip" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
report = json.loads((root / "validation_report.json").read_text(encoding="utf-8"))
import_report = json.loads((root / "import_report.json").read_text(encoding="utf-8"))
assert report["valid"] is True, report["errors"]
assert report["sample_count"] == 20
assert report["validated_mask_count"] == 20
assert import_report["imported_mask_count"] == 20
assert import_report["semantic_mask_source"] == "SegmentationClass"
assert import_report["segmentation_object_used"] is False
assert import_report["background_merged_into_ignore"] is True
assert report["class_pixel_counts"]["ON_ROAD"] > 0
assert report["class_pixel_counts"]["OFF_ROAD"] > 0
assert len(list((root / "masks").glob("*.png"))) == 20
assert len(list((root / "mask_visualizations").glob("*.png"))) == 20
for name in ("overlay_contact_sheet.jpg", "review.html", "per_image_statistics.csv"):
    assert (root / name).is_file() and (root / name).stat().st_size > 0
verification = {
    "validator_pass": True,
    "raw_dataset_unchanged": True,
    "original_cvat_zip_unchanged": True,
    "raw_dataset_fingerprint": sys.argv[2],
    "cvat_zip_sha256": sys.argv[3],
    "git_worktree_unchanged": True,
    "generated_artifacts_outside_git_worktree": True,
    "fine_tuning_performed": False,
    "full_dataset_inference_performed": False,
    "live_rover_commands_sent": False,
}
(root / "execution_verification.json").write_text(
    json.dumps(verification, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(f"Masks: {report['validated_mask_count']}")
print(f"Class pixels: {report['class_pixel_counts']}")
print(f"Class fractions: {report['class_pixel_fractions']}")
print(f"All IGNORE: {report['all_ignore_sample_ids']}")
print(f"Single class: {report['single_class_sample_ids']}")
print(f"Warnings: {report['warnings']}")
print(f"Normalized masks: {root / 'masks'}")
print(f"Validator report: {root / 'validation_report.json'}")
print(f"Overlay contact sheet: {root / 'overlay_contact_sheet.jpg'}")
print("Raw dataset unchanged: PASS")
print("Original CVAT ZIP unchanged: PASS")
print("Git exclusion: PASS")
PY

du -sh "$OUTPUT_DIR"
echo "Traversability reviewed-mask import: PASS (human overlay approval still required)"
