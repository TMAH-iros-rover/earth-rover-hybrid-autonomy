#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUNDLE_ROOT="${BUNDLE_ROOT:-$HOME/datasets/generated/traversability_dataset_v1/annotation_100_v1}"
CVAT_EXPORT="${CVAT_EXPORT:-$BUNDLE_ROOT/traversability_annotation_100_reviewed.zip}"
OUTPUT_DIR="${OUTPUT_DIR:-$BUNDLE_ROOT/reviewed_import}"
APPROVED_PILOT="${APPROVED_PILOT:-$HOME/datasets/generated/traversability_dataset_v1/pilot_20}"
DATASET_ROOT="${DATASET_ROOT:-$HOME/datasets/output_rides_0}"
EXPECTED_COUNT=100

if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON=python
else
    echo "ERROR: Python 3 was not found." >&2
    exit 1
fi

for path in \
    "$BUNDLE_ROOT/metadata.csv" \
    "$BUNDLE_ROOT/label_contract.yaml" \
    "$APPROVED_PILOT/metadata.csv" \
    "$CVAT_EXPORT"; do
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

tree_fingerprint() {
    "$PYTHON" - "$1" <<'PY'
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

echo "[1/6] Inspecting CLI interfaces and running focused tests"
"$PYTHON" training/import_cvat_traversability_masks.py --help >/dev/null
"$PYTHON" training/validate_traversability_dataset_v1.py --help >/dev/null
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -m pytest \
    -p no:cacheprovider -q tests/test_traversability_annotation.py

echo "[2/6] Recording immutable input fingerprints"
before_dataset="$(tree_fingerprint "$DATASET_ROOT")"
before_pilot="$(tree_fingerprint "$APPROVED_PILOT")"
before_zip="$(sha256sum "$CVAT_EXPORT" | awk '{print $1}')"

echo "[3/6] Importing 100 SegmentationClass masks by label name"
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" training/import_cvat_traversability_masks.py \
    --bundle "$BUNDLE_ROOT" \
    --cvat-export "$CVAT_EXPORT" \
    --output-dir "$OUTPUT_DIR" \
    --expected-count "$EXPECTED_COUNT"

echo "[4/6] Running the strict validator independently"
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" training/validate_traversability_dataset_v1.py \
    --bundle "$BUNDLE_ROOT" \
    --masks-dir "$OUTPUT_DIR/masks" \
    --report-path "$OUTPUT_DIR/validation_report.json"

echo "[5/6] Checking the approved pilot for provenance overlap"
"$PYTHON" - "$BUNDLE_ROOT" "$APPROVED_PILOT" "$OUTPUT_DIR" <<'PY'
import csv
import json
import sys
from pathlib import Path

bundle = Path(sys.argv[1]).resolve()
pilot = Path(sys.argv[2]).resolve()
output = Path(sys.argv[3]).resolve()

def rows(root: Path) -> list[dict[str, str]]:
    return list(csv.DictReader((root / "metadata.csv").open(newline="", encoding="utf-8")))

new_rows = rows(bundle)
pilot_rows = rows(pilot)
new_sample_ids = {row["sample_id"] for row in new_rows}
pilot_sample_ids = {row["sample_id"] for row in pilot_rows}
new_sources = {(row["ride_id"], row["manifest_index"]) for row in new_rows}
pilot_sources = {(row["ride_id"], row["manifest_index"]) for row in pilot_rows}
new_frames = {
    (row["ride_id"], row["frame_id"], round(float(row["timestamp"]) * 1000))
    for row in new_rows
}
pilot_frames = {
    (row["ride_id"], row["frame_id"], round(float(row["timestamp"]) * 1000))
    for row in pilot_rows
}
report = {
    "new_sample_count": len(new_rows),
    "approved_pilot_sample_count": len(pilot_rows),
    "sample_id_overlap": sorted(new_sample_ids & pilot_sample_ids),
    "source_manifest_overlap": sorted(new_sources & pilot_sources),
    "source_frame_timestamp_overlap": sorted(new_frames & pilot_frames),
}
(output / "pilot_overlap_report.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
if any(report[key] for key in (
    "sample_id_overlap", "source_manifest_overlap", "source_frame_timestamp_overlap"
)):
    raise SystemExit(json.dumps(report, indent=2, sort_keys=True))
print(json.dumps(report, indent=2, sort_keys=True))
PY

echo "[6/6] Verifying counts, immutability, reports, and Git exclusion"
after_dataset="$(tree_fingerprint "$DATASET_ROOT")"
after_pilot="$(tree_fingerprint "$APPROVED_PILOT")"
after_zip="$(sha256sum "$CVAT_EXPORT" | awk '{print $1}')"
after_git_status="$(git status --porcelain)"
if [[ "$before_dataset" != "$after_dataset" ]]; then
    echo "ERROR: Raw FrodoBots dataset changed." >&2
    exit 1
fi
if [[ "$before_pilot" != "$after_pilot" ]]; then
    echo "ERROR: Approved 20-image pilot changed." >&2
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
if [[ -n "$(git ls-files -- ':(glob)**/annotation_100_v1/reviewed_import/**')" ]]; then
    echo "ERROR: Generated review output appears in git ls-files." >&2
    exit 1
fi

"$PYTHON" - "$BUNDLE_ROOT" "$OUTPUT_DIR" "$CVAT_EXPORT" "$before_dataset" "$before_pilot" "$before_zip" <<'PY'
import csv
import json
import sys
import zipfile
from pathlib import Path

bundle = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).resolve()
export = Path(sys.argv[3]).resolve()
rows = list(csv.DictReader((bundle / "metadata.csv").open(newline="", encoding="utf-8")))
report = json.loads((output / "validation_report.json").read_text(encoding="utf-8"))
import_report = json.loads((output / "import_report.json").read_text(encoding="utf-8"))
with zipfile.ZipFile(export) as archive:
    assert archive.testzip() is None
    segmentation_class_count = sum(
        1 for name in archive.namelist()
        if name.endswith(".png") and "SegmentationClass/" in name
    )
assert len(rows) == 100
assert len({row["sample_id"] for row in rows}) == 100
assert len([path for path in (bundle / "images").iterdir() if path.is_file()]) == 100
assert segmentation_class_count == 100
assert report["valid"] is True, report["errors"]
assert report["sample_count"] == 100
assert report["validated_mask_count"] == 100
assert import_report["imported_mask_count"] == 100
assert import_report["semantic_mask_source"] == "SegmentationClass"
assert import_report["segmentation_object_used"] is False
assert set(report["class_pixel_counts"]) == {"IGNORE", "ON_ROAD", "OFF_ROAD", "OBSTACLE"}
assert all(report["class_pixel_counts"][name] > 0 for name in ("ON_ROAD", "OFF_ROAD", "OBSTACLE"))
assert len(list((output / "masks").glob("*.png"))) == 100
assert len(list((output / "mask_visualizations").glob("*.png"))) == 100
assert len(list((output / "overlays").glob("*.jpg"))) == 100
for name in (
    "overlay_contact_sheet.jpg", "review.html", "per_image_statistics.csv",
    "class_statistics.json", "pilot_overlap_report.json",
):
    assert (output / name).is_file() and (output / name).stat().st_size > 0
verification = {
    "validator_pass": True,
    "raw_dataset_unchanged": True,
    "approved_pilot_unchanged": True,
    "original_cvat_zip_unchanged": True,
    "raw_dataset_fingerprint": sys.argv[4],
    "approved_pilot_fingerprint": sys.argv[5],
    "cvat_zip_sha256": sys.argv[6],
    "git_worktree_unchanged": True,
    "generated_artifacts_outside_git_worktree": True,
    "fine_tuning_performed": False,
    "full_dataset_inference_performed": False,
    "live_rover_commands_sent": False,
}
(output / "execution_verification.json").write_text(
    json.dumps(verification, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(f"Masks: {report['validated_mask_count']}")
print(f"Class pixels: {report['class_pixel_counts']}")
print(f"Class fractions: {report['class_pixel_fractions']}")
print(f"All IGNORE: {report['all_ignore_sample_ids']}")
print(f"Single class: {report['single_class_sample_ids']}")
for name, sample_ids in report["missing_class_sample_ids"].items():
    print(f"Without {name}: count={len(sample_ids)} samples={sample_ids}")
print(f"Warnings: {report['warnings']}")
print(f"Normalized masks: {output / 'masks'}")
print(f"Validator report: {output / 'validation_report.json'}")
print(f"Overlay contact sheet: {output / 'overlay_contact_sheet.jpg'}")
print(f"Review HTML: {output / 'review.html'}")
print("Approved pilot overlap: PASS (0)")
print("Raw dataset unchanged: PASS")
print("Approved pilot unchanged: PASS")
print("Original CVAT ZIP unchanged: PASS")
print("Git exclusion: PASS")
PY

du -sh "$OUTPUT_DIR"
echo "Traversability 100-image reviewed-mask import: PASS (human overlay approval still required)"
