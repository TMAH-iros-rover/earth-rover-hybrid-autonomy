#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-$HOME/datasets/output_rides_0}"
MANIFEST_PATH="${MANIFEST_PATH:-$HOME/datasets/manifests/frodobots_2k_phase2/full_dataset/manifest.csv}"
EXISTING_PILOT="${EXISTING_PILOT:-$HOME/datasets/generated/traversability_dataset_v1/pilot_20}"
CANDIDATE_BUNDLE="${CANDIDATE_BUNDLE:-$HOME/datasets/generated/traversability_dataset_v1/annotation_100_v1_candidates_240}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/datasets/generated/traversability_dataset_v1/annotation_100_v1}"
CANDIDATE_COUNT="${CANDIDATE_COUNT:-240}"
MAXIMUM_CANDIDATE_RIDES="${MAXIMUM_CANDIDATE_RIDES:-40}"
MINIMUM_SEPARATION_SECONDS="${MINIMUM_SEPARATION_SECONDS:-10}"
MAXIMUM_PER_RIDE="${MAXIMUM_PER_RIDE:-5}"
HASH_DISTANCE_THRESHOLD="${HASH_DISTANCE_THRESHOLD:-5}"
SEED="${SEED:-20260718}"
export HF_HOME="${HF_HOME:-$HOME/datasets/generated/huggingface}"

if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON=python
else
    echo "ERROR: Python 3 was not found." >&2
    exit 1
fi

for path in "$DATASET_ROOT" "$EXISTING_PILOT"; do
    if [[ ! -d "$path" ]]; then
        echo "ERROR: Required directory is missing: $path" >&2
        exit 1
    fi
done
for path in "$MANIFEST_PATH" "$EXISTING_PILOT/metadata.csv"; do
    if [[ ! -f "$path" ]]; then
        echo "ERROR: Required file is missing: $path" >&2
        exit 1
    fi
done
for path in "$CANDIDATE_BUNDLE" "$OUTPUT_DIR"; do
    if [[ -d "$path" && -n "$(find "$path" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        echo "ERROR: Generated output is not empty; preserve it and choose a new path: $path" >&2
        exit 1
    fi
    case "$(realpath -m "$path")" in
        "$ROOT_DIR"/*)
            echo "ERROR: Dataset artifacts must remain outside the Git worktree: $path" >&2
            exit 1
            ;;
    esac
done

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

echo "[1/7] Running focused sampling and annotation-bundle tests"
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -m pytest \
    -p no:cacheprovider -q \
    tests/test_traversability_expansion.py \
    tests/test_traversability_annotation.py \
    tests/test_traversability_review.py

echo "[2/7] Checking CUDA and recording immutable-source fingerprints"
"$PYTHON" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
PY
before_dataset="$(tree_fingerprint "$DATASET_ROOT")"
before_pilot="$(tree_fingerprint "$EXISTING_PILOT")"

echo "[3/7] Building a bounded 240-frame semantic candidate pool"
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" training/build_traversability_review_bundle.py \
    --dataset-root "$DATASET_ROOT" \
    --manifest "$MANIFEST_PATH" \
    --output-dir "$CANDIDATE_BUNDLE" \
    --sample-count "$CANDIDATE_COUNT" \
    --max-rides "$MAXIMUM_CANDIDATE_RIDES" \
    --minimum-separation-seconds "$MINIMUM_SEPARATION_SECONDS" \
    --seed "$SEED" \
    --require-cuda

echo "[4/7] Selecting 100 new images and building the CVAT bundle"
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" training/build_traversability_annotation_expansion.py \
    --source-pseudo-bundle "$CANDIDATE_BUNDLE" \
    --existing-pilot "$EXISTING_PILOT" \
    --output-dir "$OUTPUT_DIR" \
    --sample-count 100 \
    --minimum-separation-seconds "$MINIMUM_SEPARATION_SECONDS" \
    --maximum-per-ride "$MAXIMUM_PER_RIDE" \
    --hash-distance-threshold "$HASH_DISTANCE_THRESHOLD" \
    --seed "$SEED"

echo "[5/7] Validating bundle counts, provenance, and CVAT seed archive"
"$PYTHON" - "$OUTPUT_DIR" "$MINIMUM_SEPARATION_SECONDS" "$HASH_DISTANCE_THRESHOLD" "$MAXIMUM_PER_RIDE" <<'PY'
import csv
import json
import sys
import zipfile
from pathlib import Path

root = Path(sys.argv[1]).resolve()
rows = list(csv.DictReader((root / "metadata.csv").open(newline="", encoding="utf-8")))
build = json.loads((root / "build_report.json").read_text(encoding="utf-8"))
selection = json.loads((root / "selection_report.json").read_text(encoding="utf-8"))
assert len(rows) == 100
assert len({row["sample_id"] for row in rows}) == 100
assert all(row["sample_id"].startswith("trav_v1_add_") for row in rows)
assert len(list((root / "images").glob("*"))) == 100
assert len(list((root / "metadata").glob("*.json"))) == 100
assert len(list((root / "initial_masks").glob("*.png"))) == 100
assert not list((root / "masks").glob("*.png"))
assert build["selected_sample_count"] == 100
assert build["model_training_performed"] is False
assert build["full_dataset_inference_performed"] is False
assert build["live_rover_commands_sent"] is False
assert selection["selected_existing_exact_overlap_count"] == 0
assert selection["minimum_selected_same_ride_time_delta_seconds"] >= float(sys.argv[2])
assert selection["minimum_selected_pair_hash_distance"] > int(sys.argv[3])
assert selection["minimum_selected_to_existing_hash_distance"] > int(sys.argv[3])
assert max(selection["ride_distribution"].values()) <= int(sys.argv[4])
with zipfile.ZipFile(root / "cvat_seed_annotations.zip") as archive:
    assert len([name for name in archive.namelist() if name.startswith("SegmentationClass/")]) == 100
    assert len([name for name in archive.namelist() if name.startswith("SegmentationObject/")]) == 100
for name in ("README.md", "contact_sheet.jpg", "metadata.csv", "cvat_labelmap.txt", "selection_report.json"):
    assert (root / name).is_file() and (root / name).stat().st_size > 0
print(f"Selected images: {len(rows)}")
print(f"Rides: {selection['selected_ride_count']}")
print(f"Ride distribution: {selection['ride_distribution']}")
print(f"Category distribution: {selection['scene_category_distribution']}")
print(f"Initial exclusions: {selection['initial_exclusion_counts']}")
print(f"Visual hash check: minimum selected pair={selection['minimum_selected_pair_hash_distance']}")
print(f"Existing pilot overlap: {selection['selected_existing_exact_overlap_count']}")
PY

echo "[6/7] Verifying raw data, approved pilot, and Git state"
after_dataset="$(tree_fingerprint "$DATASET_ROOT")"
after_pilot="$(tree_fingerprint "$EXISTING_PILOT")"
after_git_status="$(git status --porcelain)"
if [[ "$before_dataset" != "$after_dataset" ]]; then
    echo "ERROR: Raw FrodoBots dataset metadata changed." >&2
    exit 1
fi
if [[ "$before_pilot" != "$after_pilot" ]]; then
    echo "ERROR: Approved 20-image pilot changed." >&2
    exit 1
fi
if [[ "$before_git_status" != "$after_git_status" ]]; then
    echo "ERROR: Repository working state changed." >&2
    git status --short >&2
    exit 1
fi
if [[ -n "$(git ls-files -- ':(glob)**/annotation_100_v1/**')" ]]; then
    echo "ERROR: Generated annotation artifacts appear in git ls-files." >&2
    exit 1
fi

echo "[7/7] Reporting generated artifact paths"
du -sh "$CANDIDATE_BUNDLE" "$OUTPUT_DIR"
echo "Raw dataset unchanged: PASS"
echo "Approved pilot unchanged: PASS"
echo "Git exclusion: PASS"
echo "Model training/full-dataset inference/live-rover commands: NOT PERFORMED"
echo "Annotation 100 bundle: $OUTPUT_DIR"
