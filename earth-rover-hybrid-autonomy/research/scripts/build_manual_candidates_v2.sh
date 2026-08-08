#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT_1="${DATASET_ROOT_1:-$HOME/datasets/output_rides_1}"
DATASET_ROOT_2="${DATASET_ROOT_2:-$HOME/datasets/output_rides_2}"
APPROVED_METADATA="${APPROVED_METADATA:-$HOME/datasets/generated/traversability_dataset_v1/approved_120_v1/metadata.csv}"
HARD_BUNDLE="${HARD_BUNDLE:-$HOME/datasets/generated/traversability_dataset_v2/hard_examples_annotation_v1}"
HARD_SHORTFALL_REPORT="${HARD_SHORTFALL_REPORT:-$HOME/datasets/generated/traversability_dataset_v2/hard_examples_annotation_v1_candidate_shortfall.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/datasets/review_bundles/manual_candidates_v2}"
DRY_RUN_DIR="${DRY_RUN_DIR:-${OUTPUT_DIR}_dry_run}"
SAMPLE_COUNT="${SAMPLE_COUNT:-200}"
DRY_RUN_COUNT="${DRY_RUN_COUNT:-12}"
EDGE_MARGIN_SECONDS="${EDGE_MARGIN_SECONDS:-10}"
MINIMUM_PAIR_SEPARATION_SECONDS="${MINIMUM_PAIR_SEPARATION_SECONDS:-20}"
HASH_DISTANCE_THRESHOLD="${HASH_DISTANCE_THRESHOLD:-5}"
SEED="${SEED:-20260721}"

if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON=python
else
    echo "ERROR: Python 3 was not found." >&2
    exit 1
fi

for path in "$DATASET_ROOT_1" "$DATASET_ROOT_2"; do
    if [[ ! -d "$path" ]]; then
        echo "ERROR: Dataset root is missing: $path" >&2
        exit 1
    fi
done
if [[ ! -f "$APPROVED_METADATA" ]]; then
    echo "ERROR: Approved v1 metadata is missing: $APPROVED_METADATA" >&2
    exit 1
fi
for path in "$DRY_RUN_DIR" "$OUTPUT_DIR"; do
    if [[ -e "$path" ]]; then
        echo "ERROR: Output path already exists; preserve it and choose a new path: $path" >&2
        exit 1
    fi
    case "$(realpath -m "$path")" in
        "$ROOT_DIR"/*)
            echo "ERROR: Generated review artifacts must remain outside Git: $path" >&2
            exit 1
            ;;
    esac
done

exclude_args=(--exclude-metadata "$APPROVED_METADATA")
if [[ -f "$HARD_BUNDLE/metadata.csv" ]]; then
    exclude_args+=(--exclude-metadata "$HARD_BUNDLE/metadata.csv")
fi
if [[ -f "$HARD_BUNDLE/selection_report.json" ]]; then
    exclude_args+=(--exclude-report "$HARD_BUNDLE/selection_report.json")
fi
if [[ -f "$HARD_SHORTFALL_REPORT" ]]; then
    exclude_args+=(--exclude-report "$HARD_SHORTFALL_REPORT")
fi
for report in "$(dirname "$HARD_BUNDLE")"/hard_examples_annotation_v1_candidate_shortfall*.json; do
    if [[ -f "$report" && "$report" != "$HARD_SHORTFALL_REPORT" ]]; then
        exclude_args+=(--exclude-report "$report")
    fi
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

echo "[1/6] Running focused manual-candidate tests"
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -m pytest -p no:cacheprovider -q \
    tests/test_manual_candidate_sampling.py \
    tests/test_traversability_expansion.py \
    tests/test_frodobots_2k_dataset.py

echo "[2/6] Recording immutable raw-dataset fingerprints"
before_dataset_1="$(tree_fingerprint "$DATASET_ROOT_1")"
before_dataset_2="$(tree_fingerprint "$DATASET_ROOT_2")"

echo "[3/6] Running the 12-image extraction dry run"
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" training/build_manual_candidates_v2.py \
    --dataset-root "$DATASET_ROOT_1" \
    --dataset-root "$DATASET_ROOT_2" \
    "${exclude_args[@]}" \
    --output-dir "$DRY_RUN_DIR" \
    --sample-count "$DRY_RUN_COUNT" \
    --maximum-per-ride 2 \
    --minimum-pair-separation-seconds "$MINIMUM_PAIR_SEPARATION_SECONDS" \
    --edge-margin-seconds "$EDGE_MARGIN_SECONDS" \
    --hash-distance-threshold "$HASH_DISTANCE_THRESHOLD" \
    --seed "$SEED" \
    --dry-run

"$PYTHON" - "$DRY_RUN_DIR" "$DRY_RUN_COUNT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
expected = int(sys.argv[2])
report = json.loads((root / "selection_report.json").read_text(encoding="utf-8"))
assert report["success"] is True
assert report["dry_run"] is True
assert report["selected_sample_count"] == expected
assert report["camera_uid"] == "1000"
assert report["rear_camera_excluded"] is True
assert report["raw_images_have_overlays"] is False
print(f"Dry-run images: {report['selected_sample_count']}")
print("Manual candidate dry-run gate: PASS")
PY

echo "[4/6] Building the deterministic approximately 200-image review pool"
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" training/build_manual_candidates_v2.py \
    --dataset-root "$DATASET_ROOT_1" \
    --dataset-root "$DATASET_ROOT_2" \
    "${exclude_args[@]}" \
    --output-dir "$OUTPUT_DIR" \
    --sample-count "$SAMPLE_COUNT" \
    --maximum-per-ride 2 \
    --minimum-pair-separation-seconds "$MINIMUM_PAIR_SEPARATION_SECONDS" \
    --edge-margin-seconds "$EDGE_MARGIN_SECONDS" \
    --hash-distance-threshold "$HASH_DISTANCE_THRESHOLD" \
    --seed "$SEED"

echo "[5/6] Validating camera, count, provenance, diversity, and output integrity"
"$PYTHON" - "$OUTPUT_DIR" "$SAMPLE_COUNT" "$DRY_RUN_DIR" <<'PY'
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

root = Path(sys.argv[1]).resolve()
expected = int(sys.argv[2])
dry_root = Path(sys.argv[3]).resolve()
rows = list(csv.DictReader((root / "candidates.csv").open(newline="", encoding="utf-8")))
dry_rows = list(csv.DictReader((dry_root / "candidates.csv").open(newline="", encoding="utf-8")))
report = json.loads((root / "selection_report.json").read_text(encoding="utf-8"))
required = [
    "candidate_id", "dataset", "ride_id", "camera_uid", "timestamp_sec",
    "playlist_path", "image_path",
]
assert list(rows[0]) == required
assert report["success"] is True
assert len(rows) == expected == report["selected_sample_count"]
assert len({row["candidate_id"] for row in rows}) == expected
assert len({row["image_path"] for row in rows}) == expected
assert rows[:len(dry_rows)] == dry_rows
assert all(row["camera_uid"] == "1000" for row in rows)
assert all("uid_s_1000" in row["playlist_path"] for row in rows)
assert all("uid_s_1001" not in row["playlist_path"] for row in rows)
assert all(row["playlist_path"].endswith("video.m3u8") for row in rows)
assert all((root / row["image_path"]).is_file() for row in rows)
ride_counts = Counter(row["ride_id"] for row in rows)
assert max(ride_counts.values()) <= 2
timestamps = defaultdict(list)
for row in rows:
    timestamps[row["ride_id"]].append(float(row["timestamp_sec"]))
for values in timestamps.values():
    if len(values) == 2:
        assert abs(values[0] - values[1]) >= 20.0 - 1e-9
assert report["minimum_selected_hash_distance"] > report["hash_distance_threshold"]
if report["minimum_selected_to_existing_hash_distance"] is not None:
    assert report["minimum_selected_to_existing_hash_distance"] > report["hash_distance_threshold"]
assert report["contact_sheet_count"] == math.ceil(expected / 25)
assert len(list((root / "contact_sheets").glob("contact_sheet_*.jpg"))) == math.ceil(expected / 25)
assert not ({row["ride_id"] for row in rows} & set(report["existing_excluded_ride_ids"]))
assert report["raw_images_have_overlays"] is False
assert report["model_training_performed"] is False
assert report["live_rover_commands_sent"] is False
print(f"Selected images: {len(rows)}")
print(f"Used rides: {report['selected_ride_count']}")
print(f"Excluded existing rides in source: {report['existing_excluded_ride_count']}")
print(f"Minimum perceptual hash distance: {report['minimum_selected_hash_distance']}")
print(f"Contact sheets: {report['contact_sheet_count']}")
print(f"Dataset distribution: {report['dataset_distribution']}")
print("Manual candidate full gate: PASS")
PY

echo "[6/6] Verifying immutable inputs and Git exclusion"
after_dataset_1="$(tree_fingerprint "$DATASET_ROOT_1")"
after_dataset_2="$(tree_fingerprint "$DATASET_ROOT_2")"
after_git_status="$(git status --porcelain)"
if [[ "$before_dataset_1" != "$after_dataset_1" || "$before_dataset_2" != "$after_dataset_2" ]]; then
    echo "ERROR: A raw dataset changed during candidate extraction." >&2
    exit 1
fi
if [[ "$before_git_status" != "$after_git_status" ]]; then
    echo "ERROR: Git working state changed during Dell execution." >&2
    git status --short >&2
    exit 1
fi
if [[ -n "$(git ls-files -- ':(glob)**/manual_candidates_v2/**')" ]]; then
    echo "ERROR: Generated manual candidates appear in git ls-files." >&2
    exit 1
fi
du -sh "$DRY_RUN_DIR" "$OUTPUT_DIR"
echo "Raw datasets unchanged: PASS"
echo "Git exclusion: PASS"
echo "Training/inference/planner/live rover operations: NOT PERFORMED"
echo "Manual candidate bundle: $OUTPUT_DIR"
