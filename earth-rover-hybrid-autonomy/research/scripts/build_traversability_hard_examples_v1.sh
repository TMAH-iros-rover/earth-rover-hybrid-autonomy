#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-$HOME/datasets/output_rides_0}"
TEMPORAL_BUNDLE="${TEMPORAL_BUNDLE:-$HOME/datasets/review_bundles/traversability_temporal_v1}"
APPROVED_DATASET="${APPROVED_DATASET:-$HOME/datasets/generated/traversability_dataset_v1/approved_120_v1}"
CHECKPOINT="${CHECKPOINT:-$HOME/datasets/experiments/traversability_segformer_b0_v1/full_training/segformer_b0_best.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/datasets/generated/traversability_dataset_v2/hard_examples_annotation_v1}"
TARGET_COUNT=24
SEED="${SEED:-20260721}"
export HF_HOME="${HF_HOME:-$HOME/datasets/generated/huggingface}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON=python
else
    echo "ERROR: Python 3 was not found." >&2
    exit 1
fi

for path in "$DATASET_ROOT" "$TEMPORAL_BUNDLE" "$APPROVED_DATASET"; do
    if [[ ! -d "$path" ]]; then
        echo "ERROR: Required Dell directory is missing: $path" >&2
        exit 1
    fi
done
for path in "$TEMPORAL_BUNDLE/per_frame_statistics.csv" "$TEMPORAL_BUNDLE/temporal_inference_report.json" "$APPROVED_DATASET/metadata.csv" "$CHECKPOINT"; do
    if [[ ! -f "$path" ]]; then
        echo "ERROR: Required input is missing: $path" >&2
        exit 1
    fi
done
if [[ -d "$OUTPUT_DIR" && -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "ERROR: Output is not empty; preserve it and choose a new OUTPUT_DIR: $OUTPUT_DIR" >&2
    exit 1
fi
case "$(realpath -m "$OUTPUT_DIR")" in
    "$ROOT_DIR"/*)
        echo "ERROR: Generated annotation artifacts must remain outside Git: $OUTPUT_DIR" >&2
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

echo "[1/6] Running focused hard-example, annotation, and temporal tests"
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -m pytest -p no:cacheprovider -q tests/test_traversability_hard_examples.py tests/test_traversability_annotation.py tests/test_traversability_temporal_inference.py

echo "[2/6] Checking CUDA and recording immutable fingerprints"
"$PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
PY
before_raw="$(tree_fingerprint "$DATASET_ROOT")"
before_temporal="$(tree_fingerprint "$TEMPORAL_BUNDLE")"
before_approved="$(tree_fingerprint "$APPROVED_DATASET")"
before_checkpoint="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"

echo "[3/6] Mining v1 temporal errors and building the targeted 24-image CVAT bundle"
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" training/build_traversability_hard_examples.py --dataset-root "$DATASET_ROOT" --temporal-bundle "$TEMPORAL_BUNDLE" --approved-dataset "$APPROVED_DATASET" --checkpoint "$CHECKPOINT" --output-dir "$OUTPUT_DIR" --seed "$SEED" --require-cuda

echo "[4/6] Validating count, provenance, split isolation, and v1 seed masks"
"$PYTHON" - "$OUTPUT_DIR" "$APPROVED_DATASET" "$TARGET_COUNT" <<'PY'
import csv
import json
import sys
import zipfile
from pathlib import Path
import cv2
import numpy as np

root = Path(sys.argv[1]).resolve()
approved = Path(sys.argv[2]).resolve()
target_count = int(sys.argv[3])
rows = list(csv.DictReader((root / "metadata.csv").open(newline="", encoding="utf-8")))
selection = json.loads((root / "selection_report.json").read_text(encoding="utf-8"))
build = json.loads((root / "build_report.json").read_text(encoding="utf-8"))
actual = selection["selected_sample_count"]
assert 0 < actual <= target_count
assert len(rows) == actual
assert len({row["sample_id"] for row in rows}) == actual
assert len({(row["ride_id"], row["frame_id"], row["timestamp"]) for row in rows}) == actual
assert len(list((root / "images").glob("*"))) == actual
assert len(list((root / "initial_masks").glob("*.png"))) == actual
assert len(list((root / "metadata").glob("*.json"))) == actual
assert len(list((root / "context").glob("*.jpg"))) == actual
assert len(list((root / "confidence").glob("*.png"))) == actual
assert build["seed_mask_contract"] == "v1_source"
assert build["additional_training_performed"] is False
assert build["planner_or_live_rover_integration_performed"] is False
assert selection["ready_for_annotation_bundle"] is True
assert selection["approved_ride_overlap"] == []
assert selection["ride_leakage"] == []
assert selection["approved_exact_overlap_count"] == 0
assert sum(selection["split_sample_counts"].values()) == actual
assert selection["minimum_selected_hash_distance"] > selection["hash_distance_threshold"]
assert selection["minimum_same_ride_time_delta_seconds"] >= selection["minimum_separation_seconds"]
assert selection["category_targets"] == {
    "CURB_HARD_NEGATIVE": 12,
    "TRUE_OFF_ROAD": 6,
    "PAVED_HARD_CASE": 6,
}
assert all(
    selection["category_distribution"][name] + selection["category_shortfalls"][name] == target
    for name, target in selection["category_targets"].items()
)
assert build["category_target_fulfilled"] == selection["category_target_fulfilled"]
train_rides = set(selection["split_rides"]["hard_train_candidates"])
validation_rides = set(selection["split_rides"]["hard_validation_candidates"])
assert train_rides and validation_rides and not train_rides & validation_rides
approved_keys = {
    (row["ride_id"], row["frame_id"], round(float(row["timestamp"]) * 1000))
    for row in csv.DictReader((approved / "metadata.csv").open(newline="", encoding="utf-8"))
}
assert not any((row["ride_id"], row["frame_id"], round(float(row["timestamp"]) * 1000)) in approved_keys for row in rows)
for row in rows:
    mask = cv2.imread(str(root / "initial_masks" / f"{row['sample_id']}.png"), cv2.IMREAD_UNCHANGED)
    assert mask is not None and mask.ndim == 2 and mask.dtype == np.uint8
    assert set(int(value) for value in np.unique(mask)).issubset({0, 1, 2, 3})
    metadata = json.loads((root / "metadata" / f"{row['sample_id']}.json").read_text(encoding="utf-8"))
    assert len(metadata["context_frames"]) == 5
    assert sum(item["is_annotation_center"] for item in metadata["context_frames"]) == 1
    assert all(
        all(
            key in item
            for key in (
                "mean_confidence",
                "prediction_on_road_ratio",
                "prediction_off_road_ratio",
                "prediction_obstacle_ratio",
            )
        )
        for item in metadata["context_frames"]
    )
with zipfile.ZipFile(root / "cvat_seed_annotations.zip") as archive:
    assert len([name for name in archive.namelist() if name.startswith("SegmentationClass/")]) == actual
for name in ("contact_sheet.jpg", "review.html", "README.md", "cvat_labelmap.txt", "selection_report.json"):
    assert (root / name).is_file() and (root / name).stat().st_size > 0
print(f"Selected frames: {len(rows)}")
print(f"Candidates: prefilter={selection['temporal_prefilter_count']} inferred={selection['inferred_candidate_count']}")
print(f"Categories: {selection['category_distribution']}")
print(f"Category shortfalls: {selection['category_shortfalls']}")
print(f"Full 24-image target: {selection['category_target_fulfilled']}")
print(f"Rides: {selection['ride_distribution']}")
print(f"Splits: {selection['split_rides']}")
print(f"Split samples: {selection['split_sample_counts']}")
print(f"Split categories: {selection['split_category_distribution']}")
print(f"Duplicate suppression: hash={selection['minimum_selected_hash_distance']} time={selection['minimum_same_ride_time_delta_seconds']}")
print("Hard-example annotation bundle gate: PASS")
PY

echo "[5/6] Verifying immutable sources and Git exclusion"
after_raw="$(tree_fingerprint "$DATASET_ROOT")"
after_temporal="$(tree_fingerprint "$TEMPORAL_BUNDLE")"
after_approved="$(tree_fingerprint "$APPROVED_DATASET")"
after_checkpoint="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
after_git_status="$(git status --porcelain)"
if [[ "$before_raw" != "$after_raw" || "$before_temporal" != "$after_temporal" || "$before_approved" != "$after_approved" || "$before_checkpoint" != "$after_checkpoint" ]]; then
    echo "ERROR: An immutable dataset, temporal artifact, or checkpoint changed." >&2
    exit 1
fi
if [[ "$before_git_status" != "$after_git_status" ]]; then
    echo "ERROR: Git worktree changed during Dell execution." >&2
    git status --short >&2
    exit 1
fi
if [[ -n "$(git ls-files -- ':(glob)**/hard_examples_annotation_v1/**')" ]]; then
    echo "ERROR: Generated hard-example artifacts appear in git ls-files." >&2
    exit 1
fi

echo "[6/6] Reporting the portable annotation bundle"
du -sh "$OUTPUT_DIR"
echo "Raw dataset unchanged: PASS"
echo "Temporal bundle, approved_120_v1, and v1 checkpoint unchanged: PASS"
echo "Git exclusion: PASS"
echo "Retraining/smoothing/planner/live rover integration: NOT PERFORMED"
echo "Hard-example bundle: $OUTPUT_DIR"
