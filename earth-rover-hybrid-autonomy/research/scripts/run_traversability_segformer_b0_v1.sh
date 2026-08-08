#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-$HOME/datasets/output_rides_0}"
PILOT_BUNDLE="${PILOT_BUNDLE:-$HOME/datasets/generated/traversability_dataset_v1/pilot_20}"
PILOT_REVIEWED="${PILOT_REVIEWED:-$PILOT_BUNDLE/reviewed_import}"
EXPANSION_BUNDLE="${EXPANSION_BUNDLE:-$HOME/datasets/generated/traversability_dataset_v1/annotation_100_v1}"
EXPANSION_REVIEWED="${EXPANSION_REVIEWED:-$EXPANSION_BUNDLE/reviewed_import}"
APPROVED_DATASET="${APPROVED_DATASET:-$HOME/datasets/generated/traversability_dataset_v1/approved_120_v1}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$HOME/datasets/experiments/traversability_segformer_b0_v1}"
OVERFIT_DIR="$EXPERIMENT_ROOT/overfit_sanity"
TRAINING_DIR="$EXPERIMENT_ROOT/full_training"
CONFIG_PATH="$ROOT_DIR/configs/traversability_segformer_b0_v1.yaml"
SEED="${SEED:-20260718}"
REUSE_APPROVED_DATASET="${REUSE_APPROVED_DATASET:-false}"
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

for path in \
    "$DATASET_ROOT" \
    "$PILOT_BUNDLE" \
    "$PILOT_REVIEWED/masks" \
    "$EXPANSION_BUNDLE" \
    "$EXPANSION_REVIEWED/masks"; do
    if [[ ! -d "$path" ]]; then
        echo "ERROR: Required Dell dataset directory is missing: $path" >&2
        exit 1
    fi
done
for path in \
    "$PILOT_REVIEWED/validation_report.json" \
    "$EXPANSION_REVIEWED/validation_report.json" \
    "$CONFIG_PATH"; do
    if [[ ! -f "$path" ]]; then
        echo "ERROR: Required validated input is missing: $path" >&2
        exit 1
    fi
done
if [[ "$REUSE_APPROVED_DATASET" != "true" && "$REUSE_APPROVED_DATASET" != "false" ]]; then
    echo "ERROR: REUSE_APPROVED_DATASET must be true or false." >&2
    exit 1
fi
if [[ "$REUSE_APPROVED_DATASET" == "true" ]]; then
    for path in "$APPROVED_DATASET/manifest.csv" "$APPROVED_DATASET/metadata.csv" "$APPROVED_DATASET/merge_report.json"; do
        if [[ ! -f "$path" ]]; then
            echo "ERROR: Reusable approved dataset is incomplete: $path" >&2
            exit 1
        fi
    done
elif [[ -d "$APPROVED_DATASET" && -n "$(find "$APPROVED_DATASET" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "ERROR: Output is not empty; preserve it or set REUSE_APPROVED_DATASET=true: $APPROVED_DATASET" >&2
    exit 1
fi
for path in "$APPROVED_DATASET" "$OVERFIT_DIR" "$TRAINING_DIR"; do
    if [[ "$path" != "$APPROVED_DATASET" && -d "$path" && -n "$(find "$path" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        echo "ERROR: Output is not empty; preserve it and choose a new path: $path" >&2
        exit 1
    fi
    case "$(realpath -m "$path")" in
        "$ROOT_DIR"/*)
            echo "ERROR: Dataset and experiment artifacts must remain outside Git: $path" >&2
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

echo "[1/8] Running focused dataset, loader, split, metric, and annotation tests"
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -m pytest \
    -p no:cacheprovider -q \
    tests/test_traversability_segmentation.py \
    tests/test_traversability_annotation.py

echo "[2/8] Checking Dell CUDA/runtime and recording immutable fingerprints"
"$PYTHON" - <<'PY'
import importlib.metadata
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
for name in ("torch", "torchvision", "transformers", "safetensors", "Pillow"):
    print(f"{name}: {importlib.metadata.version(name)}")
PY
before_raw="$(tree_fingerprint "$DATASET_ROOT")"
before_pilot="$(tree_fingerprint "$PILOT_BUNDLE")"
before_expansion="$(tree_fingerprint "$EXPANSION_BUNDLE")"

echo "[3/8] Building immutable approved_120_v1 and deterministic ride split"
if [[ "$REUSE_APPROVED_DATASET" == "true" ]]; then
    echo "Reusing previously validated approved dataset: $APPROVED_DATASET"
else
    PYTHONDONTWRITEBYTECODE=1 "$PYTHON" training/build_approved_traversability_dataset_v1.py \
        --pilot-bundle "$PILOT_BUNDLE" \
        --pilot-reviewed "$PILOT_REVIEWED" \
        --expansion-bundle "$EXPANSION_BUNDLE" \
        --expansion-reviewed "$EXPANSION_REVIEWED" \
        --output-dir "$APPROVED_DATASET" \
        --seed "$SEED"
fi

echo "[4/8] Verifying the 120-pair dataset and split gate"
"$PYTHON" - "$APPROVED_DATASET" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
rows = list(csv.DictReader((root / "manifest.csv").open(newline="", encoding="utf-8")))
metadata_rows = list(csv.DictReader((root / "metadata.csv").open(newline="", encoding="utf-8")))
merge = json.loads((root / "merge_report.json").read_text(encoding="utf-8"))
split = json.loads((root / "split_report.json").read_text(encoding="utf-8"))
assert len(rows) == 120
assert metadata_rows == rows
assert len({row["sample_id"] for row in rows}) == 120
assert len(list((root / "images").iterdir())) == 120
assert len(list((root / "masks").glob("*.png"))) == 120
assert merge["valid"] is True
assert merge["cvat_seed_masks_included"] is False
assert merge["pseudo_labels_included"] is False
rides = {name: set(values) for name, values in split["rides"].items()}
assert not rides["train"] & rides["validation"]
assert not rides["train"] & rides["test"]
assert not rides["validation"] & rides["test"]
for name in ("train", "validation", "test"):
    assert split["statistics"][name]["sample_count"] > 0
    for class_name in ("ON_ROAD", "OFF_ROAD", "OBSTACLE"):
        assert split["statistics"][name]["class_pixel_counts"][class_name] > 0
print(f"Split rides: {split['rides']}")
print(f"Split statistics: {split['statistics']}")
print("Approved dataset and ride leakage gate: PASS")
PY
before_approved="$(tree_fingerprint "$APPROVED_DATASET")"

echo "[5/8] Running the 6-image CUDA overfit sanity gate"
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" training/train_traversability_segformer.py \
    --manifest "$APPROVED_DATASET/manifest.csv" \
    --config "$CONFIG_PATH" \
    --output-dir "$OVERFIT_DIR" \
    --mode overfit \
    --require-cuda

"$PYTHON" - "$OVERFIT_DIR/overfit_report.json" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert report["success"] is True
assert report["loss_decreased"] is True
assert report["final_metrics"]["predicted_class_count"] >= 2
assert report["checkpoint_reload_max_abs_logit_delta"] <= 1e-6
print("Overfit sanity gate: PASS")
PY

echo "[6/8] Running first unweighted 120-image SegFormer-B0 fine-tuning"
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" training/train_traversability_segformer.py \
    --manifest "$APPROVED_DATASET/manifest.csv" \
    --config "$CONFIG_PATH" \
    --output-dir "$TRAINING_DIR" \
    --mode full \
    --require-cuda

echo "[7/8] Verifying offline experiment artifacts and single-use test evaluation"
"$PYTHON" - "$TRAINING_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
report = json.loads((root / "experiment_report.json").read_text(encoding="utf-8"))
assert report["success"] is True
assert report["test_evaluation_count"] == 1
assert report["test_used_for_hyperparameter_selection"] is False
assert report["test_metrics"]["predicted_class_count"] >= 2
assert report["checkpoint_reload_max_abs_logit_delta"] <= 1e-6
assert report["live_rover_commands_sent"] is False
for name in (
    "frozen_config.yaml", "training_history.csv", "final_metrics.json",
    "confusion_matrix.json", "segformer_b0_best.pt", "review_bundle/review.html",
    "review_bundle/contact_sheet.jpg", "review_bundle/failure_examples.jpg",
):
    assert (root / name).is_file() and (root / name).stat().st_size > 0
print(f"Best epoch: {report['best_epoch']}")
print(f"Test metrics: {report['test_metrics']}")
print(f"Peak VRAM: {report['peak_vram_bytes']}")
print(f"Elapsed seconds: {report['elapsed_seconds']}")
print("Offline experiment artifact gate: PASS")
PY

echo "[8/8] Verifying source immutability and Git exclusion"
after_raw="$(tree_fingerprint "$DATASET_ROOT")"
after_pilot="$(tree_fingerprint "$PILOT_BUNDLE")"
after_expansion="$(tree_fingerprint "$EXPANSION_BUNDLE")"
after_approved="$(tree_fingerprint "$APPROVED_DATASET")"
after_git_status="$(git status --porcelain)"
if [[ "$before_raw" != "$after_raw" || "$before_pilot" != "$after_pilot" || "$before_expansion" != "$after_expansion" ]]; then
    echo "ERROR: An immutable source dataset or approved annotation changed." >&2
    exit 1
fi
if [[ "$before_approved" != "$after_approved" ]]; then
    echo "ERROR: Approved 120-image dataset changed during training." >&2
    exit 1
fi
if [[ "$before_git_status" != "$after_git_status" ]]; then
    echo "ERROR: Git worktree changed during Dell execution." >&2
    git status --short >&2
    exit 1
fi
if [[ -n "$(git ls-files -- ':(glob)**/approved_120_v1/**' ':(glob)**/traversability_segformer_b0_v1/**')" ]]; then
    echo "ERROR: Generated dataset or experiment appears in git ls-files." >&2
    exit 1
fi
du -sh "$APPROVED_DATASET" "$OVERFIT_DIR" "$TRAINING_DIR" "$TRAINING_DIR/review_bundle"
echo "Raw dataset unchanged: PASS"
echo "Approved 20 and 100 annotations unchanged: PASS"
echo "Git exclusion: PASS"
echo "Live rover integration/commands: NOT PERFORMED"
echo "Approved dataset: $APPROVED_DATASET"
echo "Best checkpoint: $TRAINING_DIR/segformer_b0_best.pt"
echo "Mac review bundle: $TRAINING_DIR/review_bundle"
