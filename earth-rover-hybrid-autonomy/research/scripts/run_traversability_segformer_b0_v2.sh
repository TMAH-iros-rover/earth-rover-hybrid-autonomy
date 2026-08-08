#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
V2_DATASET="${V2_DATASET:-$HOME/datasets/generated/traversability_dataset_v2/approved_153_v2}"
MANUAL_V2="${MANUAL_V2:-$HOME/datasets/review_bundles/traversability_manual_v2_33_imported}"
V1_CHECKPOINT="${V1_CHECKPOINT:-$HOME/datasets/experiments/traversability_segformer_b0_v1/full_training/segformer_b0_best.pt}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$HOME/datasets/experiments/traversability_segformer_b0_v2}"
TRAINING_DIR="$EXPERIMENT_ROOT/full_training"
COMPARISON_DIR="$EXPERIMENT_ROOT/v1_v2_comparison"
CONFIG_PATH="$ROOT_DIR/configs/traversability_segformer_b0_v2.yaml"
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
    "$V2_DATASET/manifest.csv" \
    "$V2_DATASET/merge_report.json" \
    "$V2_DATASET/split_report.json" \
    "$MANUAL_V2/validation_report.json" \
    "$V1_CHECKPOINT" \
    "$CONFIG_PATH"; do
    if [[ ! -f "$path" ]]; then
        echo "ERROR: Required v2 training input is missing: $path" >&2
        exit 1
    fi
done
for path in "$TRAINING_DIR" "$COMPARISON_DIR"; do
    if [[ -e "$path" ]]; then
        echo "ERROR: Output path already exists; it will not be overwritten: $path" >&2
        exit 1
    fi
    case "$(realpath -m "$path")" in
        "$ROOT_DIR"/*)
            echo "ERROR: Experiment artifacts must remain outside Git: $path" >&2
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
    digest.update(f"{path.relative_to(root).as_posix()}|{stat.st_size}|{stat.st_mtime_ns}|".encode())
    digest.update(hashlib.sha256(path.read_bytes()).digest())
print(digest.hexdigest())
PY
}

cd "$ROOT_DIR"
before_git_status="$(git status --porcelain)"
before_dataset="$(tree_fingerprint "$V2_DATASET")"
before_manual="$(tree_fingerprint "$MANUAL_V2")"
before_v1_checkpoint="$(sha256sum "$V1_CHECKPOINT" | awk '{print $1}')"

echo "[1/5] Running focused loader, checkpoint, metric, and v2 split tests"
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -m pytest -p no:cacheprovider -q \
    tests/test_traversability_dataset_v2.py \
    tests/test_traversability_checkpoint_schema.py \
    tests/test_traversability_segmentation.py

echo "[2/5] Checking CUDA and approved v1 checkpoint schema"
"$PYTHON" - "$V1_CHECKPOINT" <<'PY'
import sys
import torch
from training.models.traversability_segformer import validate_three_class_checkpoint

if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")
checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
try:
    validate_three_class_checkpoint(checkpoint)
except ValueError as exc:
    raise SystemExit(f"v1 checkpoint schema invalid: {exc}") from exc
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
print(f"v1 source epoch: {checkpoint['epoch']}")
print("Approved v1 checkpoint schema: PASS")
PY

echo "[3/5] Fine-tuning v2 from the approved v1 best checkpoint"
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" training/train_traversability_segformer.py \
    --manifest "$V2_DATASET/manifest.csv" \
    --config "$CONFIG_PATH" \
    --output-dir "$TRAINING_DIR" \
    --mode full \
    --initial-checkpoint "$V1_CHECKPOINT" \
    --require-cuda

echo "[4/5] Comparing v1 and v2 on fixed v1 evaluation and new holdout"
set +e
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" training/evaluate_traversability_v1_v2.py \
    --manifest "$V2_DATASET/manifest.csv" \
    --config "$CONFIG_PATH" \
    --v1-checkpoint "$V1_CHECKPOINT" \
    --v2-checkpoint "$TRAINING_DIR/segformer_b0_best.pt" \
    --output-dir "$COMPARISON_DIR" \
    --require-cuda
comparison_status=$?
set -e

echo "[5/5] Verifying reports, immutable inputs, and Git exclusion"
"$PYTHON" - "$TRAINING_DIR" "$COMPARISON_DIR" <<'PY'
import json
import sys
from pathlib import Path

training = Path(sys.argv[1]).resolve()
comparison = Path(sys.argv[2]).resolve()
train_report = json.loads((training / "experiment_report.json").read_text(encoding="utf-8"))
compare_report = json.loads((comparison / "comparison_report.json").read_text(encoding="utf-8"))
assert train_report["success"] is True
assert train_report["initialization"]["mode"] == "approved_v1_best_checkpoint"
assert train_report["test_used_for_hyperparameter_selection"] is False
assert train_report["manifest_sha256"]
assert train_report["git_commit"]
assert compare_report["manifest_sha256"] == train_report["manifest_sha256"]
assert compare_report["test_used_for_hyperparameter_selection"] is False
for model in ("v1", "v2"):
    for split in ("validation", "test", "new_holdout"):
        metrics = compare_report["metrics"][model][split]
        assert set(metrics["per_class_iou"]) == {"ON_ROAD", "OFF_ROAD", "OBSTACLE"}
        assert len(metrics["confusion_matrix"]) == 3
for model in ("v1", "v2"):
    for split in ("test", "new_holdout"):
        assert (comparison / "reviews" / model / split / "review.html").is_file()
print(f"Best epoch: {train_report['best_epoch']}")
print(f"Training elapsed seconds: {train_report['elapsed_seconds']}")
print(f"Peak VRAM: {train_report['peak_vram_bytes']}")
print(f"Comparison status: {compare_report['status']}")
print(f"Assessments: {compare_report['assessments']}")
PY

after_dataset="$(tree_fingerprint "$V2_DATASET")"
after_manual="$(tree_fingerprint "$MANUAL_V2")"
after_v1_checkpoint="$(sha256sum "$V1_CHECKPOINT" | awk '{print $1}')"
after_git_status="$(git status --porcelain)"
if [[ "$before_dataset" != "$after_dataset" || "$before_manual" != "$after_manual" ]]; then
    echo "ERROR: An immutable dataset changed during training." >&2
    exit 1
fi
if [[ "$before_v1_checkpoint" != "$after_v1_checkpoint" ]]; then
    echo "ERROR: The approved v1 checkpoint changed." >&2
    exit 1
fi
if [[ "$before_git_status" != "$after_git_status" ]]; then
    echo "ERROR: Git worktree changed during training." >&2
    git status --short >&2
    exit 1
fi
if [[ -n "$(git ls-files -- ':(glob)**/traversability_segformer_b0_v2/**')" ]]; then
    echo "ERROR: Generated v2 experiment appears in git ls-files." >&2
    exit 1
fi
du -sh "$TRAINING_DIR" "$COMPARISON_DIR"
echo "V2 dataset, manual annotations, and v1 checkpoint unchanged: PASS"
echo "Git exclusion: PASS"
echo "Planner/SDK/live rover operations: NOT PERFORMED"
if [[ "$comparison_status" -ne 0 ]]; then
    echo "ERROR: v2 failed the fixed-v1 evaluation regression gate. See $COMPARISON_DIR/comparison_report.json" >&2
    exit "$comparison_status"
fi
echo "SegFormer-B0 v2 fine-tuning and offline comparison: PASS"
