#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-$HOME/datasets/output_rides_0}"
MANIFEST_PATH="${MANIFEST_PATH:-$HOME/datasets/manifests/frodobots_2k_phase2/full_dataset/manifest.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/datasets/outputs/frodobots_2k_phase3/tiny_overfit}"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON=python
else
    echo "ERROR: Python 3 was not found." >&2
    exit 1
fi

if [[ ! -d "$DATASET_ROOT" ]]; then
    echo "ERROR: Dataset directory not found: $DATASET_ROOT" >&2
    exit 1
fi
if [[ ! -f "$MANIFEST_PATH" ]]; then
    echo "ERROR: Full Phase 2 manifest not found: $MANIFEST_PATH" >&2
    echo "Run ./scripts/audit_phase2_alignment.sh first." >&2
    exit 1
fi

dataset_fingerprint() {
    "$PYTHON" - "$DATASET_ROOT" <<'PY'
import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
digest = hashlib.sha256()
for path in sorted(item for item in root.rglob("*") if item.is_file()):
    stat = path.stat()
    record = f"{path.relative_to(root).as_posix()}|{stat.st_size}|{stat.st_mtime_ns}\n"
    digest.update(record.encode("utf-8"))
print(digest.hexdigest())
PY
}

cd "$ROOT_DIR"

echo "[1/4] Running focused Phase 3 tests"
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -m pytest \
    -p no:cacheprovider -q \
    tests/test_phase3_tiny_overfit.py \
    tests/test_action_labels.py

echo "[2/4] Checking CUDA and recording raw dataset fingerprint"
"$PYTHON" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
PY
before_fingerprint="$(dataset_fingerprint)"

echo "[3/4] Running the 200-sample ResNet18 tiny-overfit gate"
training_status=0
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" training/train_tiny_overfit.py \
    --dataset-root "$DATASET_ROOT" \
    --manifest "$MANIFEST_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --sample-count 200 \
    --batch-size 32 \
    --max-epochs 40 \
    --learning-rate 0.001 \
    --target-accuracy 0.95 \
    --require-cuda || training_status=$?

echo "[4/4] Verifying report, checkpoint, and raw dataset"
after_fingerprint="$(dataset_fingerprint)"
if [[ "$before_fingerprint" != "$after_fingerprint" ]]; then
    echo "ERROR: Raw dataset metadata changed during Phase 3 training." >&2
    exit 1
fi
if [[ "$training_status" -ne 0 ]]; then
    if [[ -f "$OUTPUT_DIR/tiny_overfit_report.json" ]]; then
        echo "ERROR: Tiny-overfit gate failed. Inspect $OUTPUT_DIR/tiny_overfit_report.json" >&2
    else
        echo "ERROR: Tiny-overfit gate stopped before a report was generated." >&2
    fi
    exit "$training_status"
fi

"$PYTHON" - "$OUTPUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1]).resolve()
report = json.loads((output_dir / "tiny_overfit_report.json").read_text(encoding="utf-8"))
checkpoint = Path(report["checkpoint_path"])

assert report["success"] is True
assert report["device"] == "cuda"
assert report["sample_count"] == 200
assert set(report["sample_action_distribution"].values()) == {40}
assert report["final_accuracy"] >= report["target_accuracy"]
assert report["loss_decreased"] is True
assert report["checkpoint_reload_max_abs_logit_delta"] <= 1e-6
assert checkpoint.is_file() and checkpoint.stat().st_size > 0

print(f"Epochs: {report['epochs_completed']}")
print(f"Initial loss: {report['initial_loss']:.6f}")
print(f"Final loss: {report['final_loss']:.6f}")
print(f"Final accuracy: {report['final_accuracy']:.4f}")
print(f"Checkpoint: {checkpoint}")
print(f"Reload max logit delta: {report['checkpoint_reload_max_abs_logit_delta']}")
print("Raw dataset unchanged: PASS")
print("Phase 3 tiny-overfit gate: PASS")
PY
