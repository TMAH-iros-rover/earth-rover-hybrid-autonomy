#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-$HOME/datasets/output_rides_0}"
MANIFEST_PATH="${MANIFEST_PATH:-$HOME/datasets/manifests/frodobots_2k_phase2/full_dataset/manifest.csv}"
TINY_REPORT="${TINY_REPORT:-$HOME/datasets/outputs/frodobots_2k_phase3/tiny_overfit/tiny_overfit_report.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/datasets/outputs/frodobots_2k_phase3/small_baseline}"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON=python
else
    echo "ERROR: Python 3 was not found." >&2
    exit 1
fi

if [[ ! -d "$DATASET_ROOT" || ! -f "$MANIFEST_PATH" ]]; then
    echo "ERROR: Dataset or full manifest is missing." >&2
    exit 1
fi
if [[ ! -f "$TINY_REPORT" ]]; then
    echo "ERROR: Phase 3 tiny-overfit report is missing." >&2
    echo "Run ./scripts/verify_phase3_tiny_overfit.sh first." >&2
    exit 1
fi
"$PYTHON" - "$TINY_REPORT" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if report.get("success") is not True:
    raise SystemExit("Phase 3 tiny-overfit gate has not passed")
PY

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

echo "[1/4] Running focused small-baseline tests"
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -m pytest \
    -p no:cacheprovider -q \
    tests/test_phase3_small_baseline.py \
    tests/test_phase3_tiny_overfit.py

echo "[2/4] Checking CUDA and recording raw dataset fingerprint"
"$PYTHON" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
PY
before_fingerprint="$(dataset_fingerprint)"

echo "[3/4] Training the bounded 10/2/2 ride-level baseline"
training_status=0
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" training/train_small_baseline.py \
    --dataset-root "$DATASET_ROOT" \
    --manifest "$MANIFEST_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --samples-per-ride 250 \
    --batch-size 32 \
    --max-epochs 12 \
    --patience 3 \
    --learning-rate 0.0003 \
    --video-frames-per-ride 200 \
    --require-cuda || training_status=$?

echo "[4/4] Verifying report, video, checkpoint, and raw dataset"
after_fingerprint="$(dataset_fingerprint)"
if [[ "$before_fingerprint" != "$after_fingerprint" ]]; then
    echo "ERROR: Raw dataset metadata changed during the baseline run." >&2
    exit 1
fi
if [[ "$training_status" -ne 0 ]]; then
    if [[ -f "$OUTPUT_DIR/small_baseline_report.json" ]]; then
        echo "ERROR: Baseline run failed. Inspect $OUTPUT_DIR/small_baseline_report.json" >&2
    else
        echo "ERROR: Baseline run stopped before a report was generated." >&2
    fi
    exit "$training_status"
fi

"$PYTHON" - "$OUTPUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1]).resolve()
report = json.loads((output_dir / "small_baseline_report.json").read_text(encoding="utf-8"))
checkpoint = Path(report["checkpoint_path"])
video = Path(report["prediction_video"]["path"])

assert report["execution_success"] is True
assert report["device"] == "cuda"
assert report["split_overlap"] == []
assert len(report["ride_split"]["train"]) == 10
assert len(report["ride_split"]["validation"]) == 2
assert len(report["ride_split"]["test"]) == 2
assert report["selected_samples"]["train"]["count"] == 2500
assert report["selected_samples"]["validation"]["count"] == 500
assert report["selected_samples"]["test"]["count"] == 500
assert all(item["decode_failure_count"] == 0 for item in report["selected_samples"].values())
assert report["prediction_video"]["frame_count"] == 400
assert report["prediction_video"]["codec"] == "mp4v"
assert report["prediction_video"]["resolution"] == [1024, 576]
assert checkpoint.is_file() and checkpoint.stat().st_size > 0
assert video.is_file() and video.stat().st_size > 0

metrics = report["test_metrics"]
print(f"Best epoch: {report['best_epoch']}")
print(f"Validation macro F1: {report['best_validation_macro_f1']:.4f}")
print(f"Test macro F1: {metrics['macro_f1']:.4f}")
print(f"Test balanced accuracy: {metrics['balanced_accuracy']:.4f}")
print(f"Test accuracy: {metrics['accuracy']:.4f}")
print(f"Prediction video: {video}")
print(f"Video format: {report['prediction_video']['codec']} {report['prediction_video']['resolution']} at {report['prediction_video']['fps']} fps")
print("Raw dataset unchanged: PASS")
print("Phase 3 small-baseline execution: PASS")
PY
