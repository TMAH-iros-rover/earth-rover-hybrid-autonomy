#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-$HOME/datasets/output_rides_0}"
MANIFEST_PATH="${MANIFEST_PATH:-$HOME/datasets/manifests/frodobots_2k_phase2/full_dataset/manifest.csv}"
BUNDLE_ROOT="${BUNDLE_ROOT:-$HOME/datasets/review_bundles/traversability_pilot_v1}"
SAMPLE_COUNT="${SAMPLE_COUNT:-40}"
MAX_RIDES="${MAX_RIDES:-8}"
MINIMUM_SEPARATION_SECONDS="${MINIMUM_SEPARATION_SECONDS:-5}"
SEED="${SEED:-20260716}"
export HF_HOME="${HF_HOME:-$HOME/datasets/generated/huggingface}"

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
    echo "DATASET_ROOT=$DATASET_ROOT" >&2
    echo "MANIFEST_PATH=$MANIFEST_PATH" >&2
    exit 1
fi
if [[ -d "$BUNDLE_ROOT" && -n "$(find "$BUNDLE_ROOT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "ERROR: Review bundle directory is not empty; choose a new BUNDLE_ROOT." >&2
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
    digest.update(f"{path.relative_to(root).as_posix()}|{stat.st_size}|{stat.st_mtime_ns}\n".encode())
print(digest.hexdigest())
PY
}

cd "$ROOT_DIR"

echo "[1/5] Running focused traversability review tests"
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -m pytest \
    -p no:cacheprovider -q \
    tests/test_traversability_review.py \
    tests/test_frodobots_2k_dataset.py

echo "[2/5] Checking CUDA, packages, and raw dataset fingerprint"
"$PYTHON" - <<'PY'
import importlib.metadata
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")
for package in ("transformers", "safetensors", "huggingface-hub"):
    print(f"{package}: {importlib.metadata.version(package)}")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
PY
before_fingerprint="$(dataset_fingerprint)"

echo "[3/5] Building the conservative pseudo-label review bundle"
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" training/build_traversability_review_bundle.py \
    --dataset-root "$DATASET_ROOT" \
    --manifest "$MANIFEST_PATH" \
    --output-dir "$BUNDLE_ROOT" \
    --sample-count "$SAMPLE_COUNT" \
    --max-rides "$MAX_RIDES" \
    --minimum-separation-seconds "$MINIMUM_SEPARATION_SECONDS" \
    --seed "$SEED" \
    --require-cuda

echo "[4/5] Validating bundle integrity and unreviewed gate"
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" training/validate_traversability_review.py \
    --bundle "$BUNDLE_ROOT"

echo "[5/5] Verifying raw dataset immutability and bundle report"
after_fingerprint="$(dataset_fingerprint)"
if [[ "$before_fingerprint" != "$after_fingerprint" ]]; then
    echo "ERROR: Raw dataset metadata changed." >&2
    exit 1
fi

"$PYTHON" - "$BUNDLE_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
report = json.loads((root / "bundle_report.json").read_text(encoding="utf-8"))
validation = json.loads((root / "review_validation_report.json").read_text(encoding="utf-8"))
assert report["pipeline_status"] == "REVIEW_REQUIRED"
assert report["pseudo_labels_are_ground_truth"] is False
assert report["fine_tuning_performed"] is False
assert report["live_rover_commands_sent"] is False
assert report["successful_frame_count"] > 0
assert report["processed_ride_count"] >= 2
assert validation["valid"] is True
assert validation["verified_training_sample_count"] == 0
for relative in ("README.md", "label_contract.yaml", "semantic_mapping.yaml", "cvat_labelmap.txt", "review.csv", "gallery.html", "contact_sheet.jpg"):
    path = root / relative
    assert path.is_file() and path.stat().st_size > 0
print(f"Rides: {report['processed_ride_count']}")
print(f"Frames: success={report['successful_frame_count']} failed={report['failed_frame_count']}")
print(f"Latency ms: {report['inference_latency_ms']}")
print(f"Peak VRAM bytes: {report['peak_vram_bytes']}")
print(f"Review bundle: {root}")
print("Raw dataset unchanged: PASS")
print("Human review gate: ACTIVE")
PY

du -sh "$BUNDLE_ROOT"
echo "Traversability pseudo-label pilot: PASS (review required; no training performed)"
