#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-$HOME/datasets/output_rides_0}"
PHASE1_DIR="${PHASE1_DIR:-$HOME/datasets/manifests/frodobots_2k_phase1/dell_verification_3rides}"
MANIFEST_PATH="$PHASE1_DIR/manifest.csv"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/datasets/outputs/frodobots_2k_phase2/dell_verification_20samples}"

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

echo "[1/5] Running focused Phase 2 tests"
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -m pytest \
    -p no:cacheprovider -q \
    tests/test_frodobots_2k_dataset.py

echo "[2/5] Checking the Phase 1 manifest"
if [[ ! -f "$MANIFEST_PATH" ]]; then
    echo "Phase 1 manifest not found; running Phase 1 verification first."
    ./scripts/verify_phase1_manifest.sh
fi

echo "[3/5] Recording raw dataset fingerprint"
before_fingerprint="$(dataset_fingerprint)"

echo "[4/5] Decoding and visualizing 20 aligned samples"
if ! PYTHONDONTWRITEBYTECODE=1 "$PYTHON" training/verify_frodobots_2k_hls.py \
    --dataset-root "$DATASET_ROOT" \
    --manifest "$MANIFEST_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --num-samples 20 \
    --batch-size 4; then
    after_fingerprint="$(dataset_fingerprint)"
    if [[ "$before_fingerprint" != "$after_fingerprint" ]]; then
        echo "ERROR: Raw dataset metadata changed during failed Phase 2 verification." >&2
    fi
    exit 1
fi

echo "[5/5] Checking report and raw dataset"
after_fingerprint="$(dataset_fingerprint)"
if [[ "$before_fingerprint" != "$after_fingerprint" ]]; then
    echo "ERROR: Raw dataset metadata changed during Phase 2 verification." >&2
    exit 1
fi

"$PYTHON" - "$OUTPUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1]).resolve()
report = json.loads((output_dir / "hls_verification_report.json").read_text(encoding="utf-8"))
visualization = Path(report["visualization_path"])

assert report["decoded_sample_count"] >= 20
assert report["deterministic_repeat_access"] is True
assert report["batch_shape"] == [4, 3, 224, 224]
assert visualization.is_file() and visualization.stat().st_size > 0

print(f"Decoded samples: {report['decoded_sample_count']}")
print(f"Unreadable samples: {report['unreadable_sample_count']}")
print(f"Batch shape: {report['batch_shape']}")
print(f"Visualization: {visualization}")
print("Raw dataset unchanged: PASS")
print("Phase 2 automated verification: PASS")
print("Manual check required: inspect aligned_samples.jpg for image/label plausibility.")
PY
