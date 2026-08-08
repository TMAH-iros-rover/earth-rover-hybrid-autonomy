#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-$HOME/datasets/output_rides_0}"
MANIFEST_DIR="${MANIFEST_DIR:-$HOME/datasets/manifests/frodobots_2k_phase2/full_dataset}"
MANIFEST_PATH="$MANIFEST_DIR/manifest.csv"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/datasets/outputs/frodobots_2k_phase2/semantic_alignment_audit}"

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

echo "[1/5] Running focused edge-case tests"
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -m pytest \
    -p no:cacheprovider -q \
    tests/test_frodobots_2k_alignment_audit.py \
    tests/test_frodobots_2k_dataset.py \
    tests/test_frodobots_2k_manifest.py \
    tests/test_action_labels.py

echo "[2/5] Recording raw dataset fingerprint"
before_fingerprint="$(dataset_fingerprint)"

echo "[3/5] Building a full read-only manifest for all rides"
if ! PYTHONDONTWRITEBYTECODE=1 "$PYTHON" training/build_frodobots_2k_manifest.py \
    --dataset-root "$DATASET_ROOT" \
    --output-dir "$MANIFEST_DIR" \
    --control-tolerance-ms 100; then
    after_fingerprint="$(dataset_fingerprint)"
    if [[ "$before_fingerprint" != "$after_fingerprint" ]]; then
        echo "ERROR: Raw dataset metadata changed during the failed manifest build." >&2
    fi
    exit 1
fi

echo "[4/5] Auditing transform, temporal strips, action signs, and edge cases"
audit_status=0
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" training/audit_frodobots_2k_alignment.py \
    --dataset-root "$DATASET_ROOT" \
    --manifest "$MANIFEST_PATH" \
    --output-dir "$OUTPUT_DIR" || audit_status=$?

echo "[5/5] Checking raw dataset and audit result"
after_fingerprint="$(dataset_fingerprint)"
if [[ "$before_fingerprint" != "$after_fingerprint" ]]; then
    echo "ERROR: Raw dataset metadata changed during the audit." >&2
    exit 1
fi
if [[ "$audit_status" -ne 0 ]]; then
    echo "ERROR: Phase 2 alignment audit failed. Inspect $OUTPUT_DIR/semantic_alignment_report.json" >&2
    exit "$audit_status"
fi

echo "Raw dataset unchanged: PASS"
echo "Phase 2 automated alignment audit: CONDITIONAL PASS"
echo "Manual review required: inspect left_strips.jpg and right_strips.jpg before final PASS."
echo "Audit directory: $OUTPUT_DIR"
