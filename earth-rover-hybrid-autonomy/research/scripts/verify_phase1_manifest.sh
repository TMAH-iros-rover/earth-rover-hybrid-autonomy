#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${FRODOBOTS_DATASET_ROOT:-$HOME/datasets/output_rides_0}"
OUTPUT_DIR="${FRODOBOTS_MANIFEST_OUTPUT:-$HOME/datasets/manifests/frodobots_2k_phase1/dell_verification_3rides}"

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

case "$OUTPUT_DIR/" in
    "$ROOT_DIR/"*)
        echo "ERROR: Manifest output must remain outside the Git workspace: $OUTPUT_DIR" >&2
        exit 1
        ;;
esac

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

echo "[1/4] Running focused Phase 1 tests"
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -m pytest \
    -p no:cacheprovider -q \
    tests/test_frodobots_2k_manifest.py \
    tests/test_action_labels.py

echo "[2/4] Recording raw dataset fingerprint"
before_fingerprint="$(dataset_fingerprint)"

echo "[3/4] Building a three-ride manifest"
if ! PYTHONDONTWRITEBYTECODE=1 "$PYTHON" training/build_frodobots_2k_manifest.py \
    --dataset-root "$DATASET_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --max-rides 3 \
    --control-tolerance-ms 100; then
    after_fingerprint="$(dataset_fingerprint)"
    if [[ "$before_fingerprint" != "$after_fingerprint" ]]; then
        echo "ERROR: Raw dataset metadata changed during the failed build." >&2
    fi
    exit 1
fi

echo "[4/4] Verifying manifest, report, and raw dataset"
after_fingerprint="$(dataset_fingerprint)"
if [[ "$before_fingerprint" != "$after_fingerprint" ]]; then
    echo "ERROR: Raw dataset metadata changed during manifest generation." >&2
    exit 1
fi

"$PYTHON" - "$OUTPUT_DIR" <<'PY'
import csv
import hashlib
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1]).resolve()
manifest_path = output_dir / "manifest.csv"
report_path = output_dir / "alignment_report.json"
report = json.loads(report_path.read_text(encoding="utf-8"))

with manifest_path.open(newline="", encoding="utf-8") as handle:
    manifest_rows = sum(1 for _ in csv.DictReader(handle))

manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
total = report["total_front_frame_count"]
valid = report["valid_sample_count"]
rejected = report["rejected_sample_count"]

assert report["processed_ride_count"] == 3
assert total == valid + rejected
assert manifest_rows == valid
assert report["control_delta_ms"]["max"] <= 100.0
assert report["manifest_sha256"] == manifest_sha256

print(f"Processed rides: {report['processed_ride_count']}")
print(f"Front frames: {total}")
print(f"Valid samples: {valid}")
print(f"Rejected samples: {rejected}")
print(f"Control delta: {report['control_delta_ms']}")
print(f"Manifest: {manifest_path}")
print(f"Manifest SHA-256: {manifest_sha256}")
print("Raw dataset unchanged: PASS")
print("Phase 1 verification: PASS")
PY
