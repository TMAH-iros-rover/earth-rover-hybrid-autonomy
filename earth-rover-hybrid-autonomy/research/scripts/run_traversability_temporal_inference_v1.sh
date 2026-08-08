#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-$HOME/datasets/output_rides_0}"
FULL_MANIFEST="${FULL_MANIFEST:-$HOME/datasets/manifests/frodobots_2k_phase2/full_dataset/manifest.csv}"
APPROVED_MANIFEST="${APPROVED_MANIFEST:-$HOME/datasets/generated/traversability_dataset_v1/approved_120_v1/manifest.csv}"
CHECKPOINT="${CHECKPOINT:-$HOME/datasets/experiments/traversability_segformer_b0_v1/full_training/segformer_b0_best.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/datasets/review_bundles/traversability_temporal_v1}"
CONFIG_PATH="$ROOT_DIR/configs/traversability_temporal_inference_v1.yaml"
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

for path in "$DATASET_ROOT" "$FULL_MANIFEST" "$APPROVED_MANIFEST" "$CHECKPOINT" "$CONFIG_PATH"; do
    if [[ ! -e "$path" ]]; then
        echo "ERROR: Required Dell input is missing: $path" >&2
        exit 1
    fi
done
if [[ -d "$OUTPUT_DIR" && -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "ERROR: Output is not empty; preserve it and choose a new OUTPUT_DIR: $OUTPUT_DIR" >&2
    exit 1
fi
case "$(realpath -m "$OUTPUT_DIR")" in
    "$ROOT_DIR"/*)
        echo "ERROR: Temporal review artifacts must remain outside Git: $OUTPUT_DIR" >&2
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

echo "[1/5] Running focused temporal inference and segmentation preprocessing tests"
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -m pytest -p no:cacheprovider -q tests/test_traversability_temporal_inference.py tests/test_traversability_segmentation.py

echo "[2/5] Checking CUDA and recording immutable input fingerprints"
"$PYTHON" - "$CHECKPOINT" <<'PY'
import sys
from pathlib import Path
import torch

checkpoint = Path(sys.argv[1])
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")
loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
for key in ("model_state_dict", "model_config", "epoch"):
    assert key in loaded, f"checkpoint missing {key}"
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
print(f"Checkpoint epoch: {loaded['epoch']}")
PY
before_raw="$(tree_fingerprint "$DATASET_ROOT")"
before_approved="$(tree_fingerprint "$(dirname "$APPROVED_MANIFEST")")"
before_checkpoint="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"

echo "[3/5] Running raw temporal inference on 3 unseen 30-second ride segments"
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" training/run_traversability_temporal_inference.py --dataset-root "$DATASET_ROOT" --full-manifest "$FULL_MANIFEST" --approved-manifest "$APPROVED_MANIFEST" --checkpoint "$CHECKPOINT" --config "$CONFIG_PATH" --output-dir "$OUTPUT_DIR" --require-cuda

echo "[4/5] Verifying temporal artifacts, split isolation, and raw prediction policy"
"$PYTHON" - "$OUTPUT_DIR" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
report = json.loads((root / "temporal_inference_report.json").read_text(encoding="utf-8"))
selection = json.loads((root / "selected_segments.json").read_text(encoding="utf-8"))
rows = list(csv.DictReader((root / "per_frame_statistics.csv").open(newline="", encoding="utf-8")))
assert report["success"] is True
assert report["selected_ride_count"] == 3
assert report["approved_ride_overlap"] == []
assert selection["ride_overlap"] == []
assert selection["test_split_evaluated"] is False
assert report["confidence_threshold_applied"] is False
assert report["temporal_smoothing_applied"] is False
assert report["additional_training_performed"] is False
assert report["sdk_or_live_rover_commands_sent"] is False
assert all(segment["duration_seconds"] >= 29.9 for segment in selection["segments"])
assert len(rows) == report["successful_frame_count"] + report["decode_failure_count"]
for relative in report["video_paths"]:
    path = root / relative
    assert path.is_file() and path.stat().st_size > 0
for name in ("review.html", "README.md", "anomaly_candidates.json", "frozen_config.yaml"):
    assert (root / name).is_file() and (root / name).stat().st_size > 0
print(f"Rides: {[item['ride_id'] for item in selection['segments']]}")
print(f"Frames: success={report['successful_frame_count']} failed={report['decode_failure_count']}")
print(f"Latency ms: {report['latency_ms']}")
print(f"Effective FPS: {report['effective_fps']}")
print(f"Peak VRAM: {report['peak_vram_bytes']}")
print(f"Anomaly candidates: {report['anomaly_candidate_count']}")
print("Temporal artifact and unseen-ride gate: PASS")
PY

echo "[5/5] Verifying immutable inputs and Git exclusion"
after_raw="$(tree_fingerprint "$DATASET_ROOT")"
after_approved="$(tree_fingerprint "$(dirname "$APPROVED_MANIFEST")")"
after_checkpoint="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
after_git_status="$(git status --porcelain)"
if [[ "$before_raw" != "$after_raw" || "$before_approved" != "$after_approved" || "$before_checkpoint" != "$after_checkpoint" ]]; then
    echo "ERROR: Raw data, approved dataset, or checkpoint changed." >&2
    exit 1
fi
if [[ "$before_git_status" != "$after_git_status" ]]; then
    echo "ERROR: Git worktree changed during Dell execution." >&2
    git status --short >&2
    exit 1
fi
if [[ -n "$(git ls-files -- ':(glob)**/traversability_temporal_v1/**')" ]]; then
    echo "ERROR: Generated temporal artifacts appear in git ls-files." >&2
    exit 1
fi
du -sh "$OUTPUT_DIR"
echo "Raw dataset unchanged: PASS"
echo "Approved dataset and best checkpoint unchanged: PASS"
echo "Git exclusion: PASS"
echo "Additional training: NOT PERFORMED"
echo "SDK/live rover integration: NOT PERFORMED"
echo "Review bundle: $OUTPUT_DIR"
