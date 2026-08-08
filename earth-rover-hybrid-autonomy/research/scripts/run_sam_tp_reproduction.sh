#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "$PROJECT_ROOT/../.." && pwd)"
UPSTREAM_ROOT="${UPSTREAM_ROOT:-$WORKSPACE_ROOT/external/GENIE-SAMTP}"
ENV_NAME="${ENV_NAME:-sam_tp_repro}"
ENV_BACKEND="${ENV_BACKEND:-auto}"
VENV_PATH="${VENV_PATH:-$WORKSPACE_ROOT/external/venvs/$ENV_NAME}"
PROJECT_PYTHON="${PROJECT_PYTHON:-python3}"
CONFIG="${CONFIG:-$PROJECT_ROOT/configs/sam_tp_reproduction.yaml}"
CHECKPOINT="${CHECKPOINT:-$UPSTREAM_ROOT/sam2_logs/configs/sam2.1_training_tiny/sam2_training_custom2_freezeNoneNone_f57.yaml/checkpoints/checkpoint_2.pt}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/datasets/experiments/sam_tp_reproduction/$RUN_ID}"
VIDEO_OUTPUT="${VIDEO_OUTPUT:-$HOME/datasets/review_bundles/sam_tp_reproduction/$RUN_ID}"
SMOKE_IMAGE="${SMOKE_IMAGE:-$UPSTREAM_ROOT/stretch_example/stretch_obs/rgb.png}"
DATASETS="${DATASETS:-0}"
RIDES_PER_DATASET="${RIDES_PER_DATASET:-1}"
SECONDS_PER_RIDE="${SECONDS_PER_RIDE:-30}"
DATASET_ROOT_0="${DATASET_ROOT_0:-$HOME/datasets/output_rides_0}"
DATASET_ROOT_1="${DATASET_ROOT_1:-$HOME/datasets/output_rides_1}"
DATASET_ROOT_2="${DATASET_ROOT_2:-$HOME/datasets/output_rides_2}"

for path in "$UPSTREAM_ROOT" "$CONFIG" "$CHECKPOINT" "$SMOKE_IMAGE"; do
  if [[ ! -e "$path" ]]; then
    echo "ERROR: Required input is missing: $path" >&2
    exit 2
  fi
done
if [[ -e "$OUTPUT_ROOT" || -e "$VIDEO_OUTPUT" ]]; then
  echo "ERROR: Run output already exists; choose a new RUN_ID." >&2
  exit 2
fi
if [[ "$ENV_BACKEND" == "auto" ]]; then
  if command -v conda >/dev/null 2>&1 \
    && conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    ENV_BACKEND="conda"
  elif [[ -x "$VENV_PATH/bin/python" ]]; then
    ENV_BACKEND="venv"
  else
    echo "ERROR: No SAM-TP Conda environment or venv was found; run setup first." >&2
    exit 2
  fi
fi
if [[ "$ENV_BACKEND" == "conda" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: ENV_BACKEND=conda was requested but conda is unavailable." >&2
    exit 2
  fi
  env_python() {
    conda run --no-capture-output -n "$ENV_NAME" python "$@"
  }
elif [[ "$ENV_BACKEND" == "venv" ]]; then
  if [[ ! -x "$VENV_PATH/bin/python" ]]; then
    echo "ERROR: SAM-TP venv is missing: $VENV_PATH" >&2
    exit 2
  fi
  env_python() {
    "$VENV_PATH/bin/python" "$@"
  }
else
  echo "ERROR: ENV_BACKEND must be auto, conda, or venv." >&2
  exit 2
fi
export SAM_TP_ENVIRONMENT_BACKEND="$ENV_BACKEND"
for command_name in ffmpeg ffprobe; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "ERROR: $command_name is required for the QuickTime review video." >&2
    exit 2
  fi
done
if [[ "$(ffmpeg -hide_banner -encoders 2>/dev/null)" != *libx264* ]]; then
  echo "ERROR: ffmpeg does not provide the libx264 encoder." >&2
  exit 2
fi
read -r -a dataset_arguments <<< "$DATASETS"
if [[ "${#dataset_arguments[@]}" -eq 1 && "${dataset_arguments[0]}" == "all" ]]; then
  dataset_root_indexes=(0 1 2)
else
  dataset_root_indexes=("${dataset_arguments[@]}")
fi
dataset_root_arguments=()
for dataset_index in "${dataset_root_indexes[@]}"; do
  if [[ ! "$dataset_index" =~ ^[0-9]+$ ]]; then
    echo "ERROR: DATASETS must contain numeric indexes or only 'all'." >&2
    exit 2
  fi
  variable_name="DATASET_ROOT_${dataset_index}"
  dataset_root="${!variable_name:-$HOME/datasets/output_rides_${dataset_index}}"
  dataset_root_arguments+=(--dataset-root "$dataset_index=$dataset_root")
done
cd "$PROJECT_ROOT"
before_git_status="$(git status --porcelain)"
before_checkpoint_sha="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"

mkdir -p "$OUTPUT_ROOT"

echo "[1/7] Running checkpoint-free SAM-TP unit tests"
PYTHONDONTWRITEBYTECODE=1 env_python -m pytest -p no:cacheprovider -q \
  "$PROJECT_ROOT/tests/test_sam_tp_reproduction.py" \
  "$PROJECT_ROOT/tests/test_traversability_video_review_v2.py"

echo "[2/7] Running the existing Earth Rover regression suite"
PYTHONDONTWRITEBYTECODE=1 "$PROJECT_PYTHON" -m pytest -p no:cacheprovider -q \
  "$PROJECT_ROOT/tests"

echo "[3/7] Strictly checking training config, inference config, and checkpoint"
env_python \
  "$PROJECT_ROOT/training/inspect_sam_tp_compatibility.py" \
  --reproduction-config "$CONFIG" \
  --upstream-root "$UPSTREAM_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT_ROOT/compatibility_report.json"

echo "[4/7] Running one-image CUDA smoke inference and benchmark"
SAM_TP_UPSTREAM_ROOT="$UPSTREAM_ROOT" \
SAM_TP_CHECKPOINT="$CHECKPOINT" \
SAM_TP_SMOKE_IMAGE="$SMOKE_IMAGE" \
PYTHONDONTWRITEBYTECODE=1 env_python -m pytest -p no:cacheprovider -q \
  "$PROJECT_ROOT/tests/test_sam_tp_integration.py"
env_python \
  "$PROJECT_ROOT/training/run_sam_tp_smoke.py" \
  --image "$SMOKE_IMAGE" \
  --reproduction-config "$CONFIG" \
  --upstream-root "$UPSTREAM_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --compatibility-report "$OUTPUT_ROOT/compatibility_report.json" \
  --output-dir "$OUTPUT_ROOT/single_image"

echo "[5/7] Running deterministic FrodoBots CUDA review"
env_python \
  "$PROJECT_ROOT/training/run_sam_tp_video_review.py" \
  --reproduction-config "$CONFIG" \
  --upstream-root "$UPSTREAM_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --compatibility-report "$OUTPUT_ROOT/compatibility_report.json" \
  --output-dir "$VIDEO_OUTPUT" \
  --datasets "${dataset_arguments[@]}" \
  --dataset-root-0 "$DATASET_ROOT_0" \
  --dataset-root-1 "$DATASET_ROOT_1" \
  --dataset-root-2 "$DATASET_ROOT_2" \
  "${dataset_root_arguments[@]}" \
  --rides-per-dataset "$RIDES_PER_DATASET" \
  --seconds-per-ride "$SECONDS_PER_RIDE"

echo "[6/7] Writing machine-readable and Markdown reproduction reports"
env_python \
  "$PROJECT_ROOT/training/write_sam_tp_reproduction_report.py" \
  --compatibility-report "$OUTPUT_ROOT/compatibility_report.json" \
  --smoke-report "$OUTPUT_ROOT/single_image/metadata.json" \
  --video-report "$VIDEO_OUTPUT/review_manifest.json" \
  --output-json "$OUTPUT_ROOT/reproduction_report.json" \
  --output-markdown "$OUTPUT_ROOT/reproduction_report.md"

echo "[7/7] Reporting outputs"
echo "Compatibility: $OUTPUT_ROOT/compatibility_report.json"
echo "Single image: $OUTPUT_ROOT/single_image"
echo "Video review: $VIDEO_OUTPUT"
echo "Aggregate report: $OUTPUT_ROOT/reproduction_report.json"
after_git_status="$(git status --porcelain)"
after_checkpoint_sha="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
if [[ "$before_git_status" != "$after_git_status" ]]; then
  echo "ERROR: The project worktree changed during reproduction." >&2
  git status --short >&2
  exit 2
fi
if [[ "$before_checkpoint_sha" != "$after_checkpoint_sha" ]]; then
  echo "ERROR: The official checkpoint changed during reproduction." >&2
  exit 2
fi
echo "No training, SDK call, planner integration, or live rover command was performed."
