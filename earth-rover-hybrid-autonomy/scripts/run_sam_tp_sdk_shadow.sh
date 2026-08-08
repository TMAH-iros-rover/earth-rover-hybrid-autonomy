#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"
UPSTREAM_ROOT="${UPSTREAM_ROOT:-$WORKSPACE_ROOT/external/GENIE-SAMTP}"
ENV_NAME="${ENV_NAME:-sam_tp_repro}"
ENV_BACKEND="${ENV_BACKEND:-auto}"
VENV_PATH="${VENV_PATH:-$WORKSPACE_ROOT/external/venvs/$ENV_NAME}"
MODEL_CONFIG="${MODEL_CONFIG:-$UPSTREAM_ROOT/sam2/configs/sam2.1_inference_tiny/sam2.1_custom2.yaml}"
CHECKPOINT="${CHECKPOINT:-$UPSTREAM_ROOT/sam2_logs/configs/sam2.1_training_tiny/sam2_training_custom2_freezeNoneNone_f57.yaml/checkpoints/checkpoint_2.pt}"
EXPECTED_CHECKPOINT_SHA256="${EXPECTED_CHECKPOINT_SHA256:-2607fd6049d37f17fe96132cf35459f7e0a895107632410637d812756e3f9adb}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/datasets/review_bundles/sam_tp_sdk_shadow/$RUN_ID}"
TARGET_FPS="${TARGET_FPS:-4}"
TELEMETRY_HZ="${TELEMETRY_HZ:-2}"
MAXIMUM_FRAME_AGE_SEC="${MAXIMUM_FRAME_AGE_SEC:-2.0}"
MAXIMUM_TELEMETRY_AGE_SEC="${MAXIMUM_TELEMETRY_AGE_SEC:-3.0}"
REQUEST_TIMEOUT_SEC="${REQUEST_TIMEOUT_SEC:-3.0}"
MAX_FRAMES="${MAX_FRAMES:-}"
MAXIMUM_CONSECUTIVE_FAILURES="${MAXIMUM_CONSECUTIVE_FAILURES:-60}"
SHOW_WINDOW="${SHOW_WINDOW:-false}"
DASHBOARD_HOST="${DASHBOARD_HOST:-127.0.0.1}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8001}"
MISSION_ROUTE_LATEST_OVERRIDE="${MISSION_ROUTE_LATEST_OVERRIDE:-}"
PLANNER_MODE="${PLANNER_MODE:-}"

if [[ "$ENV_BACKEND" == "auto" ]]; then
  if command -v conda >/dev/null 2>&1 \
    && conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    ENV_BACKEND="conda"
  elif [[ -x "$VENV_PATH/bin/python" ]]; then
    ENV_BACKEND="venv"
  else
    echo "ERROR: No SAM-TP environment found; run setup_sam_tp_reproduction.sh first." >&2
    exit 2
  fi
fi
if [[ "$ENV_BACKEND" == "conda" ]]; then
  env_python() {
    conda run --no-capture-output -n "$ENV_NAME" python "$@"
  }
elif [[ "$ENV_BACKEND" == "venv" ]]; then
  env_python() {
    "$VENV_PATH/bin/python" "$@"
  }
else
  echo "ERROR: ENV_BACKEND must be auto, conda, or venv." >&2
  exit 2
fi

arguments=(
  --config "$PROJECT_ROOT/configs/default.yaml"
  --upstream-root "$UPSTREAM_ROOT"
  --model-config "$MODEL_CONFIG"
  --checkpoint "$CHECKPOINT"
  --expected-checkpoint-sha256 "$EXPECTED_CHECKPOINT_SHA256"
  --output-dir "$OUTPUT_DIR"
  --target-fps "$TARGET_FPS"
  --telemetry-hz "$TELEMETRY_HZ"
  --maximum-frame-age-sec "$MAXIMUM_FRAME_AGE_SEC"
  --maximum-telemetry-age-sec "$MAXIMUM_TELEMETRY_AGE_SEC"
  --request-timeout-sec "$REQUEST_TIMEOUT_SEC"
  --maximum-consecutive-failures "$MAXIMUM_CONSECUTIVE_FAILURES"
  --dashboard-host "$DASHBOARD_HOST"
  --dashboard-port "$DASHBOARD_PORT"
)
if [[ -n "$MAX_FRAMES" ]]; then
  arguments+=(--max-frames "$MAX_FRAMES")
fi
if [[ -n "$MISSION_ROUTE_LATEST_OVERRIDE" ]]; then
  arguments+=(--mission-route-latest-override "$MISSION_ROUTE_LATEST_OVERRIDE")
fi
if [[ -n "$PLANNER_MODE" ]]; then
  arguments+=(--planner-mode "$PLANNER_MODE")
fi
if [[ "$SHOW_WINDOW" == "true" ]]; then
  arguments+=(--show-window)
fi

cd "$PROJECT_ROOT"
echo "Starting GET-only SAM-TP SDK shadow dashboard"
echo "Browser-only mode enabled; set SHOW_WINDOW=true for the legacy OpenCV window"
if [[ -n "$PLANNER_MODE" ]]; then
  echo "Planner mode override: $PLANNER_MODE"
fi
if [[ -n "$MISSION_ROUTE_LATEST_OVERRIDE" ]]; then
  echo "Read-only route preview override: latest_scanned_checkpoint=$MISSION_ROUTE_LATEST_OVERRIDE"
fi
echo "No /control or mission endpoint will be called"
env_python training/run_sam_tp_sdk_shadow.py "${arguments[@]}" "$@"
