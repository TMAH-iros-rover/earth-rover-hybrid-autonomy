#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"
UPSTREAM_ROOT="${UPSTREAM_ROOT:-$WORKSPACE_ROOT/external/GENIE-SAMTP}"
ENV_NAME="${ENV_NAME:-sam_tp_repro}"
ENV_BACKEND="${ENV_BACKEND:-auto}"
VENV_PATH="${VENV_PATH:-$WORKSPACE_ROOT/external/venvs/$ENV_NAME}"
PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-python3}"
UPSTREAM_URL="https://github.com/jiaming-ai/GENIE-SAMTP.git"
UPSTREAM_BRANCH="master"
UPSTREAM_COMMIT="728aee296cf44288356de683b1948f18b05917d6"
CHECKPOINT_RELATIVE="sam2_logs/configs/sam2.1_training_tiny/sam2_training_custom2_freezeNoneNone_f57.yaml/checkpoints/checkpoint_2.pt"
TRANSFORMERS_VERSION="5.4.0"
CHECKPOINT_URL="https://drive.google.com/drive/folders/190yHH-TcfQVoByZeB1809sPIR62CsBD1?dmr=1&ec=wgc-drive-hero-goto"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi is unavailable; this Dell reproduction requires an NVIDIA GPU." >&2
  exit 2
fi
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

if [[ "$ENV_BACKEND" == "auto" ]]; then
  if command -v conda >/dev/null 2>&1; then
    ENV_BACKEND="conda"
  else
    ENV_BACKEND="venv"
  fi
fi
if [[ "$ENV_BACKEND" != "conda" && "$ENV_BACKEND" != "venv" ]]; then
  echo "ERROR: ENV_BACKEND must be auto, conda, or venv." >&2
  exit 2
fi
if [[ "$ENV_BACKEND" == "conda" ]] && ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: ENV_BACKEND=conda was requested but conda is unavailable." >&2
  exit 2
fi
if [[ "$ENV_BACKEND" == "venv" ]] && ! command -v "$PYTHON_BOOTSTRAP" >/dev/null 2>&1; then
  echo "ERROR: Python bootstrap executable is unavailable: $PYTHON_BOOTSTRAP" >&2
  exit 2
fi

if [[ ! -d "$UPSTREAM_ROOT/.git" ]]; then
  if [[ -e "$UPSTREAM_ROOT" ]]; then
    echo "ERROR: Non-Git path already exists: $UPSTREAM_ROOT" >&2
    exit 2
  fi
  mkdir -p "$(dirname "$UPSTREAM_ROOT")"
  git clone --branch "$UPSTREAM_BRANCH" "$UPSTREAM_URL" "$UPSTREAM_ROOT"
fi

actual_remote="$(git -C "$UPSTREAM_ROOT" remote get-url origin)"
if [[ "$actual_remote" != "$UPSTREAM_URL" ]]; then
  echo "ERROR: Unexpected upstream remote: $actual_remote" >&2
  exit 2
fi
if [[ -n "$(git -C "$UPSTREAM_ROOT" status --porcelain)" ]]; then
  echo "ERROR: Upstream checkout is dirty; preserve or remove its changes manually." >&2
  exit 2
fi
git -C "$UPSTREAM_ROOT" fetch origin "$UPSTREAM_BRANCH"
git -C "$UPSTREAM_ROOT" checkout --detach "$UPSTREAM_COMMIT"
if [[ "$(git -C "$UPSTREAM_ROOT" rev-parse HEAD)" != "$UPSTREAM_COMMIT" ]]; then
  echo "ERROR: Could not pin the official upstream commit." >&2
  exit 2
fi

if [[ "$ENV_BACKEND" == "conda" ]]; then
  if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    conda env create --name "$ENV_NAME" --file "$UPSTREAM_ROOT/environment.yml"
  fi
  env_python() {
    conda run --no-capture-output -n "$ENV_NAME" python "$@"
  }
else
  "$PYTHON_BOOTSTRAP" -c \
    "import sys; assert sys.version_info[:2] == (3, 10), sys.version"
  if [[ ! -x "$VENV_PATH/bin/python" ]]; then
    if ! "$PYTHON_BOOTSTRAP" -m venv "$VENV_PATH"; then
      echo "ERROR: Could not create venv. On Ubuntu 22.04 install python3.10-venv, then rerun." >&2
      exit 2
    fi
  fi
  env_python() {
    "$VENV_PATH/bin/python" "$@"
  }
  if ! env_python -m pip --version >/dev/null 2>&1; then
    echo "pip is missing from the existing SAM-TP venv; attempting ensurepip recovery."
    if ! env_python -m ensurepip --upgrade; then
      cat >&2 <<EOF
ERROR: The SAM-TP venv exists but cannot bootstrap pip.
On Ubuntu 22.04 install the matching venv package, then rerun:

sudo apt update
sudo apt install -y python3.10-venv
./scripts/setup_sam_tp_reproduction.sh
EOF
      exit 2
    fi
  fi
  env_python -m pip install -r "$UPSTREAM_ROOT/requirements.txt"
fi
env_python -c "import sys; assert sys.version_info[:2] == (3, 10), sys.version"
env_python -m pip install "transformers==$TRANSFORMERS_VERSION"
env_python -c \
  "import transformers; assert transformers.__version__ == '$TRANSFORMERS_VERSION', transformers.__version__"

if ! env_python -c "import torch, torchvision" >/dev/null 2>&1; then
  if [[ "${INSTALL_TORCH:-false}" != "true" ]]; then
    cat >&2 <<EOF
ERROR: PyTorch is absent from the independent '$ENV_NAME' $ENV_BACKEND environment.
The official upstream environment definitions intentionally omit PyTorch.

Verify the Dell driver first with: nvidia-smi
Then rerun explicitly, without changing the existing Earth Rover environment:

INSTALL_TORCH=true TORCH_VERSION=2.7.1 TORCHVISION_VERSION=0.22.1 \
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu118 \
  ./scripts/setup_sam_tp_reproduction.sh
EOF
    exit 3
  fi
  TORCH_VERSION="${TORCH_VERSION:-2.7.1}"
  TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.22.1}"
  TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu118}"
  env_python -m pip install \
    "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" \
    --index-url "$TORCH_INDEX_URL"
fi

env_python -m pip install -e "$UPSTREAM_ROOT"
env_python -m pip install "pytest==8.4.2"

checkpoint="$UPSTREAM_ROOT/$CHECKPOINT_RELATIVE"
if [[ ! -f "$checkpoint" ]]; then
  cat >&2 <<EOF
SAM-TP environment and upstream checkout are ready.
The official checkpoint requires manual Google Drive download:

URL: $CHECKPOINT_URL
Expected filename: checkpoint_2.pt
Place it at:
$checkpoint

After placement, verify it with:
sha256sum "$checkpoint"

Then rerun this script and continue with scripts/run_sam_tp_reproduction.sh.
EOF
  exit 4
fi

echo "SAM-TP setup ready"
echo "Environment backend: $ENV_BACKEND"
if [[ "$ENV_BACKEND" == "conda" ]]; then
  echo "Environment: $ENV_NAME"
else
  echo "Environment: $VENV_PATH"
fi
echo "Upstream: $UPSTREAM_ROOT"
echo "Commit: $(git -C "$UPSTREAM_ROOT" rev-parse HEAD)"
echo "Checkpoint: $checkpoint"
