#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-$HOME/datasets/experiments/traversability_pilot_v1}"
export HF_HOME="${HF_HOME:-$HOME/datasets/generated/huggingface}"

if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON=python
else
    echo "ERROR: Python 3 was not found." >&2
    exit 1
fi

mkdir -p "$EXPERIMENT_DIR"
mkdir -p "$HF_HOME"
cd "$ROOT_DIR"

before="$($PYTHON - <<'PY'
import json
import torch
import torchvision
print(json.dumps({"torch": torch.__version__, "torchvision": torchvision.__version__, "cuda": torch.cuda.is_available()}))
PY
)"
echo "PyTorch before install: $before"

echo "[1/3] Resolving optional segmentation dependencies without installing"
"$PYTHON" -m pip install --dry-run --report "$EXPERIMENT_DIR/pip_resolution.json" \
    --upgrade-strategy only-if-needed -r requirements-segmentation.txt
"$PYTHON" - "$EXPERIMENT_DIR/pip_resolution.json" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
planned = {
    item.get("metadata", {}).get("name", "").lower().replace("_", "-")
    for item in report.get("install", [])
}
protected = planned & {"torch", "torchvision"}
if protected:
    raise SystemExit(f"Refusing dependency install because pip would modify: {sorted(protected)}")
print(f"Planned package changes: {sorted(name for name in planned if name)}")
PY

echo "[2/3] Installing the pinned optional segmentation requirement"
"$PYTHON" -m pip install --upgrade-strategy only-if-needed -r requirements-segmentation.txt

echo "[3/3] Verifying that PyTorch/CUDA did not change"
"$PYTHON" - "$before" "$EXPERIMENT_DIR" <<'PY'
import importlib.metadata
import json
import sys
from pathlib import Path

import torch
import torchvision

before = json.loads(sys.argv[1])
after = {
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "cuda": torch.cuda.is_available(),
}
if before != after:
    raise SystemExit(f"PyTorch/CUDA environment changed: before={before}, after={after}")

packages = ("torch", "torchvision", "transformers", "tokenizers", "safetensors", "huggingface-hub", "Pillow")
versions = {package: importlib.metadata.version(package) for package in packages}
report = {"before": before, "after": after, "installed_packages": versions}
path = Path(sys.argv[2]) / "installed_versions.json"
path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2, sort_keys=True))
print(f"Dependency reports: {path.parent}")
PY
