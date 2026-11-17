#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export GS_PROJECT_ROOT="$root" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export TORCH_HOME="$root/.cache/torch" TORCH_EXTENSIONS_DIR="$root/.cache/torch_extensions"
export CUDA_CACHE_PATH="$root/.cache/cuda" MPLCONFIGDIR="$root/.cache/matplotlib"
export XDG_CACHE_HOME="$root/.cache" OMP_NUM_THREADS=4
run_name=${1:-batch_local}
[[ "$run_name" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
"$root/.venv/bin/python" "$root/scripts/run_batch.py" \
  --output "$root/experiments/$run_name" --scratch "$root/.cache/runtime" \
  --trainer "$root/third_party/gsplat/examples/simple_trainer.py" --steps 6000
