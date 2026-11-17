#!/usr/bin/env bash
# Run inside a CUDA-devel image with a working matched torch/torchvision pair.
# The tested environment is Python 3.10, NVIDIA torch 24.04 and CUDA 12.4.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PIP_CONFIG_FILE=/dev/null PIP_INDEX_URL=https://pypi.org/simple
unset PIP_EXTRA_INDEX_URL
export PIP_CACHE_DIR="$root/.cache/pip" XDG_CACHE_HOME="$root/.cache"
export PYTHONDONTWRITEBYTECODE=1 MAX_JOBS=${MAX_JOBS:-4}
mkdir -p "$root/.cache"
python3 -m pip install --target "$root/.cache/bootstrap" virtualenv==20.26.6
PYTHONPATH="$root/.cache/bootstrap" python3 -m virtualenv --system-site-packages \
  --app-data "$root/.cache/virtualenv" --no-periodic-update "$root/.venv"
py="$root/.venv/bin/python"
"$py" - "$root/.cache/torch-constraints.txt" <<'PY'
import sys
from pathlib import Path
from importlib.metadata import version
import torch, torchvision
assert torch.cuda.is_available(), 'Working CUDA torch is required in the base image'
Path(sys.argv[1]).write_text('torch=='+version('torch')+'\ntorchvision=='+version('torchvision')+'\n')
print('Base torch',torch.__version__,'CUDA',torch.version.cuda)
PY
"$py" -m pip install --pre -c "$root/.cache/torch-constraints.txt" -r "$root/requirements-runtime.lock"
"$py" -m pip install --no-build-isolation --no-deps \
  "$root/third_party/pycolmap_legacy" "$root/third_party/nerfview" \
  "$root/third_party/fused_ssim" "$root/third_party/gsplat"
"$py" -c 'import torch, gsplat, pycolmap, fused_ssim; assert hasattr(pycolmap,"SceneManager"); print(torch.__version__,gsplat.__version__)'
"$py" -m pip freeze --all > "$root/.cache/environment.freeze.txt"
