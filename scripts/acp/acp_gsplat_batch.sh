#!/usr/bin/env bash
# One H100, a frozen gsplat release, one known-pose object, no AFS writes.
set -euo pipefail
project=${ACP_PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
export ACP_PROJECT_ROOT="$project" GS_PROJECT_ROOT="$project"
cache_root=${ACP_CACHE_ROOT:-$project}
export ACP_CACHE_ROOT="$cache_root"
run_name=${1:-gsplat_batch_20261104}
[[ "$run_name" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
experiment="$project/experiments/$run_name"
scratch=/tmp/3dcv-gsplat-batch
mkdir -p "$experiment" "$scratch" "$cache_root/cache/torch" "$cache_root/cache/pip"
exec > >(tee -a "$experiment/worker.log") 2>&1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PIP_CACHE_DIR="$scratch/pip_cache"
export PIP_CONFIG_FILE=/dev/null PIP_INDEX_URL=https://pypi.org/simple
export PIP_DEFAULT_TIMEOUT=30 PIP_RETRIES=2
export PIP_NO_INDEX=1 PIP_FIND_LINKS="$scratch/wheels"
unset PIP_EXTRA_INDEX_URL PIP_TRUSTED_HOST
export TORCH_HOME="$cache_root/cache/torch"
export TORCH_EXTENSIONS_DIR="$scratch/torch_extensions"
export CUDA_CACHE_PATH="$scratch/cuda_cache"
export XDG_CACHE_HOME="$scratch/cache"
export MPLCONFIGDIR="$scratch/matplotlib"
export CUDA_CACHE_DISABLE=0 MAX_JOBS=4 OMP_NUM_THREADS=4
export TORCH_CUDA_ARCH_LIST=9.0
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export EXPERIMENT_DIR="$experiment"
stage=bootstrap
finish() {
  exit_code=$?
  trap - EXIT
  python3 - "$exit_code" "$stage" <<'PY'
import json, os, sys
from datetime import datetime, timezone
from pathlib import Path
out = Path(os.environ['EXPERIMENT_DIR'])
data = {'exit_code': int(sys.argv[1]), 'last_stage': sys.argv[2],
        'finished_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'complete' if sys.argv[1] == '0' and sys.argv[2] == 'complete' else 'failed'}
(out / 'worker_status.json').write_text(json.dumps(data, indent=2) + '\n')
print(json.dumps(data), flush=True)
PY
  exit "$exit_code"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
mark_stage() {
  stage=$1
  python3 - "$stage" <<'PY'
import json, os, sys
from datetime import datetime, timezone
from pathlib import Path
data = {'stage': sys.argv[1], 'updated_at_utc': datetime.now(timezone.utc).isoformat()}
Path(os.environ['EXPERIMENT_DIR'], 'worker_progress.json').write_text(json.dumps(data)+'\n')
print(json.dumps(data), flush=True)
PY
}
mark_stage bootstrap
date -u
python3 "$project/scripts/acp/acp_preflight.py"
mkdir -p "$scratch/wheels"
cp "$cache_root"/wheels/*.whl "$scratch/wheels/"

# Preserve the image's matched NVIDIA torch/torchvision pair in an isolated venv.
python3 -m pip install --disable-pip-version-check --target "$scratch/bootstrap" virtualenv==20.26.6
PYTHONPATH="$scratch/bootstrap" python3 -m virtualenv --system-site-packages \
  --app-data "$scratch/virtualenv_appdata" --no-periodic-update "$scratch/venv"
py="$scratch/venv/bin/python"
"$py" - <<'PY'
from pathlib import Path
from importlib.metadata import version
import torch, torchvision
Path('/tmp/3dcv-gsplat-batch/torch-constraints.txt').write_text(
    'torch==' + version('torch') + '\ntorchvision==' + version('torchvision') + '\n')
assert torch.cuda.is_available(), 'GPU unavailable'
print('Using image torch', torch.__version__, 'CUDA', torch.version.cuda, flush=True)
PY
mark_stage python_dependencies
# NVIDIA's matched image reports torch as 2.3.0a0; allow that prerelease.
"$py" -m pip install --pre --disable-pip-version-check -c "$scratch/torch-constraints.txt" \
  'numpy==1.26.4' 'viser==0.2.23' 'imageio==2.36.1' 'imageio-ffmpeg==0.5.1' \
  'scikit-learn==1.5.2' 'tqdm==4.67.1' 'torchmetrics==1.4.3' \
  'opencv-python-headless==4.10.0.84' 'tyro==0.8.14' 'Pillow==11.0.0' \
  'tensorboard==2.18.0' 'tensorly==0.9.0' 'PyYAML==6.0.2' \
  'matplotlib==3.9.2' 'splines==0.3.2' 'ninja==1.11.1.1' \
  'jaxtyping==0.2.36' 'rich==13.9.4' 'typing_extensions==4.12.2' 'plyfile==1.1'

# Build locally; preserve all authoritative sources and outputs in AOSS.
mark_stage stage_sources
mkdir -p "$scratch/src"
cp -r "$project/third_party/gsplat" "$scratch/src/"
cp -r "$project/third_party/pycolmap_legacy" "$scratch/src/"
cp -r "$project/third_party/nerfview" "$scratch/src/"
cp -r "$project/third_party/fused_ssim" "$scratch/src/"
mark_stage install_verified_extensions
"$py" - <<'PYCODE'
import hashlib,json,os,shutil,torch
from pathlib import Path
root=Path(os.environ['ACP_CACHE_ROOT'])
receipt=json.loads((Path(os.environ['ACP_PROJECT_ROOT'])/'provenance/compiled_wheels.json').read_text())
assert torch.__version__==receipt['runtime']['torch']
assert torch.version.cuda==receipt['runtime']['torch_cuda']
assert torch.cuda.get_device_capability()==(9,0)
out=Path('/tmp/3dcv-gsplat-batch/built_wheels');out.mkdir(exist_ok=True)
for row in receipt['wheels']:
 src=root/row['file']
 h=hashlib.sha256(src.read_bytes()).hexdigest()
 assert h==row['sha256'],src
 shutil.copy2(src,out/src.name)
print('Verified four previously compiled wheels against SHA256 and runtime',flush=True)
PYCODE
"$py" -m pip install --no-deps "$scratch"/built_wheels/*.whl
"$py" -m pip freeze --all > "$experiment/environment.freeze.txt"
"$py" - <<'PY'
import json, os, sys, torch, numpy, gsplat, pycolmap, fused_ssim
from pathlib import Path
data = {'python': sys.version, 'torch': torch.__version__, 'numpy': numpy.__version__,
        'gsplat': gsplat.__version__, 'torch_cuda': torch.version.cuda,
        'legacy_colmap_scene_manager': hasattr(pycolmap, 'SceneManager'),
        'gpu': torch.cuda.get_device_name(0)}
Path(os.environ['EXPERIMENT_DIR'], 'runtime.json').write_text(json.dumps(data, indent=2)+'\n')
print(data, flush=True)
PY
mark_stage batch_train
"$py" "$project/scripts/run_batch.py" --output "$experiment" --scratch "$scratch" \
  --trainer "$scratch/src/gsplat/examples/simple_trainer.py" --steps 6000
mark_stage complete
