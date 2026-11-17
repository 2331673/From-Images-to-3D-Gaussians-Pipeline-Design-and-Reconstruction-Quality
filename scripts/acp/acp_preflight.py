"""Inspect one ACP GPU worker and persist only non-secret runtime details."""
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get('ACP_PROJECT_ROOT', Path(__file__).resolve().parents[2]))
OUT = ROOT / 'experiments' / 'acp_preflight'
OUT.mkdir(parents=True, exist_ok=True)
result = {
    'checked_at_utc': datetime.now(timezone.utc).isoformat(),
    'python': sys.version,
    'python_executable': sys.executable,
    'platform': platform.platform(),
    'tools': {},
    'packages': {},
}
for name in ['python', 'python3', 'pip', 'nvcc', 'gcc', 'g++', 'cmake', 'git', 'ffmpeg', 'uv', 'conda']:
    result['tools'][name] = shutil.which(name)
result['cuda_directories'] = [str(path) for path in Path('/usr/local').glob('cuda*')]
for name in ['torch', 'torchvision', 'numpy', 'PIL', 'cv2', 'gsplat', 'pycolmap']:
    try:
        module = importlib.import_module(name)
        result['packages'][name] = getattr(module, '__version__', 'importable')
    except Exception as error:
        result['packages'][name] = {'error': str(error)}
try:
    import torch
    result['torch_cuda'] = torch.version.cuda
    result['cuda_available'] = torch.cuda.is_available()
    result['gpus'] = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        result['gpus'].append({'index': index, 'name': props.name,
                               'memory_bytes': props.total_memory,
                               'compute_capability': [props.major, props.minor]})
    if result['cuda_available']:
        value = (torch.ones(16, device='cuda') * 2).sum().item()
        result['cuda_tensor_check'] = value == 32
except Exception as error:
    result['torch_error'] = str(error)
for key, command in {
    'nvidia_smi': ['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
    'nvcc_version': [shutil.which('nvcc') or '/usr/local/cuda/bin/nvcc', '--version'],
    'gxx_version': ['g++', '--version'],
}.items():
    try:
        process = subprocess.run(command, capture_output=True, text=True, timeout=20)
        result[key] = {'returncode': process.returncode,
                       'stdout': process.stdout[-3000:], 'stderr': process.stderr[-1000:]}
    except Exception as error:
        result[key] = {'error': str(error)}
(OUT / 'environment.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2), flush=True)
