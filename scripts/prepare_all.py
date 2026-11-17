#!/usr/bin/env python3
"""Rebuild all ten COLMAP-format scenes from the bundled RGBA images and poses."""
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
for obj in json.loads((root/'manifests/candidates.json').read_text())['objects']:
    subprocess.run([sys.executable, str(root/'scripts/prepare_smoke_scene.py'),
                    '--uid', obj['uid'], '--output', str(root/'data/objects'/obj['uid']),
                    '--allow-sparse'], cwd=root, check=True)
