#!/usr/bin/env python3
"""Verify and extract the bundled source data and model artifacts."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]

def main():
    manifest = json.loads((ROOT / 'archives/manifest.json').read_text())
    for entry in manifest['archives']:
        archive = ROOT / entry['path']
        digest = hashlib.sha256()
        with archive.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != entry['sha256']:
            raise ValueError(f'Checksum mismatch: {entry["path"]}')
        with zipfile.ZipFile(archive) as bundle:
            for info in bundle.infolist():
                destination = (ROOT / info.filename).resolve()
                if not destination.is_relative_to(ROOT):
                    raise ValueError(f'Unsafe archive entry: {info.filename}')
            bundle.extractall(ROOT)
        print(f'Extracted {entry["path"]}: {entry["files"]} files')
    print('Input data, sources, and experiments are ready.')

if __name__ == '__main__':
    main()
