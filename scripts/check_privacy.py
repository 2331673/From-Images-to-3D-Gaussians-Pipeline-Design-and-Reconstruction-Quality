#!/usr/bin/env python3
"""Check tracked or staged contents; report filenames/rules, never matched values."""
import argparse
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SECRET_PATTERNS = {
    'github-token': rb'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b',
    'private-key': rb'-----BEGIN (?:RSA |DSA |EC |OPENSSH |ENCRYPTED )?PRIVATE KEY-----',
    'openai-key': rb'\bsk-(?:proj|svcacct)-[A-Za-z0-9_-]{30,}',
    'aws-access-key': rb'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
    'slack-token': rb'\bxox[baprs]-[A-Za-z0-9-]{20,}',
}
PRIVATE_PATTERNS = {
    'personal-mount-path': rb'/(?:mnt/(?:afs|aoss)/(?:groups|dataset|[0-9]+)|home/[^/\s]+|root)/',
    'subscription-id': rb'/subscriptions/[0-9a-f]{8}-[0-9a-f-]{27,}',
    'internal-image-registry': rb'registry\.[a-z0-9.-]*sensecore\.(?:cn|com)/',
}


def findings(path, content):
    name = Path(path).name
    if (name in {'hosts.yml', '.git-credentials', 'id_rsa', 'id_ed25519',
                 'credentials.toml', 'acp-credentials.toml'}
            or (name.startswith('.env') and name != '.env.example')
            or name.endswith(('.local.env', '.local.toml', '.pem', '.key', '.bundle'))
            or '.private' in Path(path).parts):
        yield 'local-private-file'
    for rule, pattern in SECRET_PATTERNS.items():
        if re.search(pattern, content):
            yield rule
    # Preserve upstream examples/licenses; secret detection still applies there.
    if not path.startswith('third_party/'):
        for rule, pattern in PRIVATE_PATTERNS.items():
            if re.search(pattern, content):
                yield rule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--staged', action='store_true')
    args = parser.parse_args()
    paths = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    matches = []
    for path in filter(None, paths):
        content = (subprocess.check_output(['git', 'show', ':' + path], cwd=ROOT)
                   if args.staged else (ROOT / path).read_bytes())
        matches.extend({'file': path, 'rule': rule} for rule in findings(path, content))
    print(json.dumps({'files_checked': len([p for p in paths if p]),
                      'findings': matches}, ensure_ascii=False, indent=2))
    raise SystemExit(bool(matches))


if __name__ == '__main__':
    main()
