#!/usr/bin/env python3
"""Own-account, AOSS-only ACP short jobs; never persist authentication material.

Examples (submit allocates one GPU; dry-run does not contact ACP):
  python acp_job.py resources
  python acp_job.py status --name pt-3dcv-smoke-001
  python acp_job.py submit --name pt-3dcv-smoke-001 --worker scripts/acp/acp_gsplat_batch.sh --dry-run
"""
from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import hmac
import json
import os
import re
import shlex
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlencode

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

ROOT = Path(os.environ.get('ACP_PROJECT_ROOT', Path(__file__).resolve().parents[2])).resolve()
PROFILE = Path(os.environ.get('ACP_PROFILE', ROOT / '.private' / 'acp-credentials.toml'))
USERNAME = os.environ.get('ACP_USERNAME', '')
API = os.environ.get('ACP_API', '').rstrip('/')
JOBS = os.environ.get('ACP_JOBS_PATH', '')
CLUSTER = os.environ.get('ACP_CLUSTER_PATH', '')
SPEC = os.environ.get('ACP_SPEC', '')
IMAGE = os.environ.get('ACP_IMAGE', '')
WORKSPACE = os.environ.get('ACP_WORKSPACE', '')
RESOURCE_POOL = os.environ.get('ACP_RESOURCE_POOL', '')
MOUNT_ENDPOINT = os.environ.get('ACP_MOUNT_ENDPOINT', '')
MOUNT_ZONE = os.environ.get('ACP_MOUNT_ZONE', '')
MOUNT_PATH = os.environ.get('ACP_MOUNT_PATH', '')


def validate_configuration(action: str) -> None:
    required = {'ACP_USERNAME': USERNAME, 'ACP_API': API}
    if action in {'status', 'submit'}:
        required['ACP_JOBS_PATH'] = JOBS
    if action == 'resources':
        required.update(ACP_CLUSTER_PATH=CLUSTER, ACP_WORKSPACE=WORKSPACE,
                        ACP_SPEC=SPEC, ACP_RESOURCE_POOL=RESOURCE_POOL)
    if action == 'submit':
        required.update(ACP_SPEC=SPEC, ACP_IMAGE=IMAGE, ACP_RESOURCE_POOL=RESOURCE_POOL,
                        ACP_MOUNT_ENDPOINT=MOUNT_ENDPOINT, ACP_MOUNT_ZONE=MOUNT_ZONE,
                        ACP_MOUNT_PATH=MOUNT_PATH)
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError('Set local environment variables: ' + ', '.join(missing))
    parsed = urlsplit(API)
    if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError('ACP_API must be an HTTPS endpoint without embedded credentials')
    if action in {'submit', 'status'} and not JOBS.startswith('/'):
        raise ValueError('ACP_JOBS_PATH must be an absolute API path')
    if action == 'resources' and not CLUSTER.startswith('/'):
        raise ValueError('ACP_CLUSTER_PATH must be an absolute API path')


def credentials(profile: Path) -> dict:
    with profile.open('rb') as stream:
        data = tomllib.load(stream)
    if str(data.get('username')) != USERNAME:
        raise ValueError('Profile username must match ACP_USERNAME')
    if not all(data.get(key) for key in ('access_key_id', 'access_key_secret')):
        raise ValueError('Profile is missing ACP authentication fields')
    return data


def request(method: str, path: str, auth: dict, body: dict | None = None) -> dict:
    url = API + path
    parts = urlsplit(url)
    date = datetime.now(timezone.utc).strftime('%a, %d %b %Y %H:%M:%S GMT')
    target = parts.path + (('?' + parts.query) if parts.query else '')
    message = f'date: {date}\nhost: {parts.netloc}\n@request-target: {method.lower()} {target}'
    signature = base64.b64encode(hmac.new(auth['access_key_secret'].encode(),
                                         message.encode(), hashlib.sha256).digest()).decode()
    header = (f'hmac accesskey="{auth["access_key_id"]}", algorithm="hmac-sha256", '
              f'headers="date host @request-target", signature="{signature}"')
    req = urllib.request.Request(url, method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={'Date': date, 'X-Date': date,
                                          'Authorization': header, 'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=40) as response:
        return json.load(response)


def checked_name(name: str) -> str:
    if not re.fullmatch(r'pt-3dcv-[a-z0-9][a-z0-9-]{0,48}', name):
        raise ValueError('Job name must start pt-3dcv- and contain only lower-case letters, digits and hyphens')
    return name


def project_path(path: Path, *, must_exist: bool = False) -> Path:
    path = path.resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError('Worker and run paths must be under ACP_PROJECT_ROOT')
    if must_exist and not path.is_file():
        raise ValueError('Worker script does not exist')
    return path


def safe_job(job: dict) -> dict:
    return {
        'name': job.get('name'), 'display_name': job.get('display_name'),
        'state': job.get('state'), 'owner': job.get('ownership', {}).get('user_name'),
        'quota_type': job.get('scheduling', {}).get('quota_type'),
        'resource_pool': job.get('resource_pool', {}).get('name'),
        'roles': [{'name': role.get('name'), 'replicas': role.get('total_replicas'),
                   'resource_spec': [spec.get('name') for spec in role.get('resource_spec', [])],
                   'image_path': role.get('image_path')} for role in job.get('roles', [])],
    }


def atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    try:
        with temporary.open('x') as stream:
            json.dump(data, stream, indent=2, ensure_ascii=False)
            stream.write('\n')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def make_body(args: argparse.Namespace) -> dict:
    name = checked_name(args.name)
    worker = project_path(args.worker, must_exist=True)
    if not 1 <= args.timeout_seconds <= 3600:
        raise ValueError('Short-job timeout must be 1..3600 seconds')
    worker_args = args.worker_args
    if worker_args[:1] == ['--']:
        worker_args = worker_args[1:]
    interpreter = args.worker_interpreter
    if interpreter == 'auto':
        interpreter = 'python3' if worker.suffix == '.py' else 'bash'
    startup = shlex.join(['timeout', '--signal=TERM', '--kill-after=30',
                         str(args.timeout_seconds), interpreter, str(worker), *worker_args])
    return {
        'name': name, 'display_name': args.display_name or name,
        'framework': 'PYTORCH',
        'roles': [{'name': 'Worker', 'resource_spec': [{'name': SPEC}], 'total_replicas': 1,
                   'image_path': IMAGE, 'startup_script': startup}],
        'mount': [{'type': 'PV_AOSS', 'name': USERNAME, 'mount_path': MOUNT_PATH,
                   'endpoint': MOUNT_ENDPOINT, 'zone': MOUNT_ZONE}],
        'scheduling': {'priority': 'NORMAL', 'quota_type': args.quota.upper()},
        'resource_pool': {'name': RESOURCE_POOL}, 'fault_tolerance': {'backoff_limit': 0},
    }


def assert_same_job(job: dict, body: dict) -> None:
    if job.get('ownership', {}).get('user_name') != USERNAME:
        raise ValueError('Remote job name belongs to a different or unidentified owner')
    if len(job.get('roles', [])) != 1:
        raise ValueError('Remote job has a different role layout')
    role, expected = job['roles'][0], body['roles'][0]
    for key in ('startup_script', 'image_path', 'total_replicas'):
        if role.get(key) != expected[key]:
            raise ValueError(f'Remote job differs in {key}; use a new explicit job name')
    if [spec.get('name') for spec in role.get('resource_spec', [])] != [SPEC]:
        raise ValueError('Remote job has a different resource specification')
    if job.get('scheduling', {}).get('quota_type') != body['scheduling']['quota_type']:
        raise ValueError('Remote job has a different quota type')


def submit(args: argparse.Namespace) -> dict:
    body = make_body(args)
    if args.dry_run:
        return {'dry_run': True, 'request': body}
    auth = credentials(args.profile)
    run_dir = project_path(args.run_dir or ROOT / 'runs' / 'acp' / args.name)
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / '.submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        plan = run_dir / 'job_request.json'
        if plan.exists():
            if json.loads(plan.read_text()) != body:
                raise ValueError('Saved request differs; choose a new job name/run directory')
        else:
            atomic_json(plan, body)
        try:
            job = request('GET', JOBS + '/' + body['name'], auth)
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise
            signed = dict(body)
            signed['metadata'] = {'items': [
                {'key': 'access_key', 'value': auth['access_key_id']},
                {'key': 'secret_key', 'value': auth['access_key_secret']},
            ]}
            try:
                job = request('POST', JOBS + '?enable_queuing=true', auth, signed)
            except urllib.error.HTTPError as error:
                if error.code != 409:
                    raise
                job = request('GET', JOBS + '/' + body['name'], auth)
                assert_same_job(job, body)
            already_exists = False
        else:
            assert_same_job(job, body)
            already_exists = True
        receipt = safe_job(job)
        receipt.update(already_exists=already_exists, run_dir=str(run_dir),
                       observed_at=datetime.now(timezone.utc).isoformat())
        atomic_json(run_dir / 'job_status.json', receipt)
        return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, default=PROFILE)
    commands = parser.add_subparsers(dest='action', required=True)
    create = commands.add_parser('submit', help='Create exactly one short GPU job, or return existing same job')
    create.add_argument('--name', required=True)
    create.add_argument('--worker', type=Path, required=True)
    create.add_argument('--worker-interpreter', choices=['auto', 'bash', 'python', 'python3'], default='auto',
                        help='auto uses python3 for .py files and bash otherwise')
    create.add_argument('--display-name')
    create.add_argument('--run-dir', type=Path)
    create.add_argument('--timeout-seconds', type=int, default=1800)
    create.add_argument('--quota', choices=['reserved', 'spot'], default='reserved')
    create.add_argument('--dry-run', action='store_true')
    create.add_argument('worker_args', nargs=argparse.REMAINDER)
    status = commands.add_parser('status', help='Read a project job; print only safe fields')
    status.add_argument('--name', required=True)
    commands.add_parser('resources', help='Read advertised one-GPU spec; does not report idle capacity')
    args = parser.parse_args()
    try:
        validate_configuration(args.action)
        if args.action == 'submit':
            result = submit(args)
        elif args.action == 'status':
            job = request('GET', JOBS + '/' + checked_name(args.name), credentials(args.profile))
            if job.get('ownership', {}).get('user_name') != USERNAME:
                raise ValueError('Refusing to inspect a job belonging to another user')
            result = safe_job(job)
        else:
            data = request('GET', CLUSTER + '/resourceSpecs?' + urlencode({'workspace_name': WORKSPACE}), credentials(args.profile))
            specs = [item for item in data.get('resource_specs', []) if item.get('name') == SPEC]
            result = {'resource_pool': RESOURCE_POOL, 'idle_capacity_verified': False,
                      'specs': [{'name': item.get('name'),
                                 'cpu': item.get('cpu', {}).get('vcpu_allocatable'),
                                 'memory_gb': item.get('memory', {}).get('allocatable'),
                                 'gpu_type': item.get('device', {}).get('type'),
                                 'gpu_memory_gb': item.get('device', {}).get('memory'),
                                 'gpu_count': item.get('device', {}).get('number'),
                                 'zones': item.get('zones')} for item in specs]}
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except urllib.error.HTTPError as error:
        print(json.dumps({'error': 'ACP request failed', 'http_code': error.code}), file=sys.stderr)
        raise SystemExit(1)
    except (ValueError, OSError) as error:
        # Do not echo API response bodies, request headers, profile contents, or credentials.
        message = str(error) if isinstance(error, ValueError) else type(error).__name__
        print(json.dumps({'error': message}), file=sys.stderr)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
