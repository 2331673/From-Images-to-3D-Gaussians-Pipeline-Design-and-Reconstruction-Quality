# ACP execution template

These scripts are adapted from the worker, preflight and submission implementation used for the recorded batch. Account identifiers, resource addresses, image names and local paths have been removed. The parent `setup_environment.sh` and `run_experiments.sh` provide the portable source-build route.

Copy `acp.env.example` to a local `*.local.env` file, fill in your own resource configuration, and source that file. Set `ACP_PROFILE` to an existing local TOML profile containing `username`, `access_key_id` and `access_key_secret`. The profile username must match `ACP_USERNAME`. Keep the profile and the filled environment file outside Git; the template contains no working account or endpoint.

```bash
source /path/to/acp.local.env
python scripts/acp/acp_job.py submit --name pt-3dcv-example \
  --worker scripts/acp/acp_gsplat_batch.sh --dry-run
```

The dry run builds a request without reading credentials or contacting ACP. Its output contains your local resource configuration and must not be published. Remove `--dry-run` only when you intend to allocate a GPU. Requests and runtime logs stay in ignored local `runs/` and `experiments/` directories.

The recorded worker reuses four SHA256-verified wheels built for Python 3.10, NVIDIA torch 24.04, CUDA 12.4, compute capability 9.0. Those binaries and offline package caches remain local. Set `ACP_CACHE_ROOT` to that cache directory when reusing it; the worker reads the wheel checksums from `provenance/compiled_wheels.json`. Other machines should rebuild from the vendored sources. The parameterized launcher has been checked with dry runs; the recorded GPU results precede this privacy refactor.
