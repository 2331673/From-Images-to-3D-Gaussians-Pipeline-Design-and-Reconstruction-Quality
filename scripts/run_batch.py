#!/usr/bin/env python3
"""Train the frozen ten-object set and two controlled initialization comparisons."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(os.environ.get('GS_PROJECT_ROOT', Path(__file__).resolve().parents[1]))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def png_foreground_metrics(scene, result, steps):
    """Additional foreground metric from saved 8-bit eval pairs, not float PSNR."""
    import numpy as np
    from PIL import Image
    rows = []
    for i, index in enumerate([0, 8, 16, 24]):
        name = f'{index:03}.png'
        pair = np.asarray(Image.open(result / 'model/renders' / f'val_step{steps-1}_{i:04d}.png').convert('RGB'), dtype=np.float64) / 255.
        truth, pred = pair[:, :256], pair[:, 256:]
        mask = np.asarray(Image.open(scene / 'masks' / name)) >= 128
        assert mask.any(), f'Empty foreground in {scene.name}/{name}'
        mse = float(np.mean((truth[mask] - pred[mask]) ** 2))
        whole_mse = float(np.mean((truth - pred) ** 2))
        rows.append({'image': name, 'foreground_fraction': float(mask.mean()),
                     'fg_psnr_8bit': float(-10 * np.log10(max(mse, 1e-12))),
                     'full_psnr_8bit': float(-10 * np.log10(max(whole_mse, 1e-12)))})
    return {'definition': 'Mean per-image RGB PSNR from 8-bit saved eval pairs, evaluated where source alpha >= 128; masks used only for evaluation.',
            'views': rows, 'fg_psnr_8bit': float(np.mean([x['fg_psnr_8bit'] for x in rows])),
            'full_psnr_8bit': float(np.mean([x['full_psnr_8bit'] for x in rows]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, default=Path('/tmp/3dcv-gsplat-batch'))
    parser.add_argument('--trainer', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=6000)
    parser.add_argument('--only', nargs='*', help='Optional exact run IDs for targeted retries')
    args = parser.parse_args()
    assert args.output.resolve().is_relative_to(ROOT.resolve())
    candidates = json.loads((ROOT / 'manifests/candidates.json').read_text())['objects']
    jobs = []
    for number, obj in enumerate(candidates, 1):
        uid = obj['uid']
        stats = json.loads((ROOT / 'data/objects' / uid / 'matches/stats.json').read_text())
        init = stats['initialization_recommendation']
        jobs.append({'run_id': f'{number:02d}_{uid}', 'number': number, 'uid': uid,
                     'kind': 'primary', 'init_type': init, 'sift_points': stats['num_sparse_points'],
                     'init_num_pts': 10000 if init == 'random' else stats['num_sparse_points']})
        if number in [1, 9]:
            jobs.append({'run_id': f'ablation_random_{number:02d}', 'number': number, 'uid': uid,
                         'kind': 'initialization_ablation', 'init_type': 'random',
                         'sift_points': stats['num_sparse_points'], 'init_num_pts': stats['num_sparse_points']})
    if args.only:
        requested = set(args.only)
        assert requested <= {x['run_id'] for x in jobs}, requested
        jobs = [x for x in jobs if x['run_id'] in requested]
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / 'run_plan.json', {'steps': args.steps, 'seed': 42,
               'test_every': 8, 'train_views': 26, 'test_views': 4,
               'random_init_extent_scene_scale_multiplier': 0.3,
               'fallback_rule': 'Use 10000 random Gaussians when fewer than 20 SIFT points survive; otherwise SIFT initialization.',
               'ablation_rule': 'Objects 01 and 09: replace SIFT positions/colors with seeded random positions/colors, keeping initial point count and all other settings.',
               'jobs': jobs})
    outcomes = []
    for job in jobs:
        destination = args.output / job['run_id']
        destination.mkdir(parents=True, exist_ok=True)
        scene = ROOT / 'data/objects' / job['uid']
        local_scene = args.scratch / 'scenes' / job['uid']
        if not local_scene.exists():
            shutil.copytree(scene, local_scene)
        status_path = destination / 'status.json'
        previous = json.loads(status_path.read_text()) if status_path.exists() else {}
        if previous.get('status') == 'complete' and (destination/'orbit.gif').exists():
            outcomes.append(previous)
            continue
        started = time.monotonic()
        status = {**job, 'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat()}
        write_json(status_path, status)
        write_json(args.output/'progress.json', {'current': job, 'finished': len(outcomes), 'total': len(jobs)})
        command = [sys.executable, str(args.trainer), 'default', '--data-dir', str(local_scene),
                   '--data-factor', '1', '--result-dir', str(destination/'model'),
                   '--disable-viewer', '--disable-video', '--no-normalize-world-space',
                   '--max-steps', str(args.steps), '--eval-steps', '1', '500', '2000', str(args.steps),
                   '--save-steps', str(args.steps), '--save-ply', '--ply-steps', str(args.steps),
                   '--init-type', job['init_type'], '--init-num-pts', str(job['init_num_pts']),
                   '--init-extent', '0.3']
        write_json(destination/'command.json', {'argv': command, 'cwd': str(args.trainer.parent)})
        try:
            with (destination/'train.log').open('w') as log:
                subprocess.run(command, cwd=args.trainer.parent, stdout=log, stderr=subprocess.STDOUT, check=True)
            with (destination/'render.log').open('w') as log:
                subprocess.run([sys.executable, str(ROOT/'scripts/render_smoke_orbit.py'),
                                '--checkpoint', str(destination/'model/ckpts'/f'ckpt_{args.steps-1}_rank0.pt'),
                                '--config', str(destination/'model/cfg.yml'),
                                '--output', str(destination/'orbit.gif')], stdout=log, stderr=subprocess.STDOUT, check=True)
            metrics = json.loads((destination/'model/stats'/f'val_step{args.steps-1:04d}.json').read_text())
            fg = png_foreground_metrics(scene, destination, args.steps)
            write_json(destination/'foreground_metrics.json', fg)
            status.update(status='complete', metrics=metrics, fg_psnr_8bit=fg['fg_psnr_8bit'],
                          elapsed_seconds=time.monotonic()-started)
        except Exception as error:
            status.update(status='failed', error=f'{type(error).__name__}: {error}', elapsed_seconds=time.monotonic()-started)
        status['finished_at'] = datetime.now(timezone.utc).isoformat()
        write_json(status_path, status)
        outcomes.append(status)
        write_json(args.output/'outcomes.json', outcomes)
        print(json.dumps(status, ensure_ascii=False), flush=True)
    failed = [x['run_id'] for x in outcomes if x['status'] != 'complete']
    write_json(args.output/'progress.json', {'status': 'failed' if failed else 'complete',
               'finished': len(outcomes), 'total': len(jobs), 'failed': failed})
    if failed:
        raise SystemExit(f'Incomplete runs: {failed}')


if __name__ == '__main__':
    main()
