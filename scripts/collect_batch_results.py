#!/usr/bin/env python3
"""Collect measured batch metrics and artifacts; no document generation."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import time
from public_records import sanitize_results

NAMES = ['木箱', '紫色椅子', '白色圆柱结构', '紫色缠绕结构', '蓝色扁平板',
         '锤形物体', '深色圆柱', '编织篮', '咖啡售卖机', '灯柱模型']


def read_json(path):
    for attempt in range(10):
        try:
            return json.loads(path.read_text())
        except (ValueError, FileNotFoundError):
            if attempt == 9:
                raise
            time.sleep(.5)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--repository', type=Path, required=True)
    args = parser.parse_args()
    exp, repo = args.experiment, args.repository
    plan = read_json(exp/'run_plan.json')
    rows = []
    for job in plan['jobs']:
        run = exp/job['run_id']
        status = read_json(run/'status.json')
        assert status['status'] == 'complete', status
        row = dict(status)
        row.update(name=NAMES[job['number']-1], **status['metrics'])
        row['curve'] = {str(step): read_json(run/'model/stats'/f'val_step{step-1:04d}.json')
                        for step in [1, 500, 2000, 6000]}
        row['training'] = read_json(run/'model/stats/train_step5999_rank0.json')
        rows.append(row)
        target = repo/'results'/job['run_id']
        target.mkdir(parents=True, exist_ok=True)
        for name in ['orbit.gif', 'orbit_metadata.json', 'orbit_cameras.json', 'status.json',
                     'command.json', 'foreground_metrics.json']:
            shutil.copy2(run/name, target/name)
        shutil.copy2(run/'model/cfg.yml', target/'cfg.yml')
        shutil.copytree(run/'model/stats', target/'stats', dirs_exist_ok=True)
        hashes = {str(f.relative_to(run)): {'bytes': f.stat().st_size, 'sha256': digest(f)}
                  for f in [run/'model/ckpts/ckpt_5999_rank0.pt', run/'model/ply/point_cloud_5999.ply']}
        (target/'model_artifact_checksums.json').write_text(json.dumps(hashes, indent=2)+'\n')
    primary = sorted([x for x in rows if x['kind'] == 'primary'], key=lambda x: x['number'])
    ablations = [x for x in rows if x['kind'] == 'initialization_ablation']
    assert len(primary) == 10 and len(ablations) == 2
    result = {'scope': '10 known-pose objects and 2 matched-count initialization comparisons',
              'steps': 6000, 'results': rows}
    (repo/'results/summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    fields = ['number', 'name', 'uid', 'kind', 'init_type', 'sift_points', 'init_num_pts',
              'psnr', 'ssim', 'lpips', 'fg_psnr_8bit', 'num_GS', 'elapsed_seconds']
    with (repo/'results/metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore', lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)

    sanitize_results(repo, exp)
    print(json.dumps({"runs": len(rows), "primary": len(primary), "ablations": len(ablations)}))

if __name__ == "__main__":
    main()
