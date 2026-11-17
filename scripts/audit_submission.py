#!/usr/bin/env python3
"""Audit saved evidence and export final track correspondences; no GPU required."""
import csv
import hashlib
import itertools
import json
from pathlib import Path
from statistics import mean
from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def main():
    runs = read(ROOT / 'results/summary.json')['results']
    primary = [r for r in runs if r['kind'] == 'primary']
    assert len(primary) == 10 and len({r['uid'] for r in primary}) == 10
    checks = []
    for r in primary:
        scene = ROOT / 'data/objects' / r['uid']
        result = ROOT / 'results' / r['run_id']
        provenance = read(scene / 'provenance.json')
        train, test = set(provenance['train_images']), set(provenance['test_images'])
        assert len(train) == 26 and len(test) == 4 and not train & test
        assert test == {'000.png', '008.png', '016.png', '024.png'}
        pairs = read(scene / 'matches/pair_counts.json')
        assert len(pairs) == 325
        assert len({(p['image_a'], p['image_b']) for p in pairs}) == 325
        assert all(p['image_a'] in train and p['image_b'] in train for p in pairs)
        lines = [s for s in (scene/'sparse/0/images.txt').read_text().splitlines()
                 if not s.startswith('#')]
        assert len(lines) == 60
        images = {}
        for header, obs in zip(lines[::2], lines[1::2]):
            h, values = header.split(), obs.split()
            coords = [(float(values[j]), float(values[j+1]), int(values[j+2]))
                      for j in range(0, len(values), 3)]
            images[int(h[0])] = (h[9], coords)
            if h[9] in test:
                assert not coords
        rows = []
        points = [s.split() for s in (scene/'sparse/0/points3D.txt').read_text().splitlines()
                  if s and not s.startswith('#')]
        assert len(points) == r['sift_points']
        for p in points:
            pid = int(p[0])
            track = list(zip(map(int, p[8::2]), map(int, p[9::2])))
            assert len(track) >= 2 and len({i for i, _ in track}) == len(track)
            for i, k in track:
                assert images[i][0] in train and images[i][1][k][2] == pid
            for (i, ki), (j, kj) in itertools.combinations(sorted(track), 2):
                a, b = images[i], images[j]
                rows.append([pid, a[0], b[0], ki, kj, *a[1][ki][:2], *b[1][kj][:2]])
        with (scene/'matches/retained_correspondences.csv').open('w') as f:
            writer = csv.writer(f)
            writer.writerow(['point3D_id','image_a','image_b','keypoint_a','keypoint_b',
                             'x_a','y_a','x_b','y_b'])
            writer.writerows(rows)
        gif = result / 'orbit.gif'
        with Image.open(gif) as im:
            assert im.n_frames == 96 and im.size == (256, 256)
            for frame in range(im.n_frames):
                im.seek(frame)
                im.convert('RGB').load()
        stats = read(result/'stats/val_step5999.json')
        for key in ['psnr','ssim','lpips']:
            assert abs(stats[key]-r[key]) < 1e-9
        assert read(result/'status.json')['status'] == 'complete'
        fg_values, full_values = [], []
        for index, name in enumerate(sorted(test)):
            pair = np.asarray(Image.open(result/'eval'/f'{index:04d}.png').convert('RGB'), dtype=float)/255
            assert pair.shape == (256, 512, 3)
            error = (pair[:, :256]-pair[:, 256:])**2
            mask = np.asarray(Image.open(ROOT/'data/raw'/r['uid']/name))[:, :, 3] >= 128
            fg_values.append(float(-10*np.log10(error[mask].mean())))
            full_values.append(float(-10*np.log10(error.mean())))
        assert abs(mean(fg_values)-r['fg_psnr_8bit']) < 1e-6
        checks.append(dict(number=r['number'], uid=r['uid'], pairs=325,
                           points=len(points), retained_track_pairs=len(rows), gif_frames=96,
                           full_psnr_8bit=mean(full_values), fg_psnr_8bit=mean(fg_values),
                           gif_sha256=hashlib.sha256(gif.read_bytes()).hexdigest()))
    report = dict(scope='Saved artifact audit; does not rerun GPU training',
                  objects=checks, primary_runs=len(primary),
                  ablation_runs=sum(r['kind']=='initialization_ablation' for r in runs),
                  full_psnr_8bit_macro=mean(c['full_psnr_8bit'] for c in checks),
                  averages={k:mean(r[k] for r in primary)
                            for k in ['psnr','ssim','lpips','fg_psnr_8bit']},
                  psnr_declines_2000_to_6000=[r['number'] for r in primary
                      if r['curve']['6000']['psnr'] < r['curve']['2000']['psnr']])
    (ROOT/'results/submission_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
