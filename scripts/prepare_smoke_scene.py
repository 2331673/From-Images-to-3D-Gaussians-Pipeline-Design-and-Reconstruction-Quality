#!/usr/bin/env python3
"""Prepare a known-pose diagnostic COLMAP scene without using source geometry.

Only the 26 training images contribute SIFT observations, tracks, and 3D points.
This is not a COLMAP SfM reconstruction or an evaluation of pose estimation.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import struct
from pathlib import Path
from datetime import datetime, timezone

import cv2
import numpy as np
from PIL import Image
from public_records import sanitize_text

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UID = 'fdecd4570f78412a821d9c41d98b2cfa'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def quaternion(R):
    # Scalar-first Hamilton quaternion; eigenvector formula avoids small traces.
    r00, r01, r02, r10, r11, r12, r20, r21, r22 = R.flat
    K = np.array([[r00-r11-r22, r10+r01, r20+r02, r12-r21],
                  [r10+r01, r11-r00-r22, r21+r12, r20-r02],
                  [r20+r02, r21+r12, r22-r00-r11, r01-r10],
                  [r12-r21, r20-r02, r01-r10, r00+r11+r22]]) / 3.0
    _, v = np.linalg.eigh(K)
    # Above K corresponds to the transpose convention; conjugate to obtain R.
    q = v[:, -1][[3, 0, 1, 2]]
    q[1:] *= -1
    return q if q[0] >= 0 else -q


def rotation(q):
    w, x, y, z = q
    return np.array([[1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
                     [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
                     [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]])


def project(P, X):
    p = P @ np.r_[X, 1.0]
    return p[:2] / p[2], p[2]


def triangulate(observations, xys, projections):
    A = []
    for i, k in observations:
        x, y = xys[i][k]
        P = projections[i]
        A.extend([x * P[2] - P[0], y * P[2] - P[1]])
    _, _, vh = np.linalg.svd(np.asarray(A))
    p = vh[-1]
    if abs(p[3]) < 1e-12:
        return None
    return p[:3] / p[3]


def summary(a):
    a = np.asarray(a, dtype=float)
    if not len(a):
        return {'count': 0}
    return dict(count=len(a), min=float(a.min()), median=float(np.median(a)),
                mean=float(a.mean()), p95=float(np.percentile(a, 95)), max=float(a.max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--uid', default=DEFAULT_UID)
    parser.add_argument('--output', type=Path, default=ROOT / 'data/smoke_vending')
    parser.add_argument('--ratio', type=float, default=0.8)
    parser.add_argument('--max-reprojection-px', type=float, default=1.5)
    parser.add_argument('--max-epipolar-px', type=float, default=1.5)
    parser.add_argument('--min-parallax-deg', type=float, default=2.0)
    parser.add_argument('--allow-sparse', action='store_true',
                        help='Export even with fewer than 20 SIFT points; training must use random initialization')
    args = parser.parse_args()
    args.output = args.output.resolve()
    if not args.output.is_relative_to(ROOT):
        raise ValueError('Output must be under the authorized AOSS project root')
    manifest = ROOT / 'manifests/candidates.json'
    obj = next(x for x in json.loads(manifest.read_text())['objects'] if x['uid'] == args.uid)
    obj['images'] = [str((ROOT / path).resolve()) for path in obj['images']]
    obj['poses'] = str((ROOT / obj['poses']).resolve())
    sources = sorted(map(Path, obj['images']), key=lambda p: p.name)
    poses = np.load(obj['poses']).astype(np.float64)
    if len(sources) != len(poses) or len(sources) != 30:
        raise ValueError('Expected exactly 30 matching images and camera poses')
    for p in ('images', 'masks', 'sparse/0', 'matches'):
        (args.output / p).mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    cv2.setRNGSeed(42)
    train = [i for i in range(len(sources)) if i % 8 != 0]
    test = [i for i in range(len(sources)) if i % 8 == 0]
    train_set = set(train)
    K = np.array([[320., 0., 128.], [0., 320., 128.], [0., 0., 1.]])
    Kinv = np.linalg.inv(K)
    flip = np.diag([1., -1., -1., 1.])
    extrinsics, projections, qs, centers = [], [], [], []
    max_center_error = max_rotation_error = 0.
    for pose in poses:
        cv_c2w = pose @ flip
        U, _, V = np.linalg.svd(cv_c2w[:3, :3])
        c2w_R = U @ V
        R = c2w_R.T
        t = -R @ pose[:3, 3]
        E = np.c_[R, t]
        q = quaternion(R)
        max_rotation_error = max(max_rotation_error, float(np.abs(rotation(q)-R).max()))
        max_center_error = max(max_center_error, float(np.linalg.norm(-R.T @ t-pose[:3, 3])))
        assert np.linalg.det(R) > .999999
        assert np.linalg.norm(project(K @ E, np.zeros(3))[0] - [128, 128]) < 1e-3
        assert project(K @ E, np.zeros(3))[1] > 0
        extrinsics.append(E); projections.append(K @ E); qs.append(q); centers.append(pose[:3, 3])
    assert max_rotation_error < 1e-8 and max_center_error < 1e-8
    centers = np.asarray(centers)
    sift = cv2.SIFT_create()
    colors, keypoints, xys, descriptors, hashes = {}, {}, {}, {}, {}
    for i, src in enumerate(sources):
        with Image.open(src) as im:
            rgba = np.asarray(im.convert('RGBA'))
        if rgba.shape != (256, 256, 4):
            raise ValueError(f'Unexpected image shape: {src}: {rgba.shape}')
        rgb = np.rint(rgba[:, :, :3].astype(float) * (rgba[:, :, 3:4] / 255.)).astype(np.uint8)
        dst = args.output / 'images' / src.name
        Image.fromarray(rgb).save(dst)
        Image.fromarray(rgba[:, :, 3]).save(args.output / 'masks' / src.name)
        hashes[src.name] = dict(source_sha256=digest(src), output_rgb_sha256=digest(dst))
        if i not in train_set:
            continue
        colors[i] = rgb
        # Ignore transparent background and the immediate silhouette boundary.
        mask = cv2.erode((rgba[:, :, 3] > 250).astype(np.uint8) * 255, np.ones((3, 3), np.uint8))
        kp, desc = sift.detectAndCompute(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), mask)
        keypoints[i] = kp
        xys[i] = np.array([k.pt for k in kp], dtype=float).reshape(-1, 2)
        descriptors[i] = desc
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    edges, pair_stats, pair_matches = [], [], {}
    radius_limit = .75 * np.median(np.linalg.norm(centers, axis=1))
    for i, j in itertools.combinations(train, 2):
        di, dj = descriptors[i], descriptors[j]
        if di is None or dj is None or min(len(di), len(dj)) < 2:
            pair_stats.append(dict(image_a=sources[i].name, image_b=sources[j].name,
                                   mutual_ratio_matches=0, geometry_verified_matches=0))
            pair_matches[i, j] = []
            continue
        forward = matcher.knnMatch(di, dj, k=2)
        backward = matcher.knnMatch(dj, di, k=2)
        reverse = {m.queryIdx: m.trainIdx for m, n in backward if m.distance < args.ratio * n.distance}
        tentative = [m for m, n in forward if m.distance < args.ratio*n.distance and reverse.get(m.trainIdx) == m.queryIdx]
        Ei, Ej = extrinsics[i], extrinsics[j]
        Rij = Ej[:3, :3] @ Ei[:3, :3].T
        tij = Ej[:, 3] - Rij @ Ei[:, 3]
        tx, ty, tz = tij
        skew = np.array([[0, -tz, ty], [tz, 0, -tx], [-ty, tx, 0]])
        F = Kinv.T @ skew @ Rij @ Kinv
        accepted = []
        for m in tentative:
            pi, pj = np.r_[xys[i][m.queryIdx], 1], np.r_[xys[j][m.trainIdx], 1]
            Fpi, FTpj = F @ pi, F.T @ pj
            den = np.sum(Fpi[:2]**2) + np.sum(FTpj[:2]**2)
            epi = abs(pj @ Fpi) / np.sqrt(max(den, 1e-20))
            if epi > args.max_epipolar_px:
                continue
            obs = [(i, m.queryIdx), (j, m.trainIdx)]
            X = triangulate(obs, xys, projections)
            if X is None or not np.isfinite(X).all() or np.linalg.norm(X) > radius_limit:
                continue
            errs, depths = [], []
            for cam, k in obs:
                projected, depth = project(projections[cam], X)
                errs.append(np.linalg.norm(projected-xys[cam][k])); depths.append(depth)
            rays = np.array([X-centers[i], X-centers[j]])
            cosine = rays[0] @ rays[1] / np.prod(np.linalg.norm(rays, axis=1))
            angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
            if min(depths) <= 0 or max(errs) > args.max_reprojection_px or angle < args.min_parallax_deg:
                continue
            edges.append((float(max(errs)), float(m.distance), (i, m.queryIdx), (j, m.trainIdx)))
            accepted.append(m)
        pair_stats.append(dict(image_a=sources[i].name, image_b=sources[j].name,
                               mutual_ratio_matches=len(tentative), geometry_verified_matches=len(accepted)))
        pair_matches[i, j] = accepted
    # Merge pair matches to tracks while forbidding two keypoints of one image.
    parent, members, image_sets = {}, {}, {}
    def find(a):
        if a not in parent:
            parent[a] = a; members[a] = {a}; image_sets[a] = {a[0]}
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    conflicts = 0
    for _, _, a, b in sorted(edges):
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        if image_sets[ra] & image_sets[rb]:
            conflicts += 1
            continue
        if len(members[ra]) < len(members[rb]):
            ra, rb = rb, ra
        parent[rb] = ra; members[ra] |= members.pop(rb); image_sets[ra] |= image_sets.pop(rb)
    points, rejected_tracks = [], 0
    assignment = {i: np.full(len(xys[i]), -1, dtype=np.int64) for i in train}
    all_errors = []
    for obs_set in members.values():
        obs = sorted(obs_set)
        if len(obs) < 2:
            continue
        X = triangulate(obs, xys, projections)
        if X is None:
            rejected_tracks += 1
            continue
        errors, depths, samples = [], [], []
        for i, k in obs:
            p, depth = project(projections[i], X)
            errors.append(float(np.linalg.norm(p - xys[i][k]))); depths.append(depth)
            px = np.clip(np.rint(xys[i][k]).astype(int), 0, 255)
            samples.append(colors[i][px[1], px[0]])
        if not np.isfinite(X).all() or np.linalg.norm(X) > radius_limit or min(depths) <= 0 or max(errors) > args.max_reprojection_px:
            rejected_tracks += 1
            continue
        pid = len(points)+1
        for i, k in obs:
            assert assignment[i][k] == -1 and i in train_set
            assignment[i][k] = pid
        points.append(dict(id=pid, xyz=X.tolist(), rgb=np.rint(np.median(samples, axis=0)).astype(int).tolist(),
                           error=float(np.mean(errors)), observations=obs))
        all_errors.extend(errors)
    if len(points) < 20 and not args.allow_sparse:
        raise RuntimeError(f'Only {len(points)} valid points; do not use this scene for training')
    sparse = args.output / 'sparse/0'
    (sparse / 'cameras.txt').write_text('# CAMERA_ID MODEL WIDTH HEIGHT PARAMS\n1 PINHOLE 256 256 320 320 128 128\n')
    with (sparse / 'images.txt').open('w') as f:
        f.write('# Known synthetic poses, not estimated SfM. Test views have no keypoints.\n')
        for i, src in enumerate(sources):
            values = [*qs[i], *extrinsics[i][:, 3]]
            f.write(f'{i+1} ' + ' '.join(f'{v:.17g}' for v in values) + f' 1 {src.name}\n')
            if i in train_set:
                f.write(' '.join(f'{x:.8f} {y:.8f} {pid}' for (x, y), pid in zip(xys[i], assignment[i])))
            f.write('\n')
    with (sparse / 'points3D.txt').open('w') as f:
        f.write('# POINT3D_ID X Y Z R G B ERROR TRACK[IMAGE_ID POINT2D_IDX]\n')
        for p in points:
            f.write(f"{p['id']} " + ' '.join(f'{v:.17g}' for v in p['xyz']) + ' ' +
                    ' '.join(map(str, p['rgb'])) + f" {p['error']:.10g} " +
                    ' '.join(f'{i+1} {k}' for i, k in p['observations']) + '\n')
    # Standard little-endian COLMAP binaries avoid the legacy SceneManager's
    # Python-2-era text parser. Keep the text files for transparent inspection.
    with (sparse / 'cameras.bin').open('wb') as f:
        f.write(struct.pack('<QiiQQdddd', 1, 1, 1, 256, 256, 320., 320., 128., 128.))
    with (sparse / 'images.bin').open('wb') as f:
        f.write(struct.pack('<Q', len(sources)))
        for i, src in enumerate(sources):
            f.write(struct.pack('<i7di', i+1, *qs[i], *extrinsics[i][:, 3], 1))
            f.write(src.name.encode('utf-8') + b'\x00')
            f.write(struct.pack('<Q', len(xys[i]) if i in train_set else 0))
            if i in train_set:
                for (x, y), pid in zip(xys[i], assignment[i]):
                    f.write(struct.pack('<ddq', x, y, int(pid)))
    with (sparse / 'points3D.bin').open('wb') as f:
        f.write(struct.pack('<Q', len(points)))
        for p in points:
            f.write(struct.pack('<QdddBBBdQ', p['id'], *p['xyz'], *p['rgb'], p['error'], len(p['observations'])))
            for i, k in p['observations']:
                f.write(struct.pack('<ii', i+1, k))
    for (i, j), matches in sorted(pair_matches.items(), key=lambda item: -len(item[1]))[:6]:
        retained = [m for m in matches if assignment[i][m.queryIdx] > 0 and assignment[i][m.queryIdx] == assignment[j][m.trainIdx]]
        image = cv2.drawMatches(cv2.cvtColor(colors[i], cv2.COLOR_RGB2BGR), keypoints[i],
                                cv2.cvtColor(colors[j], cv2.COLOR_RGB2BGR), keypoints[j], retained[:60], None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        label = f'Known-pose geometry check: {sources[i].name} / {sources[j].name}; {len(retained)} retained track edges'
        panel = cv2.copyMakeBorder(image, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=(32, 32, 32))
        cv2.putText(panel, label, (5, 19), cv2.FONT_HERSHEY_SIMPLEX, .34, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(args.output / 'matches' / f'{i:03d}_{j:03d}.jpg'), panel)
    stats = dict(num_images=30, num_train_images=len(train), num_test_images=len(test),
                 initialization_recommendation='sfm' if len(points) >= 20 else 'random',
                 matching_status='sufficient' if len(points) >= 20 else 'insufficient_features',
                 num_sparse_points=len(points), track_lengths=summary([len(p['observations']) for p in points]),
                 reprojection_error_px=summary(all_errors), candidate_geometry_edges=len(edges),
                 conflicting_track_merges_rejected=conflicts, inconsistent_tracks_rejected=rejected_tracks,
                 train_sift_features={sources[i].name: len(xys[i]) for i in train},
                 associated_train_features={sources[i].name: int((assignment[i]>0).sum()) for i in train},
                 quaternion_roundtrip_max_abs=max_rotation_error, camera_center_max_error=max_center_error)
    (args.output / 'matches/pair_counts.json').write_text(json.dumps(pair_stats, indent=2)+'\n')
    (args.output / 'matches/stats.json').write_text(json.dumps(stats, indent=2)+'\n')
    provenance = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), uid=args.uid,
        purpose='Known-pose 3DGS implementation smoke test; not a COLMAP SfM result',
        input_manifest=str(manifest), input_manifest_sha256=digest(manifest), input_poses=str(obj['poses']),
        input_poses_sha256=digest(obj['poses']), input_images=[str(p) for p in sources],
        input_source_geometry_read=False, initialization='Training-image SIFT matches triangulated with known synthetic camera poses',
        train_images=[sources[i].name for i in train], test_images=[sources[i].name for i in test],
        split='Lexicographic image-name index % 8 == 0 is test; same split as gsplat examples with test_every=8',
        test_images_used_for_matching=False, test_images_used_for_initial_point_colors=False,
        image_processing='RGBA alpha composited onto black RGB background; no rescaling',
        camera_conversion='OpenCV c2w = Blender c2w @ diag(1,-1,-1,1); w2c = inverse; SVD rotation orthogonalization',
        intrinsics=dict(model='PINHOLE', width=256, height=256, fx=320., fy=320., cx=128., cy=128.),
        matching=dict(method='OpenCV SIFT, mutual Lowe ratio, known-pose Sampson distance, cheirality, parallax, DLT reprojection',
                      ratio=args.ratio, max_epipolar_px=args.max_epipolar_px,
                      max_reprojection_px=args.max_reprojection_px, min_parallax_deg=args.min_parallax_deg,
                      foreground_mask='alpha > 250 eroded by a 3x3 kernel',
                      radius_limit=float(radius_limit), radius_limit_source='0.75 times median camera distance from origin; no mesh geometry'),
        tracks='Greedy union of verified pair correspondences; at most one observation per image; multi-view DLT; inconsistent tracks rejected',
        limitations=['Synthetic camera poses are supplied, not estimated.',
                     'SIFT may give separate tracks for the same physical surface point; spatial duplicates are possible.',
                     'Sparse matches only initialize Gaussians; successful short training is not full benchmark reproduction.'],
        colmap_formats=['text', 'little-endian binary'],
        legacy_scene_manager_note='Use binary models and numpy<2; unmodified legacy pycolmap imports np.uint64(-1), which raises with NumPy 2.',
        software=dict(numpy=np.__version__, opencv=cv2.__version__), script_sha256=digest(__file__),
        image_hashes=hashes, stats=stats)
    (args.output / 'provenance.json').write_text(
        sanitize_text(json.dumps(provenance, indent=2), [(ROOT, '')]) + '\n')
    # Re-read output as independent text records and verify bidirectional tracks.
    observations, names = {}, []
    lines = [line for line in (sparse/'images.txt').read_text().splitlines() if not line.startswith('#')]
    assert len(lines) == 60
    for n in range(0, len(lines), 2):
        header, row = lines[n].split(), lines[n+1].split()
        iid = int(header[0]); names.append(header[9]); assert len(row) % 3 == 0
        observations[iid] = [int(row[j+2]) for j in range(0, len(row), 3)]
        if iid-1 in test:
            assert not row
    count = 0
    for row in (sparse/'points3D.txt').read_text().splitlines():
        if row.startswith('#'):
            continue
        vals = row.split(); pid = int(vals[0]); track = list(map(int, vals[8:])); count += 1
        assert len(track) >= 4 and len(track) % 2 == 0
        for iid, ki in zip(track[::2], track[1::2]):
            assert iid-1 in train_set and observations[iid][ki] == pid
    assert count == len(points) and len(set(names)) == 30
    print(json.dumps(dict(output=str(args.output), **stats), indent=2))


if __name__ == '__main__':
    main()
