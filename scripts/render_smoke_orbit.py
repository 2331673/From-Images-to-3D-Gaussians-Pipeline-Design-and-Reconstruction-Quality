#!/usr/bin/env python3
"""Render a project-owned gsplat v1.5.3 checkpoint into an RGB-only orbit GIF.

Train with --no-normalize-world-space. No reference mesh, training images, or
COLMAP reconstruction are opened. The checkpoint and cfg.yml are sufficient.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def project_path(path: Path, must_exist: bool = False) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT.resolve()):
        raise ValueError('Inputs and outputs must be inside the AOSS 3dcv_hw2 project')
    if must_exist and not resolved.is_file():
        raise ValueError(f'Input file is missing: {resolved}')
    return resolved


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict:
    import yaml

    class PlainConfigLoader(yaml.SafeLoader):
        pass

    def inert_python_tag(loader, suffix, node):
        # Upstream yaml.dump(vars(cfg)) tags strategy dataclasses as Python
        # objects. Read only their plain contents; never instantiate classes.
        if isinstance(node, yaml.MappingNode):
            return loader.construct_mapping(node, deep=True)
        if isinstance(node, yaml.SequenceNode):
            return loader.construct_sequence(node, deep=True)
        return loader.construct_scalar(node)

    PlainConfigLoader.add_multi_constructor('tag:yaml.org,2002:python/', inert_python_tag)
    with path.open() as stream:
        config = yaml.load(stream, Loader=PlainConfigLoader)
    if not isinstance(config, dict):
        raise ValueError('cfg.yml must contain a configuration mapping')
    if config.get('normalize_world_space') is not False:
        raise ValueError('Orbit requires normalize_world_space: false in cfg.yml; '
                         'train with --no-normalize-world-space or explicitly transform the camera path first')
    if config.get('camera_model', 'pinhole') != 'pinhole':
        raise ValueError('This orbit uses a pinhole camera')
    if config.get('app_opt', False) or config.get('use_bilateral_grid', False):
        raise ValueError('Appearance optimization and bilateral grids are not supported by this splats-only renderer')
    if not isinstance(config.get('sh_degree'), int) or not 0 <= config['sh_degree'] <= 4:
        raise ValueError('cfg.yml must contain a valid SH degree in 0..4')
    return config


def orbit_viewmats(frames: int, radius: float, elevation_degrees: float,
                   center: tuple[float, float, float]) -> list:
    """OpenCV world-to-camera matrices: camera +Z forward, +X right, +Y down.

    World +Z is up. Azimuth starts on world +X and increases toward world +Y;
    the endpoint is omitted so that a looping GIF has no duplicated end frame.
    """
    elevation = math.radians(elevation_degrees)
    ce, se = math.cos(elevation), math.sin(elevation)
    matrices = []
    for index in range(frames):
        angle = 2 * math.pi * index / frames
        ca, sa = math.cos(angle), math.sin(angle)
        offset = [radius * ce * ca, radius * ce * sa, radius * se]
        eye = [center[i] + offset[i] for i in range(3)]
        right = [-sa, ca, 0.0]
        down = [se * ca, se * sa, -ce]
        forward = [-ce * ca, -ce * sa, -se]
        rows = [axis + [-sum(axis[i] * eye[i] for i in range(3))]
                for axis in (right, down, forward)]
        matrices.append(rows + [[0.0, 0.0, 0.0, 1.0]])
    return matrices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--config', type=Path, help='Default: checkpoint parent/../cfg.yml')
    parser.add_argument('--output', type=Path, help='Exact .gif output path; mutually exclusive with --output-dir')
    parser.add_argument('--output-dir', type=Path, help='Default: checkpoint parent/../orbit')
    parser.add_argument('--frames', type=int, default=96)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--radius', type=float, default=2.4)
    parser.add_argument('--elevation', type=float, default=20.0)
    parser.add_argument('--focal-length', type=float, default=320.0)
    parser.add_argument('--center', type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument('--fps', type=float, default=24.0)
    parser.add_argument('--mp4', action='store_true', help='Also encode orbit.mp4 using imageio/ffmpeg')
    args = parser.parse_args()
    if args.output and args.output_dir:
        parser.error('Use either --output or --output-dir')
    if args.output and args.output.suffix.lower() != '.gif':
        parser.error('--output must end in .gif')
    if min(args.frames, args.width, args.height) < 1:
        parser.error('Frames and image dimensions must be positive')
    if args.radius <= 0 or args.focal_length <= 0 or not 0 < args.fps <= 100:
        parser.error('Radius/focal length must be positive, and fps must be in (0,100]')
    if not -89.0 <= args.elevation <= 89.0:
        parser.error('Elevation must be in [-89,89] degrees')
    if not all(math.isfinite(value) for value in [args.radius, args.focal_length,
                                                 args.elevation, args.fps, *args.center]):
        parser.error('Camera parameters must be finite')

    started = time.perf_counter()
    checkpoint = project_path(args.checkpoint, must_exist=True)
    config_path = project_path(args.config or checkpoint.parent.parent / 'cfg.yml', must_exist=True)
    output_dir = project_path(args.output.parent if args.output else
                              (args.output_dir or checkpoint.parent.parent / 'orbit'))
    gif_path = project_path(args.output) if args.output else output_dir / 'orbit.gif'
    config = load_config(config_path)

    import gsplat
    import numpy as np
    import torch
    from PIL import Image

    if gsplat.__version__ != '1.5.3':
        raise ValueError(f'Expected gsplat 1.5.3, found {gsplat.__version__}')
    if not torch.cuda.is_available():
        raise RuntimeError('This renderer requires an available CUDA GPU')

    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get('splats'), dict):
        raise ValueError('Expected an upstream simple_trainer checkpoint with a splats dictionary')
    splats = payload['splats']
    required = {'means', 'quats', 'scales', 'opacities', 'sh0', 'shN'}
    if not required.issubset(splats):
        raise ValueError(f'Missing checkpoint tensors: {sorted(required - splats.keys())}')
    count = int(splats['means'].shape[0])
    if count < 1:
        raise ValueError('Checkpoint contains no Gaussians')
    expected_shapes = {'means': (count, 3), 'quats': (count, 4), 'scales': (count, 3),
                       'opacities': (count,), 'sh0': (count, 1, 3),
                       'shN': (count, (config['sh_degree'] + 1) ** 2 - 1, 3)}
    for key, shape in expected_shapes.items():
        value = splats[key]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(f'Checkpoint {key} has an unexpected shape; expected {shape}')
        if not torch.isfinite(value).all().item():
            raise ValueError(f'Checkpoint {key} contains NaN/Inf')

    device = torch.device('cuda:0')
    # Match v1.5.3 simple_trainer.Runner.rasterize_splats exactly. Quaternions
    # are normalized inside gsplat; SH coefficients are not RGB probabilities.
    means = splats['means'].to(device=device, dtype=torch.float32).contiguous()
    quats = splats['quats'].to(device=device, dtype=torch.float32).contiguous()
    scales = splats['scales'].to(device=device, dtype=torch.float32).exp().contiguous()
    opacities = splats['opacities'].to(device=device, dtype=torch.float32).sigmoid().contiguous()
    colors = torch.cat([splats['sh0'], splats['shN']], dim=1).to(device=device, dtype=torch.float32).contiguous()
    if not torch.isfinite(scales).all().item():
        raise ValueError('Activated Gaussian scales contain Inf/NaN')
    viewmats = orbit_viewmats(args.frames, args.radius, args.elevation, tuple(args.center))
    intrinsics = [[args.focal_length, 0.0, args.width / 2],
                  [0.0, args.focal_length, args.height / 2], [0.0, 0.0, 1.0]]
    Ks = torch.tensor([intrinsics], dtype=torch.float32, device=device)
    background = torch.zeros((1, 3), dtype=torch.float32, device=device)
    render_options = {
        'packed': bool(config.get('packed', False)), 'absgrad': False, 'sparse_grad': False,
        'rasterize_mode': 'antialiased' if config.get('antialiased', False) else 'classic',
        'camera_model': 'pinhole', 'with_ut': bool(config.get('with_ut', False)),
        'with_eval3d': bool(config.get('with_eval3d', False)),
        'near_plane': float(config.get('near_plane', 0.01)),
        'far_plane': float(config.get('far_plane', 1e10)),
        # Upstream eval/render_traj uses cfg.sh_degree even before the final
        # SH warmup interval; untouched higher-order coefficients remain zero.
        'sh_degree': config['sh_degree'], 'render_mode': 'RGB',
    }
    rgb_frames = []
    torch.cuda.synchronize()
    render_started = time.perf_counter()
    with torch.inference_mode():
        for index, matrix in enumerate(viewmats):
            rgb, _, _ = gsplat.rasterization(
                means=means, quats=quats, scales=scales, opacities=opacities,
                colors=colors, viewmats=torch.tensor([matrix], dtype=torch.float32, device=device),
                Ks=Ks, width=args.width, height=args.height, backgrounds=background,
                **render_options,
            )
            if not torch.isfinite(rgb).all().item():
                raise RuntimeError(f'Rendered frame {index} contains NaN/Inf')
            rgb_frames.append((rgb[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
    torch.cuda.synchronize()
    render_seconds = time.perf_counter() - render_started

    output_dir.mkdir(parents=True, exist_ok=True)
    # A shared palette sampled across the complete orbit avoids frame-to-frame
    # palette changes. GIF durations have 10 ms resolution.
    palette_sample = np.concatenate([frame[::4, ::4] for frame in rgb_frames], axis=1)
    palette = Image.fromarray(palette_sample).quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    gif_frames = [Image.fromarray(frame).quantize(palette=palette, dither=Image.Dither.NONE)
                  for frame in rgb_frames]
    # Disposal=2 clears the preceding frame to the GIF's logical background.
    # Palette index 0 is not necessarily black (MEDIANCUT often puts white there).
    # An explicit black index prevents white flashes around cropped GIF frames.
    palette_rgb = np.asarray(palette.getpalette(), dtype=np.int32).reshape(-1, 3)
    background_index = int(np.square(palette_rgb).sum(axis=1).argmin())
    frame_duration_ms = max(10, round(1000 / args.fps / 10) * 10)
    gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:], loop=0,
                       duration=frame_duration_ms, disposal=2, optimize=False,
                       background=background_index)
    with Image.open(gif_path) as saved:
        stored_frames = saved.n_frames
    if stored_frames != args.frames:
        raise RuntimeError(f'GIF encoded {stored_frames}/{args.frames} distinct frames; '
                           'check for an empty or identical render sequence')
    artifacts = {'gif': {'path': str(gif_path), 'sha256': sha256(gif_path)}}
    if args.mp4:
        import imageio.v2 as imageio
        mp4_path = gif_path.with_suffix('.mp4')
        with imageio.get_writer(str(mp4_path), fps=args.fps, codec='libx264',
                                pixelformat='yuv420p', macro_block_size=1) as writer:
            for frame in rgb_frames:
                writer.append_data(frame)
        artifacts['mp4'] = {'path': str(mp4_path), 'sha256': sha256(mp4_path)}

    cameras_path = output_dir / 'orbit_cameras.json'
    cameras_path.write_text(json.dumps({'convention': 'OpenCV world-to-camera, world +Z up',
                                       'K': intrinsics, 'viewmats': viewmats}, indent=2) + '\n')
    metadata = {
        'created_at': datetime.now(timezone.utc).isoformat(), 'checkpoint': str(checkpoint),
        'checkpoint_sha256': sha256(checkpoint), 'checkpoint_step_zero_based': payload.get('step'),
        'config': str(config_path), 'config_sha256': sha256(config_path),
        'script_sha256': sha256(Path(__file__)), 'normalize_world_space': False,
        'gsplat_version': gsplat.__version__, 'torch_version': str(torch.__version__),
        'cuda_runtime': torch.version.cuda, 'gpu_name': torch.cuda.get_device_name(device),
        'num_gaussians': count, 'frames': len(rgb_frames), 'gif_frames': stored_frames,
        'width': args.width, 'height': args.height, 'radius': args.radius,
        'elevation_degrees': args.elevation, 'center': args.center, 'focal_length_pixels': args.focal_length,
        'background_rgb': [0, 0, 0], 'requested_fps': args.fps,
        'gif_frame_duration_ms': frame_duration_ms, 'gif_effective_fps': 1000 / frame_duration_ms,
        'render_options': render_options, 'render_seconds': render_seconds,
        'elapsed_seconds': time.perf_counter() - started, 'artifacts': artifacts,
        'camera_path': str(cameras_path), 'camera_path_sha256': sha256(cameras_path),
    }
    metadata_path = output_dir / 'orbit_metadata.json'
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({'gif': str(gif_path), 'metadata': str(metadata_path),
                      'frames': len(rgb_frames), 'render_seconds': render_seconds}, indent=2))


if __name__ == '__main__':
    main()
