#!/usr/bin/env python3
"""Frozen official DeepPro inference; raw grayscale NPY [T,H,W], T>=40.

Uses checkpoint-adjacent official model source when present; no training, labels,
resizing, intensity clipping, frame padding or metrics. Overlap fusion matches
official test.py. Input values must retain the original image intensity scale.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import random
from pathlib import Path
import sys
import time

import numpy as np
import torch


def tiled_logits(model, x, core, halo):
    """Exact local spatial evaluation, with original per-layer boundary padding."""
    height, width = x.shape[-2:]
    out = torch.empty((x.shape[0], 40, height, width), dtype=x.dtype, device='cpu')
    for y in range(0, height, core):
        y1 = min(y + core, height)
        for z in range(0, width, core):
            z1 = min(z + core, width)
            ya, yb = max(0, y-halo), min(height, y1+halo)
            za, zb = max(0, z-halo), min(width, z1+halo)
            logits = model(x[..., ya:yb, za:zb].contiguous())[1]
            out[..., y:y1, z:z1] = logits[..., y-ya:y1-ya, z-za:z1-za].cpu()
            del logits
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--input-npy', type=Path, required=True)
    p.add_argument('--output-npz', type=Path, required=True)
    p.add_argument('--model-type', choices=['DeepPro', 'DeepPro-Plus'], required=True)
    p.add_argument('--normalization', choices=['RGB-T', 'NUDT-MIRSDT'], required=True,
                   help='Explicit normalization domain; should match checkpoint training domain.')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--window-output-dir', type=Path, required=True, help='Passive saving of each real 40-frame window probability map.')
    p.add_argument('--seed', type=int, default=46,
                   help='Match seed_everything default in pinned official train.py:33; explicitly activated here.')
    p.add_argument('--tile-core', type=int, default=0)
    p.add_argument('--tile-halo', type=int, default=16)
    p.add_argument('--verify-tiling', action='store_true')
    args = p.parse_args()
    args.window_output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if args.tile_core and (args.model_type != 'DeepPro-Plus' or args.tile_halo < 9):
        raise ValueError('Spatial tiling audited only for DeepPro-Plus, requiring halo>=9.')
    source = args.source.resolve()
    checkpoint_path = args.checkpoint.resolve()
    frames = np.load(args.input_npy, allow_pickle=False)
    if frames.ndim != 3 or frames.shape[0] < 40:
        raise ValueError('Expected raw grayscale [T,H,W] with T>=40; official T40 model has no short-clip padding.')
    if not np.issubdtype(frames.dtype, np.number) or not np.isfinite(frames).all():
        raise ValueError('Input must contain finite numeric image intensities.')
    frames = frames.astype(np.float32, copy=False)
    mean, std = {'RGB-T': (85.0799, 47.4845),
                 'NUDT-MIRSDT': (105.4025, 26.6452)}[args.normalization]
    sys.path.insert(0, str(source))
    model_file = checkpoint_path.parent.parent / (args.model_type + '.py')
    if not model_file.is_file():
        model_file = source / 'networks/models' / (args.model_type + '.py')
    spec = importlib.util.spec_from_file_location('official_deeppro_checkpoint_model', model_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Both pinned official checkpoints contain a NumPy float64 IoU scalar.
    # Preserve weights_only loading with only these audited numeric globals.
    numpy_core = np._core if hasattr(np, '_core') else np.core
    numeric_globals = [(numpy_core.multiarray.scalar, 'numpy.core.multiarray.scalar'),
                       np.dtype, type(np.dtype(np.float64))]
    with torch.serialization.safe_globals(numeric_globals):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model = module.detector(1, 40, 40)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    device = torch.device(args.device)
    model = model.to(device).eval()
    n, h, w = frames.shape
    probabilities = np.zeros((n, h, w), dtype=np.float32)
    coverage = np.zeros(n, dtype=np.int32)
    windows = []
    tiling_check = None
    started = time.perf_counter()
    with torch.inference_mode():
        if args.verify_tiling:
            tiling_check = []
            # Fixed image-relative probes span dark sky, center, foreground and edges.
            # No model outputs or GT labels determine these positions.
            ph, pw = min(128, h), min(160, w)
            positions = [(0, 0), ((h-ph)//2, (w-pw)//2),
                         (max(0, h-ph-16), max(0, w//3-pw//2)),
                         (max(0, h-ph-16), min(w-pw, 2*w//3-pw//2)),
                         (h-ph, w-pw)]
            for py, px in positions:
                probe = torch.from_numpy(np.ascontiguousarray((frames[:40, py:py+ph, px:px+pw]-mean)/std))[None,None].to(device)
                full = model(probe)[1].cpu()
                stitched = tiled_logits(model, probe, core=48, halo=args.tile_halo)
                torch.testing.assert_close(stitched, full, rtol=1e-4, atol=1e-4)
                a_prob, b_prob = full.sigmoid(), stitched.sigmoid()
                check = {'crop_yxhw': [py, px, ph, pw], 'shape': list(probe.shape),
                         'max_absolute_logit_error': float((full-stitched).abs().max()),
                         'max_absolute_probability_error': float((a_prob-b_prob).abs().max()),
                         'threshold_disagreements': {str(t): int(((a_prob>=t)!=(b_prob>=t)).sum()) for t in [0.1, 0.5, 0.9]},
                         'rtol': 1e-4, 'atol': 1e-4, 'status': 'passed'}
                tiling_check.append(check)
                print(json.dumps({'tiling_check':check}), flush=True)
                del probe, full, stitched, a_prob, b_prob
        for i in range(math.ceil((n - 4) / 36)):
            end = min(n, (i + 1) * 36 + 4)
            start = max(0, end - 40)
            x = torch.from_numpy(np.ascontiguousarray((frames[start:end] - mean) / std))[None, None].to(device)
            logits = tiled_logits(model, x, args.tile_core, args.tile_halo) if args.tile_core else model(x)[1]
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            if probs.shape != (40, h, w):
                raise RuntimeError(f'Unexpected model output shape {probs.shape}')
            np.savez_compressed(args.window_output_dir / f'window_{i:02d}_{start:04d}_{end:04d}.npz', probabilities=probs, start=start, end=end)
            probabilities[start:end] = np.maximum(probabilities[start:end], probs)
            coverage[start:end] += 1
            windows.append([start, end])
            print(json.dumps({'window_complete': [start, end]}), flush=True)
            del x, logits, probs
    if not np.all(coverage > 0):
        raise RuntimeError('Uncovered frames')
    metadata = {
        'model_type': args.model_type, 'model_file': str(model_file),
        'wrapper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'window_output_dir': str(args.window_output_dir.resolve()),
        'checkpoint': str(checkpoint_path),
        'checkpoint_sha256': hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        'input': str(args.input_npy.resolve()), 'input_shape': list(frames.shape),
        'normalization': args.normalization, 'mean': mean, 'std': std,
        'model_source_sha256': hashlib.sha256(model_file.read_bytes()).hexdigest(),
        'torch_version': torch.__version__, 'device': str(device),
        'elapsed_inference_seconds': time.perf_counter() - started,
        'window_intervals': 'zero-based, end-exclusive', 'overlap_fusion': 'probability_max',
        'seed': args.seed, 'seed_note': 'Matches declared seed_everything default 46 in official train.py:33; original entrypoint does not call it. Explicitly activated for local inference, not a claim about checkpoint training seed.',
        'deterministic_algorithms': True,
        'tf32': False, 'spatial_tile_core': args.tile_core, 'spatial_halo': args.tile_halo if args.tile_core else None,
        'tiling_check': tiling_check,
        'temporal_causality': 'acausal within each 40-frame clip',
        'normalization_clip': None, 'checkpoint_epoch': checkpoint.get('epoch'),
    }
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, probabilities=probabilities,
                        windows=np.asarray(windows, dtype=np.int64), coverage=coverage,
                        metadata_json=np.array(json.dumps(metadata)))
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
