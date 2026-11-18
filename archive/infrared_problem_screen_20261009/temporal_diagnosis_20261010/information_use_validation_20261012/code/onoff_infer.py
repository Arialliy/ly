#!/usr/bin/env python3
"""Paired on/off forwards of one frozen trained E2, without parameter updates.

Raw logits are retained directly. Existing on probabilities are checked before
reuse; new off probabilities preserve sigmoid-then-probability-max evaluation.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
Q = ROOT.parent
OLD = Q / 'mutual_temporal_pilot_20261012'
sys.path.insert(0, str(OLD / 'code'))
import train_mutual as old_runner

STARTS = [0, 24, 36, 60, 72, 96, 108, 120]
ARRANGEMENTS = {'O': [0, 36, 72, 108, 120], 'S': [0, 24, 60, 96, 120]}


def save_json(path, value):
    old_runner.save_json(path, value)


def read_regions(path, sequence):
    with Path(path).open() as stream:
        rows = [r for r in csv.DictReader(stream) if r['sequence'] == sequence and r.get('valid', '1') == '1']
    for row in rows:
        for name in ['frame', 'x0', 'x1', 'y0', 'y1']:
            row[name] = int(row[name])
        if not (0 <= row['frame'] < 160 and 0 <= row['x0'] < row['x1'] <= 640 and 0 <= row['y0'] < row['y1'] <= 512):
            raise ValueError('Region outside the frozen image extent')
        if row['arrangement'] not in ARRANGEMENTS:
            raise ValueError('Unknown arrangement in frozen regions')
    if len({r['region_id'] for r in rows}) != len(rows):
        raise ValueError('Region IDs must be unique')
    return rows


def freeze_check(args):
    freeze = json.loads((ROOT / 'A_FREEZE.json').read_text())
    bindings = freeze.get('code_sha256', {})
    relative = str(Path(__file__).relative_to(ROOT))
    if bindings.get(relative) != old_runner.sha(__file__):
        raise AssertionError('This inference source is not bound by the new frozen protocol')
    if freeze.get('regions_sha256') != old_runner.sha(args.regions):
        raise AssertionError('Frozen on/off regions changed')
    old_freeze = json.loads((OLD / 'PROTOCOL_FREEZE.json').read_text())
    for key in ['code/train_mutual.py', 'code/module.py']:
        if old_runner.sha(OLD / key) != old_freeze['code_sha256'][key]:
            raise AssertionError('An old frozen implementation changed')
    run = json.loads((args.checkpoint.parent / 'RUN.json').read_text())
    if run['status'] != 'complete' or run['variant'] != 'E2' or run['final_step'] != 512:
        raise AssertionError('Require the completed old E2 final checkpoint')
    if old_runner.sha(args.checkpoint) != run['final_checkpoint_sha256']:
        raise AssertionError('E2 checkpoint hash mismatch')
    if old_runner.sha(args.reference_checkpoint) != run['checkpoint_sha256']:
        raise AssertionError('Released source-model checkpoint hash mismatch')
    return freeze, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['source', 'checkpoint', 'reference-checkpoint', 'dataset-root', 'regions', 'output', 'off-output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--sequence', choices=['DJI_0026_2', 'DJI_0067_2'], required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    freeze, training = freeze_check(args)
    for folder in [args.output, args.off_output]:
        if folder.exists() and any(folder.iterdir()):
            raise FileExistsError('New validation outputs must be empty; never overwrite old results')
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'windows').mkdir()
    regions = read_regions(args.regions, args.sequence)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(46); np.random.seed(46); torch.manual_seed(46); torch.cuda.manual_seed_all(46)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    helpers = old_runner.load_module('information_validation_old_helpers', Q / 'top1_background_control_20261011/code/train_pilot.py')
    model, model_file = helpers.load_model(args.source, args.reference_checkpoint)
    aggregation = old_runner.attach_aggregation(model, 'E2', radius=2)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint['variant'] != 'E2' or checkpoint['step'] != 512:
        raise AssertionError('Unexpected trained checkpoint identity')
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    initial_hash = old_runner.tensor_hash(model.state_dict().items())
    if initial_hash != training['final_model_state_sha256']:
        raise AssertionError('Loaded state is not exactly the trained E2 state')
    device = torch.device(args.device)
    model.to(device).eval()
    aggregation.collect_stats = False
    raw_path = args.dataset_root / args.sequence / 'raw_frames.npy'
    raw = np.load(raw_path, mmap_mode='r', allow_pickle=False)
    if raw.shape != (160, 512, 640) or raw.dtype != np.uint8:
        raise ValueError('Unexpected frozen development input dimensions')
    old_prediction = OLD / 'predictions/E2' / args.sequence
    old_complete = json.loads((old_prediction / 'COMPLETE.json').read_text())
    if old_complete['checkpoint_sha256'] != old_runner.sha(args.checkpoint):
        raise AssertionError('Old on predictions used a different checkpoint')
    timing = []
    checks = []
    window_norm_rows = []
    fused_off = np.zeros(raw.shape, dtype=np.float32)
    coverage = np.zeros(160, dtype=np.int32)
    context = {'capture': False}

    def capture_features(module, inputs):
        if not context['capture']:
            return
        ya, xa, height, width = context['core_local']
        core_features = inputs[0][..., ya:ya+height, xa:xa+width]
        context['feature_energy'] = core_features.square().sum(1)[0].detach().cpu().numpy()

    def capture_correction(module, inputs, output):
        if not context['capture']:
            return
        ya, xa, height, width = context['core_local']
        core_correction = output[..., ya:ya+height, xa:xa+width]
        context['correction_energy'] = core_correction.square().sum(1)[0].detach().cpu().numpy()

    aggregation.register_forward_pre_hook(capture_features)
    aggregation.output_projection.register_forward_hook(capture_correction)
    meta = {'status': 'RUNNING', 'sequence': args.sequence, 'condition': 'E2_same_checkpoint_onoff',
        'checkpoint_sha256': old_runner.sha(args.checkpoint), 'model_state_sha256': initial_hash,
        'seed': 46, 'normalization': [85.0799, 47.4845], 'starts': STARTS,
        'arrangements': ARRANGEMENTS, 'tile_core': 128, 'tile_halo': 20,
        'off_operation': "same trained E2 state; aggregation.set_mode('off') returns input feature unchanged",
        'new_training': False, 'script_sha256': old_runner.sha(__file__),
        'protocol_freeze_sha256': old_runner.sha(ROOT / 'A_FREEZE.json'),
        'regions_sha256': old_runner.sha(args.regions), 'old_prediction_receipt_sha256': old_runner.sha(old_prediction / 'COMPLETE.json'),
        'raw_sha256': old_runner.sha(raw_path), 'model_source_sha256': old_runner.sha(model_file),
        'parameters': sum(p.numel() for p in model.parameters()), 'precision': 'FP32_no_AMP_no_TF32',
        'residual_norm': 'direct output_projection output, channel squared sum on nonoverlapping tile cores',
        'inference_timing_scope': 'paired on/off with telemetry included; not a standalone speed comparison'}
    save_json(args.output / 'RUNNING.json', meta)
    begin = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for start in STARTS:
            image = torch.from_numpy(np.ascontiguousarray((raw[start:start+40].astype(np.float32)-85.0799)/47.4845))[None, None].to(device)
            on = torch.empty((1, 40, 512, 640), dtype=torch.float32, device='cpu')
            off = torch.empty_like(on)
            f2 = np.empty((40, 512, 640), dtype=np.float32)
            c2 = np.empty_like(f2)
            torch.cuda.synchronize(device); stamp = time.perf_counter()
            for y in range(0, 512, 128):
                y1 = min(y+128, 512)
                for x in range(0, 640, 128):
                    x1 = min(x+128, 640)
                    ya, yb = max(0,y-20), min(512,y1+20)
                    xa, xb = max(0,x-20), min(640,x1+20)
                    tile = image[..., ya:yb, xa:xb].contiguous()
                    context.update(capture=True, core_local=(y-ya, x-xa, y1-y, x1-x))
                    aggregation.set_mode('mutual')
                    z_on = model(tile)[1]
                    on[..., y:y1, x:x1] = z_on[..., y-ya:y1-ya, x-xa:x1-xa].cpu()
                    f2[:, y:y1, x:x1] = context.pop('feature_energy')
                    c2[:, y:y1, x:x1] = context.pop('correction_energy')
                    context['capture'] = False
                    aggregation.set_mode('off')
                    z_off = model(tile)[1]
                    off[..., y:y1, x:x1] = z_off[..., y-ya:y1-ya, x-xa:x1-xa].cpu()
                    del z_on, z_off, tile
            torch.cuda.synchronize(device)
            pair_seconds = time.perf_counter() - stamp
            if not (torch.isfinite(on).all() and torch.isfinite(off).all() and np.isfinite(f2).all() and np.isfinite(c2).all()):
                raise FloatingPointError('Nonfinite forward values')
            on_prob = on.sigmoid()[0].numpy()
            off_prob = off.sigmoid()[0].numpy()
            with np.load(old_prediction / 'windows' / f'window_{start:03d}.npz') as cached:
                old_prob = cached['probabilities']
                same = np.array_equal(on_prob, old_prob)
                max_error = float(np.max(np.abs(on_prob-old_prob)))
                decisions = int(np.count_nonzero((on_prob >= .5) != (old_prob >= .5)))
            checks.append({'start': start, 'old_on_probabilities_bitwise_identical': same,
                           'max_old_on_probability_error': max_error, 'threshold05_disagreements': decisions})
            if not same:
                raise AssertionError('On-forward probabilities differ from the audited old E2 cache; do not reuse score results silently')
            np.savez_compressed(args.output / 'windows' / f'window_{start:03d}.npz',
                                on_logits=on[0].numpy(), off_logits=off[0].numpy(), start=start, end=start+40)
            np.savez_compressed(args.off_output / 'windows' / f'window_{start:03d}.npz',
                                probabilities=off_prob, start=start, end=start+40)
            for region in regions:
                if not (start <= region['frame'] < start+40 and start in ARRANGEMENTS[region['arrangement']]):
                    continue
                slot = region['frame']-start
                sl = (slot, slice(region['y0'],region['y1']), slice(region['x0'],region['x1']))
                feature_energy = float(f2[sl].sum(dtype=np.float64))
                correction_energy = float(c2[sl].sum(dtype=np.float64))
                window_norm_rows.append({'region_id': region['region_id'], 'sequence': args.sequence,
                    'arrangement': region['arrangement'], 'frame': region['frame'], 'window_start': start,
                    'region_type': region['region_type'], 'pixels': int(f2[sl].size),
                    'feature_squared_norm': feature_energy, 'correction_squared_norm': correction_energy,
                    'relative_correction_norm': float(np.sqrt(correction_energy)/max(np.sqrt(feature_energy),1e-12))})
            timing.append({'start': start, 'paired_forward_telemetry_seconds': pair_seconds,
                'feature_squared_norm': float(f2.sum(dtype=np.float64)),
                'correction_squared_norm': float(c2.sum(dtype=np.float64)),
                'global_relative_correction_norm': float(np.sqrt(c2.sum(dtype=np.float64))/max(np.sqrt(f2.sum(dtype=np.float64)),1e-12))})
            if start in ARRANGEMENTS['O']:
                fused_off[start:start+40] = np.maximum(fused_off[start:start+40], off_prob)
                coverage[start:start+40] += 1
            print(json.dumps({'start': start, 'paired_seconds': pair_seconds, 'old_on_exact': same}), flush=True)
            del image, on, off, on_prob, off_prob, old_prob, f2, c2
    aggregation.set_mode('mutual')
    final_hash = old_runner.tensor_hash(model.state_dict().items())
    if final_hash != initial_hash:
        raise AssertionError('Model state changed during read-only on/off inference')
    np.savez_compressed(args.off_output / 'official_fused.npz', probabilities=fused_off, coverage=coverage, starts=np.array(ARRANGEMENTS['O']))
    with (args.output / 'RESIDUAL_REGION_WINDOWS.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(window_norm_rows[0]))
        writer.writeheader(); writer.writerows(window_norm_rows)
    meta.update(status='COMPLETE', elapsed_seconds=time.perf_counter()-begin,
        unchanged_final_model_state_sha256=final_hash, window_timing=timing, on_cache_checks=checks,
        residual_region_window_rows=len(window_norm_rows),
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device),
        window_sha256={f.name:old_runner.sha(f) for f in sorted((args.output/'windows').glob('*.npz'))},
        residual_region_csv_sha256=old_runner.sha(args.output / 'RESIDUAL_REGION_WINDOWS.csv'))
    save_json(args.output / 'COMPLETE.json', meta)
    off_meta = dict(meta, condition='E2_off', variant='E2_off', training_variant='E2', diagnostic='same_checkpoint_residual_off',
                    official_starts=ARRANGEMENTS['O'], diagnostic_starts=STARTS,
                    fusion='pixelwise probability maximum over official starts only',
                    fused_sha256=old_runner.sha(args.off_output / 'official_fused.npz'),
                    window_sha256={f.name:old_runner.sha(f) for f in sorted((args.off_output/'windows').glob('*.npz'))})
    save_json(args.off_output / 'COMPLETE.json', off_meta)
    print(json.dumps({'status':'COMPLETE','sequence':args.sequence,'seconds':meta['elapsed_seconds']}),flush=True)


if __name__ == '__main__':
    main()
