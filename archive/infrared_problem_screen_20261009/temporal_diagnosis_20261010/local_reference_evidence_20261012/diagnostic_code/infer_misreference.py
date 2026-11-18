#!/usr/bin/env python3
"""Conditional same-checkpoint full-frame local-reference mismatch diagnostic.

No model/loss changes, no training and no threshold selection. Construct median
and MAD on each original whole window. Roll only those reference maps on the
whole image before spatial tiling, then recompute the relative channels.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import time
import types

import numpy as np
import torch

N = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(N / 'code'))
import train_evidence as runner

STARTS = [0, 24, 36, 60, 72, 96, 108, 120]
OFFICIAL = [0, 36, 72, 108, 120]
SYMMETRIC = [0, 24, 60, 96, 120]
SEQUENCES = ['DJI_0026_2', 'DJI_0067_2']
CONDITION = 'N2_misreference'
CHECKPOINT = N / 'training/N2/run/final_model.pth'


@torch.no_grad()
def misreference(full_evidence, s_min, shifts):
    """[B,11,T,H,W] -> same shape; spatial roll is global, never tile-local."""
    if full_evidence.ndim != 5 or full_evidence.shape[1] != 11:
        raise ValueError('Expected full [B,11,T,H,W] evidence')
    result = full_evidence.clone()
    for base in (1, 6):
        valid = full_evidence[:, base + 4] != 0
        mu = torch.roll(full_evidence[:, base + 1], shifts=shifts, dims=(-2, -1))
        scale = torch.roll(full_evidence[:, base + 2], shifts=shifts, dims=(-2, -1))
        result[:, base + 1] = torch.where(valid, mu, 0)
        result[:, base + 2] = torch.where(valid, scale, 0)
        result[:, base + 3] = torch.where(valid,
            (full_evidence[:, base] - mu) / scale.clamp_min(s_min), 0)
    return result


@contextmanager
def precomputed_constructor(branch, full_evidence):
    """Supply crops to existing forward without editing its source or weights."""
    if 'construct_evidence' in branch.__dict__:
        raise AssertionError('Unexpected instance override already exists')
    bounds = [None]

    def constructor(self, image):
        ya, yb, xa, xb = bounds[0]
        values = full_evidence[..., ya:yb, xa:xb]
        b, c, t, h, w = values.shape
        if tuple(image.shape) != (b, 1, t, h, w):
            raise AssertionError('Precomputed evidence/image crop mismatch')
        self.last_input = values.permute(0, 2, 1, 3, 4).detach()
        self.last_statistics = {}
        return values

    branch.construct_evidence = types.MethodType(constructor, branch)
    try:
        yield bounds
    finally:
        del branch.construct_evidence
        branch.last_input = None
        branch.last_statistics = {}


@torch.no_grad()
def infer_from_full_evidence(model, image, full_evidence, device, core=128, halo=16):
    b, _, t, h, w = image.shape
    output = torch.empty((b, t, h, w), dtype=torch.float32, device='cpu')
    if device.type == 'cuda': torch.cuda.synchronize(device)
    stamp = time.perf_counter()
    calls = 0
    with precomputed_constructor(model.evidence, full_evidence) as bounds:
        for y in range(0, h, core):
            y1 = min(y + core, h)
            for x in range(0, w, core):
                x1 = min(x + core, w)
                ya, yb = max(0, y - halo), min(h, y1 + halo)
                xa, xb = max(0, x - halo), min(w, x1 + halo)
                bounds[0] = (ya, yb, xa, xb)
                result = model(image[..., ya:yb, xa:xb].contiguous())[1]
                output[..., y:y1, x:x1] = result[..., y-ya:y1-ya, x-xa:x1-xa].cpu()
                calls += 1
    if device.type == 'cuda': torch.cuda.synchronize(device)
    seconds = time.perf_counter() - stamp
    probabilities = output.sigmoid()[0].numpy()
    if not np.isfinite(probabilities).all():
        raise FloatingPointError('Nonfinite diagnostic output')
    return probabilities, seconds, calls


def assert_protocol_identity(freeze):
    if runner.sha(N / 'PROTOCOL.md') != freeze['protocol_sha256']:
        raise AssertionError('Original protocol was edited')
    for path, digest in freeze['code_sha256'].items():
        if runner.sha(N / path) != digest:
            raise AssertionError('Frozen source changed: ' + path)
    for path, digest in freeze['artifacts_sha256'].items():
        if runner.sha(N / path) != digest:
            raise AssertionError('Frozen artifact changed: ' + path)


def load_frozen_model():
    run = json.loads((CHECKPOINT.parent / 'RUN.json').read_text())
    source, reference = Path(run['source_path']), Path(run['checkpoint_path'])
    base = runner.load_module('misreference_frozen_batch', runner.Q / 'top1_background_control_20261011/code/train_pilot.py')
    original, model_path = base.load_model(source, reference)
    original_hash = runner.tensor_hash(original.state_dict().items())
    checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    snapshot = next(item for item in run['snapshots'] if item['step'] == 512)
    sources = runner.source_hashes(model_path)
    if (run['status'] != 'complete' or run['variant'] != 'N2' or run['mode'] != 'relative'
        or checkpoint['variant'] != 'N2' or checkpoint['mode'] != 'relative' or checkpoint['step'] != 512
        or snapshot['sha256'] != runner.sha(CHECKPOINT)
        or checkpoint['initial_checkpoint_sha256'] != runner.sha(reference)
        or checkpoint['plan_sha256'] != runner.EXPECTED['plan']
        or checkpoint['selection_sha256'] != runner.EXPECTED['selection']
        or checkpoint['label_config'] != runner.LABEL_CONFIG
        or checkpoint['effective_supervision_config'] != runner.EFFECTIVE_SUPERVISION
        or checkpoint['original_state_sha256'] != original_hash
        or run['original_initial_state_sha256'] != original_hash
        or run['original_final_state_sha256'] != original_hash
        or any(run[name] != digest or checkpoint[name] != digest for name, digest in sources.items())):
        raise AssertionError('N2 frozen checkpoint/training identity failed')
    model = runner.wrap_model(original, 'N2')
    if checkpoint['model_config'] != model.config:
        raise AssertionError('Frozen evidence configuration mismatch')
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    whole_hash = runner.tensor_hash(model.state_dict().items())
    if (whole_hash != snapshot['model_state_sha256'] or whole_hash != run['final_model_state_sha256']
        or runner.tensor_hash(model.original.state_dict().items()) != original_hash):
        raise AssertionError('Strict loaded state mismatch')
    return model, run, reference, model_path, sources, original_hash, whole_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence', choices=SEQUENCES, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    expected_output = N / 'diagnostic_predictions' / CONDITION / args.sequence
    if args.output.resolve() != expected_output.resolve():
        raise ValueError('Diagnostic output must be ' + str(expected_output))
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Output must be empty; no overwrite')
    freeze_path = N / 'PROTOCOL_FREEZE.json'
    freeze = json.loads(freeze_path.read_text())
    assert_protocol_identity(freeze)
    runner.deterministic_setup()
    model, run, reference, model_path, sources, original_hash, whole_hash = load_frozen_model()
    device = torch.device(args.device)
    model = model.to(device).eval()
    model.requires_grad_(False)
    cached_root = N / 'predictions/N2' / args.sequence
    cached = json.loads((cached_root / 'COMPLETE.json').read_text())
    raw_path = runner.Q / 'stage1_goal/data' / args.sequence / 'raw_frames.npy'
    ann_path = raw_path.with_name('annotations.json')
    if (cached['status'] != 'COMPLETE' or cached['checkpoint_sha256'] != runner.sha(CHECKPOINT)
        or cached['raw_sha256'] != runner.sha(raw_path)
        or cached['annotations_sha256'] != runner.sha(ann_path)
        or any(cached[name] != digest for name, digest in sources.items())):
        raise AssertionError('Original N2 prediction cache identity failed')
    raw = np.load(raw_path, mmap_mode='r', allow_pickle=False)
    if raw.dtype != np.uint8 or raw.shape != (160, 512, 640):
        raise AssertionError('Unexpected frozen full-frame input')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'windows').mkdir()
    fused = np.zeros(raw.shape, dtype=np.float32)
    coverage = np.zeros(160, dtype=np.int32)
    shifts = (raw.shape[1] // 2, raw.shape[2] // 2)
    metadata = {
        'sequence': args.sequence, 'condition': CONDITION, 'variant': CONDITION,
        'training_variant': 'N2', 'mode': 'relative',
        'diagnostic': 'conditional_same_checkpoint_global_local_reference_mismatch',
        'diagnostic_definition': {
            'source': str(N / 'PROTOCOL.md'), 'source_sha256': freeze['protocol_sha256'],
            'section': '同步证据记录及条件诊断', 'full_frame_hw': [512, 640],
            'global_roll_yx': list(shifts), 'roll_dims': ['height', 'width'],
            'rolled_channels': ['mu1', 's1', 'mu4', 's4'],
            'recomputed_channels': ['E1', 'E4'], 'unchanged_channels': ['I', 'D1', 'm1', 'D4', 'm4'],
            'unchanged_representation': 'original H0 and z0',
            'missing_reference': 'all five channels stay zero',
            'construction': 'whole-window whole-image statistics then shared global mu/s roll then E recomputation then spatial crop',
            'tile_local_roll': False, 's_min': model.evidence.s_min,
            'interpretation': 'distribution intervention only; not a new detector or a causal identification',
        },
        'checkpoint': str(CHECKPOINT), 'checkpoint_sha256': runner.sha(CHECKPOINT),
        'reference_checkpoint_sha256': runner.sha(reference), 'model_source': str(model_path), **sources,
        'raw_sha256': runner.sha(raw_path), 'annotations_sha256': runner.sha(ann_path),
        'script_sha256': runner.sha(__file__), 'protocol_sha256': freeze['protocol_sha256'],
        'protocol_freeze_sha256': runner.sha(freeze_path),
        'original_N2_complete_sha256': runner.sha(cached_root / 'COMPLETE.json'),
        'plan_sha256': runner.EXPECTED['plan'], 'selection_sha256': runner.EXPECTED['selection'],
        'original_initial_state_sha256': original_hash, 'original_final_state_sha256': original_hash,
        'model_initial_state_sha256': whole_hash, 'bn_initial_sha256': run['bn_initial_sha256'],
        'bn_final_sha256': run['bn_final_sha256'], 'normalization': [85.0799, 47.4845], 'seed': 46,
        'precision': 'FP32_no_AMP_no_TF32', 'torch': str(torch.__version__), 'cuda': str(torch.version.cuda),
        'official_starts': OFFICIAL, 'symmetric_starts': SYMMETRIC, 'diagnostic_starts': STARTS,
        'fusion': 'pixelwise probability maximum over official starts only', 'threshold': '>=0.5',
        'tile_core': 128, 'tile_halo': 16, 'model_labels_access': False,
        'read_only_query_labels_access': False, 'annotation_access': 'identity hash only, never parsed',
        'window_timing': [], 'model_config': model.config, 'label_config': runner.LABEL_CONFIG,
        'parameters': sum(v.numel() for v in model.parameters()),
        'trainable_parameters': 0, 'training_branch_parameter_count': 1529,
        'main_checkpoint_step': 512, 'checkpoint_step': 512, 'checkpoint_is_main': True,
        'statistics_chunk_frames': model.evidence.statistics_chunk_frames,
        'peak_memory_scope': 'full evidence construction/roll, same frozen model tiled inference and CPU output copy',
        'zero_shift_equivalence': None,
    }
    runner.save_json(args.output / 'RUNNING.json', metadata)
    begin = time.perf_counter()
    try:
        with torch.inference_mode():
            if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
            for start in STARTS:
                image = torch.from_numpy(np.ascontiguousarray(
                    (raw[start:start+40].astype(np.float32) - 85.0799) / 47.4845))[None, None].to(device)
                if device.type == 'cuda': torch.cuda.synchronize(device)
                stamp = time.perf_counter()
                original_evidence = model.evidence.construct_evidence(image)
                if device.type == 'cuda': torch.cuda.synchronize(device)
                statistic_seconds = time.perf_counter() - stamp
                if start == STARTS[0]:
                    # Validate the new computational path before applying any intervention.
                    zero_reference = misreference(original_evidence, model.evidence.s_min, (0, 0))
                    if not torch.equal(zero_reference, original_evidence):
                        raise AssertionError('Zero shift must preserve every evidence channel exactly')
                    zero_probs, zero_seconds, zero_calls = infer_from_full_evidence(model, image, zero_reference, device)
                    cache_path = cached_root / 'windows' / f'window_{start:03d}.npz'
                    if runner.sha(cache_path) != cached['window_sha256'][cache_path.name]:
                        raise AssertionError('Original N2 first-window cache hash failed')
                    with np.load(cache_path, allow_pickle=False) as obj:
                        expected_probs = obj['probabilities']
                    maximum = float(np.max(np.abs(zero_probs - expected_probs)))
                    disagree = int(np.count_nonzero((zero_probs >= .5) != (expected_probs >= .5)))
                    np.testing.assert_allclose(zero_probs, expected_probs, atol=1e-6, rtol=1e-6)
                    if disagree:
                        raise AssertionError('Zero-shift evaluation path changed threshold decisions')
                    metadata['zero_shift_equivalence'] = {
                        'status': 'PASS', 'window_start': start, 'reference_window': str(cache_path),
                        'reference_sha256': runner.sha(cache_path), 'evidence_exact_equal': True,
                        'max_probability_error': maximum, 'threshold05_disagreements': disagree,
                        'atol': 1e-6, 'rtol': 1e-6, 'seconds': zero_seconds, 'tile_model_calls': zero_calls,
                    }
                    runner.save_json(args.output / 'ZERO_SHIFT_CHECK.json', metadata['zero_shift_equivalence'])
                    del zero_reference, zero_probs, expected_probs
                if device.type == 'cuda': torch.cuda.synchronize(device)
                stamp = time.perf_counter()
                shifted = misreference(original_evidence, model.evidence.s_min, shifts)
                del original_evidence
                if device.type == 'cuda': torch.cuda.synchronize(device)
                shift_seconds = time.perf_counter() - stamp
                probabilities, model_seconds, calls = infer_from_full_evidence(model, image, shifted, device)
                stamp = time.perf_counter()
                np.savez_compressed(args.output / 'windows' / f'window_{start:03d}.npz',
                    probabilities=probabilities, start=start, end=start+40)
                write_seconds = time.perf_counter() - stamp
                if start in OFFICIAL:
                    fused[start:start+40] = np.maximum(fused[start:start+40], probabilities)
                    coverage[start:start+40] += 1
                metadata['window_timing'].append({
                    'start': start, 'full_evidence_statistics_seconds': statistic_seconds,
                    'global_misreference_seconds': shift_seconds, 'tile_model_seconds': model_seconds,
                    'model_plus_cpu_copy_seconds': statistic_seconds + shift_seconds + model_seconds,
                    'window_write_seconds': write_seconds, 'tile_model_calls': calls,
                })
                print(json.dumps({'window_complete': start, 'condition': CONDITION,
                    'sequence': args.sequence, 'elapsed_seconds': time.perf_counter()-begin}), flush=True)
                del shifted, image, probabilities
        if not np.all(coverage > 0): raise AssertionError('Uncovered output frames')
        final_original = runner.tensor_hash(model.original.state_dict().items())
        final_whole = runner.tensor_hash(model.state_dict().items())
        if final_original != original_hash or final_whole != whole_hash:
            raise AssertionError('Model parameters or buffers changed')
        if sources != runner.source_hashes(model_path): raise AssertionError('Frozen sources changed')
        assert_protocol_identity(freeze)
        if runner.sha(__file__) != metadata['script_sha256']:
            raise AssertionError('Diagnostic source changed while running')
        np.savez_compressed(args.output / 'official_fused.npz', probabilities=fused, coverage=coverage,
                            starts=np.array(OFFICIAL))
        metadata.update(status='COMPLETE', elapsed_seconds=time.perf_counter()-begin,
            new_model_inference_elapsed_seconds=time.perf_counter()-begin,
            model_final_state_sha256=final_whole, frozen_all_model_state_unchanged=True,
            frozen_original_unchanged=True,
            O_model_seconds=sum(r['model_plus_cpu_copy_seconds'] for r in metadata['window_timing'] if r['start'] in OFFICIAL),
            S_model_seconds=sum(r['model_plus_cpu_copy_seconds'] for r in metadata['window_timing'] if r['start'] in SYMMETRIC),
            total_diagnostic_tile_model_calls=160, zero_shift_check_tile_model_calls=20,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0,
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else 0,
            fused_sha256=runner.sha(args.output / 'official_fused.npz'),
            window_sha256={p.name:runner.sha(p) for p in sorted((args.output/'windows').glob('*.npz'))})
        runner.save_json(args.output / 'COMPLETE.json', metadata)
        print(json.dumps({'status': 'COMPLETE', 'sequence': args.sequence, 'condition': CONDITION,
                          'seconds': metadata['elapsed_seconds']}), flush=True)
    except BaseException as error:
        metadata.update(status='FAILED', error=repr(error), elapsed_seconds=time.perf_counter()-begin)
        runner.save_json(args.output / 'FAILED.json', metadata)
        raise


if __name__ == '__main__':
    main()
