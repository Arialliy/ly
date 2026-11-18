#!/usr/bin/env python3
"""Fixed H / HL snapshots, FP32 full-frame inference using the existing tile rule.

The eight-window union is only a computation cache for the two fixed five-window
arrangements. No target-wise window selection or checkpoint selection is done.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import train_joint as training

BASE = Path(__file__).resolve().parents[1]
O = [0, 36, 72, 108, 120]
S = [0, 24, 60, 96, 120]
STARTS = sorted(set(O + S))
STEPS = [512, 1024, 2048]
SEQUENCES = ['DJI_0026_2', 'DJI_0067_2']
HEAD_NAMES = {'conv_out1.0.weight', 'conv_out1.0.bias', 'conv_out2.weight', 'conv_out2.bias'}


def tiled_logits(model, x, core=128, halo=20):
    """Identical to frozen infer_tlfp.tiled_logits, including global Haar phase."""
    height, width = x.shape[-2:]
    output = torch.empty((x.shape[0], 40, height, width), dtype=torch.float32, device='cpu')
    for y in range(0, height, core):
        y1 = min(height, y + core)
        ya = max(0, y - halo) // 2 * 2
        yb = min(height, ((y1 + halo + 1) // 2) * 2)
        for z in range(0, width, core):
            z1 = min(width, z + core)
            za = max(0, z - halo) // 2 * 2
            zb = min(width, ((z1 + halo + 1) // 2) * 2)
            result = model(x[..., ya:yb, za:zb].contiguous())[1]
            output[..., y:y1, z:z1] = result[..., y-ya:y1-ya, z-za:z1-za].cpu()
    return output


def infer(args):
    training.deterministic_setup()
    source = training.Q.parent / 'official_deeppro'
    reference = source / 'log/sem_seg/RGB-T__2025-01-14_15-27__SoftLoUloss_DeepPro-Plus_DataL40/checkpoints/best_model.pth'
    frozen = training.load_module('joint_inference_frozen_loader', training.Q / 'top1_background_control_20261011/code/train_pilot.py')
    original, model_path = frozen.load_model(source, reference)
    initial_original = {k: v.detach().cpu().clone() for k, v in original.state_dict().items()}
    assert HEAD_NAMES <= set(dict(original.named_parameters()))
    assert sum(dict(original.named_parameters())[name].numel() for name in HEAD_NAMES) == 273
    immutable = {k: v for k, v in initial_original.items() if k not in HEAD_NAMES}
    checkpoint = BASE / 'training' / args.condition / 'run' / f'snapshot_{args.step:03d}.pth'
    run_path = checkpoint.with_name('RUN.json')
    run = json.loads(run_path.read_text())
    cp = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert run['status'] == 'complete' and cp['step'] == args.step and cp['condition'] == args.condition
    snapshots = {int(row['step']): row for row in run['snapshots']}
    assert training.sha(checkpoint) == snapshots[args.step]['sha256']
    assert cp['initial_checkpoint_sha256'] == training.sha(reference) == training.EXPECTED['checkpoint']
    sources = training.source_hashes(model_path)
    assert all(cp[key] == value and run[key] == value for key, value in sources.items())
    model = training.wrap_model(original, args.condition)
    model.load_state_dict(cp['model_state_dict'], strict=True)
    state = model.original.state_dict()
    assert set(state) == set(initial_original)
    assert all(torch.equal(state[k].detach().cpu(), v) for k, v in immutable.items())
    model = model.cuda().eval()
    model.set_inference_mode('normal')
    initial_model_hash = training.tensor_hash(model.state_dict().items())
    nonhead_hash = training.tensor_hash(immutable.items())
    assert initial_model_hash == snapshots[args.step]['model_state_sha256']
    assert nonhead_hash == cp['original_nonhead_state_sha256'] == run['original_nonhead_initial_sha256']
    raw_path = training.Q / 'stage1_goal/data' / args.sequence / 'raw_frames.npy'
    annotation_path = raw_path.with_name('annotations.json')
    raw = np.load(raw_path, mmap_mode='r', allow_pickle=False)
    assert raw.shape == (160, 512, 640) and raw.dtype == np.uint8
    destination = BASE / 'predictions' / args.condition / f'step_{args.step:04d}' / args.sequence
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(destination)
    (destination / 'windows').mkdir(parents=True, exist_ok=True)
    fused = np.zeros(raw.shape, np.float32)
    coverage = np.zeros(160, np.int32)
    metadata = {
        'status': 'RUNNING', 'condition': args.condition, 'sequence': args.sequence,
        'checkpoint': str(checkpoint), 'checkpoint_sha256': training.sha(checkpoint),
        'checkpoint_step': args.step, 'main_checkpoint_step': 2048,
        'checkpoint_is_main': args.step == 2048, 'checkpoint_selection': 'fixed2048_main;512and1024_process_only',
        'reference_checkpoint_sha256': training.sha(reference), 'training_run_sha256': training.sha(run_path),
        'normalization': [85.0799, 47.4845], 'official_starts': O, 'symmetric_starts': S,
        'diagnostic_starts': STARTS, 'raw_sha256': training.sha(raw_path),
        'annotations_sha256': training.sha(annotation_path), 'model_labels_access': False,
        'model_config': model.config, 'parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'trainable_parameter_names': [name for name, p in model.named_parameters() if p.requires_grad],
        'original_nonhead_state_sha256': nonhead_hash, 'initial_inference_model_state_sha256': initial_model_hash,
        'head_parameter_names': sorted(HEAD_NAMES), 'all_original_frozen': False,
        'tile_core': 128, 'tile_halo': 20, 'spatial_phase': 'absolute even tile input starts',
        'seed': 46, 'precision': 'FP32_no_AMP_no_TF32', 'window_timing': [], 'tiling_checks': [],
        'inference_source_path': str(Path(__file__).resolve()), 'inference_source_sha256': training.sha(__file__),
        **sources,
    }
    assert metadata['trainable_parameters'] == (273 if args.condition == 'H' else 373)
    began = time.perf_counter()
    try:
        with torch.no_grad():
            probe = torch.from_numpy(np.array((raw[:40, 176:272, 224:352].astype(np.float32)-85.0799)/47.4845, copy=True))[None, None].cuda()
            direct = model(probe)[1].cpu()
            tiled = tiled_logits(model, probe, 48, 20)
            torch.testing.assert_close(direct, tiled, atol=2e-4, rtol=2e-5)
            disagreement = int(((direct.sigmoid() >= .5) != (tiled.sigmoid() >= .5)).sum())
            assert disagreement == 0
            metadata['tiling_checks'].append({
                'crop_yxhw': [176, 224, 96, 128], 'max_logit_error': float((direct-tiled).abs().max()),
                'max_probability_error': float((direct.sigmoid()-tiled.sigmoid()).abs().max()),
                'threshold05_disagreements': disagreement, 'atol': 2e-4, 'rtol': 2e-5,
            })
            del probe, direct, tiled
            for start in STARTS:
                x = torch.from_numpy(np.array((raw[start:start+40].astype(np.float32)-85.0799)/47.4845, copy=True))[None, None].cuda()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                tick = time.perf_counter()
                logits = tiled_logits(model, x)
                torch.cuda.synchronize()
                seconds = time.perf_counter()-tick
                probabilities = logits.sigmoid()[0].numpy()
                assert np.isfinite(probabilities).all()
                tick = time.perf_counter()
                np.savez_compressed(destination/'windows'/f'window_{start:03d}.npz', probabilities=probabilities, start=start, end=start+40)
                if start in O:
                    fused[start:start+40] = np.maximum(fused[start:start+40], probabilities)
                    coverage[start:start+40] += 1
                metadata['window_timing'].append({
                    'start': start, 'model_plus_cpu_copy_seconds': seconds, 'window_write_seconds': time.perf_counter()-tick,
                    'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
                    'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(),
                })
                print(json.dumps({'condition': args.condition, 'step': args.step, 'sequence': args.sequence, 'window': start, 'seconds': seconds}), flush=True)
                del x, logits, probabilities
        assert initial_model_hash == training.tensor_hash(model.state_dict().items())
        assert nonhead_hash == training.tensor_hash((k,v) for k,v in model.original.state_dict().items() if k not in HEAD_NAMES)
        assert sources == training.source_hashes(model_path)
        assert np.all(coverage > 0) and int(coverage.sum()) == 200
        np.savez_compressed(destination/'official_fused.npz', probabilities=fused, coverage=coverage, starts=np.array(O))
        timing = metadata['window_timing']
        metadata.update(
            status='COMPLETE', elapsed_seconds=time.perf_counter()-began,
            frozen_nonhead_unchanged=True, all_model_state_unchanged_during_inference=True,
            final_inference_model_state_sha256=initial_model_hash,
            peak_cuda_allocated_bytes=max(row['peak_cuda_allocated_bytes'] for row in timing),
            peak_cuda_reserved_bytes=max(row['peak_cuda_reserved_bytes'] for row in timing),
            O_model_seconds=sum(row['model_plus_cpu_copy_seconds'] for row in timing if row['start'] in O),
            S_model_seconds=sum(row['model_plus_cpu_copy_seconds'] for row in timing if row['start'] in S),
            window_sha256={p.name: training.sha(p) for p in sorted((destination/'windows').glob('*.npz'))},
            fused_sha256=training.sha(destination/'official_fused.npz'),
        )
        training.save_json(destination/'COMPLETE.json', metadata)
    except BaseException as error:
        training.save_json(destination/'FAILED.json', {'status':'FAILED', 'error':repr(error), 'partial_metadata':metadata})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition', choices=['H','HL'], required=True)
    parser.add_argument('--step', choices=STEPS, type=int, required=True)
    parser.add_argument('--sequence', choices=SEQUENCES, required=True)
    infer(parser.parse_args())


if __name__ == '__main__':
    main()
