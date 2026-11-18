#!/usr/bin/env python3
"""Original full-frame DeepPro-Plus inference at the fixed 2048-step endpoint."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch

import runtime as rt


def verify_checkpoint(condition, checkpoint):
    run_path = checkpoint.with_name('RUN.json')
    run = rt.read_json(run_path)
    cp = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert run['status'].lower() == 'complete'
    assert run['condition'] == cp['condition'] == condition
    assert run['seed'] == cp['seed'] == 46
    assert run['final_step'] == cp['step'] == 2048
    assert run['initial_checkpoint_sha256'] == cp['initial_checkpoint_sha256'] == rt.sha(rt.CP)
    assert run['model_source_sha256'] == cp['model_source_sha256'] == rt.sha(rt.MODEL_SOURCE)
    assert run['protocol_sha256'] == cp['protocol_sha256'] == rt.sha(rt.E / 'PROTOCOL_FREEZE.json')
    assert run['frozen_frontend_unchanged'] and run['bn_unchanged']
    snapshots = {int(row['step']): row for row in run['snapshots']}
    assert snapshots[2048]['sha256'] == rt.sha(checkpoint)
    return run, run_path


def run(condition, sequence, checkpoint=None):
    assert sequence in rt.SEQS and condition in ['A', 'B', 'C']
    rt.setup()
    checkpoint = Path(checkpoint) if checkpoint else rt.E / 'training' / condition / 'run/snapshot_2048.pth'
    training, run_path = verify_checkpoint(condition, checkpoint)
    freeze = rt.read_json(rt.E / 'PROTOCOL_FREEZE.json')
    for name, expected in freeze['files'].items():
        assert rt.sha(rt.E / name) == expected, ('Frozen experiment artifact changed', name)
    model, model_path = rt.load_model(checkpoint)
    names = rt.trainable_names(model)
    count = sum(p.numel() for name, p in model.named_parameters() if name in names)
    assert names == sorted(training['trainable_parameter_names'])
    assert count == training['trainable_parameters'] == 14593
    reference_model, _ = rt.load_model(device='cpu')
    original_frozen = rt.tensor_hash(rt.frozen_state(reference_model))
    original_bn = rt.tensor_hash(rt.bn_state(reference_model))
    assert rt.tensor_hash(rt.frozen_state(model)) == original_frozen
    assert rt.tensor_hash(rt.bn_state(model)) == original_bn
    del reference_model
    initial_state = rt.tensor_hash(model.state_dict())
    raw_path = rt.DATA / sequence / 'raw_frames.npy'
    raw = np.load(raw_path, mmap_mode='r', allow_pickle=False)
    assert raw.shape == (160, 512, 640) and raw.dtype == np.uint8
    release_receipt = rt.read_json(rt.C0 / sequence / 'COMPLETE.json')
    assert rt.sha(raw_path) == release_receipt['raw_sha256']
    destination = rt.E / 'predictions' / condition / sequence
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(destination)
    (destination / 'windows').mkdir(parents=True, exist_ok=True)
    fused = np.zeros(raw.shape, np.float32)
    coverage = np.zeros(160, np.int16)
    metadata = dict(status='RUNNING', condition=condition, sequence=sequence,
        checkpoint=str(checkpoint), checkpoint_sha256=rt.sha(checkpoint), checkpoint_step=2048,
        main_checkpoint_step=2048, checkpoint_is_main=True, checkpoint_selection='fixed_2048_no_selection',
        reference_checkpoint_sha256=rt.sha(rt.CP), model_source_sha256=rt.sha(model_path),
        protocol_sha256=rt.sha(rt.E / 'PROTOCOL_FREEZE.json'), training_run_sha256=rt.sha(run_path),
        trainable_parameter_names=names, trainable_parameters=count, parameters=70913,
        raw_sha256=rt.sha(raw_path), annotations_sha256=rt.sha(raw_path.with_name('annotations.json')),
        normalization=[85.0799, 47.4845], official_starts=rt.O, symmetric_starts=rt.S,
        diagnostic_starts=rt.STARTS, seed=46, precision='FP32_no_AMP_no_TF32',
        tile_core=128, tile_halo=16, spatial_phase='absolute_even_tile_input_origins',
        model_labels_access=False, auxiliary_modules_at_inference=False, original_forward_unchanged=True,
        initial_model_state_sha256=initial_state, frozen_frontend_state_sha256=original_frozen,
        bn_state_sha256=original_bn, inference_source_sha256=rt.sha(__file__),
        runtime_source_sha256=rt.sha(rt.E / 'code/runtime.py'), torch=torch.__version__,
        window_timing=[], tiling_checks=[])
    began = time.perf_counter()
    try:
        with torch.inference_mode():
            crop = raw[:40, 176:272, 224:352].astype(np.float32)
            probe = torch.from_numpy((crop - 85.0799) / 47.4845)[None, None].cuda()
            direct = model(probe)[1].cpu()
            tiled = rt.tiled_logits(model, probe, core=48, halo=16)
            torch.testing.assert_close(direct, tiled, atol=2e-4, rtol=2e-5)
            disagreements = int(((direct.sigmoid() >= .5) != (tiled.sigmoid() >= .5)).sum())
            assert disagreements == 0
            metadata['tiling_checks'].append(dict(crop_yxhw=[176, 224, 96, 128],
                max_logit_error=float((direct - tiled).abs().max()), threshold05_disagreements=disagreements))
            del probe, direct, tiled, crop
            for start in rt.STARTS:
                x = torch.from_numpy((raw[start:start + 40].astype(np.float32) - 85.0799) / 47.4845)[None, None].cuda()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                tick = time.perf_counter()
                probabilities = rt.tiled_logits(model, x).sigmoid()[0].numpy()
                torch.cuda.synchronize()
                seconds = time.perf_counter() - tick
                assert probabilities.dtype == np.float32 and np.isfinite(probabilities).all()
                tick = time.perf_counter()
                path = destination / 'windows' / f'window_{start:03d}.npz'
                np.savez_compressed(path, probabilities=probabilities, start=start, end=start + 40)
                if start in rt.O:
                    fused[start:start + 40] = np.maximum(fused[start:start + 40], probabilities)
                    coverage[start:start + 40] += 1
                row = dict(start=start, model_plus_cpu_copy_seconds=seconds,
                    write_seconds=time.perf_counter() - tick,
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
                metadata['window_timing'].append(row)
                print({'condition': condition, 'sequence': sequence, **row}, flush=True)
                del x, probabilities
        assert initial_state == rt.tensor_hash(model.state_dict())
        assert original_frozen == rt.tensor_hash(rt.frozen_state(model))
        assert original_bn == rt.tensor_hash(rt.bn_state(model))
        assert np.all(coverage > 0) and int(coverage.sum()) == 200
        np.savez_compressed(destination / 'official_fused.npz', probabilities=fused, coverage=coverage, starts=np.array(rt.O))
        timing = metadata['window_timing']
        metadata.update(status='COMPLETE', elapsed_seconds=time.perf_counter() - began,
            all_model_state_unchanged_during_inference=True, frozen_frontend_unchanged=True, bn_unchanged=True,
            final_model_state_sha256=initial_state,
            O_model_seconds=sum(r['model_plus_cpu_copy_seconds'] for r in timing if r['start'] in rt.O),
            S_model_seconds=sum(r['model_plus_cpu_copy_seconds'] for r in timing if r['start'] in rt.S),
            window_sha256={p.name: rt.sha(p) for p in sorted((destination / 'windows').glob('*.npz'))},
            fused_sha256=rt.sha(destination / 'official_fused.npz'))
        rt.save_json(destination / 'COMPLETE.json', metadata)
    except BaseException as error:
        rt.save_json(destination / 'FAILED.json', dict(status='FAILED', error=repr(error), metadata=metadata))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition', choices=['A', 'B', 'C'], required=True)
    parser.add_argument('--sequence', choices=rt.SEQS, required=True)
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    run(args.condition, args.sequence, args.checkpoint)
