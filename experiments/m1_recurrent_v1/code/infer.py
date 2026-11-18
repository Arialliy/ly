"""Full-spatial M1 inference; the recurrent cell is never spatially tiled."""
from __future__ import annotations
import argparse
import fcntl
import json
from pathlib import Path
import time
import numpy as np
import torch
import runtime as rt


def load_trained(condition, step=2048, device='cuda'):
    checkpoint_path = rt.E / 'training' / condition / 'run' / f'snapshot_{step:03d}.pth'
    run_path = checkpoint_path.with_name('RUN.json')
    training = rt.read_json(run_path)
    assert training['status'] == 'complete' and training['final_step'] == rt.STEPS
    assert training['condition'] == condition and training['seed'] == 46
    assert training['source_sha256'] == rt.source_hashes()
    assert training['protocol_sha256'] == rt.protocol_hash()
    snapshots = {r['step']: r for r in training['snapshots']}
    assert snapshots[step]['sha256'] == rt.sha(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    assert checkpoint['step'] == step and checkpoint['condition'] == condition
    rt.setup(); model = rt.make_model(condition, device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    rt.assert_frozen(model, training['original_state_sha256'])
    assert rt.tensor_hash(model.state_dict()) == snapshots[step]['model_state_sha256']
    return model, training, checkpoint_path, run_path


def infer(condition, sequence, step=2048, device='cuda', frontend_core=128):
    """No cross-window state: model(adapter) holds state only inside each call."""
    if frontend_core != 128:
        raise ValueError('The registered inference frontend uses core128, halo16')
    destination = rt.E / 'predictions' / condition / f'step_{step:04d}' / sequence
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / 'execution.lock').open('a', encoding='utf-8') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return infer_locked(condition, sequence, step, device, destination)


def infer_locked(condition, sequence, step, device, destination):
    model, training, checkpoint, run_path = load_trained(condition, step, device)
    raw_path = rt.DATA / sequence / 'raw_frames.npy'
    raw = np.load(raw_path, mmap_mode='r', allow_pickle=False)
    assert raw.shape == (160, 512, 640) and raw.dtype == np.uint8
    model_hash = rt.tensor_hash(model.state_dict())
    identity = {'condition': condition, 'sequence': sequence, 'checkpoint_step': step,
        'main_checkpoint_step': rt.STEPS, 'checkpoint_is_main': step == rt.STEPS,
        'checkpoint_sha256': rt.sha(checkpoint), 'training_run_sha256': rt.sha(run_path),
        'raw_sha256': rt.sha(raw_path), 'annotations_sha256': rt.sha(raw_path.with_name('annotations.json')),
        'reference_checkpoint_sha256': rt.sha(rt.CP), 'source_sha256': rt.source_hashes(),
        'protocol_sha256': rt.protocol_hash(), 'initial_model_state_sha256': model_hash,
        'normalization': [85.0799, 47.4845], 'official_starts': rt.O, 'symmetric_starts': rt.S,
        'diagnostic_starts': rt.STARTS, 'seed': 46, 'precision': 'FP32_no_AMP_no_TF32',
        'spatial_execution': 'frozen frontend tiled128/halo16 stitched to full F/H; '
                             'one fullframe M1 call; original pointwise head in temporal chunks1',
        'frontend_tile_core': 128, 'frontend_tile_halo': 16, 'head_temporal_chunk': 1,
        'M1_spatial_tiling': False,
        'state_scope': 'one original40-frame window; reset for every model call and video',
        'model_labels_access': False, 'all_original_frozen': True,
        'parameters': training['parameters'], 'trainable_parameters': training['trainable_parameters'],
        'model_settings': rt.MODEL_SETTINGS, 'runtime_settings': rt.RUNTIME_SETTINGS}
    complete_path = destination / 'COMPLETE.json'
    partial_path = destination / 'IN_PROGRESS.json'
    if complete_path.exists():
        completed = rt.read_json(complete_path)
        for key, value in identity.items(): assert completed[key] == value, key
        for name, value in completed['window_sha256'].items(): assert rt.sha(destination / 'windows' / name) == value
        assert rt.sha(destination / 'official_fused.npz') == completed['fused_sha256']
        print({'condition': condition, 'sequence': sequence, 'status': 'ALREADY_COMPLETE'}, flush=True)
        return
    metadata = dict(identity, status='RUNNING', window_sha256={}, window_timing=[])
    if partial_path.exists():
        prior = rt.read_json(partial_path)
        for key, value in identity.items(): assert prior[key] == value, ('Resume identity', key)
        metadata = prior; metadata['status'] = 'RUNNING'
    (destination / 'windows').mkdir(exist_ok=True)
    fused = np.zeros(raw.shape, np.float32); coverage = np.zeros(160, np.int16)
    began = time.perf_counter()
    try:
        with torch.no_grad():
            for start in rt.STARTS:
                path = destination / 'windows' / f'window_{start:03d}.npz'
                if path.name in metadata['window_sha256']:
                    assert rt.sha(path) == metadata['window_sha256'][path.name]
                    with np.load(path, allow_pickle=False) as saved:
                        probability = saved['probabilities'].copy()
                        assert int(saved['start']) == start and int(saved['end']) == start + 40
                else:
                    if path.exists(): raise FileExistsError('Unreceipted window requires inspection: ' + str(path))
                    x = torch.from_numpy(np.array((raw[start:start+40].astype(np.float32) - 85.0799) / 47.4845,
                                                 copy=True))[None, None].to(device)
                    if torch.device(device).type == 'cuda':
                        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
                    tick = time.perf_counter()
                    logits = model.forward_full_spatial(x, frontend_core=128, frontend_halo=16,
                                                        head_temporal_chunk=1)[1]
                    probability = logits.sigmoid()[0].cpu().numpy()
                    if torch.device(device).type == 'cuda': torch.cuda.synchronize()
                    elapsed = time.perf_counter() - tick
                    assert probability.shape == (40, 512, 640) and np.isfinite(probability).all()
                    temporary = path.with_suffix('.npz.tmp')
                    with temporary.open('wb') as stream:
                        np.savez_compressed(stream, probabilities=probability, start=start, end=start+40)
                    temporary.replace(path)
                    metadata['window_sha256'][path.name] = rt.sha(path)
                    metadata['window_timing'].append({'start': start, 'model_plus_cpu_copy_seconds': elapsed,
                        'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated() if torch.device(device).type == 'cuda' else 0,
                        'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved() if torch.device(device).type == 'cuda' else 0})
                    rt.save_json(partial_path, metadata)
                    print({'condition': condition, 'sequence': sequence, 'window': start, 'seconds': elapsed}, flush=True)
                    del logits, x
                if start in rt.O:
                    fused[start:start+40] = np.maximum(fused[start:start+40], probability)
                    coverage[start:start+40] += 1
                del probability
        assert np.all(coverage > 0) and int(coverage.sum()) == 200
        rt.assert_frozen(model, training['original_state_sha256'])
        assert rt.tensor_hash(model.state_dict()) == model_hash
        fused_path = destination / 'official_fused.npz'
        temporary = fused_path.with_suffix('.npz.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, probabilities=fused, coverage=coverage, starts=np.array(rt.O))
        temporary.replace(fused_path)
        timing = metadata['window_timing']
        metadata.update(status='COMPLETE', fused_sha256=rt.sha(fused_path),
            elapsed_seconds=time.perf_counter() - began, final_model_state_sha256=model_hash,
            all_model_state_unchanged_during_inference=True, original_unchanged=True,
            O_model_seconds=sum(row['model_plus_cpu_copy_seconds'] for row in timing if row['start'] in rt.O),
            S_model_seconds=sum(row['model_plus_cpu_copy_seconds'] for row in timing if row['start'] in rt.S),
            union_model_seconds=sum(row['model_plus_cpu_copy_seconds'] for row in timing),
            peak_cuda_allocated_bytes=max(row['peak_cuda_allocated_bytes'] for row in timing))
        rt.save_json(complete_path, metadata)
    except BaseException as error:
        metadata.update(status='FAILED', error=repr(error)); rt.save_json(partial_path, metadata)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition', choices=rt.CONDITIONS, required=True)
    parser.add_argument('--sequence', choices=rt.SEQS, required=True)
    parser.add_argument('--step', choices=[128, 512, 1024, 2048], type=int, default=2048)
    parser.add_argument('--device', default='cuda'); parser.add_argument('--frontend-core', type=int, choices=[128], default=128)
    args = parser.parse_args(); infer(args.condition, args.sequence, args.step, args.device, args.frontend_core)
