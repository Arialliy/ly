#!/usr/bin/env python3
"""Re-export native FP32 pre-sigmoid scores without changing cached predictions.

Writes only beneath this script's directory. The old probability arrays remain
the sole source for actual detections. Any probability disagreement is a hard
failure, including differences that leave the 0.5 mask unchanged.
"""
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
Q = HERE.parents[1]
P = Q / 'stage1_goal'
R = Q.parent
SEQUENCES = ['DJI_0026_2', 'DJI_0067_2']
PLANS = {'O': [0, 36, 72, 108, 120], 'S': [0, 24, 60, 96, 120]}
STARTS = sorted(set().union(*map(set, PLANS.values())))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write_json(name, value):
    path = HERE / name
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    if (HERE / 'COMPLETE.json').exists():
        raise FileExistsError('Completed export already exists')
    if not (HERE.parent / 'PROTOCOL_FREEZE.json').is_file():
        raise FileNotFoundError('Root diagnostic protocol must be frozen before inference')
    raw_complete = {s: json.loads((P / 'predictions/official' / s / 'COMPLETE.json').read_text()) for s in SEQUENCES}
    reference = raw_complete[SEQUENCES[0]]
    source = R / 'official_deeppro'
    checkpoint = Path(reference['checkpoint'])
    source_files = [checkpoint, Path(reference['model_source']), Q / 'infer_with_windows.py',
                    Q / 'mechanism_followup/run_factorial.py', P / 'pilot/infer_pilot.py',
                    source / 'networks/layers/basic.py', source / 'networks/layers/TPro.py',
                    HERE.parent / 'PROTOCOL.md', HERE.parent / 'PROTOCOL_FREEZE.json', Path(__file__)]
    for seq in SEQUENCES:
        source_files.extend([P / 'data' / seq / 'raw_frames.npy', P / 'predictions/official' / seq / 'COMPLETE.json'])
        source_files.extend(P / 'predictions/official' / seq / 'windows' / f'window_{a:03d}.npz' for a in STARTS)
    hashes = {str(p): sha(p) for p in source_files}
    assert sha(checkpoint) == reference['checkpoint_sha256']
    assert sha(Path(reference['model_source'])) == reference['model_source_sha256']
    assert sha(P / 'pilot/infer_pilot.py') == reference['script_sha256']
    for seq, ref in raw_complete.items():
        assert ref['checkpoint_sha256'] == reference['checkpoint_sha256']
        assert ref['model_source_sha256'] == reference['model_source_sha256']
        assert sha(P / 'data' / seq / 'raw_frames.npy') == ref['raw_sha256']
        for a in STARTS:
            name = f'window_{a:03d}.npz'
            assert sha(P / 'predictions/official' / seq / 'windows' / name) == ref['window_sha256'][name]
    metadata = {'status': 'RUNNING', 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'sequences': SEQUENCES, 'starts': STARTS, 'plans': PLANS, 'source_sha256': hashes,
                'shape': [40, 512, 640], 'dtype': 'float32', 'seed': 46, 'tile_core': 128, 'tile_halo': 16,
                'normalization': [85.0799, 47.4845], 'interpolation': 'none: native output equals input spatial resolution',
                'score_definition': 'Native model output before sigmoid; arrangement diagnostic score is pointwise max of native window logits',
                'detection_definition': 'Unchanged old cached probability maximum and 0.5 matching; logits are diagnostic only',
                'sigmoid_location': 'CPU, after spatial tiling, exactly as cached infer_pilot.py',
                'torch_version': str(torch.__version__), 'cuda_version': str(torch.version.cuda),
                'gpu': torch.cuda.get_device_name(0), 'model_labels_access': False,
                'require_probability_bitwise_equal': True, 'window_comparisons': []}
    write_json('RUNNING.json', metadata)
    random.seed(46); np.random.seed(46); torch.manual_seed(46); torch.cuda.manual_seed_all(46)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    helper = mod('low_response_frozen_model_setup', Q / 'mechanism_followup/run_factorial.py')
    inference = mod('low_response_original_tiling', Q / 'infer_with_windows.py')
    model, model_file, epoch, radius = helper.model_setup(source, checkpoint, torch.device('cuda:0'))
    model.eval()
    metadata.update(checkpoint_epoch=epoch, receptive_radius_tyx=radius)
    begin = time.perf_counter()
    output_hashes = {}
    with torch.inference_mode():
        for seq in SEQUENCES:
            seqdir = HERE / seq
            seqdir.mkdir(exist_ok=True)
            raw = np.load(P / 'data' / seq / 'raw_frames.npy', mmap_mode='r', allow_pickle=False)
            assert raw.dtype == np.uint8 and raw.shape == (160, 512, 640)
            for start in STARTS:
                destination = seqdir / f'window_{start:03d}_logits.npy'
                if destination.exists():
                    raise FileExistsError(destination)
                x = torch.from_numpy(np.ascontiguousarray((raw[start:start+40].astype(np.float32) - 85.0799) / 47.4845))[None, None].cuda()
                logits = inference.tiled_logits(model, x, 128, 16)
                arr = logits[0].numpy()
                prob = logits.sigmoid()[0].numpy()
                assert arr.dtype == np.float32 and arr.shape == (40, 512, 640) and np.isfinite(arr).all()
                old_path = P / 'predictions/official' / seq / 'windows' / f'window_{start:03d}.npz'
                with np.load(old_path, allow_pickle=False) as cache:
                    old = cache['probabilities']
                diff = np.abs(prob - old)
                check = {'sequence': seq, 'start': start, 'probability_differing_pixels': int(np.count_nonzero(prob != old)),
                         'probability_max_abs_error': float(diff.max()),
                         'threshold_0p5_disagreements': int(np.count_nonzero((prob >= .5) != (old >= .5))),
                         'logit_min': float(arr.min()), 'logit_max': float(arr.max()),
                         'probability_saturated_0': int(np.count_nonzero(old == 0)),
                         'probability_saturated_1': int(np.count_nonzero(old == 1)),
                         'elapsed_seconds': time.perf_counter() - begin}
                metadata['window_comparisons'].append(check)
                write_json('RUNNING.json', metadata)
                if check['probability_differing_pixels']:
                    write_json('FAILED_COMPARISON.json', check)
                    raise AssertionError('Native re-export does not reproduce cached probabilities bitwise')
                np.save(destination, arr, allow_pickle=False)
                output_hashes[str(destination)] = sha(destination)
                print(json.dumps(check), flush=True)
                del x, logits, arr, prob, old, diff
            for plan, starts in PLANS.items():
                dest = seqdir / f'fused_{plan}_logits.npy'
                fused = np.lib.format.open_memmap(dest, mode='w+', dtype=np.float32, shape=(160, 512, 640))
                fused[:] = -np.inf
                coverage = np.zeros(160, dtype=np.int16)
                for start in starts:
                    arr = np.load(seqdir / f'window_{start:03d}_logits.npy', mmap_mode='r', allow_pickle=False)
                    np.maximum(fused[start:start+40], arr, out=fused[start:start+40])
                    coverage[start:start+40] += 1
                    del arr
                assert np.all(coverage > 0) and np.isfinite(fused).all()
                fused.flush()
                np.save(seqdir / f'coverage_{plan}.npy', coverage, allow_pickle=False)
                del fused
                output_hashes[str(dest)] = sha(dest)
                output_hashes[str(seqdir / f'coverage_{plan}.npy')] = sha(seqdir / f'coverage_{plan}.npy')
    assert all(sha(Path(path)) == expected for path, expected in hashes.items())
    metadata.update(status='COMPLETE', elapsed_seconds=time.perf_counter()-begin,
                    peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated()), output_sha256=output_hashes,
                    finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write_json('COMPLETE.json', metadata)
    print(json.dumps({'status': 'COMPLETE', 'elapsed_seconds': metadata['elapsed_seconds'],
                      'windows': len(metadata['window_comparisons']), 'all_cached_probabilities_bitwise_equal': True}), flush=True)


if __name__ == '__main__':
    main()
