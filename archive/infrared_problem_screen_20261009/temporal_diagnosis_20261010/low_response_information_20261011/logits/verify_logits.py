#!/usr/bin/env python3
"""Independent CPU audit of exported logits and both existing fused predictions."""
from pathlib import Path
import datetime
import hashlib
import json
import time
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
Q = HERE.parents[1]
P = Q / 'stage1_goal'
PLANS = {'O': [0, 36, 72, 108, 120], 'S': [0, 24, 60, 96, 120]}


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    begin = time.perf_counter()
    torch.set_num_threads(2)
    complete = json.loads((HERE / 'COMPLETE.json').read_text())
    assert complete['status'] == 'COMPLETE'
    source_ok = {p: sha(p) == h for p, h in complete['source_sha256'].items()}
    output_ok = {p: sha(p) == h for p, h in complete['output_sha256'].items()}
    assert all(source_ok.values()) and all(output_ok.values())
    windows = []
    fused_checks = []
    for seq in complete['sequences']:
        scores = {}
        for start in complete['starts']:
            arr = np.load(HERE / seq / f'window_{start:03d}_logits.npy', mmap_mode='r', allow_pickle=False)
            scores[start] = arr
            cache = np.load(P / 'predictions/official' / seq / 'windows' / f'window_{start:03d}.npz', allow_pickle=False)
            old = cache['probabilities']
            n_diff = 0; max_diff = 0.0; mask_diff = 0
            for frame in range(40):
                probability = torch.from_numpy(arr[frame].copy()).sigmoid().numpy()
                n_diff += int(np.count_nonzero(probability != old[frame]))
                max_diff = max(max_diff, float(np.abs(probability - old[frame]).max()))
                mask_diff += int(np.count_nonzero((probability >= .5) != (old[frame] >= .5)))
            windows.append({'sequence': seq, 'start': start, 'probability_differing_pixels': n_diff,
                            'max_probability_error': max_diff, 'mask_0p5_differing_pixels': mask_diff})
            assert n_diff == 0 and mask_diff == 0
            del old, cache
        for plan, starts in PLANS.items():
            fused = np.load(HERE / seq / f'fused_{plan}_logits.npy', mmap_mode='r', allow_pickle=False)
            probability_path = (P / 'predictions/official' / seq / 'official_fused.npz' if plan == 'O'
                                else Q / 'problem_validity_20261011/simple_controls/symmetric5_probmax' / seq / 'fused.npz')
            old = np.load(probability_path, allow_pickle=False)['probabilities']
            record = {'sequence': seq, 'plan': plan, 'frames': 160, 'pixels': int(old.size),
                      'fused_logits_not_maximum': 0, 'probability_differing_pixels': 0,
                      'max_probability_error': 0.0, 'mask_0p5_differing_pixels': 0,
                      'mask_from_logit_zero_differing_pixels': 0,
                      'old_fused_probability_sha256': sha(probability_path),
                      'probability_saturated_0': int(np.count_nonzero(old == 0)),
                      'probability_saturated_1': int(np.count_nonzero(old == 1)),
                      'fused_logit_min': float(fused.min()), 'fused_logit_max': float(fused.max())}
            coverage = np.zeros(160, dtype=np.int16)
            for t in range(160):
                refs = [scores[a][t-a] for a in starts if a <= t < a + 40]
                coverage[t] = len(refs)
                actual = np.maximum.reduce(refs)
                record['fused_logits_not_maximum'] += int(np.count_nonzero(actual != fused[t]))
                probability = torch.from_numpy(fused[t].copy()).sigmoid().numpy()
                record['probability_differing_pixels'] += int(np.count_nonzero(probability != old[t]))
                record['max_probability_error'] = max(record['max_probability_error'], float(np.abs(probability-old[t]).max()))
                record['mask_0p5_differing_pixels'] += int(np.count_nonzero((probability >= .5) != (old[t] >= .5)))
                record['mask_from_logit_zero_differing_pixels'] += int(np.count_nonzero((fused[t] >= 0) != (old[t] >= .5)))
            assert np.array_equal(coverage, np.load(HERE / seq / f'coverage_{plan}.npy', allow_pickle=False))
            assert int(coverage.sum()) == 200 and int(coverage.min()) > 0
            for key in ['fused_logits_not_maximum', 'probability_differing_pixels', 'mask_0p5_differing_pixels', 'mask_from_logit_zero_differing_pixels']:
                assert record[key] == 0, record
            fused_checks.append(record)
            del old, fused
        del scores
    report = {'status': 'PASS', 'completed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'verifier_sha256': sha(__file__), 'source_fingerprints_unchanged': all(source_ok.values()),
              'output_fingerprints_valid': all(output_ok.values()), 'source_files_checked': len(source_ok),
              'output_files_checked': len(output_ok), 'window_checks': windows, 'fused_checks': fused_checks,
              'total_window_pixels_checked': sum(40*512*640 for x in windows),
              'total_fused_pixels_checked': sum(r['pixels'] for r in fused_checks),
              'elapsed_seconds': time.perf_counter() - begin,
              'scope': 'CPU re-sigmoid and independent pixelmax; no GPU rerun, no metric changes, no inverse sigmoid'}
    (HERE / 'INDEPENDENT_VERIFICATION.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'windows': len(windows), 'arrangement_maps': len(fused_checks),
                      'fused_pixels': report['total_fused_pixels_checked'], 'elapsed_seconds': report['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
