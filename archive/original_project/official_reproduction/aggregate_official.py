#!/usr/bin/env python3
"""Restore full original sequence order, invoke original summary, compare author log."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import auc

E = Path(__file__).resolve().parent
S = E.parent / 'official_deeppro'
sys.path.insert(0, str(S))
from data_utils.TestDataLoader import TestIRSeqDataLoader
from write_results import writeNUDTMIRSDT_ROC


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--shards', type=int, default=4)
    a = p.parse_args()
    data = TestIRSeqDataLoader('NUDT-MIRSDT', str(E/'data/NUDT-MIRSDT'), 40, 4)
    ref = json.loads((E/'AUTHOR_REFERENCE_METRICS.json').read_text())
    assert data.seq_names == ref['sequence_order']
    n = len(data)
    threshold = np.array(ref['thresholds'])
    arrays = {k: np.zeros((n,len(threshold))) for k in ['false_num','true_num','target_num']}
    pixels = np.zeros(n)
    seen = set()
    intersection = union = 0
    records = []
    for i in range(a.shards):
        folder = a.run_dir / f'shard{i}'
        rec = json.loads((folder/'COMPLETE.json').read_text())
        records.append(rec)
        assert rec['status'] == 'complete' and rec['seed'] == 46 and not rec['source_dirty']
        assert rec['full_sequence_order'] == data.seq_names
        z = np.load(folder/'official_counts.npz', allow_pickle=False)
        np.testing.assert_array_equal(z['thresholds'], threshold)
        ids = z['sequence_indices'].tolist()
        assert set(ids).isdisjoint(seen)
        seen.update(ids)
        assert z['sequence_names'].tolist() == [data.seq_names[j] for j in ids]
        for k in arrays:
            arrays[k][ids] = z[k]
        pixels[ids] = z['pixels']
        intersection += int(z['intersection'])
        union += int(z['union'])
        for pr in rec['probabilities']:
            assert hashlib.sha256(Path(pr['path']).read_bytes()).hexdigest() == pr['sha256']
            q = np.load(pr['path'], allow_pickle=False)
            assert q['probabilities'].shape[0] == 100
            assert np.isfinite(q['probabilities']).all() and np.all(q['coverage'] > 0)
    assert seen == set(range(n))
    assert len({r['source_commit'] for r in records}) == 1
    assert len({r['checkpoint_sha256'] for r in records}) == 1
    logs = []
    writeNUDTMIRSDT_ROC(arrays['false_num'], arrays['true_num'], arrays['target_num'],
                       pixels, intersection, union, threshold, data, logs.append)
    (a.run_dir/'official_aggregate_eval.txt').write_text('\n'.join(logs)+'\n')
    np.savez(a.run_dir/'full_official_counts.npz', **arrays, pixels=pixels,
             thresholds=threshold, intersection=intersection, union=union,
             sequence_names=np.array(data.seq_names))
    curves = {}
    op = int(np.flatnonzero(threshold == 0.5)[0])
    low = ref['low_snr_indices']
    for group, ids in [('all',list(range(n))),('low_snr',low),('high_snr',[j for j in range(n) if j not in low])]:
        tp = arrays['true_num'][ids].sum(axis=0)
        fp = arrays['false_num'][ids].sum(axis=0)
        gt = arrays['target_num'][ids].sum(axis=0)
        pix = int(pixels[ids].sum())
        pd, fa = tp/gt, fp/pix
        curves[group] = {'auc':float(auc(fa,pd)), 'pixels':pix,
                         'threshold_0_5':{'tp':int(tp[op]),'gt':int(gt[op]),'fp_pixels':int(fp[op]),
                                          'pd_percent':float(pd[op]*100), 'fa':float(fa[op]),'fa_times_1e5':float(fa[op]*1e5)},
                         'thresholds':threshold.tolist(),'pd':pd.tolist(),'fa':fa.tolist()}
    comparison = {}
    for local, author in [('false_num','FalseNumAll'),('true_num','TrueNumAll'),('target_num','TgtNumAll')]:
        expected = np.array(ref[author])
        diff = arrays[local]-expected
        ij = np.argwhere(diff != 0)
        comparison[local] = {'exactly_equal':bool(len(ij)==0), 'different_cells':len(ij),
                             'max_absolute_difference':float(np.abs(diff).max()),
                             'differences':[{'sequence':data.seq_names[j], 'threshold':float(threshold[k]),
                                             'local':int(arrays[local][j,k]),'author':int(expected[j,k])} for j,k in ij]}
    iou = intersection/union
    report = {'status':'full_official_checkpoint_inference_completed', 'seed':46,
              'official_checkpoint_training_seed':'unknown','training_reproduced':False,
              'sequences':n,'frames':2000,'groups':curves,
              'iou':iou,'intersection':intersection,'union':union,
              'author_logged_iou':ref['author_logged_iou'],
              'author_logged_auc':ref['author_logged_auc'],
              'author_count_comparison':comparison,
              'checkpoint_sha256':records[0]['checkpoint_sha256'],
              'source_commit':records[0]['source_commit'],
              'execution':'original test.main and original evaluation functions; full native FP32 frames; 4 disjoint sequence shards; summary in original full order',
              'shard_paths':[str(a.run_dir/f'shard{i}') for i in range(a.shards)]}
    (a.run_dir/'RESULTS.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'all':curves['all']['threshold_0_5'],
                      'auc':curves['all']['auc'],'iou':iou,
                      'different_cells':{k:v['different_cells'] for k,v in comparison.items()}},indent=2))


if __name__ == '__main__':
    main()
