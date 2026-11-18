#!/usr/bin/env python3
"""Independent CPU audit of saved inference; never runs or edits the inference wrapper.

Recompute ShootingRules at 0.5 and strict-mask IoU from every probability frame.
Reassemble the saved 27-threshold counts and parse the original author log directly.
All output belongs to this script's directory. No training or new inference is run.
"""
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image
import torch
from sklearn.metrics import auc

OUT = Path(__file__).resolve().parent
E = OUT.parent
S = E.parent / 'official_deeppro'
RUN = E / 'full_test_seed46'
DATA = E / 'data/NUDT-MIRSDT'
EXPERIMENT = 'NUDT-MIRSDT__2024-12-28_16-21__SoftLoUloss_DeepPro-Plus_DataL40'
SOURCE_LOG = S / 'log/sem_seg' / EXPERIMENT / 'eval.txt'
sys.path.insert(0, str(S))
from ShootingRules import ShootingRules
from data_utils.TestDataLoader import TestIRSeqDataLoader
from write_results import writeNUDTMIRSDT_ROC


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def equal(a, b, context):
    if not np.array_equal(a, b):
        raise AssertionError(context)


def source_check():
    snapshot = read_json(E / 'SOURCE_SNAPSHOT.json')
    actual = {str(p.relative_to(S)) for p in S.rglob('*')
              if p.is_file() and '.git' not in p.relative_to(S).parts}
    assert actual == set(snapshot['tracked_sha256']), 'Source file-set changed'
    mismatches = [name for name, digest in snapshot['tracked_sha256'].items()
                  if sha(S / name) != digest]
    assert not mismatches, mismatches
    return {'tracked_files': len(actual), 'all_hashes_match': True,
            'commit': snapshot['git_commit'],
            'snapshot_sha256': sha(E / 'SOURCE_SNAPSHOT.json')}


def source_literals():
    test_ast = ast.parse((S / 'test.py').read_text())
    thresholds = None
    for node in ast.walk(test_ast):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'Th_Seg' for t in node.targets):
            thresholds = np.asarray(ast.literal_eval(node.value.args[0]), dtype=float)
    summary_ast = ast.parse((S / 'write_results.py').read_text())
    summary = next(n for n in summary_ast.body if isinstance(n, ast.FunctionDef) and n.name == 'writeNUDTMIRSDT_ROC')
    subsets = {}
    for node in summary.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ('low_snr', 'high_snr'):
            subsets[node.targets[0].id] = ast.literal_eval(node.value)
    assert thresholds is not None and len(thresholds) == 27
    return thresholds, subsets


def parse_original_log():
    sections = {}
    section = None
    logged_iou = None
    for line in SOURCE_LOG.read_text().splitlines():
        head = re.search(r'(Sequence\d+|Low SNR|High SNR|Final) results:', line)
        if head:
            section = head.group(1)
            assert section not in sections, 'Repeated author evaluation section'
            sections[section] = {'rows': []}
            match_auc = re.search(r'AUC:([\d.]+)', line)
            if match_auc:
                sections[section]['auc'] = float(match_auc.group(1))
        row = re.search(r'Th_Seg = (\S+):\s*PD:\[(\d+)/(\d+),\s*(\S+)\]\s*FA:\[(\d+),\s*(\S+)\]', line)
        if row:
            assert section is not None
            th, tp, gt, pd, fp, fa = row.groups()
            sections[section]['rows'].append({'threshold': float(th), 'tp': int(tp), 'gt': int(gt),
                                              'fp': int(fp), 'pd_logged': float(pd), 'fa_logged': float(fa)})
        match_iou = re.search(r'Eval avg class IoU of prediction:\s*([\d.]+)', line)
        if match_iou:
            assert logged_iou is None
            logged_iou = float(match_iou.group(1))
    assert len(sections) == 23 and logged_iou is not None
    assert all(len(v['rows']) == 27 for v in sections.values())
    return sections, logged_iou


def main():
    started = datetime.now(timezone.utc).isoformat()
    print('START independent CPU full-result recheck', flush=True)
    source_before = source_check()
    thresholds, subsets = source_literals()
    op = int(np.flatnonzero(thresholds == 0.5)[0])
    raw_lines = (DATA / 'test.txt').read_text().splitlines()
    assert len(raw_lines) == len(set(raw_lines)) == 2000
    sequence_order = list(dict.fromkeys(x.split('/')[0] for x in raw_lines))
    assert len(sequence_order) == 20
    assert set(Counter(x.split('/')[0] for x in raw_lines).values()) == {100}
    dataset = TestIRSeqDataLoader('NUDT-MIRSDT', str(DATA), 40, 4)
    assert sequence_order == dataset.seq_names
    reference = read_json(E / 'AUTHOR_REFERENCE_METRICS.json')
    results = read_json(RUN / 'RESULTS.json')
    report = (E / 'REPORT.md').read_text()
    equal(reference['sequence_order'], sequence_order, 'Reference sequence order')
    equal(reference['thresholds'], thresholds, 'Reference thresholds')
    equal(reference['low_snr_indices'], subsets['low_snr'], 'Reference low-SNR split')
    equal(reference['low_snr_sequence_names'], [sequence_order[x] for x in subsets['low_snr']], 'Reference low-SNR names')
    assert set(subsets['low_snr']).isdisjoint(subsets['high_snr'])
    assert sorted(subsets['low_snr'] + subsets['high_snr']) == list(range(20))
    assert results['sequences'] == 20 and results['frames'] == 2000
    assert results['source_commit'] == source_before['commit']
    checkpoint = S / 'log/sem_seg' / EXPERIMENT / 'checkpoints/best_model.pth'
    checkpoint_sha = sha(checkpoint)
    assert checkpoint_sha == results['checkpoint_sha256']
    author, author_iou = parse_original_log()
    equal([k for k in author if k.startswith('Sequence')], sequence_order, 'Original author order')
    assert reference['source_sha256'] == sha(SOURCE_LOG)
    assert Path(reference['source_path']).resolve() == SOURCE_LOG.resolve()
    author_counts = {k: np.asarray([[r[field] for r in author[name]['rows']] for name in sequence_order], dtype=np.int64)
                     for k, field in [('false_num', 'fp'), ('true_num', 'tp'), ('target_num', 'gt')]}
    for key, refkey in [('false_num', 'FalseNumAll'), ('true_num', 'TrueNumAll'), ('target_num', 'TgtNumAll')]:
        equal(author_counts[key], reference[refkey], 'Parsed author vs reference ' + key)
    for name in author:
        equal([r['threshold'] for r in author[name]['rows']], thresholds, 'Author threshold order ' + name)

    saved = {k: np.zeros((20, 27), dtype=np.int64) for k in author_counts}
    saved_pixels = np.zeros(20, dtype=np.int64)
    recomputed = np.zeros((20, 3), dtype=np.int64)
    sequence_rows = []
    frame_rows = []
    input_hashes = {str(p): sha(p) for p in [E / 'SOURCE_SNAPSHOT.json', E / 'REPORT.md',
                    E / 'AUTHOR_REFERENCE_METRICS.json', RUN / 'RESULTS.json', RUN / 'full_official_counts.npz', DATA / 'test.txt']}
    data_hashes = {}
    seen = set()
    total_intersection = total_union = 0
    recorded_intersection = recorded_union = 0
    rules = ShootingRules()
    for shard in range(4):
        folder = RUN / f'shard{shard}'
        complete_path = folder / 'COMPLETE.json'
        counts_path = folder / 'official_counts.npz'
        input_hashes[str(complete_path)] = sha(complete_path)
        input_hashes[str(counts_path)] = sha(counts_path)
        rec = read_json(complete_path)
        assert rec['status'] == 'complete' and rec['seed'] == 46 and rec['source_dirty'] == ''
        assert rec['training_performed'] is False and rec['official_checkpoint_training_seed'] == 'unknown'
        assert rec['source_commit'] == source_before['commit']
        assert rec['source_snapshot_sha256'] == source_before['snapshot_sha256']
        assert rec['checkpoint_sha256'] == checkpoint_sha
        assert Path(rec['loaded_checkpoint_path']).resolve() == checkpoint.resolve()
        equal(rec['full_sequence_order'], sequence_order, 'Complete order')
        ids = rec['sequence_indices']
        equal(ids, list(range(shard, 20, 4)), 'Shard index pattern')
        assert not seen.intersection(ids)
        seen.update(ids)
        names = [sequence_order[j] for j in ids]
        equal(rec['sequences'], names, 'Complete sequence names')
        equal([p['sequence'] for p in rec['probabilities']], names, 'Probability metadata order')
        assert len(rec['probabilities']) == 5
        assert sorted(p.name for p in (folder / 'probabilities').iterdir()) == sorted(name + '.npz' for name in names)
        expected_windows = [[name, sample[0][2], sample[-1][2]] for j, name in zip(ids, names)
                            for sample in dataset[j].samplelist]
        equal(rec['windows'], expected_windows, 'Official loader window plan')
        shard_i = shard_u = 0
        with np.load(counts_path, allow_pickle=False) as counts:
            equal(counts['thresholds'], thresholds, 'Shard thresholds')
            equal(counts['sequence_indices'], ids, 'Shard count indices')
            equal(counts['sequence_names'], names, 'Shard count order')
            for key in saved:
                assert counts[key].shape == (5, 27)
                assert np.isfinite(counts[key]).all() and (counts[key] >= 0).all()
                equal(counts[key], counts[key].astype(np.int64), 'Count is integer')
                saved[key][ids] = counts[key]
            saved_pixels[ids] = counts['pixels']
            recorded_intersection += int(counts['intersection'])
            recorded_union += int(counts['union'])
            for local_idx, (j, name, info) in enumerate(zip(ids, names, rec['probabilities'])):
                probability_path = folder / 'probabilities' / (name + '.npz')
                assert Path(info['path']).resolve() == probability_path.resolve()
                digest = sha(probability_path)
                assert digest == info['sha256'], 'Probability SHA256 mismatch'
                input_hashes[str(probability_path)] = digest
                with np.load(probability_path, allow_pickle=False) as archive:
                    probs = archive['probabilities']
                    coverage = archive['coverage']
                assert probs.dtype == np.float32 and probs.ndim == 3 and probs.shape[0] == 100
                assert list(probs.shape) == info['shape']
                assert np.isfinite(probs).all() and (probs >= 0).all() and (probs <= 1).all()
                expected_coverage = np.zeros(100, dtype=np.int32)
                for seq, first, last in expected_windows:
                    if seq == name:
                        expected_coverage[first:last + 1] += 1
                equal(coverage, expected_coverage, 'Exact overlap coverage')
                assert coverage.min() == 1 and coverage.max() == 2 and coverage.sum() == 120
                stems = [Path(line).stem for line in raw_lines if line.split('/')[0] == name]
                assert stems == [f'{frame:05d}' for frame in range(1, 101)]
                for kind in ('images', 'masks', 'masks_centroid'):
                    equal(sorted(p.name for p in (DATA / name / kind).iterdir()), [x + '.png' for x in stems], 'Data filename set ' + name + '/' + kind)
                seq_i = seq_u = equal_half = 0
                for t, stem in enumerate(stems):
                    arrays = {}
                    for kind in ('images', 'masks', 'masks_centroid'):
                        path = DATA / name / kind / (stem + '.png')
                        data_hashes[str(path)] = sha(path)
                        with Image.open(path) as image:
                            image.load()
                            arrays[kind] = np.asarray(image).copy()
                        assert arrays[kind].shape[:2] == probs.shape[1:], 'Native size mismatch'
                    mask = np.asarray(arrays['masks'], dtype=np.uint8) / 255.0
                    if mask.ndim == 3:
                        mask = mask[:, :, 0]
                    mask[mask > 0] = 1.0
                    centroid = np.asarray(arrays['masks_centroid'], dtype=np.uint8) / 255.0
                    assert centroid.ndim == 2
                    fp, tp, gt = rules(probs[t:t + 1], centroid[None], 0.5)
                    recomputed[j] += [fp, tp, gt]
                    pred = probs[t] > 0.5
                    intersection = int(np.logical_and(pred, mask > 0).sum())
                    union = int(np.logical_or(pred, mask > 0).sum())
                    seq_i += intersection
                    seq_u += union
                    equal_half += int(np.count_nonzero(probs[t] == 0.5))
                    frame_rows.append({'sequence': name, 'frame': t + 1, 'fp_pixels': int(fp), 'tp': int(tp), 'gt': int(gt),
                                       'intersection': intersection, 'union': union})
                equal(recomputed[j], [saved[k][j, op] for k in ('false_num', 'true_num', 'target_num')], 'Recomputed ShootingRules vs saved ' + name)
                equal(recomputed[j], [author_counts[k][j, op] for k in ('false_num', 'true_num', 'target_num')], 'Recomputed ShootingRules vs author ' + name)
                assert probs.size == saved_pixels[j]
                shard_i += seq_i
                shard_u += seq_u
                sequence_rows.append({'sequence': name, 'official_index': j, 'shard': shard, 'frames': 100,
                    'shape': list(probs.shape), 'dtype': str(probs.dtype), 'pixels': probs.size,
                    'probability_sha256': digest, 'coverage_histogram': dict(Counter(map(int, coverage))),
                    'fp_pixels': int(recomputed[j, 0]), 'tp': int(recomputed[j, 1]), 'gt': int(recomputed[j, 2]),
                    'intersection': seq_i, 'union': seq_u, 'probabilities_exactly_0_5': equal_half})
                print(f'CHECKED {name}: 100 frames; TP/GT {recomputed[j,1]}/{recomputed[j,2]}; FP {recomputed[j,0]}; I/U {seq_i}/{seq_u}', flush=True)
            assert shard_i == int(counts['intersection']) and shard_u == int(counts['union']), 'Shard IoU counts mismatch'
        total_intersection += shard_i
        total_union += shard_u
    assert seen == set(range(20)) and len(frame_rows) == 2000 and len(data_hashes) == 6000
    sequence_rows.sort(key=lambda x: x['official_index'])
    frame_rows.sort(key=lambda x: (sequence_order.index(x['sequence']), x['frame']))
    with np.load(RUN / 'full_official_counts.npz', allow_pickle=False) as merged:
        equal(merged['sequence_names'], sequence_order, 'Merged full sequence order')
        equal(merged['thresholds'], thresholds, 'Merged thresholds')
        equal(merged['pixels'], saved_pixels, 'Merged pixel totals')
        for key in saved:
            equal(merged[key], saved[key], 'Shard reassembly vs merged ' + key)
        assert int(merged['intersection']) == total_intersection and int(merged['union']) == total_union
    assert total_intersection == recorded_intersection == results['intersection']
    assert total_union == recorded_union == results['union']
    iou = total_intersection / total_union
    assert iou == results['iou']
    assert float(f'{iou:.6f}') == author_iou == results['author_logged_iou'] == reference['author_logged_iou']
    groups = {}
    group_ids = {'all': list(range(20)), **subsets}
    section_keys = {'all': 'Final', 'low_snr': 'Low SNR', 'high_snr': 'High SNR'}
    for group, ids in group_ids.items():
        pixels = int(saved_pixels[ids].sum())
        sums = {key: counts[ids].sum(axis=0) for key, counts in saved.items()}
        pd = sums['true_num'] / sums['target_num']
        fa = sums['false_num'] / pixels
        assert (np.diff(fa) <= 0).all()
        # Explicit adjacent trapezoids in the existing 27-threshold order.
        area = float(np.sum((fa[:-1] - fa[1:]) * (pd[:-1] + pd[1:]) * 0.5))
        assert abs(area - float(auc(fa, pd))) < 1e-15
        claimed = results['groups'][group]
        assert pixels == claimed['pixels']
        equal(claimed['thresholds'], thresholds, 'Result thresholds ' + group)
        equal(claimed['pd'], pd, 'Result Pd curve ' + group)
        equal(claimed['fa'], fa, 'Result Fa curve ' + group)
        assert abs(area - claimed['auc']) < 1e-15
        at_half = {'tp': int(sums['true_num'][op]), 'gt': int(sums['target_num'][op]), 'fp_pixels': int(sums['false_num'][op]),
                   'pd_percent': float(pd[op] * 100), 'fa': float(fa[op]), 'fa_times_1e5': float(fa[op] * 1e5)}
        assert at_half == claimed['threshold_0_5']
        author_section = author[section_keys[group]]
        author_sums = {key: counts[ids].sum(axis=0) for key, counts in author_counts.items()}
        for key, field in [('false_num', 'fp'), ('true_num', 'tp'), ('target_num', 'gt')]:
            equal(author_sums[key], [row[field] for row in author_section['rows']], 'Original author aggregate counts ' + group + '/' + key)
        author_pd = author_sums['true_num'] / author_sums['target_num']
        author_fa = author_sums['false_num'] / pixels
        author_area = float(np.sum((author_fa[:-1] - author_fa[1:]) * (author_pd[:-1] + author_pd[1:]) * 0.5))
        assert float(f'{area:.5f}') == author_section['auc'] == reference['author_logged_auc'][group] == results['author_logged_auc'][group]
        assert float(f'{author_area:.5f}') == author_section['auc']
        for row, apd, afa in zip(author_section['rows'], author_pd, author_fa):
            assert row['pd_logged'] == float(f'{apd:.5f}')
            assert row['fa_logged'] == float(f'{afa:e}')
        groups[group] = {'sequences': [sequence_order[j] for j in ids], 'pixels': pixels, 'threshold_0_5': at_half,
                         'auc_local_counts': area, 'auc_author_counts_with_verified_local_pixels': author_area,
                         'auc_local_minus_author_counts': area - author_area, 'author_logged_auc': author_section['auc'],
                         'thresholds': thresholds.tolist(), 'tp': sums['true_num'].tolist(), 'gt': sums['target_num'].tolist(),
                         'fp_pixels': sums['false_num'].tolist(), 'pd': pd.tolist(), 'fa': fa.tolist()}
    differences = {}
    for key in saved:
        delta = saved[key] - author_counts[key]
        cells = np.argwhere(delta != 0)
        details = [{'sequence': sequence_order[int(j)], 'threshold': float(thresholds[k]),
                    'local': int(saved[key][j,k]), 'author': int(author_counts[key][j,k])} for j,k in cells]
        differences[key] = {'exactly_equal': len(cells) == 0, 'different_cells': len(cells),
                            'max_absolute_difference': int(np.abs(delta).max()), 'differences': details}
        assert differences[key] == results['author_count_comparison'][key]
    assert differences['true_num']['different_cells'] == differences['target_num']['different_cells'] == 0
    assert differences['false_num']['different_cells'] == 20 and differences['false_num']['max_absolute_difference'] == 3
    assert all(row['threshold'] <= 1e-6 for row in differences['false_num']['differences'])
    # Use the untouched original summary function as an additional output witness.
    official_lines = []
    writeNUDTMIRSDT_ROC(saved['false_num'], saved['true_num'], saved['target_num'], saved_pixels,
                       total_intersection, total_union, thresholds, dataset, official_lines.append)
    official_text = '\n'.join(official_lines) + '\n'
    assert official_text == (RUN / 'official_aggregate_eval.txt').read_text()
    (OUT / 'independent_official_aggregate_eval.txt').write_text(official_text)
    readme = (S / 'README.md').read_text()
    assert '| DeepPro-Plus | 99.24 | 1.65 | 0.9955 |' in readme
    assert '| DeepPro-Plus | 99.71 | 2.69 | 0.9978 |' in readme
    for group, expected in [('all', ('99.71','2.69','0.9978')), ('low_snr', ('99.24','1.65','0.9955'))]:
        row = groups[group]
        assert (f"{row['threshold_0_5']['pd_percent']:.2f}", f"{row['threshold_0_5']['fa_times_1e5']:.2f}", f"{row['auc_local_counts']:.4f}") == expected
    assert '官方发布权重的推理复现' in report and '没有从头训练' in report
    assert '每项相差1–3个像素' in report and '≤1e-6' in report
    assert '不能把猜测的浮点原因写成已验证事实' in report
    assert '不能代替真实运动变化片段中的可靠性比较' in report
    source_after = source_check()
    assert source_before == source_after
    assert all(sha(path) == digest for path, digest in input_hashes.items()), 'Input artifact changed during recheck'
    warnings = [
        'FP differs from the original author log at 20/540 sequence-threshold cells, by 1–3 pixels, only at thresholds <= 1e-6; 0.5 counts match exactly. Cause is not established.',
        'The author did not supply floating-point probability maps or exact IoU intersection/union; pixelwise author equivalence and exact unrounded author IoU cannot be established.',
        'Only threshold 0.5 was independently recomputed from all saved probabilities and ground truths. The other 26 thresholds were independently reaggregated from saved counts, not rerun from probabilities.',
        'This verifies saved official-checkpoint inference results, not from-scratch training, the unknown checkpoint training seed, or real-video performance.',
    ]
    if '独立完整结果复算：PENDING' in report:
        warnings.append('REPORT.md still marks independent full-result recheck PENDING; this audit leaves REPORT.md untouched and supplies completion evidence separately.')
    audit = {'status': 'PASS_WITH_WARNINGS', 'started_at_utc': started, 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Saved official-checkpoint inference results only; independent CPU metric recheck, no training and no new model inference',
        'environment': {'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__,
                        'cuda_available': torch.cuda.is_available(), 'device': 'CPU',
                        'PYTHONDONTWRITEBYTECODE': os.environ.get('PYTHONDONTWRITEBYTECODE'),
                        'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS'), 'MKL_NUM_THREADS': os.environ.get('MKL_NUM_THREADS'),
                        'OPENBLAS_NUM_THREADS': os.environ.get('OPENBLAS_NUM_THREADS')},
        'source_snapshot_before_and_after': source_before, 'checkpoint_sha256': checkpoint_sha,
        'original_author_log_sha256': sha(SOURCE_LOG), 'sequence_order': sequence_order, 'sequences': 20,
        'frames_recomputed': len(frame_rows), 'decoded_data_files': len(data_hashes), 'saved_probabilities_verified': len(sequence_rows),
        'shooting_rules': {'implementation': str(S / 'ShootingRules.py'), 'threshold': 0.5, 'comparison': '>=',
                           'target': 'masks_centroid', 'all_per_sequence_counts_match_shards_and_author': True},
        'mask_iou': {'threshold': 0.5, 'comparison': '>', 'target': 'masks', 'intersection': total_intersection,
                     'union': total_union, 'iou': iou, 'author_logged_iou': author_iou, 'matches_at_author_precision': True,
                     'probabilities_exactly_0_5': sum(row['probabilities_exactly_0_5'] for row in sequence_rows)},
        'groups': groups, 'author_count_comparison': differences, 'per_sequence': sequence_rows,
        'checks': {'source_hashes_and_file_set_before_after': 'PASS', 'four_disjoint_complete_shards': 'PASS',
                   '20_sequence_order_2000_frame_coverage': 'PASS', '6000_data_files_decoded_native_dimensions': 'PASS',
                   'probability_sha256_dtype_range_shape_coverage': 'PASS', 'reference_json_vs_original_log': 'PASS',
                   'independent_all_frame_shooting_rules_at_0_5': 'PASS', 'independent_strict_mask_iou_all_frames': 'PASS',
                   '27_threshold_saved_counts_reassembly': 'PASS', 'independent_auc_formula_and_original_summary': 'PASS',
                   'readme_report_precision': 'PASS', 'report_numerical_and_scope_caveats': 'PASS'},
        'warnings': warnings, 'input_artifacts_unchanged_before_after': True,
        'script_sha256': sha(__file__)}
    (OUT / 'RESULT_RECHECK.json').write_text(json.dumps(audit, indent=2) + '\n')
    (OUT / 'per_frame_recomputed.json').write_text(json.dumps(frame_rows, indent=2) + '\n')
    (OUT / 'input_hashes.json').write_text(json.dumps({'artifacts': input_hashes, 'decoded_data_files': data_hashes}, indent=2) + '\n')
    np.savez(OUT / 'independent_reassembled_counts.npz', **saved, pixels=saved_pixels, thresholds=thresholds,
             recomputed_threshold_0_5=recomputed, intersection=total_intersection, union=total_union,
             sequence_names=np.asarray(sequence_order))
    lines = ['# Independent full-result recheck', '', '**PASS with documented numerical limitations.**', '',
        'Read-only input audit performed in the existing CPU Docker environment; no GPU, new model inference, package install, or training was required.', '',
        '## Evidence checked', '',
        '- All 20 sequences appear exactly once in the official `test.txt` order after assembling four disjoint shards. Each sequence has 100 frames: 2,000 probability frames total.',
        '- All 20 probability archive SHA256 hashes match COMPLETE metadata. Every array is finite FP32 in [0,1], uses native image dimensions, and has exact coverage for windows [0,39], [36,75], [60,99].',
        '- All 6,000 images/masks/masks_centroid PNGs were decoded, dimension-checked, and hashed. The source snapshot matches all 140 tracked files before and after; input result artifacts are unchanged.',
        '- The original author `eval.txt` was parsed independently. All 20×27 reference integer TP/GT/FP values, threshold order, sequence order, and aggregate split membership agree with AUTHOR_REFERENCE_METRICS.json.',
        '- The untouched official ShootingRules was called for every probability frame against centroid ground truth at **≥0.5**. All 20 per-sequence TP/GT/FP counts exactly match the shards, merged results, and original author log.',
        '- Pixel IoU was independently computed against full masks using **>0.5**. Intersection = 50,785; union = 59,156; IoU = 0.8584927987017378. The author log prints 0.858493. No saved probability equals exactly 0.5.',
        '- All 27 saved threshold arrays were reassembled in original order and compared with the merged NPZ. AUC was independently integrated by adjacent trapezoids and checked against the untouched official summary function; the regenerated summary log is byte-identical to the saved local aggregate log.', '',
        '| Set | TP / GT at 0.5 | FP pixels | Pixel denominator | Pd (%) | Fa (×10⁻⁵) | AUC |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for group in ('all', 'low_snr', 'high_snr'):
        row = groups[group]
        half = row['threshold_0_5']
        lines.append(f"| {group} | {half['tp']} / {half['gt']} | {half['fp_pixels']} | {row['pixels']} | {half['pd_percent']:.9f} | {half['fa_times_1e5']:.9f} | {row['auc_local_counts']:.12f} |")
    lines += ['', '## Warnings and interpretation', ''] + ['- ' + warning for warning in warnings]
    lines += ['', 'All 540 TP cells and all 540 GT cells equal the author log. FP has exactly the 20 reported differences, with maximum absolute difference 3. AUC matches the author log to 5 decimal places and the README to 4; Pd/Fa match README display precision. The report correctly discloses this boundary and does not establish the source of tiny FP differences.', '',
        'Author-count AUC calculations reuse the independently verified local pixel denominators because the original log does not publish exact denominators. Author IoU is only available at the printed 6-decimal precision.', '',
        '## Files', '', '`RESULT_RECHECK.json` contains checks and detailed comparisons; `per_frame_recomputed.json` contains all 2,000 frame counts and intersections/unions; `input_hashes.json` records artifact and decoded-data hashes; `independent_reassembled_counts.npz` records independent arrays; `independent_official_aggregate_eval.txt` records the original summary output.', '']
    (OUT / 'RESULT_RECHECK.md').write_text('\n'.join(lines))
    print(json.dumps({'status': audit['status'], 'frames': len(frame_rows), 'iou': iou,
                      'threshold_0_5': groups['all']['threshold_0_5'], 'output': str(OUT)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
