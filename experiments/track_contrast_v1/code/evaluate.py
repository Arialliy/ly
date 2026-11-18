#!/usr/bin/env python3
"""Frozen full-frame 0.5 diagnostics for release/A/B/C at step 2048.

The archived scorer, object join and trajectory continuity are reused directly.
No common-region subset, motion transform, tuned threshold or new matcher exists.
"""
import argparse
import collections
import csv
import json
from pathlib import Path

import numpy as np

import runtime as rt

E = rt.E
Q = rt.Q
OLD = Q / 'top1_background_control_20261011'
PREVIOUS = Q / 'tlfp_pilot_20261012'
OUTPUT = E / 'evaluation'
SCORED = OUTPUT / 'scored'
CONDITIONS = ['release', 'A', 'B', 'C']
FIELDS = ['gt_count', 'matched_count', 'component_count', 'unmatched_components',
          'positive_pixels', 'positive_pixels_outside_boxes']
COHORTS = ['low_all_05', 'extreme_all_001', 'both_fixed_success', 'extra_only',
           'new_window_dependent', 'high_unmatched', 'candidate_window_mixed']


def remap(path):
    value = str(path)
    old = '/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009'
    if value == old or value.startswith(old + '/'):
        return rt.R / value[len(old):].lstrip('/')
    old = '/home/ly/DeepProPlus_TLFP_Code'
    if value == old or value.startswith(old + '/'):
        return rt.ROOT / value[len(old):].lstrip('/')
    return Path(value)


def read_csv(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(k for r in rows for k in r))
    assert fields
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def matcher_sources():
    manifest = rt.read_json(PREVIOUS / 'evaluation_code/MATCHER_SOURCES.json')
    result = {}
    for path, expected in manifest['source_sha256'].items():
        actual = remap(path)
        assert rt.sha(actual) == expected, ('Frozen matcher changed', actual)
        result[str(actual)] = expected
    return result


def inference_identity(condition, sequence):
    folder = E / 'predictions' / condition / sequence
    path = folder / 'COMPLETE.json'
    receipt = rt.read_json(path)
    checkpoint = E / 'training' / condition / 'run/snapshot_2048.pth'
    run_path = checkpoint.with_name('RUN.json')
    run = rt.read_json(run_path)
    assert receipt['status'] == 'COMPLETE' and run['status'].lower() == 'complete'
    assert receipt['condition'] == run['condition'] == condition and receipt['sequence'] == sequence
    assert receipt['checkpoint_step'] == run['final_step'] == 2048 and receipt['checkpoint_is_main']
    assert receipt['checkpoint_sha256'] == rt.sha(checkpoint)
    assert receipt['training_run_sha256'] == rt.sha(run_path)
    assert receipt['reference_checkpoint_sha256'] == run['initial_checkpoint_sha256'] == rt.sha(rt.CP)
    assert receipt['model_source_sha256'] == run['model_source_sha256'] == rt.sha(rt.MODEL_SOURCE)
    assert receipt['protocol_sha256'] == run['protocol_sha256'] == rt.sha(E / 'PROTOCOL_FREEZE.json')
    assert receipt['official_starts'] == rt.O and receipt['symmetric_starts'] == rt.S
    assert receipt['diagnostic_starts'] == rt.STARTS and receipt['normalization'] == [85.0799, 47.4845]
    assert receipt['seed'] == run['seed'] == 46
    assert receipt['trainable_parameters'] == run['trainable_parameters'] == 14593
    assert sorted(receipt['trainable_parameter_names']) == sorted(run['trainable_parameter_names'])
    assert receipt['parameters'] == 70913 and receipt['precision'] == 'FP32_no_AMP_no_TF32'
    assert receipt['model_labels_access'] is False and receipt['auxiliary_modules_at_inference'] is False
    assert receipt['original_forward_unchanged'] and receipt['all_model_state_unchanged_during_inference']
    assert receipt['frozen_frontend_unchanged'] and receipt['bn_unchanged']
    assert receipt['initial_model_state_sha256'] == receipt['final_model_state_sha256']
    assert receipt['inference_source_sha256'] == rt.sha(E / 'code/infer.py')
    assert receipt['runtime_source_sha256'] == rt.sha(E / 'code/runtime.py')
    assert receipt['annotations_sha256'] == rt.sha(rt.DATA / sequence / 'annotations.json')
    assert all(r['threshold05_disagreements'] == 0 for r in receipt['tiling_checks'])
    return {str(p): rt.sha(p) for p in [path, checkpoint, run_path, E / 'PROTOCOL_FREEZE.json']}


def score(condition, sequence):
    source = matcher_sources()
    if condition != 'release':
        source.update(inference_identity(condition, sequence))
    output = SCORED / condition / sequence
    completed = output / 'COMPLETE.json'
    if completed.exists():
        receipt = rt.read_json(completed)
        assert receipt['status'] == 'COMPLETE'
        for path, expected in receipt['wrapper_source_sha256'].items():
            assert rt.sha(remap(path)) == expected
        for name, expected in receipt['artifact_sha256'].items():
            assert rt.sha(output / name) == expected
        print({'status': 'REUSED', 'condition': condition, 'sequence': sequence}, flush=True)
        return
    scorer_path = OLD / 'evaluation_code/evaluate.py'
    frozen = rt.load_module('track_original_component_scorer', scorer_path)
    frozen.E = SCORED
    frozen.ROOTS.update({c: E / 'predictions' / c for c in ['A', 'B', 'C']})
    frozen.score(condition, sequence)
    receipt = rt.read_json(completed)
    if condition == 'release':
        reference_root = OLD / 'evaluation/release' / sequence
        with np.load(output / 'scores.npz', allow_pickle=False) as now, np.load(reference_root / 'scores.npz', allow_pickle=False) as previous:
            for key in now.files:
                assert np.array_equal(now[key], previous[key], equal_nan=True) if now[key].dtype.kind not in ['U', 'S'] else np.array_equal(now[key], previous[key]), key
        with np.load(output / 'component_areas.npz', allow_pickle=False) as now, np.load(reference_root / 'component_areas.npz', allow_pickle=False) as previous:
            for key in now.files:
                assert np.array_equal(now[key], previous[key]), key
        receipt['checks'].append('All release score/area arrays exactly equal original archived release')
        source.update({str(reference_root / name): rt.sha(reference_root / name) for name in ['scores.npz', 'component_areas.npz']})
    source[str(scorer_path)] = rt.sha(scorer_path)
    source[str(E / 'code/evaluate.py')] = rt.sha(E / 'code/evaluate.py')
    source[str(E / 'code/runtime.py')] = rt.sha(E / 'code/runtime.py')
    receipt.update(wrapper_source_sha256=source, wrapper_path=str(Path(__file__).resolve()),
        artifact_sha256={name: rt.sha(output / name) for name in ['scores.npz', 'component_areas.npz', 'FRAME_COUNTS.csv']},
        checkpoint_policy='fixed2048_main_no_checkpoint_selection' if condition != 'release' else 'published_release')
    rt.save_json(completed, receipt)


def read_record(condition, sequence):
    folder = SCORED / condition / sequence
    receipt = rt.read_json(folder / 'COMPLETE.json')
    assert receipt['status'] == 'COMPLETE' and receipt['threshold'] == .5
    assert receipt['condition'] == condition and receipt['sequence'] == sequence
    for path, expected in receipt['wrapper_source_sha256'].items():
        assert rt.sha(remap(path)) == expected, ('Evaluation identity changed', path)
    for name, expected in receipt['artifact_sha256'].items():
        assert rt.sha(folder / name) == expected
    with np.load(folder / 'scores.npz', allow_pickle=False) as archive:
        scores = {key: archive[key] for key in archive.files}
    with np.load(folder / 'component_areas.npz', allow_pickle=False) as archive:
        areas = {key: archive[key] for key in archive.files}
    assert scores['plans'].tolist() == list(rt.PLANS) and float(scores['threshold']) == .5
    assert scores['matched'].shape == (2, len(receipt['objects']))
    assert scores['frame_counts'].shape == (2, 160, len(FIELDS))
    assert np.all(scores['coverage'] > 0) and np.all(scores['coverage'].sum(axis=1) == 200)
    sources = {str(folder / name): rt.sha(folder / name) for name in ['COMPLETE.json', 'scores.npz', 'component_areas.npz', 'FRAME_COUNTS.csv']}
    return scores, areas, receipt, sources


def collect(conditions):
    source_path = PREVIOUS / 'evaluation_code/evaluate.py'
    frozen = rt.load_module('track_frozen_cohort_join', source_path)
    frozen.CONDITIONS = conditions
    frozen.NEW = [c for c in conditions if c != 'release']
    frozen.COMPARISONS = conditions
    frozen.digest = lambda p: rt.sha(remap(p))
    frozen.read_record = read_record
    frozen.record_root = lambda condition: SCORED
    data = frozen.collect(conditions)
    return frozen, data


def background(records, areas, condition, pi, plan, sequences):
    counts = sum((records[condition, s]['frame_counts'][pi].sum(0) for s in sequences), np.zeros(6, np.int64))
    result = dict(zip(FIELDS, map(int, counts)))
    aa = np.concatenate([areas[condition, s][plan + '_areas'] for s in sequences])
    uu = np.concatenate([areas[condition, s][plan + '_unmatched'].astype(bool) for s in sequences])
    for name, values in [('component', aa), ('unmatched_component', aa[uu])]:
        result[name + '_area_sum'] = int(values.sum())
        result[name + '_area_max'] = int(values.max()) if len(values) else 0
        result[name + '_area_mean'] = float(values.mean()) if len(values) else 0.
        for label, q in [('q50', .5), ('q95', .95), ('q99', .99)]:
            result[name + '_area_' + label] = float(np.quantile(values, q)) if len(values) else 0.
    return result


def all_gt(records, metadata, conditions):
    objects = []
    for sequence in rt.SEQS:
        for idx, meta in enumerate(metadata[sequence]):
            for pi, plan in enumerate(rt.PLANS):
                row = dict(sequence=sequence, track_id=meta['track_id'], frame=meta['frame'], source_frame=meta['source_frame'],
                    annotation_id=meta['annotation_id'], arrangement=plan, strict_small=int(meta['strict_small']),
                    category_id=meta['category_id'], bbox=json.dumps(meta['bbox'], separators=(',', ':')), threshold=.5)
                for condition in conditions:
                    row[condition + '_matched'] = int(records[condition, sequence]['matched'][pi, idx])
                objects.append(row)
    assert len(objects) == 32594 and sum(r['strict_small'] for r in objects) == 10400
    summaries = []
    for condition in conditions:
        for plan in rt.PLANS:
            for group, sequences in [('pooled', rt.SEQS)] + [(s, [s]) for s in rt.SEQS]:
                for scope in ['all_valid_gt', 'strict_small', 'other_valid_gt']:
                    selected = [r for r in objects if r['arrangement'] == plan and r['sequence'] in sequences
                        and (scope == 'all_valid_gt' or bool(r['strict_small']) == (scope == 'strict_small'))]
                    row = dict(condition=condition, arrangement=plan, group=group, scope=scope, target_frames=len(selected),
                        tracks=len({(r['sequence'], r['track_id']) for r in selected}),
                        M=sum(r[condition + '_matched'] for r in selected))
                    for reference in conditions:
                        row[reference + '_M'] = sum(r[reference + '_matched'] for r in selected)
                        row['G_vs_' + reference] = sum(r[condition + '_matched'] and not r[reference + '_matched'] for r in selected)
                        row['L_vs_' + reference] = sum(not r[condition + '_matched'] and r[reference + '_matched'] for r in selected)
                        assert row['M'] == row[reference + '_M'] + row['G_vs_' + reference] - row['L_vs_' + reference]
                    summaries.append(row)
    write_csv(OUTPUT / 'ALL_GT_OBJECT_COMPARISON.csv', objects)
    write_csv(OUTPUT / 'ALL_GT_SUMMARY.csv', summaries)


def summarize(conditions=CONDITIONS):
    assert conditions[0] == 'release'
    collector, data = collect(conditions)
    records, areas, metadata, sources, frames, objects = data
    summaries, concentration, protection = [], [], []
    for condition in conditions:
        for pi, plan in enumerate(rt.PLANS):
            for group, sequences in [('pooled', rt.SEQS)] + [(s, [s]) for s in rt.SEQS]:
                selected = [r for r in objects if r['sequence'] in sequences and r['arrangement'] == plan]
                costs = background(records, areas, condition, pi, plan, sequences)
                row = dict(condition=condition, arrangement=plan, group=group, threshold=.5,
                    small_gt=len(selected), small_tracks=len({(r['sequence'], r['track_id']) for r in selected}),
                    clips=len(sequences), M=sum(r[condition + '_matched'] for r in selected), **costs)
                for reference in conditions:
                    before = background(records, areas, reference, pi, plan, sequences)
                    row[reference + '_M_same'] = sum(r[reference + '_matched'] for r in selected)
                    row['G_vs_' + reference + '_same'] = sum(r[condition + '_gain_vs_' + reference] for r in selected)
                    row['L_vs_' + reference + '_same'] = sum(r[condition + '_loss_vs_' + reference] for r in selected)
                    row['net_vs_' + reference + '_same'] = row['G_vs_' + reference + '_same'] - row['L_vs_' + reference + '_same']
                    assert row['M'] == row[reference + '_M_same'] + row['net_vs_' + reference + '_same']
                    for key in ['unmatched_components', 'positive_pixels_outside_boxes', 'unmatched_component_area_sum']:
                        row[reference + '_' + key] = before[key]
                        row['delta_' + key + '_vs_' + reference] = costs[key] - before[key]
                row['U_budget'] = 4015 if group == 'pooled' else ''
                row['passes_pooled_U_budget'] = int(costs['unmatched_components'] <= 4015) if group == 'pooled' else ''
                for cohort in COHORTS:
                    subset = [r for r in selected if r[cohort]]
                    matched = [r for r in subset if r[condition + '_matched']]
                    row[cohort + '_gt'] = len(subset)
                    row[cohort + '_matched'] = len(matched)
                    row[cohort + '_matched_tracks'] = len({(r['sequence'], r['track_id']) for r in matched})
                    row[cohort + '_matched_clips'] = len({r['sequence'] for r in matched})
                summaries.append(row)
                if group == 'pooled':
                    for reference in conditions:
                        for outcome in ['gain', 'loss']:
                            chosen = [r for r in selected if r[condition + '_' + outcome + '_vs_' + reference]]
                            counts = collections.Counter((r['sequence'], r['track_id']) for r in chosen)
                            ranked = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
                            concentration.append(dict(condition=condition, reference=reference, arrangement=plan, outcome=outcome,
                                frames=len(chosen), tracks=len(counts), clips=len({k[0] for k in counts}),
                                top1_frames=ranked[0][1] if ranked else 0, top5_frames=sum(v for _, v in ranked[:5]),
                                ranked_tracks=json.dumps([dict(sequence=k[0], track_id=k[1], frames=v) for k, v in ranked])))
        left = {(r['sequence'], r['frame'], r['annotation_id']): r for r in objects if r['arrangement'] == 'O'}
        right = {(r['sequence'], r['frame'], r['annotation_id']): r for r in objects if r['arrangement'] == 'S'}
        for cohort in ['all'] + COHORTS:
            selected = [k for k, r in left.items() if cohort == 'all' or r[cohort]]
            o = np.array([left[k][condition + '_matched'] for k in selected], bool)
            s = np.array([right[k][condition + '_matched'] for k in selected], bool)
            protection.append(dict(condition=condition, cohort=cohort, target_frames=len(selected),
                O_matched=int(o.sum()), S_matched=int(s.sum()), both_matched=int((o & s).sum()),
                O_only=int((o & ~s).sum()), S_only=int((~o & s).sum()), neither_matched=int((~o & ~s).sum()),
                lost_dual_success=int((~(o & s)).sum()) if cohort == 'both_fixed_success' else '',
                lost_both_arrangements=int((~o & ~s).sum()) if cohort == 'both_fixed_success' else ''))
    assert [(r['M'], r['unmatched_components']) for r in summaries if r['condition'] == 'release' and r['group'] == 'pooled'] == [(700, 4015), (720, 3967)]
    write_csv(OUTPUT / 'SUMMARY.csv', [r for r in summaries if r['group'] == 'pooled'])
    write_csv(OUTPUT / 'SEQUENCE_SUMMARY.csv', [r for r in summaries if r['group'] != 'pooled'])
    write_csv(OUTPUT / 'OBJECT_COMPARISON.csv', objects)
    write_csv(OUTPUT / 'FRAME_COUNTS.csv', frames)
    write_csv(OUTPUT / 'RECOVERY_CONCENTRATION.csv', concentration)
    write_csv(OUTPUT / 'PROTECTED_AND_WINDOW_RETENTION.csv', protection)
    write_csv(OUTPUT / 'COHORT_TRANSITIONS.csv', collector.cohort_transitions(conditions, objects))
    trajectory_path = PREVIOUS / 'evaluation_code/trajectory_tables.py'
    trajectory = rt.load_module('track_frozen_miss_runs', trajectory_path)
    trajectory.trajectory_tables(objects, conditions, conditions, OUTPUT, Q)
    all_gt(records, metadata, conditions)
    for path in [E / 'code/evaluate.py', E / 'code/runtime.py', trajectory_path,
                 PREVIOUS / 'evaluation_code/evaluate.py', Q / 'low_response_information_20261011/COHORTS.csv']:
        sources[str(path)] = rt.sha(path)
    rt.save_json(OUTPUT / 'SUMMARY_COMPLETE.json', dict(status='COMPLETE', conditions=conditions,
        threshold=.5, plans=rt.PLANS, fixed_main_step=2048, strict_small_target_frames=5200,
        strict_small_tracks=130, all_valid_target_frames=16297, other_valid_target_frames=11097,
        cohorts={'low_all_05': 4342, 'extreme_all_001': 4170, 'both_fixed_success': 646},
        requested_paired_comparisons=['C_vs_release', 'C_vs_A', 'C_vs_B'],
        source_sha256=sources, output_sha256={p.name: rt.sha(p) for p in sorted(OUTPUT.glob('*.csv'))},
        interpretation='Frozen full-frame actual-component diagnostic; not official Pd/Fa. Same 2048-step endpoint, O/S, threshold and fixed cohorts. Other-valid-GT records are reported separately.'))
    print({'status': 'COMPLETE', 'conditions': conditions,
        'rows': [{k: r[k] for k in ['condition', 'arrangement', 'M', 'G_vs_release_same', 'L_vs_release_same', 'unmatched_components', 'low_all_05_matched']} for r in summaries if r['group'] == 'pooled']}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition', choices=CONDITIONS)
    parser.add_argument('--sequence', choices=rt.SEQS)
    parser.add_argument('--summary', action='store_true')
    parser.add_argument('--conditions', choices=CONDITIONS, nargs='+', default=CONDITIONS)
    args = parser.parse_args()
    if args.summary:
        summarize(args.conditions)
    else:
        if args.condition is None:
            parser.error('--condition required for scoring')
        for sequence in [args.sequence] if args.sequence else rt.SEQS:
            score(args.condition, sequence)
