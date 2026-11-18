#!/usr/bin/env python3
"""Frozen full-frame 0.5 diagnostics for release/C_init/CONT/FRAME/GAP at global step 2304.

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
CONDITIONS = ['release', 'C_init', 'CONT', 'FRAME', 'GAP']
TRAINED = ['CONT', 'FRAME', 'GAP']
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


def initial_identity(sequence):
    """Bind the reused C predictions to the exact resumed checkpoint."""
    path = rt.V1 / 'predictions/C' / sequence / 'COMPLETE.json'
    receipt = rt.read_json(path)
    run_path = rt.INIT_CP.with_name('RUN.json')
    run = rt.read_json(run_path)
    assert receipt['status'] == 'COMPLETE' and run['status'] == 'complete'
    assert receipt['condition'] == run['condition'] == 'C'
    assert receipt['checkpoint_step'] == run['final_step'] == 2048
    assert receipt['checkpoint_sha256'] == rt.sha(rt.INIT_CP)
    assert receipt['training_run_sha256'] == rt.sha(run_path)
    assert receipt['model_source_sha256'] == rt.sha(rt.MODEL_SOURCE)
    assert receipt['reference_checkpoint_sha256'] == run['initial_checkpoint_sha256'] == rt.sha(rt.CP)
    assert receipt['protocol_sha256'] == run['protocol_sha256'] == rt.sha(rt.V1 / 'PROTOCOL_FREEZE.json')
    assert receipt['raw_sha256'] == rt.sha(rt.DATA / sequence / 'raw_frames.npy')
    assert receipt['annotations_sha256'] == rt.sha(rt.DATA / sequence / 'annotations.json')
    assert receipt['official_starts'] == rt.O and receipt['symmetric_starts'] == rt.S
    assert receipt['original_forward_unchanged'] and receipt['bn_unchanged']
    return {str(p): rt.sha(p) for p in [path, run_path, rt.INIT_CP, rt.V1 / 'PROTOCOL_FREEZE.json']}


def inference_identity(condition, sequence):
    folder = E / 'predictions' / condition / sequence
    path = folder / 'COMPLETE.json'
    receipt = rt.read_json(path)
    checkpoint = E / 'training' / condition / 'run/snapshot_2304.pth'
    run_path = checkpoint.with_name('RUN.json')
    run = rt.read_json(run_path)
    assert receipt['status'] == 'COMPLETE' and run['status'].lower() == 'complete'
    assert receipt['condition'] == run['condition'] == condition and receipt['sequence'] == sequence
    assert receipt['checkpoint_step'] == run['final_step'] == 2304 and receipt['checkpoint_is_main']
    assert receipt['checkpoint_sha256'] == rt.sha(checkpoint)
    assert receipt['training_run_sha256'] == rt.sha(run_path)
    assert receipt['reference_checkpoint_sha256'] == run['release_checkpoint_sha256'] == rt.sha(rt.CP)
    assert receipt['initial_checkpoint_sha256'] == run['initial_checkpoint_sha256'] == rt.sha(rt.INIT_CP)
    assert run['start_step'] == receipt['initial_step'] == 2048
    assert run['additional_steps'] == receipt['additional_steps'] == 256
    assert run['initial_optimizer_restored'] and receipt['initial_optimizer_restored']
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
    if condition in TRAINED:
        source.update(inference_identity(condition, sequence))
    elif condition == 'C_init':
        source.update(initial_identity(sequence))
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
    frozen.ROOTS.update({c: E / 'predictions' / c for c in TRAINED})
    frozen.ROOTS['C_init'] = rt.V1 / 'predictions/C'
    frozen.score(condition, sequence)
    receipt = rt.read_json(completed)
    if condition in ['release', 'C_init']:
        reference_root = (OLD / 'evaluation/release' / sequence if condition == 'release'
                          else rt.V1 / 'evaluation/scored/C' / sequence)
        with np.load(output / 'scores.npz', allow_pickle=False) as now, np.load(reference_root / 'scores.npz', allow_pickle=False) as previous:
            for key in now.files:
                assert np.array_equal(now[key], previous[key], equal_nan=True) if now[key].dtype.kind not in ['U', 'S'] else np.array_equal(now[key], previous[key]), key
        with np.load(output / 'component_areas.npz', allow_pickle=False) as now, np.load(reference_root / 'component_areas.npz', allow_pickle=False) as previous:
            for key in now.files:
                assert np.array_equal(now[key], previous[key]), key
        receipt['checks'].append(condition + ' score/area arrays exactly equal frozen baseline')
        source.update({str(reference_root / name): rt.sha(reference_root / name) for name in ['scores.npz', 'component_areas.npz']})
    source[str(scorer_path)] = rt.sha(scorer_path)
    source[str(E / 'code/evaluate.py')] = rt.sha(E / 'code/evaluate.py')
    source[str(E / 'code/runtime.py')] = rt.sha(E / 'code/runtime.py')
    receipt.update(wrapper_source_sha256=source, wrapper_path=str(Path(__file__).resolve()),
        artifact_sha256={name: rt.sha(output / name) for name in ['scores.npz', 'component_areas.npz', 'FRAME_COUNTS.csv']},
        checkpoint_policy='fixed2304_main_no_checkpoint_selection' if condition in TRAINED else 'frozen_baseline')
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


def contiguous(left, right):
    return int(right['frame']) == int(left['frame']) + 1 and int(right['source_frame']) == int(left['source_frame']) + 1


def runs(rows, key, value):
    result, active, previous = [], [], None
    for row in rows:
        if previous is not None and not contiguous(previous, row):
            if active:
                result.append(active)
            active = []
        if bool(row[key]) == value:
            active.append(row)
        else:
            if active:
                result.append(active)
            active = []
        previous = row
    if active:
        result.append(active)
    return result


def continuity_and_coverage(objects, conditions):
    """Full fixed tracks; isolated hits require an observed contiguous 0,1,0."""
    groups = collections.defaultdict(list)
    for row in objects:
        groups[row['arrangement'], row['sequence'], row['track_id']].append(row)
    tracks, centers, segments = [], [], []
    for (plan, sequence, tid), raw in sorted(groups.items()):
        raw = sorted(raw, key=lambda r: (int(r['frame']), int(r['source_frame'])))
        eligible = {i for i in range(1, len(raw) - 1)
                    if contiguous(raw[i - 1], raw[i]) and contiguous(raw[i], raw[i + 1])}
        matches = {c: {i for i, r in enumerate(raw) if r[c + '_matched']} for c in conditions}
        isolated = {c: {i for i in eligible if i in matches[c] and i - 1 not in matches[c]
                       and i + 1 not in matches[c]} for c in conditions}
        for i in sorted(eligible):
            r = raw[i]
            item = {k: r[k] for k in ['arrangement', 'sequence', 'track_id', 'frame', 'source_frame', 'annotation_id',
                                      'low_all_05', 'extreme_all_001', 'both_fixed_success']}
            for c in conditions:
                item[c + '_matched'] = int(i in matches[c])
                item[c + '_isolated010'] = int(i in isolated[c])
            centers.append(item)
        for c in conditions:
            success = runs(raw, c + '_matched', True)
            miss = runs(raw, c + '_matched', False)
            item = dict(condition=c, arrangement=plan, sequence=sequence, track_id=tid,
                target_frames=len(raw), matched_frames=len(matches[c]), coverage_fraction=len(matches[c]) / len(raw),
                low_target_frames=sum(r['low_all_05'] for r in raw),
                low_matched_frames=sum(r['low_all_05'] and r[c + '_matched'] for r in raw),
                extreme_matched_frames=sum(r['extreme_all_001'] and r[c + '_matched'] for r in raw),
                eligible_center_frames=len(eligible), eligible_center_matches=len(matches[c] & eligible),
                isolated010_hits=len(isolated[c]), boundary_or_gap_matched_frames=len(matches[c] - eligible),
                matched_run_count=len(success), matched_runs_ge2=sum(len(s) >= 2 for s in success),
                longest_matched_run=max(map(len, success), default=0),
                longest_miss_run=max(map(len, miss), default=0))
            for ref in conditions:
                item['gain_vs_' + ref] = len(matches[c] - matches[ref])
                item['loss_vs_' + ref] = len(matches[ref] - matches[c])
                item['net_vs_' + ref] = len(matches[c]) - len(matches[ref])
                item['isolated010_gain_vs_' + ref] = len(isolated[c] & (matches[c] - matches[ref]))
                item['isolated010_delta_vs_' + ref] = len(isolated[c]) - len(isolated[ref])
            tracks.append(item)
        if 'C_init' in conditions:
            for index, segment in enumerate(runs(raw, 'C_init_matched', False)):
                item = dict(arrangement=plan, sequence=sequence, track_id=tid, C_init_miss_segment=index,
                    frame_start=segment[0]['frame'], frame_end=segment[-1]['frame'],
                    source_frame_start=segment[0]['source_frame'], source_frame_end=segment[-1]['source_frame'],
                    target_frames=len(segment), low_all_05_frames=sum(r['low_all_05'] for r in segment))
                for c in conditions:
                    remaining = runs(segment, c + '_matched', False)
                    item[c + '_recovered_frames'] = sum(r[c + '_matched'] for r in segment)
                    item[c + '_remaining_miss_longest'] = max(map(len, remaining), default=0)
                    item[c + '_remaining_miss_runs'] = len(remaining)
                segments.append(item)
    summary = []
    for c in conditions:
        for plan in rt.PLANS:
            for group in ['pooled'] + rt.SEQS:
                chosen = [r for r in tracks if r['condition'] == c and r['arrangement'] == plan
                          and (group == 'pooled' or r['sequence'] == group)]
                item = dict(condition=c, arrangement=plan, group=group, tracks=len(chosen),
                    tracks_with_any_match=sum(r['matched_frames'] > 0 for r in chosen),
                    tracks_with_multi_frame_run=sum(r['matched_runs_ge2'] > 0 for r in chosen),
                    tracks_with_isolated010=sum(r['isolated010_hits'] > 0 for r in chosen),
                    coverage_track_mean=float(np.mean([r['coverage_fraction'] for r in chosen])),
                    longest_matched_run_track_mean=float(np.mean([r['longest_matched_run'] for r in chosen])))
                for key in ['target_frames', 'matched_frames', 'low_target_frames', 'low_matched_frames',
                            'extreme_matched_frames', 'eligible_center_frames', 'eligible_center_matches',
                            'isolated010_hits', 'boundary_or_gap_matched_frames', 'matched_run_count', 'matched_runs_ge2']:
                    item[key] = sum(r[key] for r in chosen)
                for ref in conditions:
                    for prefix in ['gain_vs_', 'loss_vs_', 'net_vs_', 'isolated010_gain_vs_', 'isolated010_delta_vs_']:
                        item[prefix + ref] = sum(r[prefix + ref] for r in chosen)
                    item['tracks_coverage_up_vs_' + ref] = sum(r['net_vs_' + ref] > 0 for r in chosen)
                    item['tracks_coverage_down_vs_' + ref] = sum(r['net_vs_' + ref] < 0 for r in chosen)
                summary.append(item)
    write_csv(OUTPUT / 'TRACK_COVERAGE.csv', tracks)
    write_csv(OUTPUT / 'COVERAGE_AND_ISOLATION_SUMMARY.csv', summary)
    write_csv(OUTPUT / 'ISOLATED_HIT_OBJECTS.csv', centers)
    if segments:
        write_csv(OUTPUT / 'C_INIT_MISS_SEGMENTS.csv', segments)
    rt.save_json(OUTPUT / 'ISOLATION_SCHEMA.json', dict(status='COMPLETE',
        scope='All fixed 5200 strict target frames and 130 tracks per arrangement; no filtering by new recoveries.',
        isolation='A hit is isolated only for pattern 0,1,0 on the same track with both local and source frame numbers consecutive. '
                  'Both observed neighbors must exist; boundaries and missing-label/source-frame gaps are ineligible.',
        coverage='Matched observed strict frames divided by observed strict frames, separately per track; not physical unobserved trajectory coverage.',
        sums='O/S are separate arrangements; do not add their recovered counts without deduplicating target frame IDs.',
        artifact_sha256={p.name: rt.sha(p) for p in [OUTPUT / 'TRACK_COVERAGE.csv', OUTPUT / 'COVERAGE_AND_ISOLATION_SUMMARY.csv',
            OUTPUT / 'ISOLATED_HIT_OBJECTS.csv']}))


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
    continuity_and_coverage(objects, conditions)
    for path in [E / 'code/evaluate.py', E / 'code/runtime.py', trajectory_path,
                 PREVIOUS / 'evaluation_code/evaluate.py', Q / 'low_response_information_20261011/COHORTS.csv']:
        sources[str(path)] = rt.sha(path)
    rt.save_json(OUTPUT / 'SUMMARY_COMPLETE.json', dict(status='COMPLETE', conditions=conditions,
        threshold=.5, plans=rt.PLANS, fixed_main_step=2304, initial_step=2048, additional_steps=256, strict_small_target_frames=5200,
        strict_small_tracks=130, all_valid_target_frames=16297, other_valid_target_frames=11097,
        cohorts={'low_all_05': 4342, 'extreme_all_001': 4170, 'both_fixed_success': 646},
        requested_paired_comparisons=['GAP_vs_C_init', 'GAP_vs_CONT', 'GAP_vs_FRAME', 'FRAME_vs_CONT'],
        source_sha256=sources, output_sha256={p.name: rt.sha(p) for p in sorted(OUTPUT.glob('*.csv'))},
        interpretation='Frozen full-frame actual-component diagnostic; not official Pd/Fa. Same global2304 endpoint, resumed C2048 and Adam for 256 matched updates, O/S, threshold and fixed cohorts. Other-valid-GT records are reported separately.'))
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
