#!/usr/bin/env python3
"""Frozen 0.5 actual-component scoring for the camera-motion controls.

The full-frame deployment path is supplied by inference, including geometric
fallback. The common-mask diagnostic keeps the original GT ordering/matcher,
but counts strict-small outcomes only for boxes wholly inside the shared mask.
No model, threshold fitting, held-out labels, or classifier is loaded here.
"""
import argparse
import collections
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
from scipy import ndimage

BASE = Path(__file__).resolve().parents[1]
PROJECT = BASE.parents[1]
Q = PROJECT / 'archive/infrared_problem_screen_20261009/temporal_diagnosis_20261010'
P = Q / 'stage1_goal'
REFERENCE = Q / 'top1_background_control_20261011/evaluation/release'
PLANS = {'O': [0, 36, 72, 108, 120], 'S': [0, 24, 60, 96, 120]}
STARTS = sorted(set(sum(PLANS.values(), [])))
SEQUENCES = ['DJI_0026_2', 'DJI_0067_2']
CONDITIONS = ['C0', 'C1', 'C2']
FIELDS = ['gt_count', 'matched_count', 'component_count', 'unmatched_components',
          'positive_pixels', 'positive_pixels_outside_boxes']
COHORTS = ['low_all_05', 'extreme_all_001', 'both_fixed_success', 'extra_only',
           'new_window_dependent', 'high_unmatched']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def read_csv(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    assert fields, path
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def remap(path):
    value = str(path)
    mappings = {
        '/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009': str(Q.parent),
        '/home/ly/DeepProPlus_TLFP_Code': str(PROJECT),
    }
    for old, new in mappings.items():
        if value == old or value.startswith(old + '/'):
            return Path(new + value[len(old):])
    return Path(value)


def frozen_sources():
    paths = [P / 'pilot/score_pilot.py', Q / 'analysis/analyze.py',
             Q / 'top1_background_control_20261011/evaluation_code/evaluate.py']
    freeze = read_json(Q / 'tlfp_pilot_20261012/evaluation_code/MATCHER_SOURCES.json')
    checked = {}
    for old, expected in freeze['source_sha256'].items():
        path = remap(old)
        assert digest(path) == expected, ('Frozen matcher source changed', path)
        checked[str(path)] = expected
    for path in paths:
        checked[str(path)] = digest(path)
    scorer = module(paths[0], 'motion_original_gt_scorer')
    helper, helper_path = scorer.helper()
    assert helper_path == paths[1]
    return scorer, helper, checked


def cohorts():
    path = Q / 'low_response_information_20261011/COHORTS.csv'
    rows = read_csv(path)
    result = {(r['sequence'], int(r['frame']), int(r['annotation_id'])): r for r in rows}
    assert len(result) == len(rows) == 5200
    for field, expected in [('low_all_05', 4342), ('extreme_all_001', 4170), ('both_fixed_success', 646)]:
        assert sum(int(r[field]) for r in rows) == expected
    assert len({(r['sequence'], int(r['track_id'])) for r in rows}) == 130
    return result


def predictions_root(condition, sequence):
    return (P / 'predictions/official' if condition == 'C0' else BASE / 'predictions' / condition) / sequence


def load_windows(condition, sequence):
    root = predictions_root(condition, sequence)
    receipt = read_json(root / 'COMPLETE.json')
    assert receipt['status'] == 'COMPLETE'
    if 'raw_sha256' in receipt:
        assert receipt['raw_sha256'] == digest(P / 'data' / sequence / 'raw_frames.npy')
    if condition != 'C0':
        baseline = read_json(predictions_root('C0', sequence) / 'COMPLETE.json')
        assert receipt['condition'] == condition and receipt['sequence'] == sequence
        assert receipt['checkpoint_sha256'] == baseline['checkpoint_sha256']
        assert receipt['model_source_sha256'] == baseline['model_source_sha256']
        assert receipt['training'] is False and receipt['model_labels_access'] is False
        assert receipt['all_model_state_unchanged'] is True
        assert receipt['O'] == PLANS['O'] and receipt['S'] == PLANS['S']
        assert receipt['starts_union'] == STARTS and receipt['anchor_offset'] == 19
        assert receipt['threshold'] == .5 and receipt['normalization'] == [85.0799, 47.4845]
        assert receipt['precision'] == 'FP32_no_AMP_no_TF32'
        assert receipt['protocol_freeze_sha256'] == digest(BASE / 'PROTOCOL_FREEZE.json')
    source = {str(root / 'COMPLETE.json'): digest(root / 'COMPLETE.json')}
    windows = {}
    for start in STARTS:
        path = root / 'windows' / f'window_{start:03d}.npz'
        checksum = digest(path)
        assert receipt['window_sha256'][path.name] == checksum, path
        with np.load(path, allow_pickle=False) as archive:
            probability = archive['probabilities']
            assert int(archive['start']) == start and int(archive['end']) == start + 40
        assert probability.shape == (40, 512, 640) and probability.dtype == np.float32
        assert np.isfinite(probability).all() and np.all((probability >= 0) & (probability <= 1))
        windows[start] = probability
        source[str(path)] = checksum
    return windows, source, receipt


def geometry_masks(sequence, required=True):
    masks, source, usable = {}, {}, {}
    for start in STARTS:
        path = BASE / 'geometry' / sequence / 'windows' / f'window_{start:03d}.npz'
        if not path.exists() and not required:
            return None, {}, {}
        with np.load(path, allow_pickle=False) as archive:
            masks[start] = archive['valid'].astype(bool)
            usable[start] = bool(archive['usable_window'])
        assert masks[start].shape == (40, 512, 640)
        if not usable[start]:
            assert not masks[start].any(), ('Untrusted window cannot supply compensated support', path)
        source[str(path)] = digest(path)
    return masks, source, usable


def fuse(windows, starts):
    result = np.zeros((160, 512, 640), np.float32)
    coverage = np.zeros(160, np.int16)
    for start in starts:
        result[start:start + 40] = np.maximum(result[start:start + 40], windows[start])
        coverage[start:start + 40] += 1
    assert np.all(coverage > 0) and coverage.sum() == 200
    return result, coverage


def common_mask(masks, starts):
    result = np.ones((160, 512, 640), bool)
    for start in starts:
        result[start:start + 40] &= masks[start]
    return result


def complete_box(mask, ann):
    x0, y0, x1, y1 = ann['crop']
    return bool(not ann['partly_outside'] and x1 > x0 and y1 > y0 and mask[y0:y1, x0:x1].all())


def score(condition, sequence, scope, verify_c0=False):
    output = BASE / 'evaluation' / scope / condition / sequence
    receipt_path = output / 'COMPLETE.json'
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        assert receipt['status'] == 'COMPLETE'
        for path, expected in receipt['source_sha256'].items():
            assert digest(remap(path)) == expected, ('Completed score source changed', path)
        for name, expected in receipt['artifact_sha256'].items():
            assert digest(output / name) == expected
        print(json.dumps({'status': 'REUSED', 'condition': condition, 'sequence': sequence, 'scope': scope}), flush=True)
        return
    began = time.perf_counter()
    scorer, helper, source = frozen_sources()
    annotation_path = P / 'data' / sequence / 'annotations.json'
    gt, metadata, offsets, excluded = scorer.gt_data(annotation_path, helper)
    source[str(annotation_path)] = digest(annotation_path)
    windows, prediction_source, inference = load_windows(condition, sequence)
    source.update(prediction_source)
    masks, mask_source, usable = geometry_masks(sequence, required=scope == 'common')
    source.update(mask_source)
    if condition != 'C0':
        assert masks is not None
        baseline_root = predictions_root('C0', sequence)
        baseline_receipt = read_json(baseline_root / 'COMPLETE.json')
        for start in STARTS:
            path = baseline_root / 'windows' / f'window_{start:03d}.npz'
            assert digest(path) == baseline_receipt['window_sha256'][path.name]
            with np.load(path, allow_pickle=False) as archive:
                original = archive['probabilities']
            assert np.array_equal(windows[start][~masks[start]], original[~masks[start]]), ('Invalid pixels must use same-window C0', condition, sequence, start)
            source[str(path)] = digest(path)
        del original
    n = len(metadata)
    arrays = dict(matched=np.zeros((2, n), np.uint8), box_max=np.zeros((2, n), np.float32),
                  peak_x=np.full((2, n), np.nan, np.float32), peak_y=np.full((2, n), np.nan, np.float32),
                  component_area=np.zeros((2, n), np.int32), frame_counts=np.zeros((2, 160, 6), np.int64),
                  coverage=np.zeros((2, 160), np.int16), strict_small=np.array([m['strict_small'] for m in metadata], bool),
                  eligible=np.ones((2, n), bool), offsets=offsets)
    area_arrays, frame_rows = {}, []
    for pi, (plan, starts) in enumerate(PLANS.items()):
        fused, arrays['coverage'][pi] = fuse(windows, starts)
        if plan == 'O':
            fused_path = predictions_root(condition, sequence) / 'official_fused.npz'
            assert digest(fused_path) == inference['fused_sha256']
            with np.load(fused_path, allow_pickle=False) as archive:
                assert np.array_equal(fused, archive['probabilities']), 'O fusion must equal original five-window maximum'
            source[str(fused_path)] = digest(fused_path)
        mask = common_mask(masks, starts) if scope == 'common' else None
        component_areas, component_unmatched, area_offsets = [], [], [0]
        for frame in range(160):
            probability = fused[frame]
            if mask is not None:
                probability = np.where(mask[frame], probability, 0).astype(np.float32)
                for idx, ann in enumerate(gt[frame], int(offsets[frame])):
                    arrays['eligible'][pi, idx] = complete_box(mask[frame], ann)
            rows, stats, components = helper.match_frame(probability, gt[frame], .5)
            for idx, row in enumerate(rows, int(offsets[frame])):
                assert (row['annotation_id'], row['track_id']) == (metadata[idx]['annotation_id'], metadata[idx]['track_id'])
                arrays['matched'][pi, idx] = row['matched']
                arrays['box_max'][pi, idx] = row['box_max'] if row['box_max'] is not None else np.nan
                if row['matched']:
                    for key in ['peak_x', 'peak_y', 'component_area']:
                        arrays[key][pi, idx] = row[key]
            arrays['frame_counts'][pi, frame] = [stats[key] for key in FIELDS]
            aa = [int(c['area']) for c in components]
            uu = [int(c['unmatched_component']) for c in components]
            component_areas.extend(aa)
            component_unmatched.extend(uu)
            area_offsets.append(len(component_areas))
            assert sum(aa) == stats['positive_pixels'] and sum(uu) == stats['unmatched_components']
            j0, j1 = int(offsets[frame]), int(offsets[frame + 1])
            eligible = arrays['eligible'][pi, j0:j1]
            boundary_count = 0
            if mask is not None and components:
                labels, _ = ndimage.label(probability >= .5, structure=np.ones((3, 3), np.uint8))
                inside_boundary = mask[frame] & ndimage.binary_dilation(~mask[frame], structure=np.ones((3, 3), bool))
                boundary_count = int(np.count_nonzero(np.unique(labels[inside_boundary])))
            frame_rows.append(dict(condition=condition, sequence=sequence, arrangement=plan, frame=frame,
                source_frame=metadata[j0]['source_frame'] if j0 < j1 else '', scope=scope, threshold=.5,
                coverage_windows=int(arrays['coverage'][pi, frame]),
                common_valid_pixels=int(mask[frame].sum()) if mask is not None else 512 * 640,
                common_valid_fraction=float(mask[frame].mean()) if mask is not None else 1.,
                eligible_gt_count=int(eligible.sum()),
                eligible_matched_count=int((arrays['matched'][pi, j0:j1].astype(bool) & eligible).sum()),
                **{key: int(stats[key]) for key in FIELDS},
                unmatched_component_area=sum(a for a, u in zip(aa, uu) if u),
                unmatched_component_max_area=max((a for a, u in zip(aa, uu) if u), default=0),
                component_max_area=max(aa, default=0), mask_boundary_components=boundary_count))
        area_arrays[plan + '_areas'] = np.array(component_areas, np.int32)
        area_arrays[plan + '_unmatched'] = np.array(component_unmatched, np.uint8)
        area_arrays[plan + '_offsets'] = np.array(area_offsets, np.int64)
        del fused, mask
    checks = []
    if condition == 'C0' and scope == 'full':
        reference_path = REFERENCE / sequence / 'scores.npz'
        reference_receipt = read_json(REFERENCE / sequence / 'COMPLETE.json')
        assert reference_receipt['objects'] == metadata
        with np.load(reference_path, allow_pickle=False) as archive:
            for key in ['matched', 'box_max', 'peak_x', 'peak_y', 'component_area', 'frame_counts', 'coverage', 'strict_small', 'offsets']:
                assert np.array_equal(arrays[key], archive[key], equal_nan=True), (sequence, key)
        with np.load(REFERENCE / sequence / 'component_areas.npz', allow_pickle=False) as archive:
            for key in area_arrays:
                assert np.array_equal(area_arrays[key], archive[key]), (sequence, key)
        source[str(reference_path)] = digest(reference_path)
        checks.append('C0 O/S matches, box scores, peaks, areas, counts and coverage exactly equal frozen release')
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / 'scores.npz', **arrays, plans=np.array(list(PLANS)), threshold=np.array(.5))
    np.savez_compressed(output / 'component_areas.npz', **area_arrays)
    write_csv(output / 'FRAME_COUNTS.csv', frame_rows)
    artifacts = {name: digest(output / name) for name in ['scores.npz', 'component_areas.npz', 'FRAME_COUNTS.csv']}
    write_json(receipt_path, dict(status='COMPLETE', condition=condition, sequence=sequence, scope=scope,
        threshold=.5, objects=metadata, excluded_annotations=excluded, checks=checks,
        source_sha256=source, artifact_sha256=artifacts, scorer_sha256=digest(__file__),
        usable_windows=usable, elapsed_seconds=time.perf_counter() - began,
        common_scope_rule='Original GT order and actual matcher on identically masked probabilities; strict-small outcome denominator requires entire original bbox crop valid. Partial GT retains original background shielding and matching eligibility. Components may split at mask boundary.',
        interpretation='Diagnostic actual-component matching and U; not official Pd/Fa.'))
    print(json.dumps({'status': 'COMPLETE', 'condition': condition, 'sequence': sequence, 'scope': scope,
                      'seconds': time.perf_counter() - began}), flush=True)


def read_record(condition, sequence, scope):
    folder = BASE / 'evaluation' / scope / condition / sequence
    receipt = read_json(folder / 'COMPLETE.json')
    assert receipt['status'] == 'COMPLETE' and receipt['threshold'] == .5
    for name, expected in receipt['artifact_sha256'].items():
        assert digest(folder / name) == expected
    with np.load(folder / 'scores.npz', allow_pickle=False) as archive:
        scores = {key: archive[key] for key in archive.files}
    with np.load(folder / 'component_areas.npz', allow_pickle=False) as archive:
        areas = {key: archive[key] for key in archive.files}
    return scores, areas, receipt, read_csv(folder / 'FRAME_COUNTS.csv')


def cost(records, areas, condition, pi, plan, sequences):
    counts = sum((records[condition, s]['frame_counts'][pi].sum(0) for s in sequences), np.zeros(6, np.int64))
    aa = np.concatenate([areas[condition, s][plan + '_areas'] for s in sequences])
    uu = np.concatenate([areas[condition, s][plan + '_unmatched'].astype(bool) for s in sequences])
    result = dict(zip(FIELDS, map(int, counts)))
    for prefix, selected in [('component', aa), ('unmatched_component', aa[uu])]:
        result[prefix + '_area_sum'] = int(selected.sum())
        result[prefix + '_area_max'] = int(selected.max()) if len(selected) else 0
        result[prefix + '_area_mean'] = float(selected.mean()) if len(selected) else 0.
        for label, quantile in [('q50', .5), ('q95', .95), ('q99', .99)]:
            result[prefix + '_area_' + label] = float(np.quantile(selected, quantile)) if len(selected) else 0.
    return result


def trajectory_tables(objects, conditions, output, scope):
    # Reuse the frozen continuity primitive; holes in common support also break runs.
    path = Q / 'tlfp_pilot_20261012/evaluation_code/trajectory_tables.py'
    frozen = module(path, 'motion_frozen_trajectory_continuity')
    groups = collections.defaultdict(list)
    for row in objects:
        if row['eligible']:
            groups[row['arrangement'], row['sequence'], row['track_id']].append(row)
    tracks, release_segments = [], []
    cohort_names = ['all', 'low_all_05', 'extreme_all_001', 'both_fixed_success']
    for (plan, sequence, track_id), raw in sorted(groups.items()):
        raw.sort(key=lambda r: (r['frame'], r['source_frame']))
        for cohort in cohort_names:
            selected = [r for r in raw if cohort == 'all' or r[cohort]]
            if not selected:
                continue
            observed = frozen.segments(selected, lambda r: True)
            misses = {c: frozen.segments(selected, lambda r, c=c: not r[c + '_matched']) for c in conditions}
            for condition in conditions:
                lengths = [len(s) for s in misses[condition]]
                original = [len(s) for s in misses['C0']]
                tracks.append(dict(scope=scope, condition=condition, arrangement=plan, sequence=sequence, track_id=track_id,
                    cohort=cohort, target_frames=len(selected), observation_segments=len(observed),
                    observation_longest=max(map(len, observed)), missed_frames=sum(lengths), miss_run_count=len(lengths),
                    longest_miss_run=max(lengths, default=0), miss_run_lengths=json.dumps(lengths),
                    longest_miss_delta_vs_C0=max(lengths, default=0) - max(original, default=0),
                    missed_frames_delta_vs_C0=sum(lengths) - sum(original)))
        for index, segment in enumerate(frozen.segments(raw, lambda r: not r['C0_matched'])):
            row = dict(scope=scope, arrangement=plan, sequence=sequence, track_id=track_id, C0_miss_segment=index,
                frame_start=segment[0]['frame'], frame_end=segment[-1]['frame'],
                source_frame_start=segment[0]['source_frame'], source_frame_end=segment[-1]['source_frame'],
                target_frames=len(segment), low_all_05_frames=sum(r['low_all_05'] for r in segment))
            for condition in conditions:
                remaining = frozen.segments(segment, lambda r, c=condition: not r[c + '_matched'])
                row[condition + '_recovered_frames'] = sum(r[condition + '_matched'] for r in segment)
                row[condition + '_remaining_miss_longest'] = max(map(len, remaining), default=0)
                row[condition + '_remaining_miss_runs'] = len(remaining)
            release_segments.append(row)
    summaries = []
    for condition in conditions:
        for plan in PLANS:
            for group in ['pooled'] + SEQUENCES:
                for cohort in cohort_names:
                    selected = [r for r in tracks if r['condition'] == condition and r['arrangement'] == plan
                                and r['cohort'] == cohort and (group == 'pooled' or r['sequence'] == group)]
                    if not selected:
                        continue
                    lengths = [r['longest_miss_run'] for r in selected]
                    deltas = [r['longest_miss_delta_vs_C0'] for r in selected]
                    summaries.append(dict(scope=scope, condition=condition, arrangement=plan, group=group, cohort=cohort,
                        target_frames=sum(r['target_frames'] for r in selected), tracks=len(selected),
                        missed_frames=sum(r['missed_frames'] for r in selected), miss_run_count=sum(r['miss_run_count'] for r in selected),
                        longest_miss_track_mean=float(np.mean(lengths)), longest_miss_track_median=float(np.median(lengths)),
                        longest_miss_track_q90=float(np.quantile(lengths, .9)), longest_miss_max=max(lengths),
                        tracks_shorter_vs_C0=sum(d < 0 for d in deltas), tracks_same_vs_C0=sum(d == 0 for d in deltas),
                        tracks_longer_vs_C0=sum(d > 0 for d in deltas), longest_delta_track_mean_vs_C0=float(np.mean(deltas))))
    if scope == 'full':
        for plan in PLANS:
            selected = [r for r in tracks if r['condition'] == 'C0' and r['arrangement'] == plan and r['cohort'] == 'all']
            assert len(selected) == 130 and sum(r['target_frames'] for r in selected) == 5200
            assert sum(r['missed_frames'] for r in selected) == (4500 if plan == 'O' else 4480)
    write_csv(output / 'CONTINUOUS_MISS_TRACKS.csv', tracks, fields=None if tracks else ['scope', 'condition', 'target_frames'])
    write_csv(output / 'CONTINUOUS_MISS_SUMMARY.csv', summaries, fields=None if summaries else ['scope', 'condition', 'target_frames'])
    write_csv(output / 'C0_MISS_SEGMENTS.csv', release_segments, fields=None if release_segments else ['scope', 'arrangement', 'target_frames'])


def summarize(scope, conditions):
    output = BASE / 'evaluation' / scope
    records, areas, metadata, frames, source = {}, {}, {}, [], {}
    frozen_cohorts = cohorts()
    for condition in conditions:
        for sequence in SEQUENCES:
            a, b, receipt, f = read_record(condition, sequence, scope)
            records[condition, sequence], areas[condition, sequence] = a, b
            if sequence in metadata:
                assert metadata[sequence] == receipt['objects']
                assert np.array_equal(a['eligible'], records['C0', sequence]['eligible'])
            metadata[sequence] = receipt['objects']
            frames.extend(f)
            path = output / condition / sequence / 'COMPLETE.json'
            source[str(path)] = digest(path)
    objects = []
    all_gt_rows = []
    for sequence in SEQUENCES:
        for idx, meta in enumerate(metadata[sequence]):
            for pi, plan in enumerate(PLANS):
                row = dict(scope=scope, sequence=sequence, frame=meta['frame'], source_frame=meta['source_frame'],
                    track_id=meta['track_id'], annotation_id=meta['annotation_id'], arrangement=plan,
                    threshold=.5, strict_small=int(meta['strict_small']), eligible=int(records['C0', sequence]['eligible'][pi, idx]),
                    bbox=json.dumps(meta['bbox'], separators=(',', ':')))
                for condition in conditions:
                    for key in ['matched', 'box_max', 'peak_x', 'peak_y', 'component_area']:
                        value = records[condition, sequence][key][pi, idx].item()
                        row[condition + '_' + key] = value if np.isfinite(value) else ''
                    row[condition + '_gain_vs_C0'] = int(row[condition + '_matched'] and not row['C0_matched'])
                    row[condition + '_loss_vs_C0'] = int(not row[condition + '_matched'] and row['C0_matched'])
                all_gt_rows.append(dict(row))
                if not meta['strict_small']:
                    continue
                cohort = frozen_cohorts[sequence, meta['frame'], meta['annotation_id']]
                assert int(cohort['track_id']) == meta['track_id']
                row['cohort'] = cohort['cohort']
                row.update({key: int(cohort[key]) for key in COHORTS})
                row['original_C0_matched'] = int(cohort['original_matched' if plan == 'O' else 'symmetric_matched'])
                row['candidate_window_mixed'] = int(0 < int(cohort['success_window_count']) < int(cohort['covering_window_count']))
                if scope == 'full':
                    assert row['C0_matched'] == row['original_C0_matched']
                objects.append(row)
    assert len(objects) == 10400 and len(all_gt_rows) == 32594
    summaries, concentration, protection = [], [], []
    for condition in conditions:
        for pi, plan in enumerate(PLANS):
            for group, sequences in [('pooled', SEQUENCES)] + [(s, [s]) for s in SEQUENCES]:
                selected = [r for r in objects if r['arrangement'] == plan and r['sequence'] in sequences and r['eligible']]
                costs = cost(records, areas, condition, pi, plan, sequences)
                baseline = cost(records, areas, 'C0', pi, plan, sequences)
                gain = [r for r in selected if r[condition + '_gain_vs_C0']]
                loss = [r for r in selected if r[condition + '_loss_vs_C0']]
                row = dict(scope=scope, condition=condition, arrangement=plan, group=group, threshold=.5,
                    small_gt=len(selected), small_tracks=len({(r['sequence'], r['track_id']) for r in selected}),
                    M=sum(r[condition + '_matched'] for r in selected), C0_M_same=sum(r['C0_matched'] for r in selected),
                    G_vs_C0_same=len(gain), L_vs_C0_same=len(loss), net_vs_C0_same=len(gain)-len(loss), **costs)
                assert row['M'] == row['C0_M_same'] + row['net_vs_C0_same']
                for key in ['unmatched_components', 'positive_pixels_outside_boxes', 'unmatched_component_area_sum']:
                    row['C0_' + key] = baseline[key]
                    row['delta_' + key + '_vs_C0'] = costs[key] - baseline[key]
                row['U_budget'] = 4015 if scope == 'full' and group == 'pooled' else ''
                row['passes_pooled_U_budget'] = int(costs['unmatched_components'] <= 4015) if row['U_budget'] else ''
                for cohort in COHORTS:
                    sub = [r for r in selected if r[cohort]]
                    restored = [r for r in sub if r[condition + '_matched']]
                    row[cohort + '_gt'] = len(sub)
                    row[cohort + '_matched'] = len(restored)
                    row[cohort + '_matched_tracks'] = len({(r['sequence'], r['track_id']) for r in restored})
                    row[cohort + '_matched_clips'] = len({r['sequence'] for r in restored})
                selected_frames = [r for r in frames if r['condition'] == condition and r['arrangement'] == plan and r['sequence'] in sequences]
                row['common_valid_pixels'] = sum(int(r['common_valid_pixels']) for r in selected_frames)
                row['common_valid_fraction'] = row['common_valid_pixels'] / (len(sequences) * 160 * 512 * 640)
                row['mask_boundary_components'] = sum(int(r['mask_boundary_components']) for r in selected_frames)
                summaries.append(row)
                if group == 'pooled':
                    for outcome, sub in [('gain_vs_C0', gain), ('loss_vs_C0', loss),
                        ('low_recovered', [r for r in selected if r['low_all_05'] and r[condition + '_matched']]),
                        ('extreme_recovered', [r for r in selected if r['extreme_all_001'] and r[condition + '_matched']])]:
                        counter = collections.Counter((r['sequence'], r['track_id']) for r in sub)
                        ranked = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
                        concentration.append(dict(scope=scope, condition=condition, arrangement=plan, outcome=outcome,
                            frames=len(sub), tracks=len(counter), clips=len({key[0] for key in counter}),
                            top1_frames=ranked[0][1] if ranked else 0, top5_frames=sum(v for _, v in ranked[:5]),
                            ranked_tracks=json.dumps([dict(sequence=k[0], track_id=k[1], frames=v) for k, v in ranked])))
        left = {(r['sequence'], r['frame'], r['annotation_id']): r for r in objects if r['arrangement'] == 'O'}
        right = {(r['sequence'], r['frame'], r['annotation_id']): r for r in objects if r['arrangement'] == 'S'}
        for cohort in ['all'] + COHORTS:
            keys = [k for k, r in left.items() if r['eligible'] and right[k]['eligible'] and (cohort == 'all' or r[cohort])]
            o = np.array([left[k][condition + '_matched'] for k in keys], bool)
            s = np.array([right[k][condition + '_matched'] for k in keys], bool)
            protection.append(dict(scope=scope, condition=condition, cohort=cohort, target_frames=len(keys),
                O_matched=int(o.sum()), S_matched=int(s.sum()), both_matched=int((o & s).sum()),
                O_only=int((o & ~s).sum()), S_only=int((~o & s).sum()), neither_matched=int((~o & ~s).sum()),
                lost_dual_success=int((~(o & s)).sum()) if cohort == 'both_fixed_success' else '',
                lost_both_arrangements=int((~o & ~s).sum()) if cohort == 'both_fixed_success' else ''))
    if scope == 'full':
        assert [(r['M'], r['unmatched_components']) for r in summaries if r['condition'] == 'C0' and r['group'] == 'pooled'] == [(700, 4015), (720, 3967)]
    write_csv(output / 'SUMMARY.csv', [r for r in summaries if r['group'] == 'pooled'])
    write_csv(output / 'SEQUENCE_SUMMARY.csv', [r for r in summaries if r['group'] != 'pooled'])
    write_csv(output / 'OBJECT_COMPARISON.csv', objects)
    write_csv(output / 'ALL_GT_OBJECT_COMPARISON.csv', all_gt_rows)
    write_csv(output / 'FRAME_COUNTS.csv', frames)
    write_csv(output / 'RECOVERY_CONCENTRATION.csv', concentration)
    write_csv(output / 'PROTECTED_AND_WINDOW_RETENTION.csv', protection)
    trajectory_tables(objects, conditions, output, scope)
    write_json(output / 'SUMMARY_COMPLETE.json', dict(status='COMPLETE', scope=scope, conditions=conditions,
        threshold=.5, plans=PLANS, frozen_strict_small_target_frames=5200, frozen_tracks=130,
        frozen_cohorts={'low_all_05': 4342, 'extreme_all_001': 4170, 'both_fixed_success': 646},
        continuity='Local and source frame both advance by one; missing labels or common-support holes break runs.',
        source_sha256=source, code_sha256=digest(__file__),
        output_sha256={p.name: digest(p) for p in output.glob('*.csv')}))
    print(json.dumps({'status': 'SUMMARIZED', 'scope': scope, 'conditions': conditions,
        'rows': [{k: r[k] for k in ['condition', 'arrangement', 'M', 'unmatched_components', 'G_vs_C0_same', 'L_vs_C0_same', 'low_all_05_matched']} for r in summaries if r['group'] == 'pooled']}), flush=True)


def transform_point(matrix, point):
    """Bbox coordinates use pixel centers .5; OpenCV H uses integer centers."""
    value = matrix @ np.array([point[0] - .5, point[1] - .5, 1.], np.float64)
    if not np.isfinite(value).all() or abs(value[2]) < 1e-12:
        return None
    return value[:2] / value[2] + .5


def inside_boxes(points, annotations):
    # Correspondence coordinates are OpenCV pixel-index coordinates.
    inside = np.zeros(len(points), bool)
    for ann in annotations:
        x, y, w, h = ann['bbox']
        inside |= ((points[:, 0] + .5 >= x) & (points[:, 0] + .5 < x + w)
                   & (points[:, 1] + .5 >= y) & (points[:, 1] + .5 < y + h))
    return inside


def optional_float(row, key):
    value = row.get(key, '')
    if value == '' or value is None:
        return None
    result = float(value)
    return result if np.isfinite(result) else None


def pair_target_overlap(path, gt, previous, current):
    with np.load(path, allow_pickle=False) as archive:
        matrix = archive['H']
        fit_indices = archive['fit_indices'].astype(int)
        fit_inlier = archive['fit_inlier_mask'].astype(bool)
        if len(fit_inlier) == len(fit_indices):
            selected = fit_indices[fit_inlier]
        else:
            assert len(fit_inlier) == len(archive['source_points'])
            selected = np.flatnonzero(fit_inlier)
        heldout = archive['heldout_indices'].astype(int)
        row = {}
        for label, indices in [('fit_inlier', selected), ('heldout', heldout)]:
            p0 = archive['source_points'][indices]
            p1 = archive['destination_points'][indices]
            inside0 = inside_boxes(p0, gt[previous])
            inside1 = inside_boxes(p1, gt[current])
            row.update({
                'posthoc_' + label + '_points_checked': len(indices),
                'posthoc_' + label + '_in_source_target_count': int(inside0.sum()),
                'posthoc_' + label + '_in_destination_target_count': int(inside1.sum()),
                'posthoc_' + label + '_in_either_target_count': int((inside0 | inside1).sum()),
                'posthoc_' + label + '_in_either_target_fraction': float((inside0 | inside1).mean()) if len(indices) else '',
            })
    return matrix, row


def motion_join():
    """Labels enter only here, after image-only estimation and trust decisions."""
    object_path = BASE / 'evaluation/full/OBJECT_COMPARISON.csv'
    objects = read_csv(object_path)
    conditions = [c for c in CONDITIONS if c + '_matched' in objects[0]]
    for row in objects:
        for key in ['frame', 'source_frame', 'track_id', 'annotation_id', 'eligible'] + COHORTS:
            row[key] = int(row[key])
        for condition in conditions:
            for suffix in ['matched', 'gain_vs_C0', 'loss_vs_C0']:
                row[condition + '_' + suffix] = int(row[condition + '_' + suffix])
    scorer, helper, source_hashes = frozen_sources()
    source_hashes[str(object_path)] = digest(object_path)
    frame_path = BASE / 'evaluation/full/FRAME_COUNTS.csv'
    frame_rows = read_csv(frame_path)
    frame_counts = {(r['sequence'], r['arrangement'], int(r['frame']), r['condition']): r for r in frame_rows}
    source_hashes[str(frame_path)] = digest(frame_path)
    frame_motion, points_quality, annotation_meta, window_quality = {}, {}, {}, {}
    pair_rows, checked_pairs = [], set()
    annotation_by_track, frame_metadata = {}, {}
    for sequence in SEQUENCES:
        annotation_path = P / 'data' / sequence / 'annotations.json'
        gt, metadata, offsets, _ = scorer.gt_data(annotation_path, helper)
        annotation_meta[sequence] = gt
        for meta in metadata:
            annotation_by_track[sequence, meta['frame'], meta['track_id']] = meta
            frame_metadata[sequence, meta['frame']] = meta['source_frame']
        adjacent_path = BASE / 'motion' / sequence / 'adjacent.csv'
        if adjacent_path.exists():
            source_hashes[str(adjacent_path)] = digest(adjacent_path)
            for item in read_csv(adjacent_path):
                previous, current = int(item['source_frame']), int(item['destination_frame'])
                assert current == previous + 1
                key = (sequence, current)
                assert key not in frame_motion
                pair_path = BASE / 'motion' / sequence / 'pairs' / f'{previous:03d}_to_{current:03d}.npz'
                matrix, overlap = pair_target_overlap(pair_path, gt, previous, current)
                assert matrix.shape == (3, 3)
                source_hashes[str(pair_path)] = digest(pair_path)
                frame_motion[key] = (item, matrix)
                points_quality[key] = overlap
                pair_rows.append(dict(sequence=sequence, source_frame=previous, destination_frame=current,
                                      usage='adjacent', **overlap))
                checked_pairs.add(pair_path)
        for start in STARTS:
            path = BASE / 'motion' / sequence / 'windows' / f'window_{start:03d}.npz'
            if path.exists():
                with np.load(path, allow_pickle=False) as archive:
                    trusted = archive['trusted'].astype(bool)
                    usable = bool(archive['window_trusted'])
                assert trusted.shape == (40,)
                window_quality[sequence, start] = (trusted, usable)
                source_hashes[str(path)] = digest(path)
        pair_root = BASE / 'motion' / sequence / 'pairs'
        for pair_path in sorted(pair_root.glob('*_to_*.npz')):
            if pair_path in checked_pairs:
                continue
            previous, current = map(int, pair_path.stem.split('_to_'))
            _, overlap = pair_target_overlap(pair_path, gt, previous, current)
            pair_rows.append(dict(sequence=sequence, source_frame=previous, destination_frame=current,
                                  usage='window_to_anchor', **overlap))
            source_hashes[str(pair_path)] = digest(pair_path)
    # Keep original C0 miss-segment identities and lengths on every observation.
    trajectory = module(Q / 'tlfp_pilot_20261012/evaluation_code/trajectory_tables.py', 'motion_join_continuity')
    partitions = collections.defaultdict(list)
    for row in objects:
        partitions[row['arrangement'], row['sequence'], row['track_id']].append(row)
    for raw in partitions.values():
        raw.sort(key=lambda r: (r['frame'], r['source_frame']))
        for condition in conditions:
            for row in raw:
                row[condition + '_containing_miss_length'] = 0
                row[condition + '_containing_miss_start'] = ''
                row[condition + '_containing_miss_end'] = ''
            for segment in trajectory.segments(raw, lambda r, c=condition: not r[c + '_matched']):
                for row in segment:
                    row[condition + '_containing_miss_length'] = len(segment)
                    row[condition + '_containing_miss_start'] = segment[0]['frame']
                    row[condition + '_containing_miss_end'] = segment[-1]['frame']
    quality_fields = ['reason', 'matches', 'fit_count', 'holdout_count', 'fit_inliers', 'fit_inlier_ratio',
        'source_hull_fraction', 'destination_hull_fraction', 'source_grid_cells', 'destination_grid_cells',
        'heldout_before_median_px', 'heldout_before_p90_px', 'heldout_after_median_px', 'heldout_after_p90_px',
        'heldout_within_3px_fraction', 'displacement_px', 'dx_px', 'dy_px', 'rotation_deg', 'scale',
        'min_local_scale', 'max_local_scale', 'no_fold', 'corner_bounded', 'roundtrip_valid_fraction']
    joined = []
    for original in objects:
        row = dict(original)
        sequence, frame, track_id = row['sequence'], row['frame'], row['track_id']
        row['frame_cost_join_key'] = f"{sequence}:{row['arrangement']}:{frame}"
        for condition in conditions:
            count = frame_counts[sequence, row['arrangement'], frame, condition]
            for key in ['unmatched_components', 'positive_pixels_outside_boxes', 'unmatched_component_area', 'component_max_area']:
                row[condition + '_frame_' + key] = int(count[key])
        row['motion_status'] = 'first_frame_no_pair' if frame == 0 else 'not_estimated'
        row['bg_trusted'] = ''
        for key in quality_fields:
            row['bg_' + key] = ''
        for key in ['bg_displacement_change_px', 'target_image_dx_px', 'target_image_dy_px', 'target_image_displacement_px',
                    'target_relative_dx_px', 'target_relative_dy_px', 'target_relative_displacement_px',
                    'target_previous_observation_contiguous']:
            row[key] = ''
        for label in ['fit_inlier', 'heldout']:
            for suffix in ['points_checked', 'in_source_target_count', 'in_destination_target_count',
                           'in_either_target_count', 'in_either_target_fraction']:
                row['posthoc_' + label + '_' + suffix] = ''
        pair = frame_motion.get((sequence, frame))
        if pair is not None:
            motion, matrix = pair
            trusted = str(motion['trusted']).lower() in ['1', 'true']
            row['motion_status'] = 'trusted' if trusted else 'untrusted'
            row['bg_trusted'] = int(trusted)
            for key in quality_fields:
                row['bg_' + key] = motion.get(key, '')
            row.update(points_quality[sequence, frame])
            previous_pair = frame_motion.get((sequence, frame - 1))
            if previous_pair is not None:
                earlier = previous_pair[0]
                if trusted and str(earlier['trusted']).lower() in ['1', 'true']:
                    values = [optional_float(motion, 'dx_px'), optional_float(motion, 'dy_px'),
                              optional_float(earlier, 'dx_px'), optional_float(earlier, 'dy_px')]
                    if all(v is not None for v in values):
                        row['bg_displacement_change_px'] = float(np.hypot(values[0] - values[2], values[1] - values[3]))
            current = annotation_by_track.get((sequence, frame, track_id))
            previous = annotation_by_track.get((sequence, frame - 1, track_id))
            contiguous = (previous is not None and current is not None
                          and current['source_frame'] == previous['source_frame'] + 1)
            row['target_previous_observation_contiguous'] = int(contiguous)
            if contiguous:
                x, y, w, h = current['bbox']
                px, py, pw, ph = previous['bbox']
                center, before = np.array([x + w / 2, y + h / 2]), np.array([px + pw / 2, py + ph / 2])
                difference = center - before
                row['target_image_dx_px'], row['target_image_dy_px'] = map(float, difference)
                row['target_image_displacement_px'] = float(np.linalg.norm(difference))
                predicted = transform_point(matrix, before)
                # Do not report a failed geometric estimate as target relative motion.
                if trusted and predicted is not None:
                    relative = center - predicted
                    row['target_relative_dx_px'], row['target_relative_dy_px'] = map(float, relative)
                    row['target_relative_displacement_px'] = float(np.linalg.norm(relative))
        covering = [s for s in PLANS[row['arrangement']] if s <= frame < s + 40]
        row['covering_windows'] = len(covering)
        available = [s for s in covering if (sequence, s) in window_quality]
        row['geometry_windows_estimated'] = len(available)
        row['trusted_covering_windows'] = sum(window_quality[sequence, s][1] for s in available)
        row['frame_to_anchor_trusted_windows'] = sum(bool(window_quality[sequence, s][0][frame - s]) for s in available)
        joined.append(row)
    common_path = BASE / 'evaluation/common/OBJECT_COMPARISON.csv'
    if common_path.exists():
        common = {(r['sequence'], int(r['frame']), int(r['annotation_id']), r['arrangement']): r for r in read_csv(common_path)}
        for row in joined:
            item = common[row['sequence'], row['frame'], row['annotation_id'], row['arrangement']]
            row['common_eligible'] = int(item['eligible'])
            for condition in conditions:
                row['common_' + condition + '_matched'] = int(item[condition + '_matched'])
        source_hashes[str(common_path)] = digest(common_path)
    write_csv(BASE / 'MOTION_AND_DETECTION.csv', joined)
    # Descriptive continuous contrasts, with each individual trajectory retained.
    # No bin, classifier, threshold, or quality gate is fitted to detection labels.
    metrics = ['bg_displacement_px', 'bg_displacement_change_px', 'bg_rotation_deg', 'bg_scale',
               'bg_fit_inlier_ratio', 'bg_heldout_before_median_px', 'bg_heldout_after_median_px', 'target_image_displacement_px',
               'target_relative_displacement_px', 'posthoc_fit_inlier_in_either_target_fraction',
               'posthoc_heldout_in_either_target_fraction']
    association = []
    for plan in PLANS:
        raw = [r for r in joined if r['arrangement'] == plan]
        partitions = [('pooled', '', '', raw)]
        partitions += [('track', seq, tid, [r for r in raw if r['sequence'] == seq and r['track_id'] == tid])
                       for seq, tid in sorted({(r['sequence'], r['track_id']) for r in raw})]
        for level, sequence, track_id, selected in partitions:
            predicates = {'C0_missed': lambda r: not r['C0_matched'], 'C0_success': lambda r: r['C0_matched']}
            for condition in conditions:
                if condition != 'C0':
                    predicates[condition + '_gain'] = lambda r, c=condition: r[c + '_gain_vs_C0']
                    predicates[condition + '_loss'] = lambda r, c=condition: r[c + '_loss_vs_C0']
            for outcome, predicate in predicates.items():
                sub = [r for r in selected if predicate(r)]
                row = dict(arrangement=plan, level=level, sequence=sequence, track_id=track_id, outcome=outcome,
                    target_frames=len(sub), tracks=len({(r['sequence'], r['track_id']) for r in sub}),
                    trusted_motion_frames=sum(r['bg_trusted'] == 1 for r in sub),
                    untrusted_motion_frames=sum(r['bg_trusted'] == 0 for r in sub),
                    unavailable_motion_frames=sum(r['bg_trusted'] == '' for r in sub))
                for key in metrics:
                    values = [optional_float(r, key) for r in sub if r['bg_trusted'] == 1]
                    values = [v for v in values if v is not None]
                    row[key + '_n'] = len(values)
                    row[key + '_mean'] = float(np.mean(values)) if values else ''
                    row[key + '_median'] = float(np.median(values)) if values else ''
                    row[key + '_q90'] = float(np.quantile(values, .9)) if values else ''
                association.append(row)
    write_csv(BASE / 'evaluation/MOTION_CONDITIONS.csv', association)
    motion_frame_rows = []
    for raw in frame_rows:
        row = dict(raw)
        pair = frame_motion.get((row['sequence'], int(row['frame'])))
        row['bg_trusted'] = int(str(pair[0]['trusted']).lower() in ['true', '1']) if pair else ''
        for key in quality_fields:
            row['bg_' + key] = pair[0].get(key, '') if pair else ''
        motion_frame_rows.append(row)
    write_csv(BASE / 'evaluation/MOTION_FRAME_COUNTS.csv', motion_frame_rows)
    write_csv(BASE / 'evaluation/POSTHOC_PAIR_TARGET_OVERLAP.csv', pair_rows,
              fields=None if pair_rows else ['sequence', 'source_frame', 'destination_frame', 'usage'])
    write_json(BASE / 'evaluation/MOTION_JOIN_COMPLETE.json', dict(status='COMPLETE', rows=len(joined),
        conditions=conditions, motion_pairs=len(frame_motion), source_sha256=source_hashes,
        output_sha256=digest(BASE / 'MOTION_AND_DETECTION.csv'), code_sha256=digest(__file__),
        interpretation='Image-only geometry is estimated first. Annotation boxes enter only this posthoc join. Continuous outcome-conditioned contrasts are associations, not causal estimates. Untrusted/missing relative motion stays blank; no classification probe or label-tuned bins.',
        frame_costs='MOTION_AND_DETECTION repeats frame costs across objects. Sum unique frames in evaluation/MOTION_FRAME_COUNTS.csv, never repeated object rows.',
        coordinate_convention='Annotation bbox center minus 0.5 -> OpenCV H -> divide homogeneous coordinate -> plus 0.5.'))
    print(json.dumps({'status': 'MOTION_JOIN_COMPLETE', 'rows': len(joined), 'motion_pairs': len(frame_motion)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition', choices=CONDITIONS)
    parser.add_argument('--sequence', choices=SEQUENCES)
    parser.add_argument('--scope', choices=['full', 'common', 'both'], default='both')
    parser.add_argument('--summary', action='store_true')
    parser.add_argument('--conditions', nargs='+', choices=CONDITIONS, default=CONDITIONS)
    parser.add_argument('--motion-join', action='store_true')
    args = parser.parse_args()
    scopes = ['full', 'common'] if args.scope == 'both' else [args.scope]
    if args.summary:
        assert args.conditions[0] == 'C0'
        for scope in scopes:
            summarize(scope, args.conditions)
    if args.motion_join:
        motion_join()
    if not args.summary and not args.motion_join:
        if not args.condition:
            parser.error('--condition is required for scoring')
        for sequence in [args.sequence] if args.sequence else SEQUENCES:
            for scope in scopes:
                score(args.condition, sequence, scope)


if __name__ == '__main__':
    main()
