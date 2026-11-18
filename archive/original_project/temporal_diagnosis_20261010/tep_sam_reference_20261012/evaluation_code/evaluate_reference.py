#!/usr/bin/env python3
"""One dense reference prediction per frame, frozen diagnostic point matching.

CPU scoring only. No model loading, training, threshold sweep or oracle selection.
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

BASE = Path(__file__).resolve().parents[1]
Q = BASE.parent
P = Q / 'stage1_goal'
OLD = Q / 'top1_background_control_20261011'
EVAL = BASE / 'evaluation'
SEQS = ['DJI_0026_2', 'DJI_0067_2']
PLANS = ['O', 'S']
FIELDS = ['gt_count', 'matched_count', 'component_count', 'unmatched_components',
          'positive_pixels', 'positive_pixels_outside_boxes']
COHORT_PATH = Q / 'low_response_information_20261011/COHORTS.csv'
ANCHOR = Q / 'bc_tpro_pilot_20261011/evaluation_preflight/SUMMARY_COMPLETE.json'
HEADS = {'tep_sam': ('probabilities.npz', 'probabilities_sha256'),
         'tep_auxiliary': ('auxiliary_probabilities.npz', 'auxiliary_probabilities_sha256')}
COHORTS = ['all', 'low_all_05', 'extreme_all_001', 'both_fixed_success']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def jsonwrite(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def csvwrite(path, rows):
    assert rows, path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def load_reference(sequence):
    """Reuse already independently audited release caches and verify frozen identity."""
    anchor = json.loads(ANCHOR.read_text())
    assert anchor['status'] == 'COMPLETE'
    folder = OLD / 'evaluation/release' / sequence
    hashes = {}
    for path in [COHORT_PATH] + [folder / name for name in
                               ['COMPLETE.json', 'scores.npz', 'component_areas.npz', 'FRAME_COUNTS.csv']]:
        actual = digest(path)
        assert actual == anchor['source_sha256'][str(path)], ('Frozen reference changed', path)
        hashes[str(path)] = actual
    receipt = json.loads((folder / 'COMPLETE.json').read_text())
    assert receipt['status'] == 'COMPLETE' and receipt['threshold'] == .5
    assert receipt['condition'] == 'release' and receipt['sequence'] == sequence
    # The release cache preceded the script's later summary extension. The final
    # top-1% receipt, already used by the E0/E1/E2 scorer, pins the current wrapper.
    final_cp = OLD / 'evaluation/lambda0_top1' / sequence / 'COMPLETE.json'
    assert digest(final_cp) == anchor['source_sha256'][str(final_cp)]
    final_receipt = json.loads(final_cp.read_text())
    assert final_receipt['script_sha256'] == digest(OLD / 'evaluation_code/evaluate.py')
    hashes[str(final_cp)] = digest(final_cp)
    for path, expected in receipt['source_sha256'].items():
        if Path(path).suffix == '.py' or Path(path).name == 'annotations.json':
            assert digest(path) == expected, ('Frozen scoring input changed', path)
    with np.load(folder / 'scores.npz', allow_pickle=False) as archive:
        scores = {key: archive[key] for key in archive.files}
    with np.load(folder / 'component_areas.npz', allow_pickle=False) as archive:
        areas = {key: archive[key] for key in archive.files}
    assert scores['plans'].tolist() == PLANS and float(scores['threshold']) == .5
    assert scores['frame_counts'].shape == (2, 160, 6)
    return scores, areas, receipt, hashes


def matching_inputs(sequence):
    scorer = load_module(P / 'pilot/score_pilot.py', 'existing_pilot_scorer')
    helper, helper_path = scorer.helper()
    ap = P / 'data' / sequence / 'annotations.json'
    gt, meta, offsets, excluded = scorer.gt_data(ap, helper)
    return helper, gt, meta, offsets, excluded, {
        str(ap): digest(ap), str(helper_path): digest(helper_path),
        str(P / 'pilot/score_pilot.py'): digest(P / 'pilot/score_pilot.py')}


def compute_scores(probabilities, sequence, condition):
    assert probabilities.shape == (160, 512, 640) and probabilities.dtype == np.float32
    assert np.isfinite(probabilities).all()
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    helper, gt, meta, offsets, excluded, hashes = matching_inputs(sequence)
    n = len(meta)
    arrays = dict(matched=np.zeros(n, np.uint8), box_max=np.zeros(n, np.float32),
                  peak_x=np.full(n, np.nan, np.float32), peak_y=np.full(n, np.nan, np.float32),
                  component_area=np.zeros(n, np.int32), frame_counts=np.zeros((160, 6), np.int64),
                  threshold_equal_pixels=np.zeros(160, np.int64),
                  strict_small=np.array([bool(row['strict_small']) for row in meta]),
                  offsets=offsets, threshold=np.array(.5))
    frame_rows, all_areas, all_unmatched, area_offsets = [], [], [], [0]
    for frame in range(160):
        rows, stats, comps = helper.match_frame(probabilities[frame], gt[frame], .5)
        for idx, row in enumerate(rows, int(offsets[frame])):
            assert (row['annotation_id'], row['track_id']) == (meta[idx]['annotation_id'], meta[idx]['track_id'])
            arrays['matched'][idx] = row['matched']
            arrays['box_max'][idx] = row['box_max']
            if row['matched']:
                for key in ['peak_x', 'peak_y', 'component_area']:
                    arrays[key][idx] = row[key]
        arrays['frame_counts'][frame] = [stats[key] for key in FIELDS]
        arrays['threshold_equal_pixels'][frame] = np.count_nonzero(probabilities[frame] == np.float32(.5))
        aa = [int(c['area']) for c in comps]
        uu = [int(c['unmatched_component']) for c in comps]
        assert sum(aa) == stats['positive_pixels'] and sum(uu) == stats['unmatched_components']
        all_areas.extend(aa)
        all_unmatched.extend(uu)
        area_offsets.append(len(all_areas))
        ua = [a for a, unmatched in zip(aa, uu) if unmatched]
        frame_rows.append(dict(condition=condition, sequence=sequence, frame=frame,
                               source_frame=meta[offsets[frame]]['source_frame'] if len(rows) else
                               int(Path(json.loads((P/'data'/sequence/'annotations.json').read_text())['images'][frame]['file_name']).stem),
                               threshold=.5, prediction_rule='single_current_frame',
                               threshold_equal_pixels=int(arrays['threshold_equal_pixels'][frame]),
                               **{key: int(stats[key]) for key in FIELDS},
                               unmatched_component_area=sum(ua),
                               unmatched_component_max_area=max(ua, default=0),
                               component_max_area=max(aa, default=0)))
    area = dict(areas=np.array(all_areas, np.int32), unmatched=np.array(all_unmatched, np.uint8),
                offsets=np.array(area_offsets, np.int64))
    return arrays, area, frame_rows, meta, excluded, hashes


def save_scores(folder, computed, condition, sequence, provenance):
    if (folder / 'COMPLETE.json').exists():
        raise FileExistsError('Refuse to overwrite completed score: ' + str(folder))
    folder.mkdir(parents=True, exist_ok=True)
    arrays, areas, frames, meta, excluded, hashes = computed
    np.savez_compressed(folder / 'scores.npz', **arrays)
    np.savez_compressed(folder / 'component_areas.npz', **areas)
    csvwrite(folder / 'FRAME_COUNTS.csv', frames)
    outputs = {name: digest(folder / name) for name in ['scores.npz', 'component_areas.npz', 'FRAME_COUNTS.csv']}
    jsonwrite(folder / 'COMPLETE.json', dict(status='COMPLETE', condition=condition, sequence=sequence,
              threshold=.5, prediction_rule='single_current_frame', objects=meta,
              excluded_annotations=excluded, source_sha256={**hashes, **provenance.get('source_sha256', {})},
              output_sha256=outputs, script_sha256=digest(__file__), provenance=provenance,
              scope='Diagnostic one-to-one component-peak-in-box matching; U is not official Fa.'))


def score(sequence, condition, predictions_root):
    began = time.perf_counter()
    reference, _, ref_receipt, _ = load_reference(sequence)
    folder = predictions_root / sequence
    filename, hash_key = HEADS[condition]
    cp, pp = folder / 'COMPLETE.json', folder / filename
    receipt = json.loads(cp.read_text())
    assert receipt['status'] == 'COMPLETE' and receipt['sequence'] == sequence
    assert receipt['raw_sha256'] == digest(P / 'data' / sequence / 'raw_frames.npy')
    assert receipt['annotations_sha256'] == digest(P / 'data' / sequence / 'annotations.json')
    assert receipt[hash_key] == digest(pp)
    protocol = Path(receipt['protocol_path'])
    assert protocol.is_file() and receipt['protocol_sha256'] == digest(protocol)
    assert len(receipt['checkpoint_sha256']) == 64 and len(receipt['repository_commit']) == 40
    if condition == 'tep_auxiliary':
        assert receipt['auxiliary_head_trained_parameters_loaded'] is True
    images = json.loads((P / 'data' / sequence / 'annotations.json').read_text())['images']
    source_frames = [int(Path(im['file_name']).stem) for im in images]
    with np.load(pp, allow_pickle=False) as archive:
        assert np.array_equal(archive['frame_indices'], np.arange(160))
        assert np.array_equal(archive['source_frames'], source_frames)
        prob = archive['probabilities']
    computed = compute_scores(prob, sequence, condition)
    assert computed[3] == ref_receipt['objects']
    assert np.array_equal(computed[0]['strict_small'], reference['strict_small'])
    provenance = dict(prediction_receipt=str(cp), checkpoint_sha256=receipt['checkpoint_sha256'],
                      repository_commit=receipt['repository_commit'], protocol_sha256=receipt['protocol_sha256'],
                      output_key='probabilities', output_file=filename, source_sha256={str(cp): digest(cp), str(pp): digest(pp),
                                                                str(protocol): digest(protocol)})
    save_scores(EVAL / condition / sequence, computed, condition, sequence, provenance)
    print(json.dumps(dict(status='COMPLETE', condition=condition, sequence=sequence,
                          M=int(computed[0]['matched'][computed[0]['strict_small']].sum()),
                          U=int(computed[0]['frame_counts'][:, 3].sum()),
                          elapsed_seconds=time.perf_counter()-began)), flush=True)


def read_new(condition, sequence, output):
    folder = output / condition / sequence
    cp = folder / 'COMPLETE.json'
    receipt = json.loads(cp.read_text())
    assert receipt['status'] == 'COMPLETE' and receipt['threshold'] == .5
    assert receipt['condition'] == condition and receipt['sequence'] == sequence
    assert receipt['script_sha256'] == digest(__file__)
    for path, expected in receipt['source_sha256'].items():
        assert digest(path) == expected, ('Scoring source changed', path)
    for name, expected in receipt['output_sha256'].items():
        assert digest(folder / name) == expected, ('Scoring output changed', name)
    with np.load(folder / 'scores.npz', allow_pickle=False) as archive:
        scores = {key: archive[key] for key in archive.files}
    with np.load(folder / 'component_areas.npz', allow_pickle=False) as archive:
        areas = {key: archive[key] for key in archive.files}
    hashes = {str(folder / name): digest(folder / name) for name in
              ['COMPLETE.json', 'scores.npz', 'component_areas.npz', 'FRAME_COUNTS.csv']}
    assert scores['matched'].ndim == 1 and scores['frame_counts'].shape == (160, 6)
    return scores, areas, receipt, hashes


def summarize(conditions, output=EVAL):
    if (output / 'SUMMARY_COMPLETE.json').exists():
        raise FileExistsError(output / 'SUMMARY_COMPLETE.json')
    records, areas, metadata, hashes = {}, {}, {}, {}
    with COHORT_PATH.open() as stream:
        cohort_rows = list(csv.DictReader(stream))
    cohorts = {(r['sequence'], int(r['frame']), int(r['annotation_id'])): r for r in cohort_rows}
    assert len(cohorts) == len(cohort_rows) == 5200
    for key, count in [('low_all_05', 4342), ('extreme_all_001', 4170), ('both_fixed_success', 646)]:
        assert sum(int(r[key]) for r in cohort_rows) == count
    for sequence in SEQS:
        ref, aa, receipt, hh = load_reference(sequence)
        metadata[sequence] = receipt['objects']
        hashes.update(hh)
        for pi, plan in enumerate(PLANS):
            name = 'release_' + plan
            records[name, sequence] = {key: ref[key][pi] for key in
                                      ['matched', 'box_max', 'peak_x', 'peak_y', 'component_area', 'frame_counts']}
            areas[name, sequence] = {'areas': aa[plan + '_areas'], 'unmatched': aa[plan + '_unmatched']}
        identities = []
        for condition in conditions:
            ss, ar, rr, hh = read_new(condition, sequence, output)
            assert rr['objects'] == metadata[sequence]
            assert np.array_equal(ss['strict_small'], ref['strict_small'])
            records[condition, sequence], areas[condition, sequence] = ss, ar
            hashes.update(hh)
            if 'checkpoint_sha256' in rr['provenance']:
                identities.append((condition, rr['provenance']['checkpoint_sha256'],
                                   rr['provenance']['protocol_sha256'], rr['provenance']['repository_commit']))
        if sequence == SEQS[0]:
            identity_reference = identities
        else:
            assert identities == identity_reference, 'Reference conditions differ between clips'
    names = ['release_O', 'release_S'] + list(conditions)
    objects = []
    for sequence in SEQS:
        for idx, meta in enumerate(metadata[sequence]):
            if not meta['strict_small']:
                continue
            cohort = cohorts[sequence, meta['frame'], meta['annotation_id']]
            assert int(cohort['track_id']) == meta['track_id']
            row = {key: meta[key] for key in ['frame', 'source_frame', 'track_id', 'annotation_id']}
            row.update(sequence=sequence, threshold=.5, cohort=cohort['cohort'],
                       **{key: int(cohort[key]) for key in COHORTS[1:]})
            for name in names:
                for key in ['matched', 'box_max', 'peak_x', 'peak_y', 'component_area']:
                    value = records[name, sequence][key][idx].item()
                    row[name + '_' + key] = value if np.isfinite(value) else ''
            assert row['release_O_matched'] == int(cohort['original_matched'])
            assert row['release_S_matched'] == int(cohort['symmetric_matched'])
            for condition in conditions:
                for plan in PLANS:
                    before, after = row['release_' + plan + '_matched'], row[condition + '_matched']
                    row[condition + '_gain_vs_' + plan] = int(after and not before)
                    row[condition + '_loss_vs_' + plan] = int(before and not after)
            objects.append(row)
    assert len(objects) == 5200 and len({(r['sequence'], r['track_id']) for r in objects}) == 130
    csvwrite(output / 'OBJECT_COMPARISON.csv', objects)
    summaries, complement, concentration = [], [], []
    for name in names:
        for group, seqs in [('pooled', SEQS)] + [(s, [s]) for s in SEQS]:
            oo = [r for r in objects if r['sequence'] in seqs]
            totals = sum((records[name, s]['frame_counts'].sum(0) for s in seqs), np.zeros(6, np.int64))
            aa = np.concatenate([areas[name, s]['areas'] for s in seqs])
            uu = np.concatenate([areas[name, s]['unmatched'].astype(bool) for s in seqs])
            after = np.array([r[name + '_matched'] for r in oo], bool)
            row = dict(condition=name, prediction_rule='single_current_frame' if name in conditions else
                       'five_window_probability_max_' + name[-1], group=group, threshold=.5,
                       target_frames=len(oo), tracks=len({(r['sequence'], r['track_id']) for r in oo}),
                       clips=len(seqs), M=int(after.sum()), **{key: int(value) for key, value in zip(FIELDS, totals)},
                       threshold_equal_pixels=sum(int(records[name,s]['threshold_equal_pixels'].sum()) for s in seqs)
                       if name in conditions else '',
                       pooled_U_budget=4015 if group == 'pooled' else '',
                       passes_pooled_U_budget=int(totals[3] <= 4015) if group == 'pooled' else '')
            for prefix, values in [('component', aa), ('unmatched_component', aa[uu])]:
                row[prefix + '_area_sum'] = int(values.sum())
                row[prefix + '_area_max'] = int(values.max()) if len(values) else 0
                for suffix, q in [('q50', .5), ('q95', .95), ('q99', .99)]:
                    row[prefix + '_area_' + suffix] = float(np.quantile(values, q)) if len(values) else 0.
            for cohort in COHORTS[1:]:
                mask = np.array([r[cohort] for r in oo], bool)
                selected = [r for r, value in zip(oo, mask & after) if value]
                row[cohort + '_frames'] = int(mask.sum())
                row[cohort + '_still_matched'] = len(selected)
                row[cohort + '_matched_tracks'] = len({(r['sequence'], r['track_id']) for r in selected})
                row[cohort + '_matched_clips'] = len({r['sequence'] for r in selected})
            for plan in PLANS:
                before = np.array([r['release_' + plan + '_matched'] for r in oo], bool)
                g, loss = after & ~before, ~after & before
                assert after.sum() == before.sum() + g.sum() - loss.sum()
                ref_totals = sum((records['release_' + plan, s]['frame_counts'].sum(0) for s in seqs), np.zeros(6, np.int64))
                ref_ua = sum(int(areas['release_' + plan, s]['areas'][areas['release_' + plan, s]['unmatched'].astype(bool)].sum()) for s in seqs)
                row.update({key + '_vs_' + plan: value for key, value in {
                    'G': int(g.sum()), 'L': int(loss.sum()), 'net_M': int(after.sum()-before.sum()),
                    'delta_U': int(totals[3]-ref_totals[3]), 'delta_outside_pixels': int(totals[5]-ref_totals[5]),
                    'delta_unmatched_component_area': int(aa[uu].sum())-ref_ua,
                    'recorded_background_costs_not_increased': int(totals[3] <= ref_totals[3] and
                        totals[5] <= ref_totals[5] and int(aa[uu].sum()) <= ref_ua)}.items()})
                if name in conditions:
                    complement.append(dict(condition=name, reference='release_' + plan, group=group,
                        scope='Post-hoc matched-set complementarity only; not fused detection performance',
                        both=int((before & after).sum()), reference_only=int(loss.sum()), reference_model_only=int(g.sum()),
                        neither=int((~before & ~after).sum()), oracle_set_union=int((before | after).sum())))
            summaries.append(row)
            if name in conditions and group == 'pooled':
                masks = {**{f'gain_vs_{p}': np.array([r[name+'_gain_vs_'+p] for r in oo], bool) for p in PLANS},
                         **{f'loss_vs_{p}': np.array([r[name+'_loss_vs_'+p] for r in oo], bool) for p in PLANS},
                         'low_recovered': after & np.array([r['low_all_05'] for r in oo], bool)}
                for outcome, mask in masks.items():
                    counts = collections.Counter((r['sequence'], r['track_id']) for r, value in zip(oo, mask) if value)
                    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
                    concentration.append(dict(condition=name, outcome=outcome, frames=int(mask.sum()),
                        tracks=len(counts), clips=len({k[0] for k in counts}), top1_frames=ranked[0][1] if ranked else 0,
                        top5_frames=sum(n for _, n in ranked[:5]),
                        ranked_tracks=json.dumps([dict(sequence=k[0], track_id=k[1], frames=n) for k, n in ranked], separators=(',', ':'))))
    pooled = [r for r in summaries if r['group'] == 'pooled']
    assert [(r['M'], r['unmatched_components'], r['positive_pixels_outside_boxes']) for r in pooled[:2]] == [(700,4015,38368),(720,3967,38641)]
    csvwrite(output / 'SUMMARY.csv', pooled)
    csvwrite(output / 'SEQUENCE_SUMMARY.csv', [r for r in summaries if r['group'] != 'pooled'])
    csvwrite(output / 'SET_COMPLEMENTARITY_DIAGNOSTIC_ONLY.csv', complement)
    csvwrite(output / 'RECOVERY_CONCENTRATION.csv', concentration)
    trajectory_tables(objects, names, conditions, output)
    if 'tep_auxiliary' in conditions and 'tep_sam' in conditions:
        auxiliary = []
        for group, seqs in [('pooled', SEQS)] + [(s, [s]) for s in SEQS]:
            for cohort in COHORTS:
                oo = [r for r in objects if r['sequence'] in seqs and (cohort == 'all' or r[cohort])]
                auxiliary.append(dict(group=group, cohort=cohort, frames=len(oo),
                    both_matched=sum(r['tep_sam_matched'] and r['tep_auxiliary_matched'] for r in oo),
                    main_only=sum(r['tep_sam_matched'] and not r['tep_auxiliary_matched'] for r in oo),
                    auxiliary_only=sum(r['tep_auxiliary_matched'] and not r['tep_sam_matched'] for r in oo),
                    neither=sum(not r['tep_sam_matched'] and not r['tep_auxiliary_matched'] for r in oo)))
        csvwrite(output / 'AUXILIARY_SUPPORT.csv', auxiliary)
    jsonwrite(output / 'SUMMARY_COMPLETE.json', dict(status='COMPLETE', threshold=.5, conditions=conditions,
        target_frames=5200, tracks=130, clips=2, main_rows=pooled, source_sha256=hashes,
        script_sha256=digest(__file__), single_prediction_per_reference_frame=True,
        limits=['Not official Pd/Fa', 'Not equal calibration or compute budget', 'No mechanism attribution from cross-model scores',
                'No deployed score for the oracle matched-set union'],
        output_sha256={p.name: digest(p) for p in sorted(output.glob('*.csv'))}))
    print(json.dumps(dict(status='COMPLETE', summary=str(output / 'SUMMARY.csv'), rows=pooled)), flush=True)


def trajectory_tables(objects, names, conditions, output):
    # Directly reuse the already audited continuity predicate and segment builder.
    old_path = Q / 'mutual_temporal_pilot_20261012/evaluation_code/trajectory_tables.py'
    old = load_module(old_path, 'frozen_trajectory_segments')
    partitions = collections.defaultdict(list)
    for row in objects:
        partitions[row['sequence'], row['track_id']].append(row)
    tracks, release_runs = [], []
    for (seq, track), raw in sorted(partitions.items()):
        raw.sort(key=lambda r: (r['frame'], r['source_frame']))
        for cohort in COHORTS:
            rows = [r for r in raw if cohort == 'all' or r[cohort]]
            if not rows:
                continue
            obs = old.segments(rows, lambda r: True)
            misses = {name: old.segments(rows, lambda r, n=name: not r[n + '_matched']) for name in names}
            for name in names:
                runs = misses[name]
                result = dict(condition=name, sequence=seq, track_id=track, cohort=cohort,
                    target_frames=len(rows), observation_segments=len(obs),
                    observation_gap_pairs=sum(not old.adjacent(a,b) for a,b in zip(rows,rows[1:])),
                    M=sum(r[name + '_matched'] for r in rows), missed_frames=sum(map(len, runs)),
                    miss_run_count=len(runs), longest_miss_run=max(map(len, runs), default=0),
                    miss_run_lengths=json.dumps([len(run) for run in runs], separators=(',', ':')))
                for plan in PLANS:
                    result['longest_miss_delta_vs_' + plan] = result['longest_miss_run'] - max(map(len, misses['release_' + plan]), default=0)
                    result['missed_frames_delta_vs_' + plan] = result['missed_frames'] - sum(map(len, misses['release_' + plan]))
                    result['G_vs_' + plan] = sum(r[name+'_matched'] and not r['release_'+plan+'_matched'] for r in rows)
                    result['L_vs_' + plan] = sum(not r[name+'_matched'] and r['release_'+plan+'_matched'] for r in rows)
                tracks.append(result)
        for plan in PLANS:
            for index, seg in enumerate(old.segments(raw, lambda r,p=plan: not r['release_'+p+'_matched'])):
                result = dict(reference='release_' + plan, sequence=seq, track_id=track, release_miss_segment=index,
                    frame_start=seg[0]['frame'], frame_end=seg[-1]['frame'], source_frame_start=seg[0]['source_frame'],
                    source_frame_end=seg[-1]['source_frame'], target_frames=len(seg), low_all_05_frames=sum(r['low_all_05'] for r in seg))
                for condition in conditions:
                    remaining = old.segments(seg, lambda r,c=condition: not r[c+'_matched'])
                    result[condition+'_recovered_frames'] = sum(r[condition+'_matched'] for r in seg)
                    result[condition+'_remaining_miss_longest'] = max(map(len,remaining), default=0)
                    result[condition+'_remaining_miss_runs'] = len(remaining)
                release_runs.append(result)
    summaries = []
    for name in names:
        for group in ['pooled'] + SEQS:
            for cohort in COHORTS:
                ts = [r for r in tracks if r['condition'] == name and r['cohort'] == cohort and (group == 'pooled' or r['sequence'] == group)]
                if not ts:
                    continue
                lengths = [r['longest_miss_run'] for r in ts]
                result = dict(condition=name, group=group, cohort=cohort, target_frames=sum(r['target_frames'] for r in ts),
                    tracks=len(ts), clips=len({r['sequence'] for r in ts}), missed_frames=sum(r['missed_frames'] for r in ts),
                    miss_run_count=sum(r['miss_run_count'] for r in ts), longest_miss_track_mean=float(np.mean(lengths)),
                    longest_miss_track_median=float(np.median(lengths)), longest_miss_track_q90=float(np.quantile(lengths,.9)),
                    longest_miss_max=max(lengths), track_equal_miss_fraction=float(np.mean([r['missed_frames']/r['target_frames'] for r in ts])))
                for plan in PLANS:
                    ds = [r['longest_miss_delta_vs_'+plan] for r in ts]
                    result.update({'tracks_shorter_vs_'+plan: sum(d<0 for d in ds), 'tracks_same_vs_'+plan: sum(d==0 for d in ds),
                                   'tracks_longer_vs_'+plan: sum(d>0 for d in ds)})
                summaries.append(result)
    for name in names:
        full = [r for r in tracks if r['condition'] == name and r['cohort'] == 'all']
        assert len(full) == 130 and sum(r['target_frames'] for r in full) == 5200
    csvwrite(output / 'TRACK_SUMMARY.csv', tracks)
    csvwrite(output / 'CONTINUOUS_MISS_SUMMARY.csv', summaries)
    csvwrite(output / 'RELEASE_MISS_SEGMENTS.csv', release_runs)
    jsonwrite(output / 'CONTINUITY_SCHEMA.json', dict(reused_source=str(old_path), reused_source_sha256=digest(old_path),
        rule='Both local and source frame must advance by one; missing labels, selection gaps and cohort exclusions break segments.',
        observation='Frozen strict-small observations, not complete physical trajectory lengths.',
        no_reference_duplication='One TEP track record per cohort, with distinct delta-vs-O and delta-vs-S columns.'))


def preflight():
    """Rescore the existing release O output as a single prediction; no TEP forward."""
    output = BASE / 'evaluation_preflight'
    condition = 'preflight_release_O_single_prediction'
    checks = []
    for sequence in SEQS:
        ref, ar, receipt, _ = load_reference(sequence)
        path = P / 'predictions/official' / sequence / 'official_fused.npz'
        assert digest(path) == receipt['source_sha256'][str(path)]
        with np.load(path, allow_pickle=False) as archive:
            probability = archive['probabilities']
        computed = compute_scores(probability, sequence, condition)
        assert computed[3] == receipt['objects']
        for key in ['matched', 'box_max', 'peak_x', 'peak_y', 'component_area', 'frame_counts']:
            assert np.array_equal(computed[0][key], ref[key][0], equal_nan=True), (sequence,key)
            checks.append(sequence + ': exact release O ' + key)
        for key in ['areas', 'unmatched', 'offsets']:
            assert np.array_equal(computed[1][key], ar['O_'+key]), (sequence,key)
            checks.append(sequence + ': exact release O component ' + key)
        save_scores(output / condition / sequence, computed, condition, sequence,
                    dict(purpose='Scorer identity preflight only; this is not a TEP-SAM prediction',
                         source_sha256={str(path): digest(path)}))
    summarize([condition], output)
    with (output / 'SUMMARY.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert int(rows[-1]['M']) == 700 and int(rows[-1]['unmatched_components']) == 4015
    assert int(rows[-1]['G_vs_O']) == 0 and int(rows[-1]['L_vs_O']) == 0
    tracks = list(csv.DictReader((output / 'TRACK_SUMMARY.csv').open()))
    assert all(int(r['longest_miss_delta_vs_O']) == 0 for r in tracks if r['condition'] == condition)
    checks += ['single prediction summary matches release O exactly', 'all miss-run deltas versus O exactly zero']
    jsonwrite(output / 'PREFLIGHT_COMPLETE.json', dict(status='COMPLETE', checks=checks,
               check_count=len(checks), detector_forward_calls=0, new_training_updates=0,
               script_sha256=digest(__file__)))


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--preflight', action='store_true')
    group.add_argument('--score', choices=sorted(HEADS))
    group.add_argument('--summary', action='store_true')
    parser.add_argument('--sequence', choices=SEQS)
    parser.add_argument('--conditions', nargs='+', choices=sorted(HEADS), default=['tep_sam'])
    parser.add_argument('--predictions-root', type=Path, default=BASE / 'predictions')
    args = parser.parse_args()
    if args.preflight:
        preflight()
    elif args.summary:
        summarize(args.conditions)
    else:
        for sequence in ([args.sequence] if args.sequence else SEQS):
            score(sequence, args.score, args.predictions_root)


if __name__ == '__main__':
    main()
