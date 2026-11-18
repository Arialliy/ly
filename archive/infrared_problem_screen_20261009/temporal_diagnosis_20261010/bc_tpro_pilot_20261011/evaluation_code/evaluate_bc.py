#!/usr/bin/env python3
"""BC-TPro pilot: reuse frozen point scorer; compare new weights with cached controls.

No model loading, training, threshold search, or candidate-window selection occurs here.
"""
import argparse
import collections
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
Q = BASE.parent
OLD = Q / 'top1_background_control_20261011'
EVAL = BASE / 'evaluation'
PLANS = {'O': [0, 36, 72, 108, 120], 'S': [0, 24, 60, 96, 120]}
SEQS = ['DJI_0026_2', 'DJI_0067_2']
REFERENCES = ['release', 'lambda0_top1']
NEW = ['single_dynamic', 'bc_tpro']
CONDITIONS = REFERENCES + NEW
FIELDS = ['gt_count', 'matched_count', 'component_count', 'unmatched_components',
          'positive_pixels', 'positive_pixels_outside_boxes']
COHORT_PATH = Q / 'low_response_information_20261011/COHORTS.csv'
COHORT_KEYS = ['low_all_05', 'extreme_all_001', 'both_fixed_success', 'extra_only',
               'new_window_dependent', 'high_unmatched']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def write_csv(path, rows):
    assert rows, path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score(condition, sequence, predictions_root=None):
    """Only redirect the old scorer's new-condition input/output directories."""
    assert condition in NEW and sequence in SEQS
    predictions_root = predictions_root or BASE / 'predictions' / condition
    inference_receipt_path = predictions_root / sequence / 'COMPLETE.json'
    inference_receipt = json.loads(inference_receipt_path.read_text())
    run_path = BASE / 'training' / condition / 'run/RUN.json'
    run = json.loads(run_path.read_text())
    checkpoint_path = BASE / 'training' / condition / 'run/final_model.pth'
    assert run['status'] == 'complete' and run['variant'] == condition
    assert run['final_step'] == run['steps'] == 512
    assert inference_receipt['status'] == 'COMPLETE'
    assert inference_receipt['variant'] == condition and inference_receipt['sequence'] == sequence
    assert Path(inference_receipt['checkpoint']).resolve() == checkpoint_path.resolve()
    assert digest(checkpoint_path) == run['final_checkpoint_sha256'] == inference_receipt['checkpoint_sha256']
    assert inference_receipt['reference_checkpoint_sha256'] == run['checkpoint_sha256']
    assert inference_receipt['model_config'] == run['model_config']
    assert inference_receipt['model_source_sha256'] == run['model_source_sha256']
    assert inference_receipt['bc_source_sha256'] == run['bc_tpro_source_sha256']
    assert inference_receipt['official_starts'] == PLANS['O']
    assert inference_receipt['diagnostic_starts'] == sorted(set(sum(PLANS.values(), [])))
    assert inference_receipt['normalization'] == [85.0799, 47.4845]
    assert inference_receipt['tile_core'] == 128 and inference_receipt['tile_halo'] >= 49
    assert inference_receipt['model_labels_access'] is False
    assert inference_receipt['tiling_checks'], 'Trained-model tiling witness missing'
    old = load_module(OLD / 'evaluation_code/evaluate.py', 'frozen_top1_scorer')
    old.E = EVAL
    old.ROOTS[condition] = predictions_root
    old.score(condition, sequence)
    completion = EVAL / condition / sequence / 'COMPLETE.json'
    receipt = json.loads(completion.read_text())
    receipt['wrapper_sha256'] = digest(__file__)
    receipt['wrapper_path'] = str(Path(__file__).resolve())
    receipt['reused_scorer_path'] = str(OLD / 'evaluation_code/evaluate.py')
    receipt['reused_scorer_sha256'] = digest(OLD / 'evaluation_code/evaluate.py')
    receipt['training_identity_checked'] = True
    receipt['training_receipt_sha256'] = digest(run_path)
    receipt['checkpoint_sha256'] = digest(checkpoint_path)
    write_json(completion, receipt)


def read_record(condition, sequence):
    root = (OLD / 'evaluation') if condition in REFERENCES else EVAL
    folder = root / condition / sequence
    receipt = json.loads((folder / 'COMPLETE.json').read_text())
    assert receipt['status'] == 'COMPLETE'
    assert receipt['condition'] == condition and receipt['sequence'] == sequence
    assert receipt['threshold'] == .5
    with np.load(folder / 'scores.npz', allow_pickle=False) as archive:
        scores = {key: archive[key] for key in archive.files}
    with np.load(folder / 'component_areas.npz', allow_pickle=False) as archive:
        areas = {key: archive[key] for key in archive.files}
    assert scores['plans'].tolist() == list(PLANS)
    assert float(scores['threshold']) == .5
    assert scores['matched'].shape == (2, len(receipt['objects']))
    assert scores['frame_counts'].shape == (2, 160, len(FIELDS))
    assert np.all(scores['coverage'] > 0)
    assert np.all(scores['coverage'].sum(axis=1) == 200)
    hashes = {str(folder / name): digest(folder / name)
              for name in ['COMPLETE.json', 'scores.npz', 'component_areas.npz', 'FRAME_COUNTS.csv']}
    return scores, areas, receipt, hashes


def frozen_cohorts():
    with COHORT_PATH.open() as stream:
        rows = list(csv.DictReader(stream))
    result = {(r['sequence'], int(r['frame']), int(r['annotation_id'])): r for r in rows}
    assert len(result) == len(rows) == 5200
    for key, count in [('low_all_05', 4342), ('extreme_all_001', 4170),
                       ('both_fixed_success', 646), ('extra_only', 104),
                       ('new_window_dependent', 73)]:
        assert sum(int(r[key]) for r in rows) == count, key
    assert sum(0 < int(r['success_window_count']) < int(r['covering_window_count'])
               for r in rows) == 286
    return result


def collect(conditions):
    comparisons = REFERENCES + (['single_dynamic'] if 'single_dynamic' in conditions else [])
    records, areas, metadata, hashes = {}, {}, {}, {str(COHORT_PATH): digest(COHORT_PATH)}
    frames = []
    for condition in conditions:
        for seq in SEQS:
            arrays, component_areas, receipt, inputs = read_record(condition, seq)
            records[condition, seq], areas[condition, seq] = arrays, component_areas
            hashes.update(inputs)
            if seq in metadata:
                assert receipt['objects'] == metadata[seq], (condition, seq, 'GT ordering')
            else:
                metadata[seq] = receipt['objects']
            root = OLD / 'evaluation' if condition in REFERENCES else EVAL
            with (root / condition / seq / 'FRAME_COUNTS.csv').open() as stream:
                frames.extend(csv.DictReader(stream))
    cohorts = frozen_cohorts()
    objects = []
    for seq in SEQS:
        for idx, meta in enumerate(metadata[seq]):
            if not meta['strict_small']:
                continue
            frozen = cohorts[seq, meta['frame'], meta['annotation_id']]
            assert int(frozen['track_id']) == meta['track_id']
            for pi, plan in enumerate(PLANS):
                row = dict(sequence=seq, track_id=meta['track_id'], frame=meta['frame'],
                           source_frame=meta['source_frame'], annotation_id=meta['annotation_id'],
                           arrangement=plan, threshold=.5, cohort=frozen['cohort'],
                           original_system_case=frozen['original_system_case'])
                row.update({key: int(frozen[key]) for key in COHORT_KEYS})
                row.update(release_O_only=int(int(frozen['original_matched']) and not int(frozen['symmetric_matched'])),
                           release_S_only=int(int(frozen['symmetric_matched']) and not int(frozen['original_matched'])),
                           candidate_window_mixed=int(0 < int(frozen['success_window_count']) < int(frozen['covering_window_count'])),
                           original_covering_windows=int(frozen['covering_window_count']),
                           original_success_windows=int(frozen['success_window_count']))
                for condition in conditions:
                    arr = records[condition, seq]
                    assert bool(arr['strict_small'][idx])
                    for key in ['matched', 'box_max', 'peak_x', 'peak_y', 'component_area']:
                        value = arr[key][pi, idx].item()
                        row[condition + '_' + key] = value if np.isfinite(value) else ''
                assert row['release_matched'] == int(frozen['original_matched' if plan == 'O' else 'symmetric_matched'])
                for condition in conditions:
                    for ref in comparisons:
                        row[condition + '_gain_vs_' + ref] = int(row[condition + '_matched'] and not row[ref + '_matched'])
                        row[condition + '_loss_vs_' + ref] = int(not row[condition + '_matched'] and row[ref + '_matched'])
                objects.append(row)
    assert len(objects) == 10400
    return records, areas, metadata, hashes, frames, objects


def summarize(conditions=CONDITIONS, output=EVAL):
    records, areas, metadata, hashes, frames, objects = collect(conditions)
    comparisons = REFERENCES + (['single_dynamic'] if 'single_dynamic' in conditions else [])
    summaries, tracks, concentration, window_rows = [], [], [], []
    for condition in conditions:
        for pi, plan in enumerate(PLANS):
            for group, seqs in [('pooled', SEQS)] + [(s, [s]) for s in SEQS]:
                obj = [r for r in objects if r['sequence'] in seqs and r['arrangement'] == plan]
                counts = sum((records[condition, s]['frame_counts'][pi].sum(0) for s in seqs), np.zeros(6, np.int64))
                ref_counts = sum((records['release', s]['frame_counts'][pi].sum(0) for s in seqs), np.zeros(6, np.int64))
                top_counts = sum((records['lambda0_top1', s]['frame_counts'][pi].sum(0) for s in seqs), np.zeros(6, np.int64))
                after = np.array([r[condition + '_matched'] for r in obj], bool)
                area = np.concatenate([areas[condition, s][plan + '_areas'] for s in seqs])
                unmatched = np.concatenate([areas[condition, s][plan + '_unmatched'].astype(bool) for s in seqs])
                row = dict(condition=condition, arrangement=plan, group=group, threshold=.5,
                           small_gt=len(obj), small_tracks=len({(r['sequence'], r['track_id']) for r in obj}),
                           clips=len(seqs), M=int(after.sum()), **dict(zip(FIELDS, map(int, counts))))
                for ref in comparisons:
                    before = np.array([r[ref + '_matched'] for r in obj], bool)
                    gain, loss = after & ~before, ~after & before
                    row.update({ref + '_M_same': int(before.sum()),
                                'G_vs_' + ref + '_same': int(gain.sum()),
                                'L_vs_' + ref + '_same': int(loss.sum()),
                                'net_vs_' + ref + '_same': int(gain.sum() - loss.sum())})
                    assert row['M'] == int(before.sum() + gain.sum() - loss.sum())
                original_o = np.concatenate([records['release', s]['matched'][0, records['release', s]['strict_small']].astype(bool) for s in seqs])
                assert len(original_o) == len(after)
                row.update(release_M_O=int(original_o.sum()), G_vs_release_O=int((after & ~original_o).sum()),
                           L_vs_release_O=int((~after & original_o).sum()))
                row.update(release_U_same=int(ref_counts[3]), delta_U_vs_release_same=int(counts[3]-ref_counts[3]),
                           release_A_same=int(ref_counts[5]), delta_A_vs_release_same=int(counts[5]-ref_counts[5]),
                           top1_U_same=int(top_counts[3]), delta_U_vs_top1_same=int(counts[3]-top_counts[3]),
                           top1_A_same=int(top_counts[5]), delta_A_vs_top1_same=int(counts[5]-top_counts[5]),
                           U_budget=4015 if group == 'pooled' else '',
                           passes_pooled_U_budget=int(counts[3] <= 4015) if group == 'pooled' else '')
                for cohort in COHORT_KEYS + ['release_O_only', 'release_S_only', 'candidate_window_mixed']:
                    selected = np.array([r[cohort] for r in obj], bool)
                    restored = selected & after
                    row[cohort + '_gt'] = int(selected.sum())
                    row[cohort + '_matched'] = int(restored.sum())
                    row[cohort + '_matched_tracks'] = len({(r['sequence'], r['track_id']) for r, yes in zip(obj, restored) if yes})
                    row[cohort + '_matched_clips'] = len({r['sequence'] for r, yes in zip(obj, restored) if yes})
                for name, aa in [('component', area), ('unmatched_component', area[unmatched])]:
                    row[name + '_area_sum'] = int(aa.sum())
                    row[name + '_area_max'] = int(aa.max()) if len(aa) else 0
                    row[name + '_area_mean'] = float(aa.mean()) if len(aa) else 0.
                    for label, q in [('q50', .5), ('q95', .95), ('q99', .99)]:
                        row[name + '_area_' + label] = float(np.quantile(aa, q)) if len(aa) else 0.
                release_area = sum(int(areas['release', s][plan+'_areas'][areas['release', s][plan+'_unmatched'].astype(bool)].sum()) for s in seqs)
                row['release_unmatched_component_area_sum'] = release_area
                row['delta_unmatched_component_area_vs_release'] = row['unmatched_component_area_sum'] - release_area
                row['recorded_background_costs_not_increased'] = int(counts[3] <= ref_counts[3] and counts[5] <= ref_counts[5] and row['unmatched_component_area_sum'] <= release_area)
                row['beats_release_S_M_within_old_U_budget'] = int(row['M'] > 720 and counts[3] <= 4015) if group == 'pooled' else ''
                summaries.append(row)
                if group == 'pooled':
                    outcomes = {'gain_vs_release': after & ~np.array([r['release_matched'] for r in obj], bool),
                                'loss_vs_release': ~after & np.array([r['release_matched'] for r in obj], bool),
                                'low_recovered': after & np.array([r['low_all_05'] for r in obj], bool),
                                'extreme_recovered': after & np.array([r['extreme_all_001'] for r in obj], bool)}
                    for label, mask in outcomes.items():
                        counter = collections.Counter((r['sequence'], r['track_id']) for r, yes in zip(obj, mask) if yes)
                        ranked = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
                        concentration.append(dict(condition=condition, arrangement=plan, outcome=label,
                                                  frames=int(mask.sum()), tracks=len(counter), clips=len({key[0] for key in counter}),
                                                  top1_frames=ranked[0][1] if ranked else 0,
                                                  top5_frames=sum(n for _, n in ranked[:5]),
                                                  ranked_tracks=json.dumps([dict(sequence=k[0], track_id=k[1], frames=v) for k, v in ranked])))
            for seq, tid in sorted({(r['sequence'], r['track_id']) for r in objects}):
                obj = [r for r in objects if r['arrangement'] == plan and r['sequence'] == seq and r['track_id'] == tid]
                row = dict(condition=condition, arrangement=plan, sequence=seq, track_id=tid,
                           target_frames=len(obj), M=sum(r[condition+'_matched'] for r in obj))
                for ref in comparisons:
                    row['G_vs_'+ref+'_same'] = sum(r[condition+'_matched'] and not r[ref+'_matched'] for r in obj)
                    row['L_vs_'+ref+'_same'] = sum(not r[condition+'_matched'] and r[ref+'_matched'] for r in obj)
                for cohort in COHORT_KEYS + ['release_O_only', 'release_S_only', 'candidate_window_mixed']:
                    row[cohort+'_gt'] = sum(r[cohort] for r in obj)
                    row[cohort+'_matched'] = sum(r[cohort] and r[condition+'_matched'] for r in obj)
                tracks.append(row)
        # Reliability under the two frozen arrangements, not arbitrary or all 11 windows.
        pairs = {(r['sequence'], r['frame'], r['annotation_id']): r for r in objects if r['arrangement'] == 'O'}
        second = {(r['sequence'], r['frame'], r['annotation_id']): r for r in objects if r['arrangement'] == 'S'}
        for group, seqs in [('pooled', SEQS)] + [(s, [s]) for s in SEQS]:
            for cohort in ['all', 'both_fixed_success', 'release_O_only', 'release_S_only', 'new_window_dependent', 'extra_only', 'candidate_window_mixed', 'low_all_05', 'extreme_all_001']:
                selected = [key for key, row in pairs.items() if row['sequence'] in seqs and (cohort == 'all' or row[cohort])]
                o = np.array([pairs[key][condition+'_matched'] for key in selected], bool)
                s = np.array([second[key][condition+'_matched'] for key in selected], bool)
                window_rows.append(dict(condition=condition, group=group, cohort=cohort, target_frames=len(selected),
                                        tracks=len({(key[0], pairs[key]['track_id']) for key in selected}),
                                        clips=len({key[0] for key in selected}), O_matched=int(o.sum()), S_matched=int(s.sum()),
                                        both_matched=int((o&s).sum()), O_only=int((o&~s).sum()), S_only=int((s&~o).sum()),
                                        neither_matched=int((~o&~s).sum()), arrangement_disagreement=int((o^s).sum())))
    expected = {'release': [(700, 4015), (720, 3967)], 'lambda0_top1': [(877, 8339), (913, 8330)]}
    for condition, values in expected.items():
        assert [(r['M'], r['unmatched_components']) for r in summaries if r['group'] == 'pooled' and r['condition'] == condition] == values
    write_csv(output/'SUMMARY.csv', [r for r in summaries if r['group'] == 'pooled'])
    write_csv(output/'SEQUENCE_SUMMARY.csv', [r for r in summaries if r['group'] != 'pooled'])
    write_csv(output/'OBJECT_COMPARISON.csv', objects)
    write_csv(output/'FRAME_COUNTS.csv', frames)
    write_csv(output/'TRACK_SUMMARY.csv', tracks)
    write_csv(output/'RECOVERY_CONCENTRATION.csv', concentration)
    write_csv(output/'WINDOW_DEPENDENCE.csv', window_rows)
    write_json(output/'SUMMARY_COMPLETE.json', dict(status='COMPLETE', conditions=conditions, plans=PLANS, threshold=.5,
               new_models_evaluated=[c for c in conditions if c in NEW], small_target_frames=5200,
               object_comparison_rows=10400, source_sha256=hashes, script_sha256=digest(__file__),
               scope='Frozen component-peak diagnostic, not official Pd/Fa. O/S reliability only; cohorts frozen from release 11-window audit.',
               cohort_counts=dict(low_all_05=4342, extreme_all_001=4170, both_fixed_success=646, extra_only=104,
                                  new_window_dependent=73, release_O_only=54, release_S_only=74, candidate_window_mixed=286)))
    print(json.dumps({'status': 'COMPLETE', 'output': str(output), 'rows': [
        {k: r[k] for k in ['condition', 'arrangement', 'M', 'unmatched_components', 'positive_pixels_outside_boxes', 'low_all_05_matched']}
        for r in summaries if r['group'] == 'pooled']}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', choices=NEW)
    parser.add_argument('--sequence', choices=SEQS)
    parser.add_argument('--predictions-root', type=Path)
    parser.add_argument('--summary', action='store_true')
    parser.add_argument('--baseline-check', action='store_true')
    args = parser.parse_args()
    if args.baseline_check:
        assert not args.summary and args.condition is None
        summarize(REFERENCES, BASE/'evaluation_preflight')
    elif args.summary:
        assert args.condition is None
        if (EVAL/'SUMMARY_COMPLETE.json').exists():
            raise FileExistsError('Completed summary already exists')
        summarize()
    else:
        assert args.condition is not None
        for seq in [args.sequence] if args.sequence else SEQS:
            score(args.condition, seq, args.predictions_root)


if __name__ == '__main__':
    main()
