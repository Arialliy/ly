#!/usr/bin/env python3
"""Local-reference evidence pilot: unchanged point scoring and frozen release controls.

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
REFERENCES = ['release']
NEW = ['N0', 'N1', 'N2']
COMPARISONS = ['release', 'N0', 'N1', 'N2']
CONDITIONS = REFERENCES + NEW
FIELDS = ['gt_count', 'matched_count', 'component_count', 'unmatched_components',
          'positive_pixels', 'positive_pixels_outside_boxes']
COHORT_PATH = Q / 'low_response_information_20261011/COHORTS.csv'
REFERENCE_ANCHOR = Q / 'bc_tpro_pilot_20261011/evaluation_preflight/SUMMARY_COMPLETE.json'
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


def record_root(condition):
    return OLD/'evaluation' if condition in REFERENCES else EVAL


def score(condition, sequence, predictions_root=None):
    from verify_identity import verify_identity
    assert condition in NEW and sequence in SEQS
    predictions_root = predictions_root or BASE/'predictions'/condition
    provenance = verify_identity(BASE, condition, sequence, predictions_root)
    reference = json.loads((OLD/'evaluation/lambda0_top1'/sequence/'COMPLETE.json').read_text())
    assert reference['script_sha256'] == digest(OLD/'evaluation_code/evaluate.py')
    for path, expected in reference['source_sha256'].items():
        if Path(path).suffix == '.py' or Path(path).name == 'annotations.json':
            assert digest(path) == expected, ('Frozen scorer changed',path)
    old = load_module(OLD/'evaluation_code/evaluate.py','frozen_top1_scorer')
    old.E = EVAL
    old.ROOTS[condition] = predictions_root
    old.score(condition, sequence)
    path = EVAL/condition/sequence/'COMPLETE.json'
    receipt = json.loads(path.read_text())
    receipt.update(wrapper_sha256=digest(__file__), wrapper_path=str(Path(__file__).resolve()),
                   reused_scorer_path=str(OLD/'evaluation_code/evaluate.py'),
                   reused_scorer_sha256=digest(OLD/'evaluation_code/evaluate.py'),
                   training_identity_checked=True, **provenance)
    write_json(path,receipt)


def read_record(condition, sequence):
    root = record_root(condition)
    folder = root / condition / sequence
    receipt = json.loads((folder / 'COMPLETE.json').read_text())
    assert receipt['status'] == 'COMPLETE'
    assert receipt['condition'] == condition and receipt['sequence'] == sequence
    assert receipt['threshold'] == .5
    if condition in NEW:
        assert receipt['training_identity_checked'] is True
        assert receipt['wrapper_sha256'] == digest(__file__)
        for path, expected in receipt['identity_source_sha256'].items():
            assert digest(path) == expected, ('Identity receipt changed',path)
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
    # Reuse the already independently recomputed release caches and cohorts.
    anchor = json.loads(REFERENCE_ANCHOR.read_text())
    assert anchor['status'] == 'COMPLETE'
    for path, expected in anchor['source_sha256'].items():
        assert digest(path) == expected, ('Frozen reference changed', path)
    comparisons = [c for c in COMPARISONS if c in conditions]
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
            root = record_root(condition)
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


def cohort_transitions(conditions, objects):
    rows = []
    keys = ['all'] + COHORT_KEYS + ['release_O_only','release_S_only','candidate_window_mixed']
    for reference in [c for c in COMPARISONS if c in conditions]:
        for condition in conditions:
            for plan in PLANS:
                for group, seqs in [('pooled',SEQS)] + [(s,[s]) for s in SEQS]:
                    obj = [r for r in objects if r['sequence'] in seqs and r['arrangement']==plan]
                    for cohort in keys:
                        selected = [r for r in obj if cohort=='all' or r[cohort]]
                        before = np.array([r[reference+'_matched'] for r in selected],bool)
                        after = np.array([r[condition+'_matched'] for r in selected],bool)
                        masks = {'both_matched':after & before,'gain':after & ~before,
                                 'loss':~after & before,'neither_matched':~after & ~before}
                        row = dict(condition=condition,reference=reference,arrangement=plan,
                                   group=group,cohort=cohort,target_frames=len(selected),
                                   target_tracks=len({(r['sequence'],r['track_id']) for r in selected}),
                                   target_clips=len({r['sequence'] for r in selected}),
                                   reference_matched=int(before.sum()),matched=int(after.sum()),
                                   net=int(after.sum()-before.sum()))
                        for label,mask in masks.items():
                            row[label]=int(mask.sum())
                            row[label+'_tracks']=len({(r['sequence'],r['track_id']) for r,y in zip(selected,mask) if y})
                            row[label+'_clips']=len({r['sequence'] for r,y in zip(selected,mask) if y})
                        assert sum(row[k] for k in masks)==len(selected)
                        assert row['matched']==row['reference_matched']+row['gain']-row['loss']
                        rows.append(row)
    return rows


def summarize(conditions=CONDITIONS, output=EVAL):
    records, areas, metadata, hashes, frames, objects = collect(conditions)
    new_conditions = [c for c in conditions if c in NEW]
    if new_conditions:
        runs = {c: json.loads((BASE/'training'/c/'run/RUN.json').read_text()) for c in new_conditions}
        shared_keys = ['seed', 'steps', 'final_step', 'lr', 'weight_decay', 'optimizer', 'betas', 'eps',
                       'precision', 'bn_policy', 'branch_initial_sha256', 'initial_model_state_sha256',
                       'original_initial_state_sha256', 'original_final_state_sha256',
                       'plan_sha256', 'selection_sha256', 'cumulative_input_sha256',
                       'cumulative_label_sha256', 'training_sequences', 'parameter_count',
                       'new_parameters', 'trainable_parameters', 'label_config',
                       'effective_supervision_config', 'loss_reduction', 'checkpoint_selection']
        reference_run = runs[new_conditions[0]]
        for condition, run in runs.items():
            for key in shared_keys:
                assert run[key] == reference_run[key], ('Unequal controlled training', condition, key)
            config = json.loads(json.dumps(run['model_config']))
            reference_config = json.loads(json.dumps(reference_run['model_config']))
            config['evidence'].pop('mode')
            reference_config['evidence'].pop('mode')
            assert config == reference_config, 'Model differs beyond the preregistered input mode'
        write_json(output/'TRAINING_COMPARABILITY.json', dict(status='PASS', conditions=new_conditions,
                   shared_fields=shared_keys, shared_values={k: reference_run[k] for k in shared_keys},
                   frozen_original_tensors_checked=True, seed=46, steps=512,
                   source_sha256={str(BASE/'training'/c/'run/RUN.json'): digest(BASE/'training'/c/'run/RUN.json')
                                  for c in new_conditions},
                   scope='Common architecture, initialization and exact data/label exposure; input information differs. Effective capacity is not claimed identical.'))
    transitions = cohort_transitions(conditions, objects)
    comparisons = [c for c in COMPARISONS if c in conditions]
    summaries, tracks, concentration, window_rows = [], [], [], []
    for condition in conditions:
        for pi, plan in enumerate(PLANS):
            for group, seqs in [('pooled', SEQS)] + [(s, [s]) for s in SEQS]:
                obj = [r for r in objects if r['sequence'] in seqs and r['arrangement'] == plan]
                counts = sum((records[condition, s]['frame_counts'][pi].sum(0) for s in seqs), np.zeros(6, np.int64))
                ref_counts = sum((records['release', s]['frame_counts'][pi].sum(0) for s in seqs), np.zeros(6, np.int64))
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
    expected = {'release': [(700, 4015), (720, 3967)]}
    for condition, values in expected.items():
        assert [(r['M'], r['unmatched_components']) for r in summaries if r['group'] == 'pooled' and r['condition'] == condition] == values
    write_csv(output/'SUMMARY.csv', [r for r in summaries if r['group'] == 'pooled'])
    write_csv(output/'SEQUENCE_SUMMARY.csv', [r for r in summaries if r['group'] != 'pooled'])
    write_csv(output/'OBJECT_COMPARISON.csv', objects)
    write_csv(output/'FRAME_COUNTS.csv', frames)
    write_csv(output/'TRACK_SUMMARY.csv', tracks)
    write_csv(output/'RECOVERY_CONCENTRATION.csv', concentration)
    write_csv(output/'WINDOW_DEPENDENCE.csv', window_rows)
    write_csv(output/'COHORT_TRANSITIONS.csv', transitions)
    from trajectory_tables import trajectory_tables
    trajectory_tables(objects, conditions, comparisons, output, Q)
    write_json(output/'SUMMARY_COMPLETE.json', dict(status='COMPLETE', conditions=conditions, plans=PLANS, threshold=.5,
               new_models_evaluated=[c for c in conditions if c in NEW], small_target_frames=5200,
               supervision_scope='Common frozen DeepPro-Plus backbone; new-branch N0 current, N1 ordinary differences, N2 local-reference differences. Registered 512-step endpoint; not author-training reproduction.',
               object_comparison_rows=10400, source_sha256=hashes, script_sha256=digest(__file__),
               frozen_reference_anchor=str(REFERENCE_ANCHOR), frozen_reference_anchor_sha256=digest(REFERENCE_ANCHOR),
               frozen_reference_hashes_unchanged=True,
               scope='Frozen component-peak diagnostic, not official Pd/Fa. O/S reliability only; cohorts frozen from release 11-window audit.',
               comparisons=comparisons, cohort_counts=dict(low_all_05=4342, extreme_all_001=4170, both_fixed_success=646, extra_only=104,
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
