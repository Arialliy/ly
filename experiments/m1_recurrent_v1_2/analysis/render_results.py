"""Render only complete M1-v1.2 evidence; no imports of model/runtime or workloads.

The decision is computed from measured paired recovery, never prefilled. This
script writes a reviewable report, not authorization for any further experiment.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

E = Path(__file__).resolve().parents[1]
V = E / 'evaluation/step_2176'
T = E / 'analysis/trace_summary'
ARMS = ('V12Current', 'V12History')
C, H, HC = 'V12CurrentCurrent', 'V12HistoryHistory', 'V12HistoryCurrent'
CONDITIONS = ('release', 'History', C, H, HC)
PAIRS = ((H, C), (H, HC), (H, 'History'), (H, 'release'), (C, 'History'))
COHORTS = ('strict_all', 'low_all_05', 'extreme_all_001', 'both_fixed_success', 'm1_low_common_recovered')
LABEL = {'release': 'release', 'History': '旧History2048', C: 'V12-Current',
         H: 'V12-History', HC: '同H权重current_only'}
SOURCES = {}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def remember(path):
    path = Path(path)
    digest = sha(path)
    if str(path) in SOURCES:
        assert SOURCES[str(path)] == digest, ('Source changed during rendering', str(path))
    SOURCES[str(path)] = digest
    return digest


def read(path):
    remember(path)
    return json.loads(Path(path).read_text(encoding='utf-8'))


def number(value):
    if value is None or value == '': return None
    result = float(value)
    assert math.isfinite(result)
    return int(result) if result.is_integer() else result


def rows(path):
    remember(path)
    with Path(path).open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def compact(row, fields):
    return {key: number(row[key]) if key not in ('condition', 'reference', 'arrangement',
        'sequence', 'track_id', 'annotation_id', 'scope', 'cohort', 'outcome', 'comparison') else row[key]
        for key in fields}


def verify_outputs(base, receipt):
    assert receipt['status'] == 'COMPLETE'
    for name, expected in receipt['output_sha256'].items():
        assert remember(base / name) == expected, str(base / name)


def key(row):
    return row['sequence'], int(row['frame']), row['annotation_id']


def table(lines, headers, values):
    lines += ['', '| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---']*len(headers)) + ' |']
    for values_row in values:
        lines.append('| ' + ' | '.join(str(v).replace('|', '/') for v in values_row) + ' |')


def fmt(value):
    return '—' if value is None else f'{value:.6g}' if isinstance(value, float) else str(value)


def matching_comparisons(summary, allgt, strict):
    si = {(r['condition'], r['arrangement']): r for r in summary}
    ai = {(r['condition'], r['arrangement']): r for r in allgt if r['scope'] == 'all_valid_gt'}
    comparisons, changed = [], []
    for condition, reference in PAIRS:
        for plan in ('O', 'S'):
            after, before = si[condition, plan], si[reference, plan]
            whole = ai[condition, plan]
            comparisons.append({'condition': condition, 'reference': reference, 'arrangement': plan,
                'strict_G': int(after[f'G_vs_{reference}_same']), 'strict_L': int(after[f'L_vs_{reference}_same']),
                'all_valid_G': int(whole[f'G_vs_{reference}']), 'all_valid_L': int(whole[f'L_vs_{reference}']),
                'delta_U': int(after['unmatched_components'])-int(before['unmatched_components']),
                'delta_outside_box_positive_pixels': int(after['positive_pixels_outside_boxes'])-int(before['positive_pixels_outside_boxes']),
                'delta_unmatched_area': int(after['unmatched_component_area_sum'])-int(before['unmatched_component_area_sum'])})
        for row in strict:
            old, new = int(row[reference+'_matched']), int(row[condition+'_matched'])
            if old == new: continue
            changed.append({name: row[name] for name in ('sequence', 'frame', 'annotation_id', 'track_id', 'arrangement')} |
                {'condition': condition, 'reference': reference, 'reference_matched': old, 'condition_matched': new,
                 'low': int(row['low_all_05']), 'extreme': int(row['extreme_all_001']),
                 'protected': int(row['both_fixed_success']),
                 'reference_box_max': float(row[reference+'_box_max']), 'condition_box_max': float(row[condition+'_box_max'])})
    return comparisons, changed


def verify_history_effects(effects, strict):
    for row in effects:
        selected = [r for r in strict if int(r[row['cohort']])]
        gain, loss = {}, {}
        for plan in ('O', 'S'):
            chosen = [r for r in selected if r['arrangement'] == plan]
            gain[plan] = {key(r) for r in chosen if int(r[H+'_matched']) and not int(r[row['reference']+'_matched'])}
            loss[plan] = {key(r) for r in chosen if not int(r[H+'_matched']) and int(r[row['reference']+'_matched'])}
        expected = {'R_hist_O': len(gain['O']), 'R_hist_S': len(gain['S']),
            'R_hist_unique_either_arrangement': len(gain['O']|gain['S']),
            'R_hist_both_arrangements': len(gain['O']&gain['S']),
            'L_hist_O': len(loss['O']), 'L_hist_S': len(loss['S']),
            'L_hist_unique_either_arrangement': len(loss['O']|loss['S']),
            'L_hist_both_arrangements': len(loss['O']&loss['S']),
            'objects_gain_in_one_plan_loss_in_other': len((gain['O']|gain['S'])&(loss['O']|loss['S']))}
        assert all(int(row[k]) == value for k, value in expected.items())
    return [{**{k: r[k] for k in ('comparison', 'condition', 'reference', 'cohort')},
             **{k: int(v) for k, v in r.items() if k.startswith(('R_hist_', 'L_hist_', 'objects_'))}}
            for r in effects]


def continuity(tracks, concentration):
    metrics = ('matched_frames', 'coverage_fraction', 'low_matched_frames', 'extreme_matched_frames',
               'isolated010_hits', 'matched_run_count', 'matched_runs_ge2', 'longest_matched_run', 'longest_miss_run')
    index = {(r['condition'], r['arrangement'], r['sequence'], r['track_id']): r for r in tracks}
    changes, never_found, summaries = [], [], []
    for condition, reference in PAIRS:
        for plan in ('O', 'S'):
            chosen = [r for r in tracks if r['condition'] == condition and r['arrangement'] == plan]
            assert chosen and len({(r['sequence'], r['track_id']) for r in chosen}) == len(chosen)
            differences = []
            recovered_tracks = []
            reference_zero_match_tracks = 0
            for row in chosen:
                before = index[reference, plan, row['sequence'], row['track_id']]
                reference_zero_match_tracks += int(int(before['matched_frames']) == 0)
                diff = {m: {'reference': number(before[m]), 'condition': number(row[m]),
                            'delta': number(row[m])-number(before[m])} for m in metrics if number(row[m]) != number(before[m])}
                identity = dict(condition=condition, reference=reference, arrangement=plan,
                                sequence=row['sequence'], track_id=row['track_id'])
                if diff: changes.append(identity | {'metrics': diff})
                if int(before['matched_frames']) == 0 and int(row['matched_frames']) > 0:
                    entry = identity | {m: number(row[m]) for m in metrics} | {
                        'target_frames': int(row['target_frames']), 'reference_matched_frames': 0}
                    never_found.append(entry); recovered_tracks.append(entry)
                differences.append({m: number(row[m])-number(before[m]) for m in metrics})
            summaries.append(dict(condition=condition, reference=reference, arrangement=plan, tracks=len(chosen),
                net_matched_frames=sum(d['matched_frames'] for d in differences),
                track_equal_coverage_delta=statistics.mean(d['coverage_fraction'] for d in differences),
                tracks_with_coverage_gain=sum(d['coverage_fraction'] > 0 for d in differences),
                tracks_with_coverage_loss=sum(d['coverage_fraction'] < 0 for d in differences),
                tracks_with_shorter_longest_miss=sum(d['longest_miss_run'] < 0 for d in differences),
                tracks_with_longer_longest_miss=sum(d['longest_miss_run'] > 0 for d in differences),
                matched_runs_ge2_delta=sum(d['matched_runs_ge2'] for d in differences),
                isolated010_hits_delta=sum(d['isolated010_hits'] for d in differences),
                reference_zero_match_tracks=reference_zero_match_tracks,
                reference_zero_match_to_positive_tracks=len(recovered_tracks),
                recovered_frames_on_reference_zero_match_tracks=sum(r['matched_frames'] for r in recovered_tracks),
                recovered_tracks_with_run_ge2=sum(r['matched_runs_ge2'] > 0 for r in recovered_tracks)))
    unique_discoveries = []
    for condition, reference in PAIRS:
        by_plan = {p: {(r['sequence'], r['track_id']) for r in never_found
            if r['condition'] == condition and r['reference'] == reference and r['arrangement'] == p} for p in ('O', 'S')}
        unique_discoveries.append(dict(condition=condition, reference=reference,
            O=len(by_plan['O']), S=len(by_plan['S']), either_arrangement=len(by_plan['O']|by_plan['S']),
            both_arrangements=len(by_plan['O']&by_plan['S'])))
    compact_concentration = []
    for row in concentration:
        if (row['condition'], row['reference']) not in PAIRS: continue
        entry = compact(row, ('condition', 'reference', 'arrangement', 'outcome', 'frames', 'tracks', 'clips', 'top1_frames', 'top5_frames'))
        entry['top1_fraction'] = entry['top1_frames']/entry['frames'] if entry['frames'] else None
        entry['top5_fraction'] = entry['top5_frames']/entry['frames'] if entry['frames'] else None
        entry['top5_tracks'] = json.loads(row['ranked_tracks'])[:5]
        compact_concentration.append(entry)
    return {'scope': 'Frozen strict5200 target frames on130 tracks; valid annotated strict intervals, not all16297 GT or full physical visibility',
        'main_continuity_cohort': 'all; low/extreme filtering can introduce new breaks and is not used as the main continuous-time claim',
        'summary': summaries, 'changed_tracks': changes, 'reference_zero_match_recovered_tracks': never_found,
        'reference_zero_match_recovered_tracks_O_S_deduplicated': unique_discoveries,
        'recovery_loss_concentration': compact_concentration}


def stage_evidence():
    metrics = ('after_pda_relative_delta_rms', 'after_tsfm_relative_delta_rms',
        'after_update_delta_rms', 'after_update_relative_delta_rms', 'injected_H_delta_rms',
        'logit_delta_rms', 'delta_z_anchor', 'delta_z_mean')
    gate_names = ('mean', 'std', 'min', 'q05', 'q50', 'q95', 'max', 'fraction_lt005', 'fraction_gt095')
    bundles, gate_rows, global_gates = {}, [], []
    for arm in ARMS:
        receipt = read(T / arm / 'TRACE_SUMMARY.json'); verify_outputs(T / arm, receipt)
        all_stage = rows(T / arm / 'STAGE_SUMMARY.csv')
        selected = [r for r in all_stage if r['arrangement'] == 'pooled_unique' and r['phase'] == 't1_39']
        index = {(r['cohort'], r['kind'], r['metric']): r for r in selected}
        margins = [r for r in rows(T / arm / 'PAIRED_MARGIN_SUMMARY.csv')
                   if r['arrangement'] == 'pooled_unique' and r['phase'] == 't1_39']
        mi = {(r['cohort'], r['metric']): r for r in margins}
        if arm == 'V12History':
            mechanism = []
            for cohort in COHORTS:
                for kind in ('target', 'background'):
                    item = dict(cohort=cohort, kind=kind)
                    for metric in metrics:
                        value = index[cohort, kind, metric]
                        item[metric] = {k: number(value[k]) for k in ('mean', 'q50', 'q90')}
                    item['eligible_target_frames'] = int(index[cohort, kind, metrics[0]]['target_frames'])
                    mechanism.append(item)
            margin_rows = [dict(cohort=cohort, metric=metric,
                **{k: number(mi[cohort, metric][k]) for k in ('mean', 'q50', 'q90', 'positive_fraction')})
                for cohort in COHORTS for metric in ('History_anchor_margin', 'HistoryCurrent_anchor_margin',
                    'history_current_anchor_margin_delta', 'history_current_peak_margin_delta')]
            bundles.update(same_History_weight_effects=mechanism, same_History_weight_margins=margin_rows)
        modes = [('HistoryCurrent', C)] if arm == 'V12Current' else [('History', H), ('HistoryCurrent', HC)]
        for mode, condition in modes:
            for cohort in COHORTS:
                for kind in ('target', 'background'):
                    item = dict(condition=condition, cohort=cohort, kind=kind)
                    for stat in gate_names:
                        row = index[cohort, kind, f'retention_{mode}_{stat}']
                        item[stat] = number(row['mean'])
                        item['eligible_target_frames'] = int(row['target_frames'])
                    for metric in ('z_mean', 'z_max', 'anchor_z'):
                        item[metric] = number(index[cohort, kind, mode+'_'+metric]['mean'])
                    gate_rows.append(item)
            global_rows = [r for r in rows(T / arm / 'GLOBAL_RETENTION_SUMMARY.csv')
                if r['arrangement'] == 'pooled_unique' and r['phase'] == 't1_39']
            gi = {r['metric']: r for r in global_rows}
            global_gates.append(dict(condition=condition, **{name: number(gi[f'retention_{mode}_{name}']['mean']) for name in gate_names}))
    return dict(bundles, three_formal_paths_gate_and_logits=gate_rows, global_frame_gate=global_gates,
        scope='t>0; overlap-window average per object/arrangement, equal available O/S then target-frame equal weighting; t0 has no gate',
        quantiles='Gate q05/q50/q95 are averages of ROI/frame quantiles, not quantiles of pooled pixels',
        background_scope='Fixed release-selected same-size hard background per target; repeated physical regions are not independent samples',
        stage_semantics='after_tsfm is bare R; after_update is actual M=(1-g)R+gA; same-History delta=history-current_only',
        t0_and_time_offsets='Separate full tables retained in trace_summary; offset populations change, no causal memory-decay rate inferred')


def training_calibration(runs):
    indices = [5, 6, 18, 25, 30, 32, 51, 7]
    readout_keys = ('target_logit_mean', 'background_logit_mean', 'logit_margin_mean',
        'probability_margin_mean', 'evidence_mean', 'target_logit_above_zero_fraction', 'background_logit_above_zero_fraction')
    state_keys = ('retention_mean', 'retention_std', 'retention_min', 'retention_max', 'aligned_rms', 'candidate_rms', 'updated_state_rms')
    summaries, fingerprints, costs = [], {}, {}
    for arm in ARMS:
        path = E / 'training_diagnostics' / arm / 'FIXED_INPUTS.json'
        data = read(path)
        assert data['status'] == 'COMPLETE' and data['condition'] == arm
        assert data['calibration_indices'] == indices and data['all_model_state_unchanged']
        assert data['backward_calls'] == data['optimizer_steps'] == 0
        assert data['source_sha256'] == runs[arm]['source_sha256']
        assert data['training_run_sha256'] == sha(E / 'training' / arm / 'RUN.json')
        snapshots = {str(r['step']): r['sha256'] for r in runs[arm]['snapshots']}
        assert data['snapshots_sha256'] == snapshots
        costs[arm] = data['elapsed_seconds']
        modes = [('current_only', C)] if arm == 'V12Current' else [('history', H), ('current_only', HC)]
        assert len(data['rows']) == 4*8*len(modes)
        for step in (2048, 2080, 2112, 2176):
            for mode, condition in modes:
                selected = [r for r in data['rows'] if r['step'] == step and r['mode'] == mode]
                assert [r['plan_index'] for r in selected] == indices
                for r in selected:
                    fingerprint = tuple(r[k] for k in ('input_sha256', 'label_sha256', 'auxiliary_input_sha256'))
                    assert r['plan_index'] not in fingerprints or fingerprints[r['plan_index']] == fingerprint
                    fingerprints[r['plan_index']] = fingerprint
                record = dict(condition=condition, additional_step=step-2048, entries=8,
                    observations_with_multiplicity=sum(r['readout']['observations_with_multiplicity'] for r in selected))
                for metric in readout_keys:
                    present = [r['readout'][metric] for r in selected if metric in r['readout']]
                    record[metric] = statistics.mean(present) if present else None
                    record[metric+'_entries_present'] = len(present)
                for metric in state_keys:
                    record[metric] = statistics.mean(r['state_update'][metric] for r in selected)
                summaries.append(record)
    index = {(r['condition'], r['additional_step']): r for r in summaries}
    for row in summaries:
        baseline = index[row['condition'], 0]
        row['change_from_own_0'] = {k: row[k]-baseline[k] if row[k] is not None and baseline[k] is not None else None for k in readout_keys}
    paired = []
    for additional in (0, 32, 64, 128):
        for reference in (C, HC):
            a, b = index[H, additional], index[reference, additional]
            paired.append(dict(condition=H, reference=reference, additional_step=additional,
                **{metric: a[metric]-b[metric] if a[metric] is not None and b[metric] is not None else None for metric in readout_keys}))
    return {'calibration_indices': indices, 'summary': summaries, 'History_minus_reference': paired,
        'elapsed_seconds': costs, 'ordered_input_fingerprints_identical': True,
        'weighting': 'Equal mean across8 fixed training entries; each source readout internally counts cached observations with multiplicity',
        'scope': 'Training calibration only; no development matching, no thresholded recovery claim; 0/32/64 are process records, final128 fixed'}


def decide(effects, summary, protected):
    by = {(r['comparison'], r['cohort']): r for r in effects}
    paired = by['independently_trained_control', 'low_all_05']
    same = by['same_History_checkpoint', 'low_all_05']
    rp, rs = paired['R_hist_unique_either_arrangement'], same['R_hist_unique_either_arrangement']
    if rp == rs == 0:
        code, text = 'NO_OBSERVED_HISTORY_RECOVERY', '两个比较均没有新增低响应实际匹配；本轮未建立 M1 的历史恢复能力。'
    elif rp and not rs:
        code, text = 'PAIRED_SIGNAL_WITHOUT_SAME_WEIGHT_SUPPORT', '独立训练比较出现低响应恢复，但同 History 权重干预没有恢复增量；历史作用尚缺第二条证据。'
    elif rs and not rp:
        code, text = 'WITHIN_WEIGHT_SIGNAL_WITHOUT_TRAINED_CURRENT_ADVANTAGE', '同权重历史干预出现低响应恢复，但未优于独立训练的同容量 Current；尚不能认为主比较通过。'
    else:
        code, text = 'RECOVERY_SIGNAL_REQUIRES_COST_AND_CONTINUITY_REVIEW', '两个比较均出现低响应实际恢复信号；仍须连同损失、轨迹分散度、连续性及代价判断，不能仅凭 R_hist 判定成功。'
    cost = {r['arrangement']: {'U': r['unmatched_components'], 'budget': r['U_budget'],
        'passes': r['unmatched_components'] <= r['U_budget']} for r in summary if r['condition'] == H}
    protection = {r['condition']: r for r in protected if r['cohort'] == 'both_fixed_success'}
    return {'code': code, 'conclusion': text, 'paired_low_R_L_unique': [rp, paired['L_hist_unique_either_arrangement']],
        'same_weight_low_R_L_unique': [rs, same['L_hist_unique_either_arrangement']], 'U_budget': cost,
        'protected_History_vs_Current': {k: protection[H][k]-protection[C][k] for k in ('both_matched', 'neither_matched')},
        'scope': 'Conservative evidence classification for this paired128-step run, not novelty/generalization or permanent feasibility judgment',
        'next_action': 'No automatic continuation, Frame control, added loss, new mechanism or M2; final interpretation is reviewed by the project owner'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--overwrite', action='store_true', help='Replace an existing rendered report from verified complete inputs')
    args = parser.parse_args()
    if not args.overwrite and any((E/name).exists() for name in ('RESULTS.json', 'RESULTS.md')):
        raise FileExistsError('Preserve existing report; use --overwrite explicitly')
    evaluation = read(V / 'SUMMARY_COMPLETE.json'); verify_outputs(V, evaluation)
    aggregate = read(T / 'SUMMARY.json'); verify_outputs(T, aggregate)
    freeze = read(E / 'PROTOCOL_FREEZE.json')
    assert freeze['status'] == 'FROZEN'
    for path, expected in freeze['source_sha256'].items(): assert remember(path) == expected
    checks = {}
    for path, expected in freeze['checks'].items():
        assert remember(path) == expected
        check = read(path); assert check['status'] == 'PASS'
        checks[Path(path).name] = {'status': check['status'], 'path': path, 'sha256': expected}
    for path in (E/'checks/PAIRED_TRAINING.json', V/'PAIRED_TRAINING_CHECK.json'):
        check = read(path); assert check['status'] in ('PASS', 'COMPLETE')
        checks[path.name + ('_evaluation' if path.parent == V else '')] = {'status': check['status'], 'path': str(path), 'sha256': sha(path)}
    runs = {arm: read(E/'training'/arm/'RUN.json') for arm in ARMS}
    for arm, run in runs.items():
        assert run['status'] == 'COMPLETE' and run['final_step'] == 2176
        assert run['completed_additional_steps'] == 128 and run['source_sha256'] == freeze['source_sha256']
        assert run['original_unchanged'] and run['initial_checkpoint_sha256'] == freeze['checkpoint_sha256']
        assert remember(E/'training'/arm/'snapshot_2176.pth') == run['checkpoint_sha256']
    assert runs[ARMS[0]]['exposure_digest_sha256'] == runs[ARMS[1]]['exposure_digest_sha256']
    artifact_path = E/'analysis/TRAINING_ARTIFACT_CHECK.json'
    artifact_check = read(artifact_path)
    assert artifact_check['status'] == 'PASS' and artifact_check['ordered_exposure_exact']
    assert artifact_check['protocol_sha256'] == sha(E/'PROTOCOL_FREEZE.json')
    assert artifact_check['initial_checkpoint_sha256'] == freeze['checkpoint_sha256']
    assert artifact_check['shared_initialization_sha256'] == remember(E/'init/initial_model.pth')
    assert all(run['v12_initial_checkpoint_sha256'] == artifact_check['shared_initialization_sha256']
               for run in runs.values())
    assert artifact_check['verification_source_sha256'] == remember(E/'analysis/verify_training_artifacts.py')
    for arm in ARMS:
        actual = artifact_check['arms'][arm]
        assert actual['run_sha256'] == sha(E/'training'/arm/'RUN.json')
        assert actual['training_log_sha256'] == remember(E/'training'/arm/'TRAIN.jsonl')
        assert actual['original_base_tensors_exact'] and actual['initial_weights_and_old_moments_exact']
        assert actual['all_parameters_and_moments_finite'] and actual['gate_loss_gradients_nonzero_after_first_step']
    checks[artifact_path.name] = {'status': 'PASS', 'path': str(artifact_path), 'sha256': sha(artifact_path)}
    summary_fields = ('condition','arrangement','M','gt_count','matched_count','component_count','unmatched_components',
        'positive_pixels_outside_boxes','unmatched_component_area_sum','component_area_sum','U_budget','passes_pooled_U_budget',
        'low_all_05_matched','extreme_all_001_matched','both_fixed_success_matched','m1_low_common_recovered_matched')
    original_summary = [r for r in rows(V/'SUMMARY.csv') if r['group'] == 'pooled']
    summary = [compact(r, summary_fields) for r in original_summary]
    assert {(r['condition'], r['arrangement']) for r in summary} == {(c,p) for c in CONDITIONS for p in ('O','S')}
    allgt = [r for r in rows(V/'ALL_GT_SUMMARY.csv') if r['group'] == 'pooled']
    strict = rows(V/'OBJECT_COMPARISON.csv')
    effects = verify_history_effects(rows(V/'LOW_RESPONSE_HISTORY_EFFECTS.csv'), strict)
    comparisons, strict_changes = matching_comparisons(original_summary, allgt, strict)
    protected = [compact(r, ('condition','cohort','target_frames','O_matched','S_matched','both_matched','O_only','S_only',
        'neither_matched','lost_dual_success','lost_both_arrangements'))
        for r in rows(V/'PROTECTED_AND_WINDOW_RETENTION.csv') if r['cohort'] in ('both_fixed_success','m1_low_common_recovered')]
    tracks = rows(V/'TRACK_COVERAGE.csv')
    continuous = continuity(tracks, rows(V/'RECOVERY_CONCENTRATION.csv'))
    focused = [{k: r[k] for k in ('sequence','frame','annotation_id','track_id','arrangement')} |
        {c+'_matched': int(r[c+'_matched']) for c in CONDITIONS} for r in strict
        if r['sequence']=='DJI_0067_2' and r['track_id']=='7702014' and int(r['frame'])==101]
    focused_tracks = [compact(r, ('condition','arrangement','sequence','track_id','target_frames','matched_frames',
        'coverage_fraction','longest_miss_run','longest_matched_run','matched_runs_ge2','isolated010_hits'))
        for r in tracks if r['sequence']=='DJI_0067_2' and r['track_id']=='7702014']
    old_v11 = E.parent/'m1_recurrent_v1_1/evaluation/step_2176/TRACK_COVERAGE.csv'
    prior_damage = [compact(r, ('condition','arrangement','sequence','track_id','target_frames','matched_frames',
        'coverage_fraction','longest_miss_run')) for r in rows(old_v11)
        if r['sequence']=='DJI_0067_2' and r['track_id']=='7702014' and r['condition'] in ('ContinuationHistory','RevisionHistory')]
    assert len(focused)==2 and len(focused_tracks)==10
    mechanism = stage_evidence()
    calibration = training_calibration(runs)
    cost = {'parameters': {}, 'training': {}, 'traced_inference': [],
        'scope': 'Inference includes3-mode shared-front-end trace, ROI/Jacobian statistics and serialization; not deployment latency. Memory is peak allocated, not reserved.'}
    for arm, run in runs.items():
        cost['parameters'][arm] = {'total':run['parameters'],'trainable':run['trainable_parameters'],'new_gate':2017}
        cost['training'][arm] = {'additional_steps':128,'elapsed_seconds':run['elapsed_seconds'],
            'peak_cuda_allocated_GiB':run['peak_cuda_allocated_bytes']/2**30}
        for sequence in ('DJI_0026_2','DJI_0067_2'):
            receipt = read(E/'traces'/arm/sequence/'COMPLETE.json')
            assert receipt['status']=='COMPLETE' and len(receipt['windows'])==8
            cost['traced_inference'].append(dict(arm=arm,sequence=sequence,windows=8,
                elapsed_seconds=receipt['elapsed_seconds'],
                peak_cuda_allocated_GiB=max(w['peak_cuda_allocated_bytes'] for w in receipt['windows'])/2**30))
    decision = decide(effects, summary, protected)
    primary_tracks = [r for r in continuous['summary'] if r['reference'] in (C, HC)]
    actual_strict_identical = all(r[H+'_matched'] == r[C+'_matched'] == r[HC+'_matched'] for r in strict)
    low_target = next(r for r in mechanism['same_History_weight_effects']
                      if r['cohort'] == 'low_all_05' and r['kind'] == 'target')
    low_background = next(r for r in mechanism['same_History_weight_effects']
                          if r['cohort'] == 'low_all_05' and r['kind'] == 'background')
    low_gate = next(r for r in mechanism['three_formal_paths_gate_and_logits']
                   if r['condition'] == H and r['cohort'] == 'low_all_05' and r['kind'] == 'target')
    interpretation = {
        'all_three_formal_strict_match_records_identical': actual_strict_identical,
        'strict_object_arrangement_records': len(strict),
        'primary_comparisons_reference_zero_match_recovered_tracks':
            sum(r['reference_zero_match_to_positive_tracks'] for r in primary_tracks),
        'low_t1_39_target_delta_logit': low_target['delta_z_anchor']['mean'],
        'low_t1_39_background_delta_logit': low_background['delta_z_anchor']['mean'],
        'low_t1_39_after_update_relative_difference': low_target['after_update_relative_delta_rms']['mean'],
        'low_t1_39_injected_H_difference_rms': low_target['injected_H_delta_rms']['mean'],
        'low_t1_39_logit_difference_rms': low_target['logit_delta_rms']['mean'],
        'low_t1_39_retention_mean': low_gate['mean'], 'low_mean_within_ROI_retention_std': low_gate['std'],
        'boundary': 'Observable effect chain at this endpoint, not a proof of the unique failure cause; no Frame control was run'}
    official = read(E/'upstream/MatAnyone_PROVENANCE.json')
    remember(E/'upstream/MATANYONE_REFERENCE_AUDIT.md')
    result = {'status':'COMPLETE','created_utc':datetime.now(timezone.utc).isoformat(),'decision':decision,
        'protocol':{'arms':list(ARMS),'mode': 'pixel in both arms','initial_step':2048,'additional_steps_each':128,'final_step':2176,
            'parent_History2048_checkpoint_sha256':freeze['checkpoint_sha256'],
            'shared_initial_checkpoint_sha256':artifact_check['shared_initialization_sha256'],
            'final_checkpoint_sha256':{a:runs[a]['checkpoint_sha256'] for a in ARMS},
            'ordered_exposure_sha256':runs[ARMS[0]]['exposure_digest_sha256'],
            'single_change':'A=PDA(previous,X); R=TSFM(A,X); supplied core M=(1-g)R+gA; M0=X0'},
        'score_table':summary,'comparisons':comparisons,'low_response_history_effects':effects,
        'endpoint_interpretation':interpretation,
        'strict_changed_objects':strict_changes,'protection_and_common9':protected,'continuity':continuous,
        'previously_damaged_case':{'sequence':'DJI_0067_2','track_id':'7702014','frame':101,'annotation_id':'876734',
            'matches':focused,'current_track_values':focused_tracks,'v11_track_values':prior_damage},
        'mechanism':mechanism,'fixed_training_calibration':calibration,'cost':cost,'checks':checks,
        'reference':{'RFR':'Unchanged pinned PDA/TSFM and real DCN from m1_recurrent_v1/upstream',
            'state_update':'Unmodified user reference core; 2017 parameters; no mask/BN/detach; g is retention, not calibrated uncertainty',
            'MatAnyone_commit':official['commit'],'paper':official['paper']['official_landing'],
            'differences':'Official UncertPred uses prior/current pixel features, prior mask, new-old values, BN and alpha-change supervision; U weights new values, whereas our g weights aligned old state'},
        'limitations':['M/U are actual-component diagnostics, not official Pd/Fa.',
            'Strict continuity covers frozen5200 annotated target frames/130 tracks, not all16297 GT or complete physical visibility.',
            'O/S union recovery and loss can intersect; their difference is not a single-config net benefit.',
            'Both trained arms include the original temporal frontend/TPro; Current only removes additional M1 recursive history.',
            'No Frame-control evidence here; pixel-selective gating benefit is not established by this comparison alone.',
            'Training calibration, larger state differences and lower background cost cannot replace actual recovery.',
            'Native DCNv2/adaptive-pooling backward is not bitwise deterministic; one short paired run does not establish generalization.'],
        'source_sha256':SOURCES,'report_source_sha256':sha(__file__)}
    lines = ['**M1-v1.2 配对128步实验已完成。'+decision['conclusion']+'**', '',
        '两臂从同一旧History2048权重、旧53个Adam状态和相同新门控初始化开始，在相同缓存曝光上训练到固定2176。'
        'Current/History均执行PDA、TSFM和pixel门控；Current只取消M1新增递归历史，原时序前端/TPro仍在。'
        '原检测头与BN冻结，首帧M0=X0、40帧窗口重置和原匹配协议不变。', '',
        'M是strict5200目标帧的匹配数，U经全部16297有效GT匹配后计数；U不是已确认假阳性数。A为GT框外正像素。'
        'low/extreme是冻结4342/4170目标帧响应分层，不等于整条轨迹持续未发现。']
    if actual_strict_identical and decision['code'] == 'NO_OBSERVED_HISTORY_RECOVERY':
        lines += ['', '三条正式路径的10400条O/S小目标匹配记录逐条相同，130条轨迹的覆盖与连续漏检没有历史增量。'
            '原有9个去重低响应检出仍被三路共同获得；此前零检出的轨迹没有因新增历史而被发现。'
            'History相对独立Current虽使U减少O5/S4，但全部有效GT额外丢失O2/S4个目标帧；'
            '相对同权重current_only，U减少11/6，同时额外丢失3/5个目标帧。', '',
            f"作用链显示历史差异已经进入更新状态、接入H和logits：low目标ΔM相对幅度{low_target['after_update_relative_delta_rms']['mean']:.6g}，"
            f"ΔH RMS={low_target['injected_H_delta_rms']['mean']:.6g}，Δz RMS={low_target['logit_delta_rms']['mean']:.6g}。"
            f"但固定目标点有符号Δz={low_target['delta_z_anchor']['mean']:.6g}，背景={low_background['delta_z_anchor']['mean']:.6g}；"
            '间隔的正变化来自背景下降更多，不能解释为目标证据增强。'
            f"目标ROI的保留系数均值{low_gate['mean']:.6g}、ROI内标准差均值{low_gate['std']:.6g}，仍非常接近0.5。"
            '结合固定训练输入上也没有目标相关历史优势，本轮不支持追加训练或展开Frame／Pixel；保留负结果，不增加M2。']
    table(lines,['条件','安排','M','U','A','未匹配面积','总组件面积','low','extreme','protect646'],
        [[LABEL[r['condition']],r['arrangement']]+[r[k] for k in ('M','unmatched_components','positive_pixels_outside_boxes',
        'unmatched_component_area_sum','component_area_sum','low_all_05_matched','extreme_all_001_matched','both_fixed_success_matched')] for r in summary])
    lines += ['', '**新增历史恢复与反向损失分别报告。**独立训练比较为H/C；同权重比较为H/同H权重current_only。'
        'unique按sequence/frame/annotation跨O/S去重，O gain与S loss可属于同一对象，不把union差称为统一配置净收益。']
    table(lines,['比较','集合','R O/S','R unique/both','L O/S','L unique/both','跨安排交叉'],
        [[r['comparison'],r['cohort'],f"{r['R_hist_O']}/{r['R_hist_S']}",
          f"{r['R_hist_unique_either_arrangement']}/{r['R_hist_both_arrangements']}",f"{r['L_hist_O']}/{r['L_hist_S']}",
          f"{r['L_hist_unique_either_arrangement']}/{r['L_hist_both_arrangements']}",r['objects_gain_in_one_plan_loss_in_other']] for r in effects])
    table(lines,['条件 / 参考','安排','strict G/L','全部GT G/L','ΔU','ΔA','Δ未匹配面积'],
        [[LABEL[r['condition']]+' / '+LABEL[r['reference']],r['arrangement'],f"{r['strict_G']}/{r['strict_L']}",
          f"{r['all_valid_G']}/{r['all_valid_L']}",r['delta_U'],r['delta_outside_box_positive_pixels'],r['delta_unmatched_area']] for r in comparisons])
    lines += ['', '**轨迹覆盖与持续漏检。**以下主表使用全部strict有效区间：5200目标帧、130轨迹，'
        '不是16297全部GT的完整物理可见轨迹。low/extreme先过滤会制造断点，故不据其子序列宣称连续漏检恢复。']
    table(lines,['条件 / 参考','安排','净匹配帧','覆盖增/减轨迹','最长漏检缩短/增长','连续命中段≥2变化','孤立命中变化','ref零检出→检出轨迹/零检出总数；恢复帧'],
        [[LABEL[r['condition']]+' / '+LABEL[r['reference']],r['arrangement'],r['net_matched_frames'],
          f"{r['tracks_with_coverage_gain']}/{r['tracks_with_coverage_loss']}",
          f"{r['tracks_with_shorter_longest_miss']}/{r['tracks_with_longer_longest_miss']}",r['matched_runs_ge2_delta'],
          r['isolated010_hits_delta'],f"{r['reference_zero_match_to_positive_tracks']}/{r['reference_zero_match_tracks']}；{r['recovered_frames_on_reference_zero_match_tracks']}"]
          for r in continuous['summary']])
    table(lines,['条件 / 参考','安排','gain/loss','帧数','轨迹数','top1/top5帧'],
        [[LABEL[r['condition']]+' / '+LABEL[r['reference']],r['arrangement'],r['outcome'],r['frames'],r['tracks'],
          f"{r['top1_frames']}/{r['top5_frames']}"] for r in continuous['recovery_loss_concentration'] if r['reference'] in (C,HC)])
    primary_changes = [r for r in strict_changes if r['reference'] in (C, HC)]
    lines += ['', f'主比较及同权重比较共有{len(primary_changes)}条strict改变记录（可重复涉及同一对象）；完整逐对象和逐轨迹变化见[RESULTS.json](RESULTS.json)、'
        '[TRACK_COVERAGE.csv](evaluation/step_2176/TRACK_COVERAGE.csv)。ref零检出指该参考配置在冻结strict区间内零匹配，不外推完整轨迹。']
    if primary_changes:
        table(lines,['比较','安排','视频 / track / frame / annotation','ref→H','low/extreme/protect'],
            [[LABEL[r['reference']]+'→H',r['arrangement'],f"{r['sequence']} / {r['track_id']} / {r['frame']} / {r['annotation_id']}",
              f"{r['reference_matched']}→{r['condition_matched']}",f"{r['low']}/{r['extreme']}/{r['protected']}"] for r in primary_changes])
    if continuous['reference_zero_match_recovered_tracks']:
        table(lines,['ref→条件','安排','视频 / track','恢复帧','最长连续命中','连续段≥2','孤立命中'],
            [[LABEL[r['reference']]+'→'+LABEL[r['condition']],r['arrangement'],r['sequence']+' / '+r['track_id'],
              r['matched_frames'],r['longest_matched_run'],r['matched_runs_ge2'],r['isolated010_hits']]
             for r in continuous['reference_zero_match_recovered_tracks']])
    lines += ['', '**保护集合与原受损对象。**下面逐安排展示DJI_0067_2 / track7702014 / frame101 / annotation876734；'
        '它是既定解释对象，未用于新增采样或选检查点。']
    table(lines,['条件','protect O/S','双安排保持','双安排均失','common9双保持'],
        [[LABEL[c], f"{next(r for r in protected if r['condition']==c and r['cohort']=='both_fixed_success')['O_matched']}/"
          f"{next(r for r in protected if r['condition']==c and r['cohort']=='both_fixed_success')['S_matched']}",
          next(r for r in protected if r['condition']==c and r['cohort']=='both_fixed_success')['both_matched'],
          next(r for r in protected if r['condition']==c and r['cohort']=='both_fixed_success')['neither_matched'],
          next(r for r in protected if r['condition']==c and r['cohort']=='m1_low_common_recovered')['both_matched']] for c in CONDITIONS])
    table(lines,['条件','安排','frame101匹配','该track匹配帧','覆盖','最长漏检'],
        [[LABEL[r['condition']],r['arrangement'],next(o for o in focused if o['arrangement']==r['arrangement'])[r['condition']+'_matched'],
          r['matched_frames'],fmt(r['coverage_fraction']),r['longest_miss_run']] for r in focused_tracks])
    lines += ['', 'v1.1已登记的同轨迹参照：'+ '；'.join(f"{r['condition']} {r['arrangement']}最长漏检={r['longest_miss_run']}" for r in prior_damage)+'。', '',
        '**同History权重的作用链。**t>0，先平均同对象重叠窗口再平均可用O/S，最后目标帧等权。'
        'Δ=正常history−同权重current_only；after_tsfm是裸候选R，after_update才是实际M。背景为发布模型预先固定的困难参照，不是独立背景样本。']
    table(lines,['集合','种类','有效对象帧','ΔM RMS','ΔM相对','ΔH RMS','Δz RMS','固定点有符号Δz'],
        [[r['cohort'],r['kind'],r['eligible_target_frames']]+[fmt(r[k]['mean']) for k in
          ('after_update_delta_rms','after_update_relative_delta_rms','injected_H_delta_rms','logit_delta_rms','delta_z_anchor')]
          for r in mechanism['same_History_weight_effects']])
    table(lines,['集合','H−current固定点margin','H−current峰margin'],
        [[cohort]+[fmt(next(r for r in mechanism['same_History_weight_margins'] if r['cohort']==cohort and r['metric']==metric)['mean'])
           for metric in ('history_current_anchor_margin_delta','history_current_peak_margin_delta')] for cohort in COHORTS])
    table(lines,['正式路','low种类','gate均值','std','q05/q50/q95','g<.05 / g>.95'],
        [[LABEL[r['condition']],r['kind'],fmt(r['mean']),fmt(r['std']),'/'.join(fmt(r[k]) for k in ('q05','q50','q95')),
          fmt(r['fraction_lt005'])+' / '+fmt(r['fraction_gt095'])] for r in mechanism['three_formal_paths_gate_and_logits'] if r['cohort']=='low_all_05'])
    lines += ['', 'gate是历史保留系数，分位数是各ROI分位数的对象均值，不是所有像素混合后的分位数。'
        '状态差异、gate分布或margin上涨都不能代替实际恢复；margin变化需同时看目标与背景的有符号变化。', '',
        '**固定8条训练校准输入，0/32/64/128步。**每条使用原固定目标/背景缓存；下表对8条等权，单条内部含重复观察计数。'
        '只作训练作用诊断，不做开发集匹配，也不以早期检查点替代预定128终点。']
    table(lines,['路径','追加步','目标logit','背景logit','margin','Δ目标/Δ背景/Δmargin vs0','gate mean/std','M RMS'],
        [[LABEL[r['condition']],r['additional_step'],fmt(r['target_logit_mean']),fmt(r['background_logit_mean']),fmt(r['logit_margin_mean']),
          '/'.join(fmt(r['change_from_own_0'][k]) for k in ('target_logit_mean','background_logit_mean','logit_margin_mean')),
          fmt(r['retention_mean'])+'/'+fmt(r['retention_std']),fmt(r['updated_state_rms'])] for r in calibration['summary']])
    table(lines,['训练校准 H−参考','追加步','Δ目标logit','Δ背景logit','Δmargin'],
        [[LABEL[r['reference']],r['additional_step'],fmt(r['target_logit_mean']),fmt(r['background_logit_mean']),fmt(r['logit_margin_mean'])]
         for r in calibration['History_minus_reference']])
    lines += ['', '**成本与实现检查。**训练时间含原日志/存储；诊断推理包含共享前端的三模式、ROI/Jacobian统计和工件写出，不能当部署延迟。']
    table(lines,['训练臂','总/可训参数','新增gate','128步秒','峰值allocated GiB'],
        [[a,f"{cost['parameters'][a]['total']}/{cost['parameters'][a]['trainable']}",2017,
          fmt(cost['training'][a]['elapsed_seconds']),fmt(cost['training'][a]['peak_cuda_allocated_GiB'])] for a in ARMS])
    table(lines,['诊断臂','视频','窗口数','秒','峰值allocated GiB'],
        [[r['arm'],r['sequence'],r['windows'],fmt(r['elapsed_seconds']),fmt(r['peak_cuda_allocated_GiB'])] for r in cost['traced_inference']])
    lines += ['', '门控原包CPU测试、真实CUDA集成及双臂身份检查均通过；保存的四阶段检查点已逐张量验证旧53个Adam状态从2048到2176、新6个参数状态从空到128，原冻结部分保持不变，曝光同序且第二步门控各层真实梯度非零。'
        '详见[训练工件核验](analysis/TRAINING_ARTIFACT_CHECK.json)、[checks](checks/)及JSON源哈希。'
        '原生DCN/池化反向不保证位级确定性。U预算逐安排结果：'+ '；'.join(f"{p} {r['U']}/{r['budget']}（{'通过' if r['passes'] else '未通过'}）" for p,r in decision['U_budget'].items())+'。'
        '即使S满足4015预算，也不代表相对release S成本不增加。', '',
        '代码保留本地固定RFR的真实PDA/TSFM和DCN，直接接入用户提供的状态更新核心。'
        '借鉴[MatAnyone（CVPR 2025）](https://openaccess.thecvf.com/content/CVPR2025/html/Yang_MatAnyone_Stable_Video_Matting_with_Consistent_Memory_Propagation_CVPR_2025_paper.html)的新旧记忆插值形式；'
        '官方UncertPred有上一帧mask、BN及alpha变化监督，官方U乘新值；本方案无这些输入/监督，g乘已对齐旧状态，不能称原模块复现或校准不确定性。'
        '来源和固定commit见[核验说明](upstream/MATANYONE_REFERENCE_AUDIT.md)。', '',
        '结论仅限本轮两个开发视频、单seed短期筛选。当前未进行Frame对照、延续训练或M2；该报告不自动触发任何后续实验。'
        '完整数值、逐对象/轨迹变化及证据链接见[RESULTS.json](RESULTS.json)。', '']
    for name, content in [('RESULTS.json',json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n'),
                          ('RESULTS.md','\n'.join(lines))]:
        temporary = E/(name+'.tmp'); temporary.write_text(content,encoding='utf-8'); temporary.replace(E/name)
    print(json.dumps({'status':'COMPLETE','decision':decision['code'],
        'RESULTS.json':sha(E/'RESULTS.json'),'RESULTS.md':sha(E/'RESULTS.md')},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
