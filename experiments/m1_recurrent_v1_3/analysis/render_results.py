"""Render only complete M1-v1.3 evidence; no imports of model/runtime or workloads.

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
V = E / 'evaluation/step_2560'
T = E / 'analysis/trace_summary'
ARMS = ('BankCurrent', 'BankHistory')
C, H, HC = 'BankCurrentCurrent', 'BankHistoryHistory', 'BankHistoryCurrent'
CONDITIONS = ('release', 'History', C, H, HC)
PAIRS = ((H, C), (H, HC), (H, 'History'), (H, 'release'), (C, 'History'))
COHORTS = ('strict_all', 'low_all_05', 'extreme_all_001', 'both_fixed_success', 'm1_low_common_recovered')
LABEL = {'release': 'release', 'History': '旧History2048', C: 'Bank-Current',
         H: 'Bank-History', HC: '同H权重current_only'}
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


def training_calibration(runs):
    indices = [5, 6, 18, 25, 30, 32, 51, 7]
    readout_keys = ('target_logit_mean', 'background_logit_mean', 'logit_margin_mean',
        'probability_margin_mean', 'evidence_mean', 'target_logit_above_zero_fraction', 'background_logit_above_zero_fraction')
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
        modes = [('current_only', C)] if arm == 'BankCurrent' else [('history', H), ('current_only', HC)]
        assert len(data['rows']) == 4*8*len(modes)
        for step in (2048, 2176, 2304, 2560):
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
                summaries.append(record)
    index = {(r['condition'], r['additional_step']): r for r in summaries}
    for row in summaries:
        baseline = index[row['condition'], 0]
        row['change_from_own_0'] = {k: row[k]-baseline[k] if row[k] is not None and baseline[k] is not None else None for k in readout_keys}
    paired = []
    for additional in (0, 128, 256, 512):
        for reference in (C, HC):
            a, b = index[H, additional], index[reference, additional]
            paired.append(dict(condition=H, reference=reference, additional_step=additional,
                **{metric: a[metric]-b[metric] if a[metric] is not None and b[metric] is not None else None for metric in readout_keys}))
    return {'calibration_indices': indices, 'summary': summaries, 'History_minus_reference': paired,
        'elapsed_seconds': costs, 'ordered_input_fingerprints_identical': True,
        'weighting': 'Equal mean across8 fixed training entries; each source readout internally counts cached observations with multiplicity',
        'scope': 'Training calibration only; no development matching, no thresholded recovery claim; 0/128/256 are process records, final512 fixed'}


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
        'scope': 'Conservative evidence classification for this paired512-step run, not novelty/generalization or permanent feasibility judgment',
        'next_action': 'No automatic continuation, added loss, new mechanism or M2; final interpretation is reviewed by the project owner'}


def stage_evidence():
    metrics = ('before_pda_relative_delta_rms', 'after_pda_relative_delta_rms',
               'after_tsfm_relative_delta_rms', 'after_tsfm_delta_rms',
               'injected_H_delta_rms', 'logit_delta_rms', 'delta_z_anchor', 'delta_z_mean')
    result, margin_result = [], []
    for arm in ARMS:
        receipt = read(T/arm/'TRACE_SUMMARY.json'); verify_outputs(T/arm, receipt)
        selected = [r for r in rows(T/arm/'STAGE_SUMMARY.csv')
                    if r['arrangement']=='pooled_unique' and r['phase']=='t1_39']
        index = {(r['cohort'],r['kind'],r['metric']):r for r in selected}
        for cohort in COHORTS:
            for kind in ('target','background'):
                record = dict(arm=arm, cohort=cohort, kind=kind)
                for metric in metrics:
                    row = index[cohort,kind,metric]
                    record[metric] = {k:number(row[k]) for k in ('mean','q50','q90','positive_fraction')}
                    record['eligible_target_frames'] = int(row['target_frames'])
                result.append(record)
        margin_result.extend(dict(r, arm=arm) for r in rows(T/arm/'PAIRED_MARGIN_SUMMARY.csv')
                             if r['arrangement']=='pooled_unique' and r['phase']=='t1_39')
    return {'same_weight_effects':result, 'margins':margin_result,
            'stage_semantics':'before_pda=mean(original source X); after_pda=mean(aligned sources); after_tsfm=actual mean candidate readout; none is recurrent fused state',
            'scope':'t>0 fixed target and background ROIs; windows averaged per object then available O/S averaged, then equal object weighting; stage attenuation is descriptive, not causal retention rate'}


def main():
    if any((E/n).exists() for n in ('RESULTS.json','RESULTS.md')):
        raise FileExistsError('Preserve existing report')
    evaluation=read(V/'SUMMARY_COMPLETE.json'); verify_outputs(V,evaluation)
    aggregate=read(T/'SUMMARY.json'); verify_outputs(T,aggregate)
    freeze=read(E/'PROTOCOL_FREEZE.json'); assert freeze['status']=='FROZEN'
    for p,h in freeze['source_sha256'].items(): assert remember(p)==h
    for p,h in freeze['checks'].items():
        assert remember(p)==h and read(p)['status']=='PASS'
    runs={arm:read(E/'training'/arm/'RUN.json') for arm in ARMS}
    pair=read(E/'checks/PAIRED_TRAINING.json'); assert pair['status']=='PASS'
    artifact=read(E/'analysis/TRAINING_ARTIFACT_CHECK.json')
    assert artifact['status']=='PASS' and artifact['ordered_exposure_exact']
    assert artifact['protocol_sha256']==sha(E/'PROTOCOL_FREEZE.json')
    assert artifact['verification_source_sha256']==remember(E/'analysis/verify_training_artifacts.py')
    for arm,run in runs.items():
        assert run['status']=='COMPLETE' and run['final_step']==2560
        assert run['completed_additional_steps']==512 and run['source_sha256']==freeze['source_sha256']
        assert run['initial_checkpoint_sha256']==freeze['checkpoint_sha256']
        assert remember(E/'training'/arm/'snapshot_2560.pth')==run['checkpoint_sha256']
        assert artifact['arms'][arm]['run_sha256']==sha(E/'training'/arm/'RUN.json')
        assert artifact['arms'][arm]['training_log_sha256']==remember(E/'training'/arm/'TRAIN.jsonl')
    fields=('condition','arrangement','M','gt_count','matched_count','component_count','unmatched_components',
            'positive_pixels_outside_boxes','unmatched_component_area_sum','component_area_sum','U_budget',
            'passes_pooled_U_budget','low_all_05_matched','extreme_all_001_matched',
            'both_fixed_success_matched','m1_low_common_recovered_matched')
    raw_summary=[r for r in rows(V/'SUMMARY.csv') if r['group']=='pooled']
    summary=[compact(r,fields) for r in raw_summary]
    strict=rows(V/'OBJECT_COMPARISON.csv')
    effects=verify_history_effects(rows(V/'LOW_RESPONSE_HISTORY_EFFECTS.csv'),strict)
    allgt=[r for r in rows(V/'ALL_GT_SUMMARY.csv') if r['group']=='pooled']
    comparisons,changed=matching_comparisons(raw_summary,allgt,strict)
    protected=[compact(r,('condition','cohort','target_frames','O_matched','S_matched','both_matched',
                          'O_only','S_only','neither_matched','lost_dual_success','lost_both_arrangements'))
               for r in rows(V/'PROTECTED_AND_WINDOW_RETENTION.csv')
               if r['cohort'] in ('both_fixed_success','m1_low_common_recovered')]
    continuous=continuity(rows(V/'TRACK_COVERAGE.csv'),rows(V/'RECOVERY_CONCENTRATION.csv'))
    mechanism=stage_evidence(); calibration=training_calibration(runs)
    decision=decide(effects,summary,protected)
    cost={'training':{arm:{'steps':512,'elapsed_seconds':r['elapsed_seconds'],
                          'peak_cuda_allocated_GiB':r['peak_cuda_allocated_bytes']/2**30,
                          'total_parameters':r['parameters'],'trainable_parameters':r['trainable_parameters']}
                      for arm,r in runs.items()},
          'resource_preflight':read(E/'checks/GPU_RESOURCES_V2.json'),
          'pairs_per_40frame_window':150,'old_recurrent_pairs':39,'added_parameters':0,
          'note':'150 pair calls versus39; new arms match each other, not original compute. Preflight inference timings exclude trace/ROI collection; runtime shared-GPU/CPU load can change throughput.'}
    result={'status':'COMPLETE','created_utc':datetime.now(timezone.utc).isoformat(),
            'decision':decision,'score_table':summary,'comparisons':comparisons,
            'low_response_history_effects':effects,'strict_changed_objects':changed,
            'protection_and_common9':protected,'continuity':continuous,
            'mechanism':mechanism,'fixed_training_calibration':calibration,'cost':cost,
            'protocol':{'initial_step':2048,'additional_steps_each':512,'final_step':2560,
                        'capacity':4,'source':'unaltered projected X; read before write; FIFO4; mean readout',
                        'shared_checkpoint_sha256':freeze['checkpoint_sha256'],
                        'final_checkpoint_sha256':{arm:r['checkpoint_sha256'] for arm,r in runs.items()}},
            'limits':['M/U are diagnostic actual matching and unmatched components, not official Pd/Fa.',
                      'O/S are arrangements of the same objects; union recovery and loss may overlap.',
                      'Strict5200/130tracks and all16297 GT are distinct; continuity is main cohort all.',
                      'Current retains the original temporal frontend and TPro; it removes only additional bank source content.',
                      'Cache correctness or source difference does not establish target evidence value.',
                      'Comparisons do not establish superiority over every equal-compute/equal-horizon recurrent method.',
                      'No M2, new gate, new loss, new data or best-checkpoint selection.'],
            'source_sha256':SOURCES}
    lines=['**M1-v1.3 配对512步实验已完成。'+decision['conclusion']+'**','',
           '两组从原History2048恢复相同53张量、Adam与RNG，固定使用同一曝光序列两遍；主终点2560。'
           '源帧X按FIFO4保存并重新对齐读取，不带v1.2插值门；原PDA/TSFM、TPro和预测头保持。']
    table(lines,['配置','安排','M','U','low匹配','extreme匹配','保护匹配'],
          [[LABEL[r['condition']],r['arrangement'],r['M'],r['unmatched_components'],
            r['low_all_05_matched'],r['extreme_all_001_matched'],r['both_fixed_success_matched']] for r in summary])
    lines += ['', 'M/U仍为诊断匹配/未匹配组件。共同U上限4015；发布S为3967，预算通过不代表相对发布S零代价。']
    table(lines,['比较','集合','O恢复/丢失','S恢复/丢失','O/S去重恢复/丢失'],
          [[r['comparison'],r['cohort'],f"{r['R_hist_O']}/{r['L_hist_O']}",f"{r['R_hist_S']}/{r['L_hist_S']}",
            f"{r['R_hist_unique_either_arrangement']}/{r['L_hist_unique_either_arrangement']}"] for r in effects])
    table(lines,['History相对','安排','strict G/L','全部有效GT G/L','ΔU','Δ未匹配面积'],
          [[LABEL[r['reference']],r['arrangement'],f"{r['strict_G']}/{r['strict_L']}",
            f"{r['all_valid_G']}/{r['all_valid_L']}",r['delta_U'],r['delta_unmatched_area']]
           for r in comparisons if r['condition']==H])
    table(lines,['配置','集合','对象数','O匹配','S匹配','两安排均匹配','两安排均漏检'],
          [[LABEL[r['condition']],r['cohort'],r['target_frames'],r['O_matched'],r['S_matched'],
            r['both_matched'],r['neither_matched']] for r in protected if r['condition'] in (C,H,HC)])
    table(lines,['比较','安排','覆盖改善/退化轨迹','最长漏检缩短/延长','原零匹配轨迹','转为检出轨迹'],
          [[LABEL[r['reference']],r['arrangement'],f"{r['tracks_with_coverage_gain']}/{r['tracks_with_coverage_loss']}",
            f"{r['tracks_with_shorter_longest_miss']}/{r['tracks_with_longer_longest_miss']}",
            r['reference_zero_match_tracks'],r['reference_zero_match_to_positive_tracks']]
           for r in continuous['summary'] if r['condition']==H and r['reference'] in (C,HC)])
    table(lines,['History相对','安排','变化','帧数','涉及轨迹','单轨最高占比','前5轨占比'],
          [[LABEL[r['reference']],r['arrangement'],r['outcome'],r['frames'],r['tracks'],
            fmt(r['top1_fraction']),fmt(r['top5_fraction'])]
           for r in continuous['recovery_loss_concentration'] if r['condition']==H and r['reference'] in (C,HC)])
    lines += ['', '逐对象变化、逐轨迹覆盖和连续漏检保存在 [评价目录](evaluation/step_2560/)。'
              '主连续区间沿冻结严格有效标注，不将low/extreme筛选造成的断点当作不可见。']
    table(lines,['同History权重，low组','源均值相对差','PDA后相对差','平均读出相对差','ΔH RMS','Δlogits RMS','固定点Δlogit'],
          [[r['kind']]+[fmt(r[m]['mean']) for m in ('before_pda_relative_delta_rms','after_pda_relative_delta_rms',
           'after_tsfm_relative_delta_rms','injected_H_delta_rms','logit_delta_rms','delta_z_anchor')]
           for r in mechanism['same_weight_effects'] if r['arm']=='BankHistory' and r['cohort']=='low_all_05'])
    lines += ['', '上述差分均为同一History权重的history−current_only；PDA/TSFM阶段是各来源结果的均值。'
              '阶段幅度下降只描述当前权重下的作用，不能单独证明信息消失或唯一失败原因。'
              '固定困难背景与目标分别汇总；正margin变化也须同时检查目标logit是否提高。']
    table(lines,['训练路径','追加步','目标logit','背景logit','margin'],
          [[LABEL[r['condition']],r['additional_step'],fmt(r['target_logit_mean']),fmt(r['background_logit_mean']),fmt(r['logit_margin_mean'])]
           for r in calibration['summary']])
    table(lines,['训练臂','512步秒','峰值allocated GiB','总/可训参数'],
          [[arm,fmt(c['elapsed_seconds']),fmt(c['peak_cuda_allocated_GiB']),f"{c['total_parameters']}/{c['trainable_parameters']}"]
           for arm,c in cost['training'].items()])
    lines += ['', '两组都执行每窗口150对PDA/TSFM，旧递归39对；参数不增加不等于同计算预算。'
              '完整分辨率实测使用全图M1，只在原冻结前端切块。CPU29项核心测试、真实CUDA状态/梯度/一致性检查及最终训练快照核验均通过。', '',
              '来源边界见 [PROVENANCE.md](PROVENANCE.md)：沿用RFR空间关联/融合，借鉴StreamVGGT保存与读取分离；'
              '原X缓存、FIFO4和均值为本项目待验证适配。缓存本身不作新颖性主张。', '',
              '本轮不自动追加训练、调整FIFO或增加M2。完整可核对数据见 [RESULTS.json](RESULTS.json)。']
    (E/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (E/'RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print({'status':'COMPLETE','decision':decision['code']})


if __name__=='__main__':main()
