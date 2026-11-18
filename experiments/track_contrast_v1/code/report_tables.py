#!/usr/bin/env python3
"""Audit complete fixed-endpoint tables and render the preregistered decision.

Standard-library only. Refuses partial conditions, changed source/results hashes,
or inconsistent object/track aggregates before writing either final artifact.
This is reporting code: no model loading, thresholds, selection or score edits.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path

E = Path(__file__).resolve().parents[1]
CONDITIONS = ('release', 'A', 'B', 'C')
PLANS = {'O': [0, 36, 72, 108, 120], 'S': [0, 24, 60, 96, 120]}
COHORTS = {'all': 5200, 'low_all_05': 4342, 'extreme_all_001': 4170,
           'both_fixed_success': 646}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_csv(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def num(row, key):
    return int(row[key])


def one(rows, **keys):
    found = [r for r in rows if all(r[k] == v for k, v in keys.items())]
    require(len(found) == 1, f'Expected one row: {keys}; found {len(found)}')
    return found[0]


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12)


def atomic_text(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(value, encoding='utf-8')
    temporary.replace(path)


def validate_inputs(folder):
    output = folder / 'evaluation'
    receipt_path = output / 'SUMMARY_COMPLETE.json'
    receipt = read_json(receipt_path)
    require(receipt['status'] == 'COMPLETE' and receipt['conditions'] == list(CONDITIONS),
            'Wait for complete release/A/B/C evaluate --summary; partial results cannot be reported')
    require(receipt['threshold'] == .5 and receipt['fixed_main_step'] == 2048
            and receipt['plans'] == PLANS, 'Frozen evaluation configuration differs')
    expected = {'strict_small_target_frames': 5200, 'strict_small_tracks': 130,
                'all_valid_target_frames': 16297, 'other_valid_target_frames': 11097}
    require(all(receipt[k] == v for k, v in expected.items()), 'Evaluation populations differ')
    require(receipt['cohorts'] == {k: v for k, v in COHORTS.items() if k != 'all'},
            'Frozen cohorts differ')
    for path, expected_hash in receipt['source_sha256'].items():
        require(sha(path) == expected_hash, f'Summary source changed: {path}')
    for name, expected_hash in receipt['output_sha256'].items():
        require(sha(output / name) == expected_hash, f'Summary table changed: {name}')
    freeze_path = folder / 'PROTOCOL_FREEZE.json'
    freeze = read_json(freeze_path)
    for name, expected_hash in freeze['files'].items():
        require(sha(folder / name) == expected_hash, f'Frozen experiment file changed: {name}')
    runs = {}
    exposure = {}
    evidence = {str(receipt_path): sha(receipt_path), str(freeze_path): sha(freeze_path)}
    for condition in CONDITIONS[1:]:
        run_path = folder / 'training' / condition / 'run/RUN.json'
        run = read_json(run_path)
        require(run['status'] == 'complete' and run['condition'] == condition
                and run['seed'] == 46 and run['final_step'] == run['steps'] == 2048,
                f'Incomplete or wrong endpoint: {condition}')
        require(run['frozen_frontend_unchanged'] and run['bn_unchanged']
                and run['trainable_parameters'] == 14593 and run['new_parameters'] == 0,
                f'Training invariants failed: {condition}')
        require(run['protocol_sha256'] == sha(freeze_path), f'Protocol mismatch: {condition}')
        require([x['step'] for x in run['snapshots']] == [0, 128, 512, 1024, 2048],
                f'Unexpected checkpoint schedule: {condition}')
        for checkpoint in run['snapshots']:
            path = run_path.parent / checkpoint['path']
            require(sha(path) == checkpoint['sha256'], f'Checkpoint hash differs: {path}')
        log_path = run_path.with_name('TRAIN.jsonl')
        logs = [json.loads(line) for line in log_path.read_text().splitlines()]
        require(len(logs) == 2048, f'Training log length differs: {condition}')
        exposure[condition] = []
        for index, row in enumerate(logs):
            require(row['step'] == index + 1 and row['plan_index'] == index % 512
                    and row['condition'] == condition, f'Training schedule differs: {condition}/{index}')
            require(math.isfinite(row['loss']) and math.isfinite(row['grad_norm']),
                    f'Nonfinite training log: {condition}/{index}')
            exposure[condition].append(tuple(row[k] for k in
                ('input_sha256', 'label_sha256', 'auxiliary_input_sha256')))
        require(exposure[condition] == exposure[condition][:512] * 4,
                f'Training exposures do not repeat the frozen 512 entries: {condition}')
        runs[condition] = run
        evidence.update({str(run_path): sha(run_path), str(log_path): sha(log_path)})
    invariant_keys = ['initial_checkpoint_sha256', 'model_source_sha256', 'protocol_sha256',
                      'calibration_sha256', 'training_sources', 'settings', 'optimizer',
                      'initial_state_sha256', 'frozen_state_sha256', 'bn_initial_sha256',
                      'trainable_parameter_names', 'precision']
    for condition in ('B', 'C'):
        require(exposure[condition] == exposure['A'], f'A/{condition} training inputs differ')
        require(all(runs[condition][key] == runs['A'][key] for key in invariant_keys),
                f'A/{condition} common training configuration differs')
    return receipt, evidence


def miss_lengths(rows, condition):
    lengths, current, previous = [], 0, None
    for row in sorted(rows, key=lambda r: (num(r, 'frame'), num(r, 'source_frame'))):
        consecutive = previous is None or (num(row, 'frame') == num(previous, 'frame') + 1
                      and num(row, 'source_frame') == num(previous, 'source_frame') + 1)
        if not consecutive or num(row, condition + '_matched'):
            if current:
                lengths.append(current)
            current = 0
        if not num(row, condition + '_matched'):
            current += 1
        previous = row
    if current:
        lengths.append(current)
    return lengths


def audit_tables(folder):
    output = folder / 'evaluation'
    names = ['SUMMARY', 'COHORT_TRANSITIONS', 'ALL_GT_SUMMARY', 'OBJECT_COMPARISON',
             'ALL_GT_OBJECT_COMPARISON', 'PROTECTED_AND_WINDOW_RETENTION',
             'CONTINUOUS_MISS_SUMMARY', 'CONTINUOUS_MISS_TRACKS']
    tables = {name: read_csv(output / (name + '.csv')) for name in names}
    objects = tables['OBJECT_COMPARISON']
    all_objects = tables['ALL_GT_OBJECT_COMPARISON']
    require(len(objects) == 10400 and len(all_objects) == 32594, 'Object table lengths differ')
    strict_keys = {(r['arrangement'], r['sequence'], r['frame'], r['annotation_id']): r for r in objects}
    require(len(strict_keys) == 10400, 'Duplicate strict object keys')
    all_keys = set()
    for row in all_objects:
        key = tuple(row[k] for k in ('arrangement', 'sequence', 'frame', 'annotation_id'))
        require(key not in all_keys, 'Duplicate all-GT object key')
        all_keys.add(key)
        require(bool(num(row, 'strict_small')) == (key in strict_keys), 'Strict population join differs')
        if key in strict_keys:
            require(all(row[c + '_matched'] == strict_keys[key][c + '_matched'] for c in CONDITIONS),
                    'Strict and all-GT match decisions disagree')
    result = {'core': {}, 'transitions': {}, 'other_gt': {}, 'continuity': {},
              'protection': {}, 'recoveries': {}, 'longest_sums': {}}
    for plan in PLANS:
        selected = [r for r in objects if r['arrangement'] == plan]
        require(len(selected) == 5200, 'Wrong strict denominator')
        for cohort, count in COHORTS.items():
            subset = selected if cohort == 'all' else [r for r in selected if num(r, cohort)]
            require(len(subset) == count, f'Wrong cohort size: {plan}/{cohort}')
            for condition in CONDITIONS:
                for reference in CONDITIONS:
                    transition = one(tables['COHORT_TRANSITIONS'], condition=condition,
                        reference=reference, arrangement=plan, group='pooled', cohort=cohort)
                    gains = [r for r in subset if num(r, condition + '_matched') and not num(r, reference + '_matched')]
                    losses = [r for r in subset if not num(r, condition + '_matched') and num(r, reference + '_matched')]
                    expected = {'target_frames': count,
                        'reference_matched': sum(num(r, reference + '_matched') for r in subset),
                        'matched': sum(num(r, condition + '_matched') for r in subset),
                        'gain': len(gains), 'loss': len(losses), 'net': len(gains) - len(losses),
                        'gain_tracks': len({(r['sequence'], r['track_id']) for r in gains}),
                        'gain_clips': len({r['sequence'] for r in gains})}
                    require(all(num(transition, k) == v for k, v in expected.items()),
                            f'Object/transition mismatch: {condition}/{reference}/{plan}/{cohort}')
                    result['transitions'][condition, reference, plan, cohort] = transition
                    if reference == 'release' and cohort in ('low_all_05', 'extreme_all_001'):
                        counts = collections.Counter((r['sequence'], r['track_id']) for r in gains)
                        ranked = [dict(sequence=s, track_id=t, frames=n)
                                  for (s, t), n in sorted(counts.items(), key=lambda x: (-x[1], x[0]))]
                        result['recoveries'][condition, plan, cohort] = dict(frames=len(gains),
                            tracks=len(counts), clips=len({s for s, _ in counts}), ranked_tracks=ranked)
        grouped = collections.defaultdict(list)
        for row in selected:
            grouped[row['sequence'], row['track_id']].append(row)
        require(len(grouped) == 130, 'Wrong strict track denominator')
        for condition in CONDITIONS:
            core = one(tables['SUMMARY'], condition=condition, arrangement=plan, group='pooled')
            require(num(core, 'small_gt') == 5200 and num(core, 'small_tracks') == 130,
                    f'Wrong core populations: {condition}/{plan}')
            strict_m = sum(num(r, condition + '_matched') for r in selected)
            require(num(core, 'M') == strict_m, 'Object/core matched count mismatch')
            for reference in CONDITIONS:
                transition = result['transitions'][condition, reference, plan, 'all']
                require(num(core, 'G_vs_' + reference + '_same') == num(transition, 'gain')
                        and num(core, 'L_vs_' + reference + '_same') == num(transition, 'loss')
                        and num(core, 'net_vs_' + reference + '_same') == num(transition, 'net'),
                        'Core/transition paired counts differ')
            require(num(core, 'component_count') == num(core, 'matched_count') + num(core, 'unmatched_components'),
                    'Component accounting mismatch')
            for cohort in COHORTS:
                if cohort != 'all':
                    count = sum(num(r, condition + '_matched') for r in selected if num(r, cohort))
                    require(num(core, cohort + '_matched') == count, 'Core cohort count mismatch')
            all_selected = [r for r in all_objects if r['arrangement'] == plan]
            for scope, population in [('all_valid_gt', all_selected), ('other_valid_gt',
                    [r for r in all_selected if not num(r, 'strict_small')])]:
                row = one(tables['ALL_GT_SUMMARY'], condition=condition, arrangement=plan, group='pooled', scope=scope)
                require(num(row, 'target_frames') == len(population), 'All-GT denominator mismatch')
                require(num(row, 'M') == sum(num(r, condition + '_matched') for r in population), 'All-GT M mismatch')
                for reference in CONDITIONS:
                    gains = sum(num(r, condition + '_matched') and not num(r, reference + '_matched') for r in population)
                    losses = sum(not num(r, condition + '_matched') and num(r, reference + '_matched') for r in population)
                    require(num(row, 'G_vs_' + reference) == gains and num(row, 'L_vs_' + reference) == losses,
                            'All-GT paired count mismatch')
                if scope == 'other_valid_gt':
                    result['other_gt'][condition, plan] = row
                else:
                    require(num(core, 'matched_count') == num(row, 'M'), 'Core/all-GT count mismatch')
            longest = {}
            for (sequence, track_id), rows in grouped.items():
                lengths = miss_lengths(rows, condition)
                row = one(tables['CONTINUOUS_MISS_TRACKS'], condition=condition, arrangement=plan,
                          cohort='all', sequence=sequence, track_id=track_id)
                require(json.loads(row['miss_run_lengths']) == lengths and num(row, 'missed_frames') == sum(lengths)
                        and num(row, 'longest_miss_run') == max(lengths, default=0), 'Continuous miss reconstruction differs')
                longest[sequence, track_id] = max(lengths, default=0)
            continuity = one(tables['CONTINUOUS_MISS_SUMMARY'], condition=condition,
                             arrangement=plan, group='pooled', cohort='all')
            require(num(continuity, 'missed_frames') == 5200 - strict_m and num(continuity, 'tracks') == 130
                    and close(continuity['longest_miss_track_mean'], sum(longest.values()) / 130),
                    'Continuity aggregate mismatch')
            for reference in CONDITIONS:
                deltas = [length - max(miss_lengths(grouped[key], reference), default=0)
                          for key, length in longest.items()]
                require(num(continuity, 'tracks_shorter_vs_' + reference) == sum(d < 0 for d in deltas)
                        and num(continuity, 'tracks_same_vs_' + reference) == sum(d == 0 for d in deltas)
                        and num(continuity, 'tracks_longer_vs_' + reference) == sum(d > 0 for d in deltas)
                        and close(continuity['longest_delta_track_mean_vs_' + reference], sum(deltas) / 130),
                        'Continuity paired track counts differ')
            result['core'][condition, plan] = core
            result['continuity'][condition, plan] = continuity
            result['longest_sums'][condition, plan] = sum(longest.values())
    for condition in CONDITIONS:
        o = {key[1:]: r for key, r in strict_keys.items() if key[0] == 'O' and num(r, 'both_fixed_success')}
        s = {key[1:]: r for key, r in strict_keys.items() if key[0] == 'S' and num(r, 'both_fixed_success')}
        require(o.keys() == s.keys() and len(o) == 646, 'Protection set differs between arrangements')
        row = one(tables['PROTECTED_AND_WINDOW_RETENTION'], condition=condition, cohort='both_fixed_success')
        joint = collections.Counter((num(o[k], condition + '_matched'), num(s[k], condition + '_matched')) for k in o)
        expected = dict(target_frames=646, both_matched=joint[1, 1], O_only=joint[1, 0],
                        S_only=joint[0, 1], neither_matched=joint[0, 0], lost_dual_success=646-joint[1, 1],
                        lost_both_arrangements=joint[0, 0], O_matched=joint[1, 1]+joint[1, 0],
                        S_matched=joint[1, 1]+joint[0, 1])
        require(all(num(row, k) == v for k, v in expected.items()), 'Protection aggregate mismatch')
        result['protection'][condition] = row
    require([(num(result['core']['release', p], 'M'), num(result['core']['release', p], 'unmatched_components'))
             for p in PLANS] == [(700, 4015), (720, 3967)], 'Release anchors differ')
    return result


def decide(data):
    gates = []
    def gate(name, label, details):
        gates.append(dict(id=name, criterion=label, passed=all(v['passed'] for v in details.values()), details=details))
    gate('background_budget', 'O/S 未匹配组件 U 均 ≤4015',
         {p: dict(observed=num(data['core']['C', p], 'unmatched_components'), maximum=4015,
                  passed=num(data['core']['C', p], 'unmatched_components') <= 4015) for p in PLANS})
    protected = num(data['protection']['C'], 'both_matched')
    gate('protected646', '646 原共同成功帧在 O/S 全部保持',
         {'joint': dict(observed=protected, required=646, passed=protected == 646)})
    for identifier, label, measure in [
        ('other_gt_retention', 'O/S 非 strict 匹配数均不低于 release',
         lambda p: dict(observed=num(data['other_gt']['C', p], 'M'), reference=num(data['other_gt']['release', p], 'M'),
                        passed=num(data['other_gt']['C', p], 'M') >= num(data['other_gt']['release', p], 'M'))),
        ('low_multitrack', 'O/S 相对 release 的 low 恢复均涉及至少 2 条轨迹',
         lambda p: dict(observed=num(data['transitions']['C', 'release', p, 'low_all_05'], 'gain_tracks'), minimum=2,
                        passed=num(data['transitions']['C', 'release', p, 'low_all_05'], 'gain_tracks') >= 2)),
        ('low_net_positive', 'O/S low 相对 release 的净增量均 >0',
         lambda p: dict(observed=num(data['transitions']['C', 'release', p, 'low_all_05'], 'net'),
                        passed=num(data['transitions']['C', 'release', p, 'low_all_05'], 'net') > 0)),
        ('continuous_miss', 'O/S 全部 130 轨平均最长连续漏检均不增加',
         lambda p: dict(observed_sum=data['longest_sums']['C', p], reference_sum=data['longest_sums']['release', p], tracks=130,
                        passed=data['longest_sums']['C', p] <= data['longest_sums']['release', p]))]:
        gate(identifier, label, {p: measure(p) for p in PLANS})
    for identifier, cohort, label in [('strict_beats_controls', 'all', 'O/S strict M 均严格优于 A 和 B'),
                                      ('low_beats_controls', 'low_all_05', 'O/S low 净恢复均严格优于 A 和 B')]:
        gate(identifier, label, {p + '_vs_' + ref: dict(net=num(data['transitions']['C', ref, p, cohort], 'net'),
            passed=num(data['transitions']['C', ref, p, cohort], 'net') > 0) for p in PLANS for ref in ('A', 'B')})
    passed = all(g['passed'] for g in gates)
    return dict(passed=passed,
        decision='ADVANCE_PAIRED_MULTISEED_AND_NEW_SOURCES' if passed else 'DO_NOT_ADVANCE_CURRENT_SETTING',
        gates=gates, failed_gates=[g['id'] for g in gates if not g['passed']],
        application='Each arrangement must pass each applicable criterion; no O/S outcome selection. '
                    'Mean longest miss uses exact integer totals across the same 130 tracks. '
                    'This gate is applied only to C and does not invent a separate gate for A/B.',
        scope='Single-seed, fixed 2048-step development screening; neither statistical significance nor convergence is established.')


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
                     + ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def render(data, decision):
    lines = ['# 固定 2048 终点结果核对', '',
        '同一发布权重起点、seed 46、原模型/输入/forward、O/S 各 5 窗 max、0.5 actual-component 匹配。'
        '每个安排为 5200 strict 帧、130 轨迹、4342 low、4170 extreme；另有 11097 非 strict 帧。'
        '以下 M/U 为开发诊断，不是官方 Pd/Fa。G/L 为同帧相对参考的得/失。', '', '## 核心表', '']
    core_rows = []
    for condition in CONDITIONS:
        for plan in PLANS:
            r = data['core'][condition, plan]
            core_rows.append([condition, plan] + [num(r, k) for k in ('M', 'G_vs_release_same', 'L_vs_release_same',
                'unmatched_components', 'low_all_05_matched', 'extreme_all_001_matched', 'positive_pixels_outside_boxes',
                'unmatched_component_area_sum')])
    lines += [table(['条件', '安排', 'M/5200', 'G', 'L', 'U≤4015', 'low/4342', 'extreme/4170', '框外前景像素', '未匹配组件面积'], core_rows),
              '', '## C 的配对增量', '']
    paired = []
    for plan in PLANS:
        for reference in ('release', 'A', 'B'):
            strict = data['transitions']['C', reference, plan, 'all']
            low = data['transitions']['C', reference, plan, 'low_all_05']
            other = data['other_gt']['C', plan]
            paired.append([plan, reference, '/'.join(str(num(strict, k)) for k in ('gain', 'loss', 'net')),
                '/'.join(str(num(low, k)) for k in ('gain', 'loss', 'net')),
                num(data['core']['C', plan], 'unmatched_components') - num(data['core'][reference, plan], 'unmatched_components'),
                '/'.join(str(v) for v in (num(other, 'G_vs_' + reference), num(other, 'L_vs_' + reference),
                    num(other, 'M') - num(data['other_gt'][reference, plan], 'M')))])
    lines += [table(['安排', '参考', 'strict G/L/净', 'low G/L/净', 'ΔU', '非 strict G/L/净'], paired),
              '', '## 646 共同成功保护帧', '']
    lines += [table(['条件', 'O保持', 'S保持', '双安排保持', '失去双安排成功', '两安排均失'],
        [[c] + [num(data['protection'][c], k) for k in ('O_matched', 'S_matched', 'both_matched', 'lost_dual_success', 'lost_both_arrangements')]
         for c in CONDITIONS]), '', '“失去双安排成功”包括只丢失一个安排；不能解读为两个安排都漏检。', '', '## 非 strict 帧', '']
    lines += [table(['条件', '安排', 'M/11097', 'G vs release', 'L vs release', '净增量'],
        [[c, p, num(data['other_gt'][c, p], 'M'), num(data['other_gt'][c, p], 'G_vs_release'),
          num(data['other_gt'][c, p], 'L_vs_release'), num(data['other_gt'][c, p], 'M') - num(data['other_gt']['release', p], 'M')]
         for c in CONDITIONS for p in PLANS]), '', '## 连续漏检（完整 130 轨，不按新恢复筛选）', '']
    continuous = []
    for c in CONDITIONS:
        for p in PLANS:
            r = data['continuity'][c, p]
            continuous.append([c, p, num(r, 'missed_frames'), f"{float(r['longest_miss_track_mean']):.6f}",
                num(r, 'longest_miss_max'), num(r, 'tracks_shorter_vs_release'), num(r, 'tracks_same_vs_release'),
                num(r, 'tracks_longer_vs_release')])
    lines += [table(['条件', '安排', '漏检帧', '平均最长漏检', '最大漏检段', '缩短轨', '不变轨', '延长轨'], continuous),
        '', '帧号或原始源帧号不连续就断开；仅统计固定观测范围，不把缺失标注补成连续轨迹。'
        '判定使用最长漏检的整数总和，表内均值保留六位小数。', '', '## low/extreme 恢复分布（相对 release）', '']
    recoveries = []
    for c in CONDITIONS[1:]:
        for p in PLANS:
            lo = data['recoveries'][c, p, 'low_all_05']
            ex = data['recoveries'][c, p, 'extreme_all_001']
            ranked = lo['ranked_tracks']
            top1 = ranked[0]['frames'] if ranked else 0
            examples = '；'.join(f"{r['sequence']}/{r['track_id']}:{r['frames']}" for r in ranked[:5]) or '—'
            recoveries.append([c, p, f"{lo['frames']}/{lo['tracks']}/{lo['clips']}",
                f"{ex['frames']}/{ex['tracks']}/{ex['clips']}", f"{top1}/{lo['frames']}" if lo['frames'] else '—', examples])
    lines += [table(['条件', '安排', 'low 帧/轨/来源', 'extreme 帧/轨/来源', 'low top1帧占比', 'low 前5轨（帧数）'], recoveries),
        '', '完整按轨排序明细见 evaluation/DECISION.json；extreme 是 low 的子集，不能相加为恢复总数。',
        '', '## 预登记判定', '',
        table(['门槛', '通过'], [[g['criterion'], '是' if g['passed'] else '否'] for g in decision['gates']]), '',
        ('全部门槛通过：仅支持进入成对多 seed 与新来源验证。' if decision['passed'] else
         '未通过全部预登记门槛：本轮固定设置不进入下一轮。不得通过选择安排、阈值或 checkpoint 改写判定。'), '',
        '单 seed、固定 2048 步不证明统计显著性或充分收敛。B/C 控制端点多重集合，但实际逐对时间间隔及跨来源关系不同；'
        '标注连续不证明无遮挡。困难背景取自全图分布采样后的裁块，不是穷举全图最大值。'
        '有效共同配对在背景筛选前仅覆盖 256/512 批、84 身份和 6 个训练来源，98.4% 对集中于 3 个来源。', '',
        '报告在所有终点完成后核验输入哈希、2048 步训练曝光一致性、CSV 哈希、逐目标配对得失、保护集合及连续漏检重建；'
        '该报告脚本不改变评分或预登记阈值。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='Verify complete results without writing final artifacts')
    args = parser.parse_args()
    receipt, evidence = validate_inputs(E)
    data = audit_tables(E)
    decision = decide(data)
    decision.update(status='COMPLETE', conditions=list(CONDITIONS), fixed_main_step=2048, threshold=.5,
        protocol_sha256=sha(E / 'PROTOCOL_FREEZE.json'), report_source_sha256=sha(__file__),
        summary_receipt_sha256=sha(E / 'evaluation/SUMMARY_COMPLETE.json'), evidence_sha256=evidence,
        table_sha256=receipt['output_sha256'],
        low_extreme_recoveries=[dict(condition=c, arrangement=p, cohort=cohort, **v)
            for (c, p, cohort), v in data['recoveries'].items()],
        audit=['Complete four-condition evaluation receipts and hashes',
               'Same 2048-step exposure hashes and common configuration across A/B/C',
               'Object-level strict/all-GT/low/extreme/protection paired aggregates',
               'Independent continuous-miss reconstruction from object outcomes'])
    if not args.check_only:
        report = E / 'RESULT_TABLES.md'
        atomic_text(report, render(data, decision))
        decision['result_tables_sha256'] = sha(report)
        atomic_text(E / 'evaluation/DECISION.json', json.dumps(decision, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: decision[k] for k in ('status', 'decision', 'passed', 'failed_gates')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
