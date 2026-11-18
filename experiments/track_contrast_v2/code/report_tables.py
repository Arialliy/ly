#!/usr/bin/env python3
"""Audit and report five-condition C-v2 direction evidence at global step2304.

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
CONDITIONS = ('release', 'C_init', 'CONT', 'FRAME', 'GAP')
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


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
                     + ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def validate_inputs():
    receipt = read_json(E / 'evaluation/SUMMARY_COMPLETE.json')
    require(receipt['status'] == 'COMPLETE' and receipt['conditions'] == list(CONDITIONS),
            'All five conditions must complete before reporting')
    require(receipt['fixed_main_step'] == 2304 and receipt['initial_step'] == 2048
            and receipt['additional_steps'] == 256 and receipt['threshold'] == .5 and receipt['plans'] == PLANS,
            'Fixed inference protocol differs')
    for path, expected in receipt['source_sha256'].items():
        require(sha(path) == expected, 'Changed summary source: ' + path)
    for name, expected in receipt['output_sha256'].items():
        require(sha(E / 'evaluation' / name) == expected, 'Changed summary table: ' + name)
    freeze = read_json(E / 'PROTOCOL_FREEZE.json')
    for name, expected in freeze['files'].items():
        require(sha(E / name) == expected, 'Changed frozen file: ' + name)
    runs, exposures = {}, {}
    for condition in CONDITIONS[2:]:
        folder = E / 'training' / condition / 'run'
        run = read_json(folder / 'RUN.json')
        require(run['status'] == 'complete' and run['condition'] == condition and run['seed'] == 46,
                'Incomplete training condition: ' + condition)
        require(run['start_step'] == 2048 and run['final_step'] == 2304 and run['additional_steps'] == 256
                and run['initial_optimizer_restored'] and run['frozen_frontend_unchanged'] and run['bn_unchanged'],
                'Continuation/freeze invariants failed: ' + condition)
        require(run['protocol_sha256'] == sha(E / 'PROTOCOL_FREEZE.json'), 'Run protocol differs')
        final = one(run['snapshots'], step=2304)
        require(final['sha256'] == sha(folder / 'snapshot_2304.pth'), 'Main checkpoint hash differs')
        logs = [json.loads(line) for line in (folder / 'TRAIN.jsonl').read_text().splitlines()]
        require(len(logs) == 256, 'Continuation log length differs')
        exposures[condition] = []
        for index, row in enumerate(logs):
            require(row['step'] == 2049 + index and row['additional_step'] == index + 1
                    and row['plan_index'] == index and row['condition'] == condition, 'Continuation schedule differs')
            require(math.isfinite(row['loss']) and math.isfinite(row['grad_norm']), 'Nonfinite continuation update')
            exposures[condition].append(tuple(row[k] for k in ('input_sha256', 'label_sha256', 'auxiliary_input_sha256')))
        runs[condition] = run
    for c in ('FRAME', 'GAP'):
        for key in ('initial_checkpoint_sha256', 'release_checkpoint_sha256', 'model_source_sha256',
                    'protocol_sha256', 'trainable_parameter_names', 'initial_optimizer_sha256',
                    'initial_state_sha256', 'settings', 'optimizer'):
            require(runs[c][key] == runs['CONT'][key], 'Common continuation identity differs: ' + key)
        require(exposures[c] == exposures['CONT'], 'Paired continuation exposures differ')
    return receipt, runs


def isolation_audit(data):
    objects = read_csv(E / 'evaluation/OBJECT_COMPARISON.csv')
    summary = read_csv(E / 'evaluation/COVERAGE_AND_ISOLATION_SUMMARY.csv')
    groups = collections.defaultdict(list)
    for row in objects:
        groups[row['arrangement'], row['sequence'], row['track_id']].append(row)
    iso = {}
    for key, rows in groups.items():
        rows.sort(key=lambda r: (num(r, 'frame'), num(r, 'source_frame')))
        eligible = []
        for i in range(1, len(rows) - 1):
            triplet = rows[i-1:i+2]
            if all(num(b, 'frame') - num(a, 'frame') == num(b, 'source_frame') - num(a, 'source_frame') == 1
                   for a, b in zip(triplet, triplet[1:])):
                eligible.append(i)
        for c in CONDITIONS:
            hits = {i for i, row in enumerate(rows) if num(row, c + '_matched')}
            isolated = {i for i in eligible if i in hits and i-1 not in hits and i+1 not in hits}
            iso[c, key] = dict(eligible=len(eligible), isolated=len(isolated), matched=len(hits),
                boundary=len(hits - set(eligible)), multi=sum(1 for i in hits if i + 1 in hits and i+1 in eligible))
    data['isolation'] = {}
    for c in CONDITIONS:
        for p in PLANS:
            chosen = [v for (condition, key), v in iso.items() if condition == c and key[0] == p]
            row = one(summary, condition=c, arrangement=p, group='pooled')
            require(num(row, 'tracks') == len(chosen) == 130, 'Coverage track count differs')
            for field, key in [('isolated010_hits', 'isolated'), ('eligible_center_frames', 'eligible'),
                               ('matched_frames', 'matched'), ('boundary_or_gap_matched_frames', 'boundary')]:
                require(num(row, field) == sum(r[key] for r in chosen), 'Isolation/coverage aggregate mismatch: ' + field)
            require(num(row, 'matched_frames') == num(data['core'][c, p], 'M'), 'Coverage/core count differs')
            data['isolation'][c, p] = row
    return objects


def direction_evidence(data, objects):
    comparisons = []
    for c, references in [('CONT', ('release', 'C_init')), ('FRAME', ('release', 'C_init', 'CONT')),
                          ('GAP', ('release', 'C_init', 'CONT', 'FRAME'))]:
        for ref in references:
            for plan in PLANS:
                current, before = data['core'][c, plan], data['core'][ref, plan]
                continuity = data['continuity'][c, plan]
                isolated = data['isolation'][c, plan]
                row = dict(condition=c, reference=ref, arrangement=plan,
                    gain=num(current, 'G_vs_' + ref + '_same'), loss=num(current, 'L_vs_' + ref + '_same'),
                    net=num(current, 'M') - num(before, 'M'))
                for cohort, label in [('low_all_05', 'low'), ('extreme_all_001', 'extreme')]:
                    transition = data['transitions'][c, ref, plan, cohort]
                    row.update({label + '_' + key: num(transition, key) for key in ['gain', 'loss', 'net', 'gain_tracks', 'gain_clips']})
                for key, label in [('unmatched_components', 'U'), ('positive_pixels_outside_boxes', 'outside_pixels'),
                                   ('unmatched_component_area_sum', 'unmatched_area')]:
                    row['delta_' + label] = num(current, key) - num(before, key)
                row.update(U=num(current, 'unmatched_components'), U_le_4015=num(current, 'unmatched_components') <= 4015,
                    longest_miss_mean_delta=float(continuity['longest_delta_track_mean_vs_' + ref]),
                    tracks_longest_miss_shorter=num(continuity, 'tracks_shorter_vs_' + ref),
                    tracks_longest_miss_longer=num(continuity, 'tracks_longer_vs_' + ref),
                    coverage_tracks_up=num(isolated, 'tracks_coverage_up_vs_' + ref),
                    coverage_tracks_down=num(isolated, 'tracks_coverage_down_vs_' + ref),
                    isolated010_gain=num(isolated, 'isolated010_gain_vs_' + ref),
                    isolated010_delta=num(isolated, 'isolated010_delta_vs_' + ref),
                    nonstrict_net=num(data['other_gt'][c, plan], 'M') - num(data['other_gt'][ref, plan], 'M'),
                    protected646_dual_delta=num(data['protection'][c], 'both_matched') - num(data['protection'][ref], 'both_matched'))
                comparisons.append(row)
    overlap = []
    for ref in ('C_init', 'CONT'):
        for p in PLANS:
            for cohort in ('all', 'low_all_05', 'extreme_all_001'):
                selected = [r for r in objects if r['arrangement'] == p and (cohort == 'all' or num(r, cohort))]
                gain = {c: {(r['sequence'], r['annotation_id']) for r in selected
                            if num(r, c + '_matched') and not num(r, ref + '_matched')} for c in ('FRAME', 'GAP')}
                overlap.append(dict(reference=ref, arrangement=p, cohort=cohort,
                    GAP_gains=len(gain['GAP']), FRAME_gains=len(gain['FRAME']), common_gains=len(gain['GAP'] & gain['FRAME']),
                    GAP_only_gains=len(gain['GAP'] - gain['FRAME']), FRAME_only_gains=len(gain['FRAME'] - gain['GAP']),
                    all_GAP_gains_also_in_FRAME=(gain['GAP'] <= gain['FRAME']) if gain['GAP'] else None))
    return dict(status='COMPLETE', assessment='FIXED_256_STEP_DIRECTION_SCREEN', comparisons=comparisons,
        FRAME_GAP_recovery_overlap=overlap,
        interpretation='No inherited v1 pass/fail gate. Read paired detection gains together with all background costs, '
            'protection losses, nonstrict changes, full-track miss runs and isolated010 gains. Shorter misses alone do not establish sustained detection.',
        FRAME_control='If FRAME accounts for GAP recovery and GAP lacks additional continuity/cost benefit, weaken the temporal-grouping claim. '
            'Matching total counts alone does not imply matching recovered objects; recovery overlap is supplied explicitly.',
        scope='One shared C2048 checkpoint/Adam state plus 256 paired updates, not converged final performance, statistical significance, '
              'repeatability or validation of the grouped mechanism. A promising direction can motivate a registered single-factor revision.')


def render(data, evidence):
    lines = ['# C-v2：固定 256 步修订方向', '',
        '三分支从同一 C2048 权重和 Adam 状态出发，终点为全局 2304。'
        '原 O/S 五窗、0.5 actual-component 匹配与全部固定人群保持一致。M/U 是开发诊断。'
        'release 和 C_init 分别为发布模型与原 C2048，均与原归档评分数组精确核对。', '', '## 核心结果', '']
    lines += [table(['条件', '安排', 'M/5200', 'low/4342', 'extreme/4170', 'U（参考预算4015）', '框外前景', '未匹配面积'],
        [[c, p] + [num(data['core'][c, p], k) for k in ('M', 'low_all_05_matched', 'extreme_all_001_matched',
          'unmatched_components', 'positive_pixels_outside_boxes', 'unmatched_component_area_sum')]
         for c in CONDITIONS for p in PLANS]), '', '## 同帧配对增量', '']
    lines += [table(['条件−参考', '安排', 'strict G/L/净', 'low G/L/净', 'low恢复轨/来源', 'ΔU', 'Δ框外像素', 'Δ非strict'],
        [[r['condition'] + '−' + r['reference'], r['arrangement'], f"{r['gain']}/{r['loss']}/{r['net']}",
          f"{r['low_gain']}/{r['low_loss']}/{r['low_net']}", f"{r['low_gain_tracks']}/{r['low_gain_clips']}",
          r['delta_U'], r['delta_outside_pixels'], r['nonstrict_net']] for r in evidence['comparisons']]),
        '', '## 保护与非 strict', '']
    lines += [table(['条件', '646 O保持', '646 S保持', '646双保持', '失去双成功', '两安排均失', '非strict O/11097', '非strict S/11097'],
        [[c] + [num(data['protection'][c], k) for k in ('O_matched', 'S_matched', 'both_matched', 'lost_dual_success', 'lost_both_arrangements')]
         + [num(data['other_gt'][c, p], 'M') for p in PLANS] for c in CONDITIONS]),
        '', '失去双成功包含只丢失一个安排，不能等同于两个安排均漏检。', '', '## 连续性与逐轨覆盖', '']
    lines += [table(['条件', '安排', '平均最长漏检（130轨）', '最大漏检段', '有检出轨', '有连续≥2帧检出轨', '逐轨平均覆盖', '孤立010数', '边界/断点检出数'],
        [[c, p, f"{float(data['continuity'][c, p]['longest_miss_track_mean']):.6f}",
          num(data['continuity'][c, p], 'longest_miss_max'), num(data['isolation'][c, p], 'tracks_with_any_match'),
          num(data['isolation'][c, p], 'tracks_with_multi_frame_run'), f"{float(data['isolation'][c, p]['coverage_track_mean']):.6f}",
          num(data['isolation'][c, p], 'isolated010_hits'), num(data['isolation'][c, p], 'boundary_or_gap_matched_frames')]
         for c in CONDITIONS for p in PLANS]), '',
        '孤立跳点严格定义为同轨迹连续三个有效观测的 0/1/0，local/source 帧号均须逐一递增。'
        '边界、缺标和帧号断点不算孤立；覆盖率仅针对固定观测帧，不推断未观测物理轨迹。', '']
    lines += [table(['条件−参考', '安排', '最长漏检均值Δ', '缩短/延长轨', '覆盖增加/减少轨', '新增匹配中的010', '010总数Δ'],
        [[r['condition'] + '−' + r['reference'], r['arrangement'], f"{r['longest_miss_mean_delta']:.6f}",
          f"{r['tracks_longest_miss_shorter']}/{r['tracks_longest_miss_longer']}",
          f"{r['coverage_tracks_up']}/{r['coverage_tracks_down']}", r['isolated010_gain'], r['isolated010_delta']]
         for r in evidence['comparisons']]), '', '## FRAME 是否覆盖 GAP 的新增恢复', '']
    lines += [table(['参考', '安排', '人群', 'GAP新增', 'FRAME新增', '共同新增', '仅GAP新增', '仅FRAME新增'],
        [[r[k] for k in ('reference', 'arrangement', 'cohort', 'GAP_gains', 'FRAME_gains', 'common_gains', 'GAP_only_gains', 'FRAME_only_gains')]
         for r in evidence['FRAME_GAP_recovery_overlap']]), '',
        '这里逐目标比较恢复集合，不以相同总数代替集合一致。O/S 必须分别解释，不能直接相加为不同对象帧。', '',
        '这是修订方向表，不套用 v1 的四项停止门槛。应同时看新增恢复、多轨覆盖、保持与全部背景代价；'
        '最长漏检缩短可能来自孤立跳点，不能单独作为持续检测成功。'
        '若 FRAME 已覆盖 GAP 的收益且没有额外连续性或代价改善，应削弱时序分组解释。'
        '即使出现额外方向收益，单次 256 步也不证明收敛或可重复性。完整数据与解释范围见 evaluation/DIRECTION.json。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    receipt, runs = validate_inputs()
    data = audit_tables(E)
    require([(num(data['core']['C_init', p], 'M'), num(data['core']['C_init', p], 'unmatched_components'))
             for p in PLANS] == [(707, 4053), (724, 4002)], 'C_init anchor differs')
    objects = isolation_audit(data)
    evidence = direction_evidence(data, objects)
    evidence.update(conditions=list(CONDITIONS), initial_step=2048, additional_steps=256, final_step=2304,
        report_source_sha256=sha(__file__), protocol_sha256=sha(E / 'PROTOCOL_FREEZE.json'),
        summary_receipt_sha256=sha(E / 'evaluation/SUMMARY_COMPLETE.json'), table_sha256=receipt['output_sha256'],
        run_sha256={c: sha(E / 'training' / c / 'run/RUN.json') for c in runs})
    if not args.check_only:
        atomic_text(E / 'RESULT_TABLES.md', render(data, evidence))
        evidence['result_tables_sha256'] = sha(E / 'RESULT_TABLES.md')
        atomic_text(E / 'evaluation/DIRECTION.json', json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status='COMPLETE', assessment=evidence['assessment'], comparison_rows=len(evidence['comparisons']))))


if __name__ == '__main__':
    main()


