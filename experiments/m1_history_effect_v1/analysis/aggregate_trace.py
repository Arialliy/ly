"""Aggregate completed same-checkpoint traces with target-frame equal weighting.

Reads immutable trace/ROI receipts. It never changes predictions, cohorts or
model code. Overlap-window means precede arrangement means; the pooled_unique
view gives one vote to each target frame, not to each O/S occurrence.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

E = Path(__file__).resolve().parents[1]
SEQUENCES = ('DJI_0026_2', 'DJI_0067_2')
PLANS = {'O': [0, 36, 72, 108, 120], 'S': [0, 24, 60, 96, 120]}
STARTS = sorted(set(sum(PLANS.values(), [])))
GROUPS = ('strict_all', 'low_all_05', 'extreme_all_001', 'both_fixed_success', 'm1_low_common_recovered')
STAGES = ('before_pda', 'after_pda', 'after_tsfm', 'residual', 'injected_H', 'head_conv1', 'head_bn', 'head_relu', 'logit')
MODES = ('History', 'HistoryCurrent', 'HistoryOff')
PHASES = ('all_offsets', 't0', 't1_39')
CHECKPOINT_SHA = 'eda063496757f368d7442a7fac99a64f7b33ef0414ef99d006716d77a6dcb725'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''): digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8', newline='') as stream:
        yield from csv.DictReader(stream)


def write_csv(path, records):
    assert records
    keys = list(dict.fromkeys(key for row in records for key in row))
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader()
        for row in records:
            writer.writerow({key: json.dumps(value, separators=(',', ':')) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def number(value):
    if value in ('', None): return None
    if value is True or value == 'True': return 1.
    if value is False or value == 'False': return 0.
    result = float(value)
    assert math.isfinite(result), value
    return result


def quantile(values, q):
    if not values: return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(position); high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


class Means:
    def __init__(self):
        self.data = {}

    def add(self, key, metadata, values, window_start=None):
        if key not in self.data:
            self.data[key] = [dict(metadata), defaultdict(float), defaultdict(int), set()]
        info, sums, counts, windows = self.data[key]
        assert info == metadata, (info, metadata)
        if window_start is not None:
            assert window_start not in windows, ('Duplicate same-object window', key, window_start)
            windows.add(window_start)
        for metric, value in values.items():
            if value is not None:
                sums[metric] += value; counts[metric] += 1

    def finish(self):
        result = []
        for _, (info, sums, counts, windows) in sorted(self.data.items()):
            result.append(dict(info, contributing_windows=len(windows),
                               metric_valid_windows=dict(counts),
                               **{metric: value / counts[metric] for metric, value in sums.items()}))
        return result


def roi_metrics(row):
    selected = {}
    for key, value in row.items():
        is_stage = any(key.startswith(stage + '_') for stage in STAGES)
        is_anchor = any(key.startswith(mode + '_anchor_') for mode in MODES)
        is_logit = any(key in (mode + '_z_mean', mode + '_z_max', mode + '_z_min') for mode in MODES)
        if is_stage or is_anchor or is_logit or key in (
            'delta_z_mean', 'delta_z_max', 'delta_z_min', 'delta_z_absmax', 'delta_z_anchor',
            'history_current_pixel_flips', 'peak_coordinates_changed', 'anchor_deltaH_l2', 'anchor_relu_flips'):
            selected[key] = number(value)
    for name in ('delta_z_anchor', 'delta_z_mean'):
        selected[name + '_absolute'] = abs(selected[name]) if selected.get(name) is not None else None
    selected['history_current_pixel_flip_fraction'] = number(row['history_current_pixel_flips']) / int(row['roi_pixels'])
    selected['anchor_relu_changed'] = float(int(row['anchor_relu_flips']) > 0)
    for mode in MODES:
        for suffix in ('anchor_J_dot_deltaH', 'anchor_linearization_error'):
            name = mode + '_' + suffix
            selected[name + '_absolute'] = abs(selected[name]) if selected.get(name) is not None else None
        selected[mode + '_anchor_zero_jacobian'] = float(selected[mode + '_anchor_jacobian_l2'] == 0)
    return selected


def pair_margins(target, background):
    result = {}
    for mode in MODES:
        for location, suffix in (('anchor', 'anchor_z'), ('peak', 'z_max')):
            result[f'{mode}_{location}_margin'] = number(target[f'{mode}_{suffix}']) - number(background[f'{mode}_{suffix}'])
    for location in ('anchor', 'peak'):
        for reference, label in (('HistoryCurrent', 'history_current'), ('HistoryOff', 'history_off')):
            difference = result[f'History_{location}_margin'] - result[f'{reference}_{location}_margin']
            result[f'{label}_{location}_margin_delta'] = difference
            result[f'{label}_{location}_margin_delta_absolute'] = abs(difference)
        result[f'current_off_{location}_margin_delta'] = result[f'HistoryCurrent_{location}_margin'] - result[f'HistoryOff_{location}_margin']
    result['target_delta_z_anchor'] = number(target['delta_z_anchor'])
    result['background_delta_z_anchor'] = number(background['delta_z_anchor'])
    assert math.isclose(result['history_current_anchor_margin_delta'],
                        result['target_delta_z_anchor'] - result['background_delta_z_anchor'], abs_tol=1e-12)
    return result


METADATA = {'sequence', 'frame', 'source_frame', 'annotation_id', 'track_id', 'object_id', 'arrangement',
            'kind', 'phase', 'offset', 'groups', 'contributing_windows', 'metric_valid_windows', 'arrangements_present'}


def pool_arrangements(records):
    buckets = defaultdict(list)
    for row in records:
        key = (row['object_id'], row['kind'], row.get('phase', ''), row.get('offset', -1))
        buckets[key].append(row)
    output = []
    for _, values in sorted(buckets.items()):
        assert len(values) <= 2 and len({row['arrangement'] for row in values}) == len(values)
        info = {key: value for key, value in values[0].items() if key in METADATA
                and key not in ('metric_valid_windows', 'contributing_windows', 'arrangements_present')}
        info['arrangement'] = 'pooled_unique'
        info['arrangements_present'] = [row['arrangement'] for row in values]
        info['contributing_windows'] = sum(row['contributing_windows'] for row in values)
        metrics = set().union(*(set(row) - METADATA for row in values))
        for metric in metrics:
            known = [row[metric] for row in values if row.get(metric) is not None]
            info[metric] = statistics.fmean(known) if known else None
        output.append(info)
    return output


def summarize(records, metric_filter=None, time_offsets=False):
    grouped = defaultdict(list)
    for row in records:
        for group in row['groups']:
            assert group in GROUPS
            grouped[(row['arrangement'], row.get('phase', 'offset_specific'), row['kind'], group,
                     row.get('offset', -1))].append(row)
    summary = []
    for (arrangement, phase, kind, group, offset), values in sorted(grouped.items()):
        assert len({row['object_id'] for row in values}) == len(values)
        metrics = set().union(*(set(row) - METADATA for row in values))
        if metric_filter is not None: metrics.intersection_update(metric_filter)
        for metric in sorted(metrics):
            numbers = [row[metric] for row in values if row.get(metric) is not None]
            summary.append({'arrangement': arrangement, 'phase': phase, 'kind': kind, 'cohort': group,
                **({'window_t': offset} if time_offsets else {}), 'metric': metric,
                'target_frames': len(values), 'target_tracks': len({(row['sequence'], row['track_id']) for row in values}),
                'values_present': len(numbers), 'values_missing': len(values) - len(numbers),
                'mean': statistics.fmean(numbers) if numbers else None,
                'q50': quantile(numbers, .5), 'q90': quantile(numbers, .9),
                'zero_fraction': sum(value == 0 for value in numbers) / len(numbers) if numbers else None,
                'positive_fraction': sum(value > 0 for value in numbers) / len(numbers) if numbers else None})
    return summary


def load_completed(root):
    missing = [str(root / 'traces' / sequence / 'COMPLETE.json') for sequence in SEQUENCES
               if not (root / 'traces' / sequence / 'COMPLETE.json').exists()]
    if missing:
        raise FileNotFoundError('WAITING_FOR_COMPLETE: no aggregate conclusions from partial traces: ' + ', '.join(missing))
    evidence = {}
    checkpoint = root.parent / 'm1_recurrent_v1/training/History/run/snapshot_2048.pth'
    assert sha(checkpoint) == CHECKPOINT_SHA
    evidence[str(checkpoint)] = CHECKPOINT_SHA
    cohort_path = root / 'cohorts/COMPLETE.json'; cohort = read_json(cohort_path)
    assert cohort['status'] == 'COMPLETE'
    evidence[str(cohort_path)] = sha(cohort_path)
    for filename in ('STRICT_COHORTS.csv', 'REGIONS.csv'):
        path = root / 'cohorts' / filename
        assert sha(path) == cohort['output_sha256'][filename]
        evidence[str(path)] = sha(path)
    strict = {}
    for row in rows(root / 'cohorts/STRICT_COHORTS.csv'):
        identifier = f"{row['sequence']}:{row['frame']}:{row['annotation_id']}"
        row['groups'] = ['strict_all'] + [name for name in GROUPS[1:] if int(row[name])]
        strict[identifier] = row
    assert len(strict) == 5200
    windows = []
    shared_model_hash = None
    parity = []
    for sequence in SEQUENCES:
        path = root / 'traces' / sequence / 'COMPLETE.json'; receipt = read_json(path)
        assert receipt['status'] == 'COMPLETE' and receipt['sequence'] == sequence
        assert receipt['checkpoint_sha256'] == CHECKPOINT_SHA and receipt['source_condition'] == 'History'
        assert receipt['checkpoint_step'] == 2048 and receipt['all_model_state_unchanged_during_inference']
        assert receipt['cohort_receipt_sha256'] == evidence[str(cohort_path)]
        for manifest in ('source_sha256', 'frozen_v1_source_sha256'):
            for filename, digest in receipt[manifest].items():
                if filename not in evidence:
                    assert sha(filename) == digest, filename
                    evidence[filename] = digest
                else:
                    assert evidence[filename] == digest
        assert receipt['initial_model_state_sha256'] == receipt['final_model_state_sha256']
        if shared_model_hash is None: shared_model_hash = receipt['initial_model_state_sha256']
        assert receipt['initial_model_state_sha256'] == shared_model_hash
        assert receipt['official_starts'] == PLANS['O'] and receipt['symmetric_starts'] == PLANS['S']
        assert sorted(record['window_start'] for record in receipt['windows']) == STARTS
        evidence[str(path)] = sha(path)
        for record in receipt['windows']:
            assert record['identity']['checkpoint_sha256'] == CHECKPOINT_SHA
            for name, digest in record['outputs'].items():
                assert sha(name) == digest, name
                evidence[name] = digest
            assert record['metadata']['shared_F_H'] and record['metadata']['state_reset_at_window_start']
            assert record['metadata']['global_m1_spatial_shape'] == [512, 640]
            assert record['checks']['History_cached']['threshold_05_pixel_differences'] == 0
            assert record['checks']['off_release']['threshold_05_pixel_differences'] == 0
            parity.append({'sequence': sequence, 'window_start': record['window_start'],
                           'checks': record['checks'], 'identity_errors': record['metadata']['maximum_identity_residual']})
            files = list(record['outputs'])
            roi_path = next(Path(name) for name in files if name.endswith('_roi.csv.gz'))
            windows.append((sequence, record['window_start'], roi_path, record['roi_rows']))
    return strict, windows, evidence, parity, shared_model_hash


def aggregate(root):
    strict, windows, evidence, parity, model_hash = load_completed(root)
    objects, margins, offsets, offset_margins = Means(), Means(), Means(), Means()
    observed_rows = 0
    for sequence, start, path, expected_rows in windows:
        pending = defaultdict(dict)
        count = 0
        for row in rows(path):
            count += 1
            assert row['sequence'] == sequence and int(row['window_start']) == start
            phase = 't0' if int(row['window_t']) == 0 else 't1_39'
            pair_id = row['pair_id']; identifier, arrangement = pair_id.rsplit(':', 1)
            assert arrangement == row['arrangement'] and start in PLANS[arrangement]
            frozen = strict[identifier]
            assert int(row['frame']) == int(frozen['frame']) == start + int(row['window_t'])
            assert json.loads(row['groups']) == frozen['groups']
            base = {'object_id': identifier, 'sequence': sequence, 'frame': int(frozen['frame']),
                    'source_frame': int(frozen['source_frame']), 'annotation_id': int(frozen['annotation_id']),
                    'track_id': int(frozen['track_id']), 'arrangement': arrangement, 'kind': row['kind'],
                    'groups': frozen['groups']}
            metric = roi_metrics(row)
            for selection in ('all_offsets', phase):
                info = dict(base, phase=selection)
                objects.add((pair_id, row['kind'], selection), info, metric, start)
            offsets.add((pair_id, row['kind'], int(row['window_t'])), dict(base, offset=int(row['window_t'])), metric, start)
            assert row['kind'] not in pending[pair_id]
            pending[pair_id][row['kind']] = row
        assert count == expected_rows
        observed_rows += count
        for pair_id, pair in pending.items():
            assert set(pair) == {'target', 'background'}
            target, background = pair['target'], pair['background']
            assert target['window_t'] == background['window_t'] and target['frame'] == background['frame']
            identifier, arrangement = pair_id.rsplit(':', 1); frozen = strict[identifier]
            base = {'object_id': identifier, 'sequence': sequence, 'frame': int(frozen['frame']),
                    'source_frame': int(frozen['source_frame']), 'annotation_id': int(frozen['annotation_id']),
                    'track_id': int(frozen['track_id']), 'arrangement': arrangement, 'kind': 'target_minus_background',
                    'groups': frozen['groups']}
            phase = 't0' if int(target['window_t']) == 0 else 't1_39'
            metric = pair_margins(target, background)
            for selection in ('all_offsets', phase):
                margins.add((pair_id, selection), dict(base, phase=selection), metric, start)
            offset_margins.add((pair_id, int(target['window_t'])), dict(base, offset=int(target['window_t'])), metric, start)
    object_rows, margin_rows = objects.finish(), margins.finish()
    object_rows += pool_arrangements(object_rows); margin_rows += pool_arrangements(margin_rows)
    offset_rows, offset_margin_rows = offsets.finish(), offset_margins.finish()
    offset_rows += pool_arrangements(offset_rows); offset_margin_rows += pool_arrangements(offset_margin_rows)
    for kind in ('target', 'background'):
        for plan in ('O', 'S', 'pooled_unique'):
            chosen = [row for row in object_rows if row['kind'] == kind and row['arrangement'] == plan and row['phase'] == 'all_offsets']
            assert len(chosen) == 5200, (kind, plan, len(chosen))
            for group, expected in zip(GROUPS, (5200, 4342, 4170, 646, 9)):
                assert sum(group in row['groups'] for row in chosen) == expected
    stage_summary = summarize(object_rows)
    margin_summary = summarize(margin_rows)
    time_metrics = {stage + suffix for stage in STAGES for suffix in ('_delta_rms', '_relative_delta_rms')}
    time_metrics.update(('delta_z_anchor', 'delta_z_anchor_absolute', 'History_anchor_jacobian_l2',
                         'HistoryCurrent_anchor_jacobian_l2', 'anchor_relu_changed'))
    time_summary = summarize(offset_rows, time_metrics, time_offsets=True)
    time_summary += summarize(offset_margin_rows, time_offsets=True)
    return {'object_rows': object_rows, 'margin_rows': margin_rows, 'stage_summary': stage_summary,
            'margin_summary': margin_summary, 'time_summary': time_summary,
            'evidence': evidence, 'parity': parity, 'model_hash': model_hash, 'raw_roi_rows': observed_rows}


def markdown(bundle):
    index = {(row['arrangement'], row['phase'], row['kind'], row['cohort'], row['metric']): row for row in bundle['stage_summary']}
    def value(group, kind, metric):
        row = index['pooled_unique', 't1_39', kind, group, metric]
        return '—' if row['mean'] is None else f"{row['mean']:.6g}"
    text = ['# 固定History-2048：历史作用链汇总',
        '状态：COMPLETE。两个开发视频的全部原始窗口已完成；这是同一History权重的推理作用诊断，不是新增训练或独立Current检查点比较。',
        '先在每个目标帧、安排和目标/背景种类内平均重叠窗口，再等权平均该目标帧的O/S安排；pooled_unique每个目标帧只计一次。t0与t1–39分别统计。表中的背景是为每个目标固定的困难背景参照，可能重复使用相同背景位置，不是独立背景样本。',
        '以下只展示t1–39、pooled_unique的对象均值；完整O/S、t0、all_offsets及mean/q50/q90见CSV。',
        '| 集合 | 种类 | ΔM RMS | ΔM相对幅度 | Δ残差 RMS | Δ接入H RMS | ΔH相对幅度 | ROI Δlogit RMS | 固定点绝对Δz |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    metrics = ('after_tsfm_delta_rms', 'after_tsfm_relative_delta_rms', 'residual_delta_rms',
               'injected_H_delta_rms', 'injected_H_relative_delta_rms', 'logit_delta_rms', 'delta_z_anchor_absolute')
    for group in GROUPS:
        for kind in ('target', 'background'):
            text.append('| ' + ' | '.join([group, kind] + [value(group, kind, metric) for metric in metrics]) + ' |')
    text += ['', '同窗口、同目标帧的target−background分数间隔在配对后才做窗口/安排均值。anchor使用固定发布位置；peak使用每个模式在同一固定ROI内的峰值，不能将峰位置变化全部解释为同一点的作用。',
             '| 集合 | History anchor间隔 | current anchor间隔 | off anchor间隔 | HC anchor间隔差 | HC peak间隔差 |',
             '| --- | --- | --- | --- | --- | --- |']
    margin_index = {(row['arrangement'], row['phase'], row['cohort'], row['metric']): row for row in bundle['margin_summary']}
    for group in GROUPS:
        values = []
        for metric in ('History_anchor_margin', 'HistoryCurrent_anchor_margin', 'HistoryOff_anchor_margin',
                       'history_current_anchor_margin_delta', 'history_current_peak_margin_delta'):
            number = margin_index['pooled_unique', 't1_39', group, metric]['mean']
            values.append('—' if number is None else f'{number:.6g}')
        text.append('| ' + ' | '.join([group] + values) + ' |')
    text += ['', 'Δ均指History−同检查点current_only。RMS先按ROI像素和通道归一化，再按对象等权；relative_delta_rms为同层两模式RMS平方平均的平方根作分母，分母≤1e−12时缺失而非补零。表中比较不同层的绝对幅度需要同时看相对尺度、实际logit与最终匹配，不能仅凭ΔH数值小认定接入无效。',
             'Jacobian为固定预测头logit对H_new的局部导数；冻结参数的梯度为空不代表头不敏感。HistoryOff位置的Jacobian也作用于HC差分，不能误读成History−off。ReLU掩码变化时局部一阶项不必等于实际有限差。',
             'TIME_OFFSET_SUMMARY.csv按局部时间位置分别报告对象统计；每个位置的目标构成不同，曲线不构成同一对象受控的记忆衰减率或因果半衰期。',
             '9个共同恢复目标是按既有v1结果确定的解释性子组，不能外推全体。内部作用不等于持续漏检恢复；需与三模式实际M/U、保护集合、背景面积和连续漏检评价一起解释。本汇总不预选M1修订。']
    return '\n\n'.join(text[:4]) + '\n\n' + '\n'.join(text[4:]) + '\n'


def self_test():
    accum = Means()
    base = {'object_id': 's:0:1', 'sequence': 's', 'frame': 0, 'source_frame': 0,
            'annotation_id': 1, 'track_id': 1, 'kind': 'target', 'groups': ['strict_all'], 'phase': 'all_offsets'}
    accum.add(('O',), dict(base, arrangement='O'), {'delta': 0.}, 0)
    accum.add(('O',), dict(base, arrangement='O'), {'delta': 2.}, 4)
    accum.add(('S',), dict(base, arrangement='S'), {'delta': 10.}, 2)
    averaged = accum.finish(); pooled = pool_arrangements(averaged)
    assert pooled[0]['delta'] == 5.5  # not raw-row mean4; equal arrangement means(1+10)/2
    summary = summarize(averaged + pooled)
    chosen = next(row for row in summary if row['arrangement'] == 'pooled_unique')
    assert chosen['target_frames'] == 1 and chosen['mean'] == chosen['q50'] == chosen['q90'] == 5.5
    target = {}; background = {}
    for index, mode in enumerate(MODES):
        target[mode+'_anchor_z'] = target[mode+'_z_max'] = str(8-index*2)
        background[mode+'_anchor_z'] = background[mode+'_z_max'] = str(3-index)
    target['delta_z_anchor'] = '2'; background['delta_z_anchor'] = '1'
    assert pair_margins(target, background)['history_current_anchor_margin_delta'] == 1.
    print(json.dumps({'status': 'TEST_FIXTURE_PASS', 'equal_arrangement_weighting': True,
                      'target_background_paired_before_mean': True, 'artifacts_written': False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--output', type=Path, default=E / 'analysis/trace_summary')
    args = parser.parse_args()
    if args.self_test: self_test(); return
    bundle = aggregate(E)
    output = args.output
    if output.exists() and any(output.iterdir()): raise FileExistsError('Preserve existing aggregate: ' + str(output))
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {'OBJECT_FRAME_TRACE_MEANS.csv': bundle['object_rows'], 'PAIRED_MARGIN_OBJECT_MEANS.csv': bundle['margin_rows'],
                 'STAGE_SUMMARY.csv': bundle['stage_summary'], 'PAIRED_MARGIN_SUMMARY.csv': bundle['margin_summary'],
                 'TIME_OFFSET_SUMMARY.csv': bundle['time_summary']}
    for filename, data in artifacts.items(): write_csv(output / filename, data)
    (output / 'TRACE_SUMMARY.md').write_text(markdown(bundle), encoding='utf-8')
    receipt = {'status': 'COMPLETE', 'source_checkpoint': 'M1-v1 History step2048 only', 'checkpoint_sha256': CHECKPOINT_SHA,
               'initial_and_final_model_state_sha256': bundle['model_hash'], 'sequences': list(SEQUENCES), 'plans': PLANS,
               'raw_roi_rows': bundle['raw_roi_rows'], 'unique_strict_target_frames': 5200,
               'groups': list(GROUPS), 'phases': list(PHASES),
               'mean_order': 'ROI channel/pixel statistic -> overlapping-window mean per object/arrangement/kind/phase -> equal mean of available O/S arrangement means -> target-frame equal-weight group statistics',
               'pooled_unique': 'Each object frame once; t0/t1_39 may have only one eligible arrangement, reported in arrangements_present',
               'background_weighting': 'Target-paired fixed references; repeated physical background ROIs are not independent samples',
               'missing_values': 'Average defined values only; counts recorded per object metric and group metric; zero is never substituted for undefined relative ratios',
               'relative_scale': 'delta_RMS/sqrt((history_RMS^2+current_RMS^2)/2), undefined below denominator1e-12',
               'quantiles': 'linear interpolation over object-frame means; q90 of signed variables is signed, separate absolute metrics provided',
               'time_offsets': 'Each local t summarized separately after object/arrangement deduplication; changing population, not a causal decay experiment',
               'diagnostic_scope': 'No detector-benefit verdict or proposed revision from trace alone',
               'reference_and_identity_checks': bundle['parity'], 'input_sha256': bundle['evidence'],
               'aggregation_source_sha256': sha(Path(__file__)), 'created_unix': time.time(),
               'output_sha256': {path.name: sha(path) for path in sorted(output.iterdir()) if path.is_file()}}
    (output / 'TRACE_SUMMARY.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'COMPLETE', 'output': str(output), 'raw_roi_rows': bundle['raw_roi_rows']}))


if __name__ == '__main__': main()
