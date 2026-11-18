"""Aggregate both complete2560 arms with the unchanged2048 ROI averaging code.

Only the completed-input loader is replaced. No model, cohort, ROI statistic,
overlap/arrangement weighting, quantile or target/background pairing is changed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import statistics

E = Path(__file__).resolve().parents[1]
DIAG = E.parent / 'm1_history_effect_v1'
LEGACY_PATH = DIAG / 'analysis/aggregate_trace.py'
LEGACY_SHA = '0ca390f3dfece255ed6c2de933706df20bf39b592ff2e3ceb36b7a3ebc78475d'
COHORT_SHA = '0f4b072348718d401ce6e49ba2bcf8036ffe817777b50e2c84bee38f39669951'
INITIAL_SHA = 'eda063496757f368d7442a7fac99a64f7b33ef0414ef99d006716d77a6dcb725'
ARMS = ('BankCurrent', 'BankHistory')
STEP = 2560


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def legacy_module():
    assert sha(LEGACY_PATH) == LEGACY_SHA, 'Frozen averaging implementation changed'
    spec = importlib.util.spec_from_file_location('_m13_original_trace_aggregation', LEGACY_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def complete_gate(root, legacy):
    required = [root / 'traces' / arm / sequence / 'COMPLETE.json'
                for arm in ARMS for sequence in legacy.SEQUENCES]
    required += [root / 'checks/PAIRED_TRAINING.json']
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError('WAITING_FOR_COMPLETE: no partial-arm conclusions: ' + ', '.join(missing))


def paired_training(root, legacy):
    """Read-only independent equality and ordered exposure checks before aggregation."""
    complete_gate(root, legacy)
    evidence = {}

    def verify(path, digest=None):
        path = Path(path)
        actual = sha(path)
        assert digest is None or actual == digest, str(path)
        if str(path) in evidence:
            assert evidence[str(path)] == actual, str(path)
        evidence[str(path)] = actual
        return actual

    verify(LEGACY_PATH, LEGACY_SHA)
    verify(DIAG / 'cohorts/COMPLETE.json', COHORT_SHA)
    verify(root.parent / 'm1_recurrent_v1/training/History/run/snapshot_2048.pth', INITIAL_SHA)
    protocol = root / 'PROTOCOL_FREEZE.json'
    protocol_sha = verify(protocol)
    sources = read(protocol)['source_sha256']
    source_paths = list((root / 'code').glob('*.py')) + list((root / 'reference/M1_v13_evidence_bank_reference').glob('*.py'))
    assert sources == {str(p): sha(p) for p in sorted(source_paths)}
    for path, digest in sources.items():
        verify(path, digest)
    diagnostic = DIAG / 'PROTOCOL_FREEZE.json'
    verify(diagnostic)
    for path, digest in read(diagnostic)['source_sha256'].items():
        verify(path, digest)
    check_path = root / 'checks/PAIRED_TRAINING.json'
    check = read(check_path)
    verify(check_path)
    assert check['status'] == 'PASS' and check['steps_each'] == 512 and check['endpoint'] == STEP
    runs, exposures = {}, {}
    for arm in ARMS:
        path = root / 'training' / arm / 'RUN.json'
        verify(path, check['runs_sha256'][arm])
        run = runs[arm] = read(path)
        assert run['status'] == 'COMPLETE' and run['condition'] == arm
        assert run['initial_step'] == 2048 and run['final_step'] == STEP
        assert run['additional_steps'] == run['completed_additional_steps'] == 512
        assert run['initial_checkpoint_sha256'] == INITIAL_SHA
        assert run['source_sha256'] == sources and run['protocol_sha256'] == protocol_sha
        assert run['original_unchanged'] and run['all_original_parameters_and_BN_frozen']
        verify(path.with_name('snapshot_2560.pth'), run['checkpoint_sha256'])
        for source, digest in run['frozen_v1_source_sha256'].items():
            verify(source, digest)
        training = path.with_name('TRAIN.jsonl')
        verify(training)
        log = [json.loads(line) for line in training.read_text(encoding='utf-8').splitlines() if line.strip()]
        assert len(log) == 512
        digest = hashlib.sha256()
        exposure = []
        for index, row in enumerate(log):
            assert row['condition'] == arm and row['plan_index'] == index % 256
            assert row['additional_step'] == index + 1 and row['step'] == 2049 + index
            keys = ('input_sha256', 'label_sha256', 'auxiliary_input_sha256')
            hashes = tuple(row[key] for key in keys)
            for value in hashes:
                assert len(value) == 64
                digest.update(value.encode())
            exposure.append(hashes)
        assert digest.hexdigest() == run['exposure_digest_sha256'] == check['exposure_digest_sha256']
        exposures[arm] = exposure
    equality = ('initial_checkpoint_sha256', 'initial_model_state_sha256', 'original_state_sha256',
                'source_sha256', 'frozen_v1_source_sha256', 'protocol_sha256', 'settings', 'cache_sha256',
                'trainable_parameter_names', 'trainable_parameters', 'parameters', 'optimizer', 'rng',
                'exposure', 'exposure_digest_sha256', 'seed', 'optimizer_migration',
                'reference_evidence_bank_sha256', 'capacity', 'pairs_per_window', 'runtime_settings')
    for field in equality:
        assert runs['BankCurrent'][field] == runs['BankHistory'][field], field
    assert exposures['BankCurrent'] == exposures['BankHistory']
    assert runs['BankCurrent']['mode'] == 'current_only' and runs['BankHistory']['mode'] == 'history'
    for run in runs.values():
        assert run['revision'] == 'source-feature evidence bank'
        assert run['capacity'] == 4 and run['pairs_per_window'] == 150
        assert run['added_parameters'] == run['added_parameter_tensors'] == 0
        assert run['parameters'] == 180263 and run['trainable_parameters'] == 109350
        assert len(run['trainable_parameter_names']) == 53
        assert run['optimizer_final_steps'] == {'old_steps':[2560.0], 'old_state_count':53, 'new_state_count':0}
        verify(root / 'reference/M1_v13_evidence_bank_reference/evidence_bank.py', run['reference_evidence_bank_sha256'])
    return runs, evidence


def load_completed(root, arm, legacy, run, shared_evidence):
    """Replacement identity/path loader; old aggregate() consumes its same tuple."""
    evidence = dict(shared_evidence)

    def verify(path, digest=None):
        path = Path(path)
        actual = evidence.get(str(path)) or sha(path)
        assert digest is None or actual == digest, str(path)
        evidence[str(path)] = actual
        return actual

    cohort_path = DIAG / 'cohorts/COMPLETE.json'
    cohort = read(cohort_path)
    assert cohort['status'] == 'COMPLETE' and verify(cohort_path) == COHORT_SHA
    for name, digest in cohort['output_sha256'].items():
        verify(DIAG / 'cohorts' / name, digest)
    strict = {}
    for row in legacy.rows(DIAG / 'cohorts/STRICT_COHORTS.csv'):
        identifier = f"{row['sequence']}:{row['frame']}:{row['annotation_id']}"
        assert identifier not in strict
        row['groups'] = ['strict_all'] + [name for name in legacy.GROUPS[1:] if int(row[name])]
        strict[identifier] = row
    assert len(strict) == 5200
    windows, parity = [], []
    for sequence in legacy.SEQUENCES:
        path = root / 'traces' / arm / sequence / 'COMPLETE.json'
        receipt = read(path)
        verify(path)
        assert receipt['status'] == 'COMPLETE' and receipt['sequence'] == sequence
        assert receipt['source_condition'] == arm and receipt['checkpoint_step'] == STEP
        assert receipt['main_checkpoint_step'] == STEP and receipt['checkpoint_is_main']
        assert receipt['checkpoint_sha256'] == run['checkpoint_sha256']
        assert receipt['trained_mode'] == run['mode']
        assert receipt['bank_capacity'] == run['capacity'] == 4
        assert receipt['pairs_per_window_per_mode'] == run['pairs_per_window'] == 150
        assert receipt['reference_evidence_bank_sha256'] == run['reference_evidence_bank_sha256']
        assert receipt['training_run_sha256'] == verify(root / 'training' / arm / 'RUN.json')
        assert receipt['all_model_state_unchanged_during_inference'] and receipt['all_original_frozen']
        assert receipt['initial_model_state_sha256'] == receipt['final_model_state_sha256'] == run['final_model_state_sha256']
        assert receipt['cohort_receipt_sha256'] == COHORT_SHA
        assert receipt['source_sha256'] == run['source_sha256']
        assert receipt['frozen_v1_source_sha256'] == run['frozen_v1_source_sha256']
        assert receipt['protocol_sha256'] == verify(root / 'PROTOCOL_FREEZE.json')
        assert receipt['diagnostic_protocol_sha256'] == verify(DIAG / 'PROTOCOL_FREEZE.json')
        assert receipt['official_starts'] == legacy.PLANS['O'] and receipt['symmetric_starts'] == legacy.PLANS['S']
        assert receipt['diagnostic_starts'] == legacy.STARTS
        assert receipt['normalization'] == [85.0799, 47.4845] and not receipt['model_labels_access']
        assert sorted(record['window_start'] for record in receipt['windows']) == legacy.STARTS
        for record in receipt['windows']:
            assert all(receipt[key] == value for key, value in record['identity'].items())
            assert record['identity']['checkpoint_sha256'] == run['checkpoint_sha256']
            for name, digest in record['outputs'].items():
                verify(name, digest)
            meta = record['metadata']
            assert meta['shared_F_H'] and meta['state_reset_at_window_start']
            assert meta['all_weights_and_buffers_unchanged'] and meta['baseline_buffers_unchanged']
            assert meta['global_m1_spatial_shape'] == [512, 640]
            assert meta['input_shape'] == [1, 1, 40, 512, 640]
            assert meta['pda_tsfm_calls_per_mode'] == {'History': 150, 'HistoryCurrent': 150, 'HistoryOff': 0}
            assert meta['bank_capacity'] == 4 and meta['read_then_write']
            assert meta['maximum_identity_residual']['bank_read_mean'] == 0
            assert record['slot_frame_rows'] == 150
            assert 'after_update' not in meta['stages'] and 'previous_input' in meta['stages']
            assert meta['empty_clipped_rois_skipped'] == 0 and not meta['roi_labels_enter_model']
            reference = record['checks']['off_release']
            assert reference['threshold_05_pixel_differences'] == 0
            assert reference['cross_runtime_release'] and reference['absolute_tolerance'] == 2e-5
            assert reference['probability_absmax'] <= reference['absolute_tolerance']
            verify(reference['reference_path'], reference['reference_sha256'])
            roi_paths = [Path(name) for name in record['outputs'] if name.endswith('_roi.csv.gz')]
            assert len(roi_paths) == 1
            windows.append((sequence, record['window_start'], roi_paths[0], record['roi_rows']))
            parity.append({'sequence': sequence, 'window_start': record['window_start'],
                           'checks': record['checks'], 'identity_errors': meta['maximum_identity_residual']})
    return strict, windows, evidence, parity, run['final_model_state_sha256']


def compare_summaries(continuation, revision, time_offsets=False):
    keys = ('arrangement', 'phase', 'kind', 'cohort', 'metric') + (('window_t',) if time_offsets else ())
    indices = [{tuple(row[key] for key in keys): row for row in records}
               for records in (continuation, revision)]
    assert len(indices[0]) == len(continuation) and len(indices[1]) == len(revision)
    assert indices[0].keys() == indices[1].keys()
    result = []
    for key in sorted(indices[0]):
        left, right = indices[0][key], indices[1][key]
        assert left['target_frames'] == right['target_frames'] and left['target_tracks'] == right['target_tracks']
        row = dict(zip(keys, key), target_frames=left['target_frames'], target_tracks=left['target_tracks'])
        for arm, source in zip(ARMS, (left, right)):
            for field in ('values_present', 'values_missing', 'mean', 'q50', 'q90', 'zero_fraction', 'positive_fraction'):
                row[arm + '_' + field] = source[field]
        for field in ('mean', 'q50', 'q90'):
            row['BankHistory_minus_BankCurrent_' + field] = (right[field] - left[field]
                if right[field] is not None and left[field] is not None else None)
        row['complete_same_object_population'] = left['values_present'] == right['values_present'] == left['target_frames']
        result.append(row)
    return result


def source_age_summary(root, legacy):
    """Compact per-source diagnostic, separate from all unchanged ROI arithmetic."""
    grouped = defaultdict(list)
    for arm in ARMS:
        for sequence in legacy.SEQUENCES:
            receipt = read(root / 'traces' / arm / sequence / 'COMPLETE.json')
            for record in receipt['windows']:
                paths = [Path(name) for name in record['outputs'] if name.endswith('_slot_frame.csv.gz')]
                assert len(paths) == 1 and sha(paths[0]) == record['outputs'][str(paths[0])]
                source_rows = list(legacy.rows(paths[0]))
                assert len(source_rows) == 150
                seen = set()
                for row in source_rows:
                    t, age = int(row['window_t']), int(row['source_age'])
                    assert row['sequence'] == sequence and int(row['window_start']) == record['window_start']
                    assert 1 <= age <= min(t,4) and int(row['active_slots']) == min(t,4)
                    assert int(row['source_window_t']) == t-age
                    assert int(row['source_frame']) == record['window_start']+t-age
                    assert int(row['current_control_actual_source_window_t']) == t
                    assert int(row['slot']) == min(t,4)-age
                    assert (t,age) not in seen
                    seen.add((t,age))
                    for stage in ('source','after_pda','candidate'):
                        for suffix in ('delta_mean','delta_rms','delta_absmax','history_rms','current_rms','relative_delta_rms'):
                            metric = stage+'_'+suffix
                            value = legacy.number(row[metric])
                            for scope in (sequence,'pooled_sequences'):
                                grouped[arm,scope,age,metric].append(value)
                assert seen == {(t,age) for t in range(1,40) for age in range(1,min(t,4)+1)}
    output = []
    for (arm,sequence,age,metric), values in sorted(grouped.items()):
        available = [value for value in values if value is not None]
        output.append({'arm':arm,'sequence':sequence,'source_age':age,'metric':metric,
            'slot_observations':len(values),'values_present':len(available),'values_missing':len(values)-len(available),
            'mean':statistics.fmean(available) if available else None,
            'q50':legacy.quantile(available,.5),'q90':legacy.quantile(available,.9),
            'minimum':min(available) if available else None,'maximum':max(available) if available else None,
            'positive_fraction':sum(value>0 for value in available)/len(available) if available else None,
            'weighting':'equal window/frame/slot occurrences; overlapping windows retained; no target conditioning',
            'comparison':'same-arm same-weight history minus current_only'})
    return output


def arm_outputs(root, output, arm, run, bundle, legacy):
    output.mkdir(parents=True)
    tables = {'OBJECT_FRAME_TRACE_MEANS.csv': 'object_rows', 'PAIRED_MARGIN_OBJECT_MEANS.csv': 'margin_rows',
              'STAGE_SUMMARY.csv': 'stage_summary', 'PAIRED_MARGIN_SUMMARY.csv': 'margin_summary',
              'TIME_OFFSET_SUMMARY.csv': 'time_summary'}
    for filename, key in tables.items():
        legacy.write_csv(output / filename, bundle[key])
    markdown = legacy.markdown(bundle)
    markdown = markdown.replace('# 固定History-2048：历史作用链汇总', f'# {arm}固定2560权重：历史作用链汇总')
    markdown = markdown.replace('这是同一History权重的推理作用诊断，不是新增训练或独立Current检查点比较。',
        f'这是{arm}完成512步后的同一2560权重history/current_only/off推理作用诊断；此臂内部差值不是独立训练Current的比较。')
    markdown = markdown.replace('本汇总不预选M1修订。', '本汇总不单独判定该修订是否实现持续漏检恢复。')
    markdown += ('\n字段语义：before_pda是previous_input的兼容别名，表示原始来源X的均值；'
                 'after_pda为各来源分别对齐后的均值；after_tsfm为各来源分别融合后的实际平均readout。'
                 'FIFO只存融合前原X，readout从不写回；这里没有v1.2门控或单状态递归更新。'
                 '均值可能抵消不同来源的差异，另附逐source_age的全图标量统计帮助区分。'
                 'BankCurrent权重下的history干预为辅助机制诊断，不是新增正式预测条件。\n')
    (output / 'TRACE_SUMMARY.md').write_text(markdown, encoding='utf-8')
    receipt = {'status': 'COMPLETE', 'arm': arm, 'checkpoint_step': STEP,
        'checkpoint_sha256': run['checkpoint_sha256'], 'shared_initial_checkpoint_sha256': INITIAL_SHA,
        'initial_and_final_inference_model_state_sha256': bundle['model_hash'],
        'raw_roi_rows': bundle['raw_roi_rows'], 'unique_strict_target_frames': 5200,
        'sequences': list(legacy.SEQUENCES), 'plans': legacy.PLANS,
        'groups': list(legacy.GROUPS), 'phases': list(legacy.PHASES),
        'mode_labels': {'History': arm + 'History' + ('_auxiliary' if arm == 'BankCurrent' else ''),
                        'HistoryCurrent': arm + 'Current', 'HistoryOff': 'same original H/head'},
        'before_pda_field_semantics': 'mean of original projected source features; compatibility alias of previous_input',
        'after_pda_field_semantics': 'mean of independently aligned source features',
        'after_tsfm_field_semantics': 'actual source-order mean of independently fused candidates; never written back to FIFO',
        'mean_order': 'overlap windows per target-frame/arrangement/kind/phase -> equal available O/S arrangement means -> equal target-frame group statistics',
        'background_weighting': 'Target-paired fixed references; repeated physical background ROIs are not independent samples',
        'relative_scale': 'delta_RMS/sqrt((history_RMS^2+current_RMS^2)/2), undefined at denominator <=1e-12; no zero fill',
        'scope': 'Within-arm same-final-weight HC effects. BankCurrent history intervention is auxiliary; independent trained-control detection comparison is in evaluation. Internal changes do not alone prove detection benefit.',
        'unchanged_aggregation_sha256': LEGACY_SHA, 'aggregation_loader_sha256': sha(__file__),
        'reference_and_identity_checks': bundle['parity'], 'input_sha256': bundle['evidence'],
        'output_sha256': {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}, 'created_unix': time.time()}
    save(output / 'TRACE_SUMMARY.json', receipt)


def summary_markdown(comparisons):
    index = {(row['arrangement'], row['phase'], row['kind'], row['cohort'], row['metric']): row
             for row in comparisons['STAGE_COMPARISON.csv']}
    lines = ['# M1-v1.3来源bank：2560终点作用链对比', '',
        '状态：COMPLETE。BankCurrent与BankHistory均从同一History-2048模型/Adam/RNG开始，完成同序列512步曝光（既有256条缓存两轮）。每臂使用自己的固定2560权重计算history/current_only/off；BankCurrent权重的history干预仅为辅助机制诊断。', '',
        '下表为pooled_unique、t1–39的目标帧等权统计。每个目标先平均重叠窗口，再等权平均可用O/S安排。背景为目标配对固定困难背景，重复背景位置不能当成独立样本。N是该phase可用目标帧数量，不等于整个静态cohort大小。', '',
        '| 集合 | 种类 | N | 指标（臂内History−current_only） | BankCurrent | BankHistory | 均值差H臂−C臂 |',
        '| --- | --- | ---: | --- | ---: | ---: | ---: |']
    metrics = ('before_pda_relative_delta_rms', 'after_pda_relative_delta_rms',
               'after_tsfm_relative_delta_rms', 'injected_H_delta_rms', 'logit_delta_rms', 'delta_z_anchor')
    for group in ('low_all_05', 'extreme_all_001', 'both_fixed_success', 'm1_low_common_recovered'):
        for kind in ('target', 'background'):
            for metric in metrics:
                row = index['pooled_unique', 't1_39', kind, group, metric]
                values = [row['BankCurrent_mean'], row['BankHistory_mean'], row['BankHistory_minus_BankCurrent_mean']]
                lines.append('| ' + ' | '.join([group, kind, str(row['target_frames']), metric]
                    + ['—' if v is None else f'{v:.6g}' for v in values]) + ' |')
    lines += ['', '| 集合 | N | HC固定点目标−背景间隔差：BankCurrent | BankHistory | H臂−C臂 |',
              '| --- | ---: | ---: | ---: | ---: |']
    for row in comparisons['PAIRED_MARGIN_COMPARISON.csv']:
        if (row['arrangement'], row['phase'], row['metric']) == ('pooled_unique', 't1_39', 'history_current_anchor_margin_delta'):
            values = [row['BankCurrent_mean'], row['BankHistory_mean'], row['BankHistory_minus_BankCurrent_mean']]
            lines.append('| ' + ' | '.join([row['cohort'], str(row['target_frames'])]
                + ['—' if v is None else f'{v:.6g}' for v in values]) + ' |')
    lines += ['', '必须同时读目标和背景的有符号delta_z_anchor：间隔提高可能来自背景下降更多，不能据此写成目标响应增强。CSV还保留两臂各自positive_fraction、mean/q50/q90与差值。分位数相减是两个边际分位数的差，不是逐对象差的分位数；仅当两臂指标在全部相同对象上都有定义，均值差才等于配对对象差的均值。缺失值不补零。', '',
        'before_pda/previous_input为所有原始源X的均值，after_pda为逐源对齐输出的均值，after_tsfm为实际逐源TSFM候选均值readout。均值会发生抵消；SOURCE_AGE_SUMMARY.csv另按age给出逐源全图标量，采用window/frame/slot出现次数等权，只是描述性补充，不沿用目标帧权重，也不代表目标证据质量。FIFO只存原X、不存readout，容量4限定新增bank来源索引年龄，不限定原前端/TPro时间感受野。', '',
        '各臂off原路径用于验证基线和固定头。HistoryOff位置Jacobian作用于HC差分，不能读成History−off；共同恢复9个对象是v1结果条件化解释子组。', '',
        '内部历史差异非零或增大只表明依赖关系存在。持续漏检恢复结论须结合evaluation/step_2560的真实匹配G/L、连续漏检、轨迹覆盖、U/面积和protect646，并分别比较BankHistoryHistory−BankCurrentCurrent及BankHistoryHistory−BankHistoryCurrent。此汇总不自动支持下一轮训练或M2。']
    return '\n'.join(lines) + '\n'


def self_test(legacy):
    legacy.self_test()
    base = dict(arrangement='pooled_unique', phase='t1_39', kind='target', cohort='strict_all', metric='delta',
                target_frames=2, target_tracks=1, values_present=2, values_missing=0,
                mean=2., q50=2., q90=2.8, zero_fraction=0., positive_fraction=1.)
    result = compare_summaries([base], [dict(base, mean=5., q50=4., q90=8.)])[0]
    assert result['BankHistory_minus_BankCurrent_mean'] == 3. and result['complete_same_object_population']
    result = compare_summaries([base], [dict(base, values_present=1, values_missing=1, mean=None)])[0]
    assert result['BankHistory_minus_BankCurrent_mean'] is None and not result['complete_same_object_population']
    try:
        compare_summaries([base], [dict(base, target_frames=3)])
    except AssertionError:
        pass
    else:
        raise AssertionError('Different target populations accepted')
    print(json.dumps({'status': 'TEST_FIXTURE_PASS', 'scope': 'Unchanged legacy arithmetic and comparison-only fixture; no experiment values',
                      'no_artifacts_written': True, 'no_gpu': True}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--output', type=Path, default=E / 'analysis/trace_summary')
    args = parser.parse_args()
    legacy = legacy_module()
    if args.self_test:
        self_test(legacy)
        return
    output = args.output
    if output.exists():
        raise FileExistsError('Preserve existing aggregate output: ' + str(output))
    runs, evidence = paired_training(E, legacy)
    # Validate both complete arms before writing even the first arm's output.
    loaded = {arm: load_completed(E, arm, legacy, runs[arm], evidence) for arm in ARMS}
    compact = {}
    output.mkdir(parents=True)
    for arm in ARMS:
        legacy.load_completed = lambda root, selection=arm: loaded[selection]
        bundle = legacy.aggregate(E)  # original function and every arithmetic helper unchanged
        arm_outputs(E, output / arm, arm, runs[arm], bundle, legacy)
        compact[arm] = {key: bundle[key] for key in ('stage_summary', 'margin_summary', 'time_summary')}
        del bundle
        gc.collect()
    comparisons = {}
    for filename, key in (('STAGE_COMPARISON.csv', 'stage_summary'), ('PAIRED_MARGIN_COMPARISON.csv', 'margin_summary'),
                          ('TIME_OFFSET_COMPARISON.csv', 'time_summary')):
        comparisons[filename] = compare_summaries(compact['BankCurrent'][key], compact['BankHistory'][key], key == 'time_summary')
        legacy.write_csv(output / filename, comparisons[filename])
    legacy.write_csv(output / 'SOURCE_AGE_SUMMARY.csv', source_age_summary(E,legacy))
    (output / 'SUMMARY.md').write_text(summary_markdown(comparisons), encoding='utf-8')
    save(output / 'SUMMARY.json', {'status': 'COMPLETE', 'arms': list(ARMS), 'final_step': STEP,
        'additional_steps_each': 512, 'shared_initial_checkpoint_sha256': INITIAL_SHA,
        'checkpoint_sha256': {arm: runs[arm]['checkpoint_sha256'] for arm in ARMS},
        'aggregation_source_sha256': sha(__file__), 'unchanged_legacy_aggregation_sha256': LEGACY_SHA,
        'frozen_cohort_receipt_sha256': COHORT_SHA, 'paired_training_source_sha256': evidence,
        'comparison_scope': 'Same-final-weight history-current effects within each arm; cross-arm summary differences are descriptive, not detection-benefit proof',
        'BankCurrent_history_mode': 'Auxiliary within-weight intervention only; official Current prediction uses current_only',
        'source_age_summary_weighting': 'Equal window/frame/slot appearances, not ROI target-frame weighting; repeated temporal windows retained',
        'bank_semantics': 'Original projected X FIFO4; independently align/fuse each source then average; no fused readout writeback;150 pairs per40frame',
        'quantile_difference': 'Difference of marginal quantiles, never quantiles of paired differences',
        'output_sha256': {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob('*')) if p.is_file()},
        'created_unix': time.time()})
    print(json.dumps({'status': 'COMPLETE', 'output': str(output), 'arms': list(ARMS), 'step': STEP}))


if __name__ == '__main__':
    main()
