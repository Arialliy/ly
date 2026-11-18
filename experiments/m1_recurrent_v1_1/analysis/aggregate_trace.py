"""Aggregate both complete2176 arms with the unchanged2048 ROI averaging code.

Only the completed-input loader is replaced. No model, cohort, ROI statistic,
overlap/arrangement weighting, quantile or target/background pairing is changed.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import time

E = Path(__file__).resolve().parents[1]
DIAG = E.parent / 'm1_history_effect_v1'
LEGACY_PATH = DIAG / 'analysis/aggregate_trace.py'
LEGACY_SHA = '0ca390f3dfece255ed6c2de933706df20bf39b592ff2e3ceb36b7a3ebc78475d'
COHORT_SHA = '0f4b072348718d401ce6e49ba2bcf8036ffe817777b50e2c84bee38f39669951'
INITIAL_SHA = 'eda063496757f368d7442a7fac99a64f7b33ef0414ef99d006716d77a6dcb725'
ARMS = ('Continuation', 'Revision')
STEP = 2176


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
    spec = importlib.util.spec_from_file_location('_m11_original_trace_aggregation', LEGACY_PATH)
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
    assert sources == {str(p): sha(p) for p in sorted((root / 'code').glob('*.py'))}
    for path, digest in sources.items():
        verify(path, digest)
    diagnostic = DIAG / 'PROTOCOL_FREEZE.json'
    verify(diagnostic)
    for path, digest in read(diagnostic)['source_sha256'].items():
        verify(path, digest)
    check_path = root / 'checks/PAIRED_TRAINING.json'
    check = read(check_path)
    verify(check_path)
    assert check['status'] == 'PASS' and check['steps_each'] == 128 and check['endpoint'] == STEP
    runs, exposures = {}, {}
    for arm in ARMS:
        path = root / 'training' / arm / 'RUN.json'
        verify(path, check['runs_sha256'][arm])
        run = runs[arm] = read(path)
        assert run['status'] == 'COMPLETE' and run['condition'] == arm
        assert run['initial_step'] == 2048 and run['final_step'] == STEP
        assert run['additional_steps'] == run['completed_additional_steps'] == 128
        assert run['initial_checkpoint_sha256'] == INITIAL_SHA
        assert run['source_sha256'] == sources and run['protocol_sha256'] == protocol_sha
        assert run['original_unchanged'] and run['all_original_parameters_and_BN_frozen']
        verify(path.with_name('snapshot_2176.pth'), run['checkpoint_sha256'])
        for source, digest in run['frozen_v1_source_sha256'].items():
            verify(source, digest)
        training = path.with_name('TRAIN.jsonl')
        verify(training)
        log = [json.loads(line) for line in training.read_text().splitlines() if line.strip()]
        assert len(log) == 128
        digest = hashlib.sha256()
        exposure = []
        for index, row in enumerate(log):
            assert row['condition'] == arm and row['plan_index'] == index
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
                'exposure', 'exposure_digest_sha256', 'seed')
    for field in equality:
        assert runs['Continuation'][field] == runs['Revision'][field], field
    assert exposures['Continuation'] == exposures['Revision']
    assert runs['Continuation']['revision'] == 'TSFM(A,X)'
    assert runs['Revision']['revision'] == 'A+TSFM(A,X)'
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
            assert meta['pda_tsfm_calls_per_mode'] == {'History': 39, 'HistoryCurrent': 39, 'HistoryOff': 0}
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
            row['Revision_minus_Continuation_' + field] = (right[field] - left[field]
                if right[field] is not None and left[field] is not None else None)
        row['complete_same_object_population'] = left['values_present'] == right['values_present'] == left['target_frames']
        result.append(row)
    return result


def arm_outputs(root, output, arm, run, bundle, legacy):
    output.mkdir(parents=True)
    tables = {'OBJECT_FRAME_TRACE_MEANS.csv': 'object_rows', 'PAIRED_MARGIN_OBJECT_MEANS.csv': 'margin_rows',
              'STAGE_SUMMARY.csv': 'stage_summary', 'PAIRED_MARGIN_SUMMARY.csv': 'margin_summary',
              'TIME_OFFSET_SUMMARY.csv': 'time_summary'}
    for filename, key in tables.items():
        legacy.write_csv(output / filename, bundle[key])
    markdown = legacy.markdown(bundle)
    markdown = markdown.replace('# 固定History-2048：历史作用链汇总', f'# {arm}固定2176权重：历史作用链汇总')
    markdown = markdown.replace('这是同一History权重的推理作用诊断，不是新增训练或独立Current检查点比较。',
        f'这是{arm}完成128步后的同一2176权重history/current_only/off推理作用诊断；不是独立训练Current的比较。')
    markdown = markdown.replace('本汇总不预选M1修订。', '本汇总不单独判定该修订是否实现持续漏检恢复。')
    markdown += ('\n字段语义：after_tsfm是状态更新的最终输出。Continuation为TSFM(A,X)，Revision为A+TSFM(A,X)；'
                 '比较表示更新位置的状态差异是否保留，不表示原TSFM门值已改变或学会保留。后续时刻的PDA输入也可能随递归改变。\n')
    (output / 'TRACE_SUMMARY.md').write_text(markdown, encoding='utf-8')
    receipt = {'status': 'COMPLETE', 'arm': arm, 'checkpoint_step': STEP,
        'checkpoint_sha256': run['checkpoint_sha256'], 'shared_initial_checkpoint_sha256': INITIAL_SHA,
        'initial_and_final_inference_model_state_sha256': bundle['model_hash'],
        'raw_roi_rows': bundle['raw_roi_rows'], 'unique_strict_target_frames': 5200,
        'sequences': list(legacy.SEQUENCES), 'plans': legacy.PLANS,
        'groups': list(legacy.GROUPS), 'phases': list(legacy.PHASES),
        'mode_labels': {'History': arm + 'History', 'HistoryCurrent': arm + 'Current', 'HistoryOff': 'same original H/head'},
        'after_tsfm_field_semantics': 'final state-update output: ' + run['revision'] + '; not an isolated bare-TSFM or gate-change measurement',
        'mean_order': 'overlap windows per target-frame/arrangement/kind/phase -> equal available O/S arrangement means -> equal target-frame group statistics',
        'background_weighting': 'Target-paired fixed references; repeated physical background ROIs are not independent samples',
        'relative_scale': 'delta_RMS/sqrt((history_RMS^2+current_RMS^2)/2), undefined at denominator <=1e-12; no zero fill',
        'scope': 'Within-arm same-final-weight HC effects; independent trained Current is not present. Internal changes do not alone prove detection benefit.',
        'unchanged_aggregation_sha256': LEGACY_SHA, 'aggregation_loader_sha256': sha(__file__),
        'reference_and_identity_checks': bundle['parity'], 'input_sha256': bundle['evidence'],
        'output_sha256': {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}, 'created_unix': time.time()}
    save(output / 'TRACE_SUMMARY.json', receipt)


def summary_markdown(comparisons):
    index = {(row['arrangement'], row['phase'], row['kind'], row['cohort'], row['metric']): row
             for row in comparisons['STAGE_COMPARISON.csv']}
    lines = ['# M1单处状态更新修订：2176终点作用链对比', '',
        '状态：COMPLETE。Continuation与Revision均从同一History-2048模型/Adam/RNG开始，完成同序列128步曝光；每臂使用自己的固定2176权重计算history/current_only/off。', '',
        '下表是pooled_unique、t1–39的目标帧等权均值。每个目标先平均重叠窗口，再等权平均可用O/S安排。背景为沿用的目标配对固定困难背景；不能把重复背景位置当成独立样本。', '',
        '| 集合 | 种类 | 指标（臂内History−current_only） | Continuation | Revision | 均值差R−C |',
        '| --- | --- | --- | --- | --- | --- |']
    metrics = ('after_pda_relative_delta_rms', 'after_tsfm_relative_delta_rms', 'injected_H_delta_rms',
               'logit_delta_rms', 'delta_z_anchor_absolute')
    for group in ('low_all_05', 'extreme_all_001', 'both_fixed_success', 'm1_low_common_recovered'):
        for kind in ('target', 'background'):
            for metric in metrics:
                row = index['pooled_unique', 't1_39', kind, group, metric]
                values = [row['Continuation_mean'], row['Revision_mean'], row['Revision_minus_Continuation_mean']]
                lines.append('| ' + ' | '.join([group, kind, metric] + ['—' if v is None else f'{v:.6g}' for v in values]) + ' |')
    lines += ['', '| 集合 | HC固定点目标−背景间隔差：Continuation | Revision | R−C |', '| --- | --- | --- | --- |']
    for row in comparisons['PAIRED_MARGIN_COMPARISON.csv']:
        if (row['arrangement'], row['phase'], row['metric']) == ('pooled_unique', 't1_39', 'history_current_anchor_margin_delta'):
            values = [row['Continuation_mean'], row['Revision_mean'], row['Revision_minus_Continuation_mean']]
            lines.append('| ' + ' | '.join([row['cohort']] + ['—' if v is None else f'{v:.6g}' for v in values]) + ' |')
    lines += ['', 'CSV保留两臂各自mean/q50/q90与差值。分位数相减是两个边际分位数的差，不是逐对象差的分位数；仅当两臂该指标在全部相同对象上都有定义，均值差才等于配对对象差的均值。缺失值不补零。', '',
        '每臂内部Δ均为该臂同权重history−current_only。修订臂after_tsfm记录A+TSFM(A,X)的最终状态，不能误称原TSFM纯输出。改动经过递归也可能改变后续PDA输入；阶段幅度比不是受控信息保持率或因果半衰期。', '',
        '各臂仍输出off原路径以验证基线和固定头。HistoryOff位置Jacobian作用于HC差分，不能读成History−off；9个既有共同恢复目标是结果条件化解释子组。', '',
        '内部历史差异增大只表明状态依赖改变。持续漏检恢复结论须结合evaluation/step_2176的实际匹配G/L、连续漏检、轨迹覆盖、U/面积和protect646，并比较RevisionHistory−ContinuationHistory及各臂自身History−Current。此汇总不自动支持下一轮训练或M2。']
    return '\n'.join(lines) + '\n'


def self_test(legacy):
    legacy.self_test()
    base = dict(arrangement='pooled_unique', phase='t1_39', kind='target', cohort='strict_all', metric='delta',
                target_frames=2, target_tracks=1, values_present=2, values_missing=0,
                mean=2., q50=2., q90=2.8, zero_fraction=0., positive_fraction=1.)
    result = compare_summaries([base], [dict(base, mean=5., q50=4., q90=8.)])[0]
    assert result['Revision_minus_Continuation_mean'] == 3. and result['complete_same_object_population']
    result = compare_summaries([base], [dict(base, values_present=1, values_missing=1, mean=None)])[0]
    assert result['Revision_minus_Continuation_mean'] is None and not result['complete_same_object_population']
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
        comparisons[filename] = compare_summaries(compact['Continuation'][key], compact['Revision'][key], key == 'time_summary')
        legacy.write_csv(output / filename, comparisons[filename])
    (output / 'SUMMARY.md').write_text(summary_markdown(comparisons), encoding='utf-8')
    save(output / 'SUMMARY.json', {'status': 'COMPLETE', 'arms': list(ARMS), 'final_step': STEP,
        'additional_steps_each': 128, 'shared_initial_checkpoint_sha256': INITIAL_SHA,
        'checkpoint_sha256': {arm: runs[arm]['checkpoint_sha256'] for arm in ARMS},
        'aggregation_source_sha256': sha(__file__), 'unchanged_legacy_aggregation_sha256': LEGACY_SHA,
        'frozen_cohort_receipt_sha256': COHORT_SHA, 'paired_training_source_sha256': evidence,
        'comparison_scope': 'Same-final-weight history-current effects within each arm; cross-arm summary differences are descriptive, not detection-benefit proof',
        'quantile_difference': 'Difference of marginal quantiles, never quantiles of paired differences',
        'output_sha256': {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob('*')) if p.is_file()},
        'created_unix': time.time()})
    print(json.dumps({'status': 'COMPLETE', 'output': str(output), 'arms': list(ARMS), 'step': STEP}))


if __name__ == '__main__':
    main()
