"""Build the main diagnostic report from completed immutable measurement tables."""
from pathlib import Path
import csv
import hashlib
import json

E = Path(__file__).resolve().parents[1]
V = E / 'evaluation/step_2048'
T = E / 'analysis/trace_summary'
inputs = {}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    inputs[str(path)] = sha(path)
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    inputs[str(path)] = sha(path)
    with Path(path).open(newline='', encoding='utf-8') as stream: return list(csv.DictReader(stream))


def main():
    evaluation = read(V / 'SUMMARY_COMPLETE.json')
    trace = read(T / 'TRACE_SUMMARY.json')
    interpretation = read(E / 'analysis/EVALUATION_INTERPRETATION.json')
    assert evaluation['status'] == trace['status'] == interpretation['status'] == 'COMPLETE'
    for base, receipt in ((V, evaluation), (T, trace)):
        for name, expected in receipt['output_sha256'].items(): assert sha(base / name) == expected
    summary = rows(V / 'SUMMARY.csv')
    allgt = rows(V / 'ALL_GT_SUMMARY.csv')
    stage = rows(T / 'STAGE_SUMMARY.csv')
    margins = rows(T / 'PAIRED_MARGIN_SUMMARY.csv')
    low = {row['metric']: row for row in stage if (row['arrangement'], row['phase'], row['kind'], row['cohort'])
           == ('pooled_unique', 't1_39', 'target', 'low_all_05')}
    low_margin = {row['metric']: row for row in margins if (row['arrangement'], row['phase'], row['cohort'])
                  == ('pooled_unique', 't1_39', 'low_all_05')}
    checkpoints = read(E / 'PROTOCOL_FREEZE.json')
    off = read(V / 'OFF_RELEASE_ARRAY_COMPARISON.json')
    parity = read(E / 'analysis/RELEASE_OFF_PARITY.json')
    jacobian = read(E / 'checks/HEAD_JACOBIAN_CPU.json')
    inputs[str(Path(__file__))] = sha(__file__)
    result = {
        'status': 'COMPLETE', 'date': '2026-10-14',
        'question': 'Does the fixed trained History2048 checkpoint gain detections or continuity from recurrent history at inference?',
        'checkpoint_sha256': evaluation['checkpoint_sha256'],
        'model_state_sha256': trace['initial_and_final_model_state_sha256'],
        'additional_training_steps': 0, 'conditions': evaluation['conditions'],
        'condition_meaning': {'release': 'Archived released baseline cache', 'History': 'Fixed v1 History2048, recurrent state',
            'HistoryCurrent': 'Same History2048 parameters, previous=X_t through the same PDA/TSFM',
            'HistoryOff': 'Same checkpoint, M1 residual including bias bypassed; original H enters frozen head'},
        'conclusion': 'History and same-weight current-only have identical target matches and continuity at the frozen '
            'threshold and O/S arrangements. The nine previously common recovered targets remain recovered without '
            'recurrent history at inference. Internal state and logits do differ, so this is not an absence-of-history claim.',
        'summary_rows': summary,
        'all_gt_pooled_rows': [row for row in allgt if row['group'] == 'pooled'],
        'same_weight_match_and_continuity': interpretation,
        'low_target_trace_t1_39_pooled_unique': low,
        'low_target_background_margin_t1_39_pooled_unique': low_margin,
        'trace_aggregation': {key: trace[key] for key in ('raw_roi_rows', 'unique_strict_target_frames', 'mean_order',
            'pooled_unique', 'background_weighting', 'missing_values', 'relative_scale')},
        'off_release_array_comparison': off,
        'cross_runtime': {'current': 'torch2.2/CUDA12.3', 'archived_release': 'torch2.7/CUDA12.8',
            'max_window_probability_difference': max(item['checks']['off_release']['probability_absmax']
                for item in trace['reference_and_identity_checks']),
            'all_window_threshold05_differences_zero': all(item['checks']['off_release']['threshold_05_pixel_differences'] == 0
                for item in trace['reference_and_identity_checks']),
            'History_matches_v1_cache_exact_every_window': all(item['checks']['History_cached']['exact_probability_equal']
                for item in trace['reference_and_identity_checks']),
            'same_runtime_off_original_parity_source': str(E / 'analysis/RELEASE_OFF_PARITY.json')},
        'interpretation_limits': [
            'PDA-to-TSFM contraction is a measured stage association, not proof that the discarded direction contains useful target evidence.',
            'The frozen head has nonzero local Jacobians and propagates differences; frozen parameters do not imply a blocked input path.',
            'Same-checkpoint mode substitution is an inference dependence test, not a replacement for independently paired training.',
            'The common9 subgroup is defined by completed v1 outcomes and is explanatory, not an independent population estimate.',
            'Two development clips and actual-component diagnostics do not establish official Pd/Fa or generalization.'],
        'revision': {'status': 'SELECTED_PENDING_PAIRED_EXECUTION',
            'only_candidate_change': 'M_t = A_t + TSFM(A_t, X_t), A_t=PDA(previous, X_t)',
            'planned_test': 'Original History continuation versus aligned-residual revision, 128 paired updates from the same History2048 checkpoint/Adam/RNG.',
            'result_claim': None, 'registration_folder': str(E.parent / 'm1_recurrent_v1_1')},
        'source_sha256': inputs}
    (E / 'RESULTS.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')

    lines = [
        '**同权重诊断已完成：历史状态产生了非零内部变化，但没有增加任何目标匹配，也没有改善记录的连续性。**（2026-10-14）',
        '',
        '本轮固定同一个 History step2048 检查点（SHA256 `'+evaluation['checkpoint_sha256']+'`），没有新增训练。'
        'History 使用上一状态；HistoryCurrent 将上一状态替换成当前 X，仍走同一 PDA/TSFM；HistoryOff 关闭完整 M1 残差及其偏置。'
        '因此 HistoryCurrent 是同权重推理控制，不是 v1 中另行训练的 Current 模型。两个视频、40帧独立窗口、O/S融合与0.5匹配阈值均沿用原设置。',
        '',
        '**检测与背景代价。**下表 M 为 strict5200 目标帧匹配数，U 为匹配所有有效GT后剩余的组件数；U不是已确认假阳性数。'
        'A为GT框外正像素数，未匹配面积为未匹配组件面积之和；总组件面积等于全图正像素数。',
        '',
        '| 条件 | 安排 | M | U | A | 未匹配组件面积 | 总组件面积 | low4342匹配 | extreme4170匹配 | protect646匹配 |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in summary:
        lines.append('| ' + ' | '.join(row[key] for key in ('condition','arrangement','M','unmatched_components',
            'positive_pixels_outside_boxes','unmatched_component_area_sum','component_area_sum',
            'low_all_05_matched','extreme_all_001_matched','both_fixed_success_matched')) + ' |')
    lines += ['',
        'History 与 HistoryCurrent 的全部10400条 strict安排×目标帧记录、32594条全部有效GT记录，匹配差异均为0，'
        '各自 O/S 的 G=L=0。全部有效GT匹配数均为3238/3353。History 相比 HistoryCurrent 只减少 O/S 的框外正像素1/2、'
        '未匹配组件面积3/4；U完全相同。O的U=4022超过固定预算4015，S的4011虽未超过预算，仍比release增加44。'
        '保护集合两条件均在S损失2个目标帧，双安排均成功数从646降至644。',
        '',
        '相对Off，History在strict集合的 O/S G/L 为9/2、9/5（净增7/4）；在全部有效GT上却为56/79、60/79（净减23/19）。'
        'low集合共同恢复的9个唯一目标帧，在HistoryCurrent中全部保留原来的O/S匹配模式（O5、S7、交集3）；Off的这9个均不匹配。'
        '因此这9个恢复在固定权重上不依赖递归历史输入。该子组按已完成v1结果定义，只作解释性分析。',
        '',
        '连续性也完全相同：260对安排×完整轨迹的覆盖/命中指标、900对轨迹×分组连续漏检记录和30对汇总记录没有任何差异。'
        '两条件 O/S 的长度至少2的匹配段均为79/80，孤立010命中均为23/21；低响应集合和极低响应集合也没有历史带来的连续漏检改善。'
        '详见[完整评分表](evaluation/step_2048/SUMMARY.csv)、[全部GT](evaluation/step_2048/ALL_GT_SUMMARY.csv)、'
        '[共同9逐对象记录](evaluation/step_2048/COMMON_RECOVERED_9_MATCHES.csv)与[连续性核对](analysis/EVALUATION_INTERPRETATION.json)。',
        '',
        '**状态差主要在TSFM后收缩，但并未变成零。**以下为low4342目标中的t1–39可观测4307个目标帧，先在对象/安排内平均重叠窗口，'
        '再平均可用O/S，最后等权平均目标帧。相对差为同层ΔRMS除以两条件RMS平方均值的平方根；不是跨层信息量度。',
        '',
        '| 位置/量 | 对象均值 |', '| --- | ---: |',
        '| PDA后相对状态差 | '+low['after_pda_relative_delta_rms']['mean']+' |',
        '| TSFM后相对状态差 | '+low['after_tsfm_relative_delta_rms']['mean']+' |',
        '| TSFM后状态差RMS | '+low['after_tsfm_delta_rms']['mean']+' |',
        '| 接入H差RMS | '+low['injected_H_delta_rms']['mean']+' |',
        '| 接入H相对差 | '+low['injected_H_relative_delta_rms']['mean']+' |',
        '| 原头输出logit差RMS | '+low['logit_delta_rms']['mean']+' |',
        '| 固定发布锚点绝对logit差 | '+low['delta_z_anchor_absolute']['mean']+' |',
        '',
        '固定target−hard-background锚点间隔的History−HistoryCurrent均值为+0.000411446，'
        '而两者间隔本身约−45.1085/−45.1089；这个小变化没有转成新增匹配。背景坐标来自release的原合法同尺寸网格最大logit，'
        '排除全部已标注目标及2px缓冲后冻结，三模式共享；重复的物理背景ROI不视为独立样本。'
        'PDA后差异较大、TSFM后差异较小，支持把TSFM处的保留路径作为可检验位置；它不证明被收缩的对齐差异已经含有目标有用信息。'
        '不能把本结果写成“完全没有历史”或“对齐信息已证实有效”。',
        '',
        '原冻结头没有阻断这些输入差异：4307个low目标帧的固定锚点局部Jacobian均非零，History的Jacobian L2均值为24.1154，'
        '|J·ΔH|均值0.000354141与实际固定点|Δz|均值0.000354132接近；接入H的微小扰动经过头部后，logit差RMS约0.000530832。'
        '这是头部对输入敏感并传递差异的证据，不支持因head被冻结而认定接入被阻塞，也不支持据此解冻预测头。'
        '真实数据逐层统计见[作用链汇总](analysis/trace_summary/TRACE_SUMMARY.md)和[完整阶段分布](analysis/trace_summary/STAGE_SUMMARY.csv)；'
        '[独立CPU Jacobian核对](checks/HEAD_JACOBIAN_CPU.json)只验证计算式，未用合成点代替真实目标证据。',
        '',
        '**关闭M1与发布结果一致性的边界。**History在所有16个视频窗口上与v1同环境缓存逐位相同。'
        'Off与当前torch2.2/CUDA12.3环境中未改动的原baseline路径，在两视频首个完整窗口上逐位相同；'
        '旧release缓存来自torch2.7/CUDA12.8，16窗最大概率尾差为1.537799835205078e-5，所有0.5像素判断差异为0。'
        '最终Off/release匹配、计数、峰坐标及所有组件面积数组完全相同，只有box_max响应值未逐位相同（两视频最大差2.8014e-6、8.37445e-6）。'
        '保留了初次严格容差失败日志和[原冻结登记](runtime/reconciliation/original_protocol_freeze.json)，'
        '没有把跨环境尾差算成M1效果。见[当前环境路径核对](analysis/RELEASE_OFF_PARITY.json)与[最终数组比较](evaluation/step_2048/OFF_RELEASE_ARRAY_COMPARISON.json)。',
        '',
        '**下一步状态：单因素修订已选定，配对验证尚未产生结果。**拟只在TSFM输出添加对齐特征残差：'
        '`A_t=PDA(previous,X_t); M_t=A_t+TSFM(A_t,X_t)`。原History继续训练与该修订从同一个History2048模型、Adam和RNG起点继续128步，'
        '用于区分额外训练与结构变化；登记和执行归于[下一轮目录](../m1_recurrent_v1_1/)。本报告不声称这处修订有效。',
        '',
        '全部结论限定于这一个已训练检查点、两个开发视频与冻结诊断定义。完整数值与来源SHA256见[RESULTS.json](RESULTS.json)；'
        '原v1代码、检查点和评分缓存保持只读，本轮评分及作用链均为实际运行结果。', '']
    (E / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'status':'COMPLETE','RESULTS.json':sha(E/'RESULTS.json'),'RESULTS.md':sha(E/'RESULTS.md')}), flush=True)


if __name__ == '__main__': main()
