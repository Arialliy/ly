"""Render the completed one-factor short experiment from final measured artifacts."""
from pathlib import Path
import csv
import hashlib
import json

E = Path(__file__).resolve().parents[1]
V = E / 'evaluation/step_2176'
T = E / 'analysis/trace_summary'
SOURCES = {}
PAIRS = [('RevisionHistory', 'ContinuationHistory'), ('RevisionHistory', 'RevisionCurrent'),
         ('ContinuationHistory', 'ContinuationCurrent'), ('RevisionHistory', 'History'),
         ('ContinuationHistory', 'History')]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def read(path):
    SOURCES[str(path)] = sha(path)
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    SOURCES[str(path)] = sha(path)
    with Path(path).open(newline='', encoding='utf-8') as stream: return list(csv.DictReader(stream))


def table(lines, headers, body):
    lines += ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    lines.extend('| ' + ' | '.join(map(str, row)) + ' |' for row in body)


def main():
    evaluation = read(V / 'SUMMARY_COMPLETE.json')
    aggregate = read(T / 'SUMMARY.json')
    training_check = read(E / 'checks/PAIRED_TRAINING.json')
    pair_check = read(V / 'PAIRED_TRAINING_CHECK.json')
    freeze = read(E / 'PROTOCOL_FREEZE.json')
    assert evaluation['status'] == aggregate['status'] == pair_check['status'] == 'COMPLETE'
    for base, receipt in ((V, evaluation), (T, aggregate)):
        for name, expected in receipt['output_sha256'].items(): assert sha(base / name) == expected
    for path, expected in freeze['source_sha256'].items(): assert sha(path) == expected
    runs = {arm: read(E / 'training' / arm / 'RUN.json') for arm in ('Continuation', 'Revision')}
    assert all(run['status'] == 'COMPLETE' and run['final_step'] == 2176 for run in runs.values())
    assert runs['Continuation']['exposure_digest_sha256'] == runs['Revision']['exposure_digest_sha256']
    costs = {'parameters': {}, 'training': {}, 'traced_inference': [],
        'inference_scope': 'Recorded three-mode shared-frontend inference with full diagnostic tracing, ROI statistics, '
            'Jacobians and artifact generation; not deployment latency or a production benchmark.',
        'memory_measure': 'torch.cuda.max_memory_allocated; GiB conversion uses 2**30; excludes unallocated reserved memory.',
        'source_sha256': {}}
    for arm, run in runs.items():
        assert run['parameters'] == 180263 and run['trainable_parameters'] == 109350
        costs['parameters'][arm] = {'total': run['parameters'], 'trainable': run['trainable_parameters'],
            'added_by_revision': 0}
        costs['training'][arm] = {'additional_steps': 128, 'elapsed_seconds': run['elapsed_seconds'],
            'peak_cuda_allocated_bytes': run['peak_cuda_allocated_bytes'],
            'peak_cuda_allocated_GiB': run['peak_cuda_allocated_bytes'] / 2**30}
        path = E / 'training' / arm / 'RUN.json'
        costs['source_sha256'][str(path)] = sha(path)
        for sequence in ('DJI_0026_2', 'DJI_0067_2'):
            path = E / 'traces' / arm / sequence / 'COMPLETE.json'
            receipt = read(path)
            assert receipt['status'] == 'COMPLETE' and len(receipt['windows']) == 8
            peak = max(window['peak_cuda_allocated_bytes'] for window in receipt['windows'])
            costs['traced_inference'].append({'arm': arm, 'sequence': sequence, 'windows': 8,
                'elapsed_seconds': receipt['elapsed_seconds'], 'peak_cuda_allocated_bytes': peak,
                'peak_cuda_allocated_GiB': peak / 2**30})
            costs['source_sha256'][str(path)] = sha(path)
    summary = rows(V / 'SUMMARY.csv')
    summary_by_key = {(row['condition'], row['arrangement']): row for row in summary}
    allgt = rows(V / 'ALL_GT_SUMMARY.csv')
    allgt_by_key = {(row['condition'], row['arrangement'], row['scope']): row for row in allgt if row['group'] == 'pooled'}
    objects = rows(V / 'ALL_GT_OBJECT_COMPARISON.csv')
    strict = rows(V / 'OBJECT_COMPARISON.csv')
    common9 = rows(V / 'COMMON_RECOVERED_9_MATCHES.csv')
    protected = rows(V / 'PROTECTED_AND_WINDOW_RETENTION.csv')
    tracks = rows(V / 'TRACK_COVERAGE.csv')
    miss = rows(V / 'CONTINUOUS_MISS_SUMMARY.csv')
    comparisons = []
    changed_objects = {}
    for condition, reference in PAIRS:
        pair = condition + '_vs_' + reference
        changed_objects[pair] = [row for row in objects if row[condition + '_matched'] != row[reference + '_matched']]
        for plan in ('O', 'S'):
            after = summary_by_key[condition, plan]
            before = summary_by_key[reference, plan]
            whole = allgt_by_key[condition, plan, 'all_valid_gt']
            comparisons.append({'condition': condition, 'reference': reference, 'arrangement': plan,
                'strict_G': int(after['G_vs_' + reference + '_same']), 'strict_L': int(after['L_vs_' + reference + '_same']),
                'all_valid_G': int(whole['G_vs_' + reference]), 'all_valid_L': int(whole['L_vs_' + reference]),
                'delta_U': int(after['unmatched_components']) - int(before['unmatched_components']),
                'delta_outside_box_positive_pixels': int(after['positive_pixels_outside_boxes']) - int(before['positive_pixels_outside_boxes']),
                'delta_unmatched_component_area': int(after['unmatched_component_area_sum']) - int(before['unmatched_component_area_sum'])})
    track_metrics = ('matched_frames', 'coverage_fraction', 'low_matched_frames', 'extreme_matched_frames',
        'isolated010_hits', 'matched_run_count', 'matched_runs_ge2', 'longest_matched_run', 'longest_miss_run')
    track_changes = {}
    for condition, reference in PAIRS:
        pair = condition + '_vs_' + reference
        indexed = {tuple(row[k] for k in ('arrangement', 'sequence', 'track_id')): row
            for row in tracks if row['condition'] == reference}
        changed = []
        for row in tracks:
            if row['condition'] != condition: continue
            before = indexed[tuple(row[k] for k in ('arrangement', 'sequence', 'track_id'))]
            differences = {key: {'reference': before[key], 'condition': row[key]} for key in track_metrics if before[key] != row[key]}
            if differences: changed.append({key: row[key] for key in ('arrangement', 'sequence', 'track_id')} | {'metrics': differences})
        track_changes[pair] = changed
    stage, margins = {}, {}
    for arm in ('Continuation', 'Revision'):
        stage[arm] = [row for row in rows(T / arm / 'STAGE_SUMMARY.csv') if row['arrangement'] == 'pooled_unique' and row['phase'] == 't1_39']
        margins[arm] = [row for row in rows(T / arm / 'PAIRED_MARGIN_SUMMARY.csv') if row['arrangement'] == 'pooled_unique' and row['phase'] == 't1_39']
    def metric(arm, cohort, kind, name):
        return next(row for row in stage[arm] if row['cohort'] == cohort and row['kind'] == kind and row['metric'] == name)
    def margin(arm, cohort, name):
        return next(row for row in margins[arm] if row['cohort'] == cohort and row['metric'] == name)
    eligibility = {}
    for cohort, total in [('strict_all', 5200), ('low_all_05', 4342), ('extreme_all_001', 4170), ('both_fixed_success', 646), ('m1_low_common_recovered', 9)]:
        eligible = int(metric('Revision', cohort, 'target', 'after_tsfm_relative_delta_rms')['target_frames'])
        assert eligible == int(metric('Continuation', cohort, 'target', 'after_tsfm_relative_delta_rms')['target_frames'])
        eligibility[cohort] = {'full_cohort_target_frames': total, 't1_39_eligible_target_frames': eligible}
    assert len([row for row in strict if row['RevisionHistory_matched'] != row['ContinuationHistory_matched']]) == 1
    assert all(row['RevisionHistory_matched'] == row['ContinuationHistory_matched'] == row['RevisionCurrent_matched']
        for row in common9)
    result = {'status': 'COMPLETE', 'date': '2026-10-14', 'decision': 'FAIL_RECOVERY_OBJECTIVE_STOP_THIS_ROUND',
        'conclusion': 'The aligned-residual state update retains and propagates much larger history-current differences '
            'but adds no low/extreme recovery, loses one protected strict target in S, and lengthens that track miss run. '
            'Its within-arm historical effect suppresses background and targets; a positive low target-background margin '
            'change comes from greater background suppression, not enhanced weak-target response.',
        'next_action': 'Stop after this completed paired128-step experiment; no longer training, second modification or M2.',
        'initial_step': 2048, 'final_step': 2176, 'additional_steps_each': 128,
        'single_change': 'A=PDA(previous,X); M=TSFM(A,X) -> M=A+TSFM(A,X); first frame remains M0=X0',
        'shared_initial_checkpoint_sha256': aggregate['shared_initial_checkpoint_sha256'],
        'final_checkpoint_sha256': aggregate['checkpoint_sha256'],
        'ordered_exposure_sha256': runs['Continuation']['exposure_digest_sha256'],
        'cost': costs,
        'paired_training_checks': {'artifact_check': training_check, 'evaluation_identity_check': pair_check},
        'summary_rows': summary, 'pooled_all_gt_rows': [row for row in allgt if row['group'] == 'pooled'],
        'comparisons': comparisons, 'all_valid_changed_objects': changed_objects,
        'strict_track_changes': track_changes, 'pooled_continuous_miss_rows': [row for row in miss if row['group'] == 'pooled'],
        'protection_and_common9': [row for row in protected if row['cohort'] in ('both_fixed_success', 'm1_low_common_recovered')],
        't1_39_eligibility': eligibility, 'stage_t1_39_pooled_unique': stage, 'margin_t1_39_pooled_unique': margins,
        'mechanism_scope': 'Within-arm deltas use each final checkpoint history-current_only. Revision after_tsfm records '
            'the final A+TSFM state, not pure TSFM output. Overlapping windows are averaged per object/arrangement before '
            'available O/S means and equal-weight target-frame aggregation. Relative differences are descriptive, not an information-retention rate.',
        'limitations': ['One paired short experiment from one trained History checkpoint; no official Pd/Fa or generalization claim.',
            'The common9 subgroup is outcome-conditioned on v1; fixed backgrounds can repeat physical positions.',
            'Native DCNv2/adaptive-pooling backward is not bitwise deterministic; identical restored inputs/optimizer/RNG do not imply bitwise reproducible training.',
            'Positive target-background margin changes require inspection of target and background signed responses separately.'],
        'source_sha256': SOURCES, 'report_source_sha256': sha(__file__)}
    (E / 'RESULTS.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    lines = [
        '**M1-v1.1 单因素短实验已完成，未通过弱目标恢复目标；本轮到此停止，不长训、不加第二处改动、不启动 M2。**（2026-10-14）',
        '',
        '唯一结构改动为 `A=PDA(previous,X); M=TSFM(A,X)` → `M=A+TSFM(A,X)`，首帧仍为 `M0=X0`。'
        'Continuation与Revision从同一History2048模型、全部Adam状态及RNG起点，在同序已有F缓存0…127上各训练128步，固定2176终点。'
        '两臂的曝光摘要相同，原DeepPro-Plus、BN和头部保持冻结；推理时各自使用同一终点权重比较History与current_only。'
        '本轮没有另训Current。身份和实际检查点核验见[PAIRED_TRAINING](checks/PAIRED_TRAINING.json)与[评分身份检查](evaluation/step_2176/PAIRED_TRAINING_CHECK.json)。',
        '',
        '**主比较没有新低响应恢复，并额外丢失一个保护目标。**下表沿用strict5200的M、全部有效GT匹配后的U、GT框外正像素A和未匹配组件面积。'
        '总组件面积等于全图正像素数；U为未匹配组件数，不等同已确认假阳性数。History指旧2048终点。', '']
    table(lines, ['条件','安排','M','U','A','未匹配面积','总组件面积','low4342','extreme4170','protect646'],
        [[row[key] for key in ('condition','arrangement','M','unmatched_components','positive_pixels_outside_boxes',
          'unmatched_component_area_sum','component_area_sum','low_all_05_matched','extreme_all_001_matched','both_fixed_success_matched')] for row in summary])
    lines += ['',
        'RevisionHistory相对ContinuationHistory：O匹配完全相同；S的G/L=0/1，M由724降到723。low仍为O5/S7，'
        '极低响应4170仍为0；共同恢复9个唯一目标帧的全部O/S匹配模式保持不变。保护集合双安排都成功由644降到643，'
        '新增损失仅发生在S，未出现两个安排都丢失。O的U=4018仍高于固定预算4015；S的U=4006在预算内，却比release的3967多39。',
        '',
        '**比较范围和代价。**G/L逐安排计算；全部有效GT为16297个目标帧，包含strict5200。差值均为左侧条件减参考条件。', '']
    table(lines, ['条件 / 参考','安排','strict G/L','全部GT G/L','ΔU','ΔA','Δ未匹配面积'],
        [[row['condition']+' / '+row['reference'], row['arrangement'], f"{row['strict_G']}/{row['strict_L']}",
          f"{row['all_valid_G']}/{row['all_valid_L']}", row['delta_U'], row['delta_outside_box_positive_pixels'],
          row['delta_unmatched_component_area']] for row in comparisons])
    lines += ['',
        'RevisionHistory相对自身同权重RevisionCurrent，确实使O/S的U减少8/15、A减少89/83、未匹配面积减少160/15；'
        '同时没有新增strict匹配，S仍损失上述保护目标，全部有效GT的G/L为4/5与2/11。'
        '因此该历史效应主要表现为抑制，不能用背景代价下降替代目标恢复。Continuation的History/current控制在全部GT匹配上完全相同，'
        '仅有轻微背景代价变化。相对旧History2048，Continuation在strict上无变化、全部GT净减1/3；Revision在strict净变0/−1、全部GT净变0/−6。'
        '额外训练自身的变化已经由Continuation单列。',
        '',
        '主比较RevisionHistory−ContinuationHistory在全部GT上只有以下4条改变；其中唯一strict改变属于既有保护集合。', '']
    table(lines, ['安排','视频','frame / source_frame','annotation_id','track_id','strict','变化'],
        [[row['arrangement'],row['sequence'],row['frame']+' / '+row['source_frame'],row['annotation_id'],row['track_id'],
          row['strict_small'],'恢复' if row['RevisionHistory_matched']=='1' else '丢失']
         for row in changed_objects['RevisionHistory_vs_ContinuationHistory']])
    lines += ['',
        '**连续性没有改善。**新增丢失对象为 `DJI_0067_2 / frame101 / annotation876734 / track7702014 / S`。'
        '该轨迹在冻结观察集合中有82个目标帧，缺标/时间断点切段，未证明是完整物理可见区间。'
        '其中匹配从5降到4，覆盖从5/82降到4/82，最长连续漏检从14延长到17。'
        '其原有孤立010命中被删除，造成全体S的孤立命中21→20、匹配段158→157；这不是连续性提升。'
        '长度至少2的匹配段仍为O79/S80。O的130条轨迹最长漏检全部相同，S有1条变长、129条相同、0条变短；'
        'low4342和extreme4170的连续漏检记录未改善。相对RevisionCurrent和旧History同样出现这条轨迹损失。'
        '逐对象与完整连续性来源见[对象改变](evaluation/step_2176/REVISIONHISTORY_VS_CONTINUATIONHISTORY_CHANGED_OBJECTS.csv)、'
        '[轨迹覆盖](evaluation/step_2176/TRACK_COVERAGE.csv)与[连续漏检](evaluation/step_2176/CONTINUOUS_MISS_SUMMARY.csv)。',
        '',
        '**历史差异被保留并传到logit，但作用方向没有形成新的弱目标恢复。**以下是每臂自己的History−current_only差，'
        '不是两臂直接相减。low表仅覆盖具有t1–39观测资格的4307/4342个目标帧：先平均重叠窗口，再平均可用O/S安排，最后等权平均目标帧。'
        'Revision中的“更新后”是 `A+TSFM(A,X)` 的最终状态。相对差的分母为同层两模式RMS平方均值的平方根，不能解释成信息保持率。', '']
    mechanism_names = [('PDA后相对状态差','after_pda_relative_delta_rms'),('更新后相对状态差','after_tsfm_relative_delta_rms'),
        ('接入H差RMS','injected_H_delta_rms'),('logit差RMS','logit_delta_rms'),('固定目标锚点有符号Δz','delta_z_anchor')]
    table(lines, ['low目标指标（4307个）','Continuation','Revision'],
        [[label, f"{float(metric('Continuation','low_all_05','target',name)['mean']):.9g}",
          f"{float(metric('Revision','low_all_05','target',name)['mean']):.9g}"] for label,name in mechanism_names])
    lines += ['',
        '更新后相对差从1.30844%增到23.7451%，接入H差从7.42786e−6增到0.000259298，logit差RMS从0.000516654增到0.0272266。'
        '这些实测说明差异已被保留和传递，不能将本轮失败仅归因于历史差异幅度消失或冻结预测头阻断。'
        '但Revision的low目标固定锚点有符号Δz均值为−0.0163735，仅11.2143%的有资格目标帧为正；配对背景为−0.0243967。'
        '**low间隔均值增加+0.00802323来自背景下降更多，不能表述为弱目标响应增强。**',
        '',
        '同样方向在保护集合和既有共同恢复子组上需要保留：', '']
    table(lines, ['集合（t1–39资格/全集合）','Revision目标Δz','配对背景Δz','目标−背景间隔变化'],
        [[label, f"{float(margin('Revision',cohort,'target_delta_z_anchor')['mean']):.9g}",
          f"{float(margin('Revision',cohort,'background_delta_z_anchor')['mean']):.9g}",
          f"{float(margin('Revision',cohort,'history_current_anchor_margin_delta')['mean']):.9g}"]
         for cohort,label in [('low_all_05','low 4307/4342'),('both_fixed_success','protect 643/646'),
                              ('m1_low_common_recovered','共同恢复 9/9')]])
    lines += ['',
        '完整t1–39资格为strict5162/5200、low4307/4342、extreme4135/4170、protect643/646及共同恢复9/9；仅首帧观测留在t0层。'
        '最终匹配评价仍覆盖全部冻结集合，未按这些资格数过滤。共同9按v1结果定义，属于解释性子组；固定困难背景坐标沿用release选择，'
        '重复背景位置不算独立样本。低响应目标状态差与背景状态差同时扩大；结合有符号响应、实际匹配与保护损失，'
        '本轮暴露的是这些新增保留差异在当前终点的判别作用和方向未满足恢复需求，不能推断PDA对齐信息已被证实对目标有用。'
        '详细分布、缺失计数与每对象数据见[两臂作用链汇总](analysis/trace_summary/SUMMARY.md)、'
        '[阶段比较](analysis/trace_summary/STAGE_COMPARISON.csv)和[有符号目标/背景间隔比较](analysis/trace_summary/PAIRED_MARGIN_COMPARISON.csv)。',
        '',
        '**实际计算成本。**两臂总参数均为180263、训练参数均为109350，修订新增参数为0。'
        '下表来自各臂RUN实测；显存是CUDA已分配内存峰值，GiB按2³⁰字节换算。', '']
    table(lines, ['训练臂','新增步数','耗时（秒）','峰值已分配显存（GiB）'],
        [[arm,128,f"{row['elapsed_seconds']:.3f}",f"{row['peak_cuda_allocated_GiB']:.4f}"]
         for arm,row in costs['training'].items()])
    lines += ['',
        '推理收据记录的是三模式共享前端、完整trace、ROI统计/Jacobian及产物生成的诊断执行。'
        '**这些耗时不能当作部署时延或生产推理基准。**每条视频使用原8个40帧窗口：', '']
    table(lines, ['诊断臂','视频','窗口数','耗时（秒）','峰值已分配显存（GiB）'],
        [[row['arm'],row['sequence'],row['windows'],f"{row['elapsed_seconds']:.3f}",f"{row['peak_cuda_allocated_GiB']:.4f}"]
         for row in costs['traced_inference']])
    lines += ['',
        '成本数值及对应训练RUN/推理COMPLETE的SHA256收录在[RESULTS.json](RESULTS.json)的 `cost` 字段；'
        '没有把冻结模型总参数数目与新增参数数目混用。',
        '',
        '**结果判定。**这一次128步、固定2176终点的单因素改动达成了“保留更多历史差异”，没有达成“新增弱目标恢复且保住原成功目标”。'
        '按照[冻结协议](PROTOCOL.md)，本轮停止，不追加长训或M2。结论限定于该配对短实验及两个开发视频的原实际组件诊断；'
        '不称官方Pd/Fa，不把一次失败推广为所有历史机制无效，也不把未发生的后续训练写成证据。',
        '',
        '两臂终点SHA256：Continuation `'+runs['Continuation']['checkpoint_sha256']+'`；Revision `'+runs['Revision']['checkpoint_sha256']+'`。'
        '训练、推理、评分及机制聚合均已完成；冻结code哈希核验通过。完整12行评分、全部GT G/L、所有改变对象、连续性变化和来源哈希见[RESULTS.json](RESULTS.json)。', '']
    (E / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'status':'COMPLETE','decision':result['decision'],'RESULTS.json':sha(E/'RESULTS.json'),'RESULTS.md':sha(E/'RESULTS.md')}), flush=True)


if __name__ == '__main__': main()
