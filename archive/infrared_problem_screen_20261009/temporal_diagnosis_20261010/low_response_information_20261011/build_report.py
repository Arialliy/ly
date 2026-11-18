#!/usr/bin/env python3
"""Render the three diagnostic tables directly from their checked CSVs."""
import csv
from pathlib import Path

BASE = Path(__file__).resolve().parent


def indexed(name):
    with (BASE/name).open() as f:
        return {(r['arrangement'],r['sequence'],r['group']):r for r in csv.DictReader(f)}


def main():
    scores = indexed('SCORE_SEPARATION.csv')
    peaks = indexed('PEAK_LOCATION.csv')
    tracks = indexed('TRACK_SUMMARY.csv')
    s = {a:scores[a,'pooled','low_all_05'] for a in ['O','S']}
    p = {a:peaks[a,'pooled','low_all_05'] for a in ['O','S']}
    t = {a:tracks[a,'pooled','low_all_05'] for a in ['O','S']}
    def val(data, key, fmt='int'):
        out=[]
        for a in ['O','S']:
            v=float(data[a][key])
            out.append(f'{v:,.0f}' if fmt=='int' else f'{v:.2%}' if fmt=='pct' else f'{v:.3f}')
        return ' | '.join(out)
    def row(label,data,key,fmt='int'):
        return f'| {label} | {val(data,key,fmt)} |'
    table1='\n'.join([
        '| 表一：4,342 个全窗低响应目标帧 | 原安排 O | 最佳共用安排 S |',
        '| --- | ---: | ---: |',
        row('目标 logit 中位数',s,'target_box_max_logit_median','float'),
        row('高于全图同尺寸背景 q95（分母均 4,342）',s,'global_advantage_q95_valid_ge20_count'),
        row('高于全图同尺寸背景 q99（分母均 4,342）',s,'global_advantage_q99_valid_ge20_count'),
        row('比目标分数更高的背景峰数量中位数',s,'global_background_peaks_stronger_than_target_median','float'),
        row('高于局部背景 q95（分母均 1,633）',s,'local_advantage_q95_valid_ge20_count'),
        row('局部、全图 q95 均占优（分母均 1,633）',s,'both_advantage_q95_valid_ge20_frames'),
        row('仅局部 q95 占优（分母均 1,633）',s,'local_only_advantage_q95_valid_ge20_frames'),
    ])
    table2='\n'.join([
        '| 表二：低响应目标附近的最高原始 logit 峰 | O | S |',
        '| --- | ---: | ---: |',
        row('落在本目标框内（全量分母 4,342）',p,'peak_inside_own_box_frames'),
        row('落在其他已标注目标框内',p,'peak_in_other_annotated_box_frames'),
        row('落在全部已标注框外，归属未由坐标确定',p,'peak_outside_all_annotated_boxes_frames'),
        row('邻域有对应概率 ≥0.5 的高峰，但高峰均在本框外',p,'outside_peak_ge05_frames'),
        row('人工核查：疑似本目标相关（45 帧、10 轨迹、2 片段）',p,'visual_suspected_target_frames'),
        row('人工核查：背景或其他目标相关（同一 45 帧）',p,'visual_background_or_other_frames'),
        row('人工核查：无法判断（同一 45 帧）',p,'visual_uncertain_frames'),
    ])
    table3='\n'.join([
        '| 表三：低响应层的轨迹持续性与背景参照 | O | S |',
        '| --- | ---: | ---: |',
        '| 涉及范围 | 4,342 帧 / 121 轨迹 / 2 片段 | 同左 |',
        '| 各片段贡献 | 0026：302 帧、7 轨；0067：4,040 帧、114 轨 | 同左 |',
        row('前五条轨迹贡献比例',t,'top5_tracks_fraction','pct'),
        '| 每轨最长连续低响应／漏检段：中位数 / 最大值 | 7 / 160 帧 | 7 / 160 帧 |',
        row('目标高于全图 q95：帧加权比例',t,'global_advantage_frame_weighted_fraction','pct'),
        row('目标高于全图 q95：轨迹等权比例',t,'global_advantage_track_equal_fraction','pct'),
        row('几何选定背景参照高于其全图 q95：帧加权',t,'reference_global_advantage_frame_weighted_fraction','pct'),
        row('目标高于固定背景参照的比例',t,'target_above_reference_frame_weighted_fraction','pct'),
        row('目标全图 q95 优势的最长连续段',t,'global_advantage_longest_run_max'),
        row('背景参照全图 q95 优势的最长连续段',t,'reference_global_advantage_longest_run_max'),
        row('移动 ≥1 像素的 1,815 对：邻域峰位移残差中位数（像素）',t,'neighborhood_moving_relative_position_change_median','float'),
        row('相同帧对：背景参照峰位移残差中位数（像素）',t,'reference_moving_relative_position_change_median','float'),
    ])
    decision_path = BASE/'INTERVENTION_DECISION.md'
    decision = decision_path.read_text() if decision_path.exists() else '下一干预等待正在执行的 λ=0 前提控制，诊断三表已完成。'
    report=f'''# 低响应目标是否仍有可用检测信息：本轮结果

本轮完成 **10,400 行逐对象帧明细及三张主表**。阈值固定 0.5，原权重、预处理、五窗安排、概率融合和匹配不变。原安排 O 的实际匹配仍为 700，最佳共用安排 S 仍为 720；本轮诊断没有产生新的模型提升。

**结果支持：部分低响应目标保留相对排序信息，但当前输出尚未提供足够强、归属明确的目标响应。不能把主缺口解释成统一的分数尺度问题，或把附近其他目标的高峰算作本目标的可恢复信息。** 是否能把这些信息变成满足背景预算的检测收益，需要一次干预验证。

{table1}

全图背景比较覆盖全部 4,342 帧。局部背景至少 20 块的主分母只有 **1,633 帧，全部来自 0067**；另外 2,708 帧有 1—19 块、1 帧没有可用块。局部不足受框形状、排除缓冲和场景密度影响，不能把该子集外推全量。所有非空局部块的辅助口径下，两类 q95 均占优为 O 898/4,341、S 926/4,341；它不替代冻结的主口径。样本不足的原始背景块分数和坐标已额外归档。

4,170 个极低响应帧涉及 116 条轨迹：高于全图 q99 的仅 O 201、S 221。成功参照 646 帧的全图 q95 优势为 646/646，目标 logit 中位数 O 14.871、S 15.439。背景峰数量是无阈值的竞争诊断，**不是未匹配组件 U，更不是官方 Fa**。q95 或 q99 占优也不等于已满足检测背景预算。

{table2}

有限人工核查总计 18 个预选案例、15 条轨迹、72 个唯一目标帧（两安排共 144 行），上表仅取其中实际属于低响应层的 45 帧。邻帧按自己的冻结分层统计，未沿用案例锚点分组；未检查对象的人工标签保持空白。核查范围内没有确认“本目标框外的可信强峰可以直接恢复低响应主体”，不能据此断言全量都没有此类峰。

原图并非全无可辨证据：0067 的轨迹 7700544 在第 121—124 帧可辨车状轮廓，而邻域最高峰属于左右其他车辆；0026 的轨迹 400006 在第 42—46 帧仍有边界旁亮团，原模型框内响应很低。单个邻域 argmax 不是附近所有响应的穷尽。另有反例表明：邻域最高峰属于他物时，本目标仍能匹配；本框内 raw-logit 峰很强时，实际概率组件匹配仍可能失败。数值峰与实际输出已经分别记录。

{table3}

低响应层来自多条轨迹，但 **93.04% 的目标帧来自 0067**；两段既有开发片段不足以代表广泛来源。所有连续段同时检查当前帧号与源帧号，缺标、筛选空缺、背景无效和不足均按对应条件断段。分层最长段只在该层内部计算，全轨迹结果另保留于主表。背景参照按固定几何偏移选择，没有按分数挑低背景。

目标比固定背景参照更强的比例约 71%，说明还有相对信息；但背景也可连续占优 13 帧。位移残差比较没有证明低响应邻域最高峰比背景更可靠地跟随目标。框中心只是几何参照，框内峰也受到搜索框限制，不能当精确目标中心或部署跟踪能力。

{decision}

**交付与复核。** [逐目标帧明细：10,400 行](LOW_RESPONSE_OBJECTS.csv)；[表一：42 行](SCORE_SEPARATION.csv)；[表二：42 行](PEAK_LOCATION.csv)；[表三：260 行](TRACK_PERSISTENCE.csv)，附[轨迹分层汇总：42 行](TRACK_SUMMARY.csv)。[冻结规则](PROTOCOL.md)、[视觉记录](visual/REPORT.md)、[独立复算](audit/REPORT.md)和[真实 logits 导出核验](logits/REPORT.md)均已保存。16 个窗口和 4 套融合 logits 的 sigmoid 与旧概率逐像素一致，0.5 前景差为零。三张表和轨迹分层统计已独立复算；本轮结果仍是诊断点匹配，未报告论文 Pd/Fa 或确认集收益。
'''
    (BASE/'REPORT.md').write_text(report)


if __name__=='__main__':
    main()
