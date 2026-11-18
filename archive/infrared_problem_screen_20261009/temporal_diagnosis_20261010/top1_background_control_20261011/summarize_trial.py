#!/usr/bin/env python3
"""Separate budget repair, released-model comparisons, and unsupported claims."""
import csv
import json
from pathlib import Path

BASE=Path(__file__).resolve().parent
LABELS={'release':'发布权重','lambda0_mean':'λ=0，全背景均值','lambda0_top1':'λ=0，top-1%'}


def read(name):
    with (BASE/'evaluation'/name).open() as f:
        return list(csv.DictReader(f))


def number(row,key):
    return int(row[key])


def main():
    rows=read('SUMMARY.csv')
    assert len(rows)==6
    lookup={(r['condition'],r['arrangement']):r for r in rows}
    seqs=read('SEQUENCE_SUMMARY.csv')
    concentrations=read('RECOVERY_CONCENTRATION.csv')
    outcomes=[]
    for arrangement in ['O','S']:
        new=lookup['lambda0_top1',arrangement]
        mean=lookup['lambda0_mean',arrangement]
        release=lookup['release',arrangement]
        m,u,a=map(lambda key:number(new,key),['M','unmatched_components','positive_pixels_outside_boxes'])
        ru=number(release,'unmatched_components');mu=number(mean,'unmatched_components')
        cost_fields=['unmatched_components','positive_pixels_outside_boxes','unmatched_component_area_sum']
        cost_deltas={key:number(new,key)-number(release,key) for key in cost_fields}
        cost_vs_mean={key:number(new,key)-number(mean,key) for key in cost_fields}
        outcomes.append(dict(arrangement=arrangement,
            M=m,U=u,A=a,U_budget=4015,budget_pass=u<=4015,
            U_reduction_vs_mean=mu-u,U_reduction_fraction_vs_mean=1-u/mu,
            recorded_background_costs_reduced_vs_mean=all(v<=0 for v in cost_vs_mean.values()) and any(v<0 for v in cost_vs_mean.values()),
            background_deltas_vs_mean=cost_vs_mean,
            delta_M_vs_release=m-number(release,'M'),
            gain_vs_release_same=number(new,'G_vs_release_same'),loss_vs_release_same=number(new,'L_vs_release_same'),
            loss_vs_release_O=number(new,'L_vs_release_O'),
            low_recovered=number(new,'low_all_05_matched'),low_recovered_tracks=number(new,'low_all_05_matched_tracks'),low_recovered_clips=number(new,'low_all_05_matched_clips'),
            extreme_recovered=number(new,'extreme_all_001_matched'),
            recorded_background_deltas_vs_release=cost_deltas,
            recorded_background_costs_not_increased_vs_release=all(v<=0 for v in cost_deltas.values()),
            exceeds_shared_720_within_U_budget=u<=4015 and m>720,
            preserves_more_original_O_successes_than_release_S=(arrangement=='S' and number(new,'L_vs_release_O')<54),
            auxiliary_old_S_joint_pass=(u<=4015 and m>=720 and number(new,'L_vs_release_O')<=54 and a<=38641) if arrangement=='S' else None,
            no_official_Pd_Fa_claim=True,no_temporal_causal_claim=True,no_novelty_claim=True))
    status={'status':'COMPLETE','one_seed':46,'updates':512,'fixed_top_fraction':.01,'threshold':.5,
            'recorded_background_cost_vector':['unmatched_components','positive_pixels_outside_boxes','unmatched_component_area_sum'],
            'plans':{'O':[0,36,72,108,120],'S':[0,24,60,96,120]},'outcomes':outcomes,
            'interpretation':'Budget pass and equal/no-greater recorded background costs are separate. M/G/L are diagnostic small-object point matches, not official Pd/Fa.'}
    (BASE/'OUTCOME.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
    table=['| 权重／训练配置 | 安排 | M | 新增 / 丢失¹ | U | 框外前景 A | 低响应恢复² |',
           '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        table.append(f"| {LABELS[r['condition']]} | {r['arrangement']} | {int(r['M']):,} | {r['G_vs_release_same']} / {r['L_vs_release_same']} | {int(r['unmatched_components']):,} | {int(r['positive_pixels_outside_boxes']):,} | {r['low_all_05_matched']} |")
    area_table=['| 背景组件面积核查 | 安排 | 总组件数 | 未匹配组件面积总和 | 面积中位数 | 面积q95 | 最大面积 |',
                '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        area_table.append(f"| {LABELS[r['condition']]} | {r['arrangement']} | {int(r['component_count']):,} | {int(r['unmatched_component_area_sum']):,} | {float(r['unmatched_component_area_q50']):g} | {float(r['unmatched_component_area_q95']):g} | {r['unmatched_component_area_max']} |")
    detail=['| top-1%：逐片段结果 | 安排 | M | 新增 / 丢失¹ | U | 低响应恢复 / 分母 | 恢复轨迹 |',
            '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for r in seqs:
        if r['condition']=='lambda0_top1':
            detail.append(f"| {r['group']} | {r['arrangement']} | {r['M']} | {r['G_vs_release_same']} / {r['L_vs_release_same']} | {int(r['unmatched_components']):,} | {r['low_all_05_matched']} / {r['low_all_05_gt']} | {r['low_all_05_matched_tracks']} |")
    paragraphs=[]
    for out in outcomes:
        ar=out['arrangement'];d=out['recorded_background_deltas_vs_release']
        line=f"**{ar}：训练失控与发布参照分开判断。** 相对全背景均值，U 减少 {out['U_reduction_vs_mean']:,}（{out['U_reduction_fraction_vs_mean']:.2%}），"
        line+=f"{'通过' if out['budget_pass'] else '仍未通过'} U≤4,015；相对发布权重同安排，M 净变化 {out['delta_M_vs_release']:+d}，新增 {out['gain_vs_release_same']}、丢失 {out['loss_vs_release_same']}。"
        line+=f"相对同安排发布参照，U 变化 {d['unmatched_components']:+,}、A 变化 {d['positive_pixels_outside_boxes']:+,}、未匹配组件面积变化 {d['unmatched_component_area_sum']:+,}。"
        line+=f"冻结低响应集合实际恢复 {out['low_recovered']}/4,342，涉及 {out['low_recovered_tracks']} 条轨迹、{out['low_recovered_clips']} 个片段；其中极低响应恢复 {out['extreme_recovered']}/4,170。"
        if not out['budget_pass']:
            line+=' 这些新增匹配发生在预算外，不能作为本轮有效检测提升。'
        elif out['exceeds_shared_720_within_U_budget']:
            line+=' 当前 U 预算内超过720，但是否同时减少各项背景成本仍由上述逐项差值判断。'
        elif out['low_recovered']==0:
            line+=' 没有恢复冻结低响应集合，低背景成本本身不足以认定本轮成功。'
        else:
            line+=' 预算内存在低响应恢复；总检出与原成功保持须分别看取舍，不能只报告恢复量。'
        paragraphs.append(line)
    dist=[]
    for r in concentrations:
        if r['condition']=='lambda0_top1' and r['outcome']=='low_recovered':
            frac=float(r['top5_fraction']) if r['top5_fraction'] else None
            dist.append(f"{r['arrangement']} 的低响应恢复涉及 {r['tracks']} 条轨迹；前五轨贡献 {r['top5_frames']} 帧"+(f"（{frac:.2%}）" if frac is not None else '（无恢复，比例不适用）'))
    text='''# 固定 top-1% 背景负监督：单次对照结果

**结果：训练失控有所缓解，仍未通过既定背景预算。** 完整结论与本轮停止边界见[结果判定](FINAL_INTERPRETATION.md)。

训练从相同发布权重重新初始化，seed46、512更新、原数据顺序/正MIL/背景mask/系数/优化器/BN策略不变，effective λ=0。仅把窗口内负损失均值改为最高损失1%均值，外层四窗等权。主 checkpoint 固定最终步；阈值0.5，O/S分别全片应用，未搜索比例、学习率、阈值或其他检查点。

'''+ '\n'.join(table)+'''

¹ 新增/丢失相对发布权重的**相同安排**；S相对原O的保持量另在下文和CSV记录。² 冻结全窗低响应4,342帧中的实际匹配；极低4,170是其子集。M分母为5,200严格小目标帧；U/A按原完整GT范围统计，不把U称为已核实误检或官方Fa。

'''+ '\n\n'.join(paragraphs)+'\n\n'+ '\n'.join(area_table)+'\n\n面积列均以像素为单位。OUTCOME中的背景成本布尔仅比较U、A、未匹配面积总和这三项总量，面积分布另按本表报告。\n\n'+ '\n'.join(detail)+'\n\n'+'；'.join(dist)+'''。这两个片段为既有开发材料，轨迹帧不是独立来源，不能据此宣称新来源泛化。

'''
    sr=lookup['release','S'];sn=lookup['lambda0_top1','S'];so=outcomes[1]
    text+=f"**保持取舍与背景代价。** 原发布 S 为 M=720、相对原O丢失54、U=3,967、A=38,641；top-1% S 为 M={sn['M']}、相对原O丢失{sn['L_vs_release_O']}、U={int(sn['unmatched_components']):,}、A={int(sn['positive_pixels_outside_boxes']):,}。预算通过不自动等于与原S相同背景成本。此前S联合检查（M≥720、L≤54、U≤4015、A≤38641）结果为{'通过' if so['auxiliary_old_S_joint_pass'] else '未通过'}，它没有代替本轮分别判定。\n\n"
    text+='''**解释边界。** 这是自定义弱框适配的工程对照，不是原作者Gaussian训练复现。若有效，证明这次负监督聚合改动改善了当前适配；若失败，只否定本次固定1%配方。没有额外强度控制，不能把效果只归因于困难位置选择：系数不变也会改变入选像素的有效负权重。更不能由此推断时序模块根因或宣称通用新颖性。

**可复核文件。** [六行主表](evaluation/SUMMARY.csv)、[逐片段](evaluation/SEQUENCE_SUMMARY.csv)、[逐轨迹](evaluation/TRACK_SUMMARY.csv)、[逐目标帧对照](evaluation/OBJECT_COMPARISON.csv)、[恢复集中度](evaluation/RECOVERY_CONCENTRATION.csv)、[帧级背景量](evaluation/FRAME_COUNTS.csv)及完整组件面积数组均已保存。配置见[冻结规则](PROTOCOL.md)，训练曲线与原均值/top均值同时见[逐步日志](training/run/TRAIN.jsonl)，单因素源码差异见[优化差异](OPTIMIZATION_ONLY.diff)。[最终独立审计](audit/COMPLETED_AUDIT.json)已通过：640帧从原窗概率重做融合与匹配，10,400对象记录和780条配置×轨迹记录核对一致。本轮结果判定与停止边界已写入[结果结论](FINAL_INTERPRETATION.md)。
'''
    (BASE/'REPORT.md').write_text(text)
    print(json.dumps(status,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
