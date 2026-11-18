# 补充诊断数字的独立复算

核对结果：**PASS**。12 项报告数字核对，0 项不一致；逐轨迹计数与 control_summary/TRACKS.csv 的差异为 0 项。

数据为 DJI_0026_2 与 DJI_0067_2，均是此前用于校准的开发材料，不是新的独立确认集。输入来自根 OBJECTS.csv、WINDOWS.csv，以及 control_summary/OBJECTS.csv、TRACKS.csv。本程序独立重建集合与逐轨迹计数；没有重新计算检测组件，也未访问 GPU。

所有控制数值对应固定阈值 0.5。读取逐对象匹配，重新计算新增与丢失，并与表内 gain/loss 校对。

## 稳定检出集合

集合要求：几何小目标、当前帧至少被两个诊断窗口覆盖、全部可用诊断窗口均匹配、原五窗融合也匹配。这里“全部”指覆盖当前帧的可用窗口，不要求每帧都被 11 个窗口覆盖。逐窗记录独立验证了 any11_matched、match_disagreement 与 multi_context 字段。

- 集合大小：422；逐序列为 {'DJI_0026_2': 233, 'DJI_0067_2': 189}。
- 原五窗概率均值控制新增丢失：0。
- 对称五窗概率 max 控制新增丢失：0。

逐对象证据保存在 STABLE_OBJECTS.csv。此集合全保留不能推出所有原本检出的对象都被保留。

## 额外窗口机会与内部槽位

集合要求：几何小目标、原融合未匹配、任一诊断窗匹配、原五窗中没有任何匹配。内部槽位定义为 8–31（含端点），通过逐窗 frame−window_start 重新计算。

- 额外窗口机会：104。
- 原五窗中已有内部槽位：59。
- 成功窗口中存在内部槽位：77。
- 两条件同时满足：32；均不满足：0。

这两个条件可以重叠，不能相加为互斥分类。自然窗口内容与槽位共同变化，以上计数不是位置的因果效应；更多窗口提供的机会也不是同计算预算可实现的收益。逐对象证据在 EXTRA_OPPORTUNITIES.csv。

## 轨迹净变化

以 (sequence, track_id) 为轨迹键，只纳入几何小目标帧。逐轨迹净值 = 新增匹配帧数 − 丢失匹配帧数；净零可能包含新增与丢失抵消。

| 控制 | 轨迹数 | 净正 | 净负 | 净零 | 新增帧合计 | 丢失帧合计 |
|---|---:|---:|---:|---:|---:|---:|
| official5_probmean | 130 | 1 | 19 | 110 | 1 | 42 |
| symmetric5_probmax | 130 | 14 | 13 | 103 | 74 | 54 |

逐轨迹数据在 TRACK_NET.csv；输入文件 SHA256 与全部核对结果在 RESULTS.json。

复算命令（仅 Python 标准库、CPU）：

```bash
python3 /home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/temporal_diagnosis_20261010/problem_validity_20261011/supplemental_diagnostics/recompute.py
```

以上是诊断点匹配，不是论文 Pd/Fa。不以相邻目标帧作为独立样本做显著性检验。
