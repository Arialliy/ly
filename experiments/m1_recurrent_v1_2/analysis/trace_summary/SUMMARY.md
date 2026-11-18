# M1-v1.2 Current/History：2176终点作用链对比

状态：COMPLETE。V12Current与V12History从同一History-2048旧权重/Adam/RNG及相同新增gate初始化开始，完成同序列128步曝光；两者均pixel更新。V12CurrentCurrent和V12HistoryHistory是独立训练主路，V12HistoryCurrent是同History权重控制。

下表是pooled_unique、t1–39的目标帧等权均值。每个目标先平均重叠窗口，再等权平均可用O/S安排。背景为沿用的目标配对固定困难背景；不能把重复背景位置当成独立样本。

| 集合 | 种类 | 指标（臂内History−current_only） | V12Current | V12History | 均值差R−C |
| --- | --- | --- | --- | --- | --- |
| low_all_05 | target | after_pda_relative_delta_rms | 0.432992 | 0.412681 | -0.020311 |
| low_all_05 | target | after_tsfm_relative_delta_rms | 0.0134432 | 0.013652 | 0.000208782 |
| low_all_05 | target | after_update_relative_delta_rms | 0.251201 | 0.23559 | -0.0156112 |
| low_all_05 | target | injected_H_delta_rms | 0.000175331 | 0.000129156 | -4.61749e-05 |
| low_all_05 | target | logit_delta_rms | 0.0167351 | 0.0134665 | -0.00326857 |
| low_all_05 | target | delta_z_anchor_absolute | 0.0119339 | 0.00880682 | -0.00312712 |
| low_all_05 | background | after_pda_relative_delta_rms | 0.944795 | 0.962724 | 0.0179287 |
| low_all_05 | background | after_tsfm_relative_delta_rms | 0.0209896 | 0.0216273 | 0.000637697 |
| low_all_05 | background | after_update_relative_delta_rms | 0.44413 | 0.451571 | 0.00744124 |
| low_all_05 | background | injected_H_delta_rms | 0.000523664 | 0.00046389 | -5.97738e-05 |
| low_all_05 | background | logit_delta_rms | 0.0306824 | 0.0245132 | -0.00616924 |
| low_all_05 | background | delta_z_anchor_absolute | 0.00968527 | 0.0111312 | 0.00144597 |
| extreme_all_001 | target | after_pda_relative_delta_rms | 0.424058 | 0.402943 | -0.0211152 |
| extreme_all_001 | target | after_tsfm_relative_delta_rms | 0.0132456 | 0.0134438 | 0.000198145 |
| extreme_all_001 | target | after_update_relative_delta_rms | 0.247567 | 0.231538 | -0.0160288 |
| extreme_all_001 | target | injected_H_delta_rms | 0.000170572 | 0.000124578 | -4.59938e-05 |
| extreme_all_001 | target | logit_delta_rms | 0.0162047 | 0.012965 | -0.00323972 |
| extreme_all_001 | target | delta_z_anchor_absolute | 0.0113545 | 0.00821492 | -0.00313955 |
| extreme_all_001 | background | after_pda_relative_delta_rms | 0.948527 | 0.966663 | 0.0181354 |
| extreme_all_001 | background | after_tsfm_relative_delta_rms | 0.0210285 | 0.0216662 | 0.000637736 |
| extreme_all_001 | background | after_update_relative_delta_rms | 0.445469 | 0.453016 | 0.00754617 |
| extreme_all_001 | background | injected_H_delta_rms | 0.000526301 | 0.000466497 | -5.98036e-05 |
| extreme_all_001 | background | logit_delta_rms | 0.030859 | 0.024682 | -0.00617693 |
| extreme_all_001 | background | delta_z_anchor_absolute | 0.0097223 | 0.0111717 | 0.00144942 |
| both_fixed_success | target | after_pda_relative_delta_rms | 0.777571 | 0.790556 | 0.0129852 |
| both_fixed_success | target | after_tsfm_relative_delta_rms | 0.0195392 | 0.0200593 | 0.00052018 |
| both_fixed_success | target | after_update_relative_delta_rms | 0.386781 | 0.389231 | 0.00244988 |
| both_fixed_success | target | injected_H_delta_rms | 0.000388546 | 0.000341991 | -4.6555e-05 |
| both_fixed_success | target | logit_delta_rms | 0.03201 | 0.0271534 | -0.0048566 |
| both_fixed_success | target | delta_z_anchor_absolute | 0.00972739 | 0.0109467 | 0.0012193 |
| both_fixed_success | background | after_pda_relative_delta_rms | 0.695789 | 0.702646 | 0.00685647 |
| both_fixed_success | background | after_tsfm_relative_delta_rms | 0.0177969 | 0.018618 | 0.000821078 |
| both_fixed_success | background | after_update_relative_delta_rms | 0.350033 | 0.351827 | 0.00179465 |
| both_fixed_success | background | injected_H_delta_rms | 0.000334685 | 0.000285008 | -4.96777e-05 |
| both_fixed_success | background | logit_delta_rms | 0.0218991 | 0.0163088 | -0.00559028 |
| both_fixed_success | background | delta_z_anchor_absolute | 0.00687088 | 0.0080376 | 0.00116672 |
| m1_low_common_recovered | target | after_pda_relative_delta_rms | 0.676003 | 0.676247 | 0.00024409 |
| m1_low_common_recovered | target | after_tsfm_relative_delta_rms | 0.0180235 | 0.0182468 | 0.000223275 |
| m1_low_common_recovered | target | after_update_relative_delta_rms | 0.345274 | 0.340285 | -0.00498907 |
| m1_low_common_recovered | target | injected_H_delta_rms | 0.000313307 | 0.000257243 | -5.60633e-05 |
| m1_low_common_recovered | target | logit_delta_rms | 0.0335637 | 0.028515 | -0.00504867 |
| m1_low_common_recovered | target | delta_z_anchor_absolute | 0.0332433 | 0.025683 | -0.00756026 |
| m1_low_common_recovered | background | after_pda_relative_delta_rms | 0.763957 | 0.772968 | 0.00901097 |
| m1_low_common_recovered | background | after_tsfm_relative_delta_rms | 0.0189397 | 0.019271 | 0.000331345 |
| m1_low_common_recovered | background | after_update_relative_delta_rms | 0.376705 | 0.377717 | 0.00101164 |
| m1_low_common_recovered | background | injected_H_delta_rms | 0.000430536 | 0.000378808 | -5.17289e-05 |
| m1_low_common_recovered | background | logit_delta_rms | 0.0271333 | 0.022209 | -0.00492423 |
| m1_low_common_recovered | background | delta_z_anchor_absolute | 0.00776847 | 0.00899553 | 0.00122706 |

| 集合 | HC固定点目标−背景间隔差：V12Current | V12History | R−C |
| --- | --- | --- | --- |
| both_fixed_success | -0.00285259 | -0.00289982 | -4.72323e-05 |
| extreme_all_001 | -0.0014521 | 0.00349011 | 0.00494221 |
| low_all_05 | -0.00206597 | 0.00285093 | 0.0049169 |
| m1_low_common_recovered | -0.0212525 | -0.0130215 | 0.00823099 |
| strict_all | -0.00247221 | 0.00172448 | 0.00419669 |

CSV保留两臂各自mean/q50/q90与差值。分位数相减是两个边际分位数的差，不是逐对象差的分位数；仅当两臂该指标在全部相同对象上都有定义，均值差才等于配对对象差的均值。缺失值不补零。

每臂内部Δ均为该臂同权重history−current_only。after_tsfm记录裸TSFM候选，after_update记录有界插值后实际状态。V12Current臂的HC差分属于额外推理干预，不能替代V12HistoryHistory−V12CurrentCurrent的独立训练比较。阶段幅度比不是因果半衰期。

各臂仍输出off原路径以验证基线和固定头。HistoryOff位置Jacobian作用于HC差分，不能读成History−off；9个既有共同恢复目标是结果条件化解释子组。

内部历史差异增大只表明状态依赖改变。持续漏检恢复结论须结合evaluation/step_2176的实际匹配G/L、连续漏检、轨迹覆盖、U/面积和protect646，并比较V12HistoryHistory−V12CurrentCurrent及V12HistoryHistory−V12HistoryCurrent。没有Frame训练对照，不能声称空间位置选择优于统一更新比例。此汇总不自动支持下一轮训练或M2。

三条正式终点路径的gate与绝对logits见GATE_MAIN_MODES.csv；各臂GLOBAL_RETENTION_SUMMARY.csv先按帧去重再汇总，独立于目标ROI。gate字段中的q05/q50/q95先在每个ROI/全帧内计算，再按原窗口和对象/帧规则取均值，不是把所有像素合成一次分位数。t0没有调用gate，保存缺失值。
