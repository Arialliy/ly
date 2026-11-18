# M1单处状态更新修订：2176终点作用链对比

状态：COMPLETE。Continuation与Revision均从同一History-2048模型/Adam/RNG开始，完成同序列128步曝光；每臂使用自己的固定2176权重计算history/current_only/off。

下表是pooled_unique、t1–39的目标帧等权均值。每个目标先平均重叠窗口，再等权平均可用O/S安排。背景为沿用的目标配对固定困难背景；不能把重复背景位置当成独立样本。

| 集合 | 种类 | 指标（臂内History−current_only） | Continuation | Revision | 均值差R−C |
| --- | --- | --- | --- | --- | --- |
| low_all_05 | target | after_pda_relative_delta_rms | 0.412333 | 0.411511 | -0.000821823 |
| low_all_05 | target | after_tsfm_relative_delta_rms | 0.0130844 | 0.237451 | 0.224367 |
| low_all_05 | target | injected_H_delta_rms | 7.42786e-06 | 0.000259298 | 0.00025187 |
| low_all_05 | target | logit_delta_rms | 0.000516654 | 0.0272266 | 0.02671 |
| low_all_05 | target | delta_z_anchor_absolute | 0.000343049 | 0.0177098 | 0.0173667 |
| low_all_05 | background | after_pda_relative_delta_rms | 0.946256 | 0.958659 | 0.0124032 |
| low_all_05 | background | after_tsfm_relative_delta_rms | 0.0204468 | 0.457716 | 0.437269 |
| low_all_05 | background | injected_H_delta_rms | 3.68766e-05 | 0.00096586 | 0.000928984 |
| low_all_05 | background | logit_delta_rms | 0.00173363 | 0.0507738 | 0.0490402 |
| low_all_05 | background | delta_z_anchor_absolute | 0.000387593 | 0.0244054 | 0.0240178 |
| extreme_all_001 | target | after_pda_relative_delta_rms | 0.402985 | 0.401922 | -0.00106351 |
| extreme_all_001 | target | after_tsfm_relative_delta_rms | 0.0128972 | 0.233402 | 0.220505 |
| extreme_all_001 | target | injected_H_delta_rms | 7.12849e-06 | 0.000249744 | 0.000242616 |
| extreme_all_001 | target | logit_delta_rms | 0.000503065 | 0.0262042 | 0.0257011 |
| extreme_all_001 | target | delta_z_anchor_absolute | 0.000328555 | 0.0165132 | 0.0161847 |
| extreme_all_001 | background | after_pda_relative_delta_rms | 0.950162 | 0.962558 | 0.0123964 |
| extreme_all_001 | background | after_tsfm_relative_delta_rms | 0.0204849 | 0.459168 | 0.438683 |
| extreme_all_001 | background | injected_H_delta_rms | 3.71188e-05 | 0.00097136 | 0.000934241 |
| extreme_all_001 | background | logit_delta_rms | 0.00174314 | 0.0511276 | 0.0493844 |
| extreme_all_001 | background | delta_z_anchor_absolute | 0.000390042 | 0.0244955 | 0.0241054 |
| both_fixed_success | target | after_pda_relative_delta_rms | 0.776804 | 0.787662 | 0.010858 |
| both_fixed_success | target | after_tsfm_relative_delta_rms | 0.0188412 | 0.393611 | 0.37477 |
| both_fixed_success | target | injected_H_delta_rms | 2.13233e-05 | 0.000709871 | 0.000688548 |
| both_fixed_success | target | logit_delta_rms | 0.00107443 | 0.0558139 | 0.0547394 |
| both_fixed_success | target | delta_z_anchor_absolute | 0.000408897 | 0.0234476 | 0.0230387 |
| both_fixed_success | background | after_pda_relative_delta_rms | 0.693973 | 0.704817 | 0.0108447 |
| both_fixed_success | background | after_tsfm_relative_delta_rms | 0.0177144 | 0.359362 | 0.341648 |
| both_fixed_success | background | injected_H_delta_rms | 2.21934e-05 | 0.000597897 | 0.000575704 |
| both_fixed_success | background | logit_delta_rms | 0.00114291 | 0.0336167 | 0.0324738 |
| both_fixed_success | background | delta_z_anchor_absolute | 0.000264158 | 0.0176247 | 0.0173605 |
| m1_low_common_recovered | target | after_pda_relative_delta_rms | 0.66606 | 0.667164 | 0.00110402 |
| m1_low_common_recovered | target | after_tsfm_relative_delta_rms | 0.0171099 | 0.340314 | 0.323204 |
| m1_low_common_recovered | target | injected_H_delta_rms | 1.5211e-05 | 0.000531835 | 0.000516624 |
| m1_low_common_recovered | target | logit_delta_rms | 0.00100543 | 0.0582526 | 0.0572471 |
| m1_low_common_recovered | target | delta_z_anchor_absolute | 0.000778635 | 0.0493506 | 0.048572 |
| m1_low_common_recovered | background | after_pda_relative_delta_rms | 0.763099 | 0.770218 | 0.00711907 |
| m1_low_common_recovered | background | after_tsfm_relative_delta_rms | 0.0180892 | 0.38193 | 0.363841 |
| m1_low_common_recovered | background | injected_H_delta_rms | 3.02106e-05 | 0.000787073 | 0.000756862 |
| m1_low_common_recovered | background | logit_delta_rms | 0.0015472 | 0.0465973 | 0.0450501 |
| m1_low_common_recovered | background | delta_z_anchor_absolute | 0.000370502 | 0.0195906 | 0.0192201 |

| 集合 | HC固定点目标−背景间隔差：Continuation | Revision | R−C |
| --- | --- | --- | --- |
| both_fixed_success | -9.86379e-05 | -0.00580565 | -0.00570701 |
| extreme_all_001 | 0.000415559 | 0.00933266 | 0.0089171 |
| low_all_05 | 0.000404171 | 0.00802323 | 0.00761906 |
| m1_low_common_recovered | 0.000427617 | -0.0229779 | -0.0234055 |
| strict_all | 0.000326365 | 0.00542723 | 0.00510087 |

CSV保留两臂各自mean/q50/q90与差值。分位数相减是两个边际分位数的差，不是逐对象差的分位数；仅当两臂该指标在全部相同对象上都有定义，均值差才等于配对对象差的均值。缺失值不补零。

每臂内部Δ均为该臂同权重history−current_only。修订臂after_tsfm记录A+TSFM(A,X)的最终状态，不能误称原TSFM纯输出。改动经过递归也可能改变后续PDA输入；阶段幅度比不是受控信息保持率或因果半衰期。

各臂仍输出off原路径以验证基线和固定头。HistoryOff位置Jacobian作用于HC差分，不能读成History−off；9个既有共同恢复目标是结果条件化解释子组。

内部历史差异增大只表明状态依赖改变。持续漏检恢复结论须结合evaluation/step_2176的实际匹配G/L、连续漏检、轨迹覆盖、U/面积和protect646，并比较RevisionHistory−ContinuationHistory及各臂自身History−Current。此汇总不自动支持下一轮训练或M2。
