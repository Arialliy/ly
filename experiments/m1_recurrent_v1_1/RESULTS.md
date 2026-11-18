**M1-v1.1 单因素短实验已完成，未通过弱目标恢复目标；本轮到此停止，不长训、不加第二处改动、不启动 M2。**（2026-10-14）

唯一结构改动为 `A=PDA(previous,X); M=TSFM(A,X)` → `M=A+TSFM(A,X)`，首帧仍为 `M0=X0`。Continuation与Revision从同一History2048模型、全部Adam状态及RNG起点，在同序已有F缓存0…127上各训练128步，固定2176终点。两臂的曝光摘要相同，原DeepPro-Plus、BN和头部保持冻结；推理时各自使用同一终点权重比较History与current_only。本轮没有另训Current。身份和实际检查点核验见[PAIRED_TRAINING](checks/PAIRED_TRAINING.json)与[评分身份检查](evaluation/step_2176/PAIRED_TRAINING_CHECK.json)。

**主比较没有新低响应恢复，并额外丢失一个保护目标。**下表沿用strict5200的M、全部有效GT匹配后的U、GT框外正像素A和未匹配组件面积。总组件面积等于全图正像素数；U为未匹配组件数，不等同已确认假阳性数。History指旧2048终点。

| 条件 | 安排 | M | U | A | 未匹配面积 | 总组件面积 | low4342 | extreme4170 | protect646 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| release | O | 700 | 4015 | 38368 | 45232 | 98381 | 0 | 0 | 646 |
| release | S | 720 | 3967 | 38641 | 45136 | 99357 | 0 | 0 | 646 |
| History | O | 707 | 4022 | 38714 | 45293 | 98569 | 5 | 0 | 646 |
| History | S | 724 | 4011 | 38961 | 45515 | 99409 | 7 | 0 | 644 |
| ContinuationHistory | O | 707 | 4016 | 38729 | 45285 | 98614 | 5 | 0 | 646 |
| ContinuationHistory | S | 724 | 4006 | 38980 | 45545 | 99458 | 7 | 0 | 644 |
| ContinuationCurrent | O | 707 | 4016 | 38731 | 45287 | 98620 | 5 | 0 | 646 |
| ContinuationCurrent | S | 724 | 4007 | 38980 | 45547 | 99463 | 7 | 0 | 644 |
| RevisionHistory | O | 707 | 4018 | 38717 | 45282 | 98587 | 5 | 0 | 646 |
| RevisionHistory | S | 723 | 4006 | 38976 | 45547 | 99438 | 7 | 0 | 643 |
| RevisionCurrent | O | 707 | 4026 | 38806 | 45442 | 98885 | 5 | 0 | 646 |
| RevisionCurrent | S | 724 | 4021 | 39059 | 45562 | 99755 | 7 | 0 | 644 |

RevisionHistory相对ContinuationHistory：O匹配完全相同；S的G/L=0/1，M由724降到723。low仍为O5/S7，极低响应4170仍为0；共同恢复9个唯一目标帧的全部O/S匹配模式保持不变。保护集合双安排都成功由644降到643，新增损失仅发生在S，未出现两个安排都丢失。O的U=4018仍高于固定预算4015；S的U=4006在预算内，却比release的3967多39。

**比较范围和代价。**G/L逐安排计算；全部有效GT为16297个目标帧，包含strict5200。差值均为左侧条件减参考条件。

| 条件 / 参考 | 安排 | strict G/L | 全部GT G/L | ΔU | ΔA | Δ未匹配面积 |
| --- | --- | --- | --- | --- | --- | --- |
| RevisionHistory / ContinuationHistory | O | 0/0 | 1/0 | 2 | -12 | -3 |
| RevisionHistory / ContinuationHistory | S | 0/1 | 0/3 | 0 | -4 | 2 |
| RevisionHistory / RevisionCurrent | O | 0/0 | 4/5 | -8 | -89 | -160 |
| RevisionHistory / RevisionCurrent | S | 0/1 | 2/11 | -15 | -83 | -15 |
| ContinuationHistory / ContinuationCurrent | O | 0/0 | 0/0 | 0 | -2 | -2 |
| ContinuationHistory / ContinuationCurrent | S | 0/0 | 0/0 | -1 | 0 | -2 |
| RevisionHistory / History | O | 0/0 | 2/2 | -4 | 3 | -11 |
| RevisionHistory / History | S | 0/1 | 1/7 | -5 | 15 | 32 |
| ContinuationHistory / History | O | 0/0 | 1/2 | -6 | 15 | -8 |
| ContinuationHistory / History | S | 0/0 | 1/4 | -5 | 19 | 30 |

RevisionHistory相对自身同权重RevisionCurrent，确实使O/S的U减少8/15、A减少89/83、未匹配面积减少160/15；同时没有新增strict匹配，S仍损失上述保护目标，全部有效GT的G/L为4/5与2/11。因此该历史效应主要表现为抑制，不能用背景代价下降替代目标恢复。Continuation的History/current控制在全部GT匹配上完全相同，仅有轻微背景代价变化。相对旧History2048，Continuation在strict上无变化、全部GT净减1/3；Revision在strict净变0/−1、全部GT净变0/−6。额外训练自身的变化已经由Continuation单列。

主比较RevisionHistory−ContinuationHistory在全部GT上只有以下4条改变；其中唯一strict改变属于既有保护集合。

| 安排 | 视频 | frame / source_frame | annotation_id | track_id | strict | 变化 |
| --- | --- | --- | --- | --- | --- | --- |
| S | DJI_0026_2 | 43 / 43 | 804942 | 400001 | 0 | 丢失 |
| S | DJI_0067_2 | 46 / 46 | 871501 | 7700508 | 0 | 丢失 |
| S | DJI_0067_2 | 101 / 101 | 876734 | 7702014 | 1 | 丢失 |
| O | DJI_0067_2 | 102 / 102 | 876775 | 7700541 | 0 | 恢复 |

**连续性没有改善。**新增丢失对象为 `DJI_0067_2 / frame101 / annotation876734 / track7702014 / S`。该轨迹在冻结观察集合中有82个目标帧，缺标/时间断点切段，未证明是完整物理可见区间。其中匹配从5降到4，覆盖从5/82降到4/82，最长连续漏检从14延长到17。其原有孤立010命中被删除，造成全体S的孤立命中21→20、匹配段158→157；这不是连续性提升。长度至少2的匹配段仍为O79/S80。O的130条轨迹最长漏检全部相同，S有1条变长、129条相同、0条变短；low4342和extreme4170的连续漏检记录未改善。相对RevisionCurrent和旧History同样出现这条轨迹损失。逐对象与完整连续性来源见[对象改变](evaluation/step_2176/REVISIONHISTORY_VS_CONTINUATIONHISTORY_CHANGED_OBJECTS.csv)、[轨迹覆盖](evaluation/step_2176/TRACK_COVERAGE.csv)与[连续漏检](evaluation/step_2176/CONTINUOUS_MISS_SUMMARY.csv)。

**历史差异被保留并传到logit，但作用方向没有形成新的弱目标恢复。**以下是每臂自己的History−current_only差，不是两臂直接相减。low表仅覆盖具有t1–39观测资格的4307/4342个目标帧：先平均重叠窗口，再平均可用O/S安排，最后等权平均目标帧。Revision中的“更新后”是 `A+TSFM(A,X)` 的最终状态。相对差的分母为同层两模式RMS平方均值的平方根，不能解释成信息保持率。

| low目标指标（4307个） | Continuation | Revision |
| --- | --- | --- |
| PDA后相对状态差 | 0.412332775 | 0.411510952 |
| 更新后相对状态差 | 0.0130843745 | 0.237451285 |
| 接入H差RMS | 7.42785661e-06 | 0.000259297797 |
| logit差RMS | 0.000516653619 | 0.0272266252 |
| 固定目标锚点有符号Δz | 4.8495945e-05 | -0.0163735093 |

更新后相对差从1.30844%增到23.7451%，接入H差从7.42786e−6增到0.000259298，logit差RMS从0.000516654增到0.0272266。这些实测说明差异已被保留和传递，不能将本轮失败仅归因于历史差异幅度消失或冻结预测头阻断。但Revision的low目标固定锚点有符号Δz均值为−0.0163735，仅11.2143%的有资格目标帧为正；配对背景为−0.0243967。**low间隔均值增加+0.00802323来自背景下降更多，不能表述为弱目标响应增强。**

同样方向在保护集合和既有共同恢复子组上需要保留：

| 集合（t1–39资格/全集合） | Revision目标Δz | 配对背景Δz | 目标−背景间隔变化 |
| --- | --- | --- | --- |
| low 4307/4342 | -0.0163735093 | -0.0243967424 | 0.00802323312 |
| protect 643/646 | -0.0234245457 | -0.0176188967 | -0.00580564899 |
| 共同恢复 9/9 | -0.0422928929 | -0.0193150308 | -0.0229778621 |

完整t1–39资格为strict5162/5200、low4307/4342、extreme4135/4170、protect643/646及共同恢复9/9；仅首帧观测留在t0层。最终匹配评价仍覆盖全部冻结集合，未按这些资格数过滤。共同9按v1结果定义，属于解释性子组；固定困难背景坐标沿用release选择，重复背景位置不算独立样本。低响应目标状态差与背景状态差同时扩大；结合有符号响应、实际匹配与保护损失，本轮暴露的是这些新增保留差异在当前终点的判别作用和方向未满足恢复需求，不能推断PDA对齐信息已被证实对目标有用。详细分布、缺失计数与每对象数据见[两臂作用链汇总](analysis/trace_summary/SUMMARY.md)、[阶段比较](analysis/trace_summary/STAGE_COMPARISON.csv)和[有符号目标/背景间隔比较](analysis/trace_summary/PAIRED_MARGIN_COMPARISON.csv)。

**实际计算成本。**两臂总参数均为180263、训练参数均为109350，修订新增参数为0。下表来自各臂RUN实测；显存是CUDA已分配内存峰值，GiB按2³⁰字节换算。

| 训练臂 | 新增步数 | 耗时（秒） | 峰值已分配显存（GiB） |
| --- | --- | --- | --- |
| Continuation | 128 | 1011.524 | 4.8934 |
| Revision | 128 | 1073.159 | 4.8944 |

推理收据记录的是三模式共享前端、完整trace、ROI统计/Jacobian及产物生成的诊断执行。**这些耗时不能当作部署时延或生产推理基准。**每条视频使用原8个40帧窗口：

| 诊断臂 | 视频 | 窗口数 | 耗时（秒） | 峰值已分配显存（GiB） |
| --- | --- | --- | --- | --- |
| Continuation | DJI_0026_2 | 8 | 199.458 | 7.2096 |
| Continuation | DJI_0067_2 | 8 | 206.745 | 7.2096 |
| Revision | DJI_0026_2 | 8 | 194.759 | 7.2096 |
| Revision | DJI_0067_2 | 8 | 198.593 | 7.2096 |

成本数值及对应训练RUN/推理COMPLETE的SHA256收录在[RESULTS.json](RESULTS.json)的 `cost` 字段；没有把冻结模型总参数数目与新增参数数目混用。

**结果判定。**这一次128步、固定2176终点的单因素改动达成了“保留更多历史差异”，没有达成“新增弱目标恢复且保住原成功目标”。按照[冻结协议](PROTOCOL.md)，本轮停止，不追加长训或M2。结论限定于该配对短实验及两个开发视频的原实际组件诊断；不称官方Pd/Fa，不把一次失败推广为所有历史机制无效，也不把未发生的后续训练写成证据。

两臂终点SHA256：Continuation `c409006206a95096b74da5f9b954831e4aa53587f69a605cf12720b5b2f5d785`；Revision `b4622c1731c2f68496bd5064e6e890ccc624c06aa6b8c69934a228adffa867d1`。训练、推理、评分及机制聚合均已完成；冻结code哈希核验通过。完整12行评分、全部GT G/L、所有改变对象、连续性变化和来源哈希见[RESULTS.json](RESULTS.json)。
