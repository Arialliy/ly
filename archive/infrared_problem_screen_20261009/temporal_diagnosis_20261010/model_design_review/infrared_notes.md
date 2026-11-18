# 同任务模型设计核查（2026-10-10）

范围：RFR、MISTNet、DTUM；补充 CVPR 2026 InvDet 作为信息保存的边界参考。以下“接入位置/风险/优先级”是针对现有 DeepPro-Plus 的工程推断，不是原论文结论。本轮没有训练或测得改进。

## 1. RFR：最直接的动态融合对照

- 归属：TGRS 2025，DOI `10.1109/TGRS.2025.3542368`。作者 [arXiv 原始记录](https://arxiv.org/abs/2409.12448)关联该 DOI；[原文](https://arxiv.org/html/2409.12448v3)。
- 原模型：PDA 用三层金字塔可变形卷积对齐上一时刻状态与当前特征；TSFM 先进行时间门控，再空间/频率调制。论文 Table VII 中 TM→TSM→TSFM 的 Pd 为 87.697→89.873→91.579，但 Fa 为 44.587→16.201→18.579（×10^-6），所以并非各指标单调改善。Table V 的滑窗 7→20 帧 Pd 从 83.558 降至 75.336，作者已明确讨论更多上下文可能有害。[方法与消融](https://arxiv.org/html/2409.12448v3#S5.SS3)
- [官方实现](https://github.com/XinyiYing/RFR/blob/main/codes/model/RFR_framework.py)的 TSFM 为 `a=sigmoid(sum_c(phi(current)*psi(aligned_history)))`，先 `concat(a*history,current)` 再 1×1 卷积，后续仍可抑制当前特征。它没有“目标响应不下降”的约束，相关性也不是收益的可靠标签。
- 可用方式：完整 RFR+ResUNet 可作为同任务模型对照。若只借鉴模块，可把 DeepPro-Plus 的 `[B,32,40,H,W]` 拆为逐帧 `[B,32,H,W]`，在 TPro 前试 PDA+TSFM；C=32 能适配，但需重新训练。若替掉 TPro，则由原本窗口联合访问改成因果递归，不能把提升全归因于门控。
- 更贴近当前因果问题的“小改动”：保留短支持/局部分支和完整 TPro 分支，在 TPro 后、`conv_out1` 前做逐帧特征融合，并比较 RFR 式相关性门控。这里的保留分支和收益监督是我们待测设计，不应标为原版 TSFM。单纯把 TSFM 接在已经塌缩的特征之后，不能保证补回丢失信息。
- 域与代价：原权重来自 IRSatVideo-LEO 半合成卫星域，不是当前真实 RGB-T 场景。官方依赖 DCN 编译；整段递归减少重复窗口计算，但状态随历史变化，训练需反传多个时间步。复杂度数字不能直接跨输入尺寸引用。[官方仓库](https://github.com/XinyiYing/RFR)

## 2. MISTNet：更直接的“去干扰同时留目标信息”参考

- 归属：官方仓库标注 TIP 2026，2026-04-27 接收；发表 DOI `10.1109/TIP.2026.3689420`，35:5196–5211。IEEE 正文访问未成功，本轮不声称复核了全部论文消融表。[官方仓库](https://github.com/ShuCvlab/MIST)；[DOI](https://doi.org/10.1109/TIP.2026.3689420)。旧作者链接 `GR-ray/MIST` 重定向到 `ShuCvlab/MIST`。
- 原模型：SNCB 用分组空间位移和多尺度邻域匹配补偿历史特征，PDD 用逐级调制过滤瓶颈压缩表示。代码以末帧作为匹配 reference，默认 5→1，四尺度通道为 8/16/32/64。[模型代码](https://github.com/ShuCvlab/MIST/blob/main/deepmist/models/multiframe/MISTNet/model_MISTNet.py)
- 充分性损失需要准确区分：官方代码对瓶颈后预测施加 GT BCE，并用空间展平后的 softmax 做 `KL(p_z || stopgrad(p_v))`；不是两个自然时间窗口的预测一致性，也不是删除上下文后仍能检测的干预充分性。[loss 源码](https://github.com/ShuCvlab/MIST/blob/main/deepmist/losses/loss_sufficiency.py)
- 工程推断：最可借鉴的是在 TPro 前后增加目标监督/诊断头，检查过滤前可检出的目标是否在过滤后失去响应，再测试目标信息保存约束。不能直接将原版 PDD 类接入，因为现有 TPro 输出单尺度、全时间帧特征，PDD 则接四层金字塔、输出末帧。
- SNCB 只应在错配/运动确有贡献时尝试。可在 TPro 前局部补偿，但把全部 40 帧对每个参考帧配准会大幅增加计算。先限制邻帧范围；同时保留原版 MISTNet 完整模型对照，以免把自创适配误写成原模型结果。
- 域与代价：MIST 是合成目标嵌入真实背景的数据，另有 NUDT-MIRSDT 权重。官方需要 NATTEN 0.14.6，5帧末帧设置不能直接与40帧全输出比计算、时延或翻转率。若需逐帧结果，滑窗运行、历史补齐及缓存规则须共同报告。[配置](https://github.com/ShuCvlab/MIST/blob/main/configs/train_MISTNet_MIST.yaml)
- 复现实装注意：`pred_v.detach()` 后教师分支是否得到有效训练，应在复现时逐路检查梯度；本轮只做静态阅读，不判定训练行为或论文错误。

## 3. DTUM：短时运动增强控制，针对性低于前两者

- 归属：TNNLS，在线 2023、卷期 2025，36(1):555–568；[论文 DOI](https://doi.org/10.1109/TNNLS.2023.3331004)，[官方实现](https://github.com/TinaLRJ/Multi-frame-infrared-small-target-detection-DTUM)。本轮出版社全文访问未成功，不给未核对的论文消融数值。
- 原设计：以池化索引编码方向，构造方向卷积与空间 U 形多尺度融合；最终连接末帧特征，原生默认短窗末帧输出。[代码](https://github.com/TinaLRJ/Multi-frame-infrared-small-target-detection-DTUM/blob/main/models/layers.py)
- 工程推断：可用作“局部运动信息是否已经足够”的对照，或为保留局部分支提供 5 帧运动特征。输入 C=32 可设，但原代码输出时间长度1，方向索引处理含5帧假设，不能只把 `num_frames` 改成40 就当兼容 TPro。
- 它不是按上下文可靠性选择收益的模块；若失败由运动特征不足主导，它可能有效。若共同支持已足够而额外内容抑制，DTUM 并未直接约束该现象。
- 代价/域：需为每个绝对帧取同样5帧历史；原代码未配准时依赖额外 alignment 选项。NUDT-MIRSDT 是合成数据；异域checkpoint测得优劣仅为迁移行为。

## 4. InvDet（CVPR 2026）：信息保存可以借鉴，当前不优先移植

- 已通过 [CVF主会记录](https://openaccess.thecvf.com/content/CVPR2026/html/Yan_Target-Aware_Invertible_Encoder_with_Reconstruction_Guidance_for_Infrared_Small_Target_CVPR_2026_paper.html)核准，非 workshop/预印本冒称主会。
- 原设计：可逆编码器配合逆向重建训练，TARM 仅在逆向分支调制，GCTM提供目标相关重建权重；推理移除重建分支。[作者论文PDF](https://github.com/SchulerYan/InvDet/blob/main/paper/InvDet_CVPR2026.pdf)
- 原文 Table 3 检查可逆深度和 block 数，模型约33–122M参数，不是轻量即插即用组件；更多深度没有一致收益。本地已保存作者 PDF 和提取文本。未读到完整补充消融时不声称 TARM/GCTM 单独增益。
- 工程判断：原问题是空间降采样丢信息，DeepPro-Plus 当前诊断是时间上下文变化；TPro 前后均保持 H/W，不应直接套用其空间失真解释。可借鉴“用可检验目标保存约束监督中间表示”，但完整可逆编码器替换会同时改变太多因素。
- [官方仓库](https://github.com/SchulerYan/InvDet)当前说明代码仍在机构审查，只有论文/图示，不能列作立即可运行模型。优先级低于 RFR、MIST。

## 研究决策建议（本项目推断）

可以将“减少不利上下文引发的漏检，并保留有利上下文带来的检出”确定为本轮研究目标；尚未确定最佳解决途径，也没有证据断言它是所有时序红外模型的共同未解问题。

1. 先把已测24个抑制、42个增益和剩余样本作为诊断分层；诊断例子只在验证集使用，训练需要独立划分的同类样本。
2. 在共同数据协议下加入 RFR 与 MISTNet。若原生模型已明显更好，先复现这一结果，再定位是哪种机制缓解问题；已有工作做过不妨碍继续比较改进。
3. 首批模型借鉴只比较：原TPro、额外容量匹配控制、RFR式相关性融合、MIST式目标信息保存监督。是否加入配准由运动/错配证据决定，不先设定 attention 是答案。
4. 判定需要同时统计抑制样本恢复、原有增益保留、新增误报、连续漏检，以及官方原生检测指标。降低翻转率可能来自双低响应，不能单独视为成功。

本轮仍有两项获取限制：MIST与DTUM论文全文/消融未完整读到；InvDet完整代码未公开。以上限制不影响RFR动态融合和MIST公开模型/loss的实际代码核查，但限制性能结论与即刻可运行性。
