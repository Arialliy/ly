# PAL 与 SPIRE：监督机制主源核验

核验日期：2026-10-11。范围：论文主源、作者代码静态阅读；没有训练或推理。本文只约束监督与输出路线的相关工作定位，不证明它们能或不能解决当前两个视频片段的失败。

## 书目与版本

| 方法 | 规范条目与去重标识 | 出版状态 | 主源 |
|---|---|---|---|
| PAL | Chuang Yu, Jinmiao Zhao, Yunpeng Liu, Sicheng Zhao, Yimian Dai, Xiangyu Yue. *From Easy to Hard: Progressive Active Learning Framework for Infrared Small Target Detection with Single Point Supervision*. ICCV 2025, pp. 2588–2598. dedup: `arxiv:2412.11154` | CVF 正式会议条目已核实。arXiv v3 是本轮可读完整方法版本；不要用早期 v1 的五人作者表代替正式六人作者表。 | [CVF](https://openaccess.thecvf.com/content/ICCV2025/html/Yu_From_Easy_to_Hard_Progressive_Active_Learning_Framework_for_Infrared_ICCV_2025_paper.html)、[完整方法 v3](https://arxiv.org/html/2412.11154v3)、[代码](https://github.com/YuChuang1205/PAL) |
| SPIRE | Rixiang Ni, Jun Chen, Boyang Li, Yonghao Li, Wujiao He, Yuji Wang, Feiyu Ren, Haoyang Yuan, Wei An. *Rethinking IRSTD: Single-Point Supervision Guided Encoder-Only Framework is Enough for Infrared Small Target Detection*. ECCV 2026, LNCS 17069, pp. 643–660. dedup: `doi:10.1007/978-3-032-37023-5_35`; alias `arxiv:2604.05363` | **已由 Springer 正式章节核实**；2026-09-11 online。不再只是作者仓库“ECCV”声称。最新 arXiv v2 为 2026-08-01；arXiv 作者排序与正式出版排序不同，引用正式版用 Springer 排序。 | [Springer](https://link.springer.com/chapter/10.1007/978-3-032-37023-5_35)、[arXiv v2](https://arxiv.org/html/2604.05363v2)、[作者页](https://nirixiang.github.io/SPIRE-IRSTD/)、[代码](https://github.com/NIRIXIANG/SPIRE-IRSTD) |

CVF 直接 open 曾返回 403，但检索返回正式条目与全文 PDF 文本；PAL 方法以完整 arXiv v3 和官方源码交叉核验。SPIRE Springer 可读元数据与摘要，完整方法读公开 arXiv v2，没有声称读取付费正式全文。

## 任务与机制对齐

| 项目 | PAL | SPIRE | 与当前实验的关系 |
|---|---|---|---|
| 时间输入 | 单帧红外图像 | 单帧红外图像 | 两者都不是本轮 40 帧上下文模型的直接时序机制控制 |
| 训练标注 | 每目标一个 coarse 或 centroid 点，逐步生成稠密伪掩码 | 每目标一个 centroid 点，构造结构化响应图 | 当前有检测框，适配稠密图时采用框内 max 正项；这既不是 PAL 点锚定伪掩码，也不是 SPIRE 点响应回归 |
| 正监督落在哪里 | 点约束伪区域；按模型能力分阶段加入困难样本并更新标签 | 点坐标映射后，Gaussian 与局部归一化辐射强度组成响应先验 | 二者均增加局部空间约束，不能描述为只把目标分数调高 |
| 学习机制 | easy 预启动；coarse outer update 加入可靠困难样本；fine inner update 修正伪标签，衰减抑制无界扩张 | 高分辨率单分支 encoder，stride 4；直接回归响应图 | PAL 属于训练框架；SPIRE 同时改变监督、结构与输出定义 |
| 输出 | 所嵌入网络的像素分数/掩码，再提取组件 | 响应图局部峰，经阈值、候选筛选、亚像素修正和逆坐标映射得到点 | 原始输出和后处理不同，不能把同一个 0.5 当成跨方法公平工作点 |
| 评价 | IoU/nIoU；组件质心匹配及未匹配组件面积型 Fa | 5 px 内一对一点匹配；Precision/Recall/F1；FP 数除图像像素数的 Fa | PAL Fa、SPIRE Fa、当前 U（组件个数）三者不能直接合并为同一种背景预算 |

PAL 的 EPG 在点附近经 Gaussian/Canny/形态学生成候选区域，利用点约束排除假区域。后续 COU 选择漏检率、误检率满足条件的样本；FIU 局部更新伪区域并衰减候选外旧标签。论文还使用 EEDM，将边缘加权与高损失像素选择结合。以上机制来自完整方法第 3 节，不能把 PAL 简化为“随机补一些难样本”。[PAL 方法](https://arxiv.org/html/2412.11154v3)

SPIRE 的 PRPS 是单点加局部响应先验；HRPE 维持 stride 4 输出，使用普通 MSE 回归，并由确定性峰提取产生目标点。最新 v2 将早版的 “gradient-aware” 表述改为 “contrast-aware”；其实现使用归一化局部强度而非显式图像梯度监督。它仍有轻量后处理，不应沿用 v1 里“没有后处理”的绝对表述。[SPIRE v2 方法](https://arxiv.org/html/2604.05363v2)

## 代码核查与可复现资产

本轮浅克隆快照：

- PAL commit `80aac56fd8d639f3a759f171c2002bddc6187ac9`，保存在 `prior_art_sources/PAL/`。
- SPIRE commit `53a236d5d3c551eaf8ec59de3340e97950c5090f`，保存在 `prior_art_sources/SPIRE-IRSTD/`。

### PAL

- 有训练、测试、指标脚本及多种 SIRST 网络；官方 README 提供整理数据下载链接。当前跟踪文件与 README 未发现直接发布预训练权重的条目；测试脚本要求用户指定训练结果目录。此结论仅限本轮检查的快照，不等于全网没有权重。[README](https://github.com/YuChuang1205/PAL/blob/80aac56fd8d639f3a759f171c2002bddc6187ac9/README.md)
- `tools/centroid_anno.m:7` 从掩码连通域计算质心。`tools/coarse_anno.m:13` 在掩码质心附近采样，并要求采样点落在真实前景掩码内。**其 coarse point 不是未经核实的粗框中心。** [标注脚本](https://github.com/YuChuang1205/PAL/blob/80aac56fd8d639f3a759f171c2002bddc6187ac9/tools/coarse_anno.m)
- `loss/Edge_loss.py:28` 将加权逐像素损失排序，并保留不小于中位位置损失的像素求均值。它是全像素高损失选择，并非当前按窗口、仅背景、固定 1% 的控制；但已足以否定“困难像素挖掘本身是新的红外监督机制”。源码边缘赋值次序还需要复现实验时单独核查，不能仅按论文描述假定实现完全相同。[损失代码](https://github.com/YuChuang1205/PAL/blob/80aac56fd8d639f3a759f171c2002bddc6187ac9/loss/Edge_loss.py)
- `test_model.py:245` 使用 sigmoid 后 0.5。`cal_PD_and_Fa.py:38` 组件质心距离 `<3` 匹配；Fa 分子是全部预测组件面积减匹配组件面积。[指标代码](https://github.com/YuChuang1205/PAL/blob/80aac56fd8d639f3a759f171c2002bddc6187ac9/cal_PD_and_Fa.py)

### SPIRE

- 仓库包含两个真实 `.pth` 文件：`SPIRE-SIRST4-512-epoch230-lr0.005.pth`、`SPIRE-UAVB-640-epoch230-lr0.005.pth`；有训练、推理、JSON 点评价和掩码适配评价代码。作者数据采用 SIRST4 与修订过单目标断连掩码的 SIRST-UAVB 版本；原版数据不能直接冒充这一发布设置。[README](https://github.com/NIRIXIANG/SPIRE-IRSTD/blob/53a236d5d3c551eaf8ec59de3340e97950c5090f/README.md)
- `utils/train_eval_utils.py:27` 实际调用 `KpLoss`；`utils/loss.py:39` 是空间 MSE 均值再按 batch 归约乘 2。文件中其他 focal 类存在不代表训练使用它们。[训练入口](https://github.com/NIRIXIANG/SPIRE-IRSTD/blob/53a236d5d3c551eaf8ec59de3340e97950c5090f/utils/train_eval_utils.py)
- `utils/transforms.py:347` 用标注附近约 11×11 的强度最大位置取得局部图块；`434` 起构造响应监督。代码将图块与 Gaussian 相乘，重叠处逐次平均。原标注映射坐标控制热图放置，而局部最大位置控制图块提取，因此不能不经执行核验就把论文里的“质心修正”理解为所有坐标都被移动。[变换代码](https://github.com/NIRIXIANG/SPIRE-IRSTD/blob/53a236d5d3c551eaf8ec59de3340e97950c5090f/utils/transforms.py)
- `evaluate.py:45` 默认响应阈值 **0.35**，`max_num_targets=8`，`tp_distance=5`。`utils/transforms.py:31` 是 3×3 局部极大；`40` 起还有候选强度差约束 `value_range`，代码并非按候选间空间距离执行该过滤。若后来进入拥挤场景比较，8 个候选上限和这一筛选必须保留原生记录，不能静默删除。[评价入口](https://github.com/NIRIXIANG/SPIRE-IRSTD/blob/53a236d5d3c551eaf8ec59de3340e97950c5090f/evaluate.py)
- `utils/metrics.py:42` 是 GT 驱动、最近未用点、距离 `<=5` 的一对一匹配；Fa 使用未匹配点数除像素数。论文 v2 写训练 500 epochs，发布 README 的权重标签为 epoch 230；其含义须视 checkpoint/选择规则核查，不能仅因文件名不同判为错误。[指标代码](https://github.com/NIRIXIANG/SPIRE-IRSTD/blob/53a236d5d3c551eaf8ec59de3340e97950c5090f/utils/metrics.py)

## 现在能借鉴什么，不能声称什么

1. 可以把 PAL 作为“样本可学性与空间标签可靠性联合管理”的监督对照，把 SPIRE 作为“点锚定结构化响应回归＋高分辨率表示”的监督/输出对照。它们提醒我们：框内 max 正监督没有指定可靠的目标锚点或局部响应结构。但这只是机制差别，不是当前失控根因已经被证明。
2. 当前框标注对框检测是完整的监督形式；它对像素分数图才是不完整定位约束。不能因为没有像素 mask，就声称数据“缺少监督”或必须研发弱监督方法。
3. 不能把粗框中心直接变成 PAL/SPIRE 所需真实目标点。必须区分真实人工点、掩码质心、框中心代理，以及由模型/亮度选择的潜在点；不同来源带来的标签误差是额外因素。
4. 现有 paired_max、选窗、top-1% 结果均未运行 PAL 或 SPIRE；不能据此主张近期监督方法共同失效。PAL 中已有高损失挖掘，SPIRE 已有响应先验与峰式输出，通用“加强难背景”“回归热图”“加一个轻量头”都不足以构成当前新颖性论证。
5. 两者使用单帧，不提供关于 DeepPro-Plus 有害时序上下文的因果证据。即使它们将来在当前片段更好，也还需分开数据、监督、预训练、空间结构、输出处理与时间输入的差异。
6. 本轮应保持为相关机制核验，不启动这两套训练。先完成 CHAL 的原生公开设置验证与目标级失败对齐，确认现代框监督背景模型仍留下什么缺口，再决定是否需要借鉴其中一个明确机制。

尚未填满的研究句式仍是：**A 使用 B，但在本项目可复测条件 C 下出现 D；E 补充 B 未提供的约束/信息。** PAL/SPIRE 的 A/B 现在已核查；它们在当前数据上的 C/D 未被测量，不能代填。
