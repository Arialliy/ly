# 原 DeepPro-Plus 监督对齐：主源核验与单次近似规则

日期：2026-10-12。范围：冻结官方代码静态核验、用户指定论文主源核对、复用已核验 SPIRE 材料；未运行 GPU、未修改旧代码、未搜索 Gaussian 超参数。本文件不是实验结果。

**可执行的控制应称为“Gaussian 派生二值支撑＋原 Soft-IoU 归约的有效域适配”，不能称为原作者连续 Gaussian 热图监督的精确复现。** 当前已知原 loader 会把任何非零标签像素转为 1；未知的是作者如何从框生成、截断及保存那张掩码。

## 1. 主源实际确认了什么

| 核验项 | 证据与结论 |
|---|---|
| RGBT-Tiny 的训练标签 | 论文 v5 §IV-F 说明用作者提供的框生成粗 Gaussian 掩码；没有给出中心、σ、截断、量化与重叠规则。[论文 §IV-F](https://arxiv.org/html/2506.12766v5#S4.SS6) |
| 损失类别 | v5 §IV-A3 明确采用 Soft-IoU；40 帧、128×128 crop、batch 4 等作者训练描述在同节。精确归约以冻结代码为准。[论文 §IV-A3](https://arxiv.org/html/2506.12766v5#S4.SS1.SSS3) |
| 掩码如何载入 | [TrainDataLoader.py](/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/official_deeppro/data_utils/TrainDataLoader.py:58)：图像文件标签转 float32 后除以 255；紧接第 59 行 `label[label > 0] = 1.`。RGB-T 的 resize 分支被注释（56–57 行）。因此保存的灰度幅值不会作为连续监督强度保留。 |
| 标签文件来源 | [标签路径](/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/official_deeppro/data_utils/TrainDataLoader.py:150)：RGB-T 从 `segmentations/<sequence>/` 读取并排序；不是在 loader 内从 JSON 框生成。此次对冻结仓库 Python/Markdown 的 Gaussian 相关搜索未找到其 RGB-T 掩码生成器；不能因此宣称全网不存在。 |
| 实际训练入口 | [训练损失选择](/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/official_deeppro/train.py:137) 实例化 `MODEL.SoftLoUloss()`；[训练调用](/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/official_deeppro/train.py:186) 将第二个模型输出 logits 与 targets 直接传入。BCE 等备选只是注释，不能列为实际配方。 |
| 精确损失 | [冻结 DeepPro-Plus.py](/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/official_deeppro/log/sem_seg/RGB-T__2025-01-14_15-27__SoftLoUloss_DeepPro-Plus_DataL40/DeepPro-Plus.py:84)：sigmoid，按维度 `(1,2,3)` 求交并比，之后 batch 均值，`smooth=0.00`。源码符号是 `SoftLoUloss`（大写 L），数学含义为 Soft-IoU。 |

冻结官方 commit：`8fa1a68b94eb22e94ccd0529e6c5ceccdaa7ec28`。checkpoint 相邻的 `DeepPro-Plus.py` 与 `networks/models/DeepPro-Plus.py` SHA-256 相同（见配套 JSON），本结论不是从 `DeepPro.py` 外推。

## 2. 原 Soft-IoU 的公式与边界

令网络 logits 为 z∈R^(N×T×H×W)，p=sigmoid(z)，二值目标为 y。每个窗口 n：

\[
 I_n=\sum_{t,h,w}p_{nthw}y_{nthw},\quad
 P_n=\sum_{t,h,w}p_{nthw},\quad
 Y_n=\sum_{t,h,w}y_{nthw},\qquad
 L=1-\frac1N\sum_n\frac{I_n}{P_n+Y_n-I_n}.
\]

它不是“先逐帧 IoU 再均值”、不是把所有 batch 像素合为一个 IoU，也没有 BCE/top-1%/配对项或正负项系数。训练损失不先将预测以 0.5 二值化；0.5 仅用于当前固定检测评价。

原式没有 epsilon。若某窗口 Y_n=0 且 P_n>0，则该窗 L_n=1、对 z 的梯度为 0；若连 P_n 也为 0 则产生 0/0。不要为保持数值稳定而静默增加 smooth 或纯背景 BCE，再称原损失。统计并披露预定计划中空支撑窗口及非有限数即可；非有限情况须显式失败。

本次若继续使用 old background mask、core=9 和忽略区，则是**masked Soft-IoU 适配**：将每个求和项限制于固定有效域 D_n，但保持窗口间等权。这一有效域不是原作者 loss 的功能。实现时应在 sigmoid 之后排除无效像素；只将其标签置 0 会把它们当背景，将其 logits 置 0 会引入 p=0.5，两者均非忽略。全 1 有效域时，应与原类核对 loss 值及 logit 梯度。

## 3. 单一近似规则：冻结后只运行一次

以下是本项目预先选定的近似，不是从论文恢复出的作者参数，也不是经开发集选择的最优参数。

1. 对沿用旧 `positive_annotation && full_core` 的每个原始浮点 bbox `(x,y,w,h)`，设 `cx=x+w/2`、`cy=y+h/2`、`sigma_x=w/6`、`sigma_y=h/6`。中心只是框几何中心代理，不声称为真实目标/强度质心。
2. 全图像素 `(i,j)` 在中心 `(j+0.5,i+0.5)` 取样。只有中心属于半开框 `[x,x+w)×[y,y+h)` 才计算：

\[
 g(i,j)=\exp\left[-\frac12\left(\frac{j+0.5-c_x}{w/6}\right)^2-\frac12\left(\frac{i+0.5-c_y}{h/6}\right)^2\right].
\]

   框外为 0。多个合格目标取逐像素 max。不要先把 bbox 取整后重新求 σ/中心；旧整数 rect 只用于枚举候选像素。若直接在 crop 内生成，应使用平移后的原浮点框，数值等价于先全图生成再裁剪。
3. `u=floor(255*g).astype(uint8)`，随后按已核验 loader 语义生成 `y=(u>0).float32`。量化中间值与最终二值标签应分清。`u>0` 等价 Gaussian 值至少 1/255（忽略浮点边界）；先将连续 g>0 二值化会得到整个矩形，改变本规则。
4. 有效域沿用同一 512 步计划/core9。默认候选形式 `D0=old_bg_mask | union(full_positive_rect)`，完整正框中量化为 0 的位置是稠密负监督；旧框外 2 像素保护带保持忽略。**已冻结重叠优先级：最终 D 从 D0 中强制去掉 ignore/crowd/partial 等排除对象的原浮点框加 2 像素保护带；排除优先于完整正框的纳入。** 该规则写入 LABEL_CONFIG；valid 域外标签同时置 0。
5. 不做图像峰值对齐、不按响应调 σ、不强制补中心正像素。预先统计每个目标量化后/有效域内的正支撑、框内占比、空支撑对象及整窗无正标签数；若出现为空，保留原规则并明确其影响。

旧监督的正对象/几何范围见 [正对象筛选](/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/temporal_diagnosis_20261010/stage1_goal/pilot/frozen_v1/train_pilot.py:63)、[完整 core 范围](/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/temporal_diagnosis_20261010/stage1_goal/pilot/frozen_v1/train_pilot.py:72)；旧负域排除所有有效标注框及 margin、再记录完整正框的执行顺序见 [旧 build_batch](/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/temporal_diagnosis_20261010/stage1_goal/pilot/frozen_v1/train_pilot.py:244)。本规则保持旧计划，但将“框内任选高点”换成具有预定空间支撑的标签并更换归约损失，因此收益仅能归因于这个**监督组合控制**，不能单独归因为 Gaussian 或 IoU。

## 4. 必须随结果披露的偏离

- 作者 Gaussian 的 σ、中心、支撑截断、保存 dtype/取整、重叠及异常标注处理未知；本规则只对齐已核验的“框生成掩码→非零二值化”语义。
- 本项目继续从发布权重做固定 512 更新的受控微调，不是论文 32 epoch 训练的重跑；固定窗口/crop 计划、训练数据范围、学习率、BN 冻结及 checkpoint 规则均需写入实际 manifest。官方入口 [官方 train 模式](/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/official_deeppro/train.py:179) 使用 `detector.train()`，loader [官方 crop 逻辑](/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/official_deeppro/data_utils/TrainDataLoader.py:86) 重新随机采样目标/随机 crop；不能把沿用旧计划和冻结 BN 描述为原配方。
- core9、保护带、ignore/crowd/partial 的有效域是当前标注适配约定；原 `SoftLoUloss` 没有 mask 参数。真实生成标签和有效域需可追溯，不能仅保存公式。
- 阈值 0.5、O/S 推理、实际匹配与背景指标继续沿用冻结诊断协议。若有收益，首先说明原架构在监督对齐近似后有新的开发集取舍；不能据此证明原时序机制根因、精确复现论文 Pd/Fa、或提出新模型贡献。

## 5. SPIRE 的位置只需复用已有核验

既有 [既有 SPIRE 机制核验](/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/temporal_diagnosis_20261010/recent_method_alignment_20261011/SUPERVISION_PRIOR_ART.md:27) 与 [既有 SPIRE 代码核验](/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/temporal_diagnosis_20261010/recent_method_alignment_20261011/SUPERVISION_PRIOR_ART.md:46) 已核实：SPIRE 以点锚定、Gaussian×归一化局部辐射强度构造连续响应，使用 MSE 及专用峰提取。它不是当前官方 loader 的二值支撑＋Soft-IoU，也不是纯框中心 Gaussian。其正式出处为 [ECCV 2026 Springer 章节](https://link.springer.com/chapter/10.1007/978-3-032-37023-5_35)，完整方法 [arXiv v2](https://arxiv.org/html/2604.05363v2)，已固定代码 commit `53a236d5d3c551eaf8ec59de3340e97950c5090f`。本轮无需再检索或运行 SPIRE；本次近似控制不应冠名 SPIRE 复现。

**结论：Gaussian 值本身并不是官方实际监督的强弱权重，保存后的非零支撑才是。单次近似可以立即执行，但必须把未知生成参数、有效域适配和微调设置与已核验的原损失语义分别记清。**
