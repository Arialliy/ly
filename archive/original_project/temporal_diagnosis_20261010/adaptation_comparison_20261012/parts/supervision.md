# 实际监督对照（2026-10-12）

本次只读取冻结源码、既有审计和 512 步日志，未训练、推理或重做坐标/帧号审计。目标固定为真实场景持续漏检的恢复、背景受控与原成功保持；适配退化不是官方时序缺陷的证据。

| 实际进入优化的量 | 原结构 top‑1% | Gaussian 派生二值支撑 + 官方 Soft‑IoU 近似控制 |
|---|---|---|
| 正对象资格 | 所有有效、非 ignore/crowd、完整位于 core9 的框；不限小目标/类别 | 量化前对象资格相同；其后排除对象的 bbox+2 区域覆盖正支撑 |
| 正监督支撑 | 每个对象帧一个 `softplus(-max z)`，没有稠密 mask；最大值唯一时仅该位置有直接正梯度 | 原浮点框中心、σx=w/6、σy=h/6、框外截断；`floor(255*g)>0`，多框最大合并；最终是二值支撑，非连续高斯幅值 |
| 背景/有效域 | core9 减去全部有效框的 bbox+2，包括 ignore/crowd/非完整 core 框 | 从同一旧背景域起步，加回正对象完整框像素；ignore/crowd/非完整 core 框+2 的排除优先；正框内量化成 0 的位置为有效负标签 |
| 窗内归约 | 每对象正项等权均值 + 背景最高损失 1% 均值（k=ceil(.01N)，并列按展平索引）；正/负系数均1 | 对同一窗全部有效时空像素求一次 IoU 比值；没有独立正/负均值、top1 或对象等权 |
| 窗间归约 | 四窗算术均值；extra 已计算但不进优化（λ=0） | 四窗算术均值，含纯负窗，不对有正窗重归一化 |
| sigmoid | 直接在 logits 上 softplus，无外部 sigmoid | 原始 logits 直接进入官方 criterion，内部 sigmoid 一次，无重复 sigmoid |
| 无正目标窗 | 正项为0；若有背景仍有 top1 负项及其梯度 | smooth=0 且 union 非零时恒为 loss=1，损失对整窗预测的梯度为0 |
| 空背景/有效域 | 空背景负项为可微0；实际2048窗均非空背景 | 空有效域报错；实际2048窗有效域均非空 |

源码证据：[旧 batch 构造](/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/temporal_diagnosis_20261010/top1_background_control_20261011/code/train_pilot.py:244)；[框 MIL](/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/temporal_diagnosis_20261010/top1_background_control_20261011/code/losses.py:12)；[top‑1% 与四窗归约](/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/temporal_diagnosis_20261010/top1_background_control_20261011/code/losses.py:48)；[实际量化支撑和有效域](/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/temporal_diagnosis_20261010/supervision_alignment_control_20261012/code/gaussian_labels.py:39)；[masked 官方损失入口](/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/temporal_diagnosis_20261010/supervision_alignment_control_20261012/code/train_supervision.py:85)；[官方 SoftLoUloss](/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/official_deeppro/log/sem_seg/RGB-T__2025-01-14_15-27__SoftLoUloss_DeepPro-Plus_DataL40/DeepPro-Plus.py:84)。官方 loader 本身确实 [将非零 mask 置为1](/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/official_deeppro/data_utils/TrainDataLoader.py:58)，但原作者 Gaussian 生成器仍未获得，不能因此将近似控制称作训练复现。

## 已有记录中可识别的监督差异

两组均有 **148,902 次正对象帧暴露**，每一步四个窗口的量化前正框计数全部相同。不是独立对象数。实际 **385/2,048 窗（18.80%）无正框/无正标签**；两组的空窗位置完全相同。16 个 batch 四窗皆空，步骤为 `27,29,30,36,54,72,111,114,118,128,195,285,301,443,467,469`。近似控制在这16步均记录 `grad_norm=0`；top1同16步梯度范数均大于0。这是既有日志的直接读取，不是重跑梯度实验。

近似控制的全计划标签预检已通过1071项；二值正支撑占原框像素的中位比例 **90%**（min83.33%、Q25 88%、Q75 92.86%、max100%）。量化本身没有清空对象支撑；排除重叠优先使 **66次**对象暴露的支撑全部清空。这没有额外制造整窗空标签：日志中零目标质量始终等价于旧配方无正框。不能把该控制称为连续峰形状回归。

| 512步重复暴露量 | 数量 | 定义 |
|---|---:|---|
| 旧背景有效像素 | 968,194,086 | 用于候选背景负项的像素总曝光 |
| 旧 top1 实选像素 | 9,682,738 | 每步每窗选择后直接负梯度支撑上限，含重复曝光 |
| 新有效像素 | 978,568,892 | 二值监督的全部有效位置曝光 |
| 新正像素 | 9,360,858 | target=1 曝光 |
| 新负像素 | 969,208,034 | target=0 且 valid；纯负整窗并不因此获得负梯度 |

复用 [top1 完成审计](/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/temporal_diagnosis_20261010/top1_background_control_20261011/audit/COMPLETED_AUDIT.json:1)、[top1 损失/掩码审计](/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/temporal_diagnosis_20261010/top1_background_control_20261011/audit/LOSS_MASK_AUDIT.json:1)、[标签预检](/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/temporal_diagnosis_20261010/supervision_alignment_control_20261012/LABEL_PREFLIGHT.json:1)、[新训练72项审计](/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/temporal_diagnosis_20261010/supervision_alignment_control_20261012/TRAINING_AUDIT.json:1)。本次新汇总只对这两份 [top1日志](/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/temporal_diagnosis_20261010/top1_background_control_20261011/training/run/TRAIN.jsonl:1)、[近似控制日志](/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/temporal_diagnosis_20261010/supervision_alignment_control_20261012/training/gaussian_softiou_approx/run/TRAIN.jsonl:1) 求和和逐步比较。

数学解释限于损失：二值标签下令 `I=Σp*y`、`D=Σp+Σy-I`，`L=1-I/D`。有目标且I>0时，正位置对p的导数为 `-1/D`，负位置为 `I/D²`，再乘 sigmoid 导数；背景强度与整窗交并比耦合。全零标签时I=0，D>0，整窗导数为0。不能从这一性质直接断言最终背景失控的根因；top1并无此空窗梯度缺席却也未通过预算。

## 仅登记一个未运行的后续适配控制

**待检验因素：近似配方的纯负窗没有直接损失梯度。** 维持二值标签、有效域、全部有正标签窗口的官方损失、四窗等权和全部训练/评价协议不变；只把 `sum(target[valid])==0` 的窗口改用已存在的 top‑1% 背景 softplus 均值，系数1。

令旧背景掩码为 `B_w`，当前二值目标质量为 `P_w`，则建议规定：

- `P_w>0`：原样调用现有 masked 官方 Soft‑IoU。
- `P_w==0`：`L_w=mean(TopK(softplus(z[B_w]), max(1,ceil(.01*|B_w|))))`，并列规则复用原 top1；不以0.5过滤。
- `B_w` 为空时按旧 top1 定义返回可微0；冻结计划中不存在此情况。整窗 valid 为空仍保留现有错误检查。
- 总损失仍为四个 `L_w` 的等权均值；不丢弃空窗、不重采样，不对正窗重归一化。空窗有效域在本计划上与旧背景域相同；实施前只需断言这一既有不变量，不能默改掩码。
- 从同一发布权重重新初始化，复用 seed46、512步、原参数/BN/优化器/LR/FP32、固定最终512、O/S与0.5。**本轮仅登记，没有实现或运行。**

把常数1替换掉会改变日志总损失的数值零点；常数无梯度，其去留不改变优化。应同时被动记录原官方损失与实际负窗损失/数量，不能通过总loss下降解释检测收益。这项试验检查的是用一项固定负监督补上空窗，不区分困难位置选择与有效负强度，不解释top1自身失败，也不构成论文新模块。最终仍同时检查原漏检恢复、原成功丢失和背景成本；不能用非零梯度或U下降单独判成功，既有512终点失败保持不变。

## 来源指纹

所有全文SHA‑256及完整逐项定义见 [supervision.json](supervision.json)。核心代码：

| 文件 | SHA‑256 |
|---|---|
| `train_pilot.py`（old_batch） | `75e4fe9b4115af837cc1769471328a0b88fca8e546ec8d6f7db73d580f460b54` |
| `losses.py`（old_loss） | `6e7021b9bd7cc362988473449596eb92212312384d51704343928af120af7ed1` |
| `gaussian_labels.py`（new_label） | `a256403d9b3df7670b7e6466e42afd40bcf881b40673c991ba8a0116eeb088b5` |
| `train_supervision.py`（new_train） | `451c6df723e7678014250752cbd890c78ee7ca44cc2277418857b4b3145ffe11` |
| `DeepPro-Plus.py`（official_model） | `fa46a034a30d8bbb0f677271d6368f033fcb389d682478d331551d8bae087668` |
| `TrainDataLoader.py`（official_loader） | `ac94cf5bbfa2a4bcb76f5eeec0e047b999c76d1423afeb1a54136bf535ffa3d4` |
