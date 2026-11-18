# 已有适配过程与检查点对照（2026-10-12）

范围：仅复用原结构 top-1% 与 Gaussian/Soft-IoU 近似控制的既有日志、源码、审计和 checkpoint。另用原不可变 Docker 环境在 CPU 上计算发布到终点的参数差异；没有模型前向、推理或训练，没有修改旧产物。

## 1. 检查点库存决定了能回答到哪里

**两条训练路径均只保存了第512步终点，没有独立的中间检查点。**

| 对照 | 保存状态 | 实际保存行为 |
|---|---|---|
| top-1% | `training/run/final_model.pth`，step=512 | 训练循环退出后一次 `torch.save`；入口拒绝非空输出目录 |
| Gaussian/Soft-IoU 近似 | `training/gaussian_softiou_approx/run/final_model.pth`，step=512 | 同样仅在完整512步后一次保存；入口拒绝非空输出目录 |

核对了上述两个实验目录的全部 `.pth/.pt/.ckpt/.safetensors`，并扩展到本项目 temporal_diagnosis 目录检查是否存在旁存副本。后者还有其它实验终点和一个旧 `smoke_real_2steps` 检查点，但都不是这两条训练路径的中间状态。没有证据表明中间权重曾被逐次覆盖；代码本身根本没有中间保存分支。`INITIAL_STATE.json`、输入/标签 hash、标量梯度日志都不是权重快照，两个终点也不含优化器状态。

因此，现有资料**不能回答**：哪一步首次超过背景预算、是否曾有早期受控恢复阶段、训练来源与开发来源的固定检测指标是否同步退化。每步日志都更换输入；训练损失趋势不能补出这些检测轨迹。本轮记录这一缺口，不重训补图，也不事后改终点。

代码定位：[top-1%入口与保存](../../top1_background_control_20261011/code/train_pilot.py) 第318、348、391行；[近似控制保存](../../supervision_alignment_control_20261012/code/train_supervision.py) 第239、322行。库存及 hash 见 [PROCESS_CHECKPOINTS.csv](PROCESS_CHECKPOINTS.csv)、[process.json](process.json)。

## 2. 实际更新规则和终点变化

| 项目 | top-1% | Gaussian/Soft-IoU 近似 |
|---|---|---|
| 模型 / 参数量 | 原始 Plus，70,913 | 相同 |
| 初始化 | 同一发布权重 | 相同 |
| 可训练参数 | 全部64个tensor，包括24个BN affine tensor | 相同 |
| BN running buffer | 36个tensor与发布权重逐位相同 | 相同 |
| 优化器 | fresh Adam；β=(0.9,0.999)，ε=1e-8 | 相同 |
| LR / weight decay | 固定1e-4 / 1e-4 | 相同 |
| 更新 / 种子 | 512 / 46 | 相同 |
| 精度 / 调度 / 裁剪 | FP32、关闭TF32；无scheduler、无梯度裁剪 | 相同 |
| 真正改变的参数 | 64/64 tensor，70,913/70,913元素 | 相同 |
| 相对发布权重全参 L2 位移 | 4.7675% | 9.1508% |

相对位移定义为 `||θ512−θreleased||₂ / ||θreleased||₂`，仅是参数变化描述；不是逐步更新范数或“过大”的判据。各层尺度和参数化不同，也不能按这个比例确定哪个模块造成退化。相同 LR 和步数并未产生相同位移，但这既不证明降低 LR 有效，也不支持直接冻结某层。

| 模块（相对 L2 位移） | top-1% | Gaussian/Soft-IoU 近似 |
|---|---:|---:|
| 空间前端 | 4.2290% | 10.4973% |
| TPro时间映射 | 9.3360% | 10.4921% |
| TPro其余参数 | 6.4197% | 3.5901% |
| 预测头 | 1.9904% | 1.5232% |
| BN affine（跨模块子集） | 1.2254% | 1.1439% |

BN affine 是跨模块子集，不与前四行相加。原架构所有 state 的键、shape、dtype均相同，全部参数有限。实际权重差异见 [PROCESS_PARAMETER_GROUPS.csv](PROCESS_PARAMETER_GROUPS.csv) 和 [PROCESS_PARAMETERS.csv](PROCESS_PARAMETERS.csv)；CPU 命令、脚本、hash 和无GPU执行回执见 [process_checkpoint_receipt.json](process_checkpoint_receipt.json)。

## 3. 日志能证明的训练过程

先固定连续64步分箱，再做统计；不是根据结果寻找某个好区间。下表梯度是当步目标函数对参数的梯度，不含随后由 Adam 处理的 weight decay，也不是参数更新量。不同监督的 loss/梯度尺度不能直接当成检测质量比较。

| 步数 | top-1%全背景被动loss均值 | top-1%梯度范数均值 | 近似控制梯度范数均值 | 近似控制纯负窗数/256 | 近似控制零梯度batch数/64 |
|---|---:|---:|---:|---:|---:|
| 1–64 | 0.002585 | 27.2470 | 2.0937 | 71 | 5 |
| 65–128 | 0.001185 | 16.3284 | 2.3126 | 48 | 5 |
| 129–192 | 0.001623 | 14.7635 | 2.1672 | 46 | 0 |
| 193–256 | 0.003382 | 12.1285 | 1.9772 | 58 | 1 |
| 257–320 | 0.006371 | 7.6377 | 1.9045 | 38 | 2 |
| 321–384 | 0.012822 | 8.2624 | 1.8210 | 30 | 0 |
| 385–448 | 0.011216 | 5.6010 | 1.7781 | 45 | 1 |
| 449–512 | 0.014659 | 6.7212 | 1.8630 | 49 | 2 |

两组均有连续512行、有限 loss/梯度。top-1% 最大梯度范数392.58出现在第7步，近似控制最大值11.63出现在第460步；这些数值既不构成“梯度爆炸”的结论，也不能证明开发集在对应步退化。top-1%全背景被动loss在后段总体更高，但每步训练内容不同、目标函数不同，不能把该列解释为固定来源/固定样本的背景检测成本曲线。

完整1024行标量投影见 [PROCESS_TRAINING_STEPS.csv](PROCESS_TRAINING_STEPS.csv)，16行分箱见 [PROCESS_TRAINING_BINS.csv](PROCESS_TRAINING_BINS.csv)。原审计已证明既有输入/监督构造和BN状态，本轮没有重做帧号、坐标或数据读取审计。

## 4. 一个已有记录直接支持的窄干预对象

Gaussian/Soft-IoU近似控制有 **385/2,048=18.80%纯负窗口暴露**，其中 **16/512=3.125%** 个batch四窗全纯负。385是曝光次数，不是385个独立背景样本。纯负窗口贯穿各64步分箱，并非只在训练末段出现。

对于其 `smooth=0` 的窗口内 Soft-IoU，标签全零、预测和大于零时交集恒零，loss恒1，预测梯度恒零。实际16个全纯负batch逐行满足loss=1、全参数当前数据损失梯度范数=0；同一计划对应的top-1%行均有正的负损失和非零梯度。逐行表见 [PROCESS_EMPTY_BATCHES.csv](PROCESS_EMPTY_BATCHES.csv)。零数据梯度不等于零参数更新：这些参数的梯度tensor仍存在，Adam动量与weight decay仍可作用。含目标窗口中的背景仍有Soft-IoU梯度，不能说全部背景都没有被监督。

因此，可供主报告选择的单因素X是：**仅让这一近似控制中的空目标窗口是否提供数据负监督信号**，保留非空窗口的标签、Soft-IoU、窗口外层归约以及所有其它配置。它比直接降LR、冻结主干或增加蒸馏具有更直接的现存记录依据。这里仅提出可检验对象，不决定补充负项的具体形式/系数，不运行训练。

必须保留反证范围：**top-1%已经有这类负信号却仍超预算**，因此不能用空窗口负梯度缺失解释top-1%的失败，也没有证据证明补足它就足以恢复背景控制。这个检验最多评价近似监督中的一个具体缺口，不能写成官方时序模型缺陷或论文结构贡献。之前固定终点失败与BC-TPro停止决定均不变。

## 5. 证据锚点

- top-1% checkpoint：`3dc5b7ea6a3e5709a8beb5375d2ab811d71cf32de86fac9f7d527cdb194b74b5`。
- 近似控制 checkpoint：`42d5c8de74f4312ceccf97cff209523963073bc26976ff17383e33517f7ca5b5`。
- 发布 checkpoint：`4fd4aa2736d9a33f2bf1459099068d8b4a7e202ff80aecdb4a3277fd13c7abcc`。
- 源码、日志、checkpoint全部hash及库存范围见 [process.json](process.json)；本次只读CPU复算结果见 [process_checkpoint_diff.json](process_checkpoint_diff.json)。
- CPU分析先遇到一次Docker入口参数错误、一次受限安全加载拒绝，均未执行前向或更新；修正入口并复用原runner既有NumPy scalar allowlist后成功。错误回执保留，未更换环境或依赖。
