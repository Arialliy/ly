# Gaussian–SoftIoU 监督近似控制：训练入口核验

状态：CPU 接口与损失检查通过；本核验未启动 GPU、未执行模型前向、未建立优化器或更新任何模型参数。训练入口为 `code/train_supervision.py`，检查脚本为 `code/test_train_supervision.py`，机器可读证据为 `checks/TRAIN_RUNNER_CHECK.json`。

## 冻结范围

- 发布结构与 checkpoint 邻接的 `DeepPro-Plus.py`，参数总数 70,913，无新增模块或参数。
- 原 TRAIN_PLAN 的全部 512 个步骤，每步四个 40×128×128 窗口；复用旧 `SequenceData` 与 `build_batch`，不重新抽样。
- seed 46，fresh Adam，lr=0.0001、weight_decay=0.0001、betas=(0.9,0.999)、eps=1e-8；FP32，关闭 AMP/TF32。
- 所有原参数可训练，包括 BN affine；BN running buffers 保持发布值，每步逐 tensor 检查。
- 固定第 512 次更新后的 checkpoint，无中途挑选或恢复优化器。
- 唯一优化损失为真实官方 `SoftLoUloss`。`frozen_plan_config` 为旧采样计划原样存档，其中历史 `extra_lambda` 不进入本轮计算。

## 官方损失与有效域

已加载 SHA256 为 `fa46a034a30d8bbb0f677271d6368f033fcb389d682478d331551d8bae087668` 的 checkpoint 邻接源码；直接实例化其 `SoftLoUloss`，未复制或改写公式。

该类内部执行 sigmoid，并沿 `(1,2,3)` 求和。官方 `[B,T,H,W]` 输入产生每个窗口的时空 IoU 后等权取均值。入口对每个窗口 boolean gather 有效 logits 与 target，再 reshape 为 `[1,1,1,N_w]`，分别调用该真实类，最后四窗等权。未二次 sigmoid、未添加平滑项、BCE、MIL、top-1%、配对项或参照分支损失。

纯负窗口沿用 smooth=0 的原行为：loss=1，对 logits 的检测梯度为零，并记录其数量。这里的“零梯度”指损失梯度；Adam 的既有动量及 weight decay 仍可能更新参数，不能把它误写成整步参数不更新。有效域为空时直接报错，不静默跳过窗口或修改外层权重。

标签由 root 所有的 `gaussian_labels.py` 提供；入口传入旧背景 mask，调用 `build_targets(entry,sequences,background_masks=masks)`。每步要求 float32 二值 target 与 bool valid，形状均为 `[4,40,128,128]`，并存档 LABEL_CONFIG、标签源文件 hash 和逐步标签 hash。

## 已完成的 29 项 CPU 检查

实际加载了原模型、发布权重、真实官方损失、八个训练来源的指纹、首个冻结训练 batch 和真实构造标签。检查通过：

1. 所有旧源码、checkpoint、TRAIN_PLAN、selection 和训练数据指纹一致，发布模型仍为 70,913 个参数。
2. 全有效域时，gather 调用与官方 `[B,T,H,W]` 调用的损失和 logits 梯度一致。
3. 有效像素数不同的四窗仍等权；部分有效域时，与显式数学 mask 公式的值和梯度一致。
4. 无效像素梯度严格为零；纯负窗口 loss=1、梯度严格为零；空有效域会被拒绝。
5. 首批输入 hash 与已经完成的控制相同；真实标签与 validity 的形状、dtype、二值性及统计契约通过。
6. 真实标签进入官方损失后，合成 logits 的损失和梯度有限；未对模型做前向或参数更新，BN 未变。

通过的入口 SHA256：`451c6df723e7678014250752cbd890c78ee7ca44cc2277418857b4b3145ffe11`。检查文件同时保存测试脚本、标签和官方源码 SHA256；若这些文件更改，应重新运行对应检查。

## 训练记录

入口保存 `INITIAL_STATE.json`（发布状态逐 tensor hash）、`RUN.json`（来源、配置、初末指纹）、512 行 `TRAIN.jsonl`（每步 loss/梯度范数/target mass/valid/positive 数量、数据及标签 hash、BN hash）、`PARAMETER_DIFF.json`（每参数实际变化）和 `final_model.pth`。每个步骤也输出日志行。完成前再次检查源码 hash，末态参数名称与发布结构一致且全部有限。

这是一项 Gaussian 标签生成与有效域均明确限定的监督近似控制，不是作者完整训练配方复现。CPU 检查证明入口与冻结约定一致，不预判训练稳定性、检测收益或背景预算是否通过。
