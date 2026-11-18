# Readout-v3 代码检查与本地适配

本地已重新运行用户包内全部 30 项 CPU 测试，全部通过；也实际执行了包内 `check_official_kernel.py`，三种形状的 DEIM 原函数全背景分支在同一均值归约下，损失值和 logit 梯度均通过对照。随后新增的 37 项本地测试全部通过。最终执行收据为 `runtime/loss_cpu_final.json`，汇总见 `resources/LOSS_VALIDATION.json`。这些是新增损失的 CPU 检查；本记录不把它们写成真实视频效果或 GPU 数值验证。

原参考包保存在 `reference/DeepProPlus_C_readout_v3/`，文件未改动。正式适配位于 `train_modules/readout_weighting.py`，只消费已经冻结的 v2 实际 ROI logits 与分段信息，不添加采样器、模型参数或推理操作。已有 F 结果由主流程核对复用条件；本模块的 `unit` 只用于独立等价检查。

## 实测问题与修正

参考 N 实现在 FP32 中用较大的负数计算 `2*logp + log_num - log_den`。它避免了直接概率幂的 0/0，但等分数组仍会出现抵消误差。对 8 个相同背景分数，理论权重均应为 1：

| 背景 logit | 原参考权重 | 本地适配权重 |
| --- | ---: | ---: |
| −40 | 0.9999923706 | 1 |
| −1000 | 1.0002441406 | 1 |
| −10000 | 0.9980487823 | 1 |
| −100000 | 0.9692332149 | 1 |

完整复测见 `NUMERICAL_REVIEW.json`。本地将 **停止梯度的 N 归一化及组合 log-weight 计算改为 float64**，组合完成后再把权重转回原损失计算类型；没有加 epsilon、截断概率或裁剪系数。模型前向和可微损失仍沿原 FP32 路径。所测极负等分数组现在逐位退化为单位权重。对于 −1000 等不可在 FP32 中表示的负概率，背景导数本身仍可能数值下溢；此修正不宣称能创造可表示范围之外的背景梯度。极低目标的正导数保持 −1。

## 实现与归约

- 正项在三种模式都直接计算 `-logsigmoid(q_positive)`；负 BCE 计算 `-logsigmoid(-q_negative)`。它们分别等价于 `softplus(-q_positive)` 和 `softplus(q_negative)`，并保留旧 v2 FRAME 的浮点计算路径。
- `unit` 独立计算新项，没有返回旧 loss 自证。在 FP32/FP64 的实际 ROI、重复分段、空窗口测试中，新旧 FRAME 值、逐段值及原像素梯度均逐位一致。
- D 直接采用作者的 `sigmoid(q_negative).detach().pow(2)` 运算；权重不反传。它会减小新增负监督总量，不能解释成统一增强背景抑制。
- N 在**每个原有 8 帧段**内计算 detached `sum(p)/sum(p^3)`。同一观察出现在多个重叠段时保留多重曝光，不先去重，也不改用整个窗口或 batch 归一化。然后保持段内帧均值、窗口内有效段均值、固定 8 窗口均值。
- 单背景或组内同分数时 N 退化为 F。存在不同分数时，预算性质仅指同一快照、同一归约的负 logit 梯度绝对值之和；不保证网络参数梯度范数、Adam 更新量或独立训练分支间的预算相同。
- 通用 `readout_terms` 提供显式固定 `sample_weight`，同时纳入 N 的分子和分母，由调用者保留原加权归约。正式 v3 段内等权，不使用额外权重。测试覆盖非等权、零权重、无效 padding 和空组。
- 适配器核对与 v2 完全相同的曝光索引、段长度、活动段和各窗口段数。原全背景或缺失分支逻辑由 v2 读出提供，本次不改变候选或掩码。

每段原负预算 `sum(p)` 还要乘 `α_s=1/(8*n_window_segments*L_s)` 才是原外层归约中的贡献。因此诊断分别保存 `raw_segment_*` 和 `*_negative_readout_gradient_l1`，并返回 `group_reduction_weight`。本地测试用窗口段数不同且存在重复观察的输入，核对后者等于对 unique `q_negative` 求导并累加后的 L1；未把 raw 段预算误称为整个新增损失的有效预算。

## 原作者源码对照范围

实际执行的是已核实官方快照中的 `DEIMCriterion.loss_labels_mal` 和 `_get_src_permutation_idx` 原函数体，通过 AST 原样提取，未改其计算。对照使用空匹配、全背景、`gamma=2`、`mal_alpha=None`，并设置同一全 logit 均值归约；另补 FP32/FP64、多类别形状检查。没有复现 DETR 匹配、IoU 质量正标签、完整 MAL、DEIM 网络或训练成绩。

官方文件来自 v2 已核实归档：commit `09d35d53d39ee3145a1e61e3a989b28b9468d1dd`，criterion SHA256 `380a7d86b71ba9a3cbbdb670b4c11a3f2bfd2e367ffe87db7ef8e9ba30806dd2`。原检查脚本本地返回 `git_commit=null`，因为此处是源码快照；该字段不承担作者版本证明。固定提交依据是已有 CVF→作者项目→官方 GitHub 来源链及已验证 Git blob，详见 `DEIM_KERNEL_AUDIT.json`。D 只借鉴作者负例加权，N 预算重标定是本次自定义适配。

没有为本轮新增测试重跑旧模型接入、完整标签核验或评分检查。是否满足正式短训条件，还需主流程的固定真实批次预检与 F 复用核验。
