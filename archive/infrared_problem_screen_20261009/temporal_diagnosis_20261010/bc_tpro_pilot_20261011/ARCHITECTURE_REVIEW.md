# BC-TPro 首版架构审计与实现范围

日期：2026-10-11。执行依据是本轮用户附件中的完整候选与单路动态控制工单。CHAL、TEP-SAM 的完整复现不是本轮前置。本文记录已实现结构和实现检查，不代替训练/检测结果。

## 1. 实际接入对象必须修正

冻结模型为发布 RGB-T 权重旁的 `DeepPro-Plus.py`，而非 `DeepPro.py`。它只有 `self.TPro`，没有 `TPro1/2/3`；32 通道分成 8 个原探针头，每头 4 通道。接入唯一 TPro 的逐头时间 Linear 输出之后、原 `norm1 → ReLU → conv` 之前。原来 8 个 Linear、偏置、BN、后续预测头及输出 tuple 全部保留。

原模型 checkpoint 相邻代码：

`official_deeppro/log/sem_seg/RGB-T__2025-01-14_15-27__SoftLoUloss_DeepPro-Plus_DataL40/DeepPro-Plus.py`

实现：[bc_tpro.py](code/bc_tpro.py)。实际原模型参数为 70,913；两种新配置均为 72,418，新增 1,505。这是本地实际对象的计数，不能直接填入其他论文表中的参数规模。

## 2. 冻结的两个候选

令 pre-TPro 原特征为 `F[B,H,W,T,32]`，背景估计为 `B`，`T=40,r=4`。新关联生成器使用同一套 `LayerNorm(32)`、`Linear(32,4,bias=False)` 查询和键投影，输出过 `tanh`。观测路和背景路共享所有这些参数。

\[
Q(Z)=\tanh(W_Q\operatorname{LN}(Z)),\quad
K(Z)=\tanh(W_K\operatorname{LN}(Z)),\quad
\mathcal A(Z)V=Q(Z)[K(Z)^\top V]/(T\sqrt r).
\]

\[
E=(F-\operatorname{sg}(B))/\sqrt T,
\quad \Delta_{\rm full}=\mathcal A(F)E-\mathcal A(\operatorname{sg}(B))E,
\quad \Delta_{\rm single}=\mathcal A(F)E.
\]

完整模型与单路控制都为 `H_original + gamma * delta`，`gamma` 是从 0 初始化、可正可负的标量。单路仍保留相同背景预测和辅助训练、Q/K 及归一化，故参数数目相同；其实际计算较少，必须分别记录耗时。两者新增模块使用独立 CPU RNG 种子 20261011，`fork_rng` 保持调用者 CPU 随机状态；新模块构建不重置 CUDA RNG。

这不是 `softmax(QK^T)` 的等价加速，也不是直接复现 Differential Transformer。没有加入附件中的可选 null token。本轮不显式存储每像素 40×40 新矩阵；实际为两次小秩 contraction。

## 3. 中心排除必须覆盖原特征感受野

原空间感受野：输入 7×7，两个 STD 残差块各扩展 6，共 19×19，半径 9。时间感受野：输入 5，两个块各扩展 6，共 17，半径 8。差分权重运算不改变上述支持域；原 BN 必须保持使用冻结运行统计，防止训练批次统计带来全空间依赖。

[仅训练标注统计](TRAIN_ANNOTATION_SIZE_STATS.json) 的八个训练片段共 16,707 个有效目标帧，框最长边最大 23。首版为中心原始区域取保守半径 23，所有上下文特征采样偏移须满足 `||offset||∞ > 23+9=32`。因此固定八方向偏移 40，而不是 32；8 个点为 `(±40,±40),(±40,0),(0,±40)`。

该规则保证采样到的 pre-TPro 特征不直接读取该中心区域；**不保证来自整段视频同一运动目标的全部轨迹都被排除，也不保证环上没有其他目标。** 超过训练尺寸范围的新目标也不享有“整框排除”的保证。它是可检查的盲区，不是背景身份的证明。

背景输入只来自 `F.detach()` 的这八个空间位置，随后是按通道时间 3×1×1 卷积、SiLU、1×1×1 卷积。没有中心残差、额外空间卷积、反射填充或循环移位。只有距 crop 边界至少 9 的源特征参与平均，使用有效样本数归一。零上下文位置的修正固定为零。每次前向记录 0…8 个有效采样数的分布、覆盖比例与完整八样本比例。

## 4. 辅助监督与检测梯度相互隔离

背景预测目标为原单位 L1：按通道和有效背景位置取均值，再对批内 4 个窗口等权平均，`lambda_ref=1`。空背景窗口贡献可微零且不改变外层归约。标签仅进入训练的 `reference_loss(background_masks)`。

辅助 mask 以原检测背景 mask 为起点（已排除全部已标注框和原 2 像素保护），再以 pre-TPro RF 作空间 9、时间 8 的排除扩张，边界无效；并要求至少一个有效环样本。实现使用等价可分离 max-pooling，避免直接扫描 17×19×19 大核。**原检测 mask、正项、top-1% 负项不修改。**

辅助目标为 `stopgrad(F)`，预测器输入也停止梯度；检测时 `B` 停止梯度，但共享 Q/K 对背景常量输入的参数梯度保留。于是：

- 检测损失不能更新背景预测器；
- 辅助损失不能更新原模型、Q/K 或 gamma；
- 两种配置均从同一参考分支初始化开始；随后原模型更新不同，参考分支见到的停止梯度特征可以不同，不能要求训练后参考权重仍相同。

`gamma=0` 时 Q/K 的初始检测梯度为零是设计性质；gamma 有非零梯度，背景预测器由辅助项学习。不能把 Q/K 也初始化为零或再放置零输出投影，否则可能令 gamma 同时失去梯度。gamma 打开后 Q/K 必须确实可训练。

## 5. 窗口、边界与资源

保持原 128×128 训练 crop、40 帧、4 窗批次及原检测 core9。新环在 crop 边缘可用样本减少，这是本轮显式记录的上下文截断条件，不使用测试标签补齐。辅助 mask 比原检测 mask 更保守，只作用于新分支。

新空间支持半径至少 `40+9=49`；因此推理 tile halo 使用 64，原 halo16 不足。增大 halo 仅改变同一真实帧的空间计算范围，不增加时间窗口；但增加计算，不能称总预算相同。应检查 halo64 的分块输出与直接前向一致，不能仅验证 gamma0 后就假定训练后的拼接没有缝隙。

`B=4,H=W=128,T=40,C=32` 时，一份 FP32 特征约 320 MiB；全通道 rank4 的单个 Q 或 K 约 40 MiB，四份两路因子约 160 MiB。若显式构造全局两张时间矩阵则约 800 MiB，若误写成原8头各一套全矩阵约 6.25 GiB，仅注意力矩阵已不必要地耗费显存。当前实现避开这部分分配。总峰值仍须实测，因为原 stem、参考预测梯度、LayerNorm 和训练反向也占显存；不能由参数增量推断耗时或显存。

## 6. 实现测试

[test_bc_tpro.py](code/test_bc_tpro.py) 在 CPU、真实发布权重及完整 40×112×112 输入上验证：gamma0 两个原输出逐元素完全相同；两候选初始化相同；原权重键和值不变；参考空间中心排除及边界归一；有效 RF mask；梯度隔离；首步 gamma、辅助参考分支和打开后的 Q/K 可训练；低秩计算与小张量显式矩阵代数一致。

仅验证 `B=F` 时修正为零不充分，因为这时 E 自身已为零。因此还使用任意非零 V，检查两路相同输入的关联输出相减为零，且单路对该 V 的输出非零。模型源码和测试源码的 SHA 随测试结果记录，并检查运行期间未改变。

最终机器可读结果：[checks/IMPLEMENTATION_CHECK.json](checks/IMPLEMENTATION_CHECK.json)。这些检查支持实现与首版设计一致，不证明实际检测收益。

## 7. Differential Transformer 的出处与借鉴边界

*Differential Transformer*，Tianzhu Ye、Li Dong、Yuqing Xia、Yutao Sun、Yi Zhu、Gao Huang、Furu Wei；arXiv:2410.05258（2024），作者机构 Microsoft Research 与 Tsinghua University，ICLR 2025 条目。原方法对同一输入生成两组 Q/K，将两个 softmax 注意力图按可学习系数相减，再作用于共享 V；它没有将第二路约束为背景参照。[论文主源](https://arxiv.org/abs/2410.05258)、[§2.1 的公式与作者代码入口](https://arxiv.org/html/2410.05258v1)、[正式 OpenReview 条目](https://openreview.net/forum?id=OvoCm1gGhN)。本轮 OpenReview 直开触发浏览器验证，作者 OpenReview 主页可检索到 ICLR2025 Oral 记录。

本实现借鉴差分关联思想，而采用低秩、无完整 softmax、辅助背景预测约束的另一种形式。差分本身不属于原创；背景约束是否带来独立收益必须由完整候选相对单路控制及后续普通双路控制验证。论文在语言建模中的噪声抵消效果不能直接证明红外检测中的因果背景分离。
