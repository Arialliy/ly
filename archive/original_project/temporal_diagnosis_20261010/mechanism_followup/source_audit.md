# 官方 DeepPro-Plus 时序实现审计

日期：2026-10-10。范围：官方 RGB-T checkpoint 及其邻接模型源码；只读 CPU 权重检查，没有 GPU 推理或修改官方源码。以下结构事实是机制候选依据，不构成漏检根因或方法提升证明。

## 实际时序处理

基准目录为 `../../official_deeppro`（相对此文件目录），检查模型为 `log/sem_seg/RGB-T__2025-01-14_15-27__SoftLoUloss_DeepPro-Plus_DataL40/DeepPro-Plus.py`，与 `networks/models/DeepPro-Plus.py` 中相关实现一致。

- `networks/layers/TPro.py:18–40`：8 个独立 `nn.Linear(40,40)` 作用在时间轴，每个 head 对其 4 个通道共享同一时间矩阵。代码中的 `QK_heads` 是容器，forward 逐 head 索引使用，不是将 8 个 Linear 依次串联。矩阵与 bias 不随视频内容改变，并非内容自适应 attention。没有额外显式位置编码，但输入／输出绝对槽位已由矩阵行列参数化。
- `TPro.py:31–36`：特征先除 `sqrt(40)`，再经过 Linear；Linear 的 bias **不除** `sqrt(40)`。一个 head 的输出槽 bias 重复作用于该 head 的 4 个通道。
- `networks/models/DeepPro-Plus.py:16–22,25–36`：入口时空卷积 → 两个 STD_Resblock → TPro → 点式检测头。TPro 无当前帧特征的显式旁路残差。
- `networks/layers/basic.py:48–57,99–107,114–118`：差分卷积最终调用零 padding 的 `F.conv3d`。时间感受野半径为入口 2 + 两块各 (2+1) = **8 帧**；空间半径为入口 3 + 两块各 (1+2) = **9 像素**。因此前端特征在时间槽 8..31 不直接受该 40 帧窗口两端 padding 影响，但 TPro 会使用所有 40 个特征槽，包括边界受影响特征。
- `temporal_diagnosis_20261010/window_offset_experiment/infer_offsets.py:75–95,129–135`：固定全局均值／标准差，加载官方 checkpoint，`eval()` 推理。BatchNorm 使用 checkpoint 中的固定统计，不能把窗口变化解释为推理时的窗口统计归一化。
- `official_deeppro/test.py:150–159`：原部署参照是重叠窗口逐像素最大值融合；任何诊断单窗口收益都必须与此实际基线区分。
- `official_deeppro/data_utils/TrainDataLoader.py:64–84,162–169`：已随机采样多个窗口结束位置，短序列有前／后补零。因此新增随机起点本身不是新训练策略；配对曝光、监督或平移约束是否不足尚待独立证据。

## 官方权重的非平移共享统计

只读 CPU 载入 `checkpoints/best_model.pth`（epoch 21）。执行环境为已有 `deeppro-official-eval:20261009` Docker 镜像，无 GPU 参数，官方目录只读挂载。计算只输出到标准输出，未改 checkpoint 或源码。

定义：Toeplitz 近似 T 将 W 中相同 `output_index-input_index` 对角线的元素替换为其均值；相对残差为 `||W-T||F/||W||F`。相邻槽位共同权重差为 `||W[1:,1:]-W[:-1,:-1]||F/||W[:-1,:-1]||F`。这些指标只刻画权重不满足时间平移共享，不能衡量检测损失或替代激活贡献。

| head | W shape | ||W||F | Toeplitz relative residual | one-step relative difference | bias min | bias max |
|---|---|---:|---:|---:|---:|---:|
| 0 | 40×40 | 2.1345773160 | 0.5139769248 | 0.6817555494 | -0.0498612858 | 0.2435310036 |
| 1 | 40×40 | 0.6276158481 | 0.9480873977 | 1.3324382181 | -0.0656082779 | 0.1262657791 |
| 2 | 40×40 | 0.9644085267 | 0.8889073859 | 1.2269293847 | -0.0347952172 | 0.1689381897 |
| 3 | 40×40 | 1.4153319644 | 0.6050441187 | 0.8157796334 | -0.0614044517 | 0.2705469429 |
| 4 | 40×40 | 2.0500285987 | 0.5399745256 | 0.7849292484 | -0.0953251943 | 0.1621774733 |
| 5 | 40×40 | 1.3718102769 | 0.9288402007 | 1.2816872831 | -0.1522122025 | 0.1716138273 |
| 6 | 40×40 | 2.0544143081 | 0.3776514736 | 0.4841127215 | -0.1185485050 | 0.1722966582 |
| 7 | 40×40 | 1.9335775973 | 0.5306891114 | 0.7894607805 | -0.0809307173 | 0.1330183297 |

## 最小推理及精确分解

1. 全 1,417 个严格小目标帧的既有最小／最大响应窗先精确重放，包含翻转、始终低、始终高对照。选择既有 min/max 窗属于探索性机制分析，不是独立泛化检验。
2. 每个原始 40 帧窗口额外取左右真实 8 帧计算前端，再截回同一 40 个绝对帧特征并通过原 TPro。该对照测“保持 TPro 槽位与对应绝对帧不变时，恢复前端缺失真实邻域的作用”。片段端点缺少真实 halo 必须单独标记，不得用复制或零填充伪称完整真实 halo。
3. 原始两窗 A、B 对同一目标帧 t，按绝对帧 u 对齐。记共同帧集合 S、对应前端特征 FA/FB、输出槽对应系数 wA/wB。对每个 head、channel、像素，原始投影差精确分为：

```text
slot = sum_shared[(wB-wA)*(FA+FB)/(2*sqrt(40))] + (biasB-biasA)
boundary = sum_shared[(wA+wB)*(FB-FA)/(2*sqrt(40))]
exclusive = sum_B_only[wB*FB/sqrt(40)] - sum_A_only[wA*FA/sqrt(40)]
zB-zA = slot + boundary + exclusive
```

`slot` 是共同帧的投影系数变化（含 bias），`boundary` 是共同帧前端特征变化，`exclusive` 是移入／移出帧贡献。最后一项同时包含上下文成员及其系数，不能命名为“纯内容因果效应”。这是冻结模型内部的代数分解；中间组合不必对应自然视频。

后端 `phi = TPro.norm1 → relu → TPro.conv → conv_out1 → conv_out2` 均在 eval 模式点式运行，因此可在 B×32×1×Hbox×Wbox 上算全部 8 个项组合，再对框内最大 logit 求三因素 Shapley。框内 logit 最大值与概率最大值位置等价，且避免 sigmoid 饱和。

## 独立审计重点

- 原数据 bbox 的像素中心规则必须与既有分析一致；空间 crop 留至少 9 像素 halo，16 是保守选择；图像真实边缘保留原 padding。
- 两窗最大响应像素可以不同。须保留同一 bbox 内的完整特征／贡献图，并在每个组合计算后取同一 bbox 的最大值；不能拿两个不同像素直接当同一像素分解。
- 正确 head/channel 映射为 32 → 8×4；时间是投影轴；bias 不除 sqrt(40)。
- 后端含 inplace ReLU；不得让组合评估修改基础 z 或贡献张量。
- 三项求和须重构原始 z 差，空组合／全组合须复现两个真实端点输出。
- 同一绝对帧若在两窗均处于时间槽 8..31，其前端特征应在数值容差内一致；不一致需先排查切片、索引、crop、dtype 或 eval 状态。
- `raw-frame roll`、逆序、重复帧改变时序连续性或运动规律，不应作为“纯位置效应”的主要证据。
- 固定共同 support 特征后切换投影行，可补充度量槽位敏感性，但因缺失特征及表示分布改变，不能声称完整分离内容／位置自然因果效应。

本文件仅陈述源码、权重及可执行诊断。尚未在此文件中报告新推理结果，也未提出已验证的模型或数据改进。

## 新推理脚本独立代码审计

已只读核对 `run_factorial.py`，包含新加入的 A/B 双锚。审计版本 SHA256：`fa2004b402add5a028f45af101c0604dd225b3e69168bc9f3acaef02bfebb238`。

结论：未发现阻止全量执行的实现问题。此结论是代码审计，不替代实际运行的数值门槛。

- bbox 裁剪逐项匹配旧 `analysis/analyze.py:44–51` 的像素中心与半开区间规则；空间 halo 16 超过已按实际层参数计算的半径 9。
- `projection` 正确按连续 4 通道映射 8 个 head，以单行 `F.linear` 投影时间轴；只缩放特征，未缩放 bias。
- `decompose` 以绝对帧交集／差集索引；三个对称贡献及 bias 分配正确。共享内部帧特征一致、真实 hook 投影一致、投影差重构都有数值门槛。
- 三项的 8 个组合逐次 `qa.clone()`；`phi` 的 inplace ReLU 操作于 BatchNorm 新输出，不修改 qa 或基础贡献张量。
- 双锚使用 `interaction_half=0.5*Δw*ΔF`。A 锚 slot 减该项、feature 加该项，分别变成 `Δw*FA+Δb` 与 `wB*ΔF`；B 锚相反，变成 `Δw*FB+Δb` 与 `wA*ΔF`。两种均保留相同端点。
- 实际前端延伸仅使用可得真实帧，记录缺失 halo；扩展前后内部特征一致有门槛。脚本没有凭空补帧。
- 对称及双锚端点已加入最大概率误差门槛及 `assert_close(logits, atol=1e-4, rtol=1e-4)`，防止概率饱和掩盖 logit 误差。
- 固定峰值基于原端点 probability 最大值、并列时按行优先取首点；脚本现记录并列数量，并在 metadata 明确饱和时它可能不同于 logit argmax。完整 bbox max-logit 读出不受此固定峰值选择影响。

## 共享投影支持脚本独立代码审计

已只读审计 `run_support_probe.py`，最终含依赖安全支持集的版本 SHA256 为 `80b249869c6779d53a45bb8a29018a8c72747ba8a2c2b6d5e705af6f0803e588`。未发现阻止运行的数学或实现问题。

- common support 按两个真实窗口的绝对帧交集取值，分别映射回各窗口的特征及其原时间投影系数。
- `q_shared` 包含一次 bias；`q_exclusive` 不含 bias。`q(lambda)=q_shared+lambda*q_exclusive` 保持共享项及 bias 不变，lambda=1 和原完整 q 均通过原冻结后端且有 logit／概率端点门槛。
- global shared 对整个双窗并集加可用真实 ±8 帧只算一次前端，随后共用同一共享帧特征张量；每端仍使用自己的原投影行及 bias。片段端点不足记录为 missing halo。
- 固定峰坐标与原 bbox 一致；普通共享项、global 共享项与 lambda 输出均在同一框中读取。
- `Ssafe={u: u-8>=shared_first 且 u+8<=shared_last}` 保证前端时间依赖全部位于共同原图帧内；用 global shared 的对应位置及原槽位系数计算，空集置 eligible=0、响应留空，并记录当前帧是否在 Ssafe 中。
- 普通 shared 特征包含 ±8 帧邻域，可能依赖共同帧区间外原图。删除独有投影项不是删除独有图像。即使 Ssafe 非空，投影项删除仍是可能偏离自然分布的内部消融，不能直接据此宣称语义证据足够或可部署改进。

执行后独立结果审计及计数见 `support_probe/AUDIT.md` 与 `support_probe/AUDIT.json`；复算脚本 `independent_support_audit.py` 不导入主推理或主分析程序。
