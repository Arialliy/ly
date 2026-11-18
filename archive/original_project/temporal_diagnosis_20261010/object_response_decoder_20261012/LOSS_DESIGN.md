# R0 / R1 / R2 直接监督与分配设计草案

状态：**实现已与父任务共同工程协议对齐，CPU 语义测试通过；正式协议冻结和 GPU 执行由父任务统一管理。** 2026-10-12。

本文件只规定标签、直接损失及分配规则。所有工程选择对 R0/R1/R2 共用。R0 关闭新增查询的时间交互；R1/R2 的结构相同，分别逐帧分配及整窗轨迹一致分配。旧实验的标签名称、512 步结论和失败决定不作修改。

## 1. 只读取训练标注的已知事实

- 既有 8 个训练来源的 16,707 条标注均含 `track_id` 和 `tracking_id`，当前存档的 `ignore=iscrowd=0`；没有显式 visibility、occlusion 或 confirmed-exit 字段。独立数据审计已核对公共来源；本轮采用 `(sequence, category_id, str(tracking_id))`，numeric `track_id` 仅保留溯源，不用于身份监督。字段一致性不是完美物理身份的证明，详见 TRAIN_ID_AND_DENSITY_AUDIT.md。
- 既有 512 更新曝光计划包含 2,048 个 40×128×128 窗口；其完整 core9 内，每帧最多 19 个正对象，整窗有效轨迹最多 19 条。385 窗没有此类有效轨迹。这些数量只由训练标注计算。
- 完整训练原图中，单帧最多 78 个正对象，任意连续 40 帧最多 79 条原作者命名身份，或 91 个 numeric ID。numeric ID 在部分同一作者身份内跳变，本轮不把它当作监督身份。**全图固定查询 K=100**，超过两种保守容量界。本损失要求每次分配的对象数不大于 K，否则立即报错，不能截断 GT。
- 训练框的 segmentation 字段本身是矩形轮廓；不能称为精确前景 mask。不存在的 annotation 也不能被解释为已确认的目标不存在。

## 2. 对象级响应标签

沿用已登记的框导出几何规则，将原有 union label 改为**保留每个对象的独立标签**：

1. 使用原浮点 bbox 的几何中心，像素坐标为 `(j+0.5, i+0.5)`；不读取亮度、模型峰值或开发预测。
2. `sigma_x=w/6, sigma_y=h/6`，原框外截断，计算 `floor(255*exp(-0.5*r²))>0` 的二值紧凑支持域。复用 `supervision_alignment_control_20261012/code/gaussian_labels.py::gaussian_region`，不重新选择 Gaussian 参数。
3. 正对象为全部类别、全部尺寸中非 ignore/crowd 且完整落入 crop core9 的有效框；不是只训练本轮评估的小目标。
4. 所有 ignore/crowd、截断或未完整落入 core9 的有效框，其 bbox+2 像素区域按既有规则排除，且排除优先。
5. 正框支持域被排除后为空：该对象帧不进入存在正项、位置正项及分配；登记原因，不造一个图像峰或强制中心像素。极小框按同一栅格规则处理。
6. 两个合法正对象的响应支持域可以相交；各自保留自己的标签。不合并轨迹，不按模型预测决定去留。

记最终非空对象支持域为 `P[j,t]`。共同有效像素域 `V[t]` 沿用旧 `build_targets` 的结果：旧全 GT 排除的背景域与完整正框矩形之并，再让 excluded-object bbox+2 排除覆盖。物理背景域 `B[t]` 完全沿用旧 `build_batch` 的背景掩码，排除**全部**已标注有效框及其 2 像素 guard。

## 3. 未知帧与负监督资格

不把缺标、裁剪外、截断或未知遮挡自动写成存在标签 0。对窗口内至少有一次有效对象响应标签的轨迹集合 `J`，定义共同的保守负监督资格：

`negative_eligible[t] = 当前帧 J 中每条轨迹均有有效 core9 标签，且没有被排除对象的扩张框与有效训练域相交。`

该资格由标注预先计算，**R0/R1/R2 完全共用，独立于预测和 Hungarian 结果**。窗口内没有有效轨迹时，第一条件为空真，但第二条件仍须满足。若数据完整性审计发现图像或标注来源存在缺页等问题，该帧也不可获得负资格。

- 有效已观察对象始终产生存在/位置正监督。
- 负资格不成立的帧：不产生未匹配查询的存在负项，也不产生实例/背景空间负项；只保留明确的有效正观察。不能让 R0/R1 将 R2 的未知位置自动视为负例。
- 负资格成立的帧：整窗所有有效轨迹都已被观察，因此三组未匹配查询数量同为 `K−|J|`，负例数量不会因时间分配方式不同而变化。
- R2 中已分配轨迹的非有效观察帧完全忽略，不跨缺标填标签，不要求输出时间平滑。

这是一项保守的新轮训练约定，**不是证明剩余像素均为物理真背景**。旧标注本身可能不完整的限制保留。实际 label builder 完成全部 2,048 个窗口的独立旧标签对照：64,555/81,920 帧（78.8037%）具有负资格；148,902 个 full-core 对象帧中有 66 个支持域被完全排除，实际有效正对象帧 148,836。详见 data_audit/frozen_target_receipts/MANIFEST.json，旧 background、valid 和 union label 全部逐像素一致。

## 4. 五个独立归约的直接损失项

模型提供 `a[B,T,K]` 与实际全分辨率动态响应 `m[B,T,K,H,W]`，或提供可精确分块求得该响应的接口。损失直接作用于 logits，**不先经过最终乘积或 max**。

对匹配到对象 `j` 的查询 `k`：

\[
e^+_{j,t}=\operatorname{softplus}(-a_{k,t}),\qquad
p_{j,t}=|P_{j,t}|^{-1}\sum_{x\in P_{j,t}}\operatorname{softplus}(-m_{k,t}(x)).
\]

在有负资格的帧，匹配查询的实例空间负项为

\[
n^{\rm inst}_{j,t}=|V_t\setminus P_{j,t}|^{-1}\sum_{x\in V_t\setminus P_{j,t}}\operatorname{softplus}(m_{k,t}(x)).
\]

**实例负项与物理背景不是同一概念。** 对 query k 来说，另一个对象的支持域不属于其被分配实例，可以是实例负例；另一个对象仍由自己的 query 接受正监督，绝不能从全模型的正目标清单或 GT 排除掩码中删除。此项避免所有查询都输出同一张多目标响应图、使身份分配形同虚设。

未匹配查询在有负资格的帧产生

\[
e^-_{k,t}=\operatorname{softplus}(a_{k,t}),\qquad
n^{\rm bg}_{k,t}=|B_t|^{-1}\sum_{x\in B_t}\operatorname{softplus}(m_{k,t}(x)).
\]

它们不在任何标注目标框内接受空间背景负监督。空背景集合对应项不进入均值；同时记录次数。均值集合全空时使用与模型连通的零值，不能除零。

每窗口损失固定为

\[
L_w=\operatorname{mean}_{(j,t)\in O}e^+_{j,t}
+\operatorname{mean}_{(k,t)\in N}e^-_{k,t}
+\operatorname{mean}_{(j,t)\in O}p_{j,t}
+\operatorname{mean}_{(j,t)\in O_-} n^{\rm inst}_{j,t}
++\operatorname{mean}_{(k,t)\in N} n^{\rm bg}_{k,t}.
\]

五项系数先登记为 1，外层仍为四窗口等权均值。不增加 top-k、focal、Dice、pairing、teacher、smoothing 或辅助重建项。`O_-` 为具有负资格的有效对象帧集合。实例空间负项与未匹配查询的物理背景负项分别归约；如果合进所有 query-frame 的一个均值，K 大而对象少时已匹配对象的负项会被约 J/K 稀释，故首版明确不采用该混合归约。各项原始值和加权值分别记录。

存在/空间正项先对每个对象帧归约，空间负项先对每个 query-frame 归约，再独立平均。因此海量背景像素不会直接稀释每个对象的正项。R2 的**训练损失**也按同一个对象帧集合归约，不能改为逐轨迹等权，否则同时改变长短轨迹曝光权重。只有下面的轨迹匹配代价按轨迹有效观察平均。

## 5. 两种 Hungarian 分配

对每个合法对象帧，候选查询与对象的代价为

\[
C_{k,j,t}=\operatorname{softplus}(-a_{k,t})+p_{k,j,t}
+\mathbf 1[\mathrm{negative\_eligible}_t]n^{\rm inst}_{k,j,t}.
\]

这里 `p` 与 `n_inst` 使用该候选 query 的 mask，且只用于分配的矩阵计算处 detach；选出匹配后的真正损失重新引用带梯度张量。禁止用 sigmoid 乘积的饱和分数替代直接代价。

- **R0/R1**：对每帧的 `[J_t,K]` 代价矩阵分别调用 Hungarian，每个目标匹配一个 query、每个 query 最多匹配一个目标。
- **R2**：对每条窗口内轨迹，取其有效观察帧上的均值 `Cbar[j,k]=mean_{t in V_j} C[k,j,t]`，在 `[J,K]` 上只调用一次 Hungarian。该轨迹所有有效观察帧使用同一个 query。
- 轨迹 key 为 `(sequence, category_id, str(tracking_id))`，不能跨来源或类别合并；numeric track_id 不参与身份分配。窗口间不维持查询身份，不输入 test GT。
- 行固定按 canonical track key/annotation ID 排序，列固定为 query index。以 CPU float64 的有限代价调用已固定 SciPy 版本 `linear_sum_assignment`；记录原矩阵摘要及 assignment。相同输入确定复现即可，不添加可能改变非并列最优解的任意 epsilon。
- `J>K`、重复同帧轨迹 ID、非有限代价、有效对象漏分配直接 fail；不得把对象截断或丢到背景以继续训练。

R2 使用额外的跨帧身份关系，应在报告承认；“相同 losses”不表示其监督信息量与 R1 相同。

## 6. 不展开 K×J×H×W 的实现

动态 mask 需要实际定义完整空间输出，但匹配代价可精确低内存计算：

1. 分帧或分查询块计算 `[K,H,W]` mask logits，保留或重计算该块用于真实损失。
2. 对每个 query 求一次 `S_valid[k]=sum_{x∈V} softplus(m_k[x])` 和 `S_bg[k]`。
3. 对每个对象只 gather 其支持域的扁平像素下标，得到 `[K,|P_j|]`；用它计算 `sum softplus(-m)` 及 `S_posneg[k,j]=sum_{x∈P_j} softplus(m)`。
4. `n_inst[k,j]=(S_valid[k]−S_posneg[k,j])/(|V|−|P_j|)`。模型及上述求和固定 FP32，三组一致；仅最终 Hungarian 代价矩阵转 CPU float64，不改变已有 FP32 代价值。
5. 这样中间匹配矩阵仅 `[K,J]`，不构造 `[K,J,H,W]`。真实正损失只 gather 已匹配 query，真实负项复用共同统计，梯度等价于显式 BCE。

全图推理必须先确定全图共享的查询状态，再分块动态空间输出。不能在每个 tile 重做 query attention 后宣称同一全图模型。计算块的大小只影响内存，不改变空间坐标、查询初始化、时间范围或 softmax 归一化域。

## 7. 必须通过的少量语义测试

1. 极低的存在和空间正 logits（例如 −20）仍产生明确有限负导数；高背景 logits 产生明确正导数。空窗具有负梯度，所有忽略窗得到有限零值。
2. 两个对象、两个 query 的手工代价中，逐帧方案允许 query 换人；轨迹方案必须在整窗固定分配。交换 GT 输入顺序不改变对应 ID 的匹配结果（真正完全相同代价的并列情况只检验确定性）。
3. 少一帧轨迹标注不会生成该轨迹存在标签 0；同帧三组的负资格及负项数量一致。截断、ignore、crowd 和极小框规则逐一用合成标签验证。
4. 另一个合法对象既出现在自己的正监督中，也被排除出全模型物理背景 `B`；它可出现在别的 query 的实例负集合，但不能被日志计为物理背景。
5. 向量化成本与小尺寸显式 `[K,J,H,W]` oracle 的值和梯度一致；cost-detach 不得切断 matched loss 梯度。
6. 每个有效对象恰好分配一次，K 足够；全网络所有原参数与 BN buffer 完全冻结。新输出无需初始化等于发布输出。
7. 同查询状态下 dense mask 与分块 mask 的 logits/最终概率一致；整个查询解码的全图处理另由模型实现测试覆盖。

## 8. 实现和验收记录

- `code/losses.py::object_response_loss(a, masks, targets, assignment)` 返回带梯度标量、小型 JSON-safe metrics 和可选的完整匹配列表；训练日志仅记录 assignment SHA，不写完整巨大列表。
- 数据接口为 `targets.py` 构造的单窗 dict：`valid/background[T,H,W]`、`negative_eligible[T]`、稀疏 `observations` 和排序的 `track_keys`。实际 batch 由父训练器分四个 microbatch，各乘 1/4 后只更新一次。
- 全窗所有对象代价组成 `[O,K]` 一次传至 CPU，再逐帧或逐轨迹求 Hungarian；不存在每帧独立 GPU 同步。五项损失数值另作一次很小的传输。
- `code/test_losses.py` 的 18 项纯 CPU 检查全部通过，包括显式逐对象 oracle 的值与梯度一致、极低 logit 梯度、缺标/空窗/无负像素、逐帧交换与固定轨迹分配、重叠对象、容量错误、无穷值拒绝及 K 增大不稀释实例负项。
- 另以预先固定训练计划第一个真实训练窗口运行 K=100 的损失前向与反向，40 个有效对象帧，3,960 个负 query-frame，两输出梯度均有限，0.864 秒纯 CPU；此项没有模型训练或参数更新。
- 完整 receipt 见 `checks/LOSS_CPU_TESTS.json`。没有为本文件读取开发预测、选择训练检查点或启动 GPU。
- 正式预算为 2,048 步，重复旧 512 步曝光计划 4 次，不是 8,192 个不同窗口。原结构和 BN 冻结。正式代码哈希及协议统一由父任务冻结；本文不自行宣布完整原型已通过 GPU 或可学习性检查。
