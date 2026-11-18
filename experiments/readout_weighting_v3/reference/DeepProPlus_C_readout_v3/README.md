# C-readout-v3：有代码依据的选择性读出修订

## 当前定位

保留DeepPro-Plus baseline与当前C。**不新增检测网络，不重启GAP长训，不将“最差连续子段”作为新的未经验证公式。** 以已完成且出现有限恢复的C-frame为直接对照，只改变新增读出项的背景权重。

本包是独立PyTorch参考、测试与实验协议。未接入本地工程、未读取本地最新权重/日志、未训练检测器或评价真实视频。不能把单元测试通过等同于效果通过。

## 依据与问题

用户最新报告：继续训练701/722、U3933/3914；逐帧读出703/724、U3970/3942；GAP701/722、U3962/3933。逐帧读出多恢复同一轨迹两帧，不足以证明持续漏检已解决；它只是当前唯一直接显示少量增量的读出修订。

下一项检验：在已有目标逐帧强监督不变时，让新增背景梯度更多落在高背景预测上，能否保留或扩大恢复、同时改善背景与保持？这尚非已确认根因。

## 来源

直接参照CVPR2025 DEIM官方 `engine/deim/deim_criterion.py` 的 `loss_labels_mal` 中“停止梯度的负例置信度加权+BCE”。完整来源和不采用的部分见 `SOURCE_MAP.md`。这不是全MAL；没有DETR matcher或合法预测框IoU，不编造这些量。

## 公式

已有逐帧读出：

    positive = softplus(-q_positive)
    negative = softplus(q_negative)

候选只改负项：

    weight = sigmoid(q_negative).detach() ** 2
    negative = weight * softplus(q_negative)

gamma=2来自查阅的作者默认设置，本轮固定不搜索。正项、C原有检测/保持/对比、取样和背景监督全部保留；本项不替换原有全图或空窗负监督。

注意候选是相对强调高分负例，不是增加绝对负梯度。w<=1，会减弱相对普通BCE的新增负约束，可能增加背景。不得先写成“必然抑制误检”。

## 一项借鉴控制、一项自己的适配

1. `unit`：已有C-frame，负权重恒为1。
2. `confidence`：忠于已核对的DEIM负分支，w_i=stopgrad(p_i^2)。这是借鉴控制，不是全MAL。
3. `matched_confidence`：自己的适配，w_i*=a*p_i^2，其中a=sum(p_i)/sum(p_i^3)，全部权重停止梯度。

第三组在同一组logits、等权负样本下，与unit的负logit梯度绝对值之和相同；只把这部分梯度更多分配到高背景概率位置。它不匹配网络参数梯度，也不能保证两个训练分支演化后继续相同。实现使用log-sum-exp防止极低分数下分母下溢。

这是一项单独声明的任务适配，不是作者MAL公式，也不能被当作创新已成立。它用于避免将“降低全部负监督强度”误解成“更好地区分背景”。

如果一个归约组只有一个有效负样本，或所有负分数完全相同，matched_confidence会退化为unit，不应在这种条件下训练重复对照。若原背景样本有不等权重，应把这些权重纳入a的分子与分母；本包不静默修改现有协议。

## 网络与数据边界

- baseline、原TPro、输出头和推理不变。
- 可训练参数清单与BN状态沿用C当前冻结协议，不猜测H或C各自细节。
- 目标、背景读出来自当前完整检测logits，不从保存概率反推，也不增加独立投影头。
- 训练源独立于旧开发与保留确认来源。4342/4170/646是评价集合，不用于新增训练掩码。
- 不把其他真实目标当背景，不改变标注、有效域、负样本数量或采样策略。
- 外层轨迹、窗口、对象曝光权重保持。调用者使用 `reduction='none'` 风格，不因代码方便改为全batch均值。

## 建议执行

### 阶段0：读取代码和本地等价测试

作者文件和本地取样/归约已经存在时直接复用。仅核对作者函数中的detach、默认gamma和负权重；固定本地commit及SHA。

运行：

```bash
python -m pytest -q --junitxml=TEST_RESULTS.xml
python check_official_kernel.py --source /path/to/DEIM/engine/deim/deim_criterion.py
```

第二条会从可信的本地文件实际执行作者方法，不要对不可信代码使用。只比较原MAL全背景分支，不是作者完整复现。

### 阶段1：单批训练作用检查

在已有固定训练批次，比较三种新增项的正/负梯度。只确认弱正梯度保持、权重停止梯度、有效背景支持一致。

若matched_confidence与unit在当前有效组上已经完全等价，登记后不运行重复组。有效负样本覆盖不足时先明确适用范围，不偷偷换采样器。

### 阶段2：从同一个C2048起点作固定256步修订

所有组seed46、相同模型/优化器/RNG初始状态、相同后续样本计划和检查点规则。

- 已完成C-frame可在权重、RNG、曝光、归约、学习率、AMP等均可确认一致时直接复用。
- 只新增confidence和matched_confidence两组，最多2次新的检测器短训。
- 不从不同2304步分支各自继续；不把额外训练收益归给权重。
- 若原对照无法精确复用，统一登记新短修订，最多三组；不事后宣称完全同条件。
- lambda_readout沿用C-frame，不再根据开发结果调整；GAP项三组均不启用。

### 阶段3：只在出现有依据的增量后扩大

评价O/S、0.5、完整后处理不变，共同U<=4015；同时报告原失败恢复、4170极低子集、最长漏检段、646保护与非严格对象、框外前景及组件面积。

matched_confidence vs unit：在同快照下保持负logit总梯度的选择性分配，是否改善实际检测取舍？
matched_confidence vs confidence：额外的预算重标定是否必要？若只有raw confidence有效，要考虑普通减弱负监督的解释，不直接声称恢复机制成立。
candidate vs release：是否改善发布系统原有失败？

本轮不凭两帧或预算内一个数值宣称时序贡献。若只有弱分数变好而无匹配，保留为训练方向信号，观察预登记过程；不事后变更0.5。

若有跨多轨迹恢复、成本可接受且优于两控制，下一阶段才移入A/B/C共同训练配方重做同预算比较，验证原同轨迹约束是否因此获得独立作用；不是立即叠加另一个GAP。

出现明确代码/梯度问题可做单因素修订；如果只是过弱背景监督导致U变差，接受该设置的局限，不把所有损失叠加或循环扫gamma。

## 接入示意

```python
terms = readout_terms(q_positive, q_negative,
    positive_valid=positive_mask, negative_valid=background_mask,
    mode='matched_confidence', gamma=2.0)
# 下述为本地接口，不是本包已实现的项目代码：
extra = existing_frame_reducer(terms.positive, terms.negative,
                              terms.positive_valid, terms.negative_valid)
loss = existing_C_loss + existing_lambda_readout * extra
```

本包返回元素级损失。`equal_mean_per_group`仅供与本地原归约确实一致时使用，不应自动替代本地归约。

## 实测范围

30项CPU合成测试通过。GPU不具备；作者函数本地执行对照、真实框采样、训练流程、性能与泛化均未验证。
直接容器网络下载GitHub API失败，未获得作者commit。网页核对不等于完整仓库下载；本包没有分发作者代码或权重。

代码无新增推理参数。方法归类是检测学习修订，不是新的时序模型结构，不是已成立的论文主线。
