# DeepPro-Plus / C-v2：直接连接检测读出的时序漏检风险约束

状态：研究提案和独立训练损失参考，不是已获得收益的方法。本包没有运行检测器训练或真实视频评价，未接入用户本地工程。

## 1. 保持什么，修改什么

DeepPro-Plus发布模型、实际代码快照、40帧输入、O/S五窗、0.5主阈值与原组件匹配均保持。C第一版所有工件与失败结论保留。

C-v2是在当前C上加入一个训练期辅助项，不改推理网络，不替换原TPro/预测头，不加入图像配准、轨迹插值、LFP、查询解码器或独立高斯热图。原检测、保持和同轨迹对比损失均原样保留，先隔离新增项的作用。未来任何删除旧损失或调整保持项都必须另立版本。

本轮不是已确认的代码修复。已有结果未证明唯一根因。设计假设是：仅提高跨帧归一化表示相似性，不直接约束实际目标logit、困难背景logit和连续可见段中的检测覆盖。新增项检验这一连接是否有用。

## 2. 近期论文及具体代码来源

以下内容已公开核对，但原论文不为本方案的新公式或效果背书。

1. DeepPro / DeepPro-Plus，作者确认TPAMI 2026接收。公开Plus实现有唯一TPro，conv_out1给出8通道特征，conv_out2为1x1x1线性读出，forward返回特征与logits。
   - https://github.com/TinaLRJ/DeepPro
   - https://raw.githubusercontent.com/TinaLRJ/DeepPro/main/networks/models/DeepPro-Plus.py
   - https://raw.githubusercontent.com/TinaLRJ/DeepPro/main/networks/layers/TPro.py
   - https://arxiv.org/html/2506.12766v5
2. SlotContrast，CVPR 2025。借鉴跨帧对象关系组织与对比训练；实际loss先归一化表示，再构造相邻帧相似度。它的slot编号假设不能复制为本项目目标身份真值。
   - https://openaccess.thecvf.com/content/CVPR2025/html/Manasyan_Temporally_Consistent_Object-Centric_Learning_by_Contrasting_Slots_CVPR_2025_paper.html
   - https://raw.githubusercontent.com/martius-lab/slotcontrast/main/slotcontrast/losses.py
3. DEIM，CVPR 2025。借鉴让辅助学习直接关联检测预测及质量的思路。其MAL使用检测logits、已匹配目标质量及BCE-with-logits；本项目没有直接采用MAL、DETR matcher或IoU伪标签。
   - https://openaccess.thecvf.com/content/CVPR2025/html/Huang_DEIM_DETR_with_Improved_Matching_for_Fast_Convergence_CVPR_2025_paper.html
   - https://raw.githubusercontent.com/Intellindust-AI-Lab/DEIM/main/engine/deim/deim_criterion.py
4. TrackMAE，CVPR 2026，作者机构已确认接收。借鉴将时间关系写成明确训练任务；不采用CoTracker生成测试查询，也不搬视频重建网络。
   - https://ivul.kaust.edu.sa/articles/2026/03/01/five-ivul-papers-accepted-cvpr-2026
   - https://arxiv.org/html/2603.27268v1
   - https://raw.githubusercontent.com/rvandeghen/TrackMAE/main/engine_for_pretraining_cotracker.py

本包Python代码为独立参考实现，未打包第三方源码或权重。公式采用常见的soft-OR聚合思想，不声称原子算子新颖；完整新颖性仍需专门检索及最近方法对照。

## 3. 核心目标

在同一条训练轨迹的连续、确实可见且标注有效的短段中，对每帧取得：

- q+：来自原检测头、按固定区域读出规则得到的目标logit。
- q-：来自有效困难背景区域、按已登记规则得到的背景logit。

背景排除所有已标注任务对象，不能只排除严格小目标。背景条件不足时沿用现有采样器的已登记处理规则，不重新用严格局部筛选丢掉大部分数据。q+不是目标概率，也不是最终组件成功的保证。

定义 r_t = sigmoid(q+_t) * sigmoid(-q-_t)。它表示一个可微的目标高、背景低评分，不是校准的“实际检测成功概率”。

对连续可见段 S，L_gap = -log[1 - product_(t in S)(1-r_t)]。

r全部很低时惩罚大；出现较强、具有目标背景区分的帧后惩罚减小。时序相关数据不满足独立性，本方案不将乘积解释成真实漏检概率。

L_total = L_C_original + lambda_gap * L_gap。

L_C_original的检测、保持、原对比及归约不变。新增项不能替代逐帧目标监督、全图背景负监督、空标签窗口监督或原保持约束。

## 4. 必须知道的限制

1. soft-OR允许一个好帧主导一段，可能产生稀疏跳点，而不是持续稳定检出。必须保留逐帧损失，并同时评价总恢复、漏检段、保持与背景。
2. 段内聚合对顺序置换不敏感；其时序含义来自连续可见段分组，不来自速度或方向估计。不能用段内乱序来声称完成时序消融。正结果后的分组控制应打乱不同段/轨迹的归属，但保持同一批(q+,q-)及段长。
3. q+来自框区域，可能存在峰归属与定位歧义。最终仍使用完整画面组件匹配，不将训练分数计为检出。
4. q-只监督采样背景，不保证所有背景U受控。
5. 本项不要求存在一个高分教师帧；全弱段也能获得梯度。但有梯度不保证图像包含可用信息或模型能学出检测。
6. 不得对缺标、遮挡未知、失去可见性或时间断点补帧后当作连续段。
7. 不得用当前开发的4342/646对象、TEP-SAM开发恢复对象或确认结果构造训练标签、保持掩码或选择训练样本。
8. 固定baseline不等于永久冻结所有参数。首轮沿用C的实际可训练参数清单和BN规则，不擅自改变。

## 5. 最短集成方式

```python
features, logits = model(video)  # 使用实际冻结版本的返回接口
loss_original = existing_c_loss(...)

# 由现有训练侧采样器给出，必须直接连接logits的梯度。
# 不使用独立辅助预测头，也不从保存概率反算极低logit。
positive, negative, valid, segment_meta = sample_current_logits(logits, labels)
result = temporal_miss_risk(positive, negative, valid, reduction='none')

# 复用已登记的窗口外层权重；同轨迹/同帧重复曝光显式计数。
# 这个函数取决于本地C采样与损失协议，不在参考包中假装实现。
loss_gap = reduce_with_existing_window_weights(
    result.per_segment, result.active, segment_meta
)
loss = loss_original + fixed_lambda_gap * loss_gap
```

`sample_current_logits`和`reduce_with_existing_window_weights`是接口占位，不是包内已经验证的项目函数。需要映射本地C代码。q+沿用现有目标区域读取；不要将逐通道max后的特征误当成同一个空间位置，也不要将框中心等同于精确质心。

## 6. 实验：先修同一个C，再决定扩大

先在固定训练批次上检查：现有C的目标背景分数、归一化对比间隔、三项已有损失及新增项到TPro/读出的梯度。只用于检查任务连接与数值规模，不重新建立开发探针，不寻找新失败集合。

短修订三组均从同一个C终点、相同优化器状态起步，固定额外256步仅作为机制pilot：

- C-resume：原配置继续训练。
- C-frame：加独立逐帧的 -log(r)，即softplus(-q+)+softplus(q-)。
- C-gap：加连续可见段的L_gap。

三组使用同一批target/background样本。C-frame和C-gap共同的取样、窗口权重及新增项量级在训练侧固定；不在开发片段上扫描系数。段长及步幅由训练连续可见性与任务容许缺失时长预登记；包内不强制“4帧一定最佳”。

256步不能判整个思想不可救，也不能作为最终方法分数。若目标分数分离/原失败实际恢复出现正向路径，则从发布权重开展相同预算的正式对照；不要只拿C-gap的额外训练去比较旧C终点。若C-frame已解释全部收益，不宣称连续段机制必要。若有梯度冲突或覆盖缺口，可在证据支持下做一个单因素v2.1修订，而不是看到未过预算就自动换题。

正向结果后增加“相同元素、打乱跨轨迹/跨段归属”的同目标容量控制及多seed。只在冻结方案后使用未参与设计的新来源。

## 7. 评价

O/S、0.5、完整画面后处理保持。两开发片段内U<=4015仍是共同组件预算，不是官方Fa；通过它也不代表相对发布S所有背景成本不增。原4342低响应集合及4170极低响应子集、持续漏检段、646双侧保护、非严格小目标、框外前景与组件面积全部保留。

不把一个区间中任意帧的训练证据当作全区间所有帧被检出，不把轨迹插值当检测。没有任意固定“多1%即可顶会”阈值。主线需要实质、可重复、机制有独立作用的取舍改善。

## 8. 文件与实测范围

- temporal_risk.py：稳定log空间损失。
- segments.py：按视频、轨迹、原帧号与可见性构造段；不跨断点。
- test_reference.py：24项CPU合成测试。
- TEST_RESULTS.xml：最终测试记录。
- TEST_ATTEMPT_1.xml：首轮测试文件括号遗漏导致收集失败，保留记录。
- VALIDATION.json：环境和测试范围。

运行：`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q`

测试覆盖极低logit非零梯度、目标/背景梯度方向、混合精度、有效帧与空段、真实线性检测头的梯度连接，也主动验证了“单帧可主导”和“段内排列不变”这两项局限。

CPU测试不是GPU工程集成、模型训练或真实数据收益。当前会话不能访问/home/ly/DeepProPlus_TLFP_Code。本包不声称已修改该目录。
