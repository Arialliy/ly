# M1-v1.3：有限窗口的源帧证据库

日期：2026-10-14。
状态：模型设计与独立存储/读取核心；不是已验证有效的方法，不是本地工程补丁。没有新增M2，不改变DeepPro-Plus baseline或M1的恢复职责。

## 1. 固定职责与本次唯一研究因素

M1维护检测前的特征/证据状态，不维护已检出目标列表。在原合法40帧窗口内进行写入、空间关联和遗忘，目标是获得Current没有的实际弱目标恢复，减少持续漏检，且控制背景与原成功丢失。

用户已报告v1.0、v1.1、v1.2历史影响进入输出，但没有历史独有恢复。不能由这些结果断言代码错误或单状态压缩是唯一根因。新假设：在决定哪些历史信息有用之前，反复融合/覆盖单一状态可能不利于保留暂时低显著性线索；把保存与读取分开，能否帮助当前判别？

本轮只改变“历史保存与读取的组织”，不同时改PDA、TSFM、检测头、监督、采样、优化器、标签或阈值。移除v1.2新加的状态插值预测器，不与其堆叠。原有M1接入保持。

## 2. 可追溯的来源

### RFR（TGRS 2025）：实际沿用的算子
- https://github.com/XinyiYing/RFR
- https://raw.githubusercontent.com/XinyiYing/RFR/main/codes/model/RFR_framework.py
- https://raw.githubusercontent.com/XinyiYing/RFR/main/codes/model/utils.py
- https://arxiv.org/abs/2409.12448

沿用本地已核对的PDA和TSFM，不重新下载浮动main覆盖已验证的代码。原源码先对齐历史，再由TSFM融合；当前与历史共享用于最终预测。TSFM本身含有当前信息与注意力，不能宣称它没有任何选择能力。它也可能主要依赖当前信息。

### StreamVGGT（作者官方仓库标注ICLR 2026）：保存/读取解耦的代码参照
- https://github.com/wzzheng/StreamVGGT
- https://raw.githubusercontent.com/wzzheng/StreamVGGT/main/src/streamvggt/layers/attention.py
- https://raw.githubusercontent.com/wzzheng/StreamVGGT/main/src/streamvggt/models/aggregator.py
- https://arxiv.org/html/2507.11539v2

`Attention.forward`在use_cache时拼接past_k/past_v和当前k/v，返回new_kv；当前查询从缓存读取。`Aggregator.forward`按层维护缓存。借鉴的是“保存按来源分开的历史表示，再按当前信息读取”，不是复制RGB编码器、几何输出头、全局token注意力或蒸馏训练。

本项目不是StreamVGGT的KV层移植：保存的是现有M1的全分辨率X_s，而不是作者的Transformer K/V。容量K=4、FIFO、逐源帧PDA/TSFM、均值读出是本项目自行登记的适配。缓存与FIFO都不是独有新算子。只有任务必要性与检测收益成立后，才能进一步论证方法贡献。

## 3. 写入、关联、遗忘

### 写入
X_t = 现有M1对原DeepPro-Plus前端特征的投影。

每个有效源帧的X_t按来源索引写入；不以概率、目标框或检测列表决定是否写入。不缓存TSFM输出或已经对齐的结果。写入发生在本帧历史读出之后。已存条目在保留期内不被后续融合覆盖。

这里的“原始观察”仅指“本次M1融合之前的投影特征”，不是原图，也不是完全单帧的特征；原前端已经包含时序运算。没有保证原编码器/投影完全保留了目标信息。

### 空间关联与读取
对仍保留的源帧s，独立执行：

    A_s_to_t = PDA(X_s, X_t)
    R_s_to_t = TSFM(A_s_to_t, X_t)

每次都从原缓存X_s重新对齐到当前帧，不把上一步已经扭曲的读出继续作为源。

    Rbar_t = mean_s(R_s_to_t)
    Hnew_t = H_t + W_out(concat(X_t, Rbar_t))

无历史的首帧使用Rbar_t=X_t。W_out、原TPro与原预测头沿用。均值是首版不引入新参数的组间归约；TSFM保留内部内容选择。均值不保证提高SNR或响应，可能稀释只出现于少数帧的线索。暂不增加另一种门控/统计量来同时检验。

### 遗忘
最多保存K=4个过去源帧，源帧过期即FIFO删除；真实窗口结束清空。K是第一版资源配置，不是已发现的最佳时距。当前模块的显式引用最多回看4个特征索引，但原TPro仍读取40帧，原特征也已有时间感受野。不能把整个模型称为仅4帧/严格因果模型。

不把FIFO称为“学会了目标可见性”或“可靠性遗忘”。背景同样会被保存；它的无用贡献仍需PDA/TSFM与原检测学习处理，不能保证背景不累积。

## 4. 参考代码的接入范围

`evidence_bank.py`只管理缓存、调用现有align/fuse、平均和分段状态。没有真实PDA/DCN、TSFM、检测器、数据或优化器。

```python
from evidence_bank import scan_features

# projected: [B,C,T,H,W]，来自现有M1投影
reads, bank = scan_features(
    projected,
    align=self.deform_align,
    fuse=self.spatio_temporal_fusion,
    mode='history',  # 或 current_only
    capacity=4,
    window_id=ordered_batch_window_id,
)
# 后续原M1 W_out([projected, reads])及原预测链保持。
```

`window_id`须标识有序batch的真实视频/窗口身份；改变顺序或换视频必须重建。代码接受全部为有效帧的张量；若本地存在填充/缺失帧，需要显式接口扩展并测试，不把它们当成真实观察。输出dtype须与现有接口兼容；AMP自动转换差异先明确后适配，不能静默改运算精度。

代码对缓存写入和读出输入clone以防原地修改，保持autograd。训练期间不自动detach；固定原时序反传策略，如需改变则另登记。FIFO只限制活动存储，训练反向保存的图仍可能随T与K增加。clone及多次PDA/TSFM会增加开销。

原代码中TSFM含全图/自适应池化的频率统计，不能假设“给tile一个小halo”就自动保持整图一致；复用本地已验证处理，再检查每个源帧读取的统计域。

## 5. 公平对照与训练

第一轮只做Bank-Current、Bank-History，两组完全相同参数和每帧读出调用次数。

- Bank-History：每个槽读取它保存的真实X_s。
- Bank-Current：保留槽数/执行路径，但每个读出的源替换为X_t。每个PDA/TSFM仍实际执行，不跳过子网络。缓存管理仍相同。

从同一个M1-v1.0 History 2048步检查点复制，保留已训练的PDA、TSFM、投影与W_out，不用v1.1或v1.2终点。通过显式参数清单加载；不使用strict=False掩盖遗漏。新缓存策略本身不添加参数，也不重置旧W_out。

两组相互的参数初值相同，但history与current输入不同，不能要求其初始输出或与旧递归History完全相同。关闭整个M1才是原发布链回归测试。

建议共同新增512步、seed46，作为一次修订预算，不是收敛保证。曝光、损失、学习率、AMP、原模型冻结/BN规则沿用当前M1。对不再有一条递归链的缓存，明确反传策略；两组不人为令一组detach另一组不detach。记录0/128/256/512步，主开发评价固定512步。若资源约束需改预算，应在看开发结果前对两组共同登记。

训练端记录全部原有对象损失与额外历史贡献；原本未检出但有标签的位置不能因为低分而跳过。训练日志按原低响应/可靠目标/背景分层即可，不增加新损失或训练样本筛选。若正训练对象本身没有恢复，和只在开发来源无恢复，应分别解释。

旧递归M1的结果是历史参照，不是同资源对照。若本候选有效，仍需追加匹配参数/计算/引用范围的递归控制，才可把收益归到“不覆盖缓存”本身；先Current/History只判断新增历史是否有价值。

## 6. 资源边界

K=4、T=40时，每窗口PDA/TSFM配对调用数为sum(min(t,4),t=0..39)=150；原递归是39次。两组新对照一致，但不能宣称与baseline或旧M1同总计算量。

缓存本身约B*K*C*H*W*bytes；B=1,C=16,H=W=256,FP32时约16MiB，不含backbone缓存、对齐中间量、clone、输出和反向图。必须先实测真实分辨率、40帧、CUDA DCN的峰值显存与耗时。资源不足时应先改实现，不暗改分辨率、源帧选择或容量后仍称同一配置。

## 7. 实现测试与作用检查

已完成的29项CPU测试：FIFO和源特征未被覆写、低/负值照常写入、读前写后、无跨窗口状态、索引校验、Current同次数计算、梯度、时间分块前向/梯度、混合精度输出、已知空间平移的索引行为、输入保护等。

这些是独立核心和替身算子测试。没有测试真实PDA/DCN估计正确、官方作者模型等价或真实检测恢复。

本地新增必要检查：
- 缓存中的X_s在后续读出后仍与写入时一致；source_id/slot年龄正确。
- 真实原地运算、AMP、空间全局统计与时间分块一致。
- 关闭整个M1仍恢复发布路径。
- 带训练图的缓存未被意外detach，窗口结束不串图。
- 同一个Bank-History终点的history/current_only及bypass输出。
- 改变将被引用的历史应能改变该分支输出；已淘汰的投影特征对未来bank读出不应再有影响。对原图扰动不能声称完全无影响，因为原TPro/时空前端仍可能读取它。

## 8. 验收与修订原则

原40帧/O、S/0.5/组件匹配规则不变。4342低响应、4170极低响应、646保护、非严格目标、完整轨迹漏检和所有背景成本都保留。U<=4015仅对应原320帧；发布S自身U=3967。分数、状态差异或下降的U不能代替历史独有恢复。

首先比较成对训练的Bank-History与Bank-Current，再做同一History权重的current_only。单列两个方向的新增/丢失，不仅报告净数；O/S同一对象去重但分别报告安排。历史带来的恢复与持续性改善必须在背景/保持约束下解释。

若Current解释全部变化，不能称历史成功；若历史有实际新增但存在具体退化，可继续修M1的关联/读出一处。若保存正确但在PDA/TSFM中失去目标判别价值，下一项修订应针对该处，不再修FIFO；若训练有收益而开发没有，转向泛化解释。若只有背景抑制，M1职责未完成，不改名成抑噪模块。

若成功，再固定M1并考虑剩余问题，不预先设计M2。若本轮没有恢复，不自动扫K、增加门控或换另一篇论文；利用已记录的源帧贡献与训练/开发差异来定向修订。

## 9. 本次来源与执行边界

公开论文和源码通过web核查；本包未复制第三方源码或权重，未取得本地作者仓库commit，不编造hash。部署前记录本地已用版本和授权条件。

未访问/home/md0/ly/DeepProPlus_TLFP_Code或/home/ly/DeepProPlus_TLFP_Code。未修改本地M1，未训练检测器，未进行真实数据评价。输出文件名是新提案，不改变任何旧实验工件与结论。
