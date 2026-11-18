# 可选择上下文与证据保护：机制核验笔记

检索日期：2026-10-10。范围为跨任务可借鉴机制的局部核验，不是新颖性终审；没有运行或复现以下方法。目标依据来自本项目既有诊断：同帧跨真实窗口有响应强翻转，部分窗口独有特征有害，另一些却是必要支持。该事实不能推出所有时序信息都应删除。

## 1. Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free

- **正式来源**：NeurIPS 2025，pure method；[正式论文](https://papers.neurips.cc/paper_files/paper/2025/file/904e89bb4e632e75fb47f093b620b257-Paper-Conference.pdf)，[作者代码](https://github.com/qiuzh20/gated_attention)。原任务是语言建模，不能以其语言成绩证明红外检测有效。
- **精确机制**：式 (5) 为 `Y' = Y ⊙ sigmoid(X Wg)`，`X` 是 prenorm hidden states。表现最好的 G1 放在每头 SDPA 输出之后、拼接投影 `W_O` 之前。可逐头标量或逐头逐元素，后者形状为 token × head × head_dim；不是对所有头共享一个帧权重。
- **证据强度**：正式论文表 1 对门的位置、粒度、头共享、激活与加/乘形式分别消融；G1 元素门的 PPL 为 5.761，基准为 6.026（该语言实验条件下）。不能迁移这个改善幅度。
- **官方代码核验**：[modeling_qwen3.py](https://github.com/qiuzh20/gated_attention/blob/f4c2a5f6ffd6ec709e0c60072c95ed4f5ce5b5d2/modeling_qwen3.py#L309) 从 `q_proj(hidden_states)` 分出 query 与 gate score；L361–366 是 SDPA 输出乘 sigmoid，再执行 `o_proj`。头门为 `[B,L,H,1]`，元素门为 `[B,L,H,D]`。不是注意力分数的 softmax。

**迁移推论（本项目候选设计，不是该论文结论）**：可以借鉴“上下文写入幅度与上下文内部配比分开”的原则。若 `c_t(W)` 是窗口上下文分支，可测试 `z_t = f_t + g(f_t,c_t(W)) ⊙ Δ_t(W)`，其中 `f_t` 来自未经过整窗混合的局部路径；门的条件应包含实际上下文，且要留能接近零的状态。

**关键限制**：sigmoid 在有限输入下只接近零，并不严格为零；它也不自动识别“有害”。如果照搬原文的 query-only 门，并把 `X` 设为同帧、无上下文的固定特征，那么同帧跨窗口的门值完全一样，不能独立区分两个窗口好坏。若把门放到已经被不利上下文压毁的特征之后，也不能重建丢失证据。残差保留了信息通路，但后续非线性和分类头仍可能使目标分数下降。

## 2. Cutie: Putting the Object Back into Video Object Segmentation

- **正式来源**：CVPR 2024，pure method；[CVF 页面](https://openaccess.thecvf.com/content/CVPR2024/html/Cheng_Putting_the_Object_Back_into_Video_Object_Segmentation_CVPR_2024_paper.html)，[全文](https://arxiv.org/html/2310.12982v2)，[作者代码](https://github.com/hkchengrex/Cutie)。在半监督 VOS 中输入首帧目标分割，随后传播；其 MOSE 对 XMem 的改善为 8.7 J&F，不是自动红外检测。
- **精确机制**：对象 query 与像素记忆双向交互；一半 query 只读预测前景，另一半只读背景。式 (2) 是 `X' = softmax(M + QKᵀ)V + X`，`M` 为 0/负无穷掩码并跨头共享。高分辨率像素特征通过残差保留。对象池化区域为空时，不更新该对象记忆。
- **官方代码核验**：[object_transformer.py](https://github.com/hkchengrex/Cutie/blob/ec5cdd4cf16f75c73ad785a2f96fb97dbad4125a/cutie/model/transformer/object_transformer.py#L184) 的辅助掩码函数将“整行全被屏蔽”的情况重置为全可见（L203）。[transformer_layers.py](https://github.com/hkchengrex/Cutie/blob/ec5cdd4cf16f75c73ad785a2f96fb97dbad4125a/cutie/model/transformer/transformer_layers.py#L45) 使用常规多头注意力与残差。

**迁移推论**：可借鉴“目标候选与干扰背景分别建模后，再判断其贡献”，而不是仅按像素相似度混合。DeepPro 的时序轮廓可尝试分为目标一致支持与背景变化两类证据；任何候选/轨迹必须由推理可获得的信息建立，不能输入测试真值 ID。

**关键限制**：Cutie 不是“拒绝全部上下文”的机制；softmax 在允许集合上仍归一化，且全掩码回退会放开全部位置。保留的像素 readout 已经包含记忆，不等于纯当前帧证据。红外小目标在 stride-16 特征上可能不足以产生稳定对象 query；无首帧分割时还需解决冷启动和目标发现。暂不优先整套移植。

## 3. Reference-Based Video Super-Resolution Using Multi-Camera Video Triplets（RefVSR）

- **正式来源**：CVPR 2022，method + benchmark；[CVF 页面](https://openaccess.thecvf.com/content/CVPR2022/html/Lee_Reference-Based_Video_Super-Resolution_Using_Multi-Camera_Video_Triplets_CVPR_2022_paper.html)，[论文](https://openaccess.thecvf.com/content/CVPR2022/papers/Lee_Reference-Based_Video_Super-Resolution_Using_Multi-Camera_Video_Triplets_CVPR_2022_paper.pdf)，[作者代码](https://github.com/codeslake/RefVSR)。输入包含不同视场的参考视频，非单路红外检测。
- **精确机制**：以 patch 余弦匹配最大值为置信度；置信图与特征一起运动对齐、传播。式 (6) 用当前匹配置信度和已传播置信度的卷积嵌入，调制参考/历史融合残差，加入已有特征；置信度用逐像素 max 累积。表 1 在相同训练损失上加该模块，PSNR 从 31.31 到 31.68。
- **代码细节**：[RefVSR.py](https://github.com/codeslake/RefVSR/blob/dfc54b023ccf672e506d5beadbac7a50a08b4b63/models/archs/RefVSR.py#L47) 的 `conf_fusion` 是两个卷积加 LeakyReLU，输出是空间×通道调制图；**并非 sigmoid [0,1] 门**。L130–131 做乘性残差融合，L147 做置信 max 累积；已有传播特征本身仍包含上下文。

**迁移推论**：更值得借鉴“连同特征一起传递其证据质量”，而非把所有上下文 token 视为等可信。可将同帧跨窗干预提供的损害/获益现象用于训练上下文质量预测，但训练标签须在训练序列上构造。

**关键限制**：其匹配置信度是相似性，不是目标存在概率，也不等价于边际检测收益；高相似背景可能产生高分。其调制为无界卷积输出，并无显式全拒绝状态；已有路径也不是纯当前帧。max 累积可能保留一次错误高置信，而本项目需要检查实际收益和误报，不能直接采用“高置信即可靠”的假设。

## 4. 面向 DeepPro-Plus 的机制决策

1. 优先对比**上下文条件化的残差调制**，但必须从时序混合前保留一条局部证据路径；是否存在可保护的局部证据先测。不要把“加 sigmoid”命名为已证实的解决方案。
2. 检查 `g≈0` 时能否真实回退到局部分支；若路径此前已混入整窗信息，这个检验不成立。还应检验有利上下文样本是否因门关闭而新增漏检。
3. 首次实验的参数选择可从 `[B,1,T,H,W]` 的逐时空位置门开始，再与 `[B,C,T,H,W]` 的通道门比较。这里是我们对现有张量结构的候选设计，三篇论文均未验证这个具体接法。
4. 仅在上下文候选间做 softmax 时，权重总和固定为 1，无法在权重层表示“全部不采纳”；若集合含当前帧/零值 null token，则可把权重移给该项。softmax 并不必然让最终向量非零（值可抵消），因此精确问题是缺少独立的拒绝自由度，而非数学上强制非零结果。
5. 决定成败的是相同误报预算下：不利上下文新增漏检减少、有利上下文带来的检出保留，以及总体 Pd/Fa 或既定正式指标。单看门值更小、波动更小或全体响应更高都不够。

## 5. 局部检索与证据状态

安全公共检索词包含 Cutie title、Gated Attention title、video restoration reliability/confidence aggregation、RefVSR title。另筛查了 EDVR（CVPR 2019 workshop，未列为主会证据）、ODPNet（CVPR 2025，机制更偏联合退化与跨尺度重建）、BasicVSR/RVRT 类方向的检索线索；后者未做全文核验，不作为推荐来源。排除了不满足来源规则的页面。

三篇的洞见/完整性/数值证据（1–5，针对原任务）分别暂记 Gated Attention 4/5/4、Cutie 4/5/4、RefVSR 4/4/4；这些是本文筛选判断，不是对红外迁移收益的评分。三篇均有正式主会来源、原文公式与作者实现支持；实际迁移效果仍未知。准确的新颖性问题应是本项目的问题定义、监督信号与接入约束带来何种额外收益，已有门控/对象建模论文不构成停止研究的理由。

本次固定代码版本和本地快照见 [selective_code_manifest.json](sources/selective_code_manifest.json)。
