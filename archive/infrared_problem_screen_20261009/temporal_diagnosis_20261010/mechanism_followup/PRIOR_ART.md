# 从实测窗口敏感性出发的已有工作核对

检索日期：2026-10-10。范围：8项最接近的原论文或官方代码；使用`research-lit`的web来源流程。仅作问题边界、已有机制和对照核对，不提出新模型。源码快照与仓库树保存在`prior_sources/`，精简条目在`PRIOR_ART_SOURCES.json`。

**判断：已有工作不会直接否定当前官方checkpoint存在窗口敏感性的实测事实，但会否定把“时序上下文可能有害”“需要相对时间”“保持时序一致性”单独当作新idea。** 更窄的待验证条件是：在共同支持已经足以检出同一目标时，额外时序内容是否反而使目标响应塌缩。本轮文献核对未找到对此精确条件的直接解决证明；这个检索结果不构成新颖性证明，共同支持充分性也尚未被本报告认定为已证实。

## 1. 八项最接近来源

|工作与原始来源|已有解决范围与直接相关证据|尚不能由该来源推出的结论|代码、权重及当前训练域可比性|
|---|---|---|---|
|Li等，DeepPro，TPAMI 2026；[arXiv v5](https://arxiv.org/html/2506.12766v5)、[官方代码](https://github.com/TinaLRJ/DeepPro)|Sec IV-C2/Fig14明确讨论相对时间距离，观察SCorM近似次对角线对称；做输入长度、探针数、阶段结构消融。官方test对重叠预测取逐像素最大值。|论文中的长度消融不是固定当前图像的窗口起点干预；近似对称不等于严格平移等变，更不等于改变真实上下文后必然一致。|现有RGB-T官方checkpoint正是当前诊断对象；多起点训练和重叠最大值融合已经存在，不能当作新增设计。|
|Li等，DTUM，TNNLS在线2023/卷期2025；[官方实现](https://github.com/TinaLRJ/Multi-frame-infrared-small-target-detection-DTUM)、[论文DOI](https://doi.org/10.1109/TNNLS.2023.3331004)|通过方向编码时序卷积增强弱目标；官方NUDT loader使用最后5帧并监督末帧，支持mask/point监督；移动背景可启用配准。|没有在本次读到的官方训练路径中实现两个自然窗口对同一绝对帧的输出一致性；末帧固定位置也不能证明40→40模型不存在窗口敏感性。|官方有NUDT-MIRSDT权重入口及SatVideo版本相关入口；与RGB-T交通/船只域不同。跨域直接测只能作迁移诊断，公平优劣需共同训练协议。|
|Ying等，RFR，TGRS 2025；[论文v3](https://arxiv.org/html/2409.12448v3)、[官方代码](https://github.com/XinyiYing/RFR)|Sec V-C1直接比较滑窗与递归、窗口内递归与整段递归；报告滑窗帧数5→7改善、7→20下降，并讨论额外信息的歧义。PDA/TSFM处理对齐、目标信息保存及动态融合。|已有平均性能提升，不代表对任意历史重置、起点或上下文都稳定；未给出本报告所需的同帧受控“共同支持充分、额外内容有害”验证。|仓库包含IRSatVideo-LEO预训练权重。该数据以真实卫星背景和合成运动/目标构建，区别于RGB-T真实交通域；还依赖DCN编译。|
|Gao等，MIST/MISTNet，TIP 2026；[官方仓库](https://github.com/ShuCvlab/MIST)、[充分性损失源码](https://github.com/ShuCvlab/MIST/blob/c95bb20805b823ddbf86a0e9864147f2a43f3a68/deepmist/losses/loss_sufficiency.py)|SNCB处理复杂运动，PDD保留目标相关表示；SufficiencyLoss包含GT BCE与两路预测KL。官方配置使用5帧→1个末帧mask，充分性损失权重0.01。|源码里的充分性不是通过“删除额外上下文后仍能检出”定义；所核对loss/data路径是同次前向内部约束，不能等同跨自然窗口的同帧一致性。|有MIST/NUDT权重入口。MIST以合成目标融入真实背景构造；不应标为纯真实目标数据或RGB-T同域权重。|
|Feng等，MTTU-Net，Remote Sensing 2025；[原论文](https://www.mdpi.com/2072-4292/17/23/3801)、[官方代码](https://github.com/ZCFengF/MTTU-Net)|用记忆和时序Transformer传递历史；官方test逐帧前向并仅在序列首帧清空memory；train默认为5帧内逐帧BCE。|持续记忆避开分块重置，却仍可能依赖初始化和不可靠历史；论文的时域长度分析不等于同帧窗口不变性保证。|代码、IRDST_real/IDSMT数据与权重下载入口公开；下载链接中的实际checkpoint未本轮加载验证。训练域不是当前RGB-T域。|
|Shen等，LVNet，arXiv 2025；[论文](https://arxiv.org/html/2503.02220v1)、[官方模型](https://github.com/ZhihuaShen/LVNet/blob/22f31b28f299cff1f6ef5555c2abd03ef522298b/model.py)|使用Video Swin式时空窗口；源码有3D相对位置偏置、shift及attention mask；Sec IV-C显示增大时域对IRDST与NUDT的影响不同。|relative bias和shifted window不自动使完整模型严格等变，也没有证明任意真实上下文扩展都不伤目标。其解码“temporal consistency”表述不等同跨裁剪配对损失。|当前核对仓库树有model.py和README，未见README所列train.py/inference.py或实际权重链接；不能记为已具备完整可复现实验包。|
|Hou等，Unsupervised Image Sequence Registration and Enhancement，TGRS 2024；[作者主页论文PDF](https://mn.cs.tsinghua.edu.cn/xinwang/PDF/papers/2024_Unsupervised%20Image%20Sequence%20Registration%20and%20Enhancement%20for%20Infrared%20Small%20Target%20Detection.pdf)、[出版社记录](https://ieeexplore.ieee.org/document/10506478/)|已使用basket层级时序一致性、篮内一致/篮间多样性及mask，处理红外序列配准与增强。|Eq7约束时序变换/配准的连续性，不是要求两个重叠检测窗口对相同GT目标给相同置信度。不能说同任务没有一致性训练。|本轮未核实官方代码和可用权重；作者PDF可读。应作为一致性含义的近邻来源，不适合作为立即可运行的同域基线。|
|Zhang等，Decoupled Motion Representation Learning，arXiv 2026；[原论文](https://arxiv.org/html/2606.15286v1)|明确讨论全局背景运动掩盖局部弱目标；使用带遮挡mask的光度/SSIM一致性维持光流对应，再以YOLOX检测监督学习目标。|其一致性对象是相邻图像对应/光流，而非同一目标帧跨两个检测窗口的概率；其边界失效讨论也不等于已解决窗口内容负作用。|IRDST/ITSDT-15K、框检测mAP/F1协议；本轮未核实官方代码或权重，且不能与当前点匹配数直接比较。|

## 2. 对三个检索问题的回答

**跨重叠窗口一致性训练：** 在已核对的DeepPro/DTUM/RFR/MIST/MTTU公开训练与loss路径中，没有看到以“同一绝对当前帧在两个真实时间裁剪中的预测”为成对输入的显式训练项。这个结论只覆盖核对路径，不覆盖所有历史分支、未公开训练版本或整个领域。MIST已有充分性约束，配准论文及2026运动解耦论文已有时序一致性，故不能仅通过换名称宣称新颖。[MIST loss](https://github.com/ShuCvlab/MIST/blob/c95bb20805b823ddbf86a0e9864147f2a43f3a68/deepmist/losses/loss_sufficiency.py)、[配准论文](https://ieeexplore.ieee.org/document/10506478/)、[运动解耦论文](https://arxiv.org/html/2606.15286v1)。

**相对时间/平移等变：** DeepPro已经提出相对时间距离的重要性；LVNet已实现含时间轴的相对位置bias。它们不构成对整个网络的严格平移等变证明。进一步，即使对保留相同有效支持的平移严格等变，也不意味着加入或移除真实帧后的预测必须相同。后者改变信息量，应允许新上下文纠正原错误。[DeepPro Sec IV-C2](https://arxiv.org/html/2506.12766v5#S4.SS3.SSS2)、[LVNet源码](https://github.com/ZhihuaShen/LVNet/blob/22f31b28f299cff1f6ef5555c2abd03ef522298b/model.py)。

**边界/有效上下文处理：** 已有实际工程策略包括DeepPro重叠最大值融合、RFR/MTTU整段状态传播、MIST只评估具备完整历史的末帧或复制首帧补历史。这些策略改变覆盖率、时延或历史长度，均不保证目标局部响应不会被上下文误导；丢弃边界也必须报告丢弃帧而非将其视为正确检出。[DeepPro test.py](https://github.com/TinaLRJ/DeepPro/blob/8fa1a68b94eb22e94ccd0529e6c5ceccdaa7ec28/test.py)、[RFR test.py](https://github.com/XinyiYing/RFR/blob/34512685a99cc13d704a6aaf424692f14e9f950c/codes/test.py)、[MTTU test.py](https://github.com/ZCFengF/MTTU-Net/blob/0a2c2034d90eac81a38c1e80dbfefdc5e47b9232/test.py)、[MIST loader](https://github.com/ShuCvlab/MIST/blob/c95bb20805b823ddbf86a0e9864147f2a43f3a68/deepmist/datasets/MISTDataset.py)。

## 3. 是否否定候选问题，以及最低必要对照

当前问题可以继续作为**DeepPro-Plus的具体能力诊断**，但要把普遍已知事实与新增可检验条件分开。RFR已经研究额外时序信息的负作用；因此，只有“给更多上下文却更差”不足以构成新研究贡献。若后续共同支持干预确认一类目标已有充分证据、额外内容却造成响应塌缩，才可进一步检查这类错误是否超出已有动态融合、记忆和充分性约束的解决范围。若共同支持本身不足，不能把变化称为负迁移；它可能反映两窗口提供不同有效证据。

最低必要对照应按问题机制组织：

1. **原始官方DeepPro-Plus融合结果。** 单窗/居中/更多重叠/改变融合方式均应作为已有推理策略的控制，报告额外算力、覆盖率以及新增/丢失检出，不能用强翻转数替代性能增益。
2. **同训练预算的原采样训练对照。** 官方已用多窗口起点；如果后续改训练数据组织，要分开“同一批双窗口各自GT监督”与“另加配对约束”的作用，不能把更多样本曝光当成约束收益。
3. **若研究一致性，比较全图无条件一致与目标条件一致。** 这是检验选择条件是否必要的控制，不能先假设选择性有效；同时检查双低塌缩、伪目标稳定化，以及本应随新增证据改变的预测。
4. **至少两个原生同任务时序范式：DTUM与RFR。** 前者是短窗末帧输出，后者是长程递归动态融合；保留RFR完整历史/重置历史对照。若要对比目标信息保存，应增加MISTNet及其已有充分性约束。MTTU可作为补充状态模型。
5. **共同域、共同目标和原生协议。** 当前RGB-T交通/船只数据与各预训练域不同。直接加载异域checkpoint只能比较迁移行为；方法优劣结论需要共同训练/划分、尺度与标注协议。5→1方法没有任意slot输出，不应硬套40→40翻转指标；应在相同绝对帧、可比较的有效历史/未来信息和算力约束下诊断，另报原生Pd/Fa或框指标。

判定反例应预先保留：如果RFR或MIST在相同条件下已稳定且检测/误报优于原方法，则问题更可能是DeepPro特定实现能力缺口；如果所有方法都因真实证据缺失而失败，应转向采样/监督或数据覆盖诊断，不能坚持模型机制故事。

## 4. 检索局限

本次是定向核对，不是系统综述。查询覆盖`temporal crop/window offset`、`overlapping windows/clips consistency`、`relative temporal/shift equivariant`、`temporal consistency loss`等组合。MTU/TIP缩写未指向明确待查算法，未据缩写自行加入时序基线。论文未写出某一术语、或网页搜索无结果，均不作为不存在该技术的证据。MDPI正文抓取不稳定，因此MTTU的训练/状态行为主要以官方代码确认；DTUM出版社全文访问未成功，未声称完成其全篇所有消融排查。
