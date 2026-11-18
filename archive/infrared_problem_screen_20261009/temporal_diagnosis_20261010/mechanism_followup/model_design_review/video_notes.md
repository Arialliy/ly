# 视频检测模型设计：原论文与官方实现核验

检索日期：2026-10-10。用途：为已经观察到的 DeepPro-Plus 跨真实窗口响应翻转寻找可检验的设计，不把文献性能视为红外结果，不重新定义研究问题。仅使用原论文、作者仓库和作者所在机构的出版记录作为依据。

## 核心判断

可借鉴的是**按当前目标/位置选择上下文贡献、保留局部证据、避免只由单帧高分决定候选是否存在**。现有视频检测论文没有直接证明其方法能满足“同一当前帧换真实窗口后不塌缩且保留有利上下文”的要求。普通残差连接、非负注意力或更大记忆均不提供该保证。

目前更适合小规模迁移的是 TF-Blender 的逐位置/通道动态聚合；SELSA 提供语义亲和选择对照；MEGA 提供局部与全局分工的设计依据；TransVOD++ 提供区域内容与查询联合选择的设计依据。后两者整体替换 DeepPro 成本较高，应先迁移机制再判断。

## Entry V1 — TF-Blender

- **作者/出处**：Yiming Cui, Liqi Yan, Zhiwen Cao, Dongfang Liu，ICCV 2021。
- **dedup_key**：`arxiv:2108.05821`。
- **primaryURLs**：[CVF记录](https://openaccess.thecvf.com/content/ICCV2021/html/Cui_TF-Blender_Temporal_Feature_Blender_for_Video_Object_Detection_ICCV_2021_paper.html)、[论文PDF](https://openaccess.thecvf.com/content/ICCV2021/papers/Cui_TF-Blender_Temporal_Feature_Blender_for_Video_Object_Detection_ICCV_2021_paper.pdf)、[官方仓库](https://github.com/goodproj13/TF-Blender)、[实际聚合代码](https://github.com/goodproj13/TF-Blender/blob/master/mmtrack/models/aggregators/tf_aggregator.py)。
- **原文定位**：Sec.3.2–3.4，Eq.(2),(6)–(9)，Table 3–5。
- **论文机制**：输入当前/邻帧特征及双向差，三层卷积生成与特征同尺寸的权重；邻帧先经过相互调整；ReLU处理权重，通道softmax处理调整后特征；最终逐元素加权。Eq.(8)将余弦相似度大于0.7的邻帧权重置零，作者用来排除冗余。
- **原消融**：Table 3，FGFA基线77.8 mAP，仅Temporal Relation为78.5，三个模块合用79.3。此处为ImageNet VID，不能拿来与本项目点匹配计数比较。
- **代码核验**：论文链接的仓库可访问，确有可读Python实现。但`tf_aggregator.py`实际拼接8组原始/归一化特征和双向差，经1×1→3×3→1×1卷积，再对**邻帧维**softmax，加权原始`ref_x`。此文件没有显式实现论文中的邻帧两两FA、最终ReLU或相似度阈值。因此不能将该文件直接称为论文全部三模块的忠实复现。
- **迁移推断**：将`channels`改为32、批次及40个输出槽位向量化，可以先做TPro前的动态贡献控制对照。由于各帧目标会移动，应在局部搜索/可审计的对齐后计算关系，不能假定同坐标就是同一目标。若计算全部40×40密集关系，要分块或限制局部候选，报告额外算力。
- **不能直接解决**：相似度不是“是否伤害当前目标”的标签；红外弱目标与背景都可能相似；softmax即使所有邻帧不可靠仍必须分配总权重1；去掉相似帧还可能丢失弱信号累积。首轮只借动态关系权重，不预设0.7阈值适用，也不预设非负权重能保证输出概率不下降。

## Entry V2 — SELSA

- **作者/出处**：Haiping Wu, Yuntao Chen, Naiyan Wang, Zhaoxiang Zhang，ICCV 2019。
- **dedup_key**：`arxiv:1907.06390`。
- **primaryURLs**：[论文PDF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wu_Sequence_Level_Semantics_Aggregation_for_Video_Object_Detection_ICCV_2019_paper.pdf)、[arXiv](https://arxiv.org/abs/1907.06390)、[官方MXNet仓库](https://github.com/happywu/Sequence-Level-Semantics-Aggregation)。
- **原文定位**：Sec.3.2 Eq.(1)–(2)，Sec.4.3 Table 1与Fig.3。
- **论文机制**：对参考RoI与全序列候选RoI计算嵌入内积亲和度，跨候选softmax后做特征加权。它选择语义邻居而不只按时间距离寻找邻居，不依赖光流对齐。
- **原消融**：Table 1，基线73.62 mAP，同帧语义聚合75.26，加入序列信息80.25。再加数据增强才得到82.69；不要将82.69全部归因于聚合模块。
- **代码核验**：作者仓库明确为官方MXNet实现，含训练、验证与结果；不是直接可导入本项目的PyTorch模块。
- **迁移推断**：可将亲和度聚合缩小为当前局部弱目标候选对邻帧局部token的选择，作为“纯相似性注意力”强对照。若保留密集输出，要重写RoI接口并防止仅保留高分proposal造成小目标提前消失；32通道亲和度质量需重新训练验证。
- **不能直接解决**：模型可能关联到相似背景或其他目标，且没有局部当前证据保护或跨窗口最弱上下文监督；扩大采样范围也不直接回答何时应拒绝上下文。

## Entry V3 — MEGA

- **作者/出处**：Yihong Chen, Yue Cao, Han Hu, Liwei Wang，CVPR 2020。
- **dedup_key**：`arxiv:2003.12063`。
- **primaryURLs**：[CVF记录](https://openaccess.thecvf.com/content_CVPR_2020/html/Chen_Memory_Enhanced_Global-Local_Aggregation_for_Video_Object_Detection_CVPR_2020_paper.html)、[论文PDF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_Memory_Enhanced_Global-Local_Aggregation_for_Video_Object_Detection_CVPR_2020_paper.pdf)、[官方PyTorch仓库](https://github.com/Scalsol/mega.pytorch)。
- **原文定位**：Sec.3.1–3.2 Eq.(1)–(4)，Fig.2，Sec.4.5 Table 3–4。
- **论文机制**：先用不含位置的全局关系模块增强局部proposal，再用含时空位置的局部关系聚合；长程记忆缓存之前的中间特征，扩大可用上下文。其逻辑是局部定位信息与全局语义信息分工。
- **原消融**：Table 3，单帧75.4、全局+局部81.4、加记忆82.9 mAP；Table 4去全局为81.3，去局部为81.8。两个方向均有用，但消融未测试“新增上下文造成当前帧退化”的率。
- **代码核验**：作者PyTorch实现，基于maskrcnn_benchmark；README同时说明实现了FGFA/RDN，不能据此将仓库里的RDN称为RDN作者官方实现。
- **迁移推断**：最小借鉴是保留独立的局部/当前目标证据分支，让40帧全局分支提供可调节的补充。迁移全套RoI记忆会引入候选生成、时序缓存与额外帧访问，不能作为相同40帧预算的直接比较。
- **不能直接解决**：更大上下文或记忆也可能累积不利内容；全局/局部分支相加仍可能经后端降低响应。若借此设计门控，门控的收益/损害判据必须来自本任务监督和独立验证，不能从MEGA的均值mAP提升推出。

## Entry V4 — TransVOD / TransVOD++

- **作者/出处**：Qianyu Zhou, Xiangtai Li, Lu He, Yibo Yang, Guangliang Cheng, Yunhai Tong, Lizhuang Ma, Dacheng Tao；IEEE TPAMI **2023**, 45(6):7853–7869；2022为在线发表年。
- **dedup_key**：`doi:10.1109/TPAMI.2022.3223955`；同一记录`arxiv:2201.05047`。不要与ACM MM 2021、`arxiv:2105.10920`的早期TransVOD版本混为相同实验。
- **primaryURLs**：[作者机构出版记录](https://livrepository.liverpool.ac.uk/3168986/)、[作者接受稿](https://livrepository.liverpool.ac.uk/3168986/6/IEEE_TPAMI_TransVOD.pdf)、[arXiv](https://arxiv.org/abs/2201.05047)、[官方++仓库](https://github.com/qianyuzqy/TransVOD_plusplus)、[官方Lite仓库](https://github.com/qianyuzqy/TransVOD_Lite)。
- **原文定位**：Sec.3.3 Fig.4和Eq.(5)，Sec.4.3.2 Table 6(a)–(c)。
- **论文机制**：QRF以查询生成动态卷积参数，融合对应RoI外观；HQM按分类得分逐层减少参考和当前查询，TQE做跨帧查询聚合，各阶段用辅助检测损失训练。
- **原消融**：Swin-B基线→QRF→加HQM的AP50为88.3→89.7→90.0；小目标AP50:95为14.2→16.0→17.6。Table6(c)不使用COCO预训练时单帧基线AP50从88.3降至44.8，迁移成本必须计入。
- **代码核验**：作者++仓库含`models/deformable_transformer_multi.py`与训练脚本；实际代码含QRF动态层，参考query使用80/50/30×参考帧数的top-k。论文与代码仍需完整复现审计后再声称模块严格一致，未实际安装运行。
- **迁移推断**：可借“局部区域外观参与上下文选择”，但DeepPro为密集32通道40→40输出，缺少RoI/query体系；整体替换已是另一检测器。小原型更适合在目标候选邻域提取内容描述后控制原TPro，而不直接照搬DETR/Swin和300查询。
- **不能直接解决**：HQM里的hard是逐级筛查询，**不等于同帧跨窗口弱一路正损失挖掘**。分数top-k可能在增强前丢弃本任务的低响应真目标；弱小红外目标需要保留低分候选通路或密集监督。其平均精度提升未验证本任务的跨窗响应保护。

## 附：RDN可作为相邻设计，不增列主候选

- Jiajun Deng等，ICCV 2019，`arxiv:1908.09511`：[原PDF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Deng_Relation_Distillation_Networks_for_Video_Object_Detection_ICCV_2019_paper.pdf)、[arXiv](https://arxiv.org/abs/1908.09511)。Sec.3.3先用全部支持proposal作基本关系推理，再按objectness选top r%支持proposal做高级关系蒸馏。它明确讨论无效proposal增多会影响关系学习稳定性。但高objectness筛选同样有提前删除弱目标风险；本轮只核验到MEGA仓库第三方实现，未核验到RDN作者独立官方仓库。

## 面向本问题的比较边界

1. 最低限度需比较原TPro、SELSA式纯亲和聚合、TF-Blender官方代码式局部动态聚合，以及拟议的“局部证据+受控上下文”候选。所有模型用相同40帧、标签、训练曝光和推理融合，变化参数/算力另报。
2. 不能直接优化“分数方差更小”：双低也能稳定。应在同误报预算下同时报告受抑制目标恢复、有利上下文样本保留、原成功样本新增漏检、连续漏检。
3. 配对真实窗口可为模型门控提供针对性训练样本，也可单独改善原模型；数据/监督路线与模型路线应拆开消融，不能把多个模块叠在一起后解释为已定位机制。
4. 当前97强翻转、24个内部干预强抑制、42个内部干预需要额外支持，只支持优先验证这一假设。共同槽特征仍有±8帧依赖，投影删除可能分布外；不将24例写成自然输入因果证明，不将本设计承诺覆盖778个所有窗口均低响应的目标帧。

检索实现备注：web浏览器部分CVF PDF返回错误，已直接从相同CVF原始URL下载并用pdftotext核验正文、公式和表格；没有以搜索摘要替代全文消融核验。原PDF和读取的两份官方代码临时保存在`/tmp/video_model_review/`。
