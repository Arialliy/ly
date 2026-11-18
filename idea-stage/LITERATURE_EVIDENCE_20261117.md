# ICME 选题证据：2026-11-17

本轮目标：基于现有 DeepPro-Plus 工程，比较图像与视频红外弱小目标检测。文献检索不是“全球无人做过”的证明；下面区分正式会议、期刊与预印本。检索覆盖 2024–2026，并专门追加 2026 年 5–11 月的工作。项目根目录没有 papers/、literature/、research-wiki/，因而没有可加载的论文库或研究 wiki。

## 会议时间

- [ICME 2027 官方常规论文截止页面](https://2027.ieeeicme.org/events/regular-paper-submission-deadline)：2026-12-30，含 Special Sessions。
- [IEEE CASS 官方会议页](https://ieee-cas.org/event/conference/2027-ieee-international-conference-multimedia-expo-icme)：2027-07-13 至 17，厦门。
- [官方 CFP](https://confcats-siteplex.s3.amazonaws.com/icme/files/icme27-cfp_web-06.pdf)列出 Image and video processing 等主题。本轮只确认范围与日期，没有核实页数、匿名细则或截止时区。不得自行假定延期。

## 主要原始来源与已读内容

| 工作 | 来源/状态 | 已读范围 | 与本项目的关系 |
|---|---|---|---|
| DeepPro / DeepPro-Plus | [arXiv:2506.12766](https://arxiv.org/html/2506.12766v2) | 摘要、引言、相关工作、时间轮廓建模 | 固定像素时间轮廓、全局时间显著性与相关性；Plus已有空间信息，不能声称首次时空结合 |
| TDCNet | [AAAI 2026 正式页](https://ojs.aaai.org/index.php/AAAI/article/view/37385) | 摘要、方法概述；[预印本](https://arxiv.org/abs/2511.09352) | 多尺度时间差分与3D卷积重参数化，IRSTD-UAV数据；普通时间差分不是空白 |
| CHAL | [CVPR 2026 正式页](https://openaccess.thecvf.com/content/CVPR2026/html/Duan_CHAL_Causal-guided_Hierarchical_Anomaly-aware_Learning_for_Moving_Infrared_Small_Target_CVPR_2026_paper.html) | 官方摘要、[作者源码](https://github.com/UESTC-nnLab/CHAL)说明 | 背景演化神经场、层次异常、因果引导；不能以“背景建模替代目标增强”单独申报新颖 |
| SPIRIT | [arXiv:2602.01843](https://arxiv.org/html/2602.01843v1) | 摘要、引言、相关工作 | 已统一单帧/多帧，物理特征与历史空间先验引导记忆；统一框架本身不是新idea |
| TEP-SAM | [arXiv:2606.27655](https://arxiv.org/html/2606.27655v1) | 摘要、引言、方法目录 | 全局/局部运动差异生成SAM提示；不是把任意时序提示加SAM就有新意 |
| Decoupled Motion Representation Learning | [arXiv:2606.15286](https://arxiv.org/html/2606.15286v1) | 摘要、引言、方法结构 | 预训练光流、全局一致运动与局部异常分离，直接威胁“相机/目标运动解耦”提案 |
| FeedbackSTS-Det | [arXiv:2601.14690](https://arxiv.org/html/2601.14690v1)、[Remote Sensing 2026](https://www.mdpi.com/2072-4292/18/12/2042) | 摘要、引言 | 稀疏帧、双向反馈、隐式对齐；稀疏历史和反馈已有明确近邻 |
| BIRD | [arXiv:2508.15415](https://arxiv.org/html/2508.15415v1) | 摘要、引言 | 双向局部/全局传播；区分离线未来信息与严格因果部署 |
| Adaptive Frame Sampling and Feature Alignment | [Applied Sciences 2024](https://www.mdpi.com/2076-3417/14/14/6360) | 摘要、引言、采样说明 | 用互信息量化运动并按运动分布采样；单纯可变帧间隔不新 |
| SAIST | [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_SAIST_Segment_Any_Infrared_Small_Target_Model_Guided_by_Contrastive_CVPR_2025_paper.html) | 官方摘要 | CLIP、SAM、场景文本；“图像+文本/SAM”已经成熟竞争 |
| Text-IRSTD | [ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Huang_Text-IRSTD_Leveraging_Semantic_Text_to_Promote_Infrared_Small_Target_Detection_ICCV_2025_paper.html) | 官方摘要 | 模糊语义提示、跨模态解码与FZDT |
| DISTA-Net | [ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Han_DISTA-Net_Dynamic_Closely-Spaced_Infrared_Small_Target_Unmixing_ICCV_2025_paper.html) | 官方摘要/来源记录 | 近距目标解混、CSIST-100K、亚像素评价已有 |
| DISTA-Net++ | [arXiv:2609.18773](https://arxiv.org/html/2609.18773v1) | 摘要、引言、方法结构 | 2026-09-16；计数约束、连续坐标修正、解混歧义，最新重要近邻 |
| SPIRE | [arXiv:2604.05363v2](https://arxiv.org/html/2604.05363v2) | 摘要、引言、监督定义 | v2为2026-08-01；中心回归、概率点响应、高分辨率encoder-only；“不用mask改用点”不构成差异 |
| HALO | [arXiv:2607.17148](https://arxiv.org/html/2607.17148v1) | 摘要、引言、方法结构 | 2026-07-19；噪声框内辐射锚点与物理Gaussian软标签，直接威胁简单框转软标签路线；仅按预印本核验，不仅凭页面Journal字段断言录用 |
| InvDet | [CVPR 2026 PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Yan_Target-Aware_Invertible_Encoder_with_Reconstruction_Guidance_for_Infrared_Small_Target_CVPR_2026_paper.pdf) | 摘要与引言 | 可逆编码与目标重建缓解下采样损失；一般“保小目标高频细节”已有强竞争 |
| NS-FPN | [CVPR 2026 PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Yuan_Seeing_Through_the_Noise_Improving_Infrared_Small_Target_Detection_and_CVPR_2026_paper.pdf) | 摘要、LFP/SFS方法描述 | 本地LFP来源路线，不能将移植LFP等同原创机制 |
| Multidomain Statistical Consistency Modeling | [IEEE LGRS 2026](https://doi.org/10.1109/LGRS.2026.3718325) | 摘要与引言 | 时域低秩/频率一致性抑制干扰，2026-07-29；去相关/频率稳定性的广义提法不足 |

## 经典方法：避免把老问题当新问题

- [Temporal noise suppression…，Optik 2015](https://doi.org/10.1016/j.ijleo.2015.09.174)：时间噪声抑制与轮廓形状早已有研究。
- [Adaptive sequential algorithms…，SPIE 1999 原论文](https://citeseerx.ist.psu.edu/document?doi=edd5afae83038cb655978aaa996322ee61a9b1e9&repid=rep1&type=pdf)：目标出生/消亡、快速检测、延迟与虚警权衡早已存在；仅“序贯检测”不新。
- [Needle Picking，MERL 2010 原论文](https://www.merl.com/publications/docs/TR2010-018.pdf)：沿假设轨迹积累证据是经典track-before-detect；轨迹积分不是新机制。
- [High-Speed Incoming Infrared Target Detection…，Sensors 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4431292/)：静止、亚像素和快速目标的不同空间/时间检测分支；停走鲁棒性需要新的可验证差异。
- [Performance study on point target detection using super-resolution reconstruction，作者机构存档](https://repository.tudelft.nl/file/File_8c6f6e77-db27-4102-8356-dc8cf448a3f0)：采样混叠、超分辨率与点目标检测已有长期研究。
- [Point-Source Target Detection and Localization in Single-Frame Infrared Imagery，作者机构PDF](https://www.nsf-shrec.org/sites/default/files/2024-03/Point-Source_Target_Detection_and_Localization_in_Single-Frame_Infrared_Imagery.pdf)：单帧亚像素定位已有先例。

## 本地实证边界（只读审计）

1. `results/tlfp_pilot_20261012/REPORT.md`：S/T各512步；严格5200目标帧没有额外恢复；4342低响应和4170极低响应恢复均0。本帧替换基本保留T背景变化。只能描述这个pilot设置的负结果，不能推广为视频信息无用。
2. `experiments/head_lfp_joint_20261012/{RESULTS,DECISION}.md`：H有有限恢复但背景代价超预算；HL没有相对H的独立恢复优势。
3. `experiments/track_contrast_v1/RESULTS.md`、`track_contrast_v2/RESULTS.md`、`readout_weighting_v3/RESULTS.md`：轨迹对比、持续段风险和负样本加权尚未证明独立检测增量。
4. `experiments/m1_history_effect_v1/RESULTS.md`、`m1_recurrent_v1_1/RESULTS.md`、`m1_recurrent_v1_2/RESULTS.md`：这些历史融合的匹配增量未成立；不能复活为已有正结果。
5. `experiments/m1_recurrent_v1_3/runtime/WORKFLOW.json`记录RUNNING，但2026-11-17核对PID已不存在，进度未到计划终点，没有RESULTS.md。只作未完成记录，不继续旧任务。
6. `experiments/camera_motion_v1/RESULTS.md`：两个开发片段背景中心位移中位约0.055/0.089px，不能支撑显著相机运动问题。目标自身运动与背景运动须分开。
7. 本地40帧TPro为固定时间轴线性映射，不是QK内容注意力；整窗结果不等于严格在线结果。实际代码：`sources/official_deeppro/networks/layers/TPro.py`与权重相邻`DeepPro-Plus.py`。
8. 数据：完整NUDT-MIRSDT为合成序列且有mask；VT-Tiny-MOT是真实视频框/轨迹数据。现有固定8训练、2开发、4确认；本轮不查看确认集。M/U为本地诊断，不能改名官方Pd/Fa。作者RGBT Gaussian监督未取得。
9. 2026-11-17只读设备检查：4张RTX5090，约32GB/卡，显存空闲；本地仅GPU1/2/3有既有授权。旧3090耗时不能直接当新硬件实测。

## 检索词族

DeepPro/DeepPro-Plus temporal profile；ICME2027 official deadline；infrared small target CVPR/ICCV2025、CVPR2026；moving infrared target temporal 2026；causal/background anomaly；memory/streaming；counterfactual temporal context；frame rate/irregular sampling；stationary/stop-go/dwell；sequential false alarm duration；correlated noise/effective observations；subpixel sampling phase；uncertain/point/box supervision。

部分CVF网页直接open返回403，但官方索引搜索能给出论文摘要；原文不足的条目已标明读取层级。没有把搜索无结果写成不存在工作。

## 追加近邻查新

- [Multidimensional Digital Filters for Point-Target Detection in Cluttered Infrared Scenes，arXiv:1408.2590](https://arxiv.org/abs/1408.2590)：三维时空预测误差滤波，按局部背景速度调节白化；V4须与之比较。
- [Moving target detection by nonlinear adaptive filtering on temporal profiles…，IPT2015](https://doi.org/10.1016/j.infrared.2015.09.003)：非线性Wiener变体、分段方差估计、噪声归一化；V4不能只比较普通深度网络。
- Adaptive whitening filters for small target detection，SPIE1992，DOI 10.1117/12.139383：搜索发现经典近邻，但本轮仅取得作者上传摘要，未取得出版社全文，不据此断言新候选完全被包含。
- [Filter design…using normalized-cross-correlation layer，arXiv:2006.08162](https://arxiv.org/abs/2006.08162)：可学习检测滤波已有先例。
- [Rethinking Evaluation of Infrared Small Target Detection，NeurIPS 2025 Datasets and Benchmarks](https://papers.neurips.cc/paper_files/paper/2025/file/a81051ae2c8b1e46bd51480917b8ab84-Paper-Datasets_and_Benchmarks_Track.pdf)、[arXiv:2509.16888](https://arxiv.org/abs/2509.16888)：已有统一评价、错误分析、跨数据集泛化工作；新评价提案须具体到视频信息可获得时间或同帧反事实，而非泛称重思评价。
- [Point target detection and subpixel position estimation in optical imagery，Applied Optics 2004](https://opg.optica.org/ao/abstract.cfm?uri=ao-43-2-257)：亚像素位置与混叠影响目标签名、GLRT/LRT检测；I2不能声称首次发现相位盲点。
- [Frequency-aware spatio-temporal difference learning…，IPT 2026-11](https://doi.org/10.1016/j.infrared.2026.106904)：FSDL-Net结合短期频率差分与长期时间轮廓，进一步限制常规短/长期融合提案。
- [From Points to Targets，arXiv:2605.00722](https://arxiv.org/abs/2605.00722)：发现在线前景亲和/背景学习的相关工作；本轮搜索命中摘要，进一步细节需原文核验。
