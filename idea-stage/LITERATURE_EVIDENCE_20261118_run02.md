# 第二轮查新证据索引

检索日期：2026-11-18（Asia/Taipei）。仅把原论文、作者代码库和数据集官网作为结论依据；未发现直接重合不等于证明没有前人。E1表示本轮读取相关原文方法/实验段落，E2表示核验原始摘要，D表示核验官方数据说明。部分读取由独立研究分支完成，原始记录在本轮trace目录。以下是会改变决策的关键来源，不是全部检索结果的堆积。

| 对应候选 | 一手来源 | 核验范围与对选择的影响 |
|---|---|---|
| L1 | [Deep-NFA，Pattern Recognition 2024](https://doi.org/10.1016/j.patcog.2024.110312) | E1，方法§3和§4：a-contrario已经进入小目标网络训练；协方差建模的是特征通道，spatial block使用注意力。不能宣称首次误报统计训练。 |
| L1 | [Statistical Modeling of Deep Features to Reduce False Alarms in Video Change Detection，JMIV 2025](https://link.springer.com/article/10.1007/s10851-025-01238-w) | E1，§4.2：使用独立近似及感受野校正，明确讨论二维依赖随机场的困难。支持继续研究依赖结构，不能据此宣布问题尚无人解决。 |
| L1 | [Towards Reliable Detection of Empty Space，2025/2026](https://arxiv.org/abs/2506.21486) | E2：标记点过程训练与区域无目标概率已有先例。目标存在过程与背景误报响应场须区分。 |
| L1 | [Normal Tail Suppression，2026](https://www.mdpi.com/2313-433X/12/10/466) | E1：惩罚正常分数上尾。仅把top-k改名不能形成差异。 |
| L1 | [Distribution-free false-alarm calibration，2026](https://arxiv.org/abs/2608.15090) | E2：有限正常样本约束误报校准精度；相关视频帧不能当独立样本。不能以先导宣称极低误报保证。 |
| L1 | [Local scattering ultrasound imaging，2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC7806797/) | E1，相关段落：Gaussian random field与有效分辨单元用于全图误报已有历史。面积修正和有效独立次数本身不是新概念。 |
| S1 | [UNSURE，ICLR 2025](https://arxiv.org/abs/2409.01985) | E1/E2：表达能力与自监督稳健性权衡；分支核验附录已有孤立亮点被Noise2Self抹除的例子。只能审查更具体尺度—噪声依赖结论。 |
| S2 | [N-AGC，ICCVW 2025](https://openaccess.thecvf.com/content/ICCV2025W/MIRA/html/Javidnia_Task-Driven_Neural_Adaptive_Gain_Control_16-bit-to-8-bit_Thermal_Tone_Mapping_for_ICCVW_2025_paper.html) | E2：任务导向16-bit到8-bit增益/偏置映射已存在。遥远区域影响的限制必须胜过经典局部曝光与稳健分位数。 |
| S2 | [RAWild，2026](https://arxiv.org/abs/2605.05941) | E2：RAW处理及跨传感器研究的直接邻居；不能把跨传感器ISP本身当创新。 |
| S3 | [HalluciDet，WACV 2024](https://arxiv.org/html/2310.04662v1) | E1，方法与loss ablation：冻结检测器监督IR任务图生成已经做过；其hallucination不等于错误对象生成。 |
| S3 | [SR4IR，CVPR 2024](https://arxiv.org/html/2404.01692v1) | E1：识别任务特征捷径及任务导向恢复偏置已被研究。尚未核实其包含本候选的目标删除因果比较。 |
| S3 | [TaskGuard，2026](https://arxiv.org/abs/2609.08011) | E2：冻结恢复器的区域效用、反事实和检测敏感度已被研究；普通“增强有害”实验不足。 |
| S3 | [HalluGen，CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Kim_HalluGen_Synthesizing_Realistic_and_Controllable_Hallucinations_for_Evaluating_Image_Restoration_CVPR_2026_paper.pdf) | E1，摘要/引言：可控幻觉及评价已有体系；本候选必须隔离训练检测梯度的因果作用。 |
| V1 | [RUO，2026-11-01预印本](https://arxiv.org/html/2611.01573v1) | E1，方法§3：按预期检测效用和实际字节成本分配区域码率，已检验不同发送/接收检测器。微小目标保留及跨检测器验证都不是新点。 |
| V1 | [FP-RDO，2025](https://arxiv.org/abs/2504.02216) | E2：特征保持率失真及块级Jacobian近似已有；逐实例决策约束需要额外实质论证。 |
| V2 | [AnytimeYOLO，2025](https://arxiv.org/html/2503.17497v1) | E1，分支读取：预算适应推理已有；更直接的部分和证书先例见独立评审，不能只与一般early exit比较。 |
| V3 | [A U-Net Based Discriminator for GANs，CVPR 2020](https://arxiv.org/html/2002.12655) | E1，jury核验§3.2：拼接输入与对应输出拼接的一致性公式已有；移到视频检测不使正则本身成为新发明。 |
| L2 | [Cut, Paste and Learn，ICCV 2017](https://arxiv.org/abs/1708.01642) | E1，§5.2：多融合及合成distractor已用于削弱粘贴边界捷径。 |
| L2 | [Learning to Generate Synthetic Data via Compositing / TERSE，CVPR 2019](https://arxiv.org/html/1904.05475) | E1，jury核验：把前景形状的背景cutout粘贴成负样本，使正负都有相同blending artifacts，是更直接先例。 |
| L2 | [The Alpha Blending Hypothesis，2026](https://arxiv.org/html/2605.10334v1) | E1，§4.3及补充：self-blended images标为real，直接解除融合伪影与fake标签的关联。 |
| L2 | [Negative Sample Copy-Paste，ICONIP 2024作者全文](https://kclpure.kcl.ac.uk/ws/portalfiles/portal/325362069/6910.pdf) | E1，§3：挖掘误报负块并seamless clone；是小目标合成训练的必要基线。 |
| L3 | [DivNoising，ICLR 2021](https://arxiv.org/abs/2006.06072) | E2及作者代码说明：从带噪图学习恢复分布、采样和下游细胞分割均有先例。 |
| L3 | [Hierarchical DivNoising，2021](https://arxiv.org/abs/2104.01374) | E2：无监督多样恢复与层次VAE已有。 |
| L3 | [Ambient-Pix2PixGAN，2024](https://arxiv.org/abs/2402.01186) | E2：带噪训练数据中的条件生成已有；不是只和无条件生成模型比较。 |
| L3 | [Learning Clean Generative Processes from Noisy Measurements，2026-10-30](https://arxiv.org/abs/2610.42036) | E2：双噪声视图、clean generative process及显微应用的非常近竞品。具体proper-scoring目标差异仍需审查。 |

## 数据证据与缺口

| 数据 | 可支持的工作 | 已核验与尚未核验 |
|---|---|---|
| 本地NUDT-MIRSDT120及既有train/dev | 合成机制调试、现有检测器可行性；不能证明真实传感器噪声规律 | 本地项目事实；确认集不读取。NUDT是合成数据。 |
| [SIRST](https://github.com/YimianDai/sirst) | 真实红外检测与像素标注，目标外区域辅助统计 | 作者仓库D；目标外裁块不能替代大量独立完整无目标图。实际可用阴性规模未审计。 |
| [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) | 完整正常图上的误报告警及工业图像跨任务检验 | 官网D：训练集无缺陷；测试含正常/异常；CC BY-NC-SA 4.0。未实际下载验证端点，不能宣称已就绪。 |
| [Anti-UAV作者项目](https://github.com/ZhaoJ9014/Anti-UAV) | 自然目标缺席帧的潜在IR误报验证 | 官方D：exist标记；各版本、许可、可下载性和独立场景数尚需核验。只作为候选数据，不改为停走/缺测方向。 |
| [FMD](https://github.com/yinhaoz/denoising-fluorescence) | 多次带噪测量用于去噪机制验证 | 作者D；按视野分组，不能把同一视野重复帧分到训练/测试。检测标签需另核验。 |
| [AODRaw](https://github.com/lzyhha/AODRaw) | RAW/处理图/检测框，用于ISP候选 | 作者D，完整包很大；未下载，许可和可获取子集待核验。现有8-bit IR不能冒充RAW。 |

本轮未下载大型数据或启动训练。来源的研究结论、我们据此提出的假设、待验证的方法效果在报告中分别标注；不把本索引当作“无人做过”的证明。
