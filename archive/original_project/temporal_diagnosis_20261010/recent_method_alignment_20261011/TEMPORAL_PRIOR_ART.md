# DeepPro-Plus、TEP-SAM 与 NS-FPN：主源及实现核验

核验日期：2026-10-11。范围是论文、正式出版页和作者实现的只读核验；本文件没有新增训练、推理或性能复现。作者报告的优势、作者承认的限制、代码事实与本项目已测现象必须分别使用。

## 规范条目及状态

| 方法 | 标题、作者 | 出版状态及去重键 | 官方实现 |
|---|---|---|---|
| DeepPro / DeepPro-Plus | *Probing Deep into Temporal Profile Makes the Infrared Small Target Detector Much Better*；Ruojing Li, Wei An, Yingqian Wang, Xinyi Ying, Yimian Dai, Longguang Wang, Miao Li, Yulan Guo, Li Liu | 作者主页列为 TPAMI 2026，并记载 2026-04 接收；本轮未获得 IEEE 卷期/DOI。`arxiv:2506.12766`，已读 v5（2026-03-27） | [TinaLRJ/DeepPro](https://github.com/TinaLRJ/DeepPro)，本地冻结 commit `8fa1a68b94eb22e94ccd0529e6c5ceccdaa7ec28` |
| TEP-SAM | *Temporal-Emerged Prompting for Segment Anything in Multiframe Infrared Small Target Detection*；Yinghui Xing, Donghao Chu, Shizhou Zhang, Di Xu | [ICML 2026 正式 PMLR 页面](https://proceedings.mlr.press/v306/xing26b.html)，卷 306，140031–140045。`arxiv:2606.27655`，v1 | [cdh8285/TEP-SAM](https://github.com/cdh8285/TEP-SAM)，commit `c5165cfa9139e7d72f1fc962ddc398d31d59093e` |
| NS-FPN | *Seeing Through the Noise: Improving Infrared Small Target Detection and Segmentation from Noise Suppression Perspective*；Maoxun Yuan, Duanni Meng, Ziteng Xi, Tianyi Zhao, Shiji Zhao, Yimian Dai, Xingxing Wei | [CVPR 2026 正式 CVF 页面](https://openaccess.thecvf.com/content/CVPR2026/html/Yuan_Seeing_Through_the_Noise_Improving_Infrared_Small_Target_Detection_and_CVPR_2026_paper.html)，27783–27792。`arxiv:2508.06878`，v2 | [mengduann/NS-FPN](https://github.com/mengduann/NS-FPN)，commit `b857bef068ba48f1258b62de6bf082f73dbafde4` |

DeepPro 的 TPAMI 状态依据[第一作者主页](https://tinalrj.github.io/)，不能把本轮没有核验的 DOI、卷期补入条目。TEP-SAM 和 NS-FPN 的会议状态均有正式论文集页面，不仅是仓库自述。

## DeepPro-Plus：可确认的机制及边界

[论文 v5](https://arxiv.org/html/2506.12766v5) III-C、IV-E、IV-F、IV-G、Appendix E/F 支持：DeepPro 以每像素时间剖面的异常与相关性为核心；Plus 加入有限空间运算，保留单层结构，理论空间感受野为 19×19，输入 40 帧、8 个探针。RGBT-Tiny 试验由框生成粗高斯掩码。作者明确展示被遮挡及静止行人的漏检；静止超过窗口时，没有典型移动目标的灰度起落，因此作者将方法定位于运动目标，并讨论更有效利用长时序的可能性。**这些是作者给出的实例与解释，不是我们对 4,342 个低响应目标的统一归因。**

本地代码补充了两个重要细节：

- [TPro.py:18](/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/official_deeppro/networks/layers/TPro.py:18) 的探针实际为多头 `Linear(seqlen,out_len)`。其参数依赖时间索引，确实可成为受控干预对象；但“存在索引依赖”并不证明它造成有害位置敏感。
- `train.py:137` 选择 SoftIoU；`TrainDataLoader.py:58–59` 读取掩码后将非零值二值化。因此“作者由框生成高斯掩码”不等于本地训练过程直接回归连续高斯概率。当前项目的 box-max 正项和背景负项仍是自定义弱框适配，不是上述原始监督复现。

当前应称“发布 DeepPro-Plus 权重在冻结两段开发片段上的持续低响应与增损”，不应笼统称“DeepPro 缺少空间信息”，也不能由两个片段推导近期所有模型都有此问题。

## TEP-SAM：时序证据对照及原生协议

[论文](https://arxiv.org/html/2606.27655v1) §3、Appendix B/C 描述：差分卷积提取特征；ConvGRU 汇聚跨帧动态，局部时间注意力及当前帧与邻帧的特征差异提供补充；这些特征用于冻结 SAM 编码器的注入和自动提示。它不是仅靠“持续出现”筛候选。作者明示适用短窗口、较一致的背景运动；强自运动、视差及复杂动态下，目标与背景的运动差异可能不可靠。这是已出版方法自身的适用声明，尚不是我们数据上的失败证据。

官方实现已确认以下事实，所有链接固定到本轮 commit：

| 项目 | 代码依据与含义 |
|---|---|
| 时间访问 | [`datasets/seqwin.py:135–150`](https://github.com/cdh8285/TEP-SAM/blob/c5165cfa9139e7d72f1fc962ddc398d31d59093e/datasets/seqwin.py#L135) 以当前帧为中心取 7 帧，即 t−3…t+3；越界复制边界。不是因果在线模型，也不是 DeepPro 的五个 40 帧窗口预算 |
| 尺度及顺序 | [`configs/sam-vit-b-token-NUDT-MIRSDT.yaml`](https://github.com/cdh8285/TEP-SAM/blob/c5165cfa9139e7d72f1fc962ddc398d31d59093e/configs/sam-vit-b-token-NUDT-MIRSDT.yaml) 设 1024²、7 帧；[`test.py:61–75`](https://github.com/cdh8285/TEP-SAM/blob/c5165cfa9139e7d72f1fc962ddc398d31d59093e/test.py#L61) 先 sigmoid，随后 area 插值回原尺寸，再以 0.5 二值化。不能把 logits 插值与概率插值互换 |
| 实际时序路径 | [`image_encoder_token.py:493–559`](https://github.com/cdh8285/TEP-SAM/blob/c5165cfa9139e7d72f1fc962ddc398d31d59093e/models/mmseg/models/sam/image_encoder_token.py#L493) 同时有局部注意力、ConvGRU、中心特征和运动差异响应，并拼接融合；不能简称只有 ConvGRU |
| 监督 | [`sam_withToken.py:304–316`](https://github.com/cdh8285/TEP-SAM/blob/c5165cfa9139e7d72f1fc962ddc398d31d59093e/models/sam_withToken.py#L304) 对最终及中间掩码使用 BCE+Dice。只有框的开发集可以做目标级检测诊断，但不能冒充其原生掩码 IoU/nIoU 评价 |
| 可获得资源 | [README](https://github.com/cdh8285/TEP-SAM/blob/c5165cfa9139e7d72f1fc962ddc398d31d59093e/README.md) 提供 NUDT-MIRSDT、TSIRMT 数据来源及各自训练权重的 Google Drive 链接；此次仅验证链接存在，没有下载权重、校验哈希或运行 |

原生评价存在需要单独记录的差异：论文 Appendix A.2 的 Eq.20 写的是 `N_FP/N_frames`；[代码 `metric_new.py:8–62`](https://github.com/cdh8285/TEP-SAM/blob/c5165cfa9139e7d72f1fc962ddc398d31d59093e/test_utils/metric_new.py#L8) 实际使用未匹配组件面积除以总像素数，中心距离匹配阈值 `<3`。第 52 行又通过“面积数值是否属于已匹配面积列表”识别未匹配组件，同面积的不同组件可能相互混淆。此轮只记录代码事实，不修改作者实现或重算论文表格。后续若运行，应保存原生代码结果，并另列对象身份明确的公共诊断，不能将二者同名拼接。

附带复现细节：README 默认命令引用 `configs/sam-vit-b-token.yaml`，当前树中实际存在的是带 `NUDT-MIRSDT`/`TSIRMT` 后缀的配置；复现须显式指定实际文件。官方仓库当前公开的是 SAM1 版本，不应把论文 SAM2 参数和计算量填到该实现旁边。

## NS-FPN：空间背景抑制参照，而非已验证时序解决方案

[论文 v2](https://arxiv.org/html/2508.06878v2) §3–4 用低频特征引导高频净化，配合门控高斯滤波，再用螺旋采样融合与目标相关的特征。其试验基于单张 IRSTD-1k、NUAA-SIRST；分别接入 MSHNet 分割和 YOLOv8n-p2 检测。它为“增强目标时同时压制背景”提供近期机制参照，却没有直接证明能修复自然窗口变化造成的响应塌缩。

[官方仓库](https://github.com/mengduann/NS-FPN/tree/b857bef068ba48f1258b62de6bf082f73dbafde4) 已公开 MSHNet 分割训练/测试入口、两数据集权重下载链接、环境文件及需编译的 SFS deformable-attention 扩展。`model/NS_FPN.py` 可核查 DWT/IDWT、低频空间注意力与高频门控；`main.py` 使用密集掩码监督。此轮未核验 YOLO 检测分支的可运行资产。不能把分割权重直接称为框检测复现。

DeepPro-Plus 的单层时空结构没有与 NS-FPN 相同的多尺度 FPN 接口；“论文称模块可插拔”不等于可以无控制地接到当前模型，并把收益归因于解决时序问题。本轮将其作为机制和强背景参照，不启动新结构。

## 对当前决策的约束

1. DeepPro-Plus 的作者明示限制与本项目的低响应现象有可检查交集，但交集大小还没有被证明；不能先把所有失败重命名为静止或遮挡。
2. TEP-SAM 提供已存在的目标—背景时序差异建模路径；若未来提出类似差异或持续性机制，必须说明相对其 GLTM/TDM/提示方式新增什么信息或约束。
3. NS-FPN 说明空间背景抑制已有近期明确设计；仅“抑制困难背景”“增加高频处理”不足以成为新颖主张。
4. 三篇论文/代码都不能填入“在我们条件 C 下重复失败 D”的完整证据。只有运行对齐的现代对照、获得逐对象结果后，才能决定其剩余缺口是否值得新方法或数据路线。
5. TEP-SAM 和 NS-FPN 的当前素材足以支撑后续实现评估，尚未构成本地成功复现。此轮优先顺序保持 CHAL 原生公开设置核验，随后才是开发片段；不并行训练所有文献方法。
