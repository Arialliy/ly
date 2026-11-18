# CHAL：主源核验与最小复现可行性

日期：2026-10-11。范围：官方论文、补充材料、当前官方代码、公开资产及既有本地审计。不运行训练，不使用 GPU，不修改旧实验；无本轮 CHAL 检测成绩。

**结论：CHAL 是直接相关的 CVPR 2026 原生框检测比较对象，但当前作者公开资产不足以完成作者设置下的性能验证。** 缺少模型依赖 `nets/module/CBAM.py`、训练完成的 CHAL 权重，以及随仓库宣称提供的作者数据划分/框标注。不能把附带的通用初始化权重当成 CHAL 已训练模型，也不能自行补 CBAM 后声称复现作者结果。机器可读清单：[ASSET_PREFLIGHT.json](ASSET_PREFLIGHT.json)。

## 1. 来源与去重

- 正式论文：[CVF 页面](https://openaccess.thecvf.com/content/CVPR2026/html/Duan_CHAL_Causal-guided_Hierarchical_Anomaly-aware_Learning_for_Moving_Infrared_Small_Target_CVPR_2026_paper.html)，Weiwei Duan、Luping Ji、Shipeng Lei、Sicheng Zhu、Jianghong Huang、Mao Ye，CVPR 2026，pp. 21357–21366。官方 PDF 已取回并阅读，不仅依赖摘要；[本地 PDF](sources/CHAL_paper.pdf)、[全文文本](sources/CHAL_paper.txt)。
- 官方代码：[UESTC-nnLab/CHAL](https://github.com/UESTC-nnLab/CHAL)，当前 main=`34661770a7e61b1534de8ceb42a4c8e3d43ebdf6`，提交时间 2026-06-03。GitHub API 核验只有 main，无 tag、release。未发现许可证文件；GitHub license 字段为空。下载 ZIP SHA256=`dd34ca0ddb3b8dfe98f45a16319b7b9673b94ee6e2b3b48eec910fc7785d02e4`。
- 用户已有 `/home/ly/CHAL` 与这次官方 ZIP 中所有文件逐字节相同，Git 状态干净，commit 相同。因此复用 [2026-09-28 代码审计](/home/ly/chal_review/CHAL_CODE_AUDIT_20260928.md)、[CPU 诊断](/home/ly/chal_review/CHAL_CODE_SMOKE_20260928.json)、[优化器诊断](/home/ly/chal_review/CHAL_OPTIMIZER_AUDIT_20260928.json)。没有把既有工作当成新发现。
- [官方补充 ZIP](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Duan_CHAL_Causal-guided_Hierarchical_CVPR_2026_supplemental.zip) 的链接由本轮 CVF HTML 核实，复用 `/home/ly/chal_review/CHAL_supplement.zip`。其中仅 `35678_supp/35678_supp.pdf`，没有遗漏代码或权重。ZIP SHA256=`f4954bfa7130eb11863be21917b0789e30d1ab137759a103a088bdff5de4e891`。
- 稳定去重键：`cvpr2026/duan/chal`。本报告未完成整个领域新颖性检索。

## 2. 真正可借鉴的机制与其证据层级

论文 §3.1–3.4：共享 CSPDarknet/FPN 提取各帧特征；SNF 由场景编码及位置编码产生背景特征；HAL 从观测和背景的差异发现异常，再用时空注意力验证；CRG 用异常引导的门控调制当前帧特征，最后以 YOLOX 风格头输出框。全部模块受检测损失反传监督，论文并没有要求背景像素真值或独立重建损失。

当前代码细节：

| 事项 | 已核验事实 | 定位 |
|---|---|---|
| 背景条件 | 场景编码使用输入序列全部特征，包括当前帧；历史帧背景估计用前缀并重复该历史帧补齐五帧 | `neural_background_field.py:91–99,200–211`；`causal_anomaly_neck.py:201–221` |
| 异常来源 | LayerNorm、L2 归一化后求余弦距离，再经卷积产生异常图 | `causal_anomaly_neck.py:108–140` |
| 时序验证 | 将五张异常图嵌入后用 3D window attention；不是对最终二值点简单做持续性筛选 | `stst.py:383–398` |
| 输出 | 末帧单尺度 stride-8 YOLOX 框头，4 个回归量+objectness+类别 | `slowfastnet.py:89–141,167–185`；`train.py:136` |
| 监督 | GT 框经匹配生成定位、objectness、分类监督；不是像素 mask 训练 | `yolo_training.py:163–222` |

这些机制解释了 CHAL 为什么适合作为“低响应目标与背景竞争”的比较对象，**没有证明它会解决或重复我们 5,200 个目标帧中的失败**。代码在当前帧条件下估计背景，只能提出目标可能进入背景估计的待验证问题，不能把它直接写成已测根因。

补充材料 §F（[本地文本](/home/ly/chal_review/CHAL_supplement.txt:556)）已报告：亮点误检、运动云边误检，以及电线附近目标漏检。作者将后者解释为目标高频结构被背景估计吸收。这可支持“当前背景异常范式仍有目标/杂波区分缺口”的文献动机，不能替代我们的逐目标帧实验，也不能推断所有慢速目标都会被吸收。

## 3. 原生设置、指标及算力

| 项目 | 论文/实现证据 |
|---|---|
| 帧访问 | 5 帧，`t−4,…,t`，输出末帧；没有未来帧。训练 loader 对早期边界重复帧 0，而默认 test 重复帧 1，文件名规则也不同，需按原数据规范恢复 |
| 输入 | 论文红外实验 512×512；灰度转三通道，除以 255，再使用 ImageNet 均值方差。训练 loader 等比缩放后灰色填充；当前默认 test 为 RsCarData、1024×1024、直接双三次缩放 |
| 置信度 | `sigmoid(objectness) × sigmoid(class)`；默认 test 保留≥0.001 候选，NMS IoU=0.65；不是 DeepPro 像素概率≥0.5 |
| 原生评价 | COCO bbox AP/AP50。test.py 所称 precision/F1 来自第一类别插值 PR 数组，不是固定置信度工作点常规定义，需要与标准 AP 及新共同指标分别记录 |
| 论文训练 | 100 epochs，batch 4，AdamW，初始 LR 8e−5，weight decay 0.05 |
| 作者资源 | README：RTX4090 24GB；正文 Table 1：15.69M 参数、137.04 GFLOPs、12.96 FPS。为作者报告，非本机实测；不要按 FPS 推定训练时长 |
| 原生 NUDT 划分 | 补充材料：训练 63 视频/6,300 帧，测试 47 视频/4,700 帧，框由原 masks 转换；缺少具体转换/分割清单不能仅据名称视为同一版本 |

论文原生参照（正文 Table 1，百分数）：

| 数据集 | mAP50 | Pr | Re | F1 |
|---|---:|---:|---:|---:|
| DAUB-H | 54.28 | 73.08 | 75.26 | 74.15 |
| NUDT-MIRSDT | 75.25 | 86.35 | 88.49 | 87.41 |
| IRDST-R | 71.37 | 86.58 | 83.02 | 84.76 |

这些是核验时使用的作者参考值，不是本轮复现成绩。最小验证的第一步可只做一个作者数据集的固定小样本管线检查，但要认定恢复了作者性能，仍须对应完整原生测试划分；少量样本不能校验上述 AP。

代码定位：`test.py:19–40,75–115,118–123,149–159`；`utils/utils.py:4–35`；`utils/utils_bbox.py:27–90`；`utils/dataloader_for_DAUB.py:112–159`。

## 4. 资产与复现阻碍

本轮用既有 PyTorch 2.7 CPU 容器、`torch.load(..., map_location='cpu', weights_only=True)` 独立核验 `model_data/pretrain.pth`。它是 462 项 OrderedDict：backbone 354、head 108、neck 0。类别头为 80 类、三尺度、128 通道；当前 CHAL 为单类、单尺度、256 通道头。`train.py:119–129` 本来就只导入同名同形状部分。因此它是初始化资产，**不是完整 CHAL 检测权重**。权重 SHA256=`6f6759630c48010aa2a23ffb4b4dc82f2612e4d2120f4b6fcc21e9eecb1761bc`。

- `causal_anomaly_neck.py:21` 强制导入的 CBAM 文件缺失。相同 commit 的既有 CPU 测试已确认 `ModuleNotFoundError: No module named 'nets.module'`。
- `logs/model.pth` 缺失；没有其他最终权重下载链接、release 或补充文件。
- 仓库不含作者 TXT/JSON 数据划分。README 的三个百度分享均已尝试读取，本环境返回 SSL EOF；这只表示当前没有下载成功，不能说公开链接无效。现有 DeepPro NUDT 数据还需确认与作者框标注、序列边界、分割清单一一对应。
- 原生环境没有 lock 文件；README 的包版本还需构建一致环境核验。现有容器的 CPU 权重解析成功不意味着原生 GPU 运行已验证。

此外，既有审计记录了论文/代码差异：论文 N=2 而代码 N=1；训练异常图是五帧加权而推理只取末帧；门控放大/压低因子固定而论文描述可学习；定位损失为 NWD 与 CIoU 混合；优化器漏收部分直接声明参数；时间比例 `.item()` 阻断梯度；每个名义 epoch 实际截半。这些不能被静默修复后统称“官方 CHAL”，也不能据此声称论文性能无效或建立新的方法贡献。

## 5. 恢复资产后的唯一最小验证路径

1. 固定与最终权重对应的完整作者实现（包括精确 CBAM）、数据版本/框生成方式、训练和测试清单、配置与权重 SHA。解决纸面/代码差异的归属，不能自行猜测哪个版本产生论文结果。
2. 在隔离且可复现的环境中先做一条真实五帧输入、原生输出框和坐标回投的检查；保持末帧语义、归一化、缩放、NMS 和原生置信度规则。恢复环境时记录种子和计算核见证，不以 import 成功代替运行验证。
3. 在一个对应的作者公开测试设置上验证原生指标，并记录作者参考值、实际值及差距；不遍历测试集挑权重。
4. 原生验证通过后才进入两个开发片段：保留 5,200 行冻结目标身份和原有分组，追加 CHAL 原生框及共同定位判定。匹配变化、误检/未匹配预测、延迟、帧访问量分别报告。
5. 仅当逐目标对照显示明确重复失败，再填写“A 用 B；条件 C 下出现 D；E 加入 B 缺少的约束/信息”。目前只有 A/B 的主源证据和文献报告的失败例，缺我们条件 C/D 的 CHAL 实验；**尚不选新模块 E**。

共同评价必须另行冻结：框中心点诊断可以与现有 component-peak 目标存在性对照，但不是原生 bbox IoU 成绩；未匹配框与未匹配连通组件不同，不能直接把旧 `U≤4015` 当作跨模型公平预算；统一写 0.5 也不代表置信度校准相同。所有旧 DeepPro 0.5/U 成绩保持原定义。
