# 来源与适配边界

本包日期：2026-10-13。独立参考实现，不是作者完整代码复现，也不是已接入本地工程的模型。

## 1. 直接代码依据

DEIM: DETR with Improved Matching for Fast Convergence，CVPR 2025。

- 正式论文：https://openaccess.thecvf.com/content/CVPR2025/html/Huang_DEIM_DETR_with_Improved_Matching_for_Fast_Convergence_CVPR_2025_paper.html
- 官方源码：https://raw.githubusercontent.com/Intellindust-AI-Lab/DEIM/main/engine/deim/deim_criterion.py
- 类与方法：DEIMCriterion.loss_labels_mal。
- 此次网页可见实现：对 `src_logits` 的 sigmoid 结果停止梯度；负例权重使用该概率的 gamma 次幂；BCE-with-logits 接收该权重。gamma 默认2。
- 作者完整 MAL 还包含匹配索引、框 IoU、正例质量软标签、特定归约。**本包没有采用这些部分**。

## 2. 本项目如何改变

| 官方/现有部分 | 本包处理 |
|---|---|
| DEIM 负例的 detached p^gamma 权重 | 在已有 C-frame 背景读出上独立实现同一计算 |
| DEIM 正例 IoU 质量标签 | 不移植，不编造 IoU，不称框内峰为高质量框匹配 |
| 本项目已有正例 softplus(-q+) | 原样保留，不按当前低置信度降低正监督 |
| 本项目已有取样、背景掩码和外层归约 | 留给本地原代码；包内不猜测这些规则 |
| 当前 GAP | 本轮所有组均不添加；旧结果保留 |
| matched_confidence | 本包自定义的梯度预算重标定，不是 DEIM 原方法 |

这是一项有作者代码参照的训练修订，不是新的时序算子，不是 DEIM 整体移植，也不以此单独主张创新。

## 3. 验证与溯源状态

- 官方源码已通过网页读取到相关方法；未在本环境下载完整仓库。
- 容器直接访问 GitHub API 失败（DNS不可用），未取得可靠 commit；不编造哈希或固定版本。
- 本包没有复制并分发作者源码、权重或许可证文件。
- `test_readout_weighting.py`核对独立实现与已读公式，不是运行作者原函数。
- `check_official_kernel.py`供本地在可信作者仓库上运行：实际执行两段原函数，仅比较全背景分支的值与梯度，并记录文件SHA256和git commit。
- 此处未运行作者函数对照，不声称已经通过该对照。

## 4. 不作为本轮直接实现依据的论文

SlotContrast、TrackMAE的思想在已有C中保留。本轮不复制其网络，不继续将它们列为本项新公式的来源。

DeepPro-Plus接口参考：https://raw.githubusercontent.com/TinaLRJ/DeepPro/main/networks/models/DeepPro-Plus.py

本地冻结快照为最终依据。本包未访问 `/home/ly/DeepProPlus_TLFP_Code`。
