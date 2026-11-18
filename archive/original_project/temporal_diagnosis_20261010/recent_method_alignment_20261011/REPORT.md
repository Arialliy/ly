# 近期机制核验与路线交接

已核验六项方法主源及实现，结果见 [时序与空间机制](TEMPORAL_PRIOR_ART.md)、[监督机制](SUPERVISION_PRIOR_ART.md)、[CHAL 可复现性](CHAL_FEASIBILITY.md)。SPIRE 已查到正式 ECCV2026 Springer 章节；DeepPro-Plus 已有有限空间信息且作者报告静止目标局限；PAL 已包含高损失像素挖掘；TEP-SAM 的原生 Fa 代码与论文公式须分开。

CHAL 官方仓库与既有审计同一commit。缺必需CBAM、完整训练权重及作者框标注划分；只有通用初始化权重，不能产生作者成绩。本轮无CHAL训练或检测结果，详见 [资产核验](ASSET_PREFLIGHT.json)。已冻结5200对象身份，原有4342/4170/646/104分层均核对一致；它们仍是DeepPro定义的开发集合。

**用户后续明确指定保留 DeepPro 基线并直接实现 BC-TPro。CHAL 不再是原型开始的前置条件。** 上述结果用于约束机制与新颖性表述，原生比较协议仅留作后续。当前工作转到 [BC-TPro 单次原型对照](../bc_tpro_pilot_20261011/PROTOCOL.md)：同一冻结DeepPro-Plus上实现完整候选和单路动态控制，保持top-1%、512更新、O/S和0.5，评估恢复、保持与背景成本。

本轮用本地旧CHAL论文及审计作为起点，联网核验当前主源；项目papers仅有一篇不相关的HierUQ分类论文，未作为本任务证据。可用环境中无Zotero/Obsidian连接器，arXiv helper不在既定路径，已用arXiv主站与正式论文集交叉核验。此处不是穷尽式查新，也没有将作者失败解释当作本项目的实验结论。
