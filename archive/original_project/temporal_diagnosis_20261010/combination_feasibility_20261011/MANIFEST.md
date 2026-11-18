# 交付入口

- [REPORT.md](REPORT.md)：主结论、全部10种共用五窗表、逐片选择与路线决策。
- [ROUTE_DECISION.md](ROUTE_DECISION.md)：本轮关闭的搜索范围、保留的问题与下一项有限诊断。
- [PROTOCOL.md](PROTOCOL.md)、[PROTOCOL_FREEZE.json](PROTOCOL_FREEZE.json)：主实验运行前冻结规则。
- [LOW_RESPONSE_GRID_SUPPLEMENT.md](LOW_RESPONSE_GRID_SUPPLEMENT.md)、[追加登记](LOW_RESPONSE_GRID_SUPPLEMENT_FREEZE.json)：在粗阈值结果之后登记、运行新增细网格之前冻结；不更改主实验。
- [LEGAL_COMBINATIONS.json](LEGAL_COMBINATIONS.json)：462个五窗子集中的10个合法安排。
- [SHARED_COMBINATIONS.csv](SHARED_COMBINATIONS.csv)、[PER_CLIP_COMBINATIONS.csv](PER_CLIP_COMBINATIONS.csv)：10种共用、100种逐片组合的实际融合计数。
- [TRADEOFF_SELECTIONS.csv](TRADEOFF_SELECTIONS.csv)、[PARETO_GAIN_LOSS_U.csv](PARETO_GAIN_LOSS_U.csv)：冻结容限下的16项选择与增益/损失/U非支配集。
- [SELECTED_OBJECT_RESULTS.csv](SELECTED_OBJECT_RESULTS.csv)、[SELECTED_TRACK_RESULTS.csv](SELECTED_TRACK_RESULTS.csv)：主要选择的逐对象和逐轨迹变化。
- [ALL11_REFERENCE.csv](ALL11_REFERENCE.csv)：超出原计算预算的11窗参照。
- [FUSION_VS_SUPPORT.csv](FUSION_VS_SUPPORT.csv)、[SUPPORT_BUT_FUSION_MISS.csv](SUPPORT_BUT_FUSION_MISS.csv)：按对象身份联查的支持—融合差额。
- [scoring/REPORT.md](scoring/REPORT.md)：实际融合脚本、逐帧与逐组件结果、组件面积分布、历史精确复核、CPU运行记录。
- [support/REPORT.md](support/REPORT.md)：5200×11三态/响应矩阵、2048子集支持枚举、不可兼顾见证。
- [support/PEAK_UPPER_BOUND.md](support/PEAK_UPPER_BOUND.md)：858→809→五窗720/721的条件和证明。
- [low_response/REPORT.md](low_response/REPORT.md)：23点已有阈值曲线与追加9点的低组恢复、背景成本、原始logit缺失范围。
- [GAIN_LOSS_BUDGET.png](GAIN_LOSS_BUDGET.png)、[SVG](GAIN_LOSS_BUDGET.svg)：全部组合增损散点图。
- [AGGREGATE_SUMMARY.json](AGGREGATE_SUMMARY.json)、[CROSS_CHECK.json](CROSS_CHECK.json)、[FINAL_VALIDATION.json](FINAL_VALIDATION.json)：聚合、支持联查、最终交付验证。
- [ARTIFACT_SHA256.json](ARTIFACT_SHA256.json)：本目录交付哈希；自文件和`__pycache__`排除。

数据/权重仅从历史缓存读取。本轮CPU评分容器及根流程记录收录于[ROOT_RUNTIME.json](ROOT_RUNTIME.json)和各子目录运行文件。历史`problem_validity_20261011`、`failure_attribution_20261011`及其冻结引用保持不变。重算主流程依次执行`scoring/score_combinations.py`、`support/analyze_support.py`、`aggregate_combinations.py`、`cross_check.py`；多数子脚本为防覆盖会拒绝在已有完整结果的目录再次运行，复现实验应使用新输出目录。匹配依赖原项目helper及原缓存路径，所有输入指纹随结果保存。
