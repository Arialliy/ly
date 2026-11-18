# 本轮交付索引

本轮任务：用原发布权重的已有预测，完成逐窗口实际匹配—最终融合联查和有限的同预算简单控制。没有新增训练，没有使用保留确认集的模型结果。该任务完成不代表研究问题已正式定题。

|工件|内容|
|---|---|
|[REPORT.md](REPORT.md)|主要结果、样本来源、实际损失、控制、解释边界和下一步|
|[PROTOCOL.md](PROTOCOL.md)、[登记哈希](PROTOCOL_FREEZE.json)|结果生成前定义；原登记及取样来源纠正另有归档|
|[TABLE.csv](TABLE.csv)、[SUMMARY.json](SUMMARY.json)|全部目标、几何小目标、多窗口层及旧1417/97队列的联表|
|[OBJECTS.csv](OBJECTS.csv)、[WINDOWS.csv](WINDOWS.csv)|21,270个对象帧与58,686条对象—窗口记录|
|[TRACKS.csv](TRACKS.csv)、[CONDITIONS.csv](CONDITIONS.csv)|逐轨迹与描述性条件；不作因果解释|
|[REMAINING_OPPORTUNITIES.csv](REMAINING_OPPORTUNITIES.csv)|主联表中的机会对象；文件名不表示已经通过控制恢复或剩余的部署收益|
|[control_summary](control_summary/SUMMARY.json)|固定0.5、开发内校准、固定方向迁移及逐对象新增/丢失|
|[simple_controls](simple_controls/README.md)|概率融合、计算/信息约束、评分曲线与完整性记录|
|[evidence_context](evidence_context/README.md)|157个小目标机会对象的500条窗口标注证据|
|[supplemental_diagnostics](supplemental_diagnostics/README.md)|稳定保护集、内部槽位和轨迹净增损的独立复算|
|[旧组审计](audit_old/REPORT.md)、[补充组审计](audit_new/REPORT.md)|实际匹配联表独立复算|
|[控制审计](audit_controls/REPORT.md)、[文字审计](audit_controls/CLAIM_REVIEW.md)|控制全部阈值、增损与结论范围核查|
|[join_matches.py](join_matches.py)、[summarize_controls.py](summarize_controls.py)|主要汇总脚本；评分命令与参数见子目录README/运行记录|
|[JOIN_REVISION.json](runtime/JOIN_REVISION.json)|NPZ解压缓存优化、原结果归档、一次拒绝覆盖退出1的记录|
|[研究计划](../stage1_goal/CVPR2027_RESEARCH_PLAN.md)|已将问题成立性验证置于训练排错之前，旧版保留|
|[FINAL_VALIDATION.json](FINAL_VALIDATION.json)|当前文件链接、审计状态、协议哈希与任务容器核查|
|[ARTIFACT_SHA256.json](ARTIFACT_SHA256.json)|本轮文件哈希，排除本哈希清单自身|
|[STATUS.json](STATUS.json)|本轮完成状态与仍待进行的研究工作|

所有匹配均是固定诊断规则下的目标帧匹配，未匹配组件不等于逐一核实的误报；没有将其改称论文Pd/Fa。补充两段已用于开发，不能当作新独立测试。
