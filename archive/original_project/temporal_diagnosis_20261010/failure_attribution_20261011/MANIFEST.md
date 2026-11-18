# 故障归因交付索引

本轮使用原发布权重的已有缓存完成CPU诊断，不训练、不新增模型前向、不使用确认集结果。研究执行完成不等于正式定题或方法成功。

|交付|入口|可追溯原始表|
|---|---|---|
|综合结论|[REPORT.md](REPORT.md)|[交叉审计](audit/REPORT.md)|
|一：错误环节表|[stages/REPORT.md](stages/REPORT.md)|[5200对象台账](stages/ERROR_STAGE_TABLE.csv)、[逐窗访问](stages/WINDOW_ACCESS.csv)、[组件映射](stages/COMPONENT_MAPPING.csv)、[五例融合损失](stages/FUSION_LOSSES.csv)|
|二：主要失败条件表|[conditions/FAILURE_CONDITIONS.md](conditions/FAILURE_CONDITIONS.md)|[逐对象条件](conditions/OBJECT_CONDITIONS.csv)、[分层分母](conditions/CONDITION_STRATA.csv)、[稳定参照](conditions/STABLE_CONTROLS.csv)、[观察事件](conditions/OBSERVATION_EVENTS.csv)|
|三：路线选择记录|[ROUTE_DECISION.md](ROUTE_DECISION.md)|原因候选、最小验证、削弱条件、尚未成立的主张|
|单窗替换验证|[one_replacement/REPORT.md](one_replacement/REPORT.md)|[完整指标](one_replacement/METRICS.csv)、[对象四状态](one_replacement/OBJECTS.csv)、[中间融合反例](one_replacement/INTERMEDIATE_FUSION_TRACE.json)|
|官方指标与背景核查|[metrics/METRIC_AUDIT.md](metrics/METRIC_AUDIT.md)|[逐对象指标](metrics/OBJECT_METRICS.csv)、[背景总体](metrics/TOTALS.csv)、[原码与版本](metrics/SOURCE_AUDIT.json)|
|完整图册|[visuals/index.html](visuals/index.html)|[128增损查看记录](visuals/ROOT_EXCHANGE_REVIEW.md)、[16失败与4稳定查看记录](visuals/ALL_FAILURE_REVIEW.md)、[五例组件图与裁剪](stages/fusion_trace_figures/MANIFEST.json)|
|协议|[PROTOCOL.md](PROTOCOL.md)|[总冻结](PROTOCOL_FREEZE.json)、[逐次替换](ONE_REPLACEMENT_PROTOCOL.md)、[代理指标](metrics/ADAPTED_METRIC_PROTOCOL.md)|
|结论审计|[audit/CLAIM_REVIEW.md](audit/CLAIM_REVIEW.md)|[机器记录](audit/CLAIM_REVIEW.json)|
|交付验证|[FINAL_VALIDATION.json](FINAL_VALIDATION.json)|[verify_delivery.py](verify_delivery.py)、[STATUS.json](STATUS.json)、[ARTIFACT_SHA256.json](ARTIFACT_SHA256.json)|

所有框、图像和评分均来自已用开发材料。模型原因、真实可见时长、遮挡与相机运动未被本轮定量确认。旧problem_validity与stage1_goal文件保持不变；新的发现和路线在本目录版本化记录。
