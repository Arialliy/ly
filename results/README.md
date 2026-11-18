# 结果与外部参照

| 内容 | 记录 |
|---|---|
| 自主模型 S0–S3 的数据、几何与模型验收 | [阶段报告目录](../experiments/irstd_independent_v1/reports/) |
| S4 四模型的当前训练和评价进展 | [子运行里程碑](../experiments/irstd_independent_v1/reports/s4_runs/s4_24k_seed46_q256v81_20261118_02/monitor/MILESTONES.md) · [父运行记录](../experiments/irstd_independent_v1/reports/s4_runs/s4_24k_seed46_20261118_01/) |
| 早期运动联合表示对照、检出身份和负结果 | [2×2 对照结果](../experiments/motion_joint_factorial/RUN.md) · [诊断](../experiments/motion_joint_followup/DIAGNOSIS.md) · [审查](../experiments/motion_joint_design_audit/EXPERIMENT_AUDIT.md) |
| DeepPro-Plus 作者发布权重与原复现 | [发布权重](references/deeppro_plus_release.pth) · [参考复现](../archive/README.md) |
| 历史 VT 片段选择和原训练计划 | [冻结数据选择](references/DATA_SELECTION.json) · [训练计划](references/TRAIN_PLAN.json) |

S4 的主要工作点采用完整开发集的峰级虚警预算。历史测试的冻结阈值迁移、描述性测试 FROC 和作者原生像素虚警率分别报告。[S4 研究协议](../1/S4_research_protocol.md)
