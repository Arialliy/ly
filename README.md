# 红外小目标视频检测

当前研究检验一个问题：给定同一段红外视频和相同的虚警预算，保留时刻与运动假设之间的对应关系，能否提高中心帧弱目标的检出率？自主模型接收连续 17 帧，预测第 9 帧的原生分辨率目标图。S4 比较 M0、Msep、Mraw 和 Mint，分别对应固定位置聚合、可加响应、完整联合响应和显式交互残差。

## 研究与实现

| 想了解什么 | 入口 |
|---|---|
| 运动假设的由来、已有工作与创新边界 | [原始研究方案](experiments/correspondence_premise_20261118_retry1/红外小目标检测_运动假设演化_研究方案.md) · [相关工作补充](<experiments/correspondence_premise_20261118_retry1/红外小目标检测_运动假设演化_研究方案 copy.md>) · [运动假设响应示意](visualizations/运动假设响应_理想示意.png) |
| 自主模型的问题定义、数据流与分阶段验收 | [自主模型方案](红外小目标视频检测_自主模型与分阶段实现方案_v1.md) · [S4 研究协议](1/S4_research_protocol.md) · [设计稿配置](1/s4_protocol_proposal.yaml) |
| 实际执行的代码及阶段证据 | [自主模型工程](experiments/irstd_independent_v1/) · [S0 数据契约](experiments/irstd_independent_v1/reports/S0_data_contract.md) · [S1 采样几何](experiments/irstd_independent_v1/reports/S1_geometry_contract.md) · [S2 验收](experiments/irstd_independent_v1/reports/S2.json) · [S3 证据模型](experiments/irstd_independent_v1/reports/S3_evidence_model.md) |
| 之前在冻结检测器上的 2×2 对照及其解释 | [原始对照](experiments/motion_joint_factorial/PROTOCOL.md) · [漏检与读出诊断](experiments/motion_joint_followup/DIAGNOSIS.md) · [结构审查](experiments/motion_joint_design_audit/EXPERIMENT_AUDIT.md) |
| 公开数据与外部参照 | [数据说明](data/README.md) · [DeepPro-Plus 作者源码](sources/official_deeppro/README.md) · [复现资料](archive/README.md) · [发布权重及结果入口](results/README.md) |

## S4 正式运行

训练种子固定为 46。三个证据模型继承同一父运行各自的前 600 次更新，再按原裁块日程继续训练。计算分块从 q64/v8 改为 q256/v81，分别表示每批查询位置数和速度假设数。M0 从配对初值训练。每个模型的共同预算先定为 24,000 步，延长至 36,000 步取决于预设训练诊断。训练进度和研究结论分别记录。[父运行协议](experiments/irstd_independent_v1/reports/s4_runs/s4_24k_seed46_20261118_01/protocol.json) · [子运行协议](experiments/irstd_independent_v1/reports/s4_runs/s4_24k_seed46_q256v81_20261118_02/protocol.json) · [当前里程碑](experiments/irstd_independent_v1/reports/s4_runs/s4_24k_seed46_q256v81_20261118_02/monitor/MILESTONES.md)

主工作点在 672 个开发中心上按至多 1 个虚警/帧选择。另用冻结的开发阈值评价 1,680 个历史测试中心。作者原生像素虚警率另列。项目位置及本机复现环境见[执行约定](AGENTS.md)。

[目录清理与完整性记录](provenance/PROJECT_CLEANUP_20261118.json)
