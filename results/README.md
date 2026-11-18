# S/T-LFP冻结结果与其他实验入口

前一轮S-LFP／T-LFP完整实验已放在本目录。最新H/HL实验在[独立实验目录](../experiments/head_lfp_joint_20261012/README.md)，全部历史结果在[archive](../archive/README.md)。先读[原始报告](tlfp_pilot_20261012/REPORT.md)，或打开[可视化浏览页](../visualizations/index.html)。

| 内容 | 路径 |
|---|---|
| 原始报告与结论 | [REPORT.md](tlfp_pilot_20261012/REPORT.md)、[DECISION.json](tlfp_pilot_20261012/DECISION.json) |
| 检出、恢复、丢失与背景汇总 | [SUMMARY.csv](tlfp_pilot_20261012/evaluation/SUMMARY.csv) |
| 逐目标帧完整明细 | [OBJECT_COMPARISON.csv](tlfp_pilot_20261012/evaluation/OBJECT_COMPARISON.csv) |
| 逐轨迹及连续漏检 | [TRACK_SUMMARY.csv](tlfp_pilot_20261012/evaluation/TRACK_SUMMARY.csv)、[CONTINUOUS_MISS_TRACKS.csv](tlfp_pilot_20261012/evaluation/CONTINUOUS_MISS_TRACKS.csv) |
| 机制开关与保持核验 | [MECHANISM_SUMMARY.csv](tlfp_pilot_20261012/evaluation/MECHANISM_SUMMARY.csv)、[BYPASS_VALIDATION.json](tlfp_pilot_20261012/evaluation/BYPASS_VALIDATION.json) |
| S-LFP终点权重 | [training/S/run/final_model.pth](tlfp_pilot_20261012/training/S/run/final_model.pth) |
| T-LFP终点权重 | [training/T/run/final_model.pth](tlfp_pilot_20261012/training/T/run/final_model.pth) |
| 中间检查点与训练日志 | [training/](tlfp_pilot_20261012/training/) |
| 完整预测缓存 | [predictions/](tlfp_pilot_20261012/predictions/) |
| 测试、原执行记录与协议 | [checks/](tlfp_pilot_20261012/checks/)、[runtime/](tlfp_pilot_20261012/runtime/)、[PROTOCOL.md](tlfp_pilot_20261012/PROTOCOL.md) |
| 发布权重 | [deeppro_plus_release.pth](references/deeppro_plus_release.pth) |
| 冻结分层、划分、训练计划与发布评分缓存 | [references/](references/) |

完整实验快照247文件，4,406,239,762字节；另有12项参照资产。已按原清单逐项核对大小与SHA256，原实验保持不变。这里的结果包括失败结论和所有控制，并非只收录有利结果。

固定阈值为0.5，两个开发片段包含5,200个严格小目标帧、130条轨迹；M/U是诊断匹配和未匹配组件，不是官方Pd/Fa。T-LFP的严格小目标匹配集合与发布权重相同，低响应集合恢复为0，不能把小幅背景变化写成目标恢复。具体边界以原报告为准。

这些冻结文件采用硬链接收录以避免重复占用磁盘，没有符号链接依赖；原路径删除后本目录文件仍存在。硬链接共享内容，请把后续分析写成新文件，不原地改写这些历史快照。完整训练输入已在`../data/stage1/`；全部历史路线在`../archive/`；完整公开数据在`../data/public/`。新H/HL实验在`../experiments/head_lfp_joint_20261012/`。旧项目根目录已整体迁入包内。
