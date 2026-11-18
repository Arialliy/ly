# 第一阶段工件记录

| Timestamp | Skill | File | Stage | Description |
|---|---|---|---|---|
| 20261009_195557 | experiment-plan | STATUS.json | implementation | 用户已激活持久目标；数据、环境与候选核查进行中 |
| 2026-10-10 | experiment-plan | [GOAL.md](GOAL.md)、[EVIDENCE_AND_GAP.md](EVIDENCE_AND_GAP.md)、[DESIGN_DECISION.md](DESIGN_DECISION.md) | scope / design | 原完成条件保留；当前执行状态与设计时记录分开 |
| 2026-10-10 | experiment-plan | [DATA_SELECTION.json](DATA_SELECTION.json)、[data/DATA_INTEGRITY.json](data/DATA_INTEGRITY.json)、[data/VISUAL_SCENE_REVIEW.md](data/VISUAL_SCENE_REVIEW.md) | data | 8训练/2校准/4确认；原图及标签核查；前缀隔离的场景限制 |
| 2026-10-10 | experiment-plan | [PILOT_PROTOCOL.md](PILOT_PROTOCOL.md)、[PILOT_FREEZE.json](PILOT_FREEZE.json)、[EVALUATION_FREEZE.json](EVALUATION_FREEZE.json) | freeze | 数据、训练、阈值网格、判据及推理源码先行冻结 |
| 2026-10-10 | run-experiment | [local_branch_probe/REPORT.md](local_branch_probe/REPORT.md) | model probe | 冻结模型局部贡献干预的恢复/损失；不是新方法成绩 |
| 2026-10-10 | run-experiment | [runtime/TRAIN_LAUNCHES.json](runtime/TRAIN_LAUNCHES.json)、[TRAINING_AUDIT.md](TRAINING_AUDIT.md) | train | 四组512步与真实终态、初始化、BN和权重变化核查 |
| 2026-10-10 | run-experiment | [summary/CALIBRATION.json](summary/CALIBRATION.json)、[summary/CALIBRATION_CURVES.csv](summary/CALIBRATION_CURVES.csv) | calibration | 四个训练配置全部无可行点；主参照官方0.5；未改阈值网格 |
| 2026-10-10 | run-experiment | [CALIBRATION_FAILURE_AUDIT.md](CALIBRATION_FAILURE_AUDIT.md)、[TRAINING_FAILURE_DIAGNOSIS.md](TRAINING_FAILURE_DIAGNOSIS.md) | audit / diagnosis | 全量独立组件重算；观察与根因假说分开 |
| 2026-10-10 | run-experiment | [summary/development/DEVELOPMENT.json](summary/development/DEVELOPMENT.json)、[DEVELOPMENT_AUDIT.md](DEVELOPMENT_AUDIT.md) | development | 预定1417/97/24/42/502/778层；raw0.5、非等预算；中心/轨迹边界明确 |
| 2026-10-10 | run-experiment | [runtime/FINAL_EXECUTION_STATUS.md](runtime/FINAL_EXECUTION_STATUS.md) | terminal audit | 官方源码/权重未变、34正式任务全退出0、确认无模型输出、GPU无本任务作业 |
| 2026-10-10 | experiment-plan / run-experiment | [PILOT_RESULTS.md](PILOT_RESULTS.md)、[IDEA.md](IDEA.md)、[STAGE1_DECISION.md](STAGE1_DECISION.md) | decision | 当前弱框实现否定；一般机制未决；提出可检验的下一阶段条件 |

当前有效开发汇总在 `summary/development/`。`summary/development_v1_20261010_0426/`保留补中心误差字段前的版本，关键计数未改变。`*_before_*.md/json`、`GOAL_design_20261010.md`等文件为历史版本，不作为当前决定。确认集没有预测或评分文件是校准失败后的预定停止结果，不是运行产物遗漏。
