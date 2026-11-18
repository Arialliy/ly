# 旧开发片段固定阈值诊断评分执行记录

日期：2026-10-10。**2个真实推理容器及8个评分容器全部退出0；8组融合曲线和88个窗口评分完整，最终核验PASS。** 本轮不启动确认集，不选择新阈值，不把raw0.5诊断当作已校准操作点。

范围：`paired_mean` / `paired_max` 两个512步checkpoint，各在 `DJI_0028_2`、`DJI_0028_4`、`DJI_0032_5`、`DJI_0047_3` 四个已有开发片段。

执行前重读根任务记录的两个Docker ID，实际复核均为exited(0)，核验8组预测的所有88个窗口、8个融合图、raw与checkpoint hash；记录于 [INFERENCE_VERIFIED.json](INFERENCE_VERIFIED.json)。没有重新推理。

每个评分容器顺序调用未修改的 `pilot/score_pilot.py` 两次：先默认23阈值的官方5窗maxfusion，再 `--context-threshold 0.5` 对11个真实40帧窗口评分。全程最多4个CPU容器并行，均无GPU DeviceRequests；所有输出存放于 `stage1_goal/diagnostic_evaluation/<variant>/<sequence>/`。完成产物没有覆盖或重跑。

最终 [DIAGNOSTIC_SCORING_VERIFIED.json](DIAGNOSTIC_SCORING_VERIFIED.json) 逐项核验：

- 2个原推理容器、8个评分容器的当前真实退出码均为0。
- 8组 `official_fused` 各23阈值，评分器已逐像素重构并验证5窗maxfusion。
- 88组 `window_*` 阈值恰为 `[0.5]`；所有96组曲线、matches和完成记录均保存hash。
- 评分源码、标注、预测引用hash与完成记录一致。
- 冻结校准文件SHA保持 `2329b4d59dc1eaf2a0cfe71b743af7e8c7e5d682d175c3a88e03b1fae96dde8e`，没有因失败而增加阈值或读取确认集结果。

[QUEUE.json](QUEUE.json)、[EVENTS.jsonl](EVENTS.jsonl) 保存调度过程，各 `variant/sequence/` 子目录保存LAUNCH、Docker ID、原始日志、EXIT与VERIFIED。队列主进程也已退出0。

仅复核完成产物可运行：

```bash
python3 /home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/temporal_diagnosis_20261010/stage1_goal/runtime/diagnostic_score_jobs/verify_diagnostic_scores.py
```

这些结果用于已知失败群组的响应与误报诊断。校准失败的结论仍有效；本执行记录不代表通过确认测试或获得官方Pd/Fa提升。
