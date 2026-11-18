# 第一阶段校准推理与评分执行记录

日期：2026-10-10。状态：**全部10次推理、10次CPU评分完成，20个真实Docker句柄均复核为 exited(0)，队列主进程也退出0。** 仅执行校准集；没有进行阈值选择，没有启动或读取确认集推理结果。

校准序列固定来自 `DATA_SELECTION.json`：`DJI_0026_2`、`DJI_0067_2`。变体为 `official`、`paired_mean`、`paired_max`、`ohem`、`random_pair_max`。

## 执行前检查

4个训练容器逐一确认 exited(0)，各 `RUN.json` 为 complete、final_step=512、BN 初末hash均为 `51832be8d623f18e372232ddefdf4214ad67b3e45f795fbaa2ea9daa7fd0ecab`。4个final_model实际SHA逐一匹配各RUN记录。

使用现有 `deeppro-official-eval:20261009` 镜像，镜像ID与此前环境审计一致；每个容器均挂载官方仓库为只读。首个空GPU优先官方参考。初始映射为GPU0官方、GPU1 paired_mean、GPU2 paired_max、GPU3 ohem；paired_max两序列完成后GPU2运行random_pair_max。每个GPU同一时刻仅一个本队列推理进程。CPU评分容器不开放GPU，最多并行4个。

## 完成检查

- 每个变体首校准序列执行3处直接/分块推理比较，共15处检查；最大概率误差1.0431e-5，脚本规定的logit断言均通过。
- 每序列保存11个真实40帧诊断窗口，但原融合仅用起点 `[0,36,72,108,120]`。评分器逐像素重新构造这5窗maxfusion并验证完全相同。
- 每序列按固定23阈值完成 `official_fused_curves.csv`、`official_fused_matches.npz` 和完成元数据。
- 逐个复核全部110个诊断窗口hash、10个融合hash、原图hash、checkpoint hash、算法脚本hash、标注hash及20个容器的实际镜像与只读mount。
- 完成后的推理与评分没有重跑；没有出现失败、替代或重启任务。

证明文件：[CALIBRATION_EXECUTION_VERIFIED.json](CALIBRATION_EXECUTION_VERIFIED.json)。完整队列状态：[QUEUE.json](QUEUE.json)；事件流：[EVENTS.jsonl](EVENTS.jsonl)。各 `variant/sequence/{infer,score}/` 目录保存 LAUNCH.json、容器ID、LAST_INSPECT.json、stdout/stderr、EXIT.json 和 VERIFIED.json。退出容器保留供独立复核。

算法使用主任务冻结的 `pilot/infer_pilot.py` / `pilot/score_pilot.py`，没有修改算法。调度器及检查器分别是 [calibration_queue.py](calibration_queue.py)、[verify_calibration_jobs.py](verify_calibration_jobs.py)。队列已有完整结果，调度器会拒绝覆盖；需要复核时只执行检查器：

```bash
python3 /home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/temporal_diagnosis_20261010/stage1_goal/runtime/evaluation_jobs/verify_calibration_jobs.py
```

推理任务内计时总和373.64秒，评分任务计时总和463.19秒；任务并行执行，这两个和不是总墙钟时间，也不是方法速度比较。

下一步由主任务运行 `summarize_pilot.py` 冻结校准阈值，再显式调度确认集。本记录只证明执行完整性，不解释方法优劣，不把诊断点匹配称作官方Pd/Fa。
