# Gaussian/SoftIoU 近似监督控制的固定评估

本目录只负责 CPU 评分、汇总与表级复核。只比较 `release`、`lambda0_top1`、`gaussian_softiou_approx`；不继续 BC-TPro，不新增模型或搜索阈值。新配置是原 Plus 上的近似标签监督控制，不能称为作者原生标签或完整训练配方复现。

## 输入接口

新权重默认位于 `training/gaussian_softiou_approx/run/final_model.pth`，同目录有训练 `RUN.json`。推理缓存默认为：

```
predictions/gaussian_softiou_approx/{DJI_0026_2,DJI_0067_2}/
  windows/window_{000,024,036,060,072,096,108,120}.npz
  official_fused.npz
  COMPLETE.json
```

每个窗口保存 FP32 有限概率图 `probabilities`，形状 `(40,512,640)`，以及 `start`、`end=start+40`。O 使用 `[0,36,72,108,120]`，S 使用 `[0,24,60,96,120]`，分别做逐像素概率 max；O 的结果必须与 `official_fused.npz` 完全相同。八个缓存窗口只是两安排的并集，每个安排仍只有五窗。

评分导入旧 `top1_background_control_20261011/evaluation_code/evaluate.py`，仅改新条件输入和输出路径。冻结阈值 0.5、坐标、预处理顺序、连通组件峰值点和一对一匹配。所有有效标注参与匹配，随后读取 5,200 个严格小目标帧的 M。U 是全部有效标注匹配后的剩余组件；A 是有效框外前景像素。两者都不是官方 Fa，U 也不等于人工确认的误检。

评分前验证：

- RUN 与推理 COMPLETE 的 `condition/variant` 均为 `gaussian_softiou_approx`，训练完成且固定第 512 步；最终权重路径与实际 SHA 一致。
- RUN 的发布权重、原 Plus 源码、计划及训练选择 hash 与旧 top-1% RUN 一致；BN 初始、最终 hash 不变。
- 推理与训练的 `label_source_sha256`、`label_config` 一致；实际原模型源码 hash 正确。
- CPU 使用 `torch.load(..., weights_only=True)` 核对最终权重内部的 condition、variant、step、初始权重、计划及标签配置，不加载 Python 模型 pickle。
- 推理归一化为 `[85.0799,47.4845]`，原 Plus 为 70,913 参数，core128、halo16；推理不访问标签。三个固定块的直接/分块检查须存在，0.5 二值差异均为零。
- 原评分源码及对应标注/匹配 helper 与旧评分凭据一致；原 scorer 进一步验证原图、窗口、融合文件 hash。

不要求 BC 源码、gamma、BC model_config 或 halo64。元数据缺失或条件不符时失败，不自动降级为无检查评分。`--predictions-root` 可指定包含两个片段目录的绝对路径，最终权重身份仍必须属于上述固定训练目录。

## 运行

在既有 `deeppro-official-eval:20261009` 容器内运行，挂载项目绝对路径，不传 GPU：

```
python evaluation_code/evaluate_alignment.py --baseline-check
python evaluation_code/validate_tables.py --directory evaluation_preflight
python evaluation_code/evaluate_alignment.py --condition gaussian_softiou_approx --sequence DJI_0026_2
python evaluation_code/evaluate_alignment.py --condition gaussian_softiou_approx --sequence DJI_0067_2
python evaluation_code/evaluate_alignment.py --summary
python evaluation_code/validate_tables.py --directory evaluation
```

两个片段评分可并行。省略 `--sequence` 则顺序评分两个片段。评分和主汇总拒绝覆盖已完成结果。`--baseline-check` 仅写 `evaluation_preflight/`，复用旧 release/top1 缓存且明确记录 `new_models_evaluated=[]`；它不是新监督方法的成绩。

## 输出与真实集合差分

完整三配置汇总写 `evaluation/`：

- `SUMMARY.csv`（6 行）和 `SEQUENCE_SUMMARY.csv`（12 行）：M/U/A、同安排相对 release 与 top1 的新增/丢失、面积成本、固定预算，以及所有冻结集合的恢复/保持和相对 top1 的真实集合新增/丢失。
- `OBJECT_COMPARISON.csv`（10,400 行）：每目标帧×安排，保留片段/轨迹/帧/标注 ID、各配置匹配状态、冻结分层、相对参照的逐对象 gain/loss，支持定位每一个差异对象。
- `COHORT_TRANSITIONS.csv`（180 行）：每配置×安排×总体/片段×冻结分组，相对 **同安排 top1** 分为共同成功、仅新配置成功（gain）、仅 top1 成功（loss）、共同失败；同时给每状态涉及帧数、轨迹数、片段数。它不把 `matched_new - matched_top1` 当成全部新增。
- `TRACK_SUMMARY.csv`（780 行）和 `RECOVERY_CONCENTRATION.csv`：逐轨迹新增/丢失、冻结低响应恢复，保留集中度。
- `WINDOW_DEPENDENCE.csv`（81 行）：在固定 O/S 下分为均成功、仅 O、仅 S、均失败；包括 286 个旧窗口混合成功对象、旧 O-only54、S-only74、额外窗口可恢复104、旧双侧成功646，以及低响应组。不是新模型全部 11 窗的稳定性实验。
- `FRAME_COUNTS.csv`（1,920 行）：全图 U/A 与面积背景成本；`SUMMARY_COMPLETE.json`：输入及脚本 hash、范围和分层数量。
- `TABLE_VALIDATION.json`：独立 stdlib 从逐对象表复算 M/G/L、低响应集合迁移、O/S 联合状态，并与帧/轨迹汇总交叉核对、验证冻结输入 hash。此项不冒充原始分数图全量独立重新评分。

分层直接复用 `low_response_information_20261011/COHORTS.csv`：低响应4,342，极低4,170，旧双侧成功646，额外窗口可恢复104，新加入窗口依赖73，旧 O-only54、S-only74、窗口成功/失败混合286。分层互有交叠，不声称构成全量的互斥划分。

旧基线缓存及分层 hash 与上一已完成 BC 评估的冻结凭据逐项一致；本轮不会改写旧结果。预期 release O/S 的 M/U/A 为 `700/4015/38368`、`720/3967/38641`，top1 为 `877/8339/48354`、`913/8330/48641`。主结论继续受 U≤4,015 约束，且要报告相对同安排基线的各项实际成本，不能只因满足较宽预算就声称相同成本。
