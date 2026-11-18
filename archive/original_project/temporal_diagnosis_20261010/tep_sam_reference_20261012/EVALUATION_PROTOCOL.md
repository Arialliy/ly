# TEP-SAM 完整画面恢复对照：诊断评分协议

本文件固定项目诊断，不替代作者原生评价。作者仓库版本、权重哈希、原生运行检查及推理还原链由本阶段主协议另行冻结；这些前提未通过时不得把评分器可运行写成 TEP-SAM 结果已经可信。

## 对象、预测与阈值

- 只读既有开发片段 `DJI_0026_2`、`DJI_0067_2`，各 160 帧、原图 512×640。每帧使用完整画面；不按目标裁剪，不接触四个确认来源。
- 固定全部 5,200 个严格小目标帧、130 条轨迹、2 个片段。冻结低响应集合 4,342、极低响应子集 4,170、原 O/S 双侧成功保护集合 646，均读取既有 `COHORTS.csv`，不根据新模型结果重分组。
- TEP-SAM 每个当前帧只有一份最终概率图。单次预测不复制为 O/S；发布 DeepPro-Plus O/S 是两份不同的比较对象。辅助头若为检查点内已训练参数且同一前向可得，则独立评分，不能补训。
- 项目诊断沿用 **probability >= 0.5**、8 邻接组件、组件最大概率像素中心及旧一对一框内峰值匹配。并列峰按原 row-major 规则；所有匹配有效标注参与匹配，随后单列冻结严格小目标。组件背景成本也使用全部匹配有效标注。
- 作者测试入口为 **probability > 0.5**。两种等号定义明确不同：项目旧规则不改；每帧和汇总记录概率恰等于 0.5 的像素数。作者原生结果保持独立。相同阈值不代表同等校准。
- 坐标还原、输出概率定义、时间访问和边界规则在运行主协议中冻结并核对后，评分器只接收原图分辨率概率，不自行插值、sigmoid、平滑、筛选或搜阈值。

## 冻结参照与评分器复用

复用 `top1_background_control_20261011/evaluation/release/` 中已审计的发布 O/S 匹配和组件缓存。其文件哈希对照 `bc_tpro_pilot_20261011/evaluation_preflight/SUMMARY_COMPLETE.json`，并核对原标注、原评分脚本及 `analysis/analyze.py` 的哈希。

新的单次图输入适配调用同一个 `stage1_goal/pilot/score_pilot.py::gt_data` 和 `analysis/analyze.py::match_frame`，不重新实现匹配。评分器本轮预检查把既有 release O 融合图当作单次图重读，对照全部 320 帧的原匹配、框峰、组件面积和背景计数；这只是评分器校验，不是一次 TEP-SAM 前向。

## 输入契约

每个片段的 `predictions/<sequence>/probabilities.npz`：

| key | 形状／含义 |
|---|---|
| `probabilities` | float32 `[160,512,640]`；有限且在 `[0,1]`，最终完整模型原图概率 |
| `frame_indices` | 整数 `[160]`，严格等于 `0..159` |
| `source_frames` | 整数 `[160]`，严格对应既有 annotation images 顺序与原文件名帧号 |

辅助头若可评价，另存同目录 `auxiliary_probabilities.npz`，包含同样的 `probabilities`、`frame_indices`、`source_frames`，形状与身份规则完全相同。此处不要求主输出 archive 再重复存储辅助图。

同目录 `COMPLETE.json` 必须包含：`status="COMPLETE"`、`sequence`、`raw_sha256`、`annotations_sha256`、`probabilities_sha256`、`protocol_path`、`protocol_sha256`、`checkpoint_sha256`、40 字符 `repository_commit`。若评价辅助头，还要求 `auxiliary_probabilities_sha256` 并声明和证明 `auxiliary_head_trained_parameters_loaded=true`。主协议和推理回执还应记录完整还原链、时间访问、参数量、前向次数、显存与计时；评分器不凭这些字段推定官方复现成功。

## 交付表

1. `OBJECT_COMPARISON.csv`：恰 5,200 行，每个严格小目标帧一行，保留发布 O/S 与 TEP 单次结果、框响应和峰坐标、冻结分层、相对 O/S 的新增和丢失；辅助头存在时添加独立列。
2. `SUMMARY.csv` 与 `SEQUENCE_SUMMARY.csv`：每个模型状态一行，TEP 不复制为两行。分别给出相对 O/S 的新增、丢失、净匹配、低响应恢复涉及帧／轨迹／片段，及 **原双侧成功对象中本模型仍检出数量**。
3. 背景列同时保留 U、框外前景像素 A、全部及未匹配组件面积总和／分位数／最大值。U 是未匹配组件诊断量，不称官方 Fa。只有完整 320 帧 pooled 行判断 `U<=4015`；同时报告相对 O、S 自身的各项背景变化，不把通过 pooled 预算等同于背景成本相同。
4. `TRACK_SUMMARY.csv`、`CONTINUOUS_MISS_SUMMARY.csv`、`RELEASE_MISS_SEGMENTS.csv`：复用已审计 `trajectory_tables.py` 的连续性判定，本地帧号与原始帧号均加 1 才连续；缺标、筛选空缺、分层排除均断开。TEP 每条轨迹每个分层一行，并列与 O/S 的最长漏检变化。报告范围是冻结观察区间，不称完整物理轨迹。
5. `RECOVERY_CONCENTRATION.csv`：恢复／丢失在片段和轨迹上的集中程度，不预设临时恢复帧数门槛。
6. `SET_COMPLEMENTARITY_DIAGNOSTIC_ONLY.csv`：成功集合交并只表征事后互补，不给其配 U、不当作可部署融合系统。
7. 辅助头可用时，`AUXILIARY_SUPPORT.csv` 交叉统计主输出与辅助输出匹配覆盖。即使同一对象被二者恢复，也不能单凭此表将收益归因于纯时间差分。

## 执行入口

`python evaluation_code/evaluate_reference.py --preflight`（仅 CPU 读取旧预测）。

预测与原生核查完成并批准为本阶段可信输入后：

`python evaluation_code/evaluate_reference.py --score tep_sam`

辅助头实际可评价才执行 `--score tep_auxiliary`。最后执行 `--summary --conditions tep_sam`，或包含辅助头的 `--summary --conditions tep_sam tep_auxiliary`。已完成输出拒绝覆盖。

固定研究边界：没有新检测器训练、没有阈值或模型配置搜索、没有将跨模型成绩解释为单机制因果证据。决定是否迁移须共同查看真实恢复、连续漏检、原成功保持、背景代价和覆盖范围；评分器不以未经登记的整数自动启动下一模块。
