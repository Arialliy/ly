# 本轮实际匹配评估

`infer_top1.py` 从原 `stage1_goal/pilot/infer_pilot.py` 复制，唯一差异为窗口起点从 11 个改为两种冻结安排的并集 8 个。预处理、模型加载、切块、CPU sigmoid 和原 O 融合保存不变。各受评安排仅使用自己的 5 个窗口。

`evaluate.py --condition release|lambda0_mean|lambda0_top1` 复用原 `score_pilot.gt_data` 和 `analysis/analyze.match_frame`。逐像素概率 max 后以 `>=0.5` 二值化；8 连通组件使用概率最大像素为峰，平局取 row-major 最早；峰在原半开 bbox 内才允许一对一最大基数匹配。排序和重匹配均不变。所有原有效 GT 参与匹配和背景统计，最终 M 只统计冻结的 5,200 个 strict_small 目标帧。不是论文 Pd/Fa。

背景 A 是原评价函数的 `positive_pixels_outside_boxes`，按原有效 GT 的像素中心裁剪区域排除，未新增 mask 或缓冲。U 是未匹配组件数，不是已确认误检。组件面积单位为像素；未匹配组件面积可能包含框内像素，故不等于 A。

每种权重、每个片段保存：

- `scores.npz`：allGT 的逐对象 matched、box_max、匹配峰及面积；strict_small、GT offsets、O/S coverage 和六种原 frame counts。
- `component_areas.npz`：O/S 的全部组件面积、unmatched 标志、逐帧 offsets；可重建所有面积统计。
- `FRAME_COUNTS.csv`：阈值 0.5 各帧组件数、U、A、面积与最大面积。
- `COMPLETE.json`：完整 GT identity 顺序、输入/脚本 hashes、旧缓存精确一致检查。

`evaluate.py --summary` 在三组缓存完整后生成：

- `SUMMARY.csv`：六行 pooled 结果。
- `SEQUENCE_SUMMARY.csv`：十二行逐片结果。
- `OBJECT_COMPARISON.csv`：10,400 行，一个原小目标帧 × O/S，三组实际结果并列，冻结 low/extreme 等标记不更新。
- `TRACK_SUMMARY.csv`：780 行，一个片段轨迹 × 三组权重 × O/S。
- `RECOVERY_CONCENTRATION.csv`：各组新增、丢失、low/extreme 恢复的帧/轨迹/片段数、top1/top5贡献及完整排序。无事件时数量为0、比例为空（NA）。

同安排发布权重是 G/L 主参照；额外记录原发布 O 和失控 lambda0_mean 的 G/L。`recorded_background_costs_not_increased` 仅检查记录的 U、A、未匹配组件总面积逐项不高于同安排发布权重。`auxiliary_old_S_joint_pass` 是历史附属条件，不替代训练修复、检出收益与保持取舍三项判断。

`evaluate_scoring_v1.py` 保留最先复算发布/mean 缓存的脚本快照；随后现用 `evaluate.py` 仅改汇总中的空事件 NA、背景三项比较和历史附属字段。每个 COMPLETE 保存其实际运行脚本 SHA256。
