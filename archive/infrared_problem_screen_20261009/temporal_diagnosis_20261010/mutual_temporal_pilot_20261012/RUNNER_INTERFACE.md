# 固定运行接口

本轮只支持 `E0`、`E1`、`E2` 三组正式训练。`train_mutual.py` 的共同参数为 `--source`、`--checkpoint`（发布权重）、`--dataset-root`、`--selection`、`--plan`、`--output`、`--variant E0|E1|E2`、`--radius 2`；固定 `--steps 512 --seed 46 --lr 0.0001`。原参数全部可学，BN运行统计冻结，所有新增参数与原参数进入同一个fresh Adam。

输出放在 `training/<variant>/run/`，包含 `RUN.json`、`INITIAL_STATE.json`、逐步 `TRAIN.jsonl`、`PARAMETER_DIFF.json`、`snapshot_000.pth`、`snapshot_128.pth`、`snapshot_256.pth`、`snapshot_384.pth`、`snapshot_512.pth`。`final_model.pth` 是512快照的硬链接，两者序列化字节完全相同。快照含模型、Adam状态和来源信息；不实现自动续训，也不根据开发指标选检查点。

`infer_mutual.py` 接口为 `--source`、`--reference-checkpoint`（发布权重）、`--checkpoint`（对应快照）、`--dataset-root`、`--sequence`、`--output`、`--variant E0|E1|E2`，可加 `--verify-tiling`。默认 `--step 512 --diagnostic none`。中间节点显式传 `--step 128|256|384`，输出目录父级使用 `E0_step128` 等名称。主终点使用 `E0`、`E1`、`E2`。

机制诊断仅接受E2最终权重，分别传 `--diagnostic mass`、`--diagnostic current`，输出父级为 `E2_mass`、`E2_current`。不产生新训练。所有条件core128、halo20、同一8起点缓存，O/S各使用固定五窗；sigmoid与概率max融合顺序不变。

最终E2正常推理加 `--provenance` 可在冻结标签中心采集只读概率归属记录。Recorder只读layer1输出，不改变模型输出；单独记录遥测耗时。此条件峰值显存明确包括遥测，不能当作排除遥测后的纯模型峰值。预测概率文件沿用历史评分器API：`windows/window_<start:03d>.npz`、`official_fused.npz`、`COMPLETE.json`。

实现核查：`checks/TRAIN_LOSS_CHECK.json`为13项CPU损失核查；`checks/NETWORK_PREFLIGHT.json`为24项真实模型/GPU核查，包括全batch前反向、零投影初始化等价、E1/E2初始分支一致、梯度启动、非零修正下全模型分块一致性及原生40×512×640推理资源。实现探针的临时模型全部丢弃，没有产生正式训练检查点。
