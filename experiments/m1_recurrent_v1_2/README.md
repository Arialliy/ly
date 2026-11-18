# M1-v1.2

此目录在原 M1 上接入用户提供的 `BoundedStateUpdate`，验证新增递归历史能否产生 Current 控制没有的实际弱目标检出。只修改状态更新；研究依据、实验范围及固定终点见 [PROTOCOL.md](PROTOCOL.md)。旧实验文件保持冻结。

本轮已完成：原包25项测试、真实CUDA集成检查、双臂各128步训练、同权重干预和完整评价均已结束。两类历史比较均无新增低响应恢复；逐对象匹配及连续漏检没有历史增量，背景减少伴随其他有效目标丢失。实际数值和作用链见 [RESULTS.md](RESULTS.md)，不据本轮结果追加训练或M2。

关键代码：

- [reference/state_update.py](reference/M1_v12_state_update_reference/state_update.py)：用户参考包原样导入的核心，`M=(1-g)R+gA`，2017 参数。
- [code/revision.py](code/revision.py)：保留原 RFR PDA／TSFM 模块对象和参数名，在候选状态之后调用参考核心。
- [code/runtime.py](code/runtime.py)：从 v1 History2048 严格加载旧权重，安装同一初值的门控，显式迁移旧 Adam 状态。
- [code/train.py](code/train.py)：新版 Current／History 各128步、0/32/64/128快照和训练侧记录。
- [code/trace.py](code/trace.py)：区分对齐状态、TSFM候选与更新后状态；追踪接入和logits。
- [code/evaluate.py](code/evaluate.py)：三条正式路径、实际匹配及两类历史恢复／损失。

[MatAnyone 官方来源核验](upstream/MATANYONE_REFERENCE_AUDIT.md)明确记录了借鉴形式和任务适配的区别。原包与官方代码均保留来源及哈希，没有用简化卷积替代原生 PDA/DCN。

模型入口为本目录的 `runtime.load_initial('V12History')` 或完成训练后的 `runtime.load_final('V12History')`。安装后仍调用原 `FrozenDeepProWithAdapter`：训练用 `model(x)`，完整开发帧推理用 `model.forward_full_spatial(x)`。`model.mode='current_only'` 只移除 M1 新增递归输入，不移除原前端或 TPro 的时间信息。不要直接用旧 v1 的构造器加载含新门控的 checkpoint。

任务由 [analysis/run_pair.py](analysis/run_pair.py) 驱动，在占用检查后启动既有 Docker 环境。运行状态见 [runtime/WORKFLOW.json](runtime/WORKFLOW.json)，固定协议见 [PROTOCOL_FREEZE.json](PROTOCOL_FREEZE.json)。该 worker 不覆盖已有运行；正在执行时不要重复启动。

原包测试结果、严格初始化迁移及真实 CUDA 集成证据位于 [checks](checks/)。工程检查不等于检测收益。实际结果以本目录 [RESULTS.md](RESULTS.md) 和可核验 CSV 为准；中间快照仅用于观察，没有用于挑选最佳终点。
