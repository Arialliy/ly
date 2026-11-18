# M1-v1.1 实现独立复核

静态检查结论：未发现破坏本轮单因素配对的实质缺陷。运行中的 code、冻结协议和旧实验均未修改，未重复前向测试。

- `runtime.load_initial` 两臂均读取 SHA 为 `eda063496757f368d7442a7fac99a64f7b33ef0414ef99d006716d77a6dcb725` 的 History-2048；Revision 只替换 TSFM 子类，随后检查整个模型 state hash 不变。
- `train` 先核对与原 History 相同的 53 个可训练参数名称及顺序，再恢复完整 Adam 状态；起点所有 step=2048。模块构造后恢复检查点 Python、NumPy、PyTorch、CUDA RNG。没有重置优化器或改变初始化权重。
- 两臂均执行缓存条目 0..127 各一次，循环恰好 128 次，日志 step=2049..2176；损失、学习率和缓存清单沿用 v1。每步要求所有可训练参数梯度存在且有限，结尾要求全部 Adam step=2176。CUDA 算子仍采用原有 warn-only 确定性设置，不构成位级训练复现承诺。
- `load_final` 先按条件构造原/修订模块，再 strict 加载各自 2176 权重并验证最终 state hash；不存在把 Revision 权重装回原 TSFM 算法的问题。
- 推理复用冻结 trace，实际调用模型中的 `cell.pda` / `cell.tsfm`。History 与 current_only 在同一检查点下共享完整 F/H，t0 都为 X0；只有 History 递归使用上一状态。Off 完全绕过 M1 残差，将同一个 H 输入原冻结头。每个窗口都重置状态，M1 使用全空间，原前端仍为 core128/halo16。
- 两开发视频使用相同的 16 个合法窗口、O/S 安排、固定 ROI、CUDA sigmoid、阈值和原匹配器。模型不接收标注。Off 逐窗继续执行已登记的跨 runtime release 参照检查。

解释边界：Revision 的 CSV 字段 `after_tsfm` 对应 **A+TSFM(A,X) 后的新状态**，不是裸 TSFM 输出。比较该字段可以判断新更新式保留的模式差异，不能据此声称原 TSFM 注意力本身已学会保留历史。该修订同时保留有符号 PDA 输出，背景传播及状态分布变化仍由实际结果判定。

最终检查点只读核验已完成，`TRAINING_ARTIFACT_CHECK.json` 状态 **PASS**。两臂实际 TRAIN 均为 128 行，连续 step2049..2176、plan_index0..127；输入、标签、辅助输入三类哈希逐行相同，曝光摘要均为 `b7cb235bd98cb01e9d39b425b5d05b512f632031e895b49e339fd5222b04aac9`。两臂各 53 个 Adam 状态全部 step=2176，参数组保持原值，moments 形状对应原有参数顺序；所有模型及 moments 张量有限。原基线 100 个权重/缓冲张量与共同起点逐项完全相同，53 个可训练张量均发生更新。新 code 和记录的旧依赖哈希均与冻结值相符。

核验只运行一次 CPU 张量读取，没有构建模型、执行前向、优化器更新或占用 GPU。这证明本轮按登记的配对方案执行，不改变本轮效果结论，也不把实现通过解释为时序恢复有效。
