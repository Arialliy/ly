# M1-v1.1：仅修订状态更新的128步配对短实验

依据已完成同权重诊断，History和current_only全部有效目标匹配及连续漏检统计相同。low4342对象上，PDA后History/current相对差异均值0.43179，TSFM后0.014015；两阶段状态RMS仍在约0.05量级。原头将约7.70e-6的接入差映为约5.31e-4 logit差，未表现为不接收差异。优先检验TSFM更新端能否保留更多已对齐状态；这不证明PDA差异已有目标价值，也不排除后续投影限制。

唯一结构改动，t>0时令A=PDA(M_prev,X)，原M=TSFM(A,X)改为M=A+TSFM(A,X)。首帧M0=X0，PDA、X投影、Wout、原DeepPro-Plus和BN、通道数、窗口重置、F监督及全部后处理保持。继承现有TSFM只增加该求和，参数名称/顺序/形状/数量不变，未增加新门控、缩放、ReLU或M2。对齐输出有符号，因此新状态分布可不同；不另作变换掩盖这一干预。

两臂Continuation/Revision都加载同一History step2048检查点（SHA eda063496757f368d7442a7fac99a64f7b33ef0414ef99d006716d77a6dcb725），恢复其全部Adam状态及RNG，在相同已有F缓存条目0..127上再训练128步，固定终点2176，不挑检查点。lr1e-5、weight_decay1e-4等优化器设置和F loss完全复用。原网络始终冻结；没有另训Current，没有新增数据。新曝光的影响由Continuation控制。这里只是短期机制修订实验，不代替正式同条件独立训练的Current/History比较。

终点每臂都在自己的同一权重下计算history/current_only，使用相同F/H与既有固定ROI，沿用两开发视频16个原40帧窗口、O/S、0.5和原匹配器。旧History终点和release直接复用缓存。报告RevisionHistory相对ContinuationHistory及每臂history相对同权重current_only的实际恢复、连续漏检、轨迹覆盖、U、面积及protect646；不会把总M提升单独归因为历史。诊断层面检查TSFM后的相对差异是否保留、是否进入logits并改善目标/固定背景间隔。

判定顺序：状态差更大只能说明该改动达成保留差异，不能证明恢复。若目标匹配/连续失败没有针对性改善或背景/原成功代价不合理，本轮不追加训练、不增加第二个改动，不启动M2。只有针对性恢复且自身同权重历史控制显示独立价值，才认为值得后续正式Current/History成对训练；本轮不执行长训。低响应背景差异本来大于目标，额外保留可能同样传播背景，必须以实测报告。

运行固定既有torch2.2/CUDA12.3镜像、FP32无AMP/TF32，GPU1/2分别执行两臂，不停止他人作业。原RFR算子梯度不保证位级确定性，不声称两臂训练位级可复现。旧release缓存跨runtime浮点尾差按已完成诊断记录处理，Off原路径及阈值/匹配不变。
