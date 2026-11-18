# S2c 执行方案：恢复、168 帧预测与完整训练资源

**下一步：执行 S2c。** S2a、S2b 已正式通过；S2c 完成后汇总 S2，进入 S3 的核心证据结构。保持第400步 M0 为本阶段冻结模型。

本轮完成的是最新仓库审核和执行方案，未运行恢复更新、168帧预测或GPU测量，不能记为 `S2c PASS`。

## 1. 已核对的前置状态

审核仓库版本：`8d39562a507780b148b8153a6ecb1b872067890e`。

| 项目 | 核对结果 |
|---|---|
| S2a | `PASS → S2b`；7/7份代码摘要一致 |
| S2b | `PASS → S2c`；6/6份代码/配置摘要一致 |
| 固定12中心清单 | 摘要与归档一致 |
| 正式结果 | `S2b.json` 与运行目录 `result.json` 字节一致 |
| 更新日志 | 连续400条，loss和gradient有限，更新后有限位全部为真 |
| 独立端点复核 | 已归档 `PASS` |

第400步总损失0.40645247，较初始化下降60.51%；固定12个训练中心的主工作点为7/8个目标、4个虚警。第350步的8/8不替代预设终点。正帧Dice损失仍为0.90545，目标峰定位和像素掩膜拟合继续分别解释；这不增加S2c的性能门槛。[S2b记录][s2b]、[短训练报告][fit]

## 2. 冻结输入与输出位置

使用原项目中的：

```text
/home/ly/DeepProPlus_TLFP_Code/experiments/irstd_independent_v1/
  reports/s2b_runs/seed46_400_20261118_01/final.pt
```

报告绑定的 SHA-256：

```text
a960e500f7f00da69de273ade47ecc021de386afb7835a9957a1dd7b90cb37de
```

加载前核对文件摘要、`global_step=400`、22,921参数以及依赖绑定。新结果写入独立的 `reports/s2c_runs/<run_id>/`，保留原S2b权重、清单、日志和输出。

Git仓库提供了权重的路径和摘要，但没有提交 `final.pt`；当前核查环境也没有原始数据和该权重。真实S2c需要在原项目环境执行。本轮对归档的核对不等于已经加载并运行该权重。

## 3. S2c-1：真正的保存恢复检查

### 3.1 复用已有状态，补充通用恢复入口

当前保存内容已有 model、AdamW参数组及一二阶矩、Python/NumPy/Torch CPU/CUDA随机状态、样本顺序、游标、配置和依赖摘要。[保存实现][train]

存在两个明确的接口限制：

1. `_checkpoint()` 将 `stage=S2b`、`global_step=400`、`next_sample_index=400` 写死，不能直接用它保存临时401步状态。
2. 原 `schedule` 恰好400项，游标400已经到末尾，不能访问不存在的 `schedule[400]` 或悄悄回绕。

新增 `engine/checkpoints.py`，显式接收步数、顺序、游标、父权重摘要和审计分支标记。保持已归档的S2b保存代码不变。

### 3.2 先验证原第400步输出

加载第400步快照后，在原12个诊断中心上，比较原生FP32 logits与归档 `logits_400.npz`。原logits的摘要记录在 `endpoint_audit.json`。

原 `evaluate_snapshot()` 会向运行目录写 logits 与 FROC 文件，不能将原S2b运行目录传给它重新计算，否则会覆盖归档。新输出应进入S2c目录，或直接逐帧比较而不重写原文件。[诊断实现][diagnostics]

该检查验证模型输出恢复；它不能代替优化器和继续训练状态检查。

### 3.3 在新建401中断点上比较继续训练

按原哈希规则扩展排序：

```text
SHA256("46|epoch|zero_based_epoch|sample_id")
```

已独立验证扩展顺序的前400项与原清单逐项相同。仅供审计的下两项为：

| 审计逻辑步 | 样本 | 属性 |
|---|---|---|
| 401 | `Sequence83/00020` | 有目标 |
| 402 | `Sequence77/00046` | 空目标 |

两项写入S2c审计协议，不改原400项清单。[排序规则][subset]

执行顺序：

1. 从400步快照的副本恢复分支A，用401样本更新一次。
2. 将完整401状态保存到S2c临时检查点。
3. A不中断，使用402样本更新，保存本步loss、更新后模型、Adam状态、游标及输出。
4. 新进程建立分支B，加载401状态，使用同一个402样本更新。
5. 比较A/B的loss、模型参数、Adam一二阶矩及step、下一样本游标和输出。

额外计算为A分支2次、B分支1次，共3次恢复审计更新。它们不延长正式400步拟合，也不用于提高正式成绩。

在同一固定环境、相同确定性设置下，模型/优化器状态以逐项完全一致为目标；元数据、游标和随机状态必须精确一致。若浮点结果不一致，先定位执行设置或恢复问题，不事后放宽标准宣布通过。整图/分块对照使用另行预设的FP32容差。

结果准确表述为：**从保存的400步快照出发，在新建401中断点上的恢复可复现性**；不声称重新证明原400步历史训练全过程。

### 3.4 显式检查随机状态

当前M0没有dropout，顺序又固定；漏恢复随机状态时，一次更新也可能碰巧一致。因此单独比较Python、NumPy、Torch CPU和各可见CUDA设备的随机状态，或在不扰动分支状态的作用域内比较下一组随机数。

先构造并加载model与optimizer，再恢复随机状态，避免初始化消耗随机数。恢复原确定性设置：关闭TF32与cuDNN benchmark、开启确定性算法，AdamW保持 `foreach=False` 和原超参数。明确调用 `train()`/`eval()`；运行模式不由 `state_dict` 自动恢复。[训练设置][train]

## 4. S2c-2：固定400步权重，完成168中心整图预测

现有模型已经提供：

```python
prediction = model(clip)
prediction = model.predict_tiled(clip, core_hw=(64, 64))
```

两者共用 `_region_logits()`。新增薄的预测调度层，复用模型，不改网络。[模型实现][model]

### 4.1 固定集合

| 来源 | 划分 | 中心 | 原尺寸 | 归档中心标签统计 |
|---|---|---|---|---|
| Sequence1 | train | 9–92，共84帧 | 157×209 | 84帧各1个目标 |
| Sequence17 | train | 9–92，共84帧 | 209×302 | 84帧全部为空 |

合计168中心、84个GT分量、84个空目标中心、8,058,204个原生输出像素。来源和中心集合来自冻结manifest；标签统计来自已归档训练侧核验。[数据清单][manifest]、[训练侧统计][coverage]

该集合属于工程预测检查。`Sequence17/00078` 已用于S2b，邻近视频窗口也有重叠，不能称为独立泛化验证。

### 4.2 预测完整性要求

- 使用S0的 `inference_item()`，推理只打开图像；评价另行读取标签。
- 使用独立加载的第400步权重，不使用401/402恢复分支或资源测量更新后的参数。
- 进入 `eval()`，保存原生FP32 logits及逐帧ID、尺寸、摘要。
- 输出ID集合恰好为168项，无重复、无缺失；所有真实像素有有限输出。
- 记录实际core处理与覆盖计数，核对每个真实像素恰好写回一次；不能用“输出非零”代替覆盖检查。
- 用第400步权重比较整图与64×64 core分块结果，采用已预设 `atol=1e-5, rtol=1e-4`。同时记录固定阈值下峰结果是否变化，不能用改阈值掩盖数值敏感性。
- 所有FP统计包括空目标中心，padding不进入峰提取或真实像素区域。

### 4.3 冻结阈值结果与描述性曲线分开记录

第400步三个预算选中的原始logit阈值恰好相同：

```text
59.03249740600586
score_domain = native_fp32_logits
```

如报告阈值迁移，就原样应用这个阈值；168帧上的实际FP/frame可能超过1，应照实记录。另可提供描述性FROC，但不能在168帧上重选阈值后称作冻结阈值结果。[末步记录][snapshot]

保留原始logits排序，不先转FP32 sigmoid或8位热图；高logit经过sigmoid可能都舍入为1，破坏排序及同分规则。

`evaluate_snapshot()` 硬编码12帧、8个GT，不能直接用于168帧。新建一般化预测清单和固定阈值评价入口，复用已有峰提取和匹配规则。现有FROC实现对单目标/空帧的限制必须显式保留；本168帧集合满足这一范围，不据此宣称已验证多目标评价。[诊断实现][diagnostics]、[FROC实现][froc]

**S2c不增加“168帧检出率必须达到87.5%”的门槛。** 本项检查完整性和推理可重复性，目标检出情况作为工程诊断如实报告。

## 5. S2c-3：完整M0训练资源

固定使用训练来源最大面积样本：`Sequence32/00009`，原尺寸272×407，输入17帧、B=1、FP32。[数据清单][manifest]

在新的诊断副本上加载第400步模型与AdamW状态：

1. 做一次包含 `optimizer.step()` 的预热，确保优化器状态存在。
2. 清梯度、同步设备、重置峰值统计。
3. 测量完整的forward、现有detection_loss、backward、optimizer.step并同步计时。
4. 记录GPU、固定镜像、输入尺寸、FP32、allocated/reserved峰值与各阶段耗时。
5. 按预定 **8,192MiB allocated峰值预算**判断。

预热和测量产生的2次诊断更新单独记录，不保存为正式检测权重。S1的1,346MiB仅为独立采样器结果，不能代替本项。单次测量用于资源可执行性检查，不宣称已得到稳定吞吐量统计。

## 6. 最小新增文件及执行顺序

| 顺序 | 文件 | 责任 |
|---|---|---|
| 1 | `engine/checkpoints.py` | 显式步数、游标、配置、RNG的保存和恢复 |
| 1 | `checks/check_s2c_resume.py` | 终点输出与401/402恢复对照 |
| 2 | `engine/predict.py` | 只读图像，调度已有full/tiled接口，输出清单 |
| 2 | `checks/s2c_prediction.py` | 168帧范围、空间覆盖、输出一致性、固定阈值诊断 |
| 3 | `checks/s2c_resources.py` | 完整M0训练资源测量 |
| 汇总 | `checks/check_s2c.py` | 前置摘要、三个子项状态、最终报告 |

按顺序完成当前子项后再扩展下一项。没有必要重写 `VideoBaseline`、重新选择12中心、重跑400步或增加一轮M0调参。

S2c产物为 `reports/S2c.json`、说明文档、恢复对照记录、168中心输出清单、完整资源记录。通过后汇总 `reports/S2.json`，明确绑定S2a/S2b/S2c及第400步权重摘要，并设置：

```json
{
  "stage": "S2",
  "status": "PASS",
  "main_fit_optimizer_updates": 400,
  "next_allowed_stage": "S3"
}
```

实际报告还应单列S2a一次工程更新、S2c恢复分支3次更新和资源分支2次更新，避免混淆主拟合预算与检查计算量。

## 7. 进入S3后的第一件事

S2c通过后，开始实现核心证据分解与编码路径，保持对外 `[B,32,Hq,Wq]` 接口、共享FrameEncoder、SpatialDecoder和Head。所有学习模块按协议参与训练。

先核对采样证据的分解、有效域、中心帧排除和梯度，再接真实训练。最终“同信息的raw-joint读出”和“显式交互结构”的公平比较继续按原方案推进；S2c不承担创新性判断，也不新增研究性能门槛。

[s2b]: https://github.com/Arialliy/ly/blob/8d39562a507780b148b8153a6ecb1b872067890e/experiments/irstd_independent_v1/reports/S2b.json
[fit]: https://github.com/Arialliy/ly/blob/8d39562a507780b148b8153a6ecb1b872067890e/experiments/irstd_independent_v1/reports/S2b_short_fit.md
[train]: https://github.com/Arialliy/ly/blob/8d39562a507780b148b8153a6ecb1b872067890e/experiments/irstd_independent_v1/engine/s2b_train.py
[diagnostics]: https://github.com/Arialliy/ly/blob/8d39562a507780b148b8153a6ecb1b872067890e/experiments/irstd_independent_v1/evaluation/s2b_diagnostics.py
[subset]: https://github.com/Arialliy/ly/blob/8d39562a507780b148b8153a6ecb1b872067890e/experiments/irstd_independent_v1/data/s2b_subset.py
[model]: https://github.com/Arialliy/ly/blob/8d39562a507780b148b8153a6ecb1b872067890e/experiments/irstd_independent_v1/models/video_baseline.py
[manifest]: https://github.com/Arialliy/ly/blob/8d39562a507780b148b8153a6ecb1b872067890e/experiments/irstd_independent_v1/manifests/nudt_mirsdt_mask17.json
[coverage]: https://github.com/Arialliy/ly/blob/8d39562a507780b148b8153a6ecb1b872067890e/experiments/irstd_independent_v1/reports/S1_grid_coverage.json
[snapshot]: https://github.com/Arialliy/ly/blob/8d39562a507780b148b8153a6ecb1b872067890e/experiments/irstd_independent_v1/reports/s2b_runs/seed46_400_20261118_01/snapshot_400.json
[froc]: https://github.com/Arialliy/ly/blob/8d39562a507780b148b8153a6ecb1b872067890e/experiments/irstd_independent_v1/evaluation/s2b_froc.py
