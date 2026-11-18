# Gaussian 标签 CPU 独立预检

状态：**PASS**；2026-10-12；1071 项检查，CPU 72.36 秒。没有读取图像或开发预测，没有 GPU 工作。

[测试脚本](/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/temporal_diagnosis_20261010/supervision_alignment_control_20261012/code/test_gaussian_labels.py) 以 Python `math.exp` 标量计算和像素中心的直接不等式比较建立独立 oracle，没有调用被测 raster helper。测试偶数/奇数/浮点框、裁剪后中心不变、uint8 量化与非矩形支撑、重叠 max、ignore/crowd 优先、partial/core9、亚像素框空支撑不强塞峰。所有 512 步都核对最终标签与有效域的逐像素一致性。

| 冻结训练计划统计 | 数量 |
|---|---:|
| 更新步骤 | 512 |
| 窗口暴露 | 2,048 |
| 帧暴露 | 81,920 |
| 正对象帧暴露 | 148,902 |
| 量化后无正支撑对象暴露 | 0 |
| 排除对象及 2px 保护优先后全部支撑被清除 | 66 |
| 整窗无正标签 | 385 |
| 四窗均无正标签的 batch | 16 |

这些是固定采样计划内的重复暴露计数，不是独立对象、轨迹或帧总数。原 Soft-IoU 在分母非零的纯负窗产生 loss=1、该窗损失梯度为 0；不等于带动量/权重衰减的优化器一定不更新。

量化支撑/框像素比例：min=0.8333，Q25=0.88，中位数=0.90，Q75=0.9286，max=1。二值化后的监督主要是框内的稠密支撑，不再保留 Gaussian 中心到边缘的连续权重。这也限制了将其称作“峰形状回归”的表述。

被测 `gaussian_labels.py` SHA-256：`a256403d9b3df7670b7e6466e42afd40bcf881b40673c991ba8a0116eeb088b5`。

512 步 target+valid 链 SHA-256：`b5c86998904cbe15a10bdc08d445e6871cfec254ed1b2676da53b15e03da5788`。每步的独立 target/valid hash、标注文件指纹、全部检查记录保存在 [LABEL_PREFLIGHT.json](/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/temporal_diagnosis_20261010/supervision_alignment_control_20261012/LABEL_PREFLIGHT.json)。本核验只证明代码实现了已冻结的近似规则，不证明该规则恢复了作者未公开的生成参数，更不证明检测收益。
