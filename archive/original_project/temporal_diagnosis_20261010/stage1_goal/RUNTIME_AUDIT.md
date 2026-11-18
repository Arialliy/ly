# 第一阶段计算环境与训练接续审计

日期：2026-10-10（Asia/Taipei）。本轮实际执行了只读 CUDA witness、合成输入前反向计时和 CPU optimizer 恢复检查；**没有执行 optimizer.step，没有保存新模型，没有训练效果指标**。GPU 已释放。官方源码 git status 为空，官方 checkpoint SHA256 前后相同。

## 1. 结论与适用范围

- 可复用现有 `deeppro-official-eval:20261009` 镜像，无需安装依赖或重建环境。其 image ID 为 `sha256:eb956873870d737f2b68dff0724b8f6f56f778829d119b0938d29b7abeaa2f92`。
- 该镜像已实际完成官方 RGB-T 权重的严格载入、train-mode FP32 前反向，以及 CPU Adam 动量恢复检查。单张 RTX 5090 足够运行 batch 4、40 帧、128×128 patch 的本模型前反向。
- 应以官方 checkpoint 初始化、各组使用相同新 optimizer 的独立微调作为第一阶段公平比较。它是本地新微调实验，不能称精确接续作者原始随机训练轨迹。完整 optimizer 恢复技术上也可行，但会带入作者已有 5,192 步的动量，模型新增参数还需单独说明初始化。
- 本审计只确认计算和模型恢复可行。RGB-T 原生分割监督是否齐备、公开 MOT 框与原始 segmentation 的关系、训练/开发/确认划分，由数据审计另行确认。不能把框直接填满作为原论文像素监督再声称原配方复现。

## 2. 按文档原样复测 witness

在主机执行（独立复核者无需复制代码，无需进入正在运行的 `spectrum_5090`）：

```bash
STAGE1_GPU=0 bash /home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/temporal_diagnosis_20261010/stage1_goal/runtime/run_witness.sh
```

预期退出码 0，并包含：

```text
torch 2.7.0+cu128, numpy 2.2.5, scikit-image 0.25.2, scikit-learn 1.5.2
WITNESS_OK, seed 46, NVIDIA GeForce RTX 5090, shape [8, 8], sum 42.68129348754883
```

实际本轮输出保存在 [runtime/WITNESS.log](runtime/WITNESS.log)。该输出是 GPU 矩阵运算和必需模块导入 witness，不是检测模型精度检查。

脚本完整命令使用现有镜像，新建自动退出容器；`/home/ly:/home/ly` 挂载让当前 raw frames、已下载数据及新产物路径均可见；再以独立只读 mount 覆盖 `official_deeppro`，避免修改官方仓库和权重。`PYTHONDONTWRITEBYTECODE=1` 禁止在官方目录写 pyc。

已运行的 `spectrum_5090` 容器只挂载 `/home/csy`；其基础镜像是 `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel`。因此不能假定直接 `docker exec spectrum_5090` 就能读取本项目，或拥有完整已审计依赖。此处无需修改它。

复用环境的通用执行模板为 [runtime/run_isolated_python.sh](runtime/run_isolated_python.sh)：

```bash
STAGE1_GPU=1 bash /home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/temporal_diagnosis_20261010/stage1_goal/runtime/run_isolated_python.sh \
  /home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/temporal_diagnosis_20261010/stage1_goal/your_adapter.py \
  --your-arguments
```

`your_adapter.py` 是待由主任务实现的训练/推理入口占位名；模板自身不选择训练配置，也没有启动训练。所有实际产物应写到 `stage1_goal` 内。若后续数据放在 `/home/ly` 以外，必须显式增加其只读 mount；容器内的软链接不会自动挂载宿主机外部目标。

## 3. 官方训练代码与 RGB-T checkpoint 的可确认事实

源码根目录：`../../official_deeppro`（从项目 `infrared_problem_screen_20261009` 根目录定位时为 `official_deeppro`）。实际载入的是 checkpoint 邻接 `DeepPro-Plus.py`；SHA256 为 `fa46a034a30d8bbb0f677271d6368f033fcb389d682478d331551d8bae087668`。权重路径：

```text
official_deeppro/log/sem_seg/RGB-T__2025-01-14_15-27__SoftLoUloss_DeepPro-Plus_DataL40/checkpoints/best_model.pth
```

checkpoint SHA256：`4fd4aa2736d9a33f2bf1459099068d8b4a7e202ff80aecdb4a3277fd13c7abcc`。

| 项目 | 本轮证据支持的内容 |
|---|---|
| checkpoint 训练位置 | `epoch=21`，按源码为完成第 22 轮；其保存的 `class_avg_iou=0.16041548053381177` 是历史 checkpoint 元数据，不是本轮测量结果。 |
| checkpoint optimizer | 64 个参数张量均有 Adam 状态，step 均为 5,192；恢复后 lr=0.00049，betas=(0.9,0.999)，eps=1e-8，weight_decay=1e-4，amsgrad=False。动量形状与参数相符，数值有限。 |
| 当前官方入口默认值 | batch 4、40 输入/输出帧、128 patch、32 epochs、Adam 初始 lr=0.001、wd=0.0001、每 10 个 epoch 乘 0.7、lr 下限 1e-5。**RGB-T 日志为空，不能声称这些全部等于该权重实际训练配置。** |
| BN 模式 | 官方每轮训练 `detector.train()`，所有 BatchNorm3d 更新 running statistics；评估使用 `.eval()`。每通道训练统计覆盖 batch、time、height、width。冻 BN 是额外实验选择，必须各组一致或明确消融。 |
| 数值精度 | 训练 images/targets 转 FP32，无 autocast/GradScaler；本轮 bench 也为 FP32，cuDNN deterministic=True、benchmark=False，TF32 显式关闭。 |
| 原始损失 | `SoftLoUloss`：每样本在 T/H/W 汇总 sigmoid 预测与像素 mask 的交并，最后平均 `1-IoU`，smooth=0。全空 mask 时，只要分母非零，原始损失常量为 1 且对预测无梯度；这需在数据/监督分析时留意，不能据此直接断言本次塌缩由损失导致。 |
| 空间采样 | 每窗一个共同 crop；有目标且随机概率<0.75 时围绕所选中间帧目标质心抖动，否则随机 crop；transform 调用被注释。 |
| 时序采样 | 构造多个真实起点的窗口，按候选长度加权有放回采样；不足40帧在归一化后补0，前/后补由索引奇偶决定。不是固定一个起点训练。 |
| RGB-T 预处理 | mean=85.0799、std=47.4845，原始空间尺度无 resize；mask 除255后正值置1。源码期望 `train2017/<seq>/01` 与 `segmentations/<seq>`，需要真正匹配的原生 mask。 |
| 随机性 | 定义 `seed_everything(46)` 但原训练入口没有调用；worker 的 NumPy seed 使用时间戳；checkpoint 无 RNG state。作者权重训练 seed 不能确认为46。 |
| 选择逻辑 | 官方入口每轮对 test split 评估并按像素 IoU 选 best，不是本项目应沿用的独立开发/确认划分；训练内评估也没有使用 standalone test 的唯一帧 maxfusion。 |

恢复细节见 [runtime/OPTIMIZER_RESTORE.json](runtime/OPTIMIZER_RESTORE.json)，本轮执行脚本为 [runtime/check_optimizer_restore.py](runtime/check_optimizer_restore.py)。载入使用 `weights_only=True` 加窄范围 NumPy scalar/dtype safe globals，没有不受限制反序列化。

必须在独立 wrapper 记录的兼容性适配：

1. NumPy 2.2.5 下 `np.random.choice(self.samplelist, p=...)` 对 ragged nested windows 不兼容。应按相同概率抽一个整数 index，再索引 list，不改变采样分布。
2. `--sample_rate` 的 parser 类型误设为 int；注释建议 RGB-T 为0.05，而通用默认值为0.1。CLI 直接传0.05会失败，不能传0或四舍五入冒充原采样。实际阶段协议须显式记录所选数量/预算。
3. 显式固定 Python、NumPy、Torch、CUDA seed，并用确定性 worker seed 或先用 num_workers=0 验证。原墙钟 worker seed 不能保留后再称可复现。
4. 原 `train.py` 的自动 resume 捕获所有异常并可能回到从零训练，且从保存目录 `best_model.pth` 自动恢复；第一阶段 wrapper 必须明确初始化路径和 strict load，载入失败立即报错。
5. 原脚本训练结束的自动 `test.py` 调用不转发非默认 savepath 到 logpath；应由独立评估入口显式指定新 checkpoint。

## 4. 前反向实测与最小 smoke 建议

GPU：单张 RTX 5090（32,607 MiB）。本轮开始时4张均空闲；这是当时快照，不代表后续可占用预约。只执行合成输入、官方权重、原始 SoftIoU 的 train-mode 前反向；每项先1次预热、再3次计时，CUDA 同步，未执行 optimizer step。它只证明计算通路与梯度有限，不证明优化收敛或模型效果。

| batch | T | patch | 前反向中位秒 | 峰值已分配 GiB | 峰值保留 GiB |
|---:|---:|---:|---:|---:|---:|
| 1 | 40 | 64 | 0.0219 | 0.793 | 1.000 |
| 4 | 40 | 64 | 0.0937 | 1.496 | 2.168 |
| 1 | 40 | 96 | 0.0560 | 1.705 | 2.375 |
| 4 | 40 | 96 | 0.1936 | 3.289 | 3.496 |
| 4 | 40 | 128 | 0.3483 | 5.795 | 6.137 |

原始逐次数字见 [runtime/BENCHMARK.json](runtime/BENCHMARK.json)，stdout 见 [runtime/BENCHMARK.log](runtime/BENCHMARK.log)。数值不含首轮 CUDA 初始化、数据读取/解码、optimizer、checkpoint保存和全图评估；3次热运行不适合宣称训练吞吐提升，也不直接等于完整训练时间。新增模块可能改变内存与时延，应在候选实现后只补测对应配置。

已提供可直接复测的 `bash .../stage1_goal/runtime/run_benchmark.sh`；它会重写本目录的 benchmark JSON，需要留存旧结果时复制到新的运行目录后再调用。无需再次复测来证明当前计算可用。

建议下一步实际训练 smoke：在数据审计通过后，以**真实训练窗** batch 1、40帧、64 patch、原始像素 mask 跑1–2个更新，检查 loss/梯度/权重变化和重载一致性；然后用各组一致的 batch 4 与128 patch做小预算微调。现有显存允许保持官方默认128 crop，没必要仅为显存主动缩小到64/96。若采用64/96用于速度试验，须所有组一致，并承认改变了空间上下文分布。先完成训练窗口、mask、划分的核验，再执行这个 smoke；本文件不把这些未完成事项写成已经通过。

为隔离所提模型机制，应记录 train/eval BN、冻结哪些参数及新旧 optimizer 策略，原方法微调对照与候选相同。测试始终保持40帧真实窗、原始空间分辨率、官方原maxfusion；全图推理仍可能OOM，应沿用已验证空间halo分块，候选若增大空间感受野则重新核验halo。
