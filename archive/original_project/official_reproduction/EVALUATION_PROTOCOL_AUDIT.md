# DeepPro-Plus 官方推理与评估协议审计

## 范围与结论

本审计只读取官方源码、随权重发布的作者评估日志及本地论文 v5，**没有运行新的模型测试**。以下目标数值来自作者日志，不能写成本地复现结果。源码锁定为 `8fa1a68b94eb22e94ccd0529e6c5ceccdaa7ec28`，审计时 `git status --porcelain` 为空。逐文件 SHA256、完整作者 ROC 与逐序列结果保存在同目录 JSON。

推荐先完成官方 NUDT-MIRSDT 全测试集、官方 DeepPro-Plus 最佳权重、原始 `ShootingRules` 与 `writeNUDTMIRSDT_ROC` 的推理复现。它验证工程与指标口径，尚不构成从头训练复现或真实运动变化能力结论。

## 1. 正式比较目标

作者随权重发布的 `eval.txt` 日期为 2025-06-16，配置为 `seqlen=40, batch_size=1, threshold_eval=0.5, epoch=None`。

| 集合 | TP / GT，阈值 0.5 | FP 像素数 | Pd，% | Fa | AUC，作者日志 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 全部测试序列 | 1724 / 1729 | 3558 | 99.7108155 | 2.689157e-5 | 0.99778 |
| SNR≤3 | 525 / 529 | 870 | 99.2438563 | 1.653845e-5 | 0.99547 |
| 高 SNR | 1199 / 1200 | 2688 | 99.9166667 | 3.372461e-5 | 0.99876 |

全局像素 IoU 在作者日志中为 **0.858493**。Fa 与 AUC 只有日志打印精度；JSON 保留原始计数及全部 27 档阈值，复现时优先对比逐序列整数计数，再对比打印/论文舍入值。

README 全集结果 `99.71 / 2.69×10^-5 / 0.9978` 与低 SNR `99.24 / 1.65×10^-5 / 0.9955` 均与日志舍入一致。论文 v5 Table VIII **低 SNR 只有 Pd/Fa 两列**；其中 `0.9978` 属于全部 NUDT-MIRSDT 的 AUC，不是低 SNR AUC。因此这里没有低 SNR AUC 的版本冲突。

证据：`official_deeppro/README.md:68–85`、`official_deeppro/log/sem_seg/NUDT-MIRSDT__2024-12-28_16-21__SoftLoUloss_DeepPro-Plus_DataL40/eval.txt`、论文 [v5 Table VIII](https://arxiv.org/html/2506.12766v5)。论文 IV-B 也说明采用 logit 阈值 0，即 sigmoid 后 0.5。

## 2. 数据顺序是协议的一部分

官方 `datasets/NUDT-MIRSDT/test.txt` 为 2000 行，20 个序列，每个序列 100 行。顺序为：

```text
Sequence85, Sequence86, Sequence87, Sequence88, Sequence89,
Sequence90, Sequence91, Sequence92, Sequence93, Sequence94,
Sequence95, Sequence96, Sequence97, Sequence47, Sequence56,
Sequence59, Sequence76, Sequence101, Sequence105, Sequence119
```

`writeNUDTMIRSDT_ROC` 硬编码低 SNR 下标 `[7,13,14,15,16,17,18,19]`，对应 `92,47,56,59,76,101,105,119`。并行分片结果汇总时必须恢复上面完整顺序；不能给部分集合或重排序集合直接调用此函数。

**加载器只用 test.txt 提取序列名称和出现顺序，再扫描序列目录中所有图像和 mask。它不按 test.txt 每一行过滤具体帧。** 因此只把 test.txt 截成 40 行而目录保留 100 帧，仍会加载 100 帧。下载与审计应核对目录完整性、同名图像/mask/centroid、文件数量和排序，防止隐藏文件影响 `os.listdir`。

图像与 mask 分别 `np.sort(os.listdir(...))` 后按位置配对，源码不会检查同名对应。NUDT centroid 来自把 mask 路径中的 `masks` 替换为 `masks_centroid`。正式运行前应独立核对这些条件。

证据：`data_utils/TestDataLoader.py:112–134,155–172`、`write_results.py:7–14`。

## 3. 输入、窗口与融合

| 设置 | 官方行为 |
| --- | --- |
| 输入尺寸 | 原生 H×W，不缩放 NUDT 图像 |
| 图像值 | PIL→float32，RGB 时取通道 0，不除 255 |
| 归一化 | `(image - 105.4025) / 26.6452` |
| 模型输入 | `[1,1,40,H,W]`，`detector(1,40,40)` |
| 输出 | `_, logits = detector(images)` 后 sigmoid |
| 推理模式 | `eval()` + `torch.no_grad()` |
| 窗口 | 长度 40，名义重叠 4，名义步长 36，尾窗向前锚定 |
| 重叠融合 | 概率逐像素 maximum |

每个 100 帧序列的 0 起点闭区间窗口为 **[0,39]、[36,75]、[60,99]**。最后重叠为 16 帧，并非 4 帧。不能替换成无重叠分块、平均融合或固定尾部 padding。序列长度不足 40 时源码没有 padding 的实际执行分支，应单独处理/报错，不能声称已经按官方协议支持。

使用 `batch_size=1`；`test.py` 的窗口拼接与 `first_frame/end_frame` 索引假定一批一个窗口。为了并行采用按完整序列分片即可，不改变序列内部 batch 或窗口顺序。

来源：`TestDataLoader.py:17–19,36–96,154–164`；`test.py:88,95–108,130–169`。

## 4. 指标精确定义

### 检出与虚警

直接调用原版：

```python
from ShootingRules import ShootingRules
evaluator = ShootingRules()
false_count, true_count, target_count = evaluator(
    probability_frame[None, ...].copy(),
    centroid_frame[None, ...].copy(),
    threshold,
)
```

- `ShootingRules` 以 **≥threshold** 判定正像素。
- `skimage.measure.label(..., connectivity=2)` 将 centroid 标注按 8 连通分组，分组数为目标数。
- 每个 centroid 连通组的任一标注像素附近 **3×3** 区域有正预测，即该目标检出。
- 所有 centroid 标注像素周围 **9×9** 区域的并集从虚警计数中排除；其余区域的正像素数为 FP。
- 这不是预测连通域与 GT 的一对一匹配，也不是框 AP。
- 源码边界区域使用普通 Python slice，没有显式边界 clipping。复现时不要静默修正；存在边界实例应另行记录。

聚合方式：`Pd=sum(TP)/sum(GT)`，`Fa=sum(FP)/sum(T×H×W)`，不是各序列率的算术平均。所有目标 mask 与 centroid 读入保持源码行为。

### ROC 与 AUC

精确阈值列表共 **27** 个：

```text
0, 1e-20, 1e-10, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3,
1e-2, 1e-1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6,
0.65, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1
```

调用 `sklearn.metrics.auc(Fa, Pd)`，对整条离散曲线直接梯形积分。**没有 Fa 范围裁剪，没有归一化，没有插入 (0,0)/(1,1) 端点。** 原序列的 Fa 随阈值递减，sklearn 可处理此方向。浮点 sigmoid 能饱和到 1，阈值 1 的 `>=` 因而仍有正像素；作者日志全部测试集该点为 TP1689/1729、FP2049。不能擅自将此点改成无检出。

### IoU

IoU 使用 `probability > threshold_eval`，注意它是严格大于，而 Pd/Fa 是大于等于。mask 除以 255 后所有大于 0 的值设为 1。汇总全部帧与序列的交集、并集再相除，即全局像素 IoU；尽管日志写 “avg class IoU”，实际上不是逐帧或逐类别平均。

来源：`ShootingRules.py:34–63`；`test.py:116–120,166–184`；`write_results.py:5–59`。

## 5. 建议调用与适配边界

优先复用原版 `TestIRSeqDataLoader('NUDT-MIRSDT', data_root=..., seq_len=40, cat_len=4, transform=None)`、原 checkpoint 附近模型源码、原版 `ShootingRules` 及 `writeNUDTMIRSDT_ROC`。

聚合后原样调用：

```python
writeNUDTMIRSDT_ROC(
    FalseNumAll, TrueNumAll, TgtNumAll, pixelsNumber,
    total_intersection, total_union, Th_Seg,
    dataset_in_official_order, log_string,
)
```

原始命令入口：

```bash
python test.py --gpu 0 --batch_size 1 --seqlen 40 \
  --dataset NUDT-MIRSDT --datapath /absolute/complete/NUDT-MIRSDT \
  --threshold_eval 0.5 --logpath /isolated/log/ \
  --log_dir NUDT-MIRSDT__2024-12-28_16-21__SoftLoUloss_DeepPro-Plus_DataL40
```

该入口会写入 checkpoint 所属目录的 `eval.txt`，因此应使用隔离的日志目录/运行适配器，保留原始 checkout 与作者日志。`model_name` 从 logs 目录第一个文件名获取，目录应只包含预期模型日志；直接按固定模型文件导入也须在适配记录说明。

当前 torch 版本读取 NumPy checkpoint 元信息可能需要窄范围安全 allowlist；这属于加载兼容，不应改动权重。若使用空间切块，应保留既有完整与切块数值校验并明确披露；原生 NUDT 尺寸的完整前向更接近原始入口。原始性能 profiling 与计时不是检测指标必需部分，不应把共享 GPU、适配器耗时拿去复现论文 FPS。

当前用户要求按照源码固定 seed，显式使用源码声明默认值 **46**。这不能证明作者发布 checkpoint 的实际训练 seed：官方函数定义了该默认值，但官方入口没有调用该函数。

## 6. 验收顺序

1. 下载工件审核确认 20 个完整原始测试序列、目录与 split 顺序、标签一致、hash 留存。
2. 保存推理配置、原始源码/权重 hash、环境、seed46、窗口与预处理设置。
3. 运行完整测试并保存每序列、每阈值 TP/FP/GT 以及像素数、IoU 计数；不能只保存最终舍入表。
4. 对照 JSON 中作者完整曲线，先看计数差异再看论文舍入值。若有差异，不通过改阈值/裁数据/挑 checkpoint 消除，而应定位数据与执行差异。
5. 完成后只能称“官方权重推理复现”；从头训练复现、真实视频条件诊断与新机制效果是后续独立阶段。
