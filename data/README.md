# 数据入口

## 当前实验实际使用的数据

[`stage1/`](stage1/) 收录冻结选择中的 **14 个片段、2,240 帧**。每个片段保留原始 JPG、`uint8[160,512,640]` 的 `raw_frames.npy`、全部类别和目标标注，以及逐帧来源、校验与下载记录。

| 冻结划分 | 片段数 | 图像帧数 | 全部标注数 |
|---|---:|---:|---:|
| train | 8 | 1,280 | 16,707 |
| calibration（已用于开发） | 2 | 320 | 16,297 |
| confirmation（保留） | 4 | 640 | 37,272 |
| 合计 | 14 | 2,240 | 70,276 |

选择以 [`DATA_SELECTION.json`](../results/references/DATA_SELECTION.json) 为准；本次收录不改变划分，不将 confirmation 加入训练。当前发布结果在两个 calibration 片段上评价，不能因为数据已全部收录就声称进行了 confirmation 验证。

每个片段的 `images/` 与 `source_archive/.../01/` 指向同一批原图的普通硬链接入口。两处路径便于使用，不代表图像帧数翻倍。原生数组在运行时按 `(float32(raw) - 85.0799) / 47.4845` 归一化；此处保留原始像素，不预先覆盖为归一化数组。

## 历轮材料

[`archive/infrared_problem_screen_20261009/`](../archive/infrared_problem_screen_20261009/) 收录原项目全部本地实验材料，包括历轮报告、代码、训练日志、权重、预测缓存、原始选取片段，以及 NUDT 已用测试子集。仅排除 Git 内部对象、Python/pytest 缓存和独立存放的第三方依赖。

第三方 Python overlay 位于 [`environment/site/`](../environment/site/)，不计为实验数据。

## 完整公开数据集

两个官方关联的公开源归档已完整下载并通过全 ZIP CRC、冻结文件列表比对、路径安全检查和提取校验。原 ZIP 保存在 [`public_archives/`](public_archives/)，完整可读数据位于 [`public/`](public/)。没有遗留 `.part`、`.extracting` 或额外分块实体。

| 公开源归档 | 原 ZIP 字节数 | 展开字节数 | 完成记录 |
|---|---:|---:|---|
| NUDT-MIRSDT | 1,027,614,117 | 1,020,055,152 | [`PUBLIC_ARCHIVE_NUDT-MIRSDT.json`](../provenance/PUBLIC_ARCHIVE_NUDT-MIRSDT.json) |
| VT-Tiny-MOT | 2,448,061,811 | 2,876,983,408 | [`PUBLIC_ARCHIVE_VT-Tiny-MOT.json`](../provenance/PUBLIC_ARCHIVE_VT-Tiny-MOT.json) |

两套原 ZIP 合计 3,475,675,928 字节；展开文件合计 3,897,038,560 字节。下载原件和解包数据均保留。

- [`NUDT-MIRSDT/`](public/NUDT-MIRSDT/)：120 个序列、12,000 张输入图、12,000 张 mask、12,000 张质心 mask、12,000 个 Mix MAT 文件，另有 120 张轨迹预览和原始 train/test 列表。原列表指定 80 个训练序列、20 个测试序列，归档另含 20 个不在这两份列表中的序列，全部原样保留，未重新分配。
- [`VT-Tiny-MOT/`](public/VT-Tiny-MOT/)：116 个序列（train2017 86 / test2017 30），93,882 张双模态 JPG，其中红外 46,941 张、可见光 46,941 张，6 份原始标注 JSON、30 个 INI。逐标注文件的图像数与标注数见 [`PUBLIC_DATA_VERIFICATION.json`](../provenance/PUBLIC_DATA_VERIFICATION.json)，不把跨模态或不同表示的标注相加当作独立目标。

额外一致性核对通过：当前 2,240 张选取原图与完整 VT 归档逐字节一致；70,276 条标注和 2,240 条 image 记录与完整红外标注逐字段一致；既有 NUDT 测试子集的 6,000 个 PNG 与完整归档逐字节一致。未运行模型，也未使用 confirmation 做训练。

VT-Tiny-MOT 是本项目通过公开 MOT 标注入口采用的完整源归档；不能据此声称已获得 DeepPro 作者自行生成的 RGBT-Tiny Gaussian masks 或其精确生成参数。NUDT 的公开像素 mask 属于另一个数据集，不能替代这项缺失。

## 完整性与保存方式

完整本地收录记录见 [`ALL_LOCAL_COLLECTION.json`](../provenance/ALL_LOCAL_COLLECTION.json)：逐文件保留来源、大小、SHA-256 和链接/复制方式。原文件内容未改动。硬链接不依赖旧路径存活，但编辑其中任一入口会改动同一文件；历史数据与结果作为只读快照使用，新实验写到新目录。
