# 红外视频数据

## NUDT-MIRSDT 与当前 S4

S4 使用公开合成数据 NUDT-MIRSDT。每个样本由连续 17 帧构成。模型以原始灰度除以 255 作为输入，预测第 9 帧；训练监督来自中心帧原始 mask。公开包有 120 个序列，发布列表列出 80 个训练序列和 20 个测试序列。S0 将这 80 个训练序列划分为 72 个训练来源和 8 个开发来源；其余 20 个未列入发布划分的序列继续保留。[数据目录](public/NUDT-MIRSDT/) · [冻结清单](../experiments/irstd_independent_v1/manifests/nudt_mirsdt_mask17.json)

| S4 划分 | 来源数 | 有效中心帧 | 空中心帧 |
|---|---:|---:|---:|
| 训练 | 72 | 6,048 | 174 |
| 开发 | 8 | 672 | 0 |
| 历史测试 | 20 | 1,680 | 214 |

正式开发评价使用全部 672 帧；32 帧监测面板只观察训练进程。S0 核验原始图像、掩模、MAT 帧对应关系与发布列表。作者指标使用 `masks_centroid`，以输入像素数作为虚警率分母。历史测试在本模型的 17 帧有效中心子集上评价。[S0 数据审计](../experiments/irstd_independent_v1/reports/S0.json) · [原始 ZIP](public_archives/NUDT-MIRSDT.zip) · [发布收据](../provenance/PUBLIC_ARCHIVE_NUDT-MIRSDT.json) · [逐文件摘要](../provenance/PUBLIC_MEMBERS_NUDT-MIRSDT.json)

## VT-Tiny-MOT 与早期研究

VT-Tiny-MOT 提供连续红外、可见光图像与目标框。完整包包含 86 个 train2017 序列、30 个 test2017 序列。项目另留有历史冻结的 14 个片段、共 2,240 帧，用于小框视频检测的工程检查、原对照和回归。使用这些片段时保留当年的训练、开发和留出用途记录。S4 的中心帧像素监督来自上表的 NUDT-MIRSDT。[完整 VT 数据](public/VT-Tiny-MOT/) · [原始 ZIP](public_archives/VT-Tiny-MOT.zip) · [发布收据](../provenance/PUBLIC_ARCHIVE_VT-Tiny-MOT.json) · [历史片段](stage1/) · [冻结选择](../results/references/DATA_SELECTION.json)
