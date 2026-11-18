# 第一阶段数据、监督与划分审计

审计日期：2026-10-10。第 1–6 节为取得第一阶段新数据前的独立只读审计快照；当时没有启动训练、模型推理或下载新图像。计数脚本为 `audit_data_availability.py`，原始计数与 SHA256 见 `DATA_AUDIT.json`；精确可下载成员见 `DATA_CANDIDATE_POOL.json`。随后按冻结选择取得的数据与验证结果见第 7 节。

**结论：现有数据足以启动明确标注为框／点监督协议变体的公平小试。没有找到官方 DeepPro RGB-T 原始生成 mask，因此现在不能把新训练称为原 Gaussian mask 配方的复现，也不能把框诊断指标称为论文原生 Pd/Fa。** 这不是停止研究的理由：所有可训练对照可以共享同一可复核弱监督与评价协议，并保留官方发布权重及其原窗口融合作为额外基线。

## 1. 本地实际可用数据

扫描范围为 `/home/ly` 的非隐藏一级目录及其子目录，跳过隐藏目录、安装依赖目录和缓存；另核查已有推理 Docker 挂载。宿主机不存在 `/data`，相关容器只绑定 `/home/ly` 与其下官方代码。没有发现任何名为 `segmentations` 的目录，也没有发现匹配官方 RGB-T 原始标签的数据树。此结论限于已扫描范围，不声称原作者未公开任何其他渠道。

| VT 热红外序列 | 实际帧范围 | 数量 | 相对项目根目录的图像目录 |
|---|---|---:|---|
| DJI_0024_2 | 00000–00159 | 160 | `runtime/public_subset_vt/VT-Tiny-MOT/train2017/DJI_0024_2/01` |
| DJI_0028_2 | 00173–00332 | 160 | `real_motion_stage2/selected_raw/VT-Tiny-MOT/train2017/DJI_0028_2/01` |
| DJI_0028_4 | 00285–00444 | 160 | `temporal_diagnosis_20261010/additional_clips/DJI_0028_4__4400005__00285__00444/VT-Tiny-MOT/train2017/DJI_0028_4/01` |
| DJI_0032_5 | 00063–00222 | 160 | `temporal_diagnosis_20261010/additional_clips/DJI_0032_5__4801536__00063__00222/VT-Tiny-MOT/train2017/DJI_0032_5/01` |
| DJI_0047_3 | 00000–00159 | 160 | `temporal_diagnosis_20261010/additional_clips/DJI_0047_3__4621500__00000__00159/VT-Tiny-MOT/train2017/DJI_0047_3/01` |

共 **800 帧，5 个序列，4 个数字前缀**。全部已有开发用途：后 4 个序列属于当前机制诊断，DJI_0024_2 是此前工程/问题筛选样本。不能把它作为新确认集。官方 45 个 RGB-T 训练序列和 15 个测试序列的真实 IR 图像当前均未在本地这一数据树中取得。

`official_reproduction/data/NUDT-MIRSDT` 的 20 个官方测试序列真实存在，共 2,000 图像、2,000 mask、2,000 centroid mask；80 个官方训练序列在该树中为 0 图像。NUDT 是合成数据，已用于官方工程协议复算；不能代替当前真实 VT 上下文问题的独立确认。

## 2. 现有 VT 标注是什么

本地完整公开 IR 训练标注：

`runtime/VT_TRAIN_01_ANNOTATIONS.json.gz`

- 86 条 video 元数据，34,111 个图像记录，456,796 条目标标注。
- **456,796／456,796 个 `segmentation` 字段逐项精确等于 bbox 四角矩形。它们是框多边形，不能作为人工描绘的目标像素支持。**
- 所有 image ID 与 file_name 各自唯一，所有已记录图像都在公开 ZIP 中。
- ZIP 的 train IR 图像为 34,351 张，比标注图像记录多 240 张。差额全部来自 `DJI_0061_2/01/00000.jpg`–`00239.jpg`，这些图像没有对应 image 元数据，**不能当成已标注纯背景**。候选池只纳入有元数据的连续帧。
- `frame_id` 不总等于文件名数值，例如 `DJI_0271_2/01/00013.jpg` 的 frame_id 为 0、mot_frame_id 为 13。必须通过原 image ID / file_name 对齐标签，并另外保存局部序号与原数值帧号。
- 候选计数保留所有类别，排除 ignore/crowd；严格小目标为 0<w,h≤16、框完全在图内且不与同帧其他 GT 框相交。此规则只服务诊断分层，不能把其他 GT 从训练背景排除规则中丢掉。

官方 45 个 train 序列在公开 VT 标注中共 87,631 条目标标注，其中满足 w,h≤16、非 ignore/crowd 的有 80,642 条；小目标中行人 39,832、船 13,766、飞机 12,037、无人机 8,053、车 3,518、骑行者 3,155、公交车 281。这是**公开 VT 版本对应序列的标注统计**，不等于已经证明这些 mask 全部进入发布权重的实际训练过程。它足以排除简单的“训练类别完全没出现”说法。

## 3. 原始 mask 与官方协议的边界

官方源码：

- `data_utils/TrainDataLoader.py` 的 RGB-T 图像路径是 `train2017/<sequence>/01`，标签是 `segmentations/<sequence>`。
- `data_utils/TestDataLoader.py` 读 `test2017/<sequence>/01` 和同名 `segmentations/<sequence>`；RGB-T 分支直接令 `centroid = label`。
- 两个 loader 都把 mask 除 255 后的所有正值设成 1。图像与标签分别排序后按位置配对，不主动验证同名对应。
- 论文 IV-F 描述按框生成 rough Gaussian masks，但本地 pinned 仓库未提供这一生成器及其 Gaussian 参数。本次核查 DeepPro/RGBT-Tiny 当前 README、链接入口与 DeepPro 全部 12 条公开 issue，仍未发现这套 RGB-T mask 的公开下载入口。[DeepPro 论文](https://arxiv.org/html/2506.12766v5)、[官方仓库](https://github.com/TinaLRJ/DeepPro)。
- RGBT-Tiny 官方提供的是带 tracking ID 的配对框；原数据下载通过表单申请，并明确链接 HGTMT 的 MOT 标注。当前公共 VT ZIP 94,270 个条目中，没有名称包含 mask 或 segmentation 的条目。[RGBT-Tiny](https://github.com/XinyiYing/RGBT-Tiny)、[HGTMT](https://github.com/xuqingyu26/HGTMT)。

因此不能把 bbox 实心填充、任意 Gaussian 或 bbox 中心点称为“作者原 mask”。若采用可追溯框／点弱监督，应预先写明正样本、框内不确定区域、框外负样本以及 ignore 区域的规则，并让原结构、改模型和改监督对照共享该数据版本。

已存在 `official_reproduction/EVALUATION_PROTOCOL_AUDIT.md`，其详细审计对象是 **NUDT**。不能直接推定其结论覆盖原 RGB-T 论文结果：RGB-T 作者结果文件名为 `eval_ShootingRules_LocLen2_4.txt`，当前 `ShootingRules.py` 则显式写 `LocLen1=1, LocLen2=4`；文件名本身不能证明历史实际代码。因此还需保持 RGB-T 原指标不可复现的边界，而不是通过自行改半径追平论文。

一个有限的正面校验：15 个官方 RGB-T test 序列在公开 VT ZIP 中的逐序列帧数，与作者日志按 `FP / Fa / (640×512)` 推算的帧数逐一相符（浮点差仅来自 Fa 的打印舍入）。这支持序列帧数对应，不证明图像字节一致、mask 相同或协议等价。精确序列、文件名起止与数值保存在 JSON。

## 4. 场景与 checkpoint 暴露边界

官方 train/test 清单分别有 45／15 个序列，序列全名无交叉；以去掉最后一段的命名前缀分组，存在 **DJI_0257、DJI_0277、DJI_0307、DJI_0309** 四个交叉前缀。

这个前缀可以作保守的源视频分组代理，不能凭名称断言同一拍摄事件，更不能凭不同前缀断言地理/视觉场景独立。第一阶段建议：训练、校准、确认之间按前缀完全分离；确认/校准额外排除官方 45 个 train 的所有前缀和已有 5 个开发序列的全部前缀；图像取得后再记录视觉相似性/近重复检查，避免把“命名前缀独立”写成“已证实独立场景”。

发布 RGB-T checkpoint 邻接训练日志为空。官方 45 序列清单和论文描述代表**预期训练划分**，无法用它们证明具体发布 checkpoint 绝对没有见过额外数据。报告须披露此历史边界。对我们自己的小试训练，实际样本与更新必须完整记录。

## 5. 可立即冻结与获取的候选池

公共入口：[VT-Tiny-MOT.zip](https://drive.google.com/file/d/17CgPc3huCf-KMr-I9zFVqmV0hzBVky9w/view)，完整 archive 为 2,448,061,811 bytes；现有 `runtime/fetch_public_zip_subset.py` 支持正常公开下载表单及 HTTP 206 范围读取。此次没有下载新图像。应在执行时重新确认 206、预算、成员 CRC/hash，不需要完整下载 archive。

`DATA_CANDIDATE_POOL.json` 提供 3 组：

| 字段 | 条件 | 序列数 | 唯一前缀数 |
|---|---|---:|---:|
| `official_train_candidates` | 官方 train 全名；首个有元数据、ZIP 存在、数值连续的 160 帧；至少 1 个严格小目标帧 | 42 | 37 |
| `new_prefix_candidates` | 同上长度/标注规则；排除官方 train 和既有开发的全部前缀 | 28 | 26 |
| `all_candidates` | 满足 160 连续已标注帧的全部 VT train 序列，无需有严格小目标 | 见 JSON | 见 JSON |

未进 160 帧训练池的 3 个官方 train 序列是 DJI_0101_2（120 帧）、DJI_0107_1（70 帧）、DJI_0271_2（70 帧）。没有为了补长度复制或填充这些序列。

每个候选有 `selected_members`（160 个精确 ZIP 成员名）、`selected_image_ids`、原文件名数值起止、frame_id/mot_frame_id 起止、目标数、小目标类别、压缩图像字节。父任务可按固定 seed46 从不同前缀选择 8 个训练片段，再从新前缀池选择 2 个校准、4 个确认；不得用任何候选模型分数改变选取。计数分层用于预先保障足够小目标与类型覆盖，训练/确认划分由实验协议单独冻结。

为便于立即判断数据预算，若按序列字典序取新前缀池前 6 个，可得到以下候选（**仅示例，不是实验划分**）：

| 序列 | 范围 | 严格小目标帧 | 压缩图像 bytes |
|---|---|---:|---:|
| DJI_0026_2 | 00000–00159 | 606 | 1,358,821 |
| DJI_0051_3 | 00000–00159 | 160 | 4,688,439 |
| DJI_0067_2 | 00000–00159 | 4,594 | 2,785,985 |
| DJI_0071_2 | 00000–00159 | 458 | 2,414,832 |
| DJI_0075_3 | 00000–00159 | 13,750 | 6,764,416 |
| DJI_0077_2 | 00000–00159 | 744 | 5,308,993 |

不要只训练空中单目标、再把车辆/行人密集场景当唯一确认域；不让极长或极密目标轨迹以像素数量支配训练。这个配置可由类别配额或每轨迹上限预先规定，而不是用确认结果回改。

## 6. 推荐的小试比较口径

1. 以官方发布 checkpoint 初始化所有可训练对照，保留冻结原 checkpoint + 官方 40 帧/步长 36/尾窗回锚/重叠概率 max 融合作为基线。
2. 原结构继续训练、改模型、改监督使用同一原始框版本、同一训练片段、同等图像曝光/更新数/优化器；明确为共有弱监督小试，不能称已精确恢复原训练配方。
3. 校准集只选阈值/固定参数，确认集不回流选模型。按全部有效目标计检测，同时单列严格小目标、原已检出目标保留率、上下文依赖收益组及持续低响应组；框不确定区域、未标注区域的 false positive 处理必须写入协议。
4. 同误报约束下比较检测变化；报告实际 FP 计数与原始分母。框/点诊断规则不能冒称论文原生 Pd/Fa。独立确认若失败，结论为候选未通过当前小试，而非抹去已有机制证据。
5. 完整训练、原 RGB-T 论文指标对比需后续取得/重建并核验原 mask 与历史评价协议；这与当前小试的内部公平比较是两项独立要求。

## 7. 冻结选择后的取数与实体验证

父任务先冻结 `DATA_SELECTION.json`，SHA256 `1f5e617764c0fe27a63440cc5b26d0e27a05db7dbc8c997b0bc5fd74a4b39c92`，再按其 2,240 个精确成员从公开 ZIP 获取。没有因为模型结果改变选择，也没有取得选择以外的图像。

| 划分 | 序列 | 原始帧 | 全部原始目标标注 |
|---|---:|---:|---:|
| train | 8 | 1,280 | 16,707 |
| calibration | 2 | 320 | 16,297 |
| confirmation | 4 | 640 | 37,272 |
| 合计 | 14 | 2,240 | 70,276 |

数据目录为 `data/<sequence>/`，包含原 JPEG 的 `images` 链接、原尺寸 `raw_frames.npy`（uint8，160×512×640）、完整 `annotations.json`、逐帧原 image ID/帧号/hash 的 `FRAME_MANIFEST.json` 和逐文件 `DOWNLOAD_PROVENANCE.json`。没有 resize、伪造像素 mask、改框或过滤类别。

VT ZIP 使用 data descriptor；旧下载助手因此不能用 local header 验证 CRC。第一次材料检查主动捕获了这一缺口，随后只补取 ZIP 中央目录元数据，逐个验证全部原 JPEG 的 CRC32。**最终全部 2,240 原图 CRC 与公共 ZIP 中央目录一致**；没有把助手初始的 `crc_verified=false` 隐去或当作通过。最终独立脚本 `verify_pilot_materialization.py` 又逐项验证了全部 2,240 数组帧与 JPEG 解码逐像素相等、全部 70,276 原标注字段保留以及所有 SHA/CRC，结果 `data/DATA_INTEGRITY.json` 为 PASS。

全部 HTTP body 实际读取 **68,026,770 bytes**；包含下载表单、图像范围、ZIP footer/中央目录以及重试读数。为应对断线，脚本每次 read 前保守扣减最大读取量，累计扣减 **68,061,860／250,000,000 bytes**。未下载整个 ZIP，无 GPU 操作。精确账本为 `data/NETWORK_TRANSFER.json`，容器记录与入口修正为 `data/ACQUISITION_EXECUTION.json`。

原图 SHA 未发现跨序列完全重复，但这不证明独立物理场景。首/中/末原图概览及明确相似场景已记录在 `data/VISUAL_SCENE_REVIEW.md`。特别是训练内 DJI_0043/0057 和 DJI_0113/0115 有明显相似场地，不能把不同数字前缀当独立物理场景。当前只支持命名前缀分离和有限图像审查的边界；不新增可见性真值，也没有运行确认集模型预测。
