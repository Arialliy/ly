# DeepPro-Plus 研究工程：源码、完整数据与实验结果

**唯一项目位置：`/home/ly/DeepProPlus_TLFP_Code`。** 原项目已经迁入本目录，旧的`.../idea-stage/infrared_problem_screen_20261009/`位置不再保留。所有新训练、推理与评价在本目录内执行。

| 当前入口 | 内容 |
|---|---|
| [H／H＋LFP本轮结果](experiments/head_lfp_joint_20261012/RESULTS.md) | 两组2048步及全部固定节点已完成；[结论与边界](experiments/head_lfp_joint_20261012/DECISION.md) |
| [本轮结果图](experiments/head_lfp_joint_20261012/figures/final_results.png) | H/HL固定2048步结果；[PDF](experiments/head_lfp_joint_20261012/figures/final_results.pdf) |
| [数据总说明](data/README.md) | 完整NUDT-MIRSDT、VT-Tiny-MOT、原ZIP与冻结14片段输入 |
| [当前工作流状态](experiments/head_lfp_joint_20261012/runtime/WORKFLOW.json) | 每项运行入口、状态和完成记录 |
| [全部历史材料](archive/README.md) | 历轮源码、数据、训练权重、预测、报告与日志 |

当前模型、训练、推理、评价、测试、实际依赖及官方源码统一放在本目录。`sources/` 提供55份常用来源文件，其中53份Python源码，内容与原文件逐字节一致；完整历轮源码另存于 `archive/`，新H/HL代码位于 `experiments/head_lfp_joint_20261012/code/`。 没有改动已冻结实验的模型运算、损失、阈值或评价规则。

**前一轮S/T-LFP完整实验结果已放入本目录**：247个原始实验文件，以及含发布权重在内的12项参照资产，含S/T检查点、预测缓存、逐目标结果表和日志。另有3张统计图、6组完整画面对照及2组低响应目标局部图，均提供PNG和PDF。

| 快速查看 | 入口 |
|---|---|
| 实验结果与权重目录 | [results/README.md](results/README.md) |
| S/T-LFP原始完整报告 | [REPORT.md](results/tlfp_pilot_20261012/REPORT.md) |
| 可视化浏览页（浏览器打开） | [visualizations/index.html](visualizations/index.html) |
| 可视化说明与图片索引 | [visualizations/README.md](visualizations/README.md) |
| 汇总结果CSV | [SUMMARY.csv](results/tlfp_pilot_20261012/evaluation/SUMMARY.csv) |

## 从这里看代码

| 内容 | 入口 |
|---|---|
| 新H/HL接入与训练 | [joint_adaptation.py](experiments/head_lfp_joint_20261012/code/joint_adaptation.py)、[train_joint.py](experiments/head_lfp_joint_20261012/code/train_joint.py) |
| 新H/HL执行入口 | [launch.py](experiments/head_lfp_joint_20261012/code/launch.py)、[workflow.py](experiments/head_lfp_joint_20261012/code/workflow.py)；已有工作流不可重复启动 |
| 当前时序／空间适配器 | [temporal_lfp.py](sources/temporal_diagnosis_20261010/tlfp_pilot_20261012/code/temporal_lfp.py) |
| 空间Haar/LFP运算 | [haar_reference.py](sources/temporal_diagnosis_20261010/route_closeout_20261012/haar_reference.py) |
| 原网络接入与空间分块 | [framewise_adapter.py](sources/temporal_diagnosis_20261010/route_closeout_20261012/framewise_adapter.py) |
| 官方DeepPro-Plus公开定义 | [DeepPro-Plus.py](sources/official_deeppro/networks/models/DeepPro-Plus.py) |
| **本轮实际加载的官方模型与Soft-IoU快照** | [权重对应的DeepPro-Plus.py](sources/official_deeppro/log/sem_seg/RGB-T__2025-01-14_15-27__SoftLoUloss_DeepPro-Plus_DataL40/DeepPro-Plus.py) |
| 官方TPro | [TPro.py](sources/official_deeppro/networks/layers/TPro.py) |
| S/T共同训练 | [train_tlfp.py](sources/temporal_diagnosis_20261010/tlfp_pilot_20261012/code/train_tlfp.py) |
| 完整画面推理及T三种输出 | [infer_tlfp.py](sources/temporal_diagnosis_20261010/tlfp_pilot_20261012/code/infer_tlfp.py) |
| 实际匹配与结果汇总 | [evaluate.py](sources/temporal_diagnosis_20261010/tlfp_pilot_20261012/evaluation_code/evaluate.py) |
| 持续漏检统计 | [trajectory_tables.py](sources/temporal_diagnosis_20261010/tlfp_pilot_20261012/evaluation_code/trajectory_tables.py) |
| 时间条件与关闭分支核验 | [mechanism_tables.py](sources/temporal_diagnosis_20261010/tlfp_pilot_20261012/evaluation_code/mechanism_tables.py) |
| 新时间扩展37项测试 | [test_temporal_lfp.py](sources/temporal_diagnosis_20261010/tlfp_pilot_20261012/code/test_temporal_lfp.py) |
| 所借鉴的作者LFP原始源码，仅作来源参照 | [NS_FPN.py](sources/temporal_diagnosis_20261010/route_closeout_20261012/upstream/NS_FPN.py) |

完整文件表见[CODE_INDEX.md](CODE_INDEX.md)，官方原说明见[官方README](sources/official_deeppro/README.md)。本地LFP是已登记差异的自写参考，不能称为NS-FPN作者整模型复现。

## 为什么保留部分历史目录名称

当前训练实际调用 `train_mutual → train_supervision → train_pilot` 中的公共标签、损失、采样和校验函数；评分实际调用旧 `evaluate → score_pilot → analyze`。这些必要源码已经一起收进来。`sources/`的目录层级保留用于兼容动态导入；BC/E2/N2/R2等已停止模型的完整历史材料保存在`archive/`，不作为本轮新增实验。

`sources/`原文件含固定绝对路径、相邻目录查找和源码哈希校验。为保持历史实验身份，源码内容与相对层级原样保留。完整输入与凭据现已归入包内；新实验通过`experiments/head_lfp_joint_20261012/code/launch.py`进行容器路径映射。**不应绕过映射直接执行历史`launch.py`。** 当前运行仍需本机已验证的Docker镜像及GPU环境，不等于纯Python跨机器直接启动。

## 验证本目录自身

只检查源码完整性和语法，不需要GPU或深度学习环境：

```bash
python3 check.py
```

使用本机已经验证的Docker镜像，检查完整导入链并运行已有37项CPU测试：

```bash
python3 check.py --tests
```

此入口把本目录`sources/`挂到容器中原代码约定的位置，因此能发现缺漏的源码依赖。第三方pytest等依赖使用包内`environment/site/`；其他环境可传`--test-deps /path/to/site`。不安装新环境、不使用GPU、不训练或重新评估检测。

实际验证已完成：55份来源文件校验通过；模型、损失、训练／推理／评分导入通过，原模型70,913参数、S/T新增100/296参数确认，37项CPU测试通过。记录见[validation/](validation/)。这仅验证归拢后的源码完整性，不改变此前T-LFP未恢复持续漏检的实验结论。

## 数据、权重与结果

`results/tlfp_pilot_20261012/` 收录S/T-LFP实验的完整冻结目录，约4.41 GB；`results/references/` 收录发布权重及直接参照。所有其他历史实验已归入`archive/`。新H/HL实验在`experiments/head_lfp_joint_20261012/`独立写入。历史快照使用硬链接避免重复占用空间；后续改动写为新文件，不原地改写冻结结果。

`data/stage1/`包含冻结的14片段、2240帧原图、原始数组及70276条标注；`data/public/`包含完整公开NUDT-MIRSDT和VT-Tiny-MOT，`data/public_archives/`保留完整原ZIP。下载、CRC、安全解包及原选取数据一致性均已核对。作者另行生成的RGBT Gaussian masks及精确参数仍未获得，不能由完整公开框标注推称原监督已复现。

原路径只作为冻结报告中的历史来源文字保留。兼容旧源码时，容器将包内`archive/infrared_problem_screen_20261009/`映射到其约定路径；宿主机实际输入、权重和输出均在本目录。迁移记录见[PROJECT_RELOCATION.json](provenance/PROJECT_RELOCATION.json)，资产现位置见[EXTERNAL_ASSETS.json](provenance/EXTERNAL_ASSETS.json)。

前次清理删除了71个本机生成缓存文件、31个空目录，共1,529,466字节。源码、实验结果、权重、日志、数据与依赖均已保留或迁入包内。原树及Git来源存于`archive/original_project/`，常用可执行归档去除缓存后位于`archive/infrared_problem_screen_20261009/`。

删除范围与逐文件记录见[CLEANUP_COMPLETED.json](provenance/CLEANUP_COMPLETED.json)。源码在`sources/`，冻结实验在`results/`，新生成图片在`visualizations/`。

## 来源与环境

- [源码来源与SHA256](provenance/SOURCE_MANIFEST.json)：55份文件的原路径与新路径。
- [冻结实验协议](provenance/EXPERIMENT_PROTOCOL.md)：原协议副本，不是新实验授权。
- [原环境说明](provenance/ORIGINAL_ENV_SPEC.json)、[基础环境说明](provenance/BASE_ENV_SPEC.json)：复用镜像`sha256:eb956873870d737f2b68dff0724b8f6f56f778829d119b0938d29b7abeaa2f92`。
- [当前执行链依赖版本](requirements-current.txt)：实际导入检查中的版本；完整官方训练程序可能另需作者README列出的依赖。

历史S/T-LFP整理和绘图没有修改旧实验结论。用户随后批准的新H/HL实验另设协议和结果目录，不覆盖旧结果，也不调整阈值。

完整现状见[PACKAGE_STATUS.json](PACKAGE_STATUS.json)，全包逐文件SHA清单见[PACKAGE_MANIFEST.json](PACKAGE_MANIFEST.json)。可用`python3 tools/package_manifest.py verify`重新核对；新增或改动文件后应重新生成清单。
