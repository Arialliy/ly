# BC-TPro 首轮训练底座与梯度隔离审计

日期：2026-10-11。范围：只读旧训练代码、计划、日志与 8 个训练序列标注；新增训练入口 `code/train_bc.py`，没有执行 GPU 工作、没有修改模型文件。未读取 calibration/confirmation 的标注文件、原始帧或预测。最终参数以本目录 `PROTOCOL.md` 为准；本审计落实一个预注册配置，不进行参数搜索。

## 先修正接入对象

本地发布权重对应 **DeepPro-Plus**，checkpoint 邻接 `DeepPro-Plus.py` 只有一个 `self.TPro`，32 通道、8 heads、40→40 时间映射、空间分辨率不变。不存在 TPro1/TPro2/TPro3。旧训练的 `load_model` 强制加载这一邻接源码，故新代码应在此唯一 TPro 的 `norm1` 前接入；“最高空间分辨率分支”的意图在当前实现对应这个唯一分支。不能换成远端另一模型再称同一个 baseline。

冻结源码与资产：

| 资产 | SHA256 |
|---|---|
| `stage1_goal/pilot/TRAIN_PLAN.json` | `6f86d929ad1275bbb120456dd23a25c7ea9571517f60b4c64a9d5ce6413c975c` |
| `stage1_goal/DATA_SELECTION.json` | `1f5e617764c0fe27a63440cc5b26d0e27a05db7dbc8c997b0bc5fd74a4b39c92` |
| 发布 `best_model.pth` | `4fd4aa2736d9a33f2bf1459099068d8b4a7e202ff80aecdb4a3277fd13c7abcc` |
| checkpoint 邻接模型源码 | `fa46a034a30d8bbb0f677271d6368f033fcb389d682478d331551d8bae087668` |
| top1 `code/train_pilot.py` | `75e4fe9b4115af837cc1769471328a0b88fca8e546ec8d6f7db73d580f460b54` |
| top1 `code/losses.py` | `6e7021b9bd7cc362988473449596eb92212312384d51704343928af120af7ed1` |

## 可以直接复用的设置

| 项 | 固定值 |
|---|---|
| 初始化 | 两配置都从发布 best_model.pth 严格加载旧参数，不能从 top1 final checkpoint 继续 |
| 步数/取点 | 512 更新；只评价最终一步，不挑 best checkpoint |
| 优化器 | fresh Adam，lr=0.0001，weight_decay=0.0001，默认 betas=(0.9,0.999)、eps=1e-8 |
| 原参数 | 全部可训练，包括 BN affine；BN running buffers 固定 |
| 精度 | FP32，无 AMP，TF32 false；torch deterministic、cudnn deterministic true、benchmark false |
| 随机种子 | Python/NumPy/Torch/CUDA 均 46 |
| 输入 | 四个 `[1,40,128,128]` 窗口；uint8 原帧 `(x−85.0799)/47.4845`；无新增随机增强 |
| 数据顺序 | 原 TRAIN_PLAN 512 entries；每 step 2 段、每段 2 个窗口；不用重新抽样 |
| 检测损失 | 每窗框内 max 的 softplus 正损失先按对象平均；每窗背景最高 ceil(1%×N) softplus 损失均值；四窗等权；正负系数都是1 |
| 配对项 | effective lambda=0；旧 plan.config.extra_lambda=1 只是历史采样元数据，不能重新启用 |
| top1 ties | 截断值相同按扁平化索引先后选到 k 个，沿用旧实现 |

启动时可复用 `top1_background_control_20261011/training/LAUNCH.json` 的 command 数组，替换容器名、脚本路径和输出目录，然后显式加入新模型配置参数。不要直接运行旧 launcher：它固定旧目录与容器名。

原容器镜像：`deeppro-official-eval:20261009`，ID `sha256:eb956873870d737f2b68dff0724b8f6f56f778829d119b0938d29b7abeaa2f92`。原启动资源为 `--cpus 4 --memory 12g --user 1004:1004 --gpus all`，`CUDA_VISIBLE_DEVICES=0`，线程数2，`CUBLAS_WORKSPACE_CONFIG=:4096:8`，`--entrypoint python`。新模块增加的实际显存需 smoke test 记录，不能把原 12 GiB host memory 限制当成 GPU 内存预算。

新训练入口应继续接受以下**原有参数契约**（路径前缀 Q 为本轮 `temporal_diagnosis_20261010`；R 为其父目录）：

```text
--dataset-root Q/stage1_goal/data
--selection Q/stage1_goal/DATA_SELECTION.json
--plan Q/stage1_goal/pilot/TRAIN_PLAN.json
--source R/official_deeppro
--checkpoint R/official_deeppro/log/sem_seg/RGB-T__2025-01-14_15-27__SoftLoUloss_DeepPro-Plus_DataL40/checkpoints/best_model.pth
--strategy paired_mean --steps 512 --lr 0.0001 --seed 46
```

`paired_mean` 仍只是未优化额外项的遗留参数名称，不代表新增训练策略。完整可复用原 argv 已另存 `REUSED_TRAINING_COMMAND.json`，作为具体路径清单；它不是本轮已执行记录。

## 掩码审计

`build_batch` 的检测背景掩码先取 `[9:119,9:119]` 核心范围，再遍历该窗口每一帧的**全部有效框**：不限类别、不限尺寸、不限 strict-small、不忽略 ignore/crowd。每框向外扩2 px，再按 `ceil(x−0.5)` 的像素中心/半开矩形规则裁剪并置 False。正样本则要求非 ignore/crowd 且完整位于核心；两者条件故意不同。应原样复用，不能只拿 positive_boxes 构造背景掩码。

辅助参考损失可以复用全部框排除原则，但需保留独立 `ref_background_mask`，不能顺手改检测 mask。进入 TPro 的特征已经含邻域信息：初始卷积空间半径3、时间半径2；两个 STD block 各增加空间1+2、时间2+1，因此进入 TPro 的特征支持域上界为**空间±9、时间±8**。

实现已采用更严格的 ref mask：在原全框扩2 px、core9背景mask上，再按空间±9、时间±8侵蚀，时间与裁剪外边界记无效，并要求有效环采样 count>0；这只影响辅助目标，检测 mask 完全不变。两配置使用完全相同定义，记录每窗口有效数与空集合数。空集合为可微零，外层仍四窗等权。

## 只用训练标注固定空间支持域

`TRAIN_ANNOTATION_SIZE_STATS.json` 来自 TRAIN_PLAN 中8段训练标注，nearest-rank 分位数，按目标帧计数：

| 集合 | N | 中位 max(w,h) | q95 | q99 | 最大 |
|---|---:|---:|---:|---:|---:|
| 全部有效目标框 | 16,707 | 9 | 15 | 19 | 23 |
| max(w,h)≤16 | 16,251 | 9 | 14 | 16 | 16 |

最终采用全部训练框 maxside 的最大值23，预定采样 offset40，满足 `40>23+9`。因此对任意训练框内目标位置，参照源特征的空间感受野不直接覆盖该中心目标框；它不保证更大目标或邻近其他目标不进入参照。使用8个固定方向，实际有效采样点归一化；参考预测器随后仅空间1×1、时间卷积，不进行中心跳接、反射/循环填充或全空间统计归一化。

不能将“仅排除 F 中心格点”升级为原图 blind spot；当前更远的40偏移专门计入了原 backbone 的空间9半径。新增路径的空间半径为49，推理采用 halo64，并按新模型成本记录；旧 halo16 不足以重现这一上下文。训练 crop128保持原协议，因此部分位置不能取得完整8方向上下文，使用有效点归一并记录 count，而不是扩大裁剪或用测试标注调整半径。

## 辅助项、归一化和梯度流

本轮固定 **lambda_ref=1.0，原特征单位 L1**，窗口内对有效位置和32通道平均、再四窗等权。另记录 `L1 / max(mean(abs(F_bg)),1e-6)` 作为被动相对误差；该量不进入优化、不用于挑 checkpoint。不为了“总损失同量级”扫描 lambda。

理由是设计要求两条学习路径严格隔离：

```text
F = original_backbone(x)
B_train = background_predictor(context(F.detach()))
L_ref = valid_background_mean(abs(B_train - F.detach()))
B_det = B_train.detach()
E = F - B_det
delta = association(F, B_det, E)
H_new = H_original + gamma * delta
L = L_det + 1.0 * L_ref
```

`L_ref` 只更新参考预测器；`L_det` 更新原模型、共享 Q/K 与 gamma，而不更新参考预测器。因此 lambda 不承担两任务在共享主干上的梯度平衡作用。它仍会经 Adam eps、weight decay 等影响参考分支更新，不能称其数学上无关。原优化器统一 weight_decay 可保留；需明确参考分支除了 L_ref 还受到同样的 optimizer 正则。

必须在训练前检查：

1. L_ref-only backward：原主干/TPro/预测头/QK/gamma 的梯度为 None 或严格0；ref 参数有有限非零梯度。
2. L_det-only backward：ref 参数梯度为 None；gamma 梯度应可训练。**不能因 gamma=0 就跳过计算 delta 或把整个新增分支置于 no_grad**，否则会切断 gamma 的启动梯度。
3. ref 输入/target 都 detach；检测侧 B detach，但 shared association 对 Q/K 参数仍有梯度，不能把整个 A_bg 计算 no_grad。
4. gamma=0 时输出与原模型一致；新分支有限，防 `0×NaN` 破坏等价性。初始检测损失及旧参数梯度可对照冻结 top1 的第一步。gamma=0 会令 Q/K 检测梯度为0，这是预期，不是死分支证明。
5. F=B 时差分修正为0；单路控制应保留相同参考分支、L_ref、共享输入与容量规则。

## 同数据、同初始条件与 BN

- 原 TRAIN_PLAN 的 entries 固定了所有图像、crop、起点、对象列表和 random_pairs。逐步比对输入/正框/背景mask指纹，是数据一致性保证；只写相同 seed 不够。
- 新模块初始化应放在 `torch.random.fork_rng` 中使用固定初始化 seed，之后恢复全局 RNG。两配置的新参考模块和共享 Q/K 应复制同一个初始 state_dict，不能仅靠不同构造路径各自 seed46。
- 模型初始零 gamma 确保旧模型前向一致，不确保训练后保持。检测支路完整与单路产生不同旧参数更新后，参考分支监督 F 也会不同，这是同一联训方案的自然结果；要记录，不能称参考分支全程权重相同。
- `model.train()` 后遍历原 `_BatchNorm` 置 eval，每个 step 保持；旧 buffer 哈希初值为 `51832be8d623f18e372232ddefdf4214ad67b3e45f795fbaa2ea9daa7fd0ecab`。记录旧 BN affine 可训练、旧 running buffers bitwise 固定。新增模块尽量不引入 BN；若引入必须单独声明训练状态，不能给新 BN 也写“发布统计量”。
- 启动前、完成后记录源码/计划/权重/输入指纹；512步日志保留 Ldet 正负项、Lref原值与加权值、ref有效数、相对误差、gamma、分支梯度范数、原参数梯度范数、峰值显存和耗时。依旧固定最终一步。

本审计不扩展训练矩阵。它支持先运行完整 BC-TPro 与单路动态控制；未约束双路差分配置留待现有两配置结果有继续价值时再做。

## 已实现训练入口检查

`code/train_bc.py` 已使用旧 `train_pilot` 的 SequenceData/build_batch/load_model 和旧 losses.py，没有复制重写数据或检测损失逻辑。启动时强制核验表中全部 SHA 与逐训练序列数据指纹；附加模块前后核验原 state_dict 键和值完全不变，记录两配置新增模块初始 tensor hash 与每步输入/正框/背景mask hash。每 step 核验旧 BN running buffers，并日志保存其 hash 与显存。

首步实际训练数据还执行 `autograd.grad` 隔离审计：L_ref 对原参数、关联生成器和 gamma 不产生梯度；L_det 对 reference 参数不产生梯度；L_ref 自身梯度有限且非零。初始正/负检测项必须与已冻结 top1 第一步在 1e-6 容差内一致。失败保存 `GRADIENT_ISOLATION.json` 与 RUN.failed 并立即停止；这不增加优化次数。

CPU AST 与命令行 `--help` 静态检查通过，记录 `TRAIN_RUNNER_STATIC_CHECK.json`。梯度审计属于首步运行时检查，不在静态报告中冒充已经执行。训练入口还在 attach 之后恢复 CUDA seed46，避免新模块 CPU fork 中 `torch.manual_seed` 意外改写 CUDA lazy seed；两个新增模块保持 seed20261011 初始值。

工程导入修正：首次启动在 diagnostics 模块的 `from losses import background_top1pct_mean` 处失败，位于读取计划、加载模型和 forward/update 之前。现已把经过 SHA 验证的 losses 模块显式注册为 `sys.modules['losses']` 后再导入 diagnostics；没有改变损失或训练参数。随后在同一冻结 Docker 镜像内完成训练入口、base、losses、diagnostics 的真实 CPU 导入和一次合成诊断调用，并核实 helper 函数身份一致，记录 `TRAIN_RUNNER_IMPORT_SMOKE.json=PASS`。旧静态检查只代表当时源码，不能代替这次运行时导入检查。
