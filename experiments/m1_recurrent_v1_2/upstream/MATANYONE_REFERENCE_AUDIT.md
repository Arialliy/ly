# MatAnyone 官方参考核验

本次仅阅读、保存作者源码与论文，没有安装或运行 MatAnyone 依赖、下载模型权重，未编写替代的 `BoundedStateUpdate`。官方仓库保持未修改；文件哈希见 `MatAnyone_PROVENANCE.json`。

来源：[CVPR 2025 官方论文](https://openaccess.thecvf.com/content/CVPR2025/html/Yang_MatAnyone_Stable_Video_Matting_with_Consistent_Memory_Propagation_CVPR_2025_paper.html)，7299–7308 页。已下载 CVF PDF 并核读 §3.1、式(1)；不是仅凭题目或摘要引用。官方仓库固定到 [e5ddc534c1fff9bb9e54cf476095d29071b7cb4f](https://github.com/pq-yang/MatAnyone/tree/e5ddc534c1fff9bb9e54cf476095d29071b7cb4f)，提交日期 2026-03-04。这是核验时的 main，不声称是 CVPR 投稿时的代码快照。原始 S-Lab License 1.0 随仓库保存。

## 精确实现边界

| 项目 | 固定版本中的实际实现 | 与用户 v1.2 方案的关系 |
| --- | --- | --- |
| 输入 | 上帧像素特征、当前像素特征、上帧预测 mask、当前记忆读出减存储旧值；mask 用 area 缩至特征尺寸 | `[A,X,R−A]` 没有 mask，A 也不是原作的上帧像素特征，是自定义适配 |
| 通道 | 默认 pixel_dim=value_dim=256，输入 769；三卷积 769→64→32→1，核 1/3/3，均无 bias | 小通道预测器不是原 UncertPred 同构同参实现 |
| 归一化 | 前两层卷积后各有 BN、ReLU；train() 强制子模块 eval，冻结 BN 统计，但不关闭可训练参数梯度 | 方案没有 BN，需要明确列为差异 |
| 梯度 | read_memory、UncertPred、TrainWrapper 中该差分路径没有显式 detach | 不能以 MatAnyone 为依据自行加 detach；参考包若有需单列来源和梯度含义 |
| 组合方向 | U 乘新读出，1−U 乘旧记忆 | 用户 g 乘旧状态，是保留系数；对应方向是 g=1−U |
| 初值 | UncertPred 未特别将输出层置零 | g 初始 0.5 是当前参考设计的初始化，不是原作保证 |

代码依据：[UncertPred](https://github.com/pq-yang/MatAnyone/blob/e5ddc534c1fff9bb9e54cf476095d29071b7cb4f/matanyone/model/big_modules.py#L22)、[read_memory](https://github.com/pq-yang/MatAnyone/blob/e5ddc534c1fff9bb9e54cf476095d29071b7cb4f/matanyone/model/matanyone.py#L127)、[默认配置](https://github.com/pq-yang/MatAnyone/blob/e5ddc534c1fff9bb9e54cf476095d29071b7cb4f/matanyone/config/model/base.yaml#L4)。read_memory 注明为训练路径，推理由 [MemoryManager](https://github.com/pq-yang/MatAnyone/blob/e5ddc534c1fff9bb9e54cf476095d29071b7cb4f/matanyone/inference/memory_manager.py#L249) 实现相同新/旧值混合方向；不能把整个 memory-bank 读出误称为 RFR 的 PDA 对齐。

## 任务和监督条件

论文主设置通过首帧分割 mask 指定目标，先得到 alpha，再编码进记忆。§3.1 变化预测有真实相邻 alpha 差的二分类监督，并用 BCE；U 表示内容变化概率，不能直接等同于未知小弱目标的可靠性。视频 matting 代码将相邻 GT alpha 绝对差 area 下采样 1/16，再以 0.001 为阈值产生标签；图像/分割分支使用不同阈值和监督条件。依据：[训练首帧与递归路径](https://github.com/pq-yang/MatAnyone/blob/e5ddc534c1fff9bb9e54cf476095d29071b7cb4f/matanyone/model/train_wrapper.py#L30)、[监督代码](https://github.com/pq-yang/MatAnyone/blob/e5ddc534c1fff9bb9e54cf476095d29071b7cb4f/matanyone/model/losses.py#L326)。

DeepPro-Plus 当前任务没有指定目标的首帧 mask 或 alpha 变化标签；只通过已有检测损失学习状态选择。因此适当表述是“借鉴 MatAnyone 的内容条件化新旧记忆插值形式，适配到 RFR 检测前状态”，不是复现其不确定性模块，也没有已校准概率的证据。

## 参考包待核对

用户要求复用 `M1_v12_state_update_reference.zip`，本核验不以官方完整仓库替代该包。随后已取得用户原包并实读 `reference/M1_v12_state_update_reference/state_update.py`：确为 48→16→8→1、三层带 bias、无 BN，共 2017 参数；gate 为 `[B,1,H,W]`，descriptor 无 detach，输出层 weight/bias 零初始化。包内文档也明确这是独立任务适配，非作者完整模块。没有自行补写或覆盖该参考核心；新版训练分组以用户最新正文的 Current/History 为准，不能误沿用包内较早的 Frame/Pixel 主比较。

RFR 继续复用 v1 已固定的 PDA/TSFM 和原生 DCN；此次保存 MatAnyone 是增加来源证据，不改变本地 RFR 母方法、时序访问、训练范围或状态更新代码。
