# M1-v1.2：先检验历史恢复

本轮仅将 M1 的状态更新改为内容条件化插值。研究对象仍是既定有效轨迹区间中的持续漏检；门控幅度、状态差异或背景减少均不能替代实际目标恢复。先执行新版 Current／History 的同容量短实验，Frame／Pixel 消融留待出现恢复证据后再登记。

## 代码与来源

原样导入用户提供的 `M1_v12_state_update_reference.zip`，核心为 `reference/M1_v12_state_update_reference/state_update.py`；哈希见 `REFERENCE_IMPORT.json`。包内旧 README 的 Frame／Pixel 主比较由用户最新正文覆盖，核心实现不改。

继续使用 v1 固定的 RFR PDA、TSFM 和原生 DCNv2，提交 `fd71ccdf17f056f80e0a5ea8726b87c82c27fb69`。[RFR，IEEE TGRS 2025](https://doi.org/10.1109/TGRS.2025.3542368)。内容条件化的新旧值插值参考 [MatAnyone，CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Yang_MatAnyone_Stable_Video_Matting_with_Consistent_Memory_Propagation_CVPR_2025_paper.html)，官方代码固定到 `e5ddc534c1fff9bb9e54cf476095d29071b7cb4f`。精确源码边界见 `upstream/MATANYONE_REFERENCE_AUDIT.md`。

MatAnyone 的变化概率乘新读出，本模块的 g 乘旧状态；方向相反。这里的 `[A,X,R−A]`、48→16→8→1 通道、无 mask／BN、输出零初始化和原检测任务监督均为任务适配。没有复现作者完整 matting 模型，没有导入其预训练权重，不把 g 称作目标概率或已校准不确定性。

## 唯一结构变化

非首帧计算 `A=PDA(previous,X)`、`R=TSFM(A,X)`、`(M,g)=BoundedStateUpdate(A,R,X,scope='pixel')`。即 `M=g*A+(1-g)*R`，撤回 v1.1 的无条件 `A+R`。更新器新增 2017 参数、6 个参数张量；原 53 个可训练张量的名称、初值及优化器历史逐一保持。整个模型 182280 参数，可训练 111367 参数。

首帧 `M=X`，不执行 PDA／TSFM／更新器；每个原合法 40 帧窗口重新初始化。原 TPro、输入投影、`H+Wout[X,M]` 接口与预测头保持。g 初始为 0.5，不等价于旧 M=R；保留已经训练的 Wout，不以重新置零掩盖变化。原跨帧反传和激活检查点设置保持，无新增 detach。

## 成对训练

两组均从 v1.0 History 的 `snapshot_2048.pth` 开始，SHA256 `eda063496757f368d7442a7fac99a64f7b33ef0414ef99d006716d77a6dcb725`。

| 条件 | M1 递归输入 | 更新器 |
| --- | --- | --- |
| V12Current | 每个非首帧用当前 X 替代 previous | 同一 Pixel 更新器，全部 PDA／TSFM 运算执行 |
| V12History | 上一步实际 M | 同一 Pixel 更新器 |

门控统一 seed46 初始化；两组先严格加载原权重，再安装新更新器。按父 RUN 原 53 名称顺序恢复 Adam，之后加入新门控参数组。旧状态从 step2048 延续，新门控 Adam 状态为空起步；lr=1e−5，weight_decay=1e−4，betas=(0.9,0.999)，eps=1e−8。恢复相同旧检查点 RNG。每组各 128 步，原冻结 F 曝光 entry0…127 各一次；不扩大数据、不重选样本。保存额外步数 0／32／64／128，即总步数 2048／2080／2112／2176，2176 为唯一主终点。

原损失及系数、4 检测+4 辅助窗口、microbatch2、128 像素训练裁剪、FP32、无 AMP／TF32、原网络参数及 BN 全冻结保持。CUDA DCN 原子加法和 adaptive pooling 反向有已有非确定性，不宣称逐 bit 训练可复现。八训练来源、两开发来源及四未参与设计来源不变。

每步保存原训练损失、已有固定 cache 的目标／背景 logit 指标、门控统计和门控合并梯度；首两步及 32／64／128 步另存逐层梯度与实际更新量。状态作用由终点同权重 trace 完整记录。另在保存的 0／32／64／128 步对既有校准索引 `[5,6,18,25,30,32,51,7]` 做只读诊断，同一批次和固定背景用于观察学习变化；不产生梯度或新的训练曝光，不据此选终点。

## 推理和诊断

原 O={0,36,72,108,120}、S={0,24,60,96,120}，合并八种窗口起点。归一化、概率最大值融合、0.5 阈值和真实组件峰点匹配均复用原实现。M/U 是诊断匹配／未匹配组件数，不改称论文 Pd/Fa。

终点主表包含 `V12CurrentCurrent`（训练 Current 正常模式）、`V12HistoryHistory` 和 `V12HistoryCurrent`（同一 History 权重的 current_only 干预）。Current 仍拥有原前端和 TPro 的时间信息，只取消 M1 额外递归历史。完整轨迹区间、4342 低响应、4170 极低响应、646 保护对象、共同恢复 9 个对象和固定困难背景继续复用，不输入模型、不重新筛选。

History 同权重追踪 before_PDA → after_PDA → TSFM 候选 → after_update → 接入残差／H → 原 head → logits，明确区分候选与最终状态。保存目标和固定背景的门控、logit、状态差异；首帧无门控，单列，不伪记 g=0。额外 Current 权重的 history 干预若为复用 trace 而计算，仅作机制诊断，不进入三路主比较。

前端沿用已验证 core128／halo16 分块并拼回完整 F/H，M1 全图执行，head 按时间分块。TSFM 含全局池化，不能把有限 halo 的 M1 空间分块宣称为等价。新门控自身只验证 halo2 的局部空间一致性；时序分块不得重置／detach 状态。

## 验收和决策

先看低响应实际恢复、连续漏检及逐轨迹覆盖，再看历史独立增量，最后看 U≤4015、背景面积、保护及其他对象损失。成对训练和同 History 权重两种比较分别单列 R_hist 与 L_hist，不只报净增。O/S 分别报告，再按(sequence,frame,annotation_id)去重；二者不是独立视频证据。共同 U 预算通过不等于相对发布 S 的 U=3967 没有新增代价。

实现检查仅覆盖新核心、严格迁移、优化器映射、历史输入语义、初值／梯度启动、时间隔离及门控局部分块；通过不代表检测有效。若只有背景下降或 Current／History 共同适配收益，本轮不宣称完成历史恢复，不自动长训或增设 M2。是否登记成对延续由训练侧固定输入信号与真实匹配结果共同判断，不能事后更换主终点。

所有新工件位于本目录；旧 v1、同权重诊断及 v1.1 文件保持冻结。运行沿用已实际使用的 Docker 镜像 `sha256:8470a68886ff2f480d76858d930fb72d0e680e69276fc1d068fe8b6625b4cb9f`，仅在空闲的已授权 GPU1／2 启动。代码、核心包、检查和协议在训练前登记哈希。
