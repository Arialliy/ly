# C-readout-v3：只修改新增逐帧项的负权重

本轮执行用户提供的 v3 方案。先审查和修正代码，再从同一 C2048 模型与完整 Adam 状态新增 D、N 各 256 步短训。已有 F（v2 FRAME）、CONT、C_init 与 release 直接引用其已保存预测及评分。主终点固定为 2304；2112、2176 仅保存过程，不据此选择模型。无新网络、采样器、阈值调整、GAP 训练、超参数搜索或额外校准。

## 唯一改动和原归约

总损失仍为原 C 的 det + lambda_keep*keep + lambda_temporal*C，加原 lambda_readout 乘逐帧项。正项保持旧计算路径 -logsigmoid(q_positive)。负 BCE 保持 -logsigmoid(-q_negative)。F 权重为 1；D 为 stopgrad(sigmoid(q_negative)^2)；N 为 stopgrad(a*sigmoid(q_negative)^2)，a=sum(p)/sum(p^3)。gamma 固定 2。

N 的原有归约组为每个已有 8 帧段。保留段内均值、窗口内有效段均值、固定 8 窗口均值和重叠段的重复曝光。每段外层系数为 1/(8*n_window_segments*L_segment)。不将分组替换成全 batch，不改变目标 bbox 均值读出或背景局部/全局合法 ROI 均值的最大值读出。直接读取旧 v1/v2 缓存及掩码。

N 在同一快照上保存负读出 logit 梯度 L1 总量；这不等于保存参数梯度范数、Adam 更新量，也不表示独立演化分支的梯度预算相同。D 会减小总体新增负监督。原 C 自身的背景监督保持不变。

参考实现的极低等分数组存在 FP32 抵消误差。本地仅把停止梯度的 N log-weight 组合与归一化改为 float64，随后转回原损失类型；无 epsilon 或裁剪。模型和可微损失沿用 FP32。原 ZIP 和提取源码未改动，修正及实测见 resources/CODE_REVIEW.md。

## 固定训练合同

- 初态：track_contrast_v1/training/C/run/snapshot_2048.pth；完整恢复 28 个 Adam 状态及 step2048。
- seed46、原计划索引 0..255、原 4 检测窗 + 4 公共辅助窗、microbatch2；输入、标签、辅助输入逐步哈希与旧 F 对齐。
- Adam lr1e-5、weight_decay1e-4、betas(0.9,0.999)、eps1e-8、无 scheduler。
- temperature=0.47569116950035095、lambda_temporal=0.10885460558865433、lambda_keep=5.75443767610971、lambda_readout=0.0023821861727945912，直接使用 v2 CALIBRATION.json。
- 唯一可训练前缀 TPro.、conv_out1.、conv_out2.；28 张量、14,593 参数，总参数 70,913。全部 model.eval，BN buffers 与冻结前端不变。
- FP32、确定性设置、无 AMP/TF32；镜像 sha256:8470a68886ff2f480d76858d930fb72d0e680e69276fc1d068fe8b6625b4cb9f。
- 保存2048、2112、2176、2304；固定训练诊断8批[5,6,18,25,30,32,51,7]，不使用开发集决定系数。

## 训练前证据与 F 复用

原参考30项及本地37项 CPU 测试通过；实际 DEIM 原负分支损失/梯度对照通过。GPU2 的固定8批预检不更新模型：unit 与旧 F 损失误差0，所有模式正 q 梯度误差0，unit 负 q 梯度与旧 F 的 assert_close 通过。未单独持久化最后一项的最大误差，因此不声称其独立实测误差为0。

1,222个有效段均存在不同背景分数，单样本及等分数组均为0，N 不退化为 F。N 实际累加后的负 q 梯度 L1 比值范围[0.9999999856354176,1.0000000402987224]；D 范围[1.631823395489083e-10,0.20543108042870406]。见 REAL_WEIGHTING_CHECK.json 与 F_REUSE_AUDIT.json。

F 初态、完整优化器、seed、确定性设置、原实现与256步曝光核对通过，unit 检查通过，允许复用。旧 F 未保存 RNG 状态快照；一致性依据为相同 seed/setup/初始化和不包含随机操作的数据/损失执行路径，不能宣称已逐位比对旧 CUDA RNG 快照。新预检测量 CPU torch RNG 未变，未把该测量扩大为所有 RNG 的现场检查。

## 评价与推进规则

只为新 D/N 运行两个既有开发序列 DJI_0026_2、DJI_0067_2 的推理。O=[0,36,72,108,120]、S=[0,24,60,96,120]；原五窗融合、0.5阈值、原实际连通域匹配不变。引用旧四组缓存，核查新别名及 join，不重跑旧完整标注/模型/匹配审计。

报告 strict5200、low4342、extreme4170、protect646、其他GT11097、130条轨迹的持续漏检与010；同时报告 U、未匹配域面积和标注外像素。M/U 仍为诊断指标，不称官方 Pd/Fa。比较 N-F、N-D、D-F、候选-release，并保留 CONT 和 C_init。使用逐对象/逐轨迹配对变化定位收益和代价。

若收益仍是少量临界帧且极低响应和持续漏检无改善，保留有限结果，不扩大长训。只有恢复覆盖多条轨迹、代价可接受且优于两组控制，才支持下一阶段成对重复及 A/B/C 共同协议；本轮不自动开启该阶段。缺少独立可见性/遮挡标签，连续有效标注仅作代理，不能称已验证可见区间。

GPU安排：GPU2已经执行预检；GPU0/1分别运行D/N，后续4个新推理任务按空闲情况分配到3张卡。不为填满3卡重训F。所有新产物位于 readout_weighting_v3；历史目录只读。

## 来源边界

D 仅借鉴 DEIM loss_labels_mal 中停止梯度的负样本置信度平方权重；N 为本次自定义梯度预算适配。固定官方提交 09d35d53d39ee3145a1e61e3a989b28b9468d1dd，criterion 文件 SHA256 380a7d86b71ba9a3cbbdb670b4c11a3f2bfd2e367ffe87db7ef8e9ba30806dd2。原函数全背景分支对照不等于复现完整 MAL、DEIM 匹配、质量正标签或作者训练成绩。该加权本身不作为原创时序贡献。
