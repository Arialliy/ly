# BC-TPro 独立实现审阅

2026-10-11；审阅范围为 `code/bc_tpro.py`、`train_bc.py`、`infer_bc.py`、`launch_jobs.py` 与 `PROTOCOL.md`。只读审阅实现；未训练、未用GPU。评估入口防误用检查由评估作者在自己的 `evaluation_code/evaluate_bc.py` 内补充。当前源码指纹及六项冻结输入核验见 [INDEPENDENT_FROZEN_INPUT_CHECK.json](INDEPENDENT_FROZEN_INPUT_CHECK.json)。

## 模型和训练计算：未发现阻止固定首轮实验的结构性错误

1. 实际发布 DeepPro-Plus 只有一个32通道TPro；包装器原样保存QK_heads、norm1、conv及其状态名称。残差确实插在原时间Linear组合之后、第一BN/ReLU之前。分支rank=4，Q/K共享LayerNorm与投影，不生成新的T×T张量。
2. `reference_context`只读取8个偏移40的邻域特征，排除邻域特征在crop边缘9像素的不完整stem区；没有中心残差、空间反射/循环填充。`features.detach()`阻断参照输入到主干的梯度，检测读取`background.detach()`，而ref辅助目标读取`features.detach()`。这支持设计要求的两条梯度通道隔离；实际梯度测试仍须通过实现门和训练首步门。
3. `gamma=0`、Q/K非零初始化有合理首步梯度路径。训练参数分组有完备/无重复断言；实际优化器直接接收`model.parameters()`，不存在CHAL审计中漏掉直接声明Parameter的问题。新模块随机初始化隔离CPU RNG；两配置新参数和数据顺序可直接比较。所有原BN被置eval且每步核验buffer不变。
4. reference loss按通道先平均、有效位置按窗口归一、最后四窗等权，空背景窗口为可反传零；ref mask另侵蚀空间±9、时间±8，未修改原检测mask。top1检测项直接调用冻结旧函数，无paired额外项混入。
5. 原stem最大空间半径是3+2×(1+2)=9、时间半径2+2×(2+1)=8。新环偏移40使新增空间依赖达到49；参照仅有时间卷积及1×1通道投影，动态关联只在每像素的时间轴和通道轴运算，后级也为1×1。因此halo64在结构上充分。几何count对核心像素的计算在该halo下与完整图相同；BN eval避免跨空间统计使平铺失效。真实非零gamma训练权重的直接前向/平铺检查仍必须完成，数学局部性不能代替数值见证。
6. 训练严格核验计划、划分、原checkpoint、旧训练脚本、旧loss、checkpoint邻接模型源码；本轮独立重算这6个hash均匹配。更新512、最终检查点、同Adam等配置与协议一致。

## 已向主执行者指出的必要证据绑定修正

### A. 实现PASS必须绑定当前源码与配置

审阅时`launch_jobs.py`只检查`IMPLEMENTATION_CHECK.json.status == PASS`，并没有对检查时的`bc_tpro.py`、测试脚本hash与MODEL_CONFIG进行比对。若实现测试后代码变化，旧PASS可能被继续使用。需在启动前把PASS与当前版本/配置绑定。当前源码虽无已知计算错误，不能让门禁成为一个与代码无关的字符串。

### B. 新权重身份必须与实验条件绑定

审阅时`infer_bc.py`只验证`model_config`及strict state加载，没有校验：请求variant与checkpoint.variant相同、step=512、原初始化checkpoint/plan、训练RUN已complete，以及final checkpoint hash。单路权重与完整权重参数形状完全相同，因此严格载入本身无法防止错误权重被记入另一个实验条件。

主执行者应在launch/infer处绑定这些证据。评估入口已追加第二道门：COMPLETE.variant/sequence、训练RUN的complete/step512、最终权重路径与hash、原checkpoint hash、原模型和新增模型源码hash、MODEL_CONFIG、归一化、O/S窗集合、halo≥49、无标签推理标记及训练后平铺见证。缺少任一条件即拒绝进入评分。此门不替代推理前身份检查。

## 解释和后续核验边界

- 128训练crop内环上下文不全，而全图推理可有更多环采样；这是已写入协议的输入边界差异，不应事后称为模型泛化机制证据。
- 背景参照是由辅助目标学习的估计，排除中心并不能保证邻域没有其他目标，或没有同一目标在其他时刻的运动轨迹。不能把邻域采样本身当成已知背景真值。
- 实现门通过、训练首步loss/梯度隔离通过、最终BN/hash及训练后平铺检查通过后，方可据实际M/U/G/L输出研究结论。当前审阅没有新模型检测成绩。
