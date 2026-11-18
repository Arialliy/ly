# 从 DeepPro-Plus baseline 新建来源关系 M1

这里从官方 DeepPro-Plus baseline 新建 M1。**只加载 baseline 发布权重；M1 的输入投影、PDA、归约器、读取器及可选 M2 全部重新初始化。** 不加载历史 M1 的代码、状态、权重、优化器或训练缓存。目录名只用于隔离这次代码开发，不表示已完成检测实验。

## 实现入口

| 文件 | 职责 |
| --- | --- |
| [adapter.py](code/adapter.py) | 新建 FIFO4 状态、逐源 PDA、关系归约、简单读取/M2 与 baseline 接入 |
| [归约器](../../source_relational_reducer.py) | 用户给出的 `relations` / `early_merge` 公式，默认 672 参数 |
| [runtime.py](code/runtime.py) | 严格加载官方发布 baseline，按 seed 构造全新 M1/M2 |
| [pda.py](code/pda.py) | 惰性加载独立存放的 RFR PDA 母方法源码，返回新初始化 PDA |
| [母方法源码与构建说明](upstream/rfr/README.md) | 固定 RFR commit、源码校验和原生 DCN 编译 |
| [check.py](check.py) | 在项目已有固定镜像中运行 CPU 检查，保存真实测试收据 |

新代码以 wrapper 接入，官方 baseline 文件保持原样。保留唯一 TPro、`conv_out1`、`conv_out2` 和返回 `(features, logits)` 的接口；sigmoid、五窗融合、0.5 阈值与原匹配逻辑继续由调用端处理。

## 新 M1 的计算路径

```text
官方前端 F [B,32,T,H,W] ─────────── 唯一 TPro ───── H0
        │
        └─ 新投影 32→16 ─ X_t
                             │
             原始 X_j FIFO4 ─┴─ 逐源 PDA(X_j, X_t)
                                      │
                             [B,K,16,H,W]
                                      │
                        共享 16→8 + Tanh
                                      │
                  内容均值 μ + 成对关系均值 R
                                      │
                           E_t=[μ,R]，16通道
                                      │
       简单读取 H0+W[X,E] 或 M2(H0,X,E) ─ 原预测头
```

状态每次 forward 重新创建，每个 batch 行对应一个独立窗口。先读取旧源，再写入当前原始投影；关联结果、关系摘要和检测结果均不写回状态。FIFO 固定为4，不跨窗口传播；内部时间分块继续同一状态，保留梯度。来源索引必须唯一、连续、早于当前帧。

空历史输出 `[psi(X_t),0]`；单历史源输出 `[V_j,0]`。Current 在 PDA **之前**用当前特征替换各历史槽，保留相同槽位时序及调用次数。40帧窗口执行150次 PDA 和220次关系网络调用。这里的 Current 仍保留 baseline 自身的多帧前端/TPro。

归约使用用户方案原式：`D(a,b)=[a+b,abs(a-b),a*b]`，共享 `24→16→8` 关系网络，先编码每一对再均值。`early_merge` 先求来源均值，再对其两个副本使用同一网络。未加入额外损失、来源筛选、目标初始化 mask 或新增归约模式。

## 模型构建与对照

从项目根目录、在已配置 PyTorch/native DCN 的环境中使用：

```python
from experiments.m1_source_relations_v1.code.runtime import load_initial

model, metadata = load_initial("HistoryRelations", device="cuda", seed=46)
model.train()  # 默认只训练新模块；baseline 参数和 BN 维持冻结
features, logits = model(video)  # video: 已按原流程归一化的 [B,1,40,H,W]
```

| 配置 | 新增历史 | 归约 | 读取 |
| --- | --- | --- | --- |
| `CurrentRelations` | 当前替换 | 成对关系 | 简单 |
| `HistoryEarlyMerge` | 真实历史 | 提前归约 | 简单 |
| `HistoryRelations` | 真实历史 | 成对关系 | 简单 |
| `CurrentRelationsM2` | 当前替换 | 成对关系 | M2 |
| `HistoryEarlyMergeM2` | 真实历史 | 提前归约 | M2 |
| `HistoryRelationsM2` | 真实历史 | 成对关系 | M2 |

三组同读取条件使用完全相同的参数初值；跨 simple/M2 的公共投影、PDA、归约器初值相同。简单读取输出投影为零初始化；M2 为 `H0+sigmoid(G(Z))*P(Z)`，`Z=GELU(Conv1x1([H0,X,E]))`，`P` 零初始化、门初始0.5。因此初始预测继承 baseline。第一步上游梯度为零是零残差初始化的预期现象；输出投影更新后梯度进入 M1/M2。

同权重干预：

```python
model.mode = "current_only"    # 只去掉新增 M1 历史
model.adapter.enabled = False # 完全旁路新 M1/M2，直接使用 H0
```

干预不替代独立训练对照；比较后由调用端恢复 `mode` 和 `enabled`。新训练应创建自己的优化器，不恢复任何旧 M1 优化器。该交付只实现模型与接口，不擅自启动训练或确定新的数据曝光方案。

## 约束与验证边界

- 正式模型使用 RFR 原生 PDA/DCN；CPU 接口测试中的 `TestOnlyPDA` 仅用于隔离状态与梯度逻辑，不是运行时替代后端。
- 固定发布模型为40输入、40输出；时间长度不一致时明确报错，不猜测时间映射。PDA 的 H/W 必须是4的倍数，使用 CUDA float32/float64，完整新 M1 不支持 AMP。
- M2 不建立第二套记忆，也不输出独立检测概率。
- `early_merge` 匹配名义参数及调用次数，其差异块恒零，不能宣称匹配了全部有效容量。来源均值与关系摘要也不保证保留来源数量、身份、顺序或统计独立性。
- 原型 ZIP 保持原样；根目录归约器只增加输入验证、返回 dtype 一致性和准确语义说明，核心公式不变。

运行 CPU 检查：

```bash
python3 experiments/m1_source_relations_v1/check.py
```

检查包含归约公式、来源合法性、FIFO写入、Current替换、窗口/batch隔离、分块与checkpoint梯度一致性，以及真实发布 baseline 的输出等价、唯一 TPro 和冻结预测头的梯度通路。测试收据位于 `checks/CPU_*.json`。CUDA 扩展编译/导入状态见 `upstream/rfr/BUILD_REPORT.json`，它不等于 CUDA 数值前后向或真实尺寸显存验证。

当前没有训练结果、目标恢复结果或 CVPR 创新成立的结论。
