# 来源映射：作者实现与本项目改写

核对日期2026-10-14。没有将自行设计的缓存称为作者原方法。

| 来源 | 实际核对内容 | 本项目保留/借鉴 | 本项目不采用 |
|---|---|---|---|
| RFR（TGRS2025），RFR_framework.py | forward_train/forward_test的PDA→TSFM递归更新；PDA和TSFM定义 | 本地既有PDA/TSFM算子及接口，用于逐源帧关联 | 作者完整检测头；不另复制DCN版本 |
| StreamVGGT（官方仓库标注ICLR2026），layers/attention.py | Attention.forward在use_cache时拼接历史k/v、返回new_kv并供当前attention读取 | 按来源保存、不由本次读出覆盖历史值、读写分离 | 作者KV编码、RoPE、DINO、全局attention、几何输出和蒸馏 |
| StreamVGGT，models/aggregator.py | past_key_values在block之间传递，streaming与完整序列两种执行 | 将显式状态作为接口；训练/推理对应检查 | 其全局Transformer与数据预处理 |
| DeepPro-Plus，DeepPro-Plus.py | 唯一TPro和原预测头 | 现有本地原网络及M1接入位置 | 不替换baseline或检测规则 |

本项目新增/自定：缓存X而非K/V；K=4；FIFO；源帧到当前逐次PDA/TSFM；均值读出；同计算路径current_only；状态身份和窗口边界检查。这些都是待验证适配，不是作者论文已验证的红外恢复方案。

缓存中的投影特征没有被该缓存后续融合覆盖，不等于原始图像无损，也不等于其中必然包含可恢复目标信息。简单均值与FIFO没有原创性主张。当前原型只验证“为检测前弱特征保留重新读取机会”是否有作用。

已查阅的源码URL：

- https://raw.githubusercontent.com/XinyiYing/RFR/main/codes/model/RFR_framework.py
- https://raw.githubusercontent.com/XinyiYing/RFR/main/codes/model/utils.py
- https://raw.githubusercontent.com/wzzheng/StreamVGGT/main/src/streamvggt/layers/attention.py
- https://raw.githubusercontent.com/wzzheng/StreamVGGT/main/src/streamvggt/models/aggregator.py
- https://raw.githubusercontent.com/TinaLRJ/DeepPro/main/networks/models/DeepPro-Plus.py

母方法与论文：

- https://github.com/XinyiYing/RFR
- https://arxiv.org/abs/2409.12448
- https://github.com/wzzheng/StreamVGGT
- https://arxiv.org/html/2507.11539v2

本包无第三方代码复制和权重分发；参考代码为独立实现。将来使用官方代码须保存其许可证、来源commit/SHA以及本地变更。
