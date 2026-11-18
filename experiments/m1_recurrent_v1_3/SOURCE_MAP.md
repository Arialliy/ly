# M1-v1.3 源码语义对应

本表补充并精确化用户原包的来源表，不改写原包。固定版本及文件哈希见 [PROVENANCE.md](PROVENANCE.md)。

| 来源与定位 | 已核验的代码行为 | v1.3 的使用边界 |
| --- | --- | --- |
| [StreamVGGT attention.py，固定提交 L45–106](https://github.com/wzzheng/StreamVGGT/blob/05682bcb05683c4c71522e9cb618fc8ca349063c/src/streamvggt/layers/attention.py#L45) | 当前隐状态先生成 Q/K/V。缓存分支把当前 K/V 加一维并沿帧维拼接旧缓存，形成 `[B,heads,frames,tokens,head_dim]`；返回的 `new_kv` 在 Q/K norm 和 RoPE 之前形成。当前 Q 读取包括当前帧在内的缓存。 | 借鉴“保存与当前读出分离”；本项目保存的是已有投影 X，不是该层 K/V，也不使用注意力权重。 |
| [同文件缓存与输出位置](https://github.com/wzzheng/StreamVGGT/blob/05682bcb05683c4c71522e9cb618fc8ca349063c/src/streamvggt/layers/attention.py#L57) | 旧槽由 concat 保留，当前 attention 输出没有覆盖旧槽；该实现没有容量 K=4、FIFO 驱逐或显式 detach。 | 本项目的容量、驱逐、read-before-write 和无 detach 训练是单独登记的语义，不能说来自作者的同一实现。 |
| [StreamVGGT block.py，L72–90](https://github.com/wzzheng/StreamVGGT/blob/05682bcb05683c4c71522e9cb618fc8ca349063c/src/streamvggt/layers/block.py#L72) | block 传入自己的历史缓存，将注意力结果用于残差路径，另行返回 `new_kv`。 | 只用作接口参照；没有移植其残差块、归一化或 MLP。 |
| [StreamVGGT aggregator.py，L261–294](https://github.com/wzzheng/StreamVGGT/blob/05682bcb05683c4c71522e9cb618fc8ca349063c/src/streamvggt/models/aggregator.py#L261) | 每个 global block 使用 `past_key_values[global_idx]`，更新对应层槽并返回整份列表；frame attention 不走这一缓存。 | 这是逐层状态列表，不是把一层的同一 K/V 直接给所有层，也不是源帧 PDA bank。 |
| [同文件 L205–214、L342–367](https://github.com/wzzheng/StreamVGGT/blob/05682bcb05683c4c71522e9cb618fc8ca349063c/src/streamvggt/models/aggregator.py#L342) | 缓存路径预期单帧输入；非缓存路径按帧位置构造阻断未来帧的 mask。缓存路径以已有过去及当前 K/V 为可读范围。 | 不能把两条作者路径的存在写成已证明本项目训练/推理等价；本项目必须有自己的 chunk、reset、梯度和真实算子检查。 |
| [RFR 的 PDA/TSFM](https://github.com/XinyiYing/RFR/blob/fd71ccdf17f056f80e0a5ea8726b87c82c27fb69/codes/model/RFR_framework.py) 与 [utils.py](https://github.com/XinyiYing/RFR/blob/fd71ccdf17f056f80e0a5ea8726b87c82c27fb69/codes/model/utils.py) | 沿用已核验的真实对齐和融合算子及 native DCN。 | 16 通道、逐源帧调用顺序和源帧读出均值是现有 M1 接口上的适配；TSFM 含全域频率池化，不能把整个 M1 空间分块后声称等价。 |
| [用户 evidence_bank.py](reference/M1_v13_evidence_bank_reference/evidence_bank.py) | `read_then_write` 先 clone 当前 X 作为待存副本；逐槽 align/fuse 后均值；输出不回写；最后追加当前副本并 FIFO 截断。首帧直接返回 X。 | 正式代码直接导入这一原包核心。bank 保持源槽身份与自动微分关系；clone 有内存/复制成本，FIFO 容量也不等于训练反向激活的总上限。 |
| 同一用户核心的 `current_only` | 逐槽将历史输入替换为当前 X，仍执行同数目的 align/fuse 与均值。40 帧、K=4 为每窗口每模式 150 对算子。 | 这是新增 M1 历史的对照，原 DeepPro-Plus 时空前端/TPro 仍有时间信息，不能称整模型单帧基线。 |

缓存位置“属于某一帧”不意味着其数值只含该帧原始观测：StreamVGGT 的深层隐状态可已融合过去；本项目 X 也来自既有时空前端。K=4 只限制新增 bank 的源索引年龄，不限制整个检测器的原始时间感受野。两种实现都不能被简化为“保存四张独立单帧观测”。

可用于研究计划的准确表述：参照 StreamVGGT 显式保存与读取分离的实现方式，在固定 DeepPro-Plus/RFR 算子上，用用户提供的有限源特征 bank 检验持续漏检恢复。缓存、FIFO、均值和来源索引本身均不是独有新算子；该试验不预先建立恢复能力或新颖性。
