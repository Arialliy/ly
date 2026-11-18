# 本候选与近期机制的有限关系

TEP-SAM作者原文 https://arxiv.org/html/2606.27655v1 在冻结SAM适配中构造时序线索并注入主表示。本候选只借鉴先构造可辨别线索的思路；没有移植其网络、提示、辅助监督或预测。此前作者权重在本开发协议未通过预算，不能称为本候选有效的实验证明。

TDCNet作者论文 https://ojs.aaai.org/index.php/AAAI/article/view/37385 研究时序差分；时间差分本身不是新贡献。本轮N1是必须面对的简化控制。局部median/MAD和残差读出也不能仅凭组合宣称新颖性。当前唯一待检验命题是局部变化参照是否在同样训练/读出下提供预算内的额外恢复。

这不同于已停止的BC-TPro深层背景关联相减，也不同于E2在深层特征上检索和双向核验跨帧对应；区别支持独立控制实验，不自动支持机制或论文贡献。

实现兼容性：实际使用Torch2.7，官方确定性文档 https://docs.pytorch.org/docs/2.7/generated/torch.use_deterministic_algorithms.html 说明CUDA kthvalue在严格确定性下不支持。因此在正式GPU运行前改用精确稳定排序取两个中央项，保留数学定义和确定性设置；未按性能结果改变方案。
