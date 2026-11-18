# 红外视频路径矩研究计划审阅

日期：2026-11-18。对象：[原研究计划](../infrared_video_detection_research_plan.md)。按用户提供文件的“坚持视频检测”范围审阅，原计划未修改，未改变全局研究方向状态，未启动模型训练。

**判断：可以作为视频机制先导候选，暂不能认定创新性已经充分。** 固定图上的数学递推成立；已有文献没有在本轮被核实为完整同构，但通用矩计算和二阶结构学习的先例比原文列出的更多。剩余贡献必须由“路径联合关系本身提供额外检测信息”来承担。查新结论为PROCEED WITH CAUTION，same-family provisional。

## 需要优先修正的内容

1. **时间编码的具体形式。** 原文223行允许“包含相对时间编码”，但h_t=F_t+p_t或简单拼接后求和仍会删掉顺序。应明确时间与内容的交互，例如h_t=B_tF_t或b_j(t)F_t；这只是补全原机制，不应增加为独立创新点。
2. **检测监督与多路径保留的关系。** 对固定路径分数，期望logit对q线性，而BCE对正样本单调递减、对负样本单调递增。若单样本q可自由选择，正例可集中于最高分路径、负例可集中于最低分路径。共享网络实际未必塌缩，但仅凭检测监督不能假设它会主动保留不确定性。必须分开固定q机制实验与端到端训练结果，不预设加熵正则就能解决。
3. **低秩实现的线性分量。** 原文327–368行只写R个二次投影，不能一般恢复任意wᵀμ。可额外递推一条w投影的一阶累积，最终s=b+l_T+Σλ_r n_T^r；无需存完整μ。
4. **将“保边缘、去依赖”设为决定性实验。** A3优于g(μ)只证明总方差信息有用，其中包含逐帧方差。必须另比较同一终点条件边缘的乘积分布q_ind=∏q_t，它保留逐帧均值和方差，却去掉跨时协方差。只有完整联合版本仍有增量，才支持“路径联系”这项主张。q_ind可能含原图不允许的跳跃，应作为统计信息消融，不称为合法tracker。
5. **明确冻结和公平训练的不同结论。** 固定同一encoder/P后分别训练head，用于识别统计信息；各方法端到端训练，用于比较整体方案。不能把后者也描述成唯一差别是对同一分布的计算顺序。还应有保留逐帧对齐特征、按时间拼接的强均值基线，避免过早压缩削弱对手。

完整推导、行号和滑窗缓存条件见[数学审查](PATH_MOMENT_MATH_REVIEW_20261118.md)。

## 查新中的实质更新

- **IRDINO的未知项已补齐。** 论文§3.3使用二阶时间差分；公开代码也计算相邻历史特征的二阶差分。它不是不同对应路径的协方差，因此不能仅凭“二阶”这个词判定重合。论文的中心差分与代码的后向差分并不完全一致，基线复现还需核对因果输入条件。[论文PDF](https://openaccess.thecvf.com/content/CVPR2026F/papers/Xu_IRDINO_Adapting_DINOv3_with_Second-Order_Motion_Awareness_for_Moving_Infrared_CVPRF_2026_paper.pdf)；[固定版本代码](https://github.com/TR797963/IRDINO/blob/187f762100c9bed7dcd74f8bfcbfb6d665fe6848/engine/deim/irdino_modules.py#L261)
- **应补Magic Moments（JMLR 2008）。** 它已经通过动态规划求结构化输出的评分矩，并用均值/协方差构造判别训练目标。它的结构空间平均与本方案学习的视频条件路径分布不同，预测形式也不同；但“用路径矩做判别”不能再作为宽泛首创。[原论文](https://www.jmlr.org/papers/volume9/ricci08a/ricci08a.pdf)
- **应补d²p（ICML 2026）与A²M²-Net（2025）。** 前者覆盖可微软结构及协方差导数；后者已经结合时间建模、可变形采样、二阶统计与视频对齐。它们不自动覆盖本方案，但约束组合创新的表述。[d²p官方记录](https://proceedings.mlr.press/v306/mogilevsky26a.html)；[A²M²-Net全文](https://arxiv.org/html/2509.17638v1)
- **仍有一项重要全文缺口。** Kerekes与Kumar的2008年二次多帧相关滤波已经结合概率运动与阈值前软检测输出。当前核到机构摘要，尚不能确定其二次项是否跨整条随机路径。正式立项前应继续核对，不能把“概率运动＋二次检测”视为已经排除前人的贡献。[ORNL记录](https://www.ornl.gov/publications/multiple-target-detection-in-video-using-quadratic-multi-frame-correlation-filtering/)

原文对期望半环/P-STN的承认是必要的；TCPNet基线也应保留其已有跨帧乘积，不能弱化为逐帧方差。详见[先验审查](PATH_MOMENT_PRIOR_REVIEW_20261118.md)。

## 这次实际验证了什么

新写独立小图复现，20个随机种子、3个终点、含边界掩码。矩递推与逐条路径枚举的最大绝对误差为1.78×10⁻¹⁵；递推分数的有限差分与路径解析梯度最大误差约6.05×10⁻¹¹。

这是新的双精度标量参考检查，不是对原表具体随机实例的追溯，也没有验证PyTorch自动微分、GPU运行时间或真实视频检测。脚本显式补全独立线性项，不能把它说成原文所有实现细节已经完备。

[复现脚本](../experiments/path_moment_review_20261118/check_math.py) · [精确结果](../experiments/path_moment_review_20261118/RESULTS.json)

## 数据与下一决策点

本轮从作者来源复核：IRDST页面列出85个真实场景、40,650帧，并有序列版入口；SatVideoIRSDT README确认20,126帧武汉一号真实数据及另外的模拟组成。原计划把真实部分单列的安排合理。尚未核验实际下载内容、完整划分、身份标注与许可证。DQAligner作者README的track_loss默认关闭描述也已核实。[IRDST官方页](https://xzbai.buaa.edu.cn/datasets.html)；[SatVideoIRSDT](https://github.com/TinaLRJ/SatVideoIRSDT)；[DQAligner](https://github.com/dengfa02/DQAligner_MIRSTD)

下一步应把时间映射、线性项和两个训练协议写清，随后用同一冻结对应图比较：均值判别、完整路径二阶判别、保边缘去跨时依赖的二阶判别，以及充分训练的强时序头。若完整路径版本没有独立收益，不把普通二阶池化涨点解释成路径机制成功。这个条件可检验，且不需要把题目再收窄到某种特殊运动情形。

