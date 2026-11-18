# 路径矩方案：先验工作审阅（2026-11-18）

审阅对象：`infrared_video_detection_research_plan.md`。只读评估；未修改原方案，未训练，未读取或使用 confirmation 数据。路线：novelty-check 的 fresh-agent 文献核验分支；**same-family / provisional**。这是早期新颖性关口，不是正式确定研究主线或论文录用预测。

## 判断

**PROCEED WITH CAUTION，5.5/10：通过进入先导实验的查新关口。** 本次没有找到一篇已核实论文完整包含“学习局部跨帧路径分布 → 时间化加性路径特征 → 精确二阶矩 → 带符号低秩二次期望检测分数”这一具体实现，不能据此宣称首次，也没有理由 ABANDON。

CAUTION 的具体原因是：期望半环已覆盖核心递推；Magic Moments 已把结构化输出的均值与协方差用于判别学习；TCPNet 及 A²M²-Net 已组合时间建模、二阶统计和对齐。因此，仅以“概率对应＋时间编码＋二阶矩＋低秩二次头”为新机制，容易被审稿人判断为已有数学和模块的直接组合。值得先导的是更窄且可验证的新发现：**在相同跨帧对应边缘统计下，路径联合关系是否含有均值融合和普通时序二阶池化无法替代的弱目标判别信息。** 尚无训练证据不取消做先导的资格；原文数值 toy 只支持递推算术一致。

## 核心主张与最接近的对应

| 主张 | 最强已核近邻 | 仍可成立的具体差异 |
|---|---|---|
| 不枚举路径求一、二阶矩 | Li & Eisner 2009；Ricci et al. 2008 | 新意不能落在递推本身；应落在视频对应随机变量、特征和检测判别器的联合设计 |
| 在结构分布上使用二阶判别信息 | Magic Moments；概率 STN | Magic Moments 用全结构输出空间的评分矩设计训练目标，推断仍为线性结构评分的 argmax；本方案对终点条件下学习的对应分布取二次检测分数期望 |
| 视频中的跨时二阶统计与对齐 | TCPNet 2021；A²M²-Net 2025 | 它们建模特征/空间样本二阶统计；本方案统计对象是同一终点的不同空间对应路径，含同一路径内的跨时乘积 |
| 可学习软结构和协方差高效计算 | Structured Attention Networks 2017；d²p 2026 | 已有工作让结构边缘量参与端到端学习，协方差出现在导数中；这里拟把路径二阶量直接作为前向检测统计。不能声称首次可微概率路径推理 |
| 红外“二阶运动感知” | IRDINO 2026 | 已核论文及代码是时间二阶差分，和路径不确定性的二阶矩不是同一个量 |

## 最接近文献的具体核对

**1. IRDINO：原方案的关键未知项已补齐。** CVF PDF 在 web 工具返回 403，但直接 HTTP 成功取得 11 页论文；PDF 为图像式，已渲染并阅读第 4 页（印刷页 8411）§3.3。它明确采用二阶时间差分，并给出 `|2 f_t − f_(t−1) − f_(t+1)|`。公开仓库 `engine/deim/irdino_modules.py` 中 SMAM 的第 268–279 行取前两项特征并计算 `feat − 2.0 * prev1 + prev2`，且检查序列 ID 与帧间隔；第 284 行也明确描述为 second-order differences。论文的中心差分绝对值与代码的后向差分形式不完全一致，不据此宣称代码已忠实复现全文。二者都足以排除把这里的“二阶”直接等同于 `Cov_q(v_path)`。检查的仓库树 SHA 为 `187f762100c9bed7dcd74f8bfcbfb6d665fe6848`。[CVF 论文](https://openaccess.thecvf.com/content/CVPR2026F/papers/Xu_IRDINO_Adapting_DINOv3_with_Second-Order_Motion_Awareness_for_Moving_Infrared_CVPRF_2026_paper.pdf)；[代码](https://github.com/TR797963/IRDINO/blob/187f762100c9bed7dcd74f8bfcbfb6d665fe6848/engine/deim/irdino_modules.py#L261)

**2. Magic Moments for Structured Output Prediction，JMLR 2008，是应补充的强基础近邻。** §3 明确对全部结构化输出的线性评分计算均值与协方差，式 (4) 给出特征协方差，随后用 DP 避免枚举；应用包括序列对齐、标注、RNA 结构。其定义使用 `1/N` 对输出空间平均，训练目标包括 Z-score 与 SODA，预测仍为 `argmax_y θᵀφ(x,y)`。这先于 2009 期望半环，直接削弱“第一次用路径矩完成判别”的笼统说法；但它没有因此自动覆盖本方案的“学习视频条件对应权重＋终点检测期望二次分数”。[JMLR 论文](https://jmlr.csail.mit.edu/papers/volume9/ricci08a/ricci08a.pdf)

**3. d²p: Structured Soft Attention Is All You Need，ICML 2026，是最新可微结构推理边界。** 官方摘要明确将软 DP 的边缘量作为导数，推导 Hessian-vector products、cross-Jacobians 及 Gibbs 分布下的协方差表达，并学习对齐/编辑/解析参数。其摘要所列实验是蛋白结构对齐与句法解析。和本方案的可核差异是“学习结构边缘量所需的二阶导数”与“直接用于前向检测的路径特征二阶矩”，不应仅因出现 covariance 就视为同构。本轮核验到官方摘要，尚未逐页核全文；该差异为 provisional。[PMLR 官方记录](https://proceedings.mlr.press/v306/mogilevsky26a.html)

**4. A²M²-Net，2025，是比普通协方差池化更强的视觉近邻。** §3.3 的式 (2) 是空间位置上的外积平均；式 (3) 先进行时间卷积及可变形空间卷积，偏移由时间差分产生；§3.2/3.4 再以 EMD 比较 query/support 视频的二阶描述子。它已经把轨迹启发的采样、二阶统计和自适应对齐放在同一个视频识别方法里，必须纳入组合创新论证；但其统计和对齐顺序未形成对同一视频的随机对应路径做二次分数边缘化。[作者预印本全文](https://arxiv.org/html/2509.17638v1)

**5. TCPNet，NeurIPS 2021，不应被弱化成“逐帧方差”。** 式 (5) 已含投影后的帧内外积及跨帧交叉乘积，式 (8) 还用前三帧的空间注意力校准特征。公平基线应保留它的跨时交叉项与注意力，不能只拿逐帧二阶统计代表该工作。其随机性来自视频特征样本，和本方案固定终点后的对应路径随机性可明确区分。[论文全文](https://proceedings.neurips.cc/paper_files/paper/2021/file/70efdf2ec9b086079795c442636b55fb-Paper.pdf)

**6. 概率 STN / 期望半环仍是必引基础。** P-STN 已对变换分布采样并对分类预测平均，因此“多个对齐假设分别预测后融合”的思想不新；本方案的可核差异是局部路径图和特定二次分数的精确矩计算。Li & Eisner 的二阶期望半环已经给出紧凑结构上的二阶期望计算范式。[P-STN 全文](https://proceedings.mlr.press/v180/schwobel22a/schwobel22a.pdf)；[期望半环论文](https://aclanthology.org/D09-1005/)

**7. 仍需下一轮全文排除的传统检测近邻。** Kerekes & Kumar 的 *Multiple target detection in video using quadratic multi-frame correlation filtering*（2008）在作者机构摘要中明确把概率运动模型作用于阈值前软相关输出，再纳入二次相关滤波器。它非常接近“概率运动＋二次检测”的宽表述；当前只核实摘要，无法判定其二次项是否作用于整条随机路径的加性特征，不能把它写成已证明同构或已彻底排除。[ORNL 原始机构记录](https://www.ornl.gov/publications/multiple-target-detection-in-video-using-quadratic-multi-frame-correlation-filtering/)

## 最值得先做的一项决定性实验

**固定表示和对应边缘分布的路径依赖消融。** 在相同编码器、时间基函数、对应图、二次头和预算下，比较完整 `E_q[g(v)]`、`g(E_q[v])`，以及用同一终点条件边缘分布 `q_t(x_t | X,end)` 的乘积分布替换完整路径联合分布的二阶版本。第三组保留每帧均值与方差，却移除跨时依赖；作为机制诊断它可以包含非局部跳跃，不应冒充合法跟踪基线。同时给充分训练、同等预算的 TCP 风格跨时二阶基线和均值 MLP 一次公平比较。

目标证据是：完整联合版本的优势在对应歧义较高的真实视频段明显；把联合关系替换为边缘乘积后，这一优势消失，而非仅仅受益于时间编码、额外容量或普通二阶池化。共享权重删项能作定位，但不足以替代分别训练的公平比较。原 toy 不承担这一结论。如果该实验给出支持，就是可发表的新发现候选；如果无支持，应调整机制假设而非继续堆模块。本审阅不启动该实验。

## 可采用的一句定位

“我们研究固定终点的跨帧对应歧义，将已知结构化矩计算用于时间化路径特征的二次期望判别，并检验同一路径的跨时依赖是否提供均值对齐及普通时序二阶池化所缺失的弱目标证据。”

## 搜索和核验边界

本轮检索覆盖红外、视觉跟踪、结构化预测、软时间对齐、期望半环、二次多帧滤波与二阶视频识别，使用了 `path covariance`、`second-order soft alignment`、`expectation semiring vision`、`quadratic path marginalization`、`moment propagation correspondence` 等多种表述，并检索了 2025–2026 及 2026-05-18 后的 arXiv 相关条目。检索未发现完全同名方法不构成创新性证据。官方来源确认了 d²p 的 2026 会议信息；未完成所有 ICLR/NeurIPS/ICML 近两年论文的穷尽筛查。所有新颖性结论保持 same-family provisional。

归档：`.aris/traces/novelty-check/2026-11-18_path_moment/001-prior.*`；IRDINO PDF 与第 4 页图在同目录 `sources/`。
