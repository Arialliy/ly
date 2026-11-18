# 第一轮官方权重诊断完成（seed46）

## 种子与复现边界

用户要求采用源码种子。DeepPro `train.py:33` 定义 `seed_everything(seed=46)`，原训练入口未调用，训练worker还使用时钟种子。当前推理wrapper显式固定Python、NumPy、PyTorch CPU/CUDA为46，并固定确定性设置；未修改官方源码和权重。不能声称官方checkpoint已确认按46训练。seed0历史输出保留，不作为正式对照。

官方commit `8fa1a68b94eb22e94ccd0529e6c5ceccdaa7ec28`，完整checkout保持clean。

## 实际完成

| 输入 | 权重与设置 | 结果与用途 |
|---|---|---|
| Anti-UAV-RGBT val固定首序列前160帧 | 官方RGB-T DeepPro-Plus；seed46 | 推理完成，原尺寸512×640；与历史seed0同权重eval输出最大差0。工程检查，不是训练种子比较。 |
| NUDT-MIRSDT官方首测试序列Sequence85前40帧 | 官方NUDT DeepPro-Plus；seed46 | 推理完成；原尺寸200×300；局部mask IoU=0.992794，只有这一窗口，不是论文全量结果。 |
| VT-Tiny-MOT额外开发序列DJI_0024_2前160帧 | 官方RGB-T DeepPro-Plus；seed46 | 原尺寸512×640，保留1920条原标注。属于真实开发诊断，不保证与训练场景独立。 |

NUDT标准Pd/Fa尚未计算：原`ShootingRules`依赖的skimage在当前镜像缺失，未用替代实现冒充官方评估。原始像素输出和mask均已保留。

## 对问题选择的实际影响

预先只按标注选定的目标3300000，是连续160帧11×11船体框，标注中心位移中位数与90分位均为0。其每帧框内最大概率均高于0.9998。**这不支持“标注位置近静止就一定没有响应”。** 框内响应也不是经过一对一匹配的检测召回。

同一片段中另有5个原始ID在各自160帧内的框内最大概率均未超过0.5，同时也有间歇响应和持续强响应对象。全部逐帧记录保留，不只展示失败案例。现阶段不能把这些差异归因于运动、对比度、目标大小、配准或记忆不足；这些因素尚未控制。

原始标注包含一个仅2帧的额外track_id，未擅自合并ID。此数据没有运动→停止→离开的完整事件标签；当前不能报告停动恢复或退出残影指标。

下一步应先核对全部目标与标注，再比较同场景强响应、弱响应目标的尺度、局部对比度和时序强度曲线；需要单帧与简单时序对照后，才能把现象缩为具体方法问题。暂不增加记忆模块。

## 显存执行适配

整幅FP32确定性推理首次OOM。根据源码确认19×19局部空间感受野，采用128核心、16像素halo，同时保留40帧完整时间维度。模型、分辨率、归一化均未改变。

三个输入均通过整块/分块数值检查。最大概率差分别为Anti-UAV约8.79e-6、NUDT约5.96e-7、VT约2.73e-11；0.5阈值差异均为0。源码核查与独立doc-follow均完成。当前GPU共享、空间分块下的速度不当作论文FPS复现。

## 工件

- `runtime/SEED_POLICY.json`：当前固定种子与原代码调用事实。
- `runtime/SEED46_PROBE_RESULTS.json`：三组结果与解释边界。
- `runtime/vt_small_seed46_frame_responses.csv`：完整逐目标逐帧响应。
- `runtime/doc_follow_seed46_result.json`：独立原样命令检查，PASS。
- `DATA_ENTRY_AUDIT.md`：数据来源、划分、选择规则、ZIP分段提取记录。
- `runtime/public_subset_vt/` 与 `runtime/public_subset_nudt/`：可重用公开样本。

本轮没有训练、没有新模型性能提升结论。
