# DeepPro-Plus 官方权重推理复现结果

## 结论

完成 NUDT-MIRSDT 完整官方测试集：20个序列、2,000帧。官方阈值0.5下的TP、GT、FP像素计数与作者附带日志一致，README中的Pd、Fa、AUC达到原报告精度。此次是**官方发布权重的推理复现**；没有从头训练，也没有新方法提升。

| 集合 | 本地 Pd (%) | 本地 Fa (×10⁻⁵) | 本地 AUC | 作者 README Pd / Fa / AUC |
|---|---:|---:|---:|---|
| 完整测试集 | 99.710816 | 2.689157 | 0.9977783921 | 99.71 / 2.69 / 0.9978 |
| SNR≤3 | 99.243856 | 1.653845 | 0.9954679508 | 99.24 / 1.65 / 0.9955 |

- 全集：TP/GT = 1724/1729，FP像素 = 3558。
- 低SNR：TP/GT = 525/529，FP像素 = 870。
- 全局像素IoU = 0.8584927987；作者日志0.858493，按作者显示精度一致。

## 数值一致性的准确边界

全部20×27个TP、GT计数与作者日志完全一致。FP计数540项中20项不同，每项相差1–3个像素，均发生在≤1e-6的极低阈值。官方阈值0.5的逐序列与汇总计数均一致。AUC在作者日志5位小数及README4位小数精度一致。

因此称为**官方指标在原报告精度下复现**，不称逐像素或全部阈值计数完全一致。作者没有提供原始浮点概率图及完整软件栈，微小差异的具体来源尚未逐像素定位；不能把猜测的浮点原因写成已验证事实。

## 固定协议与实际执行

- 源码提交：8fa1a68b94eb22e94ccd0529e6c5ceccdaa7ec28。官方源码文件未改，运行时只读挂载；140个跟踪文件在前后核对SHA256和文件集合。
- 权重：官方 NUDT-MIRSDT DeepPro-Plus best_model.pth，SHA256 e2ad2d5c9b39c3d428f57d8d50edefd6a09f8b45dff53fb9d31134272e5cc83e。
- 数据：官方test.txt逐字节保留，6,000张images/masks/masks_centroid PNG全部CRC、SHA与解码检查通过；各序列保持原尺寸。
- 模型推理：原始FP32整幅输入，无空间切片、无缩放、无半精度；40帧窗口为[0,39]、[36,75]、[60,99]，重叠概率取最大值。
- 直接调用官方test.main、TestIRSeqDataLoader、ShootingRules与writeNUDTMIRSDT_ROC。27档原始阈值、原始AUC积分和IoU规则保持不变。
- GPU0–3各处理5个不重叠序列，最终恢复官方20序列顺序，再用原函数计算完整和低SNR结果。四个容器均exit0，无OOM。
- 本地显式seed46，来自源码seed_everything默认值。作者训练入口没有调用该函数，权重实际训练seed未知。
- 额外明确本地数值设置：cuDNN deterministic、benchmark=False、deterministic_algorithms=True、TF32关闭。不能声称这些设置与作者运行环境完全一致。
- 适配器仅处理种子、NumPy安全加载、日志位置、序列分片、被动输出记录与汇总；保留完整源码、各序列概率、原始计数、命令、容器日志、环境规格和失败修复历史。
- 独立文档跟跑v4已通过环境见证及完整Sequence85验证。独立完整结果复算：PASS_WITH_WARNINGS。全部2,000帧从保存概率及原标注重算0.5计数和IoU，与原输出、作者日志一致；全部27阈值重新汇总一致。保留上述极低阈值FP差异。见 `result_recheck/RESULT_RECHECK.md`。

## 为什么下一步进入真实视频

NUDT-MIRSDT是合成数据。本次证明基准数据上的官方实现和评估已正确衔接，不能代替真实运动变化片段中的可靠性比较。

根据用户2026-10-10的进一步澄清，下一步先分析已复现的DeepPro-Plus：核对真实序列、检查逐帧输出、找到稳定错误，再验证具体时序原因。之后核对已有多帧方法是否已解决该问题，确定值得改进的剩余缺口。单帧不作为当前必做主对照。研究问题与机制仍未预定。

## 工件入口

- `full_test_seed46/RESULTS.json`：完整指标、20项差异的逐项清单。
- `full_test_seed46/full_official_counts.npz`、`official_aggregate_eval.txt`：原始统计和官方汇总日志。
- `full_test_seed46/shard*/probabilities/`：所有2,000帧FP32概率图及窗口覆盖。
- `full_test_seed46/LAUNCH.json`、`CONTAINER_STATUS.json`：实际命令、GPU、镜像ID、退出状态。
- `AUTHOR_REFERENCE_METRICS.json`：作者日志参照，明确与本地结果分开。
- `FULL_TEST_DATA_AUDIT.md`、`EVALUATION_PROTOCOL_AUDIT.md`、`TRAINING_RECIPE_AUDIT.md`：数据、评估、训练来源审计。
- `COMPUTE.md`、`doc_follow_v4_result.json`：环境可执行说明及独立跟跑证据。
- `../RESEARCH_SEQUENCE.md`：用户确认的后续执行顺序。
