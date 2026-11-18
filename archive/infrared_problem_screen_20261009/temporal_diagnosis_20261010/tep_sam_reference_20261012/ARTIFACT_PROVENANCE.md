# TEP-SAM 作者工件与运行口径核对

本轮只取得作者 NUDT-MIRSDT 权重，使用固定作者源码；不训练、不选择第二套权重、不修改作者模型或指标代码。

## 工件

- 作者仓库：<https://github.com/cdh8285/TEP-SAM>
- 固定 commit：`c5165cfa9139e7d72f1fc962ddc398d31d59093e`，提交日期 2026-05-26。
- 原 README 指向的 NUDT 权重：<https://drive.google.com/file/d/1_hbtkHZiVZeT-h9chdUagmAPfsTBlqoa/view?usp=sharing>。
- 下载文件：`weights/model_NUDT-MIRSDT_SAM1-B.pth`，382,496,611 bytes。
- SHA256：`1125ea28b4dfe0745be1f33500f4e5463f1cbd8e837a3c1c733b34bb0de9e6f0`。
- [CPU 严格加载收据](runtime/CPU_STRICT_LOAD.json)：472 state keys，95,576,030 参数，456 参数张量，missing/unexpected 均为空，FP32。辅助头全部 16 state keys 存在，包括 BatchNorm buffers。
- 作者源码保持干净；53 个 git tracked files 的哈希见 [源码清单](ARTIFACT_PROVENANCE_SOURCE_MANIFEST.json)。下载 HTTP 回执另存于 `weights/download_receipt.json`。

`test.py:22–28` 按 `state_dict`、`model` 顺序解包后 `strict=True`。本文件为完整 OrderedDict，严格加载成功。配置中 `sam_checkpoint` 字段没有被 `test.py` 或模型构造读取；无需再下载 SAM 基础权重。README 通用示例中的 `configs/sam-vit-b-token.yaml` 不存在，实际 NUDT 配置为 `configs/sam-vit-b-token-NUDT-MIRSDT.yaml`。

## 实际输入输出

[原生数据包装](author_source/datasets/seqwin.py) 的 test 类使用每帧中心 `t-3,…,t+3` 共 7 帧，在当前序列边界内 clamp；不会读取另一个序列。先 PIL `convert('RGB')`，整幅变形为 1024×1024，ToTensor 后按 ImageNet mean `[0.485,0.456,0.406]` / std `[0.229,0.224,0.225]` 标准化；不裁剪目标、不保持长宽比。真正使用的是包装器的变换，YAML 中另一套 `data_norm` 未用于 test。mask 全部非零值二值化，原图 GT 保留用于评价。

`models/sam_withToken.py:254–302` 的 `infer` 返回原始 logits，函数注释中的 binary masks 不是实际数值定义。decoder 的 256×256 logits 先 bilinear 到 1024×1024，`align_corners=False`；[test.py:57–80](author_source/test.py) 再执行 sigmoid，然后对概率图用 area 插值还原原图，最后严格 `>0.5`。不能交换 sigmoid 与插值。

原生测试入口 `boxes=None`，不输入标注或人工点提示。完整开发评估每个当前帧只有一次中心预测，不构造假的 O/S 两套 TEP-SAM 安排。

## 已训练辅助输出

`image_encoder.prompt_generator.middle_mask_decoder` 是 16→32→16→1 的卷积/BN/GELU 预测头。`image_encoder.forward` 第三个输出就是 256×256 原始 `middle_mask` logits；`infer` 丢弃它。训练源码 `models/sam_withToken.py:304–318` 对它施加 BCEWithLogits + Dice，各系数为 1。检查点确实包含该头参数和非零 BN 更新计数，因此可以同次前向用只读 hook 导出，不创建或训练新读出器。

作者没有提供辅助头原生评价。本阶段已预登记项目辅助输出还原：**256 logits → bilinear 1024（align_corners=False）→ sigmoid → area 原图**，与主头处理顺序相同；只用这一种顺序，不进行选择。它是项目诊断适配，不命名为作者辅助头官方成绩。

## 原生指标需单列的实现口径

| 打印名称 | 当前作者代码的实际计算 | 源码位置 |
|---|---|---|
| IoU | 每帧交并比的算术均值 | `test_utils/metric_new.py:68–85` |
| nIoU | 累计交集除累计并集 | `test_utils/metric_new.py:112–131` |
| PD | 8 连通组件，质心距离 `<3` 的逐目标贪心匹配 | `test_utils/metric_new.py:23–60` |
| FA | 未匹配组件面积 / 总像素数；未匹配按面积值是否属于匹配面积列表判断 | `test_utils/metric_new.py:52–60` |

最后一项意味着，与某个匹配组件面积相同的未匹配组件可能被排除；这是当前公开实现的可核查性质。本阶段原生检查保留该实现，不修指标后声称运行了原生入口。项目诊断继续独立使用冻结的组件峰点匹配、U、框外像素与未匹配组件面积，不将 U 写成官方 Fa。

原生任务固定为 **Sequence85 全 100 帧运行检查**，不是完整官方划分的发布权重推理复核，也不是从头训练复现。该子集结果不能与 README 全测试成绩直接比较。

详尽机器可读参数及源码行见 [ARTIFACT_PROVENANCE.json](ARTIFACT_PROVENANCE.json)。
