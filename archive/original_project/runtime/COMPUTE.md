# DeepPro 官方权重推理：固定真实验证片段

本地 Docker，复用不可变镜像，不安装新依赖。宿主GPU0共享使用；不停止其他进程。

环境规格 env-spec.json；canonical SHA256 644494fe6dcff11df0ee5550b3007531c71661043de5c1a58756b026929837c0.

数据：Anti-UAV-RGBT val 字典序首个完整配对序列，前160帧。原始512×640，20fps，框标注。模型：官方RGB-T DeepPro-Plus。属于跨数据工程诊断，不是论文指标复现；目标约数十像素，不能代表1–3像素点目标。

用户要求遵循源码种子值：本地显式固定seed=46，对应官方train.py:33默认值；官方函数未被原入口调用，故不能确认checkpoint训练实际seed。模型与checkpoint内容不改。

## 1. CUDA见证命令（原样执行）

```bash
docker run --rm --gpus all --user 1004:1004 --cpus 4 --memory 12g -e CUDA_VISIBLE_DEVICES=0 -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -v /home/ly:/home/ly --entrypoint python ppg/ultralytics:8.3.203 /home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/runtime/gpu_witness.py
```

预期WITNESS_OK，torch2.7.0+cu128，RTX5090，shape[8,8]；新seed46见证数值由本次运行记录。

## 2. 实际模型推理命令（原样执行一次）

```bash
docker run --rm --gpus all --user 1004:1004 --cpus 4 --memory 12g -e CUDA_VISIBLE_DEVICES=0 -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -v /home/ly:/home/ly --entrypoint python ppg/ultralytics:8.3.203 /home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/runtime/infer_checkpoint.py --source /home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/runtime/deeppro_audit_source --checkpoint /home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/runtime/deeppro_audit_source/log/sem_seg/RGB-T__2025-01-14_15-27__SoftLoUloss_DeepPro-Plus_DataL40/checkpoints/best_model.pth --input-npy /home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/runtime/first_val_clip/raw_frames.npy --output-npz /home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/runtime/first_val_seed46.npz --model-type DeepPro-Plus --normalization RGB-T --device cuda:0 --seed 46 --tile-core 128 --tile-halo 16 --verify-tiling
```

预期严格加载成功，写first_val_seed46.npz，概率形状[160,512,640]且有限，覆盖160帧。会读取未来帧，不宣称在线性能。运行日志应由执行者保存到doc_follow_seed46.log。约束：不修改命令或补安装；出错保留异常交由父agent修复。

## 已诊断的首次失败与修正

首版整幅 FP32 conv3d 请求21.09GiB连续工作空间，在共享GPU0上OOM。原记录保留doc_follow_inference.log。

当前空间分块核心128×128、halo16，仍保留原始512×640和完整40帧。源码感受野半径9，无全局空间操作；各层保留原padding，拼接核心区域。它是内存执行适配，不是新检测方法。不使用缩放或半精度。

实际推理前，必须通过真实96×112输入的整体/分块FP32对照，rtol=atol=1e-4，记录最大logit、概率误差及0.5阈值差异。若对照失败，程序立即停止。

## Seed correction

Earlier seed0 runs remain engineering history. User instruction supersedes that choice: all new runs use source-declared46. Source checkout is unchanged. This invocation writes a new seed46 archive, preserving previous output.
