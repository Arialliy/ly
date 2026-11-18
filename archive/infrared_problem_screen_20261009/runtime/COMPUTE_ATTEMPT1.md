# DeepPro 官方权重推理：固定真实验证片段

本地 Docker，复用不可变镜像，不安装新依赖。宿主GPU0共享使用；不停止其他进程。

环境规格 env-spec.json；canonical SHA256 ac6c0ce74e36b5fa36f9c041a77d46fc4f82cab8fc81fc5866880857d73f40ef.

数据：Anti-UAV-RGBT val 字典序首个完整配对序列，前160帧。原始512×640，20fps，框标注。模型：官方RGB-T DeepPro-Plus。属于跨数据工程诊断，不是论文指标复现；目标约数十像素，不能代表1–3像素点目标。

seed=0仅为本地固定推理设置。官方训练seed函数46没有被调用，官方test不设seed。模型与checkpoint内容不改。

## 1. CUDA见证命令（原样执行）

```bash
docker run --rm --gpus all --user 1004:1004 --cpus 4 --memory 12g -e CUDA_VISIBLE_DEVICES=0 -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -v /home/ly:/home/ly --entrypoint python ppg/ultralytics:8.3.203 /home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/runtime/gpu_witness.py
```

预期WITNESS_OK，torch2.7.0+cu128，RTX5090，shape[8,8]，sum46.30007553100586。

## 2. 实际模型推理命令（原样执行一次）

```bash
docker run --rm --gpus all --user 1004:1004 --cpus 4 --memory 12g -e CUDA_VISIBLE_DEVICES=0 -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -v /home/ly:/home/ly --entrypoint python ppg/ultralytics:8.3.203 /home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/runtime/infer_checkpoint.py --source /home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/runtime/deeppro_audit_source --checkpoint /home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/runtime/deeppro_audit_source/log/sem_seg/RGB-T__2025-01-14_15-27__SoftLoUloss_DeepPro-Plus_DataL40/checkpoints/best_model.pth --input-npy /home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/runtime/first_val_clip/raw_frames.npy --output-npz /home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/runtime/first_val_inference.npz --model-type DeepPro-Plus --normalization RGB-T --device cuda:0 --seed 0
```

预期严格加载成功，写first_val_inference.npz，概率形状[160,512,640]且有限，覆盖160帧。会读取未来帧，不宣称在线性能。运行日志应由执行者保存到doc_follow_inference.log。约束：不修改命令或补安装；出错保留异常交由父agent修复。
