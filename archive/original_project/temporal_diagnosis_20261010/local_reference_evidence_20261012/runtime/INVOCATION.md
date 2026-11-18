# Local GPU invocation

Warm reuse immutable image `sha256:eb956873870d737f2b68dff0724b8f6f56f778829d119b0938d29b7abeaa2f92`, canonical environment spec `de8a7eb93aac154354646ae9b59c7ba4086c2ad1d8a704894c9c1de86a4d6d86`. No dependency installation or rebuild. User authorizes GPUs 2/3; launcher requires less than 500 MiB before each launch and never stops external jobs. Original project read-only; new stage writable. Explicit Python entrypoint, FP32 deterministic seed46, no TF32/AMP. No notification service.

Fresh documentation-follow agent executes verbatim once:

```bash
python3 /home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/temporal_diagnosis_20261010/local_reference_evidence_20261012/code/launch_gpu.py --kind preflight --gpu 3
```

This checks seeded CUDA execution, real original checkpoint, all three zero projections, forced nonzero correction direct versus tiled on a training crop, disabled correction equivalence, original freeze and finite new gradients. It performs zero optimizer updates and no development inference. Successful receipt is `checks/GPU_INTEGRATION.json`; log is `runtime/preflight.log`. Existing files are not overwritten. Failure must be inspected and recorded before repair.

Only after gates pass and PROTOCOL_FREEZE.json exists, root launches each fixed training once with `--kind train --condition N0` (also N1/N2) and an idle authorized `--gpu`. Then final-only inference uses `--kind infer --condition N0 --sequence DJI_0026_2` (three conditions and two fixed clips). Same launcher and environment. Evaluator is CPU-only `evaluation_code/launch_score.py`.
