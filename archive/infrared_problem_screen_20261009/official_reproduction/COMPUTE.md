# Full official DeepPro-Plus NUDT-MIRSDT inference reproduction

Spec canonical SHA256: `de8a7eb93aac154354646ae9b59c7ba4086c2ad1d8a704894c9c1de86a4d6d86`. Provider: local Docker, user1004:1004, shared RTX5090; no other jobs stopped. New isolated dependency layer atop verified local base image03c5b4808c64. Source remains read-only. Full original FP32 frames, no resize/tile/AMP, original40-frame windows and max fusion.

Seed46 is explicitly activated from the source declaration; the official checkpoint training seed is unknown. This validates inference, not retraining.

## 1. Imports and CUDA witness (run verbatim)

```bash
docker run --rm --gpus all --user 1004:1004 --cpus 4 --memory 12g -e CUDA_VISIBLE_DEVICES=0 -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 -e MPLCONFIGDIR=/tmp/deeppro-matplotlib -v /home/ly:/home/ly -v /home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/official_deeppro:/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/official_deeppro:ro --entrypoint python deeppro-official-eval:20261009 /home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/official_reproduction/environment_witness.py
```

Expected imports succeed, torch2.7.0+cu128, NumPy2.2.5, WITNESS_OK, RTX5090, shape[8,8], seed46, sum42.68129348754883.

## 2. Complete first official sequence through original test.main (run verbatim once)

```bash
docker run --rm --gpus all --user 1004:1004 --cpus 4 --memory 12g -e CUDA_VISIBLE_DEVICES=0 -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 -e MPLCONFIGDIR=/tmp/deeppro-matplotlib -v /home/ly:/home/ly -v /home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/official_deeppro:/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/official_deeppro:ro --entrypoint python deeppro-official-eval:20261009 /home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/official_reproduction/run_official_shard.py --gpu 0 --shards 1 --shard-index 0 --max-sequences 1 --output-dir /home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/official_reproduction/doc_follow_probe_v4
```

Expected strict official checkpoint load; native100-frame Sequence85 completes; doc_follow_probe_v4/COMPLETE.json exists, original_counts in official_counts.npz and full probability output shape[100,200,300]. There are three windows [0,39],[36,75],[60,99]. No full-paper metrics claim from this single-sequence gate.

The wrapper imports unchanged official test.py and invokes main. Only source seed activation, narrow NumPy safe loading globals, disjoint sequence selection, output relocation, recording hook and deferred full-order summary are adapters. Official post-forward ShootingRules is unchanged. The recording hook does not modify outputs.

The fresh reviewer runs both commands without fixing them. Save stdout/exit codes to doc_follow.log and doc_follow_result.json. If anything fails, report the exact failing step.

## Recorded first dependency failure

First fresh doc-follow failed at matplotlib import: missing cycler. Metadata audit also found contourpy, fonttools, kiwisolver and python-dateutil absent in base. The new declarative phase pins all five, then rebuilds the isolated image. Prior failed logs are retained as attempt1. No scientific inference was run in that failed attempt.

## Local arithmetic policy

The wrapper additionally sets cudnn.benchmark=False, cudnn.deterministic=True, torch.use_deterministic_algorithms(True), CUDA matmul TF32=False and cuDNN TF32=False. These are explicit local reproducibility settings beyond seed activation; original source does not record the author runtime settings. All operations remain float32. Numerical differences from author logs will be reported rather than concealed. Existing experiment symlinks must resolve to the pinned source before loading; metadata hashes the actual loaded checkpoint.

## Recorded adapter repair

Attempt2 imports/CUDA witness passed, but model creation failed before inference: replacing the detector class name broke the original explicit super(detector,self). The adapter now preserves the class and wraps its initializer only to register a read-only forward hook. Pinned source text is unchanged. Source mount is now explicitly :ro. Attempt3 uses a fresh probe output directory; previous logs retained.

## Recorded metadata repair

Attempt3 completed native Sequence85 inference and original metric counts, but writing provenance failed because git CLI is absent inside the image. Metadata now uses SOURCE_SNAPSHOT.json, generated on the host from the clean pinned repository. The runtime verifies every tracked SHA256 and the source file set before and after inference. It records this validation method explicitly; it does not claim to run git inside the image. Attempt4 uses a fresh output directory.
