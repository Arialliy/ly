# Frozen development inference on user-allocated GPU 2 / 3

This file adds resource-specific development invocations after the science protocol was frozen. It does not modify the original native documentation, frozen model runner, model code, checkpoint, data, or environment. User allocation is physical GPUs **2 and 3**; other processes are never stopped. Resource occupancy is rechecked immediately before each job. If GPU 2 remains occupied, use GPU 3 sequentially after the native job exits.

Prerequisites: root has reviewed the fresh GPU3 native doc-follow PASS; `native_check/COMPLETE.json` reports `PASS_NATIVE_RUNNING_CHECK_ONLY` and exact 100-frame metric input agreement. `PROTOCOL.json` SHA must be `9c73da144a4a4743c2cc36b6c57a33298cacf5b15468a03e0214f56e18b73374`. This launch script mechanically verifies the native receipt, immutable image and all source hashes frozen in the protocol before starting a model.

Preferred sequential commands, each after the previous one exits successfully:

```bash
bash /home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/temporal_diagnosis_20261010/tep_sam_reference_20261012/runtime/launch_development.sh DJI_0026_2 3
```

```bash
bash /home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/temporal_diagnosis_20261010/tep_sam_reference_20261012/runtime/launch_development.sh DJI_0067_2 3
```

The only permitted alternative is replacing the last resource argument with `2` if physical GPU 2 has <500 MiB memory occupancy. No model parameter, input, precision, threshold, or scoring change accompanies a different allocated GPU. If neither is free, wait; do not kill another task or change the batch/image size. Root controls launch authorization and sequencing.

Each command writes a resource receipt under `runtime/RESOURCE_RUN_<sequence>.json` and the complete log `runtime/inference_<sequence>.log`. The output directory must not exist. The runtime creates `predictions/<sequence>/probabilities.npz` and `auxiliary_probabilities.npz`, each FP32 `[160,512,640]`, and `FRAME_ACCESS.json` / `COMPLETE.json`. Main and auxiliary use the frozen, separately documented output chains. One author model infer per current frame, 160 per clip, with the author's original internal computations preserved. No training or GT prompts.

Each job: 4 CPU, 20 GiB host RAM, 2 GiB shared memory, one visible GPU, seed 46, FP32, no autocast/TF32, no cropping or tiling, batch 1. Source and all prior experiment inputs stay read-only; outputs are isolated in this new stage. Same immutable `deeppro-official-eval:20261009` and environment as the independently verified native check. No new environment package or native rerun is required for this resource wrapper.
