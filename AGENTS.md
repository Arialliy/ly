# Project location and execution

- The user requires all project code, data, checkpoints, logs, predictions and results to live under `/home/ly/DeepProPlus_TLFP_Code`.
- The previous project path under `small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009` was moved into this folder. Do not recreate that host project tree.
- `archive/infrared_problem_screen_20261009` is the materialized immutable execution archive. `archive/original_project` preserves the migrated original tree and Git provenance. Many files are hardlinks: do not edit frozen historical assets in place.
- Existing absolute historical paths are supported by read-only container mappings from the package archive. Host bind-mount sources must stay inside this folder.
- New authorized experiments write under `experiments/`; current work is `experiments/head_lfp_joint_20261012`.

## Local environment

- Local CUDA GPUs1,2,3 are user-authorized. Check occupancy before each launch; never stop another task or take an occupied GPU.
- Reuse immutable Docker image `sha256:eb956873870d737f2b68dff0724b8f6f56f778829d119b0938d29b7abeaa2f92` and package `environment/site`. No package upgrades are needed.
- Existing launch wrapper is `experiments/head_lfp_joint_20261012/code/launch.py`; task/CPU/GPU bounds and receipts are explicit there.
- Current workflow was already started. Read its `runtime/WORKFLOW.json` and process state before doing anything; do not rerun training or overwrite output paths.
- No W&B or external notifications are configured for this task.

## Research identity

- H and HL both use seed46 and a fixed2048step budget; 512/1024 are process observations,2048 the main endpoint. Preserve the frozen protocol and source hashes.
- The complete public datasets are archived inputs, not an instruction to change training exposure. Current experiment retains eight training, two development and four untouched confirmation clips.
- Threshold remains0.5 and existing actual-component matching is reused. Diagnostic M/U are not official Pd/Fa.
- Historical negative results remain valid. Separate H-v-release recovery from HL-v-H incremental effects.
