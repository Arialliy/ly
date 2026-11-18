# TEP-SAM immutable environment and native running check

This reuses image `deeppro-official-eval:20261009`, expected image ID `sha256:eb956873870d737f2b68dff0724b8f6f56f778829d119b0938d29b7abeaa2f92`. No pip installation, environment rebuild or original source edit. Existing canonical environment spec SHA is `de8a7eb93aac154354646ae9b59c7ba4086c2ad1d8a704894c9c1de86a4d6d86`. See `WARM_REUSE.json` for actual versions; the author's old requirement versions have not been recreated. CPU imports and strict loading pass; this invocation validates the actual forward and original metrics.

Run **exactly once**, from the host; do not repair or retry silently if it fails:

```bash
bash /home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/temporal_diagnosis_20261010/tep_sam_reference_20261012/runtime/launch_native_check_gpu3.sh
```

The command refuses to overwrite `native_check`, checks GPU 3 has <500 MiB use, then starts a single read-only-source container with 4 CPU / 20 GiB host memory / GPU 3. Existing project inputs are read-only; only the new stage is writable, with author source overmounted read-only. The environment is network-disabled; all weights and data are local. The command captures stdout/stderr in `runtime/native_check_gpu3.log`.

Expected evidence:

- `WITNESS_OK` from explicit seed 46, followed by the original author's native `test.main` loop on fixed **Sequence85, 100 frames**. This is a running check, not full author test-set replication, and not a training reproduction.
- Strict load of the 382,496,611-byte NUDT checkpoint, SHA `1125ea28b4dfe0745be1f33500f4e5463f1cbd8e837a3c1c733b34bb0de9e6f0`.
- 95,576,030 parameters; the native final output has shape `[1,1,1024,1024]`, trained auxiliary output `[1,1,256,256]`. No model training and unchanged all parameter/buffer bytes after inference.
- `native_check/COMPLETE.json` status `PASS_NATIVE_RUNNING_CHECK_ONLY`, 100 native frames and 100 exact comparisons of native metric input to the exported probability's `>0.5` binary mask. Parameter count, GPU model/time/memory and source before/after hashes are recorded.
- Exported `probabilities.npz` and `auxiliary_probabilities.npz` both hold FP32 `[100,200,300]` maps plus `frame_indices`, `source_frames`.

The original `test.py` is imported and its main loop executes unchanged. The wrapper only seeds, records unchanged infer return values, captures the existing image-encoder auxiliary return with a passive hook, verifies native metric input, and records provenance. The native config copies `configs/sam-vit-b-token-NUDT-MIRSDT.yaml` and changes only the two test data root paths and adds `seq_list=runtime/native_sequence.txt`. The README's shorter config filename does not exist; this actual source config is used. No access to ground truth enters model inference or prompt generation.

Data: the previously verified original NUDT-MIRSDT Sequence85 PNG images and masks in `official_reproduction/data/NUDT-MIRSDT/Sequence85`. Original source loader converts grayscale to RGB, stretches the whole image to 1024 square, uses ImageNet normalization and a centered seven-frame window, clamping indices within Sequence85. Author inference upsamples logits with bilinear to 1024; author test then performs sigmoid and area interpolation to 200×300 before strict `>0.5` thresholding. The project development scorer uses its frozen `>=0.5`, with exact equality pixels counted separately.

Auxiliary output is **not an author metric**: author infer discards it, and no auxiliary evaluation postprocess is published. The preregistered project adapter reuses the author's entire `postprocess_masks` chain on auxiliary logits (bilinear 1024, original crop, second bilinear 1024), then sigmoid and area to original dimensions. No alternative interpolation is selected. The author's duplicate temporal-difference calls remain untouched.

FP32, batch 1, no autocast, no spatial tiling, no object crops; TF32 disabled, cuDNN benchmark disabled and deterministic enabled. DataLoader workers set to 0 for a deterministic lightweight check. The author's native loop itself remains the point of comparison. Forward timings exclude export/metric processing but include the passive GPU reference hook; wall timing includes native data loading and export. Do not compare this to DeepPro as an equal-compute replacement.

After a fresh agent follows this invocation, root reviews the receipt before authorizing the two development sequences. No development inference is launched by this command.

Resource-only attempt 2: original GPU0 launch refused occupied GPU before creating output or loading a model. GPU3 had 0MiB/no CUDA process over multiple checks; exposing a GPU to another container is not an exclusive device allocation. This command again refuses a GPU already using >=500MiB. No model, dataset, scorer or infer code change. Preserve original blocked receipt; write fresh receipt as DOC_FOLLOW_GPU3_RESULT.json.
