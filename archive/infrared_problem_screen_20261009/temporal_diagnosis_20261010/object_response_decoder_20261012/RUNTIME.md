# Immutable local execution

Warm reuse image sha256:eb956873870d737f2b68dff0724b8f6f56f778829d119b0938d29b7abeaa2f92 and canonical spec de8a7eb93aac154354646ae9b59c7ba4086c2ad1d8a704894c9c1de86a4d6d86. No installs or network. Host Python is stdlib only; numerical commands run inside the image. User authorized GPUs2/3. Launcher refuses ≥500MiB occupied card and never stops unrelated work.

Fresh documentation-follow check, execute exactly:

```bash
python3 /home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/temporal_diagnosis_20261010/object_response_decoder_20261012/code/launch_gpu.py --kind witness --gpu 3
```

Expected: exit0, runtime/witness_1.json status COMPLETE, log with deterministic CUDA synthetic model/gradient/chunk checks. Zero optimizer updates. This is a seeded kernel/interface witness, separate from the completed actual-backbone training-only preflight. The launcher uses explicit /opt/conda/bin/python entrypoint, original project read-only and only current experiment writable. No hidden path setup or shell activation required.

Formal invocations after PROTOCOL_FREEZE.json and GPU_PREFLIGHT PASS:

```bash
python3 /home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/temporal_diagnosis_20261010/object_response_decoder_20261012/code/launch_gpu.py --kind train --gpu 2 --condition R0
python3 /home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/temporal_diagnosis_20261010/object_response_decoder_20261012/code/launch_gpu.py --kind train --gpu 3 --condition R1
```

R2 uses a free authorized card after previous process exits. Inference uses `--kind infer --gpu 2 --condition R0 --sequence DJI_0026_2` (all three conditions and both sequences). Receipts/logs cannot be overwritten; `--attempt` creates a new infrastructure-attempt receipt, never permits replacing completed formal outputs or selecting a checkpoint.
