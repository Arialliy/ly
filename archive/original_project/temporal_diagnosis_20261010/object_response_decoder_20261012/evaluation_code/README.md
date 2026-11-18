# R0 / R1 / R2 fixed evaluation

`evaluate.py` invokes the unchanged historical `top1_background_control_20261011/evaluation_code/evaluate.py` for probability max fusion, component extraction and GT matching. Only condition names, new training identity checks and aggregate comparison columns are adapted from the completed local-reference experiment.

Fixed main endpoint: 2048 updates, seed 46, R0/R1/R2; two development sequences; original O and symmetric S five-window schedules; threshold 0.5. No checkpoint, window or threshold selection. The frozen 5,200 target frames, 4,342 low-response frames, 4,170 extreme frames, 646 protected frames and all-GT background matching remain unchanged.

After both prediction receipts for a condition are COMPLETE:

```bash
python3 evaluation_code/launch_score.py --condition R0
python3 evaluation_code/launch_score.py --condition R1
python3 evaluation_code/launch_score.py --condition R2
python3 evaluation_code/launch_score.py --summary
python3 evaluation_code/validate_tables.py --directory evaluation
```

The launcher uses the already verified immutable CPU-only Docker image. It does not allocate a GPU. Commands refuse an existing launch receipt, and completed scoring/summary outputs are not overwritten.

`verify_identity.py` verifies actual released backbone tensors and buffers, common initial state, only the new branch being trainable, 31,194 new / 102,107 total parameters, final2048 checkpoint identity, frozen code/protocol, per-update inputs against the historical 512-entry plan repeated four times, and per-update new-label equality across R0/R1/R2. Historical weak-supervision label/loss equality is intentionally not required. R0 temporal parameters must remain unchanged. New global queries and full-frame inference provenance are required.

Delivered tables include the 10,400-row object comparison, true gain/loss sets against release and each R control, background components/pixels/areas, per-sequence and per-track counts, protected-set O/S consistency, concentration, and continuous missed runs. Missing or filtered frames break continuity. `validate_tables.py` independently recomputes table sums and partitions with the standard library; it is not a second raw-probability matcher.

Preflight used released caches only and recovered O `(M=700,U=4015,A=38368)` and S `(720,3967,38641)` exactly. The independent table checker passed 5,136 checks. Those receipts are in `../evaluation_preflight/`; no R predictions were scored during preflight.
