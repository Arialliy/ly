# R0/R1/R2 training integrity audit

**PASS.** CPU-only audit ran after all three formal trainings completed. This certifies the checked exposure, state, update, and loss-accounting artifacts; it makes no claim about detection effectiveness. Receipt: `checks/TRAINING_AUDIT.json`. Reproducible checker: `audit_code/verify_training.py` (SHA-256 `c848d9b53505ee4baeebb39d8a1b4067da9d7c1dbb6a5a2a9458ecbd6e474d0e`).

| Run | Seed | Updates | Window exposures | Train seconds | Active Adam parameter tensors |
|---|---:|---:|---:|---:|---:|
| R0 | 46 | 2,048 | 8,192 | 932.197 | 43 |
| R1 | 46 | 2,048 | 8,192 | 950.179 | 63 |
| R2 | 46 | 2,048 | 8,192 | 957.763 | 63 |

Each model contains 70,913 original parameters and 31,194 newly declared parameters, total 102,107. R0 bypasses 20 temporal parameter tensors; those tensors receive no gradients, obtain no Adam state, and remain bitwise identical to initialization. R1/R2 update the same 63 new tensors. The parameter-tensor count is distinct from the scalar-parameter count.

All three complete initial states, including dormant temporal parameters, are bitwise identical. Initial state SHA: `eb09a29234d9e54300d2a52845f93ff0880d8ba048e2d9c511856ada7bda30b1`.

Every step's input hash equals the corresponding historical entry under exactly four repetitions of the original 512-step plan. Independently recomputed hashes from the frozen per-window label manifest equal every training label hash. The three complete streams are identical:

- Input stream: `ec4f79859f46208f884a14ecfa2886fba9508a9042223d88bcd2d332bc4d436d`.
- Label stream: `5a6e2d6ff45d5a6029b3062a50d361bb61f5383a1b77915ef68ebd6ed70f91c0`.

Per run, the verified totals are 595,344 valid object-frame labels, 258,220 negative-eligible frames, 25,494,016 unmatched-query existence/background negative entries, and 327,984 matched-object instance-negative entries. These counts are the same in all three runs; R2 changes assignment rather than exposure weights or negative eligibility. Each step uses five separate unit-weight loss means and the arithmetic mean over four windows. Recorded losses and gradients remain finite; checkpoint tensors are finite.

All 12 saved checkpoints at steps 0, 512, 1,024, and 2,048 were checked directly. Every original parameter and buffer, including BN running statistics and counters, equals the released checkpoint bitwise. Anchors are unchanged. Fresh Adam has empty state at step 0; each active parameter has the prescribed update count at later snapshots. Each final model is the hardlink to the fixed 2,048-step snapshot, not an earlier selected checkpoint. Final log hashes, full state hashes, parameter-change receipts, and checkpoint file hashes agree.

Frozen source/protocol/label-plan hashes and the referenced old checkpoint, original model/input helper, selection, raw data, and annotations match their existing receipts. This check concerns those explicitly recorded files, not an assertion that every unrelated historical file has been audited.

The user-authorized addition of GPU 1 is recorded transparently in `PROTOCOL_FREEZE.json` and linked to `PROTOCOL_FREEZE_INITIAL.json`. Only the launcher GPU-choice hash changed; training/model/loss/data hashes and the 2,048-step endpoint did not change. Subsequent inference scheduling does not alter the completed training-state checks above. The audit container had no GPU access and did not modify training artifacts.
