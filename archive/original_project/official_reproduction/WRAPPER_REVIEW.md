# Official inference wrapper review

Initial review: 2026-10-09. Updated review: 2026-10-10 (Asia/Taipei), after the second doc-follow attempt exposed a missed constructor failure. Scope: static, read-only comparison of `run_official_shard.py` and `aggregate_official.py` with pinned official `test.py`, `data_utils/TestDataLoader.py`, `ShootingRules.py`, and `write_results.py`, plus inspection of the recorded doc-follow failure. Relevant detector/layer source and checkpoint pickle opcodes were also read. No official Python module or Torch was imported, no checkpoint was unpickled, no GPU was used, and this reviewer changed only this report. A minimal standard-library constructor check executed the wrapper's AST-extracted initializer against a stub class. Runtime tests are performed by the separate doc-follow agent.

## Verdict

**The earlier static PASS missed a real blocker and is superseded. The constructor repair is now checked statically; runtime attempt 3 remains pending.** Attempt 2 passed imports/CUDA witness but failed before checkpoint loading or any model forward. Its wrapper replaced the module's `detector` class with a function, invalidating the original constructor's `super(detector, self)` call. This interaction was visible in the source and should have been caught in the first review; checking hook non-mutation alone was insufficient.

The current wrapper preserves the `detector` class symbol and wraps its initializer to register the recording hook after original initialization. The isolated constructor check passes, and the unchanged overlap and metric adapters remain semantically consistent on static inspection. The aggregator still restores all 20 sequences in the original order before calling the official ROC writer. None of these findings establishes successful inference or author-count agreement; attempt 3 must demonstrate the actual constructor, strict checkpoint load, model forward, metric callback, and completed output artifacts.

## Findings and verified resolutions

1. **Runtime blocker missed by this review; code repair checked, runtime confirmation pending.** `doc_follow_v2_result.json` records command 1 exit 0 and command 2 exit 1; `doc_follow_v2.log:13–27` captures `TypeError: super() argument 1 must be a type, not function`. The failing official constructor is `DeepPro-Plus.py:14`. The initial wrapper assigned `model_module.detector = recording_detector`; the original class initializer resolves the global name `detector` in that module at call time, so it received a function in `super`. No strict checkpoint load or model forward was reached, and no counts/probability/COMPLETE artifacts were produced. Current `run_official_shard.py:119–127` keeps `model_module.detector` bound to the original class, saves `original_detector.__init__`, and replaces only `__init__` with an adapter that calls the saved initializer before registering one hook. The saved initializer's explicit `super` therefore resolves the original class. The adapter returns `None`, preserves constructor arguments, and does not change `forward`. A standard-library stub with the same explicit `super(detector, self)` reproduced the old error and passed with the current AST-extracted adapter. This checks Python constructor semantics; it does not replace the pending attempt 3 execution.

2. **Resolved in documented commands: source protection is now an explicit read-only mount.** Attempt 2's parent `/home/ly` bind mount was writable, so the earlier statement that the source was read-only was not enforced, although the source remained clean. Both current `COMPUTE.md` commands additionally bind the exact `official_deeppro` directory onto itself with `:ro`. This child mount protects the source used through output symlinks while leaving the output directory writable. `PYTHONDONTWRITEBYTECODE=1` remains set. Runtime confirmation of the new mount belongs to attempt 3.

3. **Resolved: existing model/checkpoint links now require the pinned source paths.** The initial revision retained pre-existing output experiment entries without validating their targets and hashed the source checkpoint instead of the checkpoint path used for loading. Current `run_official_shard.py:86–87` rejects any existing `logs`, `checkpoints`, or `DeepPro-Plus.py` entry whose resolved path differs from its expected source entry. Lines 169–171 hash the checkpoint through the actual output path and record resolved checkpoint/model paths. These changes close the identified stale-output provenance mismatch. A broken old symlink also fails closed when link creation fails.

4. **Resolved: the additional arithmetic policy is now explicit.** Besides matching the source helper's seed/cuDNN settings, `run_official_shard.py:53–55` disables CUDA matmul TF32 and cuDNN TF32 and requests deterministic algorithms. Those extra switches are absent from the official seed helper and test entrypoint. They preserve the algorithm but can select different floating-point kernels from an unmodified historical run. The current `COMPUTE.md` has an explicit “Local arithmetic policy” section stating these switches and the unknown author runtime settings. `run_official_shard.py:178` now records the policy in `COMPLETE.json`. No historical bitwise equivalence claim is justified by this static review; any numerical difference must still be assessed using the actual author-count comparison.

## Adapter checks

| Adapter | Assessment and evidence |
|---|---|
| Seed 46 | Explicitly applied before dataset/model construction (`run_official_shard.py:47–52`). Official `train.py:33–40` declares default 46 but official `test.py` does not activate it. The wrapper reimplements the seeding operations; it does not literally call the official helper. The source helper's separate `torch.cuda.manual_seed` is redundant after the wrapper's all-device seeding. The record correctly calls the released checkpoint's training seed unknown. |
| Checkpoint compatibility | The narrow safe-global context encloses the unchanged `official.main` and thus its unchanged `torch.load` and strict `load_state_dict`. Static inspection of the checkpoint ZIP's `data.pkl` found `numpy.core.multiarray.scalar` once and `numpy.dtype` once, with `class_avg_iou` and dtype `f8`. These match the declared scalar alias, dtype constructor, and float64 dtype class. The adapter does not replace tensors or state-dict keys. Runtime compatibility remains the separate environment gate. |
| Sequence sharding | The subclass first executes the original constructor, asserts identical original sequence order, and changes only `seq_names`. `__len__` and `__getitem__` both derive from that list. Each sequence keeps its original sorted frame files and original preprocessing: image float32, first channel if needed, `(image - 105.4025) / 26.6452`, original masks/centroids, and native dimensions. |
| Disjointness and completeness | With four shards, IDs are `[0,4,8,12,16]`, `[1,5,9,13,17]`, `[2,6,10,14,18]`, and `[3,7,11,15,19]`. Thus all 20 official sequences occur once. The aggregator verifies the data order against the 20-entry author reference, rejects inter-shard ID overlap, and requires the final ID set to equal every dataset index. A one-sequence doc-follow probe cannot satisfy that full-run merge check. |
| Output relocation | `logs`, `checkpoints`, and the archived detector module must resolve to the pinned source; `eval.txt` is created in the output experiment directory. The original log model-name selection resolves to `DeepPro-Plus` because the source logs directory contains only `DeepPro-Plus.txt`. Model-module preimport and the later official import share that module object, so the updated class initializer is used. The original class symbol is preserved. Existing-entry resolution checks also cover retries. Current documented containers additionally mount the source directory read-only. |
| Forward hook | The current adapter runs the original class initializer first, then registers the forward hook on that initialized instance. It leaves the module's class name intact, preserving the original explicit `super` call. `record_forward` returns `None`, computes sigmoid on `outputs[1]`, and copies the result to a separate NumPy buffer. It does not mutate the returned logits, intermediate features, parameters, or official prediction buffer. Batch size is fixed at one, so `[0]` removes exactly the batch dimension. The detector returns `(seq_feats, seq_midseg)` with `seq_midseg` shaped `[B,T,H,W]`. Full execution is pending attempt 3. |
| Forward-call alignment | `plans` comes from the original loader's `samplelist` in the selected sequence order. Official iteration uses the same loader and `shuffle=False`, one detector call per window. Before counts are accepted, the callback requires `capture['calls'] == len(plans)`. The later official THOP profiling call is ignored by the hook after all prediction windows; it does not become a saved probability sequence. |
| Metric preservation | Official IoU still uses the original segmentation mask and strict `prediction > 0.5`. Official Pd/Fa still uses centroid masks, all 27 thresholds, and unchanged `ShootingRules`, including its `>=` detection threshold, target-component count, 3×3 hit neighborhoods, 9×9 exclusion neighborhoods, and original slice boundary behavior. No alternative metric code is inserted into inference. |
| Deferred official ROC | Intercepting only the final NUDT summary is necessary because `writeNUDTMIRSDT_ROC` hard-codes original low-SNR indices `[7,13,14,15,16,17,18,19]`. The wrapper saves the already computed official per-sequence FP/TP/GT arrays, pixel denominators, threshold vector, and IoU intersection/union. `aggregate_official.py:40–61` restores rows by original indices, sums the shard IoU totals, and invokes that same official writer only after complete coverage. |
| Author comparison | Aggregation compares every FP/TP/GT cell against the author reference, reports exact equality and differences, and computes Pd/Fa from summed counts and pixel denominators rather than averaging per-sequence rates. It retains the logged author IoU/AUC with their original limited precision. It does not force local counts to equal the reference or substitute reference counts for local observations. |

## Probability reconstruction: overlap maximum is correct

All current sequences contain exactly 100 frames. The official sampling formula, with `seq_len=40` and `cat_len=4`, yields inclusive, zero-based windows:

| Window | Frame range | Overlap with accumulated prediction |
|---|---|---|
| 0 | 0–39 | None |
| 1 | 36–75 | 36–39, four frames |
| 2 | 60–99 | 60–75, sixteen frames |

The final overlap is **16 frames**, because the last window shifts backward to retain 40 input frames. The wrapper correctly reads these endpoints from the official sample lists; it does not assume a fixed four-frame overlap.

Official `test.py:150–159` first copies window 0, replaces each later overlap with `torch.maximum`, and appends the newly covered suffix. The wrapper's `np.maximum(buffer[start:end+1], probability, out=...)` performs the same maximum over all contributing windows, in the same order. Uncovered buffer entries start at zero; sigmoid probabilities are nonnegative, so this initialization does not alter first writes. Both paths apply Torch sigmoid to the same float32 logits before copying to CPU. Each produces 100 complete probability frames, with coverage one for 80 frames and coverage two for 20 frames. The stored probabilities are the fused probabilities before thresholding, matching the official `seq_midpred_all[0]` for these data and settings.

## Verification performed

- Read the full wrapper, aggregator, four requested official files, detector forward, and relevant layer definitions. Parsed both adapter files with Python's AST parser without importing or executing the full modules.
- After the recorded attempt 2 failure, re-read the official constructor and current adapter. Executed only the AST-extracted `recording_init` function against a standard-library stub class: the class identity, base initialization, argument passthrough, and exactly one hook registration all passed. Rebinding the class symbol to the old factory reproduced the exact `super()` TypeError. The earlier static PASS is withdrawn rather than treated as successful execution evidence.
- Re-read the entire aggregator after the constructor fix; its SHA-256 is unchanged. Its order assertion uses the actual 20-name reference, rejects overlap across shard ID sets, requires the final ID set to equal all 20 dataset indices, then invokes the official ROC writer. Complete counts, matching thresholds, common source/checkpoint provenance, probability hashes/finiteness, and positive coverage remain merge gates.
- Inspected both current documented Docker commands: each now contains the exact source-directory child bind with `:ro`, plus `PYTHONDONTWRITEBYTECODE=1`; the fresh output path is `doc_follow_probe_v3`.
- Checked the actual local split yields 20 names in exactly the author-reference order; checked every sequence has filenames `00001.png` through `00100.png` in images, masks, and centroid masks. This is a file-coverage check, not a repetition of the separate PNG decode/CRC audit.
- Checked the actual reference's low-SNR indices and its 27 thresholds against the hard-coded official grouping and threshold use.
- Exercised the append/max and scatter/max equations on 20 synthetic, nonnegative scalar-window cases using only the Python standard library; all 100 reconstructed positions matched exactly in every case. This validates indexing, not CUDA numerical parity.
- Used `pickletools` on the checkpoint's raw pickle bytecode without deserializing its objects. The only NumPy globals were the single scalar constructor and dtype constructor described above.
- Confirmed the official Git working tree is clean and pinned at `8fa1a68b94eb22e94ccd0529e6c5ceccdaa7ec28`.

Reviewed adapter SHA-256 values:

```text
run_official_shard.py  10ebe4dd954eebdbcbba41f3ed59de22173d8cafd61fb75a8fe09a947b398e2b
aggregate_official.py 77d412be91dc4f1d8a0ea550dd1a540b9da05168a798b5064f88c8208cbbef87
```

Official file SHA-256 values:

```text
test.py                         99b58962061f101bcaf1cb4e787a46599ab947805444af910d405324cf314b9b
data_utils/TestDataLoader.py     03210e84ac9acfc3fd85ef5bcd2c787cfc126e2580e622fca9c3297c3e1c022e
ShootingRules.py                888d4a26ff2877dc64550bc919fa60bf18e04e0af795aa8e8c304aa0b1e5c4e2
write_results.py                47984378234a8f6e1ec3e18d83a50af8b93cce52645f1e04042afe52f04654ef
```

Attempt 3 runtime success, actual read-only mount enforcement, saved-probability finiteness/hashes, actual GPU memory capacity, and final full-dataset author-count agreement remain pending the planned execution and aggregation gates. The attempt 2 failure record and logs must remain available; they establish that the previous static verdict did not catch the constructor blocker.
