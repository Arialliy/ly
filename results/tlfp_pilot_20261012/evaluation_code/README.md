# Fixed S / T endpoint scoring

This package evaluates the registered 512-step S and T checkpoints, plus the mandatory T-current and T-bypass inference modes from exactly the same T file. It does not train models, select checkpoints, adjust thresholds, add NMS, truncate predicted detections, or evaluate official Pd/Fa.

`evaluate.py` calls `top1_background_control_20261011/evaluation_code/evaluate.py::score` directly after verifying its frozen SHA and source dependencies. That function still calls `stage1_goal/pilot/score_pilot.py` and `analysis/analyze.py::match_frame`. All matching-valid annotations participate in one-to-one matching and background accounting; only the existing strict-small 5,200 target frames enter the reported M. Thresholding is the actual frozen **probability >= 0.5** rule with 8-connected components. A component's maximum-response point is matched to the existing annotation rectangle under the unchanged ordering/tie handling. No IoU detector rule is substituted.

## Inputs and identity

Four prediction roots: `predictions/S`, `predictions/T`, `predictions/T_current`, `predictions/T_bypass`. Each contains both existing 160-frame source clips. Each clip provides:

- `windows/window_{start:03d}.npz` for starts 0, 24, 36, 60, 72, 96, 108, 120, containing float32 `probabilities` shaped [40,512,640] and scalar `start`,`end`;
- `official_fused.npz`, float32 `probabilities` [160,512,640];
- `COMPLETE.json`, binding actual data, checkpoint, mode, probabilities and code hashes.

Only the frozen five-window O and S plans are scored: O=[0,36,72,108,120], S=[0,24,60,96,120]. They are merged by unchanged pixelwise max of the already processed probability maps. All 160 frames must be covered.

`verify_identity.py` verifies the fixed endpoint, original tensors/BN against released weights, unchanged input and label exposure, frozen protocol code hashes, and the mandatory three T diagnostic files. S and T may differ in new parameter counts (100 / 296); their common initialization mapping is a separate model preflight check. T-current is an inference intervention, not a retrained single-frame baseline or a claim of causality.

The historical original-463-object protection tolerance cannot be mapped to the disjoint current 646-object protection cohort. This evaluator reports all four O/S matching states in the frozen 646 cohort and does not invent a new allowable loss count.

## Commands

Use the already recorded immutable Docker image/environment, CPU only for scoring. From the project root:

```bash
python3 evaluation_code/launch_score.py --condition S
python3 evaluation_code/launch_score.py --condition T
python3 evaluation_code/launch_score.py --condition T_current
python3 evaluation_code/launch_score.py --condition T_bypass
```

Each command creates a detached CPU-only container; wait for successful completion of all four before launching:

```bash
python3 evaluation_code/launch_score.py --summary
```

After that container finishes successfully:

```bash
python3 evaluation_code/validate_tables.py --directory evaluation
```

For historical release-only validation, invoke `evaluation_code/evaluate.py --baseline-check` inside the immutable image; it writes `evaluation_preflight` and loads no new checkpoint or prediction. Then use the stdlib validator against that directory.

## Deliverables

- `SUMMARY.csv`, `SEQUENCE_SUMMARY.csv`: M, gain/loss against each condition, U, outside-box foreground pixels, complete component/unmatched-area distributions, low/extreme recovery and source/track counts.
- `OBJECT_COMPARISON.csv`: all 10,400 target-frame/arrangement rows, including frozen cohort flags and exact per-condition matching sets.
- `TRACK_SUMMARY.csv`, `RECOVERY_CONCENTRATION.csv`: track outcomes and concentration.
- `WINDOW_DEPENDENCE.csv`: both matched / O only / S only / neither for the same objects, including the 646 protected target frames.
- `COHORT_TRANSITIONS.csv`: actual gain/loss/intersection/neither sets for all condition comparisons; totals are not mistaken for set differences.
- `CONTINUOUS_MISS_TRACKS.csv`, `CONTINUOUS_MISS_SUMMARY.csv`, `RELEASE_MISS_SEGMENTS.csv`: full frozen observation population, local/source frame gaps breaking runs, track counts and longest misses.
- `MECHANISM_SUMMARY.csv`: T-vs-S and T-vs-current sets, plus how many T restorations survive the current-only guide intervention. This is descriptive evidence, not automatically a mechanism success.
- `BYPASS_PROBABILITY_COMPARISON.csv`, `BYPASS_VALIDATION.json`: all eight windows plus O fusion per clip compared to historical release maps; predeclared max absolute error <= 1e-4, zero >=0.5 disagreements, exact match/peak/background/area identity. Bitwise equality is reported separately. A failing receipt is preserved and stops completion; tolerances are not adjusted.
- `TRAINING_COMPARABILITY.json`, `SUMMARY_COMPLETE.json`, `TABLE_VALIDATION.json`: provenance, actual checks and independently recomputed table identities.

U counts unmatched predicted components, not certified false alarms. The common diagnostic U budget is 4,015; publication Pd/Fa is not computed here. Passing U alone does not establish target recovery, retention, or all background costs being unchanged. S's release U is 3,967, so a higher value below 4,015 still raises that component cost.

`SOURCE_MAPPING.json` records reused script provenance. Old project artifacts and failed routes are not edited.
