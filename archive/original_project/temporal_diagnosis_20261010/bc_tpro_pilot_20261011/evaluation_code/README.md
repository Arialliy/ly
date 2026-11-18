# BC-TPro pilot evaluation interface

This directory owns CPU scoring and summary only. Model implementation, training, inference and GPU launch belong to separate files. Old experiments are read-only.

## Inputs

New conditions: `single_dynamic`, `bc_tpro`. For each condition and each of `DJI_0026_2`, `DJI_0067_2`, inference writes:

```
predictions/<condition>/<sequence>/
  windows/window_000.npz
  windows/window_024.npz
  windows/window_036.npz
  windows/window_060.npz
  windows/window_072.npz
  windows/window_096.npz
  windows/window_108.npz
  windows/window_120.npz
  official_fused.npz
  COMPLETE.json
```

Every window NPZ has `probabilities`: finite FP32 `(40,512,640)`, and scalar `start`, `end=start+40`. Probabilities follow the frozen native processing order; do not substitute averaged logits, reordered sigmoid/interpolation, resized images or a different tiling extent without registering the implementation difference. New-module halo/tiling correctness is the inference implementation's responsibility and must be verified before interpreting results.

`official_fused.npz` holds the exact pixelwise maximum probability from O starts `[0,36,72,108,120]`. Only these five windows enter O. S is re-fused from `[0,24,60,96,120]`. The other three windows needed for the second arrangement do not increase either arrangement's five-window budget.

`COMPLETE.json` needs `status='COMPLETE'`, `raw_sha256`, `fused_sha256`, `window_sha256` keyed by NPZ filenames. Include checkpoint/model/config hashes, timing and tiling witness metadata for provenance; the scoring wrapper checks the mandatory cache hashes through the old scorer. It refuses to overwrite a completed new scoring result.

Before scoring, the wrapper also binds inference to `training/<condition>/run/RUN.json` and `final_model.pth`: condition/sequence identity, complete training, step512, exact checkpoint path/hash, original and BC model source hashes, original checkpoint, model configuration, normalization, starts, no-label inference, halo at least49 and nonempty trained-model tiling checks must agree. Missing metadata fails closed rather than assigning the wrong same-shaped checkpoint to a condition.

References are read directly from `../top1_background_control_20261011/evaluation/{release,lambda0_top1}/<sequence>`: `scores.npz`, `component_areas.npz`, `FRAME_COUNTS.csv`, `COMPLETE.json`. No new reference inference or rescoring is required.

## Commands

Use the established `deeppro-official-eval:20261009` image. These commands are CPU only; invoke them inside that container with the project mounted at its absolute path.

```
python evaluation_code/evaluate_bc.py --baseline-check
python evaluation_code/evaluate_bc.py --condition single_dynamic --sequence DJI_0026_2
python evaluation_code/evaluate_bc.py --condition single_dynamic --sequence DJI_0067_2
python evaluation_code/evaluate_bc.py --condition bc_tpro --sequence DJI_0026_2
python evaluation_code/evaluate_bc.py --condition bc_tpro --sequence DJI_0067_2
python evaluation_code/evaluate_bc.py --summary
```

Omit `--sequence` to score both clips in sequence. Optional `--predictions-root /absolute/path` points directly to the directory containing both sequence subdirectories for the selected condition. Without it, the default path above is used. Run `--summary` only after all four new condition×sequence scores exist. It fails on missing weights' results instead of inserting zeros. It refuses to overwrite a completed summary.

`--baseline-check` writes only `evaluation_preflight/`; it validates cached references/cohorts and summary logic, not new-model performance. The baseline result receipt explicitly has `new_models_evaluated=[]`.

## Frozen evaluation and outputs

- Threshold 0.5, all matching-valid GT considered before counting the 5,200 strict-small target frames. Original `prepare_gt`, maximum-cardinality component-peak matching, half-open boxes and pixel-center coordinates are reused exactly.
- Score wrapper imports the old `top1_background_control_20261011/evaluation_code/evaluate.py` and redirects only the condition's prediction root and output root. No architecture-, score-, or response-specific matching changes.
- `evaluation/<condition>/<sequence>` contains the same score/area/per-frame/cache-provenance format as the original experiment.
- `SUMMARY.csv`: release, same-training top1, single dynamic and full BC-TPro, each O/S. M, same-arrangement gain/loss versus release and top1 and single dynamic, gain/loss versus original release O, U, framewise costs, low/extreme recovery, trajectory/source counts, component area distributions, old U budget and release-relative background deltas.
- `SEQUENCE_SUMMARY.csv`, `TRACK_SUMMARY.csv`, `RECOVERY_CONCENTRATION.csv` retain source and trajectory distributions. `OBJECT_COMPARISON.csv` has 10,400 rows with all four conditions' results, identity and frozen strata; `FRAME_COUNTS.csv` retains all per-frame background quantities.
- `WINDOW_DEPENDENCE.csv`: per condition, pooled/per clip, total and each frozen cohort; reports O/S matched, both matched, O-only, S-only, neither matched, arrangement disagreement. It explicitly includes old O-only54, old S-only74, new-window-dependent73, extra-only104, and 286 targets with both successful and unsuccessful originally sampled covering windows.
- `SUMMARY_COMPLETE.json` records hashes and exact claim scope. U is unmatched components, not confirmed FP or official Fa; the scores are diagnostic point matches, not official Pd. New-model O/S stability is not evidence that every one of the original 11 windows is now stable.

Frozen cohorts come from `low_response_information_20261011/COHORTS.csv`; the full and low-response populations are never redefined from a new model. The four main strata overlap and are not asserted to partition the 5,200 rows.

Validation asserts original O/S `(M,U)=(700,4015)/(720,3967)`, top1 `(877,8339)/(913,8330)`, exact GT row/order equality across all models, 10,400 object-arrangement rows, low4,342, extreme4,170, both-success646, extra104, new-window-dependent73 and sampled mixed-success286.
