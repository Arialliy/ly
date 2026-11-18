# Native logit export contract

This export adds diagnostic pre-sigmoid scores to the two existing development clips. It does not alter the released checkpoint, preprocessing, cached probabilities, detection threshold, connected components, or matching.

## Existing computation, checked in local source

`stage1_goal/pilot/infer_pilot.py` reads uint8 `raw_frames.npy` with shape `[160, 512, 640]`, normalizes each real 40-frame window in FP32 using `(x - 85.0799) / 47.4845`, and loads the checkpoint-adjacent `DeepPro-Plus.py` through `mechanism_followup/run_factorial.py:model_setup`.

`infer_with_windows.py:tiled_logits` uses spatial core 128 and halo 16. Each tile is evaluated by the original model. The tile's native output core is copied to a CPU tensor `[1, 40, 512, 640]`. There is no interpolation or image resizing in this selected pipeline. The model output before sigmoid has exactly the input spatial resolution. The previously audited longest local receptive radius is `(8, 9, 9)` in temporal, y, x axes.

The existing wrapper then computes CPU sigmoid and saves FP32 probabilities. Original O uses starts `[0,36,72,108,120]`; symmetric S uses `[0,24,60,96,120]`. Their actual output remains the pixelwise maximum of those saved probability maps. Existing window archives contain only `probabilities`, `start`, and `end`; no reusable native-logit cache was found for these clips.

## New diagnostic arrays

- `SEQUENCE/window_START_logits.npy`: native pre-sigmoid float32 scores `[40,512,640]`, obtained by the identical inference path.
- `SEQUENCE/fused_O_logits.npy` and `fused_S_logits.npy`: pixelwise maximum of the native window logits, float32 `[160,512,640]`.
- `SEQUENCE/coverage_O.npy` and `coverage_S.npy`: covering-window counts, int16 `[160]`.

There are eight starts per clip, the union of O and S. The model does not read annotations. Files are written only to this new logit directory.

Since sigmoid is monotone and no interpolation intervenes, the maximum-logit diagnostic map corresponds to the existing maximum-probability map. This relationship is verified against the saved arrays rather than merely assumed. Native logits can distinguish score differences that FP32 probabilities have saturated into ties. Consequently, a **diagnostic logit argmax** is not silently substituted for the original **probability component peak**. Actual matching must continue using the original probability outputs and their unchanged tie rules.

## Reproduction and checks

`export_logits.py` matches the original seed 46, FP32, two Torch CPU threads, deterministic algorithms, disabled cuDNN benchmarking, and disabled TF32. It uses the existing Docker image `deeppro-official-eval:20261009` and one otherwise idle GPU. Re-export is allowed only after the root diagnostic protocol has been frozen.

Every window is re-sigmoided on CPU and compared with its old probability archive at every pixel. Any nonzero difference, including one that leaves the 0.5 mask unchanged, is a hard failure. Old predictions are never overwritten. Source files, raw inputs, checkpoint, and old window caches are hashed before and after execution. `COMPLETE.json` records comparisons, resource use, and output fingerprints.

`verify_logits.py` is a separate CPU-only check. It reads the saved native logit files, recomputes each arrangement's maxima independently, and compares their sigmoid probabilities, threshold-0.5 masks, and logit-zero masks with the existing O/S fused arrays. It checks all input and output fingerprints. Its final result is `INDEPENDENT_VERIFICATION.json`.

Neither operation reconstructs logits from probabilities. Neither operation modifies sigmoid/interpolation order. This is a score export and provenance check, not a new detection experiment.
