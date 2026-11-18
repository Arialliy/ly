# Independent audit: real-clip DeepPro-Plus inference

**Verdict: PASS_WITH_LIMITATIONS for failure diagnosis. No blocking implementation mismatch found.**

Audit date: 2026-10-10. This audit read the actual adapter, pinned model and imported layers, official test loader and test entrypoint, launch records, and all four saved prediction sets. It did not modify inference/model code, install packages, or run GPU inference. The saved-output check ran on CPU in the existing `deeppro-official-eval:20261009` image.

## 1. Identity and scope

- Source commit: `8fa1a68b94eb22e94ccd0529e6c5ceccdaa7ec28`; `git status --porcelain` was empty during audit.
- Adapter SHA256: `7527aa474ec4e0fd4e85fd493dee871aa795238ff4ad703cde9956f113824625`.
- Actual model is the checkpoint-adjacent `RGB-T__2025-01-14_15-27__SoftLoUloss_DeepPro-Plus_DataL40/DeepPro-Plus.py`, with imported `networks/layers/basic.py` and `TPro.py` from the pinned source. The adapter uses strict state loading and `detector(1, 40, 40).eval()`.
- RGB-T checkpoint SHA256: `4fd4aa2736d9a33f2bf1459099068d8b4a7e202ff80aecdb4a3277fd13c7abcc`; saved metadata reports epoch 21 for all clips.
- Launch commands retain native 640×512 resolution, FP32, all 40 temporal frames per model call, core 128 and spatial halo 16. No resizing, temporal subdivision, AMP, new training, or changed weights occurs in this adapter.
- Full-frame native attempts ended with CUDA OOM; failed logs/status remain under each run directory. Successful tiled containers all exited 0.

Source-file hashes are in [INFERENCE_SOURCE_HASHES.json](INFERENCE_SOURCE_HASHES.json). This audit concerns the current real-clip inference adapter. It does not turn these additional clips into a reproduction of the full official RGB-T benchmark.

## 2. Spatial dependency and tiling

The maximum spatial dependency radius of this specific Plus eval path is **9 pixels** in each spatial direction:

| Layer/path | Added spatial radius |
|---|---:|
| Input `SDifferenceConv`, spatial kernel 7 and dilation 1 | 3 |
| First STD block: TD kernel 3/dilation 1, then SD kernel 3/dilation 2 | 1 + 2 |
| Second STD block: same | 1 + 2 |
| Residual 1×1 branches, remaining 1×1 convolutions | 0 |
| TPro temporal linear maps | 0 |

TD/SD operations construct modified weights, then call local `conv3d`; they do not enlarge the stated kernel support. All spatial strides are 1. BatchNorm3d modules are in eval mode with their saved running statistics, so they do not recompute statistics across the tile. TPro applies eight temporal Linear(40,40) maps at each spatial position and does not pool spatially.

The adapter passes the full 40-frame tensor into each spatial tile, crops each tile with halo 16, and only copies its core logits. Internal tile boundaries are farther away than the 9-pixel dependency radius. At true image boundaries the crop stops at the true boundary, allowing the original per-layer zero padding to operate there. Thus this tiling preserves the full-frame mathematical dependency; it does not add a new model-level selection or fusion mechanism.

Five fixed 128×160 patches per clip compare native patch inference against core-48/halo-16 tiling. All 20 comparisons pass `rtol=1e-4, atol=1e-4`:

| Clip | Maximum absolute logit difference | Maximum absolute probability difference | Threshold disagreements at 0.1 / 0.5 / 0.9 |
|---|---:|---:|---|
| DJI_0028_2 | 4.1962e-5 | 4.1723e-6 | 0 / 0 / 0 |
| DJI_0028_4 | 5.7220e-5 | 8.2850e-6 | 0 / 0 / 0 |
| DJI_0032_5 | 6.1512e-5 | 1.0192e-5 | 0 / 0 / 0 |
| DJI_0047_3 | 5.6744e-5 | 8.2850e-6 | 0 / 0 / 0 |

These checks support numerical agreement for the tested patches. Native full 640×512 inference did not complete, so there is **no direct native-versus-tiled full-frame numerical comparison**. Production core 128 also has a different convolution shape from the core-48 check. The dependency argument applies, but floating-point kernel differences preclude a bitwise-equivalence claim.

## 3. Preprocessing, windows, and passive saves

- Official `data_utils/TestDataLoader.py` uses RGB-T mean 85.0799 and standard deviation 47.4845 directly on original image intensities. The adapter uses the same values on float32 inputs, with no division by 255, resize, or clipping. Input JPEG-to-NPY fidelity is outside this code audit; earlier data manifests remain its source of evidence.
- The official loader uses sequence length 40 and overlap length 4. The adapter's `ceil((n-4)/36)`, end anchoring, and start indices reproduce this rule for the 160-frame diagnostic clip: `[0,40), [36,76), [72,112), [108,148), [120,160)`.
- Official `test.py` fuses overlapping probabilities with maximum. The adapter does the same. Reconstructing every saved fused array from its five saved windows gives **exact array equality** for all four clips, and independently counted coverage exactly matches the saved vector: 120 frames with coverage 1 and 40 with coverage 2 per clip.
- All four fused arrays have shape `[160,512,640]`, dtype float32, finite values in [0,1]. All twenty window arrays have shape `[40,512,640]` and finite values.
- The added `np.savez_compressed` writes `probs` without modifying it or model state. Fixed probe calls occur in eval/inference mode and do not update BatchNorm running statistics. Source inspection found no forward-time random operations or mutable recurrent state on this path.
- These are windows of the extracted diagnostic clip. They do not reproduce the temporal context or window phase of running the entire original video; clip-boundary effects must be reported explicitly.

Independent CPU check code and full evidence: [check_saved_inference.py](check_saved_inference.py), [SAVED_INFERENCE_CHECK.json](SAVED_INFERENCE_CHECK.json).

## 4. Seed and numerical/runtime distinctions

The official `train.py` declares `seed_everything(seed=46)` but contains no invocation of that function; worker initialization separately uses `time.time()`. This adapter explicitly activates 46 for Python, NumPy and Torch, disables TF32, and requests deterministic algorithms. Its metadata correctly distinguishes the **local inference seed 46** from the **unknown training seed of the supplied checkpoint**. This audit supports that distinction and does not establish author-training reproducibility.

The tiled branch copies logits to CPU before sigmoid; official native `test.py` applies sigmoid on GPU and then copies probabilities. The expression is mathematically identical, but this is an additional reason to avoid asserting bitwise identity. The patch check compares logits first; both patch probabilities are then computed on CPU.

`elapsed_inference_seconds` includes fixed numerical probes, tile execution/transfers, and per-window compressed saves. It is a diagnostic job duration, **not clean model latency, FPS, or a fair deployable inference-cost measurement**.

## 5. What this audit permits

The saved arrays are suitable for the declared real-clip error and context-consistency diagnostics, subject to the limitations above. The audit does not assess the downstream bounding-box/component association code, annotations, data independence, or whether an apparent failure is caused by motion. Acausal 40-frame context, fixed temporal linear maps, or a low response are not by themselves proof of an unresolved temporal detection mechanism.
