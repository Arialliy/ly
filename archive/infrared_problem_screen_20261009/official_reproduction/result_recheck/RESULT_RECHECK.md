# Independent full-result recheck

**PASS with documented numerical limitations.**

Read-only input audit performed in the existing CPU Docker environment; no GPU, new model inference, package install, or training was required.

## Evidence checked

- All 20 sequences appear exactly once in the official `test.txt` order after assembling four disjoint shards. Each sequence has 100 frames: 2,000 probability frames total.
- All 20 probability archive SHA256 hashes match COMPLETE metadata. Every array is finite FP32 in [0,1], uses native image dimensions, and has exact coverage for windows [0,39], [36,75], [60,99].
- All 6,000 images/masks/masks_centroid PNGs were decoded, dimension-checked, and hashed. The source snapshot matches all 140 tracked files before and after; input result artifacts are unchanged.
- The original author `eval.txt` was parsed independently. All 20×27 reference integer TP/GT/FP values, threshold order, sequence order, and aggregate split membership agree with AUTHOR_REFERENCE_METRICS.json.
- The untouched official ShootingRules was called for every probability frame against centroid ground truth at **≥0.5**. All 20 per-sequence TP/GT/FP counts exactly match the shards, merged results, and original author log.
- Pixel IoU was independently computed against full masks using **>0.5**. Intersection = 50,785; union = 59,156; IoU = 0.8584927987017378. The author log prints 0.858493. No saved probability equals exactly 0.5.
- All 27 saved threshold arrays were reassembled in original order and compared with the merged NPZ. AUC was independently integrated by adjacent trapezoids and checked against the untouched official summary function; the regenerated summary log is byte-identical to the saved local aggregate log.

| Set | TP / GT at 0.5 | FP pixels | Pixel denominator | Pd (%) | Fa (×10⁻⁵) | AUC |
|---|---:|---:|---:|---:|---:|---:|
| all | 1724 / 1729 | 3558 | 132309100 | 99.710815500 | 2.689157435 | 0.997778392104 |
| low_snr | 525 / 529 | 870 | 52604700 | 99.243856333 | 1.653844618 | 0.995467950776 |
| high_snr | 1199 / 1200 | 2688 | 79704400 | 99.916666667 | 3.372461244 | 0.998759016639 |

## Warnings and interpretation

- FP differs from the original author log at 20/540 sequence-threshold cells, by 1–3 pixels, only at thresholds <= 1e-6; 0.5 counts match exactly. Cause is not established.
- The author did not supply floating-point probability maps or exact IoU intersection/union; pixelwise author equivalence and exact unrounded author IoU cannot be established.
- Only threshold 0.5 was independently recomputed from all saved probabilities and ground truths. The other 26 thresholds were independently reaggregated from saved counts, not rerun from probabilities.
- This verifies saved official-checkpoint inference results, not from-scratch training, the unknown checkpoint training seed, or real-video performance.
- REPORT.md still marks independent full-result recheck PENDING; this audit leaves REPORT.md untouched and supplies completion evidence separately.

All 540 TP cells and all 540 GT cells equal the author log. FP has exactly the 20 reported differences, with maximum absolute difference 3. AUC matches the author log to 5 decimal places and the README to 4; Pd/Fa match README display precision. The report correctly discloses this boundary and does not establish the source of tiny FP differences.

Author-count AUC calculations reuse the independently verified local pixel denominators because the original log does not publish exact denominators. Author IoU is only available at the printed 6-decimal precision.

## Files

`RESULT_RECHECK.json` contains checks and detailed comparisons; `per_frame_recomputed.json` contains all 2,000 frame counts and intersections/unions; `input_hashes.json` records artifact and decoded-data hashes; `independent_reassembled_counts.npz` records independent arrays; `independent_official_aggregate_eval.txt` records the original summary output.
