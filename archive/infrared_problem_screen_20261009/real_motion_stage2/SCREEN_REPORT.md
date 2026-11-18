# VT-Tiny-MOT real motion annotation prescreen

Status: completed annotation-only candidate screen. These are image-coordinate motion candidates, with no established physical stop/restart or model failure. Raw video inspection is the next stage.

## Fixed rule and source separation

`PRESCREEN_RULE.json` was written and hashed before the first motion scan. SHA-256: `99d2eeefdf413cb4c30f667721e000dd438775f2b4aad43eacec27fcfa14efd7`. The scan uses the existing thermal `instances_01_train2017.json` source, totaling 34,111 images, 456,796 annotations and 85 nonempty sequences (86 video metadata records). Exact sequence IDs from **both** official DeepPro RGB-T train (45 IDs) and test (15 IDs) manifests are excluded. The source contains 45 train matches and zero test matches, leaving **40 development sequences** and 369,165 annotations. This is extra development data, not an official validation/test split; absent sequence IDs do not prove independent capture scenes.

Valid original tracks require one annotation per raw numeric frame, positive in-bounds boxes, ignore=iscrowd=0, unchanged category/video, and no missing raw frames. All 277 duplicate source track/frame keys (554 annotation rows) are rejected before size/ignore filtering. Compacted `frame_id` is never the continuity key. The source has no visibility/occlusion field, so continuity means annotated presence pending visual verification.

The primary clip has 160 frames with max(width,height)≤16 px in every frame. Maximal valid runs of 120–159 frames produce a separately reported 120-frame fallback. Every sliding window is enumerated with stride one. Four (or three) consecutive disjoint 40-frame blocks are compared. Block speed is median 5-frame displacement/5, with lower-quartile and 3-of-4 ten-frame-quarter support for movement. Low motion has p90 speed≤0.15 px/frame, net speed≤0.15, and all quarter speeds≤0.20; movement requires median≥0.50, p25≥0.30 and at least three quarter speeds≥0.30. Acceleration/deceleration requires ≥2× robust speed ratio and ≥0.40 px/frame change between moving blocks; a turn requires ≥45° between coherent moving blocks. Whole-clip raw center-step and box-side guards reject conspicuous jumps. All thresholds remain those frozen before scanning.

## Scale and candidate coverage

| Stratum | Continuous runs | Evaluated windows | Passing windows | Passing all-frame ≤12 px windows | Tracks | Sequences |
|---|---:|---:|---:|---:|---:|---:|
| Primary 160 frames | 456 | 54,535 | 1,534 | 1,206 | 47 | 16 |
| Fallback 120 frames | 192 | 3,273 | 227 | 86 | 15 | 5 |

There are 237,732 valid primary-scale annotation rows, including 169,784 at ≤12 px. The separate size-continuity count finds 296 uninterrupted ≤12 px runs of at least 160 frames and 128 of 120–159 frames. The nested candidate stratum always requires **every frame of the evaluated window** to satisfy ≤12 px; the standalone run counts do not change fallback eligibility.

Artifact guards reject 723 primary windows, including 5 with an otherwise qualifying tag; fallback guard rejections are 74. All 57,808 evaluated windows and all 1,761 accepted windows are retained. **Windows overlap, tags overlap, and counts are not independent events.** `77` first-window representatives are retained per sequence/track/tag/length stratum.

## Motion tag coverage and first examples

The examples below are the earliest 160-frame candidates in the same fixed sequence/track/frame order, without ranking by motion magnitude.

| Annotation tag | 160-frame windows | ≤12 px windows | Tracks | First candidate: sequence / track / raw frame interval | Compared median speeds (px/frame) |
|---|---:|---:|---:|---|---|
| low_to_moving | 103 | 92 | 5 | `DJI_0167_1` / `1801501` / 122–281 | 0.000 → 0.600 |
| moving_to_low | 21 | 19 | 1 | `DJI_0331_3` / `2101685` / 43–202 | 0.539 → 0.100 |
| acceleration | 410 | 293 | 24 | `DJI_0028_2` / `3800006` / 173–332 | 0.500 → 1.005 |
| deceleration | 380 | 348 | 10 | `DJI_0028_4` / `4400005` / 285–444 | 1.603 → 0.583 |
| turn | 992 | 799 | 20 | `DJI_0028_4` / `4400005` / 287–446 | 1.600 → 0.632 |
| low_moving_low | 0 | 0 | 0 | None under frozen rule | — |
| moving_low_moving | 14 | 12 | 1 | `DJI_0331_3` / `2101685` / 83–242 | 0.539 → 0.100 → 0.894 |

## Selected next inspection clip

**`DJI_0028_2` / track `3800006` / raw frames 173–332 inclusive** is the first primary candidate. Category: `ship` (source category ID 0); source video ID 38; metadata FPS 15 (time between first/last frames 10.6 seconds). Boxes have width 5–9 px, height 6–9 px, and median maximum side 8 px. The selected tag is `acceleration`: blocks 213–252 and 253–292 have median lag-5 speeds 0.500 and 1.00499 px/frame (ratio 2.00998). This selection was determined by sequence, numeric track ID, and earliest start, not model outputs or candidate strength.

`SELECTED_CANDIDATE.json` contains all 160 raw filenames, original image/annotation/tracking IDs, bbox centers/size, and all four block statistics. `INSPECTION_QUEUE.json` contains the first candidate from each of the first 12 passing sequences for bounded later review. `ELIGIBLE_TRAJECTORIES.jsonl.gz` resolves every retained candidate's `run_id`, `run_frame_offset`, and `frame_count` to exact source IDs. `SPLIT_PROVENANCE.json` records source hashes and excluded manifest IDs.

## Checks and limits

`MANIFEST_VALIDATION.json` reports PASS for all 1,761 accepted windows, 57,808 evaluated windows and 648 eligible runs: hashes match, source IDs resolve, candidate sets equal passing evaluated rows, all source frames are consecutive, all size strata are correct, exclusions hold and deterministic selections agree. A separate agent audited source schema, implementation and numeric edge cases. Numerical tests covered low/moving states, acceleration/deceleration, 45° turns, a single impulse/persistent jump at every block position, 119/120/159/160/161-frame enumeration, and removal of only affected windows from the 12 px stratum. No blocking issue remained.

No model predictions, masks, scores, or error results were used. Camera motion, annotation interpolation/occlusion, physical motion and actual failures remain unverified. Coordinate acceleration/deceleration labels do not establish physical acceleration/deceleration. Detection boxes and rectangular segmentation polygons are not verified pixel masks. All eligible-run metadata report 15 FPS; this is source metadata, not an independently measured capture rate.

Reproduce with `python3 real_motion_stage2/screen_annotations.py` followed by `python3 real_motion_stage2/verify_manifests.py` from the project directory. Both use the standard library on CPU.

## Authorized raw-video handoff

After the annotation-only screen and independent checks completed, the next-stage instruction authorized downloading exactly the fixed selected clip. `selected_raw/VALIDATION.json` reports PASS: **160 original thermal JPEGs, raw frames 173–332 inclusive, all native grayscale mode L and 640×512**. Raw bytes are unchanged and total 1,518,675 bytes; returned HTTP body bytes total 1,231,344 including the public download form, below the 50 MB bound. An initial TLS handshake failure returned no counted body bytes and is preserved in `selected_raw/NETWORK_TRANSFER_ATTEMPT_1.json`; the exact retry succeeded.

`selected_raw/FRAME_MANIFEST.json` binds every raw filename to original image/annotation IDs, selected-target boxes, local path and SHA-256. `selected_raw/annotations.json` preserves **all 1,280 source annotations across 8 tracks** in the selected frames, including other targets. `selected_raw/DOWNLOAD_PROVENANCE.json` records source, helper, reduced manifest and transfer details.

The fixed visual inspection frames 173, 213, 253, 293 and 332 are shown in `selected_raw/CONTACT_SHEET.png`; individual `INSPECTION_FRAME_00xxx.png` panels retain raw-view, box overlay and zoomed crop views. These are derived inspection aids; raw JPEGs remain unchanged. Qualitative camera-motion/visibility inspection is still pending. No detector inference or compensated-motion claim has been made.
