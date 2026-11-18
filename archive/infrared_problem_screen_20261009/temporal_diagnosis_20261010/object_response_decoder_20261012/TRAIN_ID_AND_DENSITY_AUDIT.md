# Training identity and query-capacity audit

This audit reads only the eight frozen training annotation subsets, their original public source JSON, selection manifest, and old 512-step exposure plan. It does not read development annotations, predictions, model outputs, or image pixels. No old data, annotations, or evaluation identities are changed.

**Decision:** the explicitly namespaced author field `(sequence, category_id, str(tracking_id))` is available for R2 training supervision. Retain the numeric `track_id` unchanged for provenance and the frozen evaluation. Source provenance and consistency checks support using the provided annotation identity; they do not certify perfect physical identity or visibility at every frame.

## Provenance and identity limitation

The [RGBT-Tiny author repository](https://github.com/XinyiYing/RGBT-Tiny#mot-label) describes bounding boxes with tracking IDs and links its MOT labels to [HGTMT](https://github.com/xuqingyu26/HGTMT). The latter links the VT-Tiny-MOT public archive already acquired by this project. Re-read on 2026-10-12; no contact or additional dataset access requested.

All **16,707 training annotation objects and 1,280 image records** match their public-source JSON objects exactly, including both `tracking_id` and `track_id`. Thus neither field is an identity invented by this project's diagnostic grouping. The uncompressed source hash is `39f19472ad2dddd5b4cbf1aebb35450ccf412841b9661d4904ffbe08eccce125`, identical to the original download receipt. Its locally compressed copy has SHA-256 `d60553eb3c537f91fbf778d995d1c978071cdd978e006834186e58de55b8082b`.

The numeric field is **not interchangeable** with the original label: 129 numeric IDs map to 112 `(sequence, category, tracking_id)` groups. Seventeen original-label groups contain two numeric IDs. There are 143 changes of numeric ID between adjacent annotated frames within those groups. Their center displacement has median 1.118 pixels and maximum 4.123 pixels. For example, pedestrian `tracking_id='64'` in DJI_0043_2 has numeric ID 8721564 at frames 121–124, 8601564 at frame 125, and 8721564 at frame 126, with continuous boxes. These changes are already present in the public source.

The [HGTMT training loader](https://github.com/xuqingyu26/HGTMT/blob/main/datasets/generic_dataset_rgbt_gnnloss.py) consumes numeric `track_id` for temporal correspondence. No public conversion rule explaining the observed numeric aliases was found. This audit therefore does **not** claim to reproduce the HGTMT identity handling, reverse its conversion, or repair author labels. Using the source `tracking_id` with category and sequence namespaces is an explicit supervision choice in this new experiment. It must not silently change the historical 130-track evaluation grouping.

Checks over the eight training clips:

- No missing identity fields, duplicated image/annotation records, or per-frame collisions under either numeric ID or the namespaced original label.
- Every numeric ID maps to one category and one original label. Category is part of the training key because the same bare original ID can be reused across categories.
- All 112 original-label groups have one contiguous annotated run within the selected material: 16,595 adjacent transitions = 16,707 observations − 112 groups. There are no internal annotation gaps in these groups.
- Largest adjacent center motion is 17.5 pixels for the plane in DJI_0309_1; maximum displacement divided by the larger adjacent box diagonal is 1.646. Numeric-ID switch transitions are small as reported above. These are integrity observations, not visual proof of identity.
- There are no visibility, occlusion, or explicit absent-state fields. `ignore` and `iscrowd` exist and happen to be zero for all these training annotations. Absence before or after a track, clipping by a crop, excluded annotations, and any future gap must not automatically become a negative existence label for that assigned object.

Native source frame numbering is preserved: DJI_0309_1 starts at source frame 1; DJI_0323_2 starts at 234; the other six start at 0. All selected native frame numbers and `mot_frame_id` values advance by one. Array indices remain 0–159. No frame relabeling is required.

## Training-only density

| Training clip | Object frames | Numeric IDs | Original category/ID groups | Max original groups / 40 frames | Max concurrent objects |
|---|---:|---:|---:|---:|---:|
| DJI_0030_9 | 2,240 | 15 | 14 | 14 | 14 |
| DJI_0043_2 | 11,758 | 96 | 81 | 79 | 78 |
| DJI_0057_1 | 480 | 3 | 3 | 3 | 3 |
| DJI_0113_3 | 160 | 1 | 1 | 1 | 1 |
| DJI_0115_2 | 629 | 5 | 4 | 4 | 4 |
| DJI_0217_2 | 480 | 3 | 3 | 3 | 3 |
| DJI_0309_1 | 160 | 1 | 1 | 1 | 1 |
| DJI_0323_2 | 800 | 5 | 5 | 5 | 5 |

All 968 legal 40-frame training windows (121 starts × 8 clips) were enumerated. Maximum numeric-ID union is **91**; maximum original-category-ID union is **79**; maximum per-frame concurrency is **78**.

The original 512-step exposure plan has 2,048 crop-window exposures. For full-core-9 positives in those 128×128 crops, distinct original tracks per window have median 1, 90th percentile 4, 99th percentile 14, and maximum **19**. There are 385 empty crop windows and 148,902 positive object-frame exposures. These counts include all eligible categories and sizes, not only the strict small-target evaluation class. Intersecting but non-full-core objects are separately retained in the audit CSV so they cannot silently become background.

**Engineering choice approved by the parent before any new development evaluation: K=100, a fixed 10×10 global anchor grid.** This exceeds even the conservative numeric-ID 40-frame maximum of 91. K is not chosen from the crop maximum of 19 and must not reset to a smaller per-tile query set at inference. Each query may output multiple connected components, so K is not itself a hard connected-component or false-positive cap.

The new 2,048-step protocol repeats the frozen 512-step plan four times: 8,192 crop-window exposures, 1,540 empty exposures and 595,608 positive object-frame exposures. This increases updates, not unique scenes or unique training clips. R0/R1/R2 must share this exact exposure order.

## Crop-to-full-frame condition

Training still uses the old 128×128 crops. The decoder retains the global 10×10 anchors and absolute original-frame coordinates; crop offsets must enter positional encoding correctly. Full-frame inference may let each query read substantially more spatial context than a training crop. This is a distribution difference inherent to the fixed prototype, not evidence of a temporal mechanism. Chunking the full-frame token/query computation must reproduce the same global operator; independent reinitialization of queries in each tile would be another architecture.

## Reproducible artifacts

- `data_audit/audit_training_identity.py`: standard-library-only audit.
- `data_audit/TRAIN_ID_DENSITY_SUMMARY.json`: hashes, checks, and capacity decision.
- `data_audit/TRAIN_SEQUENCE_IDENTITY.csv`: native source-frame offsets and sequence totals.
- `data_audit/TRAIN_FULL40_DENSITY.csv`: all 968 training windows.
- `data_audit/TRAIN_CROP_DENSITY.csv`: all 2,048 old crop-window exposures, including excluded/intersecting objects.
- `data_audit/TRAIN_CANONICAL_IDENTITY.csv` and `TRAIN_CANONICAL_ADJACENT_MOTION.csv`: original source-key runs and all adjacent motion measurements.
- `data_audit/SOURCE_ID_ALIAS_GROUPS.csv` and `SOURCE_ID_ALIAS_ADJACENT_SWITCHES.csv`: 17 alias groups and 143 adjacent numeric-ID switches.
- `data_audit/TRAIN_NUMERIC_TRACK_RUNS.csv`: original numeric-ID fragmentation retained explicitly.

This is sufficient to start an author-ID-supervised prototype with the stated limits. It does not establish that trajectory supervision will improve detection, that all missing observations are absent objects, or that the prior failures were caused by identity inconsistency.

## Implemented target gate

`code/targets.py` and `code/test_targets.py` are complete. Sixteen CPU semantic tests pass in the frozen Docker image. An exhaustive pass over all 2,048 old window exposures also passes: every reconstructed background domain, valid domain, and union of object response supports is exactly equal to the legacy generator. The check took 34.56 seconds on CPU and performed no optimizer or GPU work. It writes the complete per-window semantic receipt without persisting dense masks.

The final builder counts differ appropriately from the earlier geometric approximation: **148,836 effective object-frame labels** remain from 148,902 full-core boxes; 66 supports are completely removed by excluded-object guards. No fallback points are created. The common negative qualification is true for **64,555/81,920 frames (78.80249%)**; the other 17,365 frames retain only valid positive supervision. Four repetitions yield 595,344 effective object-frame exposures and 258,220 frames with negative qualification. Background and valid domains are never empty in this plan.

`data_audit/frozen_target_receipts/MANIFEST.json` contains every window's identity, origin, counts, and semantic SHA. The ordered semantic stream SHA is `4fc387fa3921c1a8774f95446d6d42a33ab2223affbbd0460b6688740e649ceb`. Final target-source SHA is `79013f3650ae0ca18b447fc5fa4631e073ee81c2ee0ce84600d88879af23f06e`; test-source SHA is `981c7c9547393b1498692d821d208dec96d4d8d2f33ff82ec6fbd30d59e950fd`. `checks/TARGET_SEMANTIC_TESTS.json` binds those files and the exhaustive receipt.
