# Independent diagnostic audit

**PASS.** All requested numeric outputs were checked without importing the authors' analysis or trajectory functions. No model forward pass or training was added during this audit, and no existing logit or analysis output was modified.

| Layer | Independent coverage | Result |
|---|---|---|
| Native logit reproduction | Previously checked 16 windows and four O/S fused maps | Bitwise sigmoid and 0.5 masks unchanged |
| Raw object recomputation | 8 distinct frames and sizes per clip, both arrangements: 32 rows × all 132 numeric/identity columns | PASS |
| Fixed background references | All 130 tracks, all 16 candidate geometric offsets | PASS |
| Frozen full cohort | All 10,400 rows, labels, identities and old 0.5 matches | PASS |
| Table 1 | All 42 rows / 3,822 cells | PASS |
| Table 2 and Table 3 | 42 peak rows, 260 track rows, 42 track-summary rows / 95,686 cells | PASS |
| Final assembly | Original 132 columns preserved verbatim, 144 unique reviewed arrangement rows | PASS |

## Independent computation

`SAMPLE_SELECTION.csv` selects 16 target frames by identity hash with unique frame/size combinations per clip, using only the frozen cohort. Both arrangements are audited. The verifier reads raw native logits and original annotations, traverses same-sized background rectangles literally, computes quantiles manually, finds local maxima through eight shifted comparisons, and merges equal-valued peak platforms using an explicit stack. It does not call the authors' reshape reduction, block helper, maximum filter, connected-component labeler, quantile routine, or summary function.

All 130 tracks' background offset candidates were independently chosen from geometry alone. The local reference excludes itself plus a 2-pixel buffer; its global reference excludes all grid blocks intersecting itself. Every target and background block has the same per-frame crop dimensions and maximum-score definition. All annotation boxes participate in exclusion, including ignored ones in the implementation; this particular data contains zero ignore/iscrowd flags and integer positive-area boxes.

The finite peak-platform rule removes an entire equal-valued local maximum platform if any of its pixels touches the annotation buffer. This conservative choice is explicitly documented in the analysis README; its representative is the first row-major pixel. It cannot be mistaken for official false alarm counting.

## Precision finding and resolution

The first independent mathematical float64 reference differed in 30 `neighborhood_max_minus_q95` cells, with a maximum difference of 3.8623809786031416e-6. NumPy 2.2.5 keeps the original float32 array dtype for scalar quantiles and the following subtraction, including float32 rank/interpolation rounding. These were not data errors.

The initial failure report and logs remain in `INITIAL_PRECISION_DIFFERENCES.json` and `INITIAL_OBJECT_AUDIT.*`. The final verifier reconstructs rank, interpolation and subtraction with explicit float32 rounding, without calling NumPy's quantile and without loosening the numeric tolerance. All 5,524 sampled object/reference cells then pass; the largest residual is 3.552713678800501e-15 from ordinary float64 arithmetic. Discrete identities, counts, coordinates and flags agree exactly. The main target/background q95 comparisons use float64 block-score arrays and had no substantive discrepancy in either run. No research output was changed to obtain agreement.

## Denominators and continuity

The Table 1 main local/global comparison retains the preregistered minimum 20 valid blocks. Missing, 1–19 and >=20 denominators are separately reported; joint advantage categories use the intersection of valid denominators. The locally valid subset is geometry dependent, so its event rate must not be generalized to every low-response object or directly compared with a differently shaped success population.

Every Table 3 field was reconstructed independently. Local-frame and source-frame indices must both advance by one. Missing observations, false predicates, invalid references, insufficient backgrounds and group-filtered-away observations break the applicable runs; neither gap is bridged. Frame-weighted ratios pool numerators/valid denominators, while equal-track ratios average within-track rates only among tracks having a valid denominator. Top-track contributions use each selected group's frame count. Stationary, less-than-one-pixel, and moving pairs remain distinct. Box-center motion is a geometric reference, not a ground-truth target center or a causal attribution.

The final table contains 72 uniquely reviewed target frames, or 144 arrangement rows; all other 10,256 rows have no visual attribution. Each neighboring frame retains its own actual frozen group. Counts of own-box, other-box and outside-all-box peaks partition each denominator exactly. Reviewed target/background/uncertain categories partition only the reviewed rows. Manual labels are not extrapolated to all 5,200 target frames.

## Scope

This audit establishes computation and claim-scope consistency for a bounded diagnostic round. It does not validate visual labels as objective ground truth, establish a deployable detector gain, or replace background cost with local maxima counts. The original 0.5 matching result is reused exactly.

The applied `analyze-results` skill requires raw tables, key findings and next-experiment implications; the deliverables satisfy those requirements. `run-experiment` requires resource/process commands and logs, already present for the warm-reused native-logit export. Neither skill mandates an additional final review loop for this unchanged environment.

Evidence: [sampled raw-object recomputation](OBJECT_RECOMPUTATION.json), [Table 1 recomputation](SCORE_SUMMARY_AUDIT.json), [final tables and continuity audit](FINAL_TABLES_AUDIT.json).
