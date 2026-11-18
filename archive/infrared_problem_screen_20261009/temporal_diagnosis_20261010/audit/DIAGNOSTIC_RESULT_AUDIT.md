# Independent audit of four-clip diagnostic results

**Verdict: PASS for the declared threshold-0.5 point-response counts, with interpretation limitations below. No numerical correction is required.**

Audit date: 2026-10-10. The audit independently recomputed all 640 frames from saved fused probabilities and original annotation JSON files. It did not import, copy, or call `analysis/analyze.py` functions. It ran on CPU in the existing image without installing packages or executing a detector.

## Independent implementation

[recheck_diagnostic.py](recheck_diagnostic.py) implements:

1. Eight-neighbor connected components by explicit flood fill over foreground pixels with probability >= 0.5.
2. Component maxima and ties from sorted linear pixel indices, so the selected tied maximum is explicitly the first pixel in row-major order.
3. Pixel-center containment in original half-open xywh boxes, followed by an independently implemented augmenting-path maximum-cardinality matching. GT order and candidate ordering follow the recorded protocol: track/annotation ID, normalized squared center distance, descending score, component ID.
4. An additional exhaustive dynamic program over reachable GT subsets, independently verifying the maximum matching cardinality for every frame.
5. Per-track consecutive unmatched intervals using original source frame indices, breaking at missing frames or matched frames.

For every retained threshold-0.5 annotation, the independent matched flag agrees with `per_track_frame.csv`. Every component area, peak probability and peak coordinate agrees with `components.csv`. Every track's denominator, match count, longest unmatched interval, and complete unmatched interval list agrees with `track_summary.json`. Per-sequence summaries agree exactly with their entries in `ALL_CLIPS_SUMMARY.json`.

Full independent counts and intervals: [DIAGNOSTIC_RESULT_RECHECK.json](DIAGNOSTIC_RESULT_RECHECK.json).

## Recomputed results

| Clip | All GT point matches / GT frames | Components | Selected track point matches / GT frames | Selected track longest unmatched interval |
|---|---:|---:|---:|---:|
| DJI_0028_2 | 221 / 1280 | 234 | 154 / 160 | 4 |
| DJI_0028_4 | 354 / 1280 | 1313 | 158 / 160 | 1 |
| DJI_0032_5 | 1037 / 1748 | 10323 | 0 / 160 | 160 |
| DJI_0047_3 | 196 / 665 | 6052 | 17 / 160 | 142 |
| Sum | 1808 / 4973 | 17922 | 329 / 640 | Not additive |

There are 16114 unmatched components under this association rule. They are **not 16114 established false positives**. The summed denominators count annotated target-frames, including targets of different sizes and categories, rather than independent objects or independent trials. Do not treat the pooled ratio as a new benchmark result.

## Annotation and coordinate checks

- All four annotation files have no duplicate annotation IDs and no duplicate `(image_id, track_id)` keys.
- There are 128 partially outside boxes, all with negative x, in DJI_0028_2. They retain valid in-frame support and remain included. Independent pixel-center crop bounds reproduce every reported box maximum, including these boxes; no negative-index wraparound is present in the resulting values.
- Distinct GT identities can have identical boxes: DJI_0028_2 has 18 target-frame records with an identical other-track box, representing two tracks in nine frames. This is distinct from duplicate annotation records. A maximum-cardinality count remains defined, but assigning a particular identity in these frames is not observationally unique.
- All annotations were retained in these four clips; none failed the declared valid/nonignored/in-frame-area filter.

## Saturated plateaus materially limit the interpretation of point misses

The component representative is a protocol choice. Probability saturation often creates multiple maximal pixels with value exactly 1, and row-major ordering then picks the uppermost/leftmost maximum. This is correctly implemented, but a point miss is not necessarily a missing response.

| Clip | Components with tied maxima | Components with saturated maximum plateau | Point-unmatched GT frames with box max >= 0.5 | Such frames with another tied maximal pixel inside the box while chosen peak is outside |
|---|---:|---:|---:|---:|
| DJI_0028_2 | 194 | 194 | 127 | 82 |
| DJI_0028_4 | 662 | 649 | 2 | 2 |
| DJI_0032_5 | 2552 | 2423 | 48 | 2 |
| DJI_0047_3 | 775 | 715 | 22 | 1 |

For the **selected tracks**, all six point misses in DJI_0028_2 and both point misses in DJI_0028_4 contain a probability-1 maximal plateau pixel in the box while the row-major representative lies outside it. Thus the selected-track longest point-unmatched intervals of four and one frames cannot be presented as periods with absent model response.

For selected track 4801536 in DJI_0032_5, point matches are 0/160, but 23 of those frames have box maximum >= 0.5. Therefore “no associated component peak in any frame” is supported; “no response throughout all 160 frames” is not supported. In DJI_0047_3, the selected track's 143 point-unmatched frames all have box maximum below 0.5 in the saved fused predictions.

An alternative maximal pixel inside a box is not an independent detected object and does not prove that a different representative would recover the target without affecting another assignment. Likewise, a positive box maximum can come from a neighboring response or merged component. These supplementary observations explain representation sensitivity; they do not replace the frozen metric or justify selecting a favorable postprocessing rule.

## Claim boundaries

The recorded results are **one-to-one bounding-box point-response diagnostics**. They are not the paper's Pd, Fa, mask IoU, AP, or a calibrated object recall. Official evaluation uses different ground-truth representations and association rules. The original reports already disclose this distinction.

The independent recheck supports the arithmetic and engineering protocol, not a conclusion that acceleration, turning, fixed temporal weights, or lack of alignment caused the misses. Annotation visibility, scene differences, target contrast, box overlap, clip context and temporal position still matter. Four prescreened clips do not establish a general defect or a CVPR-level method gap.

This audit fully recomputed threshold 0.5 only. It did not independently recompute all threshold-0.1/0.9 statistics or the separate pre-fusion overlap-window comparisons and supplemental centroid sensitivity tables. Those outputs must not inherit this audit's numerical scope.
