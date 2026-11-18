# Independent screen audit

Reviewer: separate collaboration agent `/root/real_motion_annotation_screen/rule_audit`.

Status: no remaining blocking issue. Read-only audit conducted before raw-image acquisition; no prediction/model-output files read.

Source checks identified 277 duplicate raw track/frame keys, category changes on DJI_0083_2 / 6602004, out-of-bounds boxes, and compacted frame_id. Implementation rejects ambiguous frames before eligibility filtering, breaks runs on category changes, requires in-bounds boxes, and uses raw numeric filenames. Train/test manifests are parsed separately then unioned. Six frozen source/reference hashes matched.

Implementation checks covered all frozen thresholds, linear quantiles, disjoint 40-frame blocks, lag-5/quarter/net statistics, numeric track ordering, length fallback, all-frame 12 px membership, retention metadata and deterministic queue ordering.

AST-extracted numerical tests passed low/moving, acceleration/deceleration, 45-degree turn and low-moving-low classifications. A single 80 px impulse or persistent step at every block position never established moving. Run lengths 119/120/159/160/161 produced 0/1/40/1/2 evaluated windows. A single 13 px frame removed only windows containing it from the nested 12 px stratum. Final bookkeeping changes did not change the event rule or window eligibility.
