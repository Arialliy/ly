# CPU feasibility audit: dwell/lifetime proxies

Reproduce: `python3 /home/ly/DeepProPlus_TLFP_Code/idea-stage/pilot_feasibility_20261117/diagnose.py`

No training, GPU inference, confirmation data, or historical-file edits. See RESULTS.json for complete per-track records, source hashes and limitations.

| Split | Tracks | Target frames | Low >=8 tracks/runs | Low >=16 tracks/runs | Low >=32 tracks/runs | Stop-go >=8 runs |
|---|---:|---:|---:|---:|---:|---:|
|train|129|16707|49/156|27/55|12/16|132|
|calibration|226|16297|13/37|9/18|5/6|33|

Release O longest miss distribution: {"n": 130, "min": 0, "median": 7.0, "p75": 16.0, "p90": 25.300000000000026, "p95": 71.24999999999996, "max": 160, "mean": 15.823076923076924, "ge8": 59, "ge16": 35, "ge32": 11}
Never matched within full strict observation: 82/130 tracks.

## Limits

- 5-frame net displacement means Euclidean distance center[t]-center[t-5], requiring 6 consecutive observations; low endpoint iff distance <1px. Run length counts consecutive qualifying endpoint frames (not the union of their preceding five frames).
- All nonignore/noncrowd finite positive-area bounding boxes are used for motion feasibility, without strict-small/nonoverlap filtering. Track counts are clip+track-ID units, not independent physical entities or scenes.
- Integer/half-pixel box centers, annotation quantization/jitter, reversal/circling, camera motion and unknown visibility/occlusion prevent treating low displacement as physical stationary truth. No motion compensation or image inspection performed.
- Appearance/disappearance means annotation presence boundary inside the frozen 160-frame clip/window. It may reflect occlusion, missing annotation or truncation; not verified physical birth/death. Overlapping windows repeat events.
- Low/extreme cohorts are defined by low released-model response, so their complete missing is expected/selection-conditioned. Cohort filtering creates gaps; full-strict track never-detection is separately reported.
- Only release O once per annotation; strict observation may omit larger/overlapping portions of a physical track. No official Pd/Fa, no new prediction or model-success conclusion.
- No confirmation annotation/image/prediction was read. DATA_SELECTION contains confirmation names only. No GPU library or model loaded.
- False-alarm run lengths cannot be inferred from per-target matching tables without temporal association of unmatched components; they were not estimated.

## Development overlap with strict evaluation

{"8": {"strict_frames": 302, "strict_tracks": 7, "matched": 82, "low_frames": 220, "low_tracks": 6, "extreme_frames": 220, "extreme_tracks": 6, "per_clip": {"DJI_0026_2": {"strict_frames": 211, "low_frames": 129, "extreme_frames": 129}, "DJI_0067_2": {"strict_frames": 91, "low_frames": 91, "extreme_frames": 91}}}, "16": {"strict_frames": 190, "strict_tracks": 4, "matched": 49, "low_frames": 141, "low_tracks": 3, "extreme_frames": 141, "extreme_tracks": 3, "per_clip": {"DJI_0026_2": {"strict_frames": 166, "low_frames": 117, "extreme_frames": 117}, "DJI_0067_2": {"strict_frames": 24, "low_frames": 24, "extreme_frames": 24}}}, "32": {"strict_frames": 103, "strict_tracks": 2, "matched": 0, "low_frames": 103, "low_tracks": 2, "extreme_frames": 103, "extreme_tracks": 2, "per_clip": {"DJI_0026_2": {"strict_frames": 91, "low_frames": 91, "extreme_frames": 91}, "DJI_0067_2": {"strict_frames": 12, "low_frames": 12, "extreme_frames": 12}}}}

In particular, all dwell >=16 endpoints contain only 3 low-response track(s). This is data coverage, not causal evidence that dwelling causes a miss.
