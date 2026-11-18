# CPU diagnostics

`analyze.py` implements the frozen parent `DIAGNOSTIC_PROTOCOL.json` for the first four entries of the earlier annotation-only inspection queue. No model is trained or executed here. Parameters are not selected from observed probabilities.

## Reproduce

```bash
docker run --rm --user 1004:1004 --cpus 4 --memory 12g -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 -e PYTHONDONTWRITEBYTECODE=1 -e MPLCONFIGDIR=/tmp/deeppro-matplotlib -v /home/ly:/home/ly --entrypoint python deeppro-official-eval:20261009 /home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/temporal_diagnosis_20261010/analysis/analyze.py --root /home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009
```

An optional `--sequence DJI_0028_2` runs one completed clip. Missing predictions are explicitly recorded as pending; incomplete ZIP output should be allowed to finish before running. The existing inference environment is reused without GPU exposure, package installation, or image modification.

## Exact choices

- Probability threshold comparison is `>=` at 0.1, 0.5 and 0.9.
- Components are 8-connected. Peak ties are explicitly resolved by row-major pixel index; SciPy's unspecified maximum-position tie behavior is not relied upon.
- Pixel centers `(x+0.5,y+0.5)` must be inside original half-open COCO xywh boxes. Any positive in-frame portion of a valid, nonignored box is retained; negative and overflowing crop indices are clamped before indexing. Partly outside boxes remain flagged.
- GTs are ordered by track ID and annotation ID. Maximum-cardinality augmenting-path matching uses adjacency sorted by normalized squared center distance, descending peak score and component ID. This is deterministic but does **not** claim global minimum-distance assignment.
- Both prediction-graph ambiguity and annotation overlap/identical boxes are recorded. The conservative unambiguous overlap comparison excludes GT boxes intersecting any other retained GT as well as nonunique candidate associations. Primary counts retain all valid targets.
- The same original frame in two overlapping windows is compared before fusion. Saved full probabilities must equal the exact elementwise maximum of saved window probabilities, and coverage must agree; both are asserted.
- Global frame/track/threshold tables are primary evidence. Selected/weak-track crop illustrations are generated only after those tables, with their deterministic post-statistic selection rule recorded. The lowest-response illustration requires at least40 annotated frames. Such illustrations are not independent validation.
- Matching, row-major ties, maximum-cardinality rerouting, and partially/fully outside boxes receive small functional self-checks before any input analysis. These are engineering checks, not measured detection outcomes.

## Outputs and limits

Every sequence directory includes per-track/frame CSV, frame counts, all components, track summaries with consecutive unmatched intervals, four-block summaries, same-frame overlap comparisons, all-track plots, illustrative contacts, input/source hashes and artifact hashes.

The metric is a **bbox point-response diagnostic**, not official Pd, Fa, IoU, AP or calibrated object recall. Unmatched components are not automatically false positives. Low thresholds can merge components, so object matching need not change monotonically with pixel threshold. Annotation presence does not prove full visibility, identical boxes can make identity assignments arbitrary, and window-context changes do not isolate slot weighting from surrounding content or temporal padding.

This script does not establish that motion caused any observed error, nor that a new mechanism is required. Four prescreened development clips do not constitute a new independently sampled benchmark.
