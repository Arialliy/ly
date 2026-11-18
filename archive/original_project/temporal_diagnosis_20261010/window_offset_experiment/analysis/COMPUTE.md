# CPU replay

This analysis reuses the existing `deeppro-official-eval:20261009` environment without GPU access, installation, model changes or training.

```bash
docker run --rm --user 1004:1004 --cpus 4 --memory 12g -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 -e PYTHONDONTWRITEBYTECODE=1 -e MPLCONFIGDIR=/tmp/deeppro-matplotlib -v /home/ly:/home/ly --entrypoint python deeppro-official-eval:20261009 /home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/temporal_diagnosis_20261010/window_offset_experiment/analysis/analyze_offsets.py /home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009
```

The completed run exited0; stdout is retained in `run.log`. Each clip reuses all five original prediction windows and six new windows. Window hashes, frozen protocol hash, shared primary matching implementation hash and offset-analysis implementation hash are recorded in each `SUMMARY.json`.

All target annotations, fixed thresholds0.1/0.5/0.9, original matching definitions and all four annotation-prescreened clips are retained. Additional conservative and scale strata do not replace primary counts. These are diagnostic GT-frame counts, not independent samples, official Pd/Fa/AP or proposed-method improvements.
