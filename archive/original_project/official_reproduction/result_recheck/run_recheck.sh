#!/usr/bin/env bash
set -euo pipefail
task_output_dir=/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/official_reproduction/result_recheck
task_source_dir=/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/official_deeppro
docker run --rm --network none --user 1004:1004 \
  --entrypoint /opt/conda/bin/python \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 \
  -v /home/ly:/home/ly:ro \
  -v "$task_source_dir:$task_source_dir:ro" \
  -v "$task_output_dir:$task_output_dir:rw" \
  -w "$task_output_dir" \
  deeppro-official-eval:20261009 -u recheck_full_results.py \
  > "$task_output_dir/recheck.log" 2>&1
