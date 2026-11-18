#!/usr/bin/env bash
set -euo pipefail
project_root=/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009
docker run --rm --gpus all --user 1004:1004 --cpus 4 --memory 12g \
  -e CUDA_VISIBLE_DEVICES="${STAGE1_GPU:-0}" \
  -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 \
  -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONUNBUFFERED=1 -e MPLCONFIGDIR=/tmp/deeppro-matplotlib \
  -v /home/ly:/home/ly \
  -v "$project_root/official_deeppro:$project_root/official_deeppro:ro" \
  --entrypoint python deeppro-official-eval:20261009 \
  "$project_root/official_reproduction/environment_witness.py"
