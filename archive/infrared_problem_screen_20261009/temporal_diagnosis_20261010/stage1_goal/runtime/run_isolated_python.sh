#!/usr/bin/env bash
# Usage: STAGE1_GPU=0 bash run_isolated_python.sh /absolute/stage1_goal/script.py [args...]
# Adapter only. It neither chooses a training protocol nor launches one by itself.
set -euo pipefail
project_root=/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009
if [[ $# -lt 1 ]]; then
  echo 'usage: run_isolated_python.sh /absolute/stage1_goal/script.py [args...]' >&2
  exit 2
fi
case "$1" in
  "$project_root"/temporal_diagnosis_20261010/stage1_goal/*.py) ;;
  *) echo 'script must be inside stage1_goal' >&2; exit 2 ;;
esac
docker run --rm --gpus all --user 1004:1004 --cpus 4 --memory 12g \
  -e CUDA_VISIBLE_DEVICES="${STAGE1_GPU:-0}" \
  -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 \
  -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONUNBUFFERED=1 -e MPLCONFIGDIR=/tmp/deeppro-matplotlib \
  -v /home/ly:/home/ly \
  -v "$project_root/official_deeppro:$project_root/official_deeppro:ro" \
  --entrypoint python deeppro-official-eval:20261009 "$@"
