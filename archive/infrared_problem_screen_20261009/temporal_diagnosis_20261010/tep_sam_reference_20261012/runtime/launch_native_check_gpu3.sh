#!/usr/bin/env bash
set -euo pipefail
tep_stage=/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/temporal_diagnosis_20261010/tep_sam_reference_20261012
tep_project=/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009
tep_expected=sha256:eb956873870d737f2b68dff0724b8f6f56f778829d119b0938d29b7abeaa2f92
[[ $(docker image inspect deeppro-official-eval:20261009 --format '{{.Id}}') == "$tep_expected" ]]
[[ ! -e "$tep_stage/native_check" ]]
tep_usage=$(nvidia-smi --id=3 --query-gpu=memory.used --format=csv,noheader,nounits)
if (( tep_usage >= 500 )); then printf 'GPU3 not idle: %s MiB\n' "$tep_usage" >&2;exit 1;fi
docker run --rm --name tep_sam_native_check_gpu3_20261012 --network none --user 1004:1004 --cpus 4 --memory 20g --shm-size 2g --gpus all -e CUDA_VISIBLE_DEVICES=3 -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -e MPLCONFIGDIR=/tmp/tep-mpl -v "$tep_project:$tep_project:ro" -v "$tep_stage:$tep_stage:rw" -v "$tep_stage/author_source:$tep_stage/author_source:ro" -w "$tep_stage/author_source" --entrypoint python deeppro-official-eval:20261009 "$tep_stage/runtime/native_check.py" --output "$tep_stage/native_check" 2>&1 | tee "$tep_stage/runtime/native_check_gpu3.log"
