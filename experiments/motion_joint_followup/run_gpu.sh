#!/usr/bin/env bash
set -euo pipefail
gpu="$1"
shift
case "$gpu" in 1|2|3) ;; *) exit 2 ;; esac
project="$(cd "$(dirname "$0")/../.." && pwd)"
experiment_dir="$(cd "$(dirname "$0")" && pwd)"
exp="${experiment_dir#"$project/"}"
exec docker run --rm --entrypoint /opt/conda/bin/python --gpus "device=$gpu" --user "$(id -u):$(id -g)" --cpus 4 --memory 20g --shm-size 1g \
    -v "$project:/project:ro" -v "$project/$exp/runs:/project/$exp/runs:rw" \
    -e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=2 -e PYTHONPATH=/project/environment/site \
    -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -w "/project/$exp" \
    sha256:eb956873870d737f2b68dff0724b8f6f56f778829d119b0938d29b7abeaa2f92 "$@"
