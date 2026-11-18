#!/usr/bin/env bash
set -euo pipefail
tep_stage=/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/temporal_diagnosis_20261010/tep_sam_reference_20261012
tep_project=/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009
tep_sequence=${1:?Pass DJI_0026_2 or DJI_0067_2}
tep_gpu=${2:?Pass permitted physical GPU 2 or 3}
case "$tep_sequence" in DJI_0026_2|DJI_0067_2) ;; *) exit 2 ;; esac
case "$tep_gpu" in 2|3) ;; *) printf 'User allocated GPUs 2 and 3 only\n' >&2;exit 2 ;; esac
tep_expected=sha256:eb956873870d737f2b68dff0724b8f6f56f778829d119b0938d29b7abeaa2f92
[[ $(docker image inspect deeppro-official-eval:20261009 --format '{{.Id}}') == "$tep_expected" ]]
[[ ! -e "$tep_stage/predictions/$tep_sequence" ]]
tep_usage=$(nvidia-smi --id="$tep_gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
if (( tep_usage >= 500 )); then printf 'GPU%s not idle: %s MiB\n' "$tep_gpu" "$tep_usage" >&2;exit 1;fi
python3 - "$tep_stage" "$tep_sequence" "$tep_gpu" "$tep_usage" <<'PY'
import datetime,hashlib,json,sys
from pathlib import Path
n=Path(sys.argv[1]);seq=sys.argv[2];gpu=int(sys.argv[3]);usage=int(sys.argv[4])
p=n/'PROTOCOL.json'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='9c73da144a4a4743c2cc36b6c57a33298cacf5b15468a03e0214f56e18b73374'
protocol=json.loads(p.read_text())
for name,value in protocol['source_sha256'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==value,name
native=json.loads((n/'native_check/COMPLETE.json').read_text())
assert native['status']=='PASS_NATIVE_RUNNING_CHECK_ONLY'
assert native['checkpoint_sha256']==protocol['author']['checkpoint_sha256']
assert native['state_unchanged'] and native['metric_export_equality_frames']==100
record={'status':'LAUNCHING','timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'user_allocated_physical_gpus':[2,3],'selected_physical_gpu':gpu,'memory_before_mib':usage,'sequence':seq,'seed':46,'batch_size':1,'fp32':True,'image_id':'sha256:eb956873870d737f2b68dff0724b8f6f56f778829d119b0938d29b7abeaa2f92','native_receipt_sha256':hashlib.sha256((n/'native_check/COMPLETE.json').read_bytes()).hexdigest(),'protocol_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'frozen_science_inputs_verified':len(protocol['source_sha256']),'inference_code_changed':False,'container_cuda_visible_devices':str(gpu),'gpu_access':'all CUDA devices mounted; visibility bound to one user-allocated physical GPU; no other GPU accessed by torch'}
(n/'runtime'/('RESOURCE_RUN_'+seq+'.json')).write_text(json.dumps(record,indent=2))
PY
docker run --rm --name "tep_sam_${tep_sequence}_20261012" --network none --user 1004:1004 --cpus 4 --memory 20g --shm-size 2g --gpus all -e CUDA_VISIBLE_DEVICES="$tep_gpu" -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -e MPLCONFIGDIR=/tmp/tep-mpl -v "$tep_project:$tep_project:ro" -v "$tep_stage:$tep_stage:rw" -v "$tep_stage/author_source:$tep_stage/author_source:ro" -w "$tep_stage/author_source" --entrypoint python deeppro-official-eval:20261009 "$tep_stage/runtime/infer_full_frames.py" --sequence "$tep_sequence" --output "$tep_stage/predictions/$tep_sequence" --native-receipt "$tep_stage/native_check/COMPLETE.json" --protocol "$tep_stage/PROTOCOL.json" 2>&1 | tee "$tep_stage/runtime/inference_${tep_sequence}.log"
