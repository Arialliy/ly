"""Single preregistered frozen-checkpoint auxiliary inference job."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import time

N=Path(__file__).resolve().parents[1]
R=N.parent.parent
IMAGE='sha256:eb956873870d737f2b68dff0724b8f6f56f778829d119b0938d29b7abeaa2f92'
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--sequence',choices=['DJI_0026_2','DJI_0067_2'],required=True)
    p.add_argument('--gpu',type=int,choices=[2,3],required=True)
    a=p.parse_args()
    out=N/'diagnostic_predictions/N2_misreference'/a.sequence
    receipt=N/'runtime'/f'misreference_{a.sequence}.json'
    log=receipt.with_suffix('.log')
    if receipt.exists() or log.exists():raise FileExistsError('No duplicate diagnostic invocation')
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    freeze=json.loads((N/'PROTOCOL_FREEZE.json').read_text())
    assert sha(N/'PROTOCOL.md')==freeze['protocol_sha256']
    for path,expected in freeze['code_sha256'].items():assert sha(N/path)==expected,path
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    usage={int(r.split(',')[0]):int(r.split(',')[1]) for r in rows.splitlines()}
    if usage[a.gpu]>=500:raise RuntimeError(f'GPU{a.gpu} occupied; no sharing or termination')
    script=N/'diagnostic_code/infer_misreference.py'
    command=['docker','run','--rm','--name','lre_misref_'+a.sequence.lower(),'--network','none','--gpus','all',
             '--user',f'{os.getuid()}:{os.getgid()}','--cpus','4','--memory','20g','--shm-size','2g',
             '-e',f'CUDA_VISIBLE_DEVICES={a.gpu}','-e','OMP_NUM_THREADS=2','-e','MKL_NUM_THREADS=2',
             '-e','OPENBLAS_NUM_THREADS=2','-e','PYTHONDONTWRITEBYTECODE=1','-e','PYTHONUNBUFFERED=1',
             '-e','CUBLAS_WORKSPACE_CONFIG=:4096:8','-e','MPLCONFIGDIR=/tmp/mpl',
             '-v',f'{R}:{R}:ro','-v',f'{N}:{N}:rw','-w',str(N),'--entrypoint','/opt/conda/bin/python',IMAGE,
             str(script),'--sequence',a.sequence,'--output',str(out)]
    record={'status':'RUNNING','command':command,'gpu':a.gpu,'gpu_before':usage,'start_unix':time.time(),
            'diagnostic_source_sha256':sha(script),'protocol_freeze_sha256':sha(N/'PROTOCOL_FREEZE.json'),
            'trigger_sha256':sha(N/'DIAGNOSTIC_TRIGGER.md'),'formal_training_updates':0}
    receipt.write_text(json.dumps(record,indent=2)+'\n')
    print('START',a.sequence,'GPU',a.gpu,flush=True)
    with log.open('w') as stream:result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
    record.update(status='COMPLETE' if result.returncode==0 else 'FAILED',exit_code=result.returncode,end_unix=time.time())
    receipt.write_text(json.dumps(record,indent=2)+'\n')
    print('END',a.sequence,record['status'],flush=True)
    raise SystemExit(result.returncode)
if __name__=='__main__':main()
