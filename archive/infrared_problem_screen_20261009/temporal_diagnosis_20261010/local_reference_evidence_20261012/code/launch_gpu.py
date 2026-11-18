"""One fixed isolated GPU job; never stop or share an occupied device."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

N=Path(__file__).resolve().parents[1]
Q=N.parent
R=Q.parent
IMAGE='sha256:eb956873870d737f2b68dff0724b8f6f56f778829d119b0938d29b7abeaa2f92'
SOURCE=R/'official_deeppro'
CHECKPOINT=SOURCE/'log/sem_seg/RGB-T__2025-01-14_15-27__SoftLoUloss_DeepPro-Plus_DataL40/checkpoints/best_model.pth'

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--kind',choices=['preflight','train','infer'],required=True)
    p.add_argument('--gpu',type=int,choices=[2,3],required=True)
    p.add_argument('--condition',choices=['N0','N1','N2'])
    p.add_argument('--sequence',choices=['DJI_0026_2','DJI_0067_2'])
    a=p.parse_args()
    if a.kind!='preflight' and a.condition is None: p.error('condition required')
    if a.kind=='infer' and a.sequence is None: p.error('sequence required')
    label='_'.join(str(v) for v in [a.kind,a.condition,a.sequence] if v)
    receipt=N/'runtime'/f'{label}.json'
    log=N/'runtime'/f'{label}.log'
    if receipt.exists() or log.exists(): raise FileExistsError('Immutable job already exists: '+label)
    report=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    rows={int(r.split(',')[0]):[int(x.strip()) for x in r.split(',')[1:]] for r in report.splitlines()}
    if rows[a.gpu][0]>=500: raise RuntimeError(f'GPU {a.gpu} occupied: {rows[a.gpu]}; no launch')
    if a.kind=='preflight':
        call=[str(N/'code/preflight.py'),'--output',str(N/'checks/GPU_INTEGRATION.json')]
    else:
        freeze=json.loads((N/'PROTOCOL_FREEZE.json').read_text())
        import hashlib
        for path,digest in freeze['code_sha256'].items():
            assert hashlib.sha256((N/path).read_bytes()).hexdigest()==digest,path
        assert hashlib.sha256((N/'PROTOCOL.md').read_bytes()).hexdigest()==freeze['protocol_sha256']
        call=[str(N/('code/train_evidence.py' if a.kind=='train' else 'code/infer_evidence.py')),
              '--source',str(SOURCE),'--dataset-root',str(Q/'stage1_goal/data'),'--variant',a.condition]
        if a.kind=='train':
            call+=['--checkpoint',str(CHECKPOINT),'--selection',str(Q/'stage1_goal/DATA_SELECTION.json'),
                   '--plan',str(Q/'stage1_goal/pilot/TRAIN_PLAN.json'),'--output',str(N/'training'/a.condition/'run')]
        else:
            call+=['--reference-checkpoint',str(CHECKPOINT),'--checkpoint',str(N/'training'/a.condition/'run/final_model.pth'),
                   '--sequence',a.sequence,'--output',str(N/'predictions'/a.condition/a.sequence),'--verify-tiling']
    command=['docker','run','--rm','--name','lre_'+label.lower(),'--network','none','--gpus','all',
             '--user',f'{os.getuid()}:{os.getgid()}','--cpus','4','--memory','20g','--shm-size','2g',
             '-e',f'CUDA_VISIBLE_DEVICES={a.gpu}','-e','OMP_NUM_THREADS=2','-e','MKL_NUM_THREADS=2',
             '-e','OPENBLAS_NUM_THREADS=2','-e','PYTHONDONTWRITEBYTECODE=1','-e','PYTHONUNBUFFERED=1',
             '-e','CUBLAS_WORKSPACE_CONFIG=:4096:8','-e','MPLCONFIGDIR=/tmp/mpl',
             '-v',f'{R}:{R}:ro','-v',f'{N}:{N}:rw','-w',str(N),'--entrypoint','/opt/conda/bin/python',IMAGE,*call]
    record={'status':'RUNNING','gpu':a.gpu,'job':label,'image':IMAGE,'command':command,'gpu_before':rows,'start_unix':time.time()}
    receipt.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)
    with log.open('w') as stream:
        result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
    record.update(status='COMPLETE' if result.returncode==0 else 'FAILED',exit_code=result.returncode,end_unix=time.time())
    receipt.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)
    raise SystemExit(result.returncode)

if __name__=='__main__': main()
