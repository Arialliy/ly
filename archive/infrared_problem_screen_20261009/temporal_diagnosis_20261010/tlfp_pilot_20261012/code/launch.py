"""Fixed isolated TLFP jobs; immutable receipts and occupied-GPU refusal."""
import argparse, hashlib, json, os, subprocess, time
from pathlib import Path
N=Path(__file__).resolve().parents[1];Q=N.parent;R=Q.parent
IMAGE='sha256:eb956873870d737f2b68dff0724b8f6f56f778829d119b0938d29b7abeaa2f92'
SOURCE=R/'official_deeppro'
CHECKPOINT=SOURCE/'log/sem_seg/RGB-T__2025-01-14_15-27__SoftLoUloss_DeepPro-Plus_DataL40/checkpoints/best_model.pth'
def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['init','preflight','train','infer'],required=True)
    p.add_argument('--gpu',type=int,choices=[1,2,3]);p.add_argument('--condition',choices=['S','T'])
    p.add_argument('--sequence',choices=['DJI_0026_2','DJI_0067_2']);p.add_argument('--tag',default='main')
    a=p.parse_args()
    if a.kind!='init' and a.gpu is None:p.error('GPU required')
    if a.kind in ['train','infer'] and a.condition is None:p.error('condition required')
    if a.kind=='infer' and a.sequence is None:p.error('sequence required')
    label='_'.join(str(v) for v in [a.kind,a.condition,a.sequence,a.tag] if v)
    receipt=N/'runtime'/f'{label}.json';log=receipt.with_suffix('.log')
    if receipt.exists() or log.exists():raise FileExistsError(label)
    record={'status':'PREPARING','gpu':a.gpu,'job':label,'image':IMAGE,'start_unix':time.time()}
    def save():receipt.write_text(json.dumps(record,indent=2)+'\n')
    if a.gpu is not None:
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        rows={int(line.split(',')[0]):[int(x.strip()) for x in line.split(',')[1:]] for line in raw.splitlines()};record['gpu_before']=rows
        if rows[a.gpu][0]>=500:
            record.update(status='BLOCKED_OCCUPIED_BEFORE_DOCKER',exit_code=3);save();raise SystemExit(3)
    if a.kind in ['train','infer']:
        freeze=json.loads((N/'PROTOCOL_FREEZE.json').read_text())
        for path,digest in freeze['code_sha256'].items():
            file=Path(path) if Path(path).is_absolute() else N/path
            assert hashlib.sha256(file.read_bytes()).hexdigest()==digest,path
        assert hashlib.sha256((N/'PROTOCOL.md').read_bytes()).hexdigest()==freeze['protocol_sha256']
    if a.kind=='init':call=[str(N/'code/prepare_init.py')]
    elif a.kind=='preflight':call=[str(N/'code/preflight.py'),'--output',str(N/'checks'/f'GPU_INTEGRATION_{a.tag}.json')]
    elif a.kind=='train':
        call=[str(N/'code/train_tlfp.py'),'--source',str(SOURCE),'--checkpoint',str(CHECKPOINT),'--dataset-root',str(Q/'stage1_goal/data'),'--selection',str(Q/'stage1_goal/DATA_SELECTION.json'),'--plan',str(Q/'stage1_goal/pilot/TRAIN_PLAN.json'),'--variant',a.condition,'--output',str(N/'training'/a.condition/'run')]
    else:call=[str(N/'code/infer_tlfp.py'),'--condition',a.condition,'--sequence',a.sequence]
    command=['docker','run','--rm','--name','tlfp_'+label.lower(),'--network','none','--user',f'{os.getuid()}:{os.getgid()}','--cpus','4','--memory','20g','--shm-size','2g']
    if a.gpu is not None:command+=['--gpus','all','-e',f'CUDA_VISIBLE_DEVICES={a.gpu}']
    env={'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8','MPLCONFIGDIR':'/tmp/mpl','PYTHONPATH':str(Q/'route_closeout_20261012/dependencies/site')+':'+str(N/'code')}
    for k,v in env.items():command+=['-e',f'{k}={v}']
    command+=['-v',f'{R}:{R}:ro','-v',f'{N}:{N}:rw','-w',str(N),'--entrypoint','/opt/conda/bin/python',IMAGE,*call]
    record.update(status='RUNNING',command=command,code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((N/'code').glob('*.py'))});save();print(json.dumps({'job':label,'status':'RUNNING'}),flush=True)
    with log.open('w') as stream:result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
    record.update(status='COMPLETE' if result.returncode==0 else 'FAILED',exit_code=result.returncode,end_unix=time.time());save()
    print(json.dumps({'job':label,'status':record['status'],'exit_code':result.returncode,'log':str(log)}),flush=True);raise SystemExit(result.returncode)
if __name__=='__main__':main()
