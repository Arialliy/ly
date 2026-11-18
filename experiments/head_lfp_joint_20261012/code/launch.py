"""Isolated package-backed jobs, explicit receipts, no occupied-GPU takeover."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

E=Path(__file__).resolve().parents[1];B=E.parents[1]
R=Path('/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009');Q=R/'temporal_diagnosis_20261010'
IMAGE='sha256:eb956873870d737f2b68dff0724b8f6f56f778829d119b0938d29b7abeaa2f92'
SOURCE=R/'official_deeppro';CP=SOURCE/'log/sem_seg/RGB-T__2025-01-14_15-27__SoftLoUloss_DeepPro-Plus_DataL40/checkpoints/best_model.pth'

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',required=True,choices=['preflight','train','infer','observe','score','summary','cpu'])
    p.add_argument('--condition',choices=['H','HL','release']);p.add_argument('--step',type=int)
    p.add_argument('--sequence',choices=['DJI_0026_2','DJI_0067_2']);p.add_argument('--gpu',type=int,choices=[1,2,3])
    p.add_argument('--tag',default='main');a=p.parse_args()
    gpu=a.kind in ['preflight','train','infer','observe']
    if gpu and a.gpu is None:p.error('GPU required')
    if a.kind in ['train','infer','observe','score'] and a.condition is None:p.error('condition required')
    if a.kind in ['infer','observe','score'] and a.step is None:p.error('step required')
    if a.kind in ['infer','score'] and a.sequence is None:p.error('sequence required')
    label='_'.join(str(x) for x in [a.kind,a.condition,a.step,a.sequence,a.tag] if x is not None)
    receipt=E/'runtime'/f'{label}.json';log=receipt.with_suffix('.log')
    if receipt.exists() or log.exists():raise FileExistsError(label)
    record={'status':'PREPARING','job':label,'gpu':a.gpu,'image':IMAGE,'start_unix':time.time(),'old_project_mounted':False}
    def save():receipt.write_text(json.dumps(record,indent=2)+'\n')
    if gpu:
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        rows={int(line.split(',')[0]):[int(x.strip()) for x in line.split(',')[1:]] for line in raw.splitlines()}
        record['gpu_before']=rows
        if rows[a.gpu][0]>=500:
            record.update(status='BLOCKED_OCCUPIED',exit_code=3);save();raise SystemExit(3)
    if a.kind not in ['preflight','cpu']:
        freeze=json.loads((E/'PROTOCOL_FREEZE.json').read_text())
        for name,h in freeze['files'].items():assert digest(E/name)==h,name
    if a.kind=='preflight':call=[str(E/'code/preflight.py'),'--output',str(E/'checks'/f'GPU_INTEGRATION_{a.tag}.json'),'--device','cuda:0']
    elif a.kind=='train':
        if a.condition not in ['H','HL']:p.error('H or HL train only')
        call=[str(E/'code/train_joint.py'),'--source',str(SOURCE),'--checkpoint',str(CP),'--dataset-root',str(Q/'stage1_goal/data'),'--selection',str(Q/'stage1_goal/DATA_SELECTION.json'),'--plan',str(Q/'stage1_goal/pilot/TRAIN_PLAN.json'),'--variant',a.condition,'--output',str(E/'training'/a.condition/'run')]
    elif a.kind in ['infer','score']:
        script='infer_joint.py' if a.kind=='infer' else 'evaluate_joint.py'
        call=[str(E/'code'/script),'--condition',a.condition,'--step',str(a.step),'--sequence',a.sequence]
    elif a.kind=='observe':call=[str(E/'code/observe_training.py'),'--condition',a.condition,'--step',str(a.step)]
    elif a.kind=='summary':call=[str(E/'code/summarize_joint.py'),'--step',str(a.step)] if a.step else [str(E/'code/summarize_joint.py'),'--all']
    else:call=['-m','pytest','-q','-p','no:cacheprovider',str(E/'code/test_joint.py'),'--junitxml',str(E/'checks/CPU_ACTUAL.xml')]
    cmd=['docker','run','--rm','--name','head_joint_'+label.lower(),'--network','none','--user',f'{os.getuid()}:{os.getgid()}','--cpus','4','--memory','20g','--shm-size','2g']
    if gpu:cmd+=['--gpus','all','-e',f'CUDA_VISIBLE_DEVICES={a.gpu}']
    env={'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1',
         'PYTHONUNBUFFERED':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8','MPLCONFIGDIR':'/tmp/head-joint-mpl',
         'PYTHONPATH':f'{B}/environment/site:{E}/code'}
    for k,v in env.items():cmd+=['-e',f'{k}={v}']
    cmd+=['-v',f'{B}:{B}:ro','-v',f'{B}/archive/infrared_problem_screen_20261009:{R}:ro',
          '-v',f'{E}:{E}:rw','-w',str(E),'--entrypoint','/opt/conda/bin/python',IMAGE,*call]
    record.update(status='RUNNING',command=cmd);save();print(json.dumps({'job':label,'status':'RUNNING'}),flush=True)
    with log.open('w') as f:result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
    record.update(status='COMPLETE' if result.returncode==0 else 'FAILED',exit_code=result.returncode,end_unix=time.time());save()
    print(json.dumps({'job':label,'status':record['status'],'exit_code':result.returncode,'log':str(log)}),flush=True)
    raise SystemExit(result.returncode)

if __name__=='__main__':main()
