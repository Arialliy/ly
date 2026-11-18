"""Launch task code with read-only source/data mounts and bounded resources."""
import argparse
import os
from pathlib import Path
import subprocess
import time
from runtime import B, E, save

IMAGE='nvcr.io/nvidia/pytorch:24.01-py3'
OLD='/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009'

def main():
    p=argparse.ArgumentParser(); p.add_argument('--gpu',type=int,choices=[0,1,2]);p.add_argument('--tag',required=True)
    p.add_argument('script');p.add_argument('args',nargs=argparse.REMAINDER);a=p.parse_args()
    receipt=E/'runtime'/f'{a.tag}.json';log=receipt.with_suffix('.log')
    if receipt.exists(): raise FileExistsError(receipt)
    image_id=subprocess.check_output(['docker','image','inspect',IMAGE,'--format','{{.Id}}'],text=True).strip()
    record={'status':'PREPARING','image':image_id,'gpu':a.gpu,'start_unix':time.time()}
    if a.gpu is not None:
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        rows={int(l.split(',')[0]):[int(x.strip()) for x in l.split(',')[1:]] for l in raw.splitlines()}
        record['gpu_before']=rows
        if rows[a.gpu][0]>=500: raise RuntimeError('GPU occupied; will not launch')
    cmd=['docker','run','--rm','--name','camera_motion_'+a.tag,'--network','none','--user',f'{os.getuid()}:{os.getgid()}',
         '--cpus','4','--memory','20g','--shm-size','2g']
    if a.gpu is not None:cmd+=['--gpus',f'device={a.gpu}','-e','CUDA_VISIBLE_DEVICES=0']
    env={'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1',
         'PYTHONUNBUFFERED':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8','MPLCONFIGDIR':'/tmp/camera-motion-mpl',
         'PYTHONPATH':str(E/'code')}
    for k,v in env.items():cmd+=['-e',f'{k}={v}']
    cmd+=['-v',f'{B}:{B}:ro','-v',f'{B}/archive/infrared_problem_screen_20261009:{OLD}:ro',
          '-v',f'{E}:{E}:rw','-w',str(E),'--entrypoint','python',image_id,str(E/'code'/a.script),*a.args]
    record.update(status='RUNNING',command=cmd);save(receipt,record)
    with log.open('w',encoding='utf-8') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
    record.update(status='COMPLETE' if r.returncode==0 else 'FAILED',exit_code=r.returncode,end_unix=time.time());save(receipt,record)
    print(record['status'],str(log),flush=True)
    raise SystemExit(r.returncode)
if __name__=='__main__':main()
