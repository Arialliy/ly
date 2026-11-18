#!/usr/bin/env python3
"""Launch the one authorized original-architecture supervision control."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

N=Path(__file__).resolve().parents[1]
Q=N.parent
R=Q.parent
SOURCE=R/'official_deeppro'
REF=SOURCE/'log/sem_seg/RGB-T__2025-01-14_15-27__SoftLoUloss_DeepPro-Plus_DataL40/checkpoints/best_model.pth'
IMAGE='deeppro-official-eval:20261009'
IMAGE_ID='sha256:eb956873870d737f2b68dff0724b8f6f56f778829d119b0938d29b7abeaa2f92'
CONDITION='gaussian_softiou_approx'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('job',choices=['train','infer'])
    p.add_argument('--gpu',required=True,type=int)
    p.add_argument('--sequence',choices=['DJI_0026_2','DJI_0067_2'])
    a=p.parse_args()
    image=subprocess.check_output(['docker','image','inspect',IMAGE,'--format','{{.Id}}'],text=True).strip()
    assert image==IMAGE_ID
    spec=R/'official_reproduction/env-spec.json'
    spec_hash=hashlib.sha256(json.dumps(json.loads(spec.read_text()),sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert spec_hash=='de8a7eb93aac154354646ae9b59c7ba4086c2ad1d8a704894c9c1de86a4d6d86'
    frozen=json.loads((N/'PROTOCOL_FREEZE.json').read_text())
    assert frozen['protocol_sha256']==sha(N/'PROTOCOL.md')
    for path,fingerprint in frozen['code_sha256'].items():assert sha(N/path)==fingerprint,path
    for path,fingerprint in frozen['preflight_sha256'].items():
        assert sha(N/path)==fingerprint,path
        assert json.loads((N/path).read_text())['status']=='PASS',path
    used=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits','-i',str(a.gpu)],text=True).strip())
    assert used<500,f'GPU {a.gpu} occupied: {used} MiB'
    if a.job=='infer':assert a.sequence
    tail='' if a.job=='train' else '_'+a.sequence
    name='supervision_'+a.job+'_'+CONDITION+tail+'_20261012'
    logdir=N/('training' if a.job=='train' else 'inference_logs')/CONDITION
    if a.sequence:logdir=logdir/a.sequence
    logdir.mkdir(parents=True,exist_ok=True)
    assert not (logdir/'LAUNCH.json').exists(),'Refusing duplicate launch'
    cmd=['docker','run','-d','--name',name,'--gpus','all','--user','1004:1004','--cpus','4','--memory','16g','--shm-size','2g']
    for env in [f'CUDA_VISIBLE_DEVICES={a.gpu}','OMP_NUM_THREADS=2','MKL_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2',
                'CUBLAS_WORKSPACE_CONFIG=:4096:8','PYTHONDONTWRITEBYTECODE=1','PYTHONUNBUFFERED=1','MPLCONFIGDIR=/tmp/deeppro-matplotlib']:
        cmd+=['-e',env]
    cmd+=['-v','/home/ly:/home/ly']
    for path in [SOURCE,Q/'stage1_goal/data',Q/'stage1_goal/pilot',Q/'stage1_goal/DATA_SELECTION.json',Q/'top1_background_control_20261011',Q/'low_response_information_20261011',Q/'bc_tpro_pilot_20261011']:
        cmd+=['-v',f'{path}:{path}:ro']
    cmd+=['--entrypoint','python',IMAGE]
    if a.job=='train':
        cmd += [str(N/'code/train_supervision.py'),'--source',str(SOURCE),'--checkpoint',str(REF),
                '--dataset-root',str(Q/'stage1_goal/data'),'--selection',str(Q/'stage1_goal/DATA_SELECTION.json'),
                '--plan',str(Q/'stage1_goal/pilot/TRAIN_PLAN.json'),'--output',str(logdir/'run'),
                '--steps','512','--seed','46','--lr','0.0001','--device','cuda:0']
    else:
        cp=N/'training'/CONDITION/'run/final_model.pth'
        assert cp.is_file()
        cmd += [str(N/'code/infer_supervision.py'),'--source',str(SOURCE),'--reference-checkpoint',str(REF),
                '--checkpoint',str(cp),'--dataset-root',str(Q/'stage1_goal/data'),'--sequence',a.sequence,
                '--variant',CONDITION,'--output',str(N/'predictions'/CONDITION/a.sequence),'--verify-tiling']
    record={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':cmd,'container_name':name,
            'gpu':a.gpu,'gpu_preflight_memory_mib':used,'image_id':image,'environment_spec_sha256':spec_hash,
            'reuse':'unchanged immutable validated image; no installs','protocol_freeze_sha256':sha(N/'PROTOCOL_FREEZE.json')}
    result=subprocess.run(cmd,capture_output=True,text=True)
    record.update(launch_returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
    (logdir/'LAUNCH.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)
    result.check_returncode()

if __name__=='__main__':main()
