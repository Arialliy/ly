#!/usr/bin/env python3
"""Launch one specified, bounded BC-TPro job in the unchanged validated image."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

Q=Path(__file__).resolve().parents[2]
B=Q/'bc_tpro_pilot_20261011'
R=Q.parent
SOURCE=R/'official_deeppro'
REF=SOURCE/'log/sem_seg/RGB-T__2025-01-14_15-27__SoftLoUloss_DeepPro-Plus_DataL40/checkpoints/best_model.pth'
IMAGE='deeppro-official-eval:20261009'
IMAGE_ID='sha256:eb956873870d737f2b68dff0724b8f6f56f778829d119b0938d29b7abeaa2f92'

def main():
    p=argparse.ArgumentParser()
    p.add_argument('job',choices=['train','infer'])
    p.add_argument('--variant',required=True,choices=['single_dynamic','bc_tpro'])
    p.add_argument('--gpu',required=True,type=int)
    p.add_argument('--sequence',choices=['DJI_0026_2','DJI_0067_2'])
    p.add_argument('--benchmark-release',action='store_true')
    a=p.parse_args()
    image=subprocess.check_output(['docker','image','inspect',IMAGE,'--format','{{.Id}}'],text=True).strip()
    assert image==IMAGE_ID
    spec=R/'official_reproduction/env-spec.json'
    spec_hash=hashlib.sha256(json.dumps(json.loads(spec.read_text()),sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert spec_hash=='de8a7eb93aac154354646ae9b59c7ba4086c2ad1d8a704894c9c1de86a4d6d86'
    used=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits','-i',str(a.gpu)],text=True).strip())
    assert used<500, f'GPU {a.gpu} occupied: {used} MiB'
    tail='' if a.job=='train' else '_'+str(a.sequence)
    name='bc_tpro_'+a.job+'_'+a.variant+tail+'_20261011'
    logdir=B/('training' if a.job=='train' else 'inference_logs')/a.variant
    if a.sequence:logdir=logdir/a.sequence
    logdir.mkdir(parents=True,exist_ok=True)
    assert not (logdir/'LAUNCH.json').exists(), 'Refusing duplicate launch'
    cmd=['docker','run','-d','--name',name,'--gpus','all','--user','1004:1004',
         '--cpus','4','--memory','16g','--shm-size','2g']
    for env in [f'CUDA_VISIBLE_DEVICES={a.gpu}','OMP_NUM_THREADS=2','MKL_NUM_THREADS=2',
                'OPENBLAS_NUM_THREADS=2','CUBLAS_WORKSPACE_CONFIG=:4096:8',
                'PYTHONDONTWRITEBYTECODE=1','PYTHONUNBUFFERED=1','MPLCONFIGDIR=/tmp/deeppro-matplotlib']:
        cmd+=['-e',env]
    cmd+=['-v','/home/ly:/home/ly']
    for path in [SOURCE,Q/'stage1_goal/data',Q/'stage1_goal/pilot',Q/'stage1_goal/DATA_SELECTION.json',Q/'top1_background_control_20261011',Q/'low_response_information_20261011']:
        cmd+=['-v',f'{path}:{path}:ro']
    cmd+=['--entrypoint','python',IMAGE]
    if a.job=='train':
        assert (B/'checks/IMPLEMENTATION_CHECK.json').is_file(), 'Implementation gate absent'
        check=json.loads((B/'checks/IMPLEMENTATION_CHECK.json').read_text())
        assert check.get('status')=='PASS',check
        for filename, fingerprint in check['tested_source_sha256'].items():
            assert hashlib.sha256((B/'code'/filename).read_bytes()).hexdigest()==fingerprint, 'Tested source changed'
        frozen=json.loads((B/'PROTOCOL_FREEZE.json').read_text())
        assert frozen['model_config']==check['model_config']
        for filename, fingerprint in frozen['training_code_sha256'].items():
            assert hashlib.sha256((B/'code'/filename).read_bytes()).hexdigest()==fingerprint, 'Frozen training source changed'
        cmd += [str(B/'code/train_bc.py'),'--variant',a.variant,'--source',str(SOURCE),
                '--checkpoint',str(REF),'--dataset-root',str(Q/'stage1_goal/data'),
                '--selection',str(Q/'stage1_goal/DATA_SELECTION.json'),
                '--plan',str(Q/'stage1_goal/pilot/TRAIN_PLAN.json'),
                '--output',str(logdir/'run'),'--steps','512','--seed','46','--lr','0.0001','--device','cuda:0']
    else:
        assert a.sequence
        checkpoint=B/'training'/a.variant/'run/final_model.pth'
        assert checkpoint.is_file()
        cmd += [str(B/'code/infer_bc.py'),'--source',str(SOURCE),'--reference-checkpoint',str(REF),
                '--checkpoint',str(checkpoint),'--dataset-root',str(Q/'stage1_goal/data'),
                '--sequence',a.sequence,'--variant',a.variant,'--output',str(B/'predictions'/a.variant/a.sequence),'--verify-tiling']
        if a.benchmark_release:cmd+=['--benchmark-release']
    record={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':cmd,'container_name':name,
            'gpu':a.gpu,'gpu_preflight_memory_mib':used,'image_id':image,'environment_spec_sha256':spec_hash,
            'reuse':'unchanged immutable validated image; no installs','code_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((B/'code').glob('*.py'))},
            'protocol_sha256':hashlib.sha256((B/'PROTOCOL.md').read_bytes()).hexdigest()}
    result=subprocess.run(cmd,capture_output=True,text=True)
    record.update(launch_returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
    (logdir/'LAUNCH.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'container':name,'job':a.job,'variant':a.variant,'gpu':a.gpu,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr}),flush=True)
    result.check_returncode()

if __name__=='__main__':main()
