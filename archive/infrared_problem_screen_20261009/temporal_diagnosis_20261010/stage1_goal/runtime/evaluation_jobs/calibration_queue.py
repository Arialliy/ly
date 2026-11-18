"""Handle-tracked calibration-only queue; algorithm scripts are never modified."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE=Path('/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009')
STAGE=BASE/'temporal_diagnosis_20261010/stage1_goal'
HERE=Path(__file__).resolve().parent
SOURCE=BASE/'official_deeppro'
REFERENCE=SOURCE/'log/sem_seg/RGB-T__2025-01-14_15-27__SoftLoUloss_DeepPro-Plus_DataL40/checkpoints/best_model.pth'
IMAGE='deeppro-official-eval:20261009'
STRATEGIES=['paired_mean','paired_max','ohem','random_pair_max']
VARIANTS=['official']+STRATEGIES
EXPECTED_BN='51832be8d623f18e372232ddefdf4214ad67b3e45f795fbaa2ea9daa7fd0ecab'

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
    return h.hexdigest()

def write(path,value):
    temp=Path(str(path)+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)

def call(args,check=True):
    r=subprocess.run(args,text=True,capture_output=True,timeout=30)
    if check and r.returncode:raise RuntimeError({'command':args,'returncode':r.returncode,'stderr':r.stderr})
    return r

def inspect(handle):
    r=call(['docker','inspect',handle],check=False)
    if r.returncode:
        # A transient inspect failure never authorizes restart or GPU reuse.
        return {'inspection_error':r.stderr,'observed_utc':now()}
    obj=json.loads(r.stdout)[0]
    return {'id':obj['Id'],'name':obj['Name'].lstrip('/'),'state':obj['State'],'observed_utc':now()}

def log_event(kind,**kwargs):
    event={'utc':now(),'kind':kind,**kwargs}
    with (HERE/'EVENTS.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
    print(json.dumps(event),flush=True)

def docker_base(name,gpu=None):
    cmd=['docker','run','-d','--name',name,'--user','1004:1004','--cpus','4' if gpu is not None else '2','--memory','12g' if gpu is not None else '8g']
    if gpu is not None:cmd+=['--gpus','all']
    for entry in [f'CUDA_VISIBLE_DEVICES={gpu if gpu is not None else ""}','OMP_NUM_THREADS=2','MKL_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','CUBLAS_WORKSPACE_CONFIG=:4096:8','PYTHONDONTWRITEBYTECODE=1','PYTHONUNBUFFERED=1','MPLCONFIGDIR=/tmp/deeppro-matplotlib']:
        cmd+=['-e',entry]
    cmd+=['-v','/home/ly:/home/ly','-v',f'{SOURCE}:{SOURCE}:ro','--entrypoint','python',IMAGE]
    return cmd

def launch_job(kind,variant,seq,gpu,frozen_hashes,first_seq):
    for path,value in frozen_hashes.items():
        if sha(path)!=value:raise RuntimeError('Frozen algorithm or selection changed: '+path)
    folder=HERE/variant/seq/kind;folder.mkdir(parents=True,exist_ok=True)
    name=f'deeppro_stage1_cal_{kind}_{variant}_{seq.lower()}'
    if (folder/'LAUNCH.json').exists():raise RuntimeError('Existing launch needs handle recovery, never restart: '+str(folder))
    pred=STAGE/'predictions'/variant/seq
    evaluation=STAGE/'evaluation'/variant/seq
    if kind=='infer':
        completion=pred/'COMPLETE.json'
        if pred.exists() and any(pred.iterdir()):raise RuntimeError('Prediction output already exists; inspect before any restart: '+str(pred))
        checkpoint=REFERENCE if variant=='official' else STAGE/'runs'/variant/'final_model.pth'
        cmd=docker_base(name,gpu)+[str(STAGE/'pilot/infer_pilot.py'),'--source',str(SOURCE),'--reference-checkpoint',str(REFERENCE),
             '--checkpoint',str(checkpoint),'--dataset-root',str(STAGE/'data'),'--sequence',seq,'--output',str(pred)]
        if seq==first_seq:cmd+=['--verify-tiling']
    else:
        completion=evaluation/'official_fused_COMPLETE.json'
        if evaluation.exists() and any(evaluation.iterdir()):raise RuntimeError('Score output already exists; inspect before any restart: '+str(evaluation))
        cmd=docker_base(name)+[str(STAGE/'pilot/score_pilot.py'),'--data',str(STAGE/'data'/seq),'--predictions',str(pred),'--output',str(evaluation)]
    record={'created_utc':now(),'kind':kind,'variant':variant,'sequence':seq,'gpu':gpu,'container_name':name,'command':cmd,
            'completion_path':str(completion),'frozen_sources':frozen_hashes,'status':'launching'}
    write(folder/'LAUNCH.json',record)
    r=call(cmd,check=False)
    record.update(launch_exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr)
    if r.returncode:
        record['status']='launch_failed';write(folder/'LAUNCH.json',record)
        raise RuntimeError(record)
    record.update(container_id=r.stdout.strip(),status='running',folder=str(folder))
    write(folder/'LAUNCH.json',record)
    log_event('launch',kind_job=kind,variant=variant,sequence=seq,gpu=gpu,container_id=record['container_id'])
    return record

def collect_job(job):
    observed=inspect(job['container_id'])
    folder=Path(job['folder'])
    write(folder/'LAST_INSPECT.json',observed)
    if 'inspection_error' in observed:return False
    if observed['state']['Running']:return False
    if observed['state']['Status']!='exited':raise RuntimeError('Non-exited terminal job state: '+str(observed))
    logs=call(['docker','logs',job['container_id']],check=False)
    (folder/'stdout.log').write_text(logs.stdout)
    (folder/'stderr.log').write_text(logs.stderr)
    job.update(status='exited',exit_code=observed['state']['ExitCode'],finished_utc=observed['state']['FinishedAt'])
    write(folder/'EXIT.json',{'job':job,'inspect':observed})
    if job['exit_code']!=0:raise RuntimeError('Job failed: '+str(folder))
    cp=Path(job['completion_path'])
    if not cp.exists():raise RuntimeError('Exited0 without completion: '+str(cp))
    completion=json.loads(cp.read_text())
    if completion['status']!='COMPLETE':raise RuntimeError('Wrong completion status: '+str(cp))
    if job['kind']=='infer':
        if completion['sequence']!=job['sequence'] or completion['official_starts']!=[0,36,72,108,120]:raise RuntimeError('Inference protocol mismatch')
        if len(completion['window_sha256'])!=11:raise RuntimeError('Missing diagnostic windows')
        if completion['raw_sha256']!=sha(STAGE/'data'/job['sequence']/'raw_frames.npy'):raise RuntimeError('Raw input hash changed')
    else:
        if len(completion['thresholds'])!=23 or not completion['official_fusion_exactly_reconstructed']:raise RuntimeError('Scoring protocol mismatch')
    job.update(status='complete',completion_sha256=sha(cp))
    write(folder/'VERIFIED.json',job)
    log_event('complete',kind_job=job['kind'],variant=job['variant'],sequence=job['sequence'],gpu=job['gpu'],container_id=job['container_id'])
    return True

def main():
    if (HERE/'QUEUE.json').exists():raise RuntimeError('Existing queue state must be recovered, not overwritten')
    selection=json.loads((STAGE/'DATA_SELECTION.json').read_text())
    sequences=selection['splits']['calibration']
    assert sequences==['DJI_0026_2','DJI_0067_2']
    frozen_hashes={str(p):sha(p) for p in [STAGE/'DATA_SELECTION.json',STAGE/'pilot/infer_pilot.py',STAGE/'pilot/score_pilot.py']}
    initial_training={s:inspect('deeppro_stage1_train_'+s) for s in STRATEGIES}
    if any('inspection_error' in v for v in initial_training.values()):raise RuntimeError(initial_training)
    write(HERE/'INITIAL_TRAINING_HANDLES.json',initial_training)
    queue={'status':'running','pid':os.getpid(),'created_utc':now(),'split':'calibration_only','sequences':sequences,
           'pending_variants':VARIANTS.copy(),'gpu_assignments':{},'training_verified':{},'active_infer':{},'active_score':[],
           'pending_score':[],'inference_completed':[],'scoring_completed':[],'frozen_sources':frozen_hashes,
           'image_id':call(['docker','image','inspect',IMAGE,'--format','{{.Id}}']).stdout.strip()}
    write(HERE/'QUEUE.json',queue)
    while True:
        for gpu,s in enumerate(STRATEGIES):
            if s in queue['training_verified']:continue
            observation=inspect(initial_training[s]['id'])
            write(HERE/('TRAIN_LAST_'+s+'.json'),observation)
            if 'inspection_error' in observation or observation['state']['Running']:continue
            if observation['state']['Status']!='exited' or observation['state']['ExitCode']!=0:
                raise RuntimeError('Training failed/not clean exit: '+s)
            run_path=STAGE/'runs'/s/'RUN.json'
            run=json.loads(run_path.read_text())
            if run['status']!='complete' or run['final_step']!=512 or run['bn_initial_sha256']!=EXPECTED_BN or run['bn_final_sha256']!=EXPECTED_BN:
                raise RuntimeError('Training completion/BN mismatch: '+s)
            ckpt=STAGE/'runs'/s/'final_model.pth'
            if not ckpt.is_file():raise RuntimeError('Final checkpoint missing: '+s)
            queue['training_verified'][s]={'gpu':gpu,'container':observation,'run_sha256':sha(run_path),'checkpoint_sha256':sha(ckpt),
                                            'final_step':512,'bn_initial_sha256':run['bn_initial_sha256'],'bn_final_sha256':run['bn_final_sha256']}
            log_event('training_verified',variant=s,gpu=gpu)
        for gpu,job in list(queue['active_infer'].items()):
            if collect_job(job):
                variant,seq=job['variant'],job['sequence']
                queue['inference_completed'].append([variant,seq])
                queue['pending_score'].append([variant,seq])
                del queue['active_infer'][gpu]
                if seq==sequences[-1]:
                    del queue['gpu_assignments'][gpu]
                else:
                    next_seq=sequences[sequences.index(seq)+1]
                    queue['active_infer'][gpu]=launch_job('infer',variant,next_seq,int(gpu),frozen_hashes,sequences[0])
        for job in queue['active_score'][:]:
            if collect_job(job):
                queue['scoring_completed'].append([job['variant'],job['sequence']]);queue['active_score'].remove(job)
        for gpu,s in enumerate(STRATEGIES):
            key=str(gpu)
            if s not in queue['training_verified'] or key in queue['gpu_assignments']:continue
            eligible=[v for v in queue['pending_variants'] if v=='official' or v in queue['training_verified']]
            if not eligible:continue
            # Physical GPU must now be clear, in addition to clean Docker exit.
            memory=call(['nvidia-smi','--id='+key,'--query-gpu=memory.used','--format=csv,noheader,nounits']).stdout.strip()
            if int(memory)>64:continue
            variant=eligible[0]
            queue['gpu_assignments'][key]=variant
            queue['pending_variants'].remove(variant)
            queue['active_infer'][key]=launch_job('infer',variant,sequences[0],gpu,frozen_hashes,sequences[0])
        while queue['pending_score'] and len(queue['active_score'])<4:
            variant,seq=queue['pending_score'].pop(0)
            queue['active_score'].append(launch_job('score',variant,seq,None,frozen_hashes,sequences[0]))
        queue['last_poll_utc']=now()
        if len(queue['inference_completed'])==10 and len(queue['scoring_completed'])==10:
            queue.update(status='complete',completed_utc=now());write(HERE/'QUEUE.json',queue)
            log_event('queue_complete',inferences=10,scores=10)
            return
        write(HERE/'QUEUE.json',queue)
        time.sleep(5)

if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(HERE/'QUEUE_ERROR.json',{'utc':now(),'error':repr(error),'note':'No active job was terminated or restarted. Inspect saved handles before recovery.'})
        raise
