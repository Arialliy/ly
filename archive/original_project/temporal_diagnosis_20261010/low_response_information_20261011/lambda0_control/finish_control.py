import json,subprocess,time
from pathlib import Path
from datetime import datetime,timezone
L=Path(__file__).resolve().parent;Q=L.parent.parent;P=Q/'stage1_goal';R=Q.parent
def state(name):return json.loads(subprocess.check_output(['docker','inspect',name,'--format','{{json .State}}'],text=True))
def save(name,x):(L/name).write_text(json.dumps(x,indent=2)+'\n')
def logs(name,target):
    r=subprocess.run(['docker','logs',name],capture_output=True,text=True);(L/target).write_text(r.stdout+r.stderr)
def launch(name,gpu,script,args,cpuonly=False):
    cmd=['docker','run','-d','--name',name,'--user','1004:1004','--cpus','4','--memory','12g']
    if not cpuonly:cmd+=['--gpus','all']
    for e in [f'CUDA_VISIBLE_DEVICES={gpu}','OMP_NUM_THREADS=2','MKL_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','CUBLAS_WORKSPACE_CONFIG=:4096:8','PYTHONDONTWRITEBYTECODE=1','PYTHONUNBUFFERED=1','MPLCONFIGDIR=/tmp/deeppro-matplotlib']:cmd+=['-e',e]
    cmd+=['-v','/home/ly:/home/ly','-v',f'{R}/official_deeppro:{R}/official_deeppro:ro','--entrypoint','python','deeppro-official-eval:20261009',str(script)]+list(map(str,args))
    r=subprocess.run(cmd,capture_output=True,text=True);save(name+'_LAUNCH.json',dict(utc=datetime.now(timezone.utc).isoformat(),command=cmd,stdout=r.stdout,stderr=r.stderr,exit_code=r.returncode));assert r.returncode==0,r.stderr
    print(json.dumps({'launched':name,'gpu':gpu,'utc':datetime.now(timezone.utc).isoformat()}),flush=True)
def waitall(names,timeout=900):
    start=time.monotonic()
    while time.monotonic()-start<timeout:
        ss={n:state(n) for n in names}
        if all(not s['Running'] for s in ss.values()):
            for n,s in ss.items():logs(n,n+'.log');assert s['ExitCode']==0,(n,s)
            return ss
        time.sleep(5)
    raise TimeoutError(names)
train='deeppro_lowinfo_lambda0_train_20261011';ss=waitall([train]);save('TRAIN_CONTAINER_STATE.json',ss)
run=json.loads((L/'run/RUN.json').read_text());assert run['status']=='complete' and run['final_step']==512
print(json.dumps({'train_complete_seconds':run['duration_seconds']}),flush=True)
cp=R/'official_deeppro/log/sem_seg/RGB-T__2025-01-14_15-27__SoftLoUloss_DeepPro-Plus_DataL40/checkpoints/best_model.pth';names=[]
for gpu,seq in enumerate(['DJI_0026_2','DJI_0067_2']):
    mem=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip());assert mem<500,(gpu,mem)
    name='deeppro_lowinfo_lambda0_infer_'+seq.lower();names.append(name)
    launch(name,gpu,P/'pilot/infer_pilot.py',['--source',R/'official_deeppro','--reference-checkpoint',cp,'--checkpoint',L/'run/final_model.pth','--dataset-root',P/'data','--sequence',seq,'--output',L/'predictions'/seq])
save('INFERENCE_CONTAINER_STATES.json',waitall(names))
name='deeppro_lowinfo_lambda0_score_20261011';launch(name,'',L/'score_lambda0.py',[],True);save('SCORE_CONTAINER_STATE.json',waitall([name]))
print(json.dumps({'status':'COMPLETE','summary':str(L/'SUMMARY.json')}),flush=True)
