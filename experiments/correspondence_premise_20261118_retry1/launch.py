from pathlib import Path
import datetime,hashlib,json,os,subprocess,time

E=Path(__file__).resolve().parent
P=E.parents[1]
def write(name,value):
    (E/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
assert not (E/'LAUNCH.json').exists(), 'New launch requires a new output directory.'
freeze={'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':{n:hashlib.sha256((E/n).read_bytes()).hexdigest() for n in ['PROTOCOL.md','audit.py']}}
write('FREEZE.json',freeze)
image='sha256:eb956873870d737f2b68dff0724b8f6f56f778829d119b0938d29b7abeaa2f92'
name='correspondence_premise_20261118_retry1_cpu'
cmd=['docker','run','--rm','--name',name,'--network','none','--user',f'{os.getuid()}:{os.getgid()}','--cpus','4','--memory','20g','--shm-size','1g']
for v in [f'PYTHONPATH={P}/environment/site','PYTHONDONTWRITEBYTECODE=1','PYTHONUNBUFFERED=1','OMP_NUM_THREADS=2','MKL_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2']:
    cmd+=['-e',v]
cmd+=['-v',f'{P}:{P}:ro','-v',f'{E}:{E}:rw','-w',str(E),'--entrypoint','/opt/conda/bin/python',image,str(E/'audit.py')]
receipt={'status':'RUNNING','start_unix':time.time(),'image':image,'device':'cpu','gpu_hours':0,'command':cmd,'timeout_seconds':1200}
write('LAUNCH.json',receipt);print(json.dumps(receipt),flush=True)
with (E/'run.log').open('w') as log:
    try:
        proc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
        receipt.update(status='COMPLETE' if proc.returncode==0 else 'FAILED',exit_code=proc.returncode)
    except subprocess.TimeoutExpired:
        subprocess.run(['docker','stop','--time','5',name],stdout=log,stderr=subprocess.STDOUT)
        receipt.update(status='TIMEOUT',exit_code=124)
receipt['end_unix']=time.time();write('LAUNCH.json',receipt)
print(json.dumps({'status':receipt['status'],'exit_code':receipt['exit_code'],'elapsed_seconds':receipt['end_unix']-receipt['start_unix']}),flush=True)
