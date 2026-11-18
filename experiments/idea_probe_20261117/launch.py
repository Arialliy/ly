import hashlib,json,os,re,subprocess,time
from pathlib import Path
E=Path(__file__).resolve().parent;B=E.parents[1]
IMAGE=re.search(r'sha256:[0-9a-f]{64}',(B/'AGENTS.md').read_text()).group()
assert not (E/'LAUNCH.json').exists()
freeze={'frozen_unix':time.time(),'files':{n:hashlib.sha256((E/n).read_bytes()).hexdigest() for n in ['PROTOCOL.md','probe.py','launch.py']}}
(E/'FREEZE.json').write_text(json.dumps(freeze,indent=2)+'\n')
gpu_before=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
cmd=['docker','run','--rm','--name','idea_probe_20261117_cpu','--network','none','--user',f'{os.getuid()}:{os.getgid()}','--cpus','4','--memory','20g','--shm-size','1g']
env={'PYTHONPATH':str(B/'environment/site'),'PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'}
for k,v in env.items():cmd+=['-e',f'{k}={v}']
cmd+=['-v',f'{B}:{B}:ro','-v',f'{E}:{E}:rw','-w',str(E),'--entrypoint','/opt/conda/bin/python',IMAGE,str(E/'probe.py')]
record={'status':'RUNNING','start_unix':time.time(),'image':IMAGE,'device':'cpu','gpu_before':gpu_before,'command':cmd,'timeout_seconds':900}
def save():(E/'LAUNCH.json').write_text(json.dumps(record,indent=2)+'\n')
save()
try:
    with (E/'run.log').open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=900)
    record.update(status='COMPLETE' if r.returncode==0 else 'FAILED',exit_code=r.returncode)
except subprocess.TimeoutExpired:
    subprocess.run(['docker','stop','-t','3','idea_probe_20261117_cpu'],check=False)
    record.update(status='TIMEOUT',exit_code=124)
record['end_unix']=time.time();save();print(json.dumps(record,indent=2));raise SystemExit(record['exit_code'])
