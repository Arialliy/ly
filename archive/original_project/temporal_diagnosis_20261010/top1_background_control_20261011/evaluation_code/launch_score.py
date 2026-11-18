from pathlib import Path
import argparse,datetime,hashlib,json,subprocess
L=Path(__file__).resolve().parents[1];R=L.parent.parent
p=argparse.ArgumentParser();p.add_argument('--condition',required=True);p.add_argument('--sequence');p.add_argument('--summary',action='store_true');a=p.parse_args()
spec=R/'official_reproduction/env-spec.json';h=hashlib.sha256(json.dumps(json.loads(spec.read_text()),sort_keys=True,separators=(',',':')).encode()).hexdigest();assert h=='de8a7eb93aac154354646ae9b59c7ba4086c2ad1d8a704894c9c1de86a4d6d86'
image='deeppro-official-eval:20261009';iid=subprocess.check_output(['docker','image','inspect',image,'--format','{{.Id}}'],text=True).strip();assert iid=='sha256:eb956873870d737f2b68dff0724b8f6f56f778829d119b0938d29b7abeaa2f92'
name='deeppro_top1_score_'+a.condition+('_'+a.sequence.lower() if a.sequence else '')
cmd=['docker','run','-d','--name',name,'--user','1004:1004','--cpus','2','--memory','8g']
for env in ['OMP_NUM_THREADS=2','MKL_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','PYTHONDONTWRITEBYTECODE=1','PYTHONUNBUFFERED=1','MPLCONFIGDIR=/tmp/deeppro-matplotlib']:cmd+=['-e',env]
cmd+=['-v','/home/ly:/home/ly','--entrypoint','python',image,str(L/'evaluation_code/evaluate.py')]
if a.summary:cmd+=['--summary']
else:
    cmd+=['--condition',a.condition]
    if a.sequence:cmd+=['--sequence',a.sequence]
r=subprocess.run(cmd,text=True,capture_output=True)
(L/'evaluation'/(name+'_LAUNCH.json')).write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=cmd,stdout=r.stdout,stderr=r.stderr,exit_code=r.returncode,image_id=iid,spec_sha256=h,warm_reuse=True),indent=2)+'\n')
print(r.stdout,r.stderr);assert r.returncode==0
