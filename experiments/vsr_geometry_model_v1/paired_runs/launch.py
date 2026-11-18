"""Launch an immutable paired trial; keep authentication, logs, and eval per run."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


here=Path(__file__).resolve().parent
exp=here.parent
root=exp.parents[1]
image='sha256:eb956873870d737f2b68dff0724b8f6f56f778829d119b0938d29b7abeaa2f92'
parser=argparse.ArgumentParser()
parser.add_argument('--gpu',type=int,choices=(1,2,3),default=1)
args=parser.parse_args()
occupancy=subprocess.check_output(['nvidia-smi',f'--id={args.gpu}','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
used,util=map(int,occupancy.split(','))
if used>=500 or util>=20:
    raise RuntimeError(f'GPU {args.gpu} currently in use, reported {occupancy}')
attempt=len(list(here.glob('receipt_*.json')))+1
label=f'{attempt:03}'
cmd=['docker','run','--rm','--network','none','--name',f'paired_vsr_{os.getuid()}_{label}',
     '--cpus','4','--memory','14g','--shm-size','1g','--gpus',f'device={args.gpu}',
     '--user',f'{os.getuid()}:{os.getgid()}','-v',f'{root}:{root}:ro','-v',f'{here}:{here}:rw','-w',str(exp)]
for x in [f'PYTHONPATH={root}/environment/site','PYTHONDONTWRITEBYTECODE=1','PYTHONUNBUFFERED=1',
          'CUBLAS_WORKSPACE_CONFIG=:4096:8','OMP_NUM_THREADS=2','MKL_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2']:
    cmd+=['-e',x]
cmd+=['--entrypoint','/opt/conda/bin/python',image,'paired.py','--out',f'paired_runs/run_{label}']
files=['model.py','geometry.py','transport.py','paired.py','PAIRED_PROTOCOL.md','cache/MANIFEST.json',
       'cache/RECIPE.json','data/MANIFEST.json','paired_runs/launch.py','third_party/DUF_SOURCE.json']
receipt={'cmd':cmd,'started':time.time(),'gpu':args.gpu,'occupancy_before_launch':occupancy,
         'sha256':{s:hashlib.sha256((exp/s).read_bytes()).hexdigest() for s in files}}
path=here/f'receipt_{label}.json'
path.write_text(json.dumps(receipt,indent=2)+'\n')
with (here/f'run_{label}.log').open('x') as log:
    code=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT).returncode
receipt.update(exit_code=code,ended=time.time())
path.write_text(json.dumps(receipt,indent=2)+'\n')
raise SystemExit(code)
