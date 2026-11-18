import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


here = Path(__file__).resolve().parent
root = here.parents[1]
image = 'sha256:eb956873870d737f2b68dff0724b8f6f56f778829d119b0938d29b7abeaa2f92'
cmd = ['docker','run','--rm','--network','none','--cpus','4','--memory','8g',
       '--user',f'{os.getuid()}:{os.getgid()}','-v',f'{root}:{root}:ro',
       '-v',f'{here}:{here}:rw','-v',f'{here}/inputs:{here}/inputs:ro','-w',str(here)]
for var in [f'PYTHONPATH={root}/environment/site','PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=1',
            'OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','PYTHONUNBUFFERED=1']:
    cmd+=['-e',var]
cmd+=['--entrypoint','/opt/conda/bin/python',image,'run.py']
files = ['PROTOCOL.md','solver.py','registration.py','run.py','launch.py','inputs/MANIFEST.json','inputs/download_reds.py']
attempt = len(list(here.glob('launch_*.json'))) + 1
if (here/'METRICS.json').exists():
    raise FileExistsError('The experiment already has final metrics.')
receipt = here/f'launch_{attempt}.json'
data = dict(command=cmd,start_unix=time.time(),
            sha256={f: hashlib.sha256((here/f).read_bytes()).hexdigest() for f in files})
receipt.write_text(json.dumps(data,indent=2)+'\n')
with (here/f'run_{attempt}.log').open('x') as log:
    code = subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT).returncode
data.update(exit_code=code,end_unix=time.time())
receipt.write_text(json.dumps(data,indent=2)+'\n')
raise SystemExit(code)
