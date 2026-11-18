"""Schedule only the preregistered remaining jobs on idle user GPUs 2/3."""
from pathlib import Path
import json
import subprocess
import time

N=Path(__file__).resolve().parents[1]
jobs=[dict(kind='train',condition=c) for c in ['N0','N1']]
jobs += [dict(kind='infer',condition=c,sequence=s) for c in ['N2','N0','N1'] for s in ['DJI_0026_2','DJI_0067_2']]
running=[]
completed=[]
while jobs or running:
    for item in list(running):
        code=item['process'].poll()
        if code is None:continue
        item['stream'].close()
        if code: raise RuntimeError(f"Fixed job failed: {item['job']}, exit {code}; inspect receipt, no retry")
        print('FINISHED',item['job'],flush=True)
        completed.append(item['job']);running.remove(item)
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    occupied={int(row.split(',')[0]):int(row.split(',')[1]) for row in raw.splitlines()}
    reserved={item['gpu'] for item in running}
    # Parent-launched N2 training must finish before GPU3 can be reused.
    initial=json.loads((N/'runtime/train_N2.json').read_text())
    if initial['status']=='FAILED':raise RuntimeError('Initial N2 training failed; no retry')
    if initial['status']=='RUNNING':reserved.add(initial['gpu'])
    for gpu in [2,3]:
        if gpu in reserved or occupied[gpu]>=500:continue
        selected=None
        for job in jobs:
            if job['kind']=='train':selected=job;break
            path=N/'training'/job['condition']/'run/RUN.json'
            if path.exists() and json.loads(path.read_text())['status']=='complete':selected=job;break
        if selected is None:continue
        call=['python3',str(N/'code/launch_gpu.py'),'--gpu',str(gpu)]
        for key,value in selected.items():call += ['--'+key,value]
        label='_'.join(selected.values())
        stream=(N/'runtime'/('queue_'+label+'.log')).open('x')
        process=subprocess.Popen(call,stdout=stream,stderr=subprocess.STDOUT)
        running.append(dict(job=selected,gpu=gpu,process=process,stream=stream));jobs.remove(selected)
        print('STARTED',selected,'GPU',gpu,flush=True)
    (N/'runtime/QUEUE_STATUS.json').write_text(json.dumps({'pending':jobs,'running':[{'job':i['job'],'gpu':i['gpu'],'pid':i['process'].pid} for i in running],'completed':completed,'unix':time.time()},indent=2)+'\n')
    if jobs or running:time.sleep(5)
print('ALL_FIXED_JOBS_COMPLETE',flush=True)
