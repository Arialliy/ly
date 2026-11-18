"""Durable paired short train -> own same-weight diagnostic -> CPU scoring."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import threading
import time

E=Path(__file__).resolve().parents[1]
RECEIPT=E/'runtime/WORKFLOW.json'
LOCK=threading.Lock()
STATE={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def update(**values):
    with LOCK:
        STATE.update(values)
        temp=RECEIPT.with_suffix('.tmp');temp.write_text(json.dumps(STATE,indent=2)+'\n');temp.replace(RECEIPT)

def job(tag,script,args,gpu=None):
    command=[sys.executable,str(E/'code/launch.py'),'--tag',tag]
    if gpu is not None:command+=['--gpu',str(gpu)]
    command+=[script,*args]
    update(**{tag:'RUNNING'})
    result=subprocess.run(command,cwd=E)
    receipt=read(E/'runtime'/(tag+'.json'))
    assert result.returncode==0 and receipt['status']=='COMPLETE' and receipt['exit_code']==0,(tag,result.returncode,receipt)
    update(**{tag:'COMPLETE'})

def arm(condition,gpu):
    tag=condition.lower()
    job('train_'+tag,'train.py',['--condition',condition],gpu)
    job('infer_'+tag,'infer.py',['--condition',condition],gpu)

def main():
    RECEIPT.parent.mkdir(exist_ok=True)
    if RECEIPT.exists():raise FileExistsError(RECEIPT)
    update(status='RUNNING',pid=os.getpid(),start_unix=time.time(),worker_sha256=sha(__file__),
           protocol_sha256=sha(E/'PROTOCOL_FREEZE.json'))
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(arm,'Continuation',1),pool.submit(arm,'Revision',2)]
            for future in futures:future.result()
        a,b=[read(E/'training'/name/'RUN.json') for name in ('Continuation','Revision')]
        shared=('initial_checkpoint_sha256','initial_model_state_sha256','original_state_sha256','source_sha256',
                'settings','cache_sha256','trainable_parameter_names','trainable_parameters','exposure_digest_sha256')
        assert all(a[k]==b[k] for k in shared)
        assert a['additional_steps']==b['additional_steps']==128 and a['final_step']==b['final_step']==2176
        pair={'status':'PASS','steps_each':128,'endpoint':2176,'matching_fields':list(shared),
              'runs_sha256':{n:sha(E/'training'/n/'RUN.json') for n in ('Continuation','Revision')},
              'exposure_digest_sha256':a['exposure_digest_sha256']}
        (E/'checks/PAIRED_TRAINING.json').write_text(json.dumps(pair,indent=2)+'\n')
        update(status='SCORING')
        for condition in ('ContinuationHistory','ContinuationCurrent','RevisionHistory','RevisionCurrent'):
            job('score_'+condition.lower(),'evaluate.py',['--condition',condition])
        job('score_summary','evaluate.py',['--summary'])
        update(status='EVALUATED',end_unix=time.time())
    except BaseException as exc:
        update(status='FAILED',error=repr(exc),end_unix=time.time());raise

if __name__=='__main__':main()
