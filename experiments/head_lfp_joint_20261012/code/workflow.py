"""Run the registered two-arm experiment once; stop on any failed subprocess."""
import concurrent.futures
import json
from pathlib import Path
import subprocess
import sys
import time

E=Path(__file__).resolve().parents[1]
LAUNCH=E/'code/launch.py'

def main():
    output=E/'runtime/WORKFLOW.json'
    if output.exists():raise FileExistsError('Workflow already exists; do not repeat training')
    record={'status':'RUNNING','start_unix':time.time(),'jobs':[],'gpu':3,'main_step':2048}
    def save():output.write_text(json.dumps(record,indent=2)+'\n')
    def job(args):
        cmd=[sys.executable,str(LAUNCH),*args]
        result=subprocess.run(cmd)
        return {'command':cmd,'exit_code':result.returncode,'end_unix':time.time()}
    def run(args):
        record['current_job']=args;save();item=job(args);record['jobs'].append(item);save()
        if item['exit_code']:raise RuntimeError('Failed job '+str(args))
    save()
    try:
        run(['--kind','observe','--condition','release','--step','0','--gpu','3'])
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            scoring=[]
            for condition in ['H','HL']:
                run(['--kind','train','--condition',condition,'--gpu','3'])
                for step in [128,512,1024,2048]:
                    run(['--kind','observe','--condition',condition,'--step',str(step),'--gpu','3'])
                for step in [512,1024,2048]:
                    for seq in ['DJI_0026_2','DJI_0067_2']:
                        run(['--kind','infer','--condition',condition,'--step',str(step),'--sequence',seq,'--gpu','3'])
                        scoring.append(pool.submit(job,['--kind','score','--condition',condition,'--step',str(step),'--sequence',seq]))
            for future in scoring:
                item=future.result();record['jobs'].append(item);save()
                if item['exit_code']:raise RuntimeError('CPU scoring failed '+str(item['command']))
        for step in [512,1024,2048]:run(['--kind','summary','--step',str(step)])
        run(['--kind','summary'])
        record.update(status='COMPLETE',end_unix=time.time(),current_job=None);save()
    except BaseException as error:
        record.update(status='FAILED',error=repr(error),end_unix=time.time());save();raise

if __name__=='__main__':main()
