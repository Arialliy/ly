"""Durably finish immutable endpoint evidence after the registered pair worker."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

E=Path(__file__).resolve().parents[1]
ROOT=E.parents[1]
RECEIPT=E/'runtime/FINALIZER.json'
IMAGE='sha256:8470a68886ff2f480d76858d930fb72d0e680e69276fc1d068fe8b6625b4cb9f'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,d):
    q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(d,indent=2)+'\n');q.replace(p)

def task(name,cmd):
    path=E/'runtime'/(name+'.log')
    with path.open('x') as f:
        p=subprocess.run(cmd,cwd=E,stdout=f,stderr=subprocess.STDOUT)
    assert p.returncode==0,(name,p.returncode,str(path))

def main():
    if RECEIPT.exists():raise FileExistsError(RECEIPT)
    state={'status':'WAITING_FOR_EVALUATED','pid':os.getpid(),'start_unix':time.time(),
           'script_sha256':sha(__file__),'protocol_sha256':sha(E/'PROTOCOL_FREEZE.json'),
           'analysis_source_sha256':{str(p):sha(p) for p in sorted((E/'analysis').glob('*.py'))}}
    save(RECEIPT,state)
    try:
        while True:
            workflow=read(E/'runtime/WORKFLOW.json')
            if workflow['status']=='EVALUATED':break
            if workflow['status'] in ('FAILED','COMPLETE'):raise RuntimeError(workflow['status'])
            time.sleep(30)
        assert all(sha(p)==h for p,h in state['analysis_source_sha256'].items())
        state['status']='AGGREGATING';save(RECEIPT,state)
        task('aggregate_trace',[sys.executable,str(E/'analysis/aggregate_trace.py')])
        state['status']='VERIFYING_CHECKPOINTS';save(RECEIPT,state)
        task('verify_training_artifacts',['docker','run','--rm','--network','none',
             '--user',f'{os.getuid()}:{os.getgid()}','--cpus','4','--memory','8g',
             '-e','OMP_NUM_THREADS=2','-e','MKL_NUM_THREADS=2','-e','OPENBLAS_NUM_THREADS=2',
             '-e','PYTHONDONTWRITEBYTECODE=1','-v',f'{ROOT}:{ROOT}:ro','-v',f'{E}:{E}:rw',
             '-w',str(E),'--entrypoint','python',IMAGE,str(E/'analysis/verify_training_artifacts.py')])
        state['status']='RENDERING';save(RECEIPT,state)
        task('render_results',[sys.executable,str(E/'analysis/render_results.py')])
        result=read(E/'RESULTS.json');assert result['status']=='COMPLETE'
        workflow=read(E/'runtime/WORKFLOW.json');assert workflow['status']=='EVALUATED'
        workflow.update(status='COMPLETE',decision=result['decision']['code'],completed_unix=time.time(),
                        additional_steps_each=512,M2_added=False,no_further_training_launched=True,
                        report_sha256={str(E/n):sha(E/n) for n in ('RESULTS.json','RESULTS.md')},
                        training_artifact_check_sha256=sha(E/'analysis/TRAINING_ARTIFACT_CHECK.json'),
                        trace_summary_sha256=sha(E/'analysis/trace_summary/SUMMARY.json'),
                        evaluation_summary_sha256=sha(E/'evaluation/step_2560/SUMMARY_COMPLETE.json'))
        save(E/'runtime/WORKFLOW.json',workflow)
        state.update(status='COMPLETE',decision=result['decision']['code'],end_unix=time.time());save(RECEIPT,state)
    except BaseException as exc:
        state.update(status='FAILED',error=repr(exc),end_unix=time.time());save(RECEIPT,state);raise

if __name__=='__main__':main()
