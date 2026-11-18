"""CPU-only scoring execution controller. Scientific/scorer sources remain frozen."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

BASE=Path(__file__).resolve().parents[1]
PROJECT=BASE.parents[2]
IMAGE='deeppro-official-eval:20261009'
JOBS=json.loads((BASE/'INFERENCE_QUEUE.json').read_text())['jobs']
LOGS=BASE/'evaluation_logs'
STATUS=BASE/'CPU_SCORING_QUEUE.json'
START=time.time()
processes={}
rows=[dict(condition=r['condition'],sequence=r['sequence'],status='pending') for r in JOBS]
extras={'main_summary':'pending','main_validation':'pending','path_summary':'pending','path_validation':'pending'}

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(status='RUNNING'):
    obj=dict(status=status,updated_utc=now(),elapsed_seconds=time.time()-START,jobs=rows,extra=extras,
             controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),parallel_cpu_jobs=2,
             image=IMAGE,scientific_sources_modified=False)
    temp=STATUS.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2)+'\n');temp.replace(STATUS)
def command(script,args):
    return ['docker','run','--rm','--user','1004:1004','--network','none','--entrypoint','python',
            '-e','OMP_NUM_THREADS=2','-e','OPENBLAS_NUM_THREADS=2',
            '-v',str(PROJECT)+':'+str(PROJECT)+':ro','-v',str(BASE)+':'+str(BASE)+':rw',IMAGE,
            str(BASE/'evaluation_code'/script)]+args

def start(key,script,args):
    logfile=LOGS/(key+'.log');cmd=command(script,args)
    (LOGS/(key+'.command.json')).write_text(json.dumps(dict(command=cmd,started_utc=now()),indent=2)+'\n')
    stream=logfile.open('w');process=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT)
    processes[key]=(process,stream)
    print(json.dumps(dict(started=key,pid=process.pid)),flush=True)

while True:
    failure=False
    for key,(process,stream) in list(processes.items()):
        rc=process.poll()
        if rc is None:continue
        stream.close();del processes[key]
        result=dict(status='COMPLETE' if rc==0 else 'FAILED',exit_code=rc,finished_utc=now())
        (LOGS/(key+'.result.json')).write_text(json.dumps(result,indent=2)+'\n')
        if key.startswith('score_'):
            row=next(r for r in rows if key=='score_'+r['condition']+'_'+r['sequence']);row.update(result)
        else:extras[key]=result['status']
        print(json.dumps(dict(finished=key,**result)),flush=True)
    failed=any(r['status']=='FAILED' for r in rows) or any(s=='FAILED' for s in extras.values())
    if failed:
        write('FAILED')
        if not processes:raise SystemExit('CPU scoring failed; inspect saved logs. No sources modified.')
        time.sleep(5);continue
    for row in rows:
        if len(processes)>=2:break
        if row['status']!='pending':continue
        complete=BASE/'predictions'/row['condition']/row['sequence']/'COMPLETE.json'
        if not complete.exists():continue
        receipt=json.loads(complete.read_text())
        if receipt.get('status')!='COMPLETE':continue
        prior=BASE/'evaluation'/row['condition']/row['sequence']/'COMPLETE.json'
        if prior.exists():
            existing=json.loads(prior.read_text())
            if existing.get('training_identity_checked'):
                row.update(status='COMPLETE',reused_completed_score=True);continue
            raise RuntimeError('Incomplete existing score receipt '+str(prior))
        row.update(status='RUNNING',started_utc=now(),inference_receipt_sha256=hashlib.sha256(complete.read_bytes()).hexdigest())
        start('score_'+row['condition']+'_'+row['sequence'],'evaluate_mutual.py',['--condition',row['condition'],'--sequence',row['sequence']])
    main_ready=all(r['status']=='COMPLETE' for r in rows if '_step' not in r['condition'])
    all_ready=all(r['status']=='COMPLETE' for r in rows)
    tasks=[('main_summary',main_ready,['--summary'],'evaluate_mutual.py'),
           ('main_validation',extras['main_summary']=='COMPLETE',['--directory',str(BASE/'evaluation')],'validate_tables.py'),
           ('path_summary',all_ready and extras['main_validation']=='COMPLETE',['--path-summary'],'evaluate_mutual.py'),
           ('path_validation',extras['path_summary']=='COMPLETE',['--directory',str(BASE/'evaluation_path')],'validate_tables.py')]
    for key,ready,args,script in tasks:
        if ready and extras[key]=='pending' and not processes:
            extras[key]='RUNNING';start(key,script,args)
    complete=all_ready and all(s=='COMPLETE' for s in extras.values())
    write('COMPLETE' if complete else 'RUNNING')
    if complete:break
    time.sleep(10)
