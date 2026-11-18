"""Wait for already authorized fixed training; run two fixed final inferences."""
import argparse,json,subprocess,time
from pathlib import Path
N=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--condition',required=True);p.add_argument('--gpu',type=int,required=True)
a=p.parse_args();record=N/'runtime'/f'pipeline_{a.condition}.json'
state={'condition':a.condition,'gpu':a.gpu,'status':'WAITING_FOR_FIXED_TRAINING','inferences':[]}
def save():record.write_text(json.dumps(state,indent=2)+'\n')
save()
while True:
    path=N/'runtime'/f'train_{a.condition}_1.json'
    r=json.loads(path.read_text())
    if r['status']=='FAILED':raise RuntimeError('Training failed; no dependent inference')
    if r['status']=='COMPLETE':break
    time.sleep(5)
state['status']='FINAL_INFERENCE';save()
for sequence in ['DJI_0026_2','DJI_0067_2']:
    command=['python3',str(N/'code/launch_gpu.py'),'--kind','infer','--gpu',str(a.gpu),'--condition',a.condition,'--sequence',sequence]
    result=subprocess.run(command,stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
    state['inferences'].append({'sequence':sequence,'exit_code':result.returncode});save()
    if result.returncode!=0:
        state['status']='FAILED';save();raise SystemExit(result.returncode)
state['status']='INFERENCES_COMPLETE';save()
result=subprocess.run(['python3',str(N/'evaluation_code/launch_score.py'),'--condition',a.condition],capture_output=True,text=True)
state.update(status='SCORING_LAUNCHED' if result.returncode==0 else 'FAILED',score_launcher_exit_code=result.returncode,
    score_stdout=result.stdout,score_stderr=result.stderr);save()
print(json.dumps(state),flush=True)
raise SystemExit(result.returncode)
