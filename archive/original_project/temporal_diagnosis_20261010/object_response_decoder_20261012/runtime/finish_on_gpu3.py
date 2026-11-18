"""Resource-only fallback: serialize remaining fixed inferences on GPU3."""
import json,subprocess,time
from pathlib import Path
N=Path(__file__).resolve().parents[1]
receipt=N/'runtime/REMAINING_GPU3_PIPELINE.json'
state={'status':'WAITING_FOR_R1_INFERENCE','gpu':3,'jobs':[]}
def save():receipt.write_text(json.dumps(state,indent=2)+'\n')
save()
while json.loads((N/'runtime/pipeline_R1.json').read_text())['status']=='FINAL_INFERENCE':time.sleep(3)
assert json.loads((N/'runtime/pipeline_R1.json').read_text())['status']=='SCORING_LAUNCHED'
for condition in ['R0','R2']:
    while json.loads((N/'runtime'/f'train_{condition}_1.json').read_text())['status']=='RUNNING':time.sleep(3)
    assert json.loads((N/'training'/condition/'run/RUN.json').read_text())['status']=='complete'
    state['status']='INFER_'+condition;save()
    for sequence in ['DJI_0026_2','DJI_0067_2']:
        # The original-card attempt was blocked before Docker/model execution.
        command=['python3',str(N/'code/launch_gpu.py'),'--kind','infer','--gpu','3','--condition',condition,'--sequence',sequence,'--attempt','gpu3']
        with (N/'runtime'/f'fallback_{condition}_{sequence}.log').open('w') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        state['jobs'].append({'condition':condition,'sequence':sequence,'exit_code':result.returncode});save()
        if result.returncode:state['status']='FAILED';save();raise SystemExit(result.returncode)
    result=subprocess.run(['python3',str(N/'evaluation_code/launch_score.py'),'--condition',condition],capture_output=True,text=True)
    state['jobs'].append({'condition':condition,'kind':'score_launch','exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr});save()
    if result.returncode:state['status']='FAILED';save();raise SystemExit(result.returncode)
state['status']='ALL_FIXED_INFERENCES_COMPLETE_SCORES_LAUNCHED';save();print(json.dumps(state),flush=True)
