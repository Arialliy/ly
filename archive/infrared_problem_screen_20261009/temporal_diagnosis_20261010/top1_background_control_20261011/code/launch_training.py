#!/usr/bin/env python3
"""Run exactly one authorized top-1% training invocation and record its lifecycle."""
import datetime, hashlib, json, math, subprocess, time
from pathlib import Path
Q=Path('/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/temporal_diagnosis_20261010')
D=Q/'top1_background_control_20261011'
OLD=Q/'low_response_information_20261011/lambda0_control'
T=D/'training'
NAME='deeppro_top1_background_train_20261011'
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
# Warm-reuse contract already verified; require immutable image match at launch too.
image=subprocess.check_output(['docker','image','inspect','deeppro-official-eval:20261009','--format','{{.Id}}'],text=True).strip()
assert image=='sha256:eb956873870d737f2b68dff0724b8f6f56f778829d119b0938d29b7abeaa2f92'
mem=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits','-i','0'],text=True).strip())
assert mem<500, f'GPU0 is occupied: {mem} MiB'
cmd=json.loads((OLD/'LAUNCH.json').read_text())['command']
cmd[cmd.index('--name')+1]=NAME
cmd[cmd.index(str(OLD/'code/train_pilot.py'))]=str(D/'code/train_pilot.py')
cmd[cmd.index('--output')+1]=str(T/'run')
# Extra read-only overlays protect reused evidence and original training data.
pos=cmd.index('--entrypoint')
for path in [OLD,Q/'stage1_goal/data',Q/'stage1_goal/pilot',Q/'stage1_goal/DATA_SELECTION.json']:
    cmd[pos:pos]=['-v',f'{path}:{path}:ro'];pos+=2
record={'utc':utc(),'command':cmd,'preflight_gpu0_memory_mib':mem,'image_id':image,'effective_extra_lambda':0,'top_fraction':0.01,'training_protocol_sha256':sha(T/'TRAINING_PROTOCOL.json')}
result=subprocess.run(cmd,text=True,capture_output=True)
record.update(stdout=result.stdout,stderr=result.stderr,launch_exit_code=result.returncode)
save(T/'LAUNCH.json',record)
print(json.dumps({'launched':NAME,**record}),flush=True)
result.check_returncode()
with (T/'GPU_TELEMETRY.jsonl').open('w',buffering=1) as log:
    while True:
        state=json.loads(subprocess.check_output(['docker','inspect',NAME,'--format','{{json .State}}'],text=True))
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu,power.draw','--format=csv,noheader,nounits','-i','0'],text=True).strip()
        log.write(json.dumps({'utc':utc(),'gpu_csv':gpu,'running':state['Running']})+'\n')
        if not state['Running']:break
        time.sleep(10)
save(T/'CONTAINER_STATE.json',state)
logs=subprocess.run(['docker','logs',NAME],text=True,capture_output=True)
(T/'stdout.log').write_text(logs.stdout);(T/'stderr.log').write_text(logs.stderr)
check={'utc':utc(),'exit_code':state['ExitCode'],'old_sources_unchanged':{},'new_code_sha256':{str(p):sha(p) for p in (D/'code').glob('*.py')}}
for p,h in json.loads((T/'SOURCE_BEFORE.json').read_text()).items():check['old_sources_unchanged'][p]=sha(Path(p))==h
assert all(check['old_sources_unchanged'].values())
if state['ExitCode']==0:
    run=json.loads((T/'run/RUN.json').read_text())
    rows=[json.loads(x) for x in (T/'run/TRAIN.jsonl').read_text().splitlines()]
    old_first=json.loads((OLD/'run/TRAIN.jsonl').read_text().splitlines()[0])
    checks={'exact_512_steps':len(rows)==512 and [r['step'] for r in rows]==list(range(1,513)),
      'bn_fixed':run['bn_initial_sha256']==run['bn_final_sha256'],
      'checkpoint_sha256':sha(T/'run/final_model.pth')==run['final_checkpoint_sha256'],
      'first_step_positive_equals_mean_control':rows[0]['positive']==old_first['positive'],
      'first_step_full_negative_equals_mean_control':rows[0]['background_full_mean']==old_first['negative'],
      'loss_equals_positive_plus_top_negative':all(abs(r['loss']-(r['positive']+r['negative']))<1e-5 for r in rows),
      'negative_equals_passive_top_diagnostic':all(r['negative']==r['background_top1pct_mean'] for r in rows),
      'per_window_k_rule':all(k==(max(1,math.ceil(.01*n)) if n else 0) for r in rows for n,k in zip(r['background_N_per_window'],r['background_k_per_window'])),
      'full_background_count_unchanged':all(r['background_N_per_window']==r['background_pixels_per_window'] for r in rows),
      'effective_lambda_zero':all(r['effective_extra_lambda']==0 for r in rows),
      'all_steps_finite':all(r['finite'] and all(math.isfinite(r[k]) for k in ['loss','positive','negative','background_full_mean','grad_norm']) for r in rows)}
    check.update(checks=checks,first_step=rows[0],last_step=rows[-1],run=run)
    save(T/'COMPLETION_CHECK.json',check)
    assert all(checks.values()),checks
    print(json.dumps({'complete':True,'checkpoint':str(T/'run/final_model.pth'),'checks':checks,'duration_seconds':run['duration_seconds']}),flush=True)
else:
    save(T/'COMPLETION_CHECK.json',check)
    raise SystemExit(state['ExitCode'])
