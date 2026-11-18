#!/usr/bin/env python3
"""Finite queue of preregistered inference jobs; never retries or chooses scores."""
import datetime
import json
from pathlib import Path
import subprocess
import time

N = Path(__file__).resolve().parents[1]
LAUNCH = N/'code/launch_jobs.py'
SEQUENCES = ['DJI_0026_2', 'DJI_0067_2']
TASKS = []
for variant, diagnostic, step in ([('E0','none',512), ('E1','none',512), ('E2','none',512),
                                  ('E2','mass',512), ('E2','current',512)] +
                                 [(v,'none',s) for s in (128,256,384) for v in ('E0','E1','E2')]):
    condition = variant if diagnostic == 'none' else 'E2_' + diagnostic
    if step != 512:
        condition += f'_step{step}'
    for sequence in SEQUENCES:
        TASKS.append(dict(variant=variant, diagnostic=diagnostic, step=step,
                          condition=condition, sequence=sequence, status='pending'))


def persist(status):
    (N/'INFERENCE_QUEUE.json').write_text(json.dumps(dict(status=status,
        updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), jobs=TASKS), indent=2)+'\n')


def capture(task):
    logdir = N/'inference_logs'/task['condition']/task['sequence']
    name = task['container']
    state = json.loads(subprocess.check_output(['docker','inspect',name], text=True))[0]['State']
    if state['Running']:
        return False
    logs = subprocess.run(['docker','logs',name], capture_output=True)
    (logdir/'stdout.log').write_bytes(logs.stdout)
    (logdir/'stderr.log').write_bytes(logs.stderr)
    (logdir/'CONTAINER_STATE.json').write_text(json.dumps(state, indent=2)+'\n')
    task['exit_code'] = state['ExitCode']
    complete = N/'predictions'/task['condition']/task['sequence']/'COMPLETE.json'
    if state['ExitCode'] != 0 or not complete.is_file():
        task['status'] = 'failed'
        persist('FAILED_NO_AUTOMATIC_RETRY')
        raise RuntimeError(f"Inference failed: {task}")
    task['status'] = 'complete'
    print(json.dumps({'complete':task['condition'], 'sequence':task['sequence']}), flush=True)
    return True


def main():
    if (N/'INFERENCE_QUEUE.json').exists():
        raise FileExistsError('Queue execution is single-use; inspect failures before any continuation')
    persist('RUNNING')
    while not all(t['status'] == 'complete' for t in TASKS):
        for task in TASKS:
            if task['status'] == 'running':
                capture(task)
        active = {t['gpu'] for t in TASKS if t['status'] == 'running'}
        for gpu in range(4):
            if gpu in active:
                continue
            used = int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used',
                '--format=csv,noheader,nounits','-i',str(gpu)], text=True).strip())
            if used >= 500:
                continue
            chosen = None
            for task in TASKS:
                if task['status'] != 'pending':
                    continue
                run = N/'training'/task['variant']/'run/RUN.json'
                if run.is_file() and json.loads(run.read_text())['status'] == 'complete':
                    chosen = task
                    break
            if chosen is None:
                continue
            cmd = ['python3', str(LAUNCH), 'infer', '--gpu', str(gpu), '--variant', chosen['variant'],
                   '--diagnostic', chosen['diagnostic'], '--step', str(chosen['step']), '--sequence', chosen['sequence']]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode:
                chosen.update(status='launch_failed', error=result.stderr)
                persist('FAILED_NO_AUTOMATIC_RETRY')
                raise RuntimeError(result.stderr)
            receipt = json.loads(result.stdout)
            chosen.update(status='running', gpu=gpu, container=receipt['container_name'])
            print(json.dumps({'started':chosen['condition'], 'sequence':chosen['sequence'], 'gpu':gpu}), flush=True)
        persist('RUNNING')
        if not all(t['status'] == 'complete' for t in TASKS):
            time.sleep(10)
    persist('COMPLETE')


if __name__ == '__main__':
    main()
