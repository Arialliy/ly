"""Durable Current-lane postprocessing of the already launched formal run.

No training or resume command is issued here. GPU ownership is checked afresh
by the frozen launcher before each inference job; any rejection stops the lane.
"""
from __future__ import annotations
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

E = Path(__file__).resolve().parents[1]
RUNTIME = E / 'runtime'
WORKFLOW = RUNTIME / 'Current_WORKFLOW.json'
TRAIN_RUN = E / 'training/Current/run/RUN.json'
TRAIN_RECEIPT = RUNTIME / 'train_current.json'
FREEZE = E / 'PROTOCOL_FREEZE.json'
LAUNCH = E / 'code/launch.py'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save(value):
    value['updated_unix'] = time.time()
    temporary = WORKFLOW.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(WORKFLOW)


def latest_step():
    path = TRAIN_RUN.with_name('TRAIN.jsonl')
    if not path.exists(): return 0
    with path.open('rb') as stream:
        stream.seek(0, 2); length = stream.tell(); stream.seek(max(0, length - 131072))
        lines = stream.read().splitlines()
    for line in reversed(lines):
        try: return int(json.loads(line)['step'])
        except (ValueError, KeyError): pass
    return 0


def verify_frozen(record):
    assert sha(FREEZE) == record['protocol_sha256'], 'Frozen protocol changed'
    manifest = read(FREEZE)
    for path, expected in manifest['source_sha256'].items():
        assert sha(path) == expected, ('Frozen executable source changed', path)
    assert sha(__file__) == record['worker_source_sha256'], 'Worker changed while running'


def launch_job(record, tag, script, arguments, gpu=None):
    verify_frozen(record)
    command = [sys.executable, str(LAUNCH)]
    if gpu is not None: command += ['--gpu', str(gpu)]
    command += ['--tag', tag, script, *arguments]
    receipt = RUNTIME / (tag + '.json')
    assert not receipt.exists() and not receipt.with_suffix('.log').exists(), ('Existing job tag', tag)
    job = {'tag': tag, 'command': command, 'gpu': gpu, 'status': 'RUNNING', 'start_unix': time.time()}
    record['jobs'].append(job); record['status'] = 'RUNNING_' + script.split('.')[0].upper()
    record['active_job'] = tag; save(record)
    print(json.dumps({'status': record['status'], 'command': command}), flush=True)
    process = subprocess.Popen(command, cwd=E)
    job['launcher_pid'] = process.pid; save(record)
    returncode = process.wait()
    job.update(exit_code=returncode, end_unix=time.time())
    if returncode:
        job['status'] = 'FAILED'
        raise RuntimeError(f'{tag} launcher exited {returncode}; no automatic retry or GPU takeover')
    result = read(receipt)
    assert result['status'] == 'COMPLETE' and result['exit_code'] == 0
    job.update(status='COMPLETE', receipt=str(receipt), receipt_sha256=sha(receipt))
    record['active_job'] = None; save(record)


def main():
    with (RUNTIME / 'Current_worker.lock').open('a', encoding='utf-8') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if WORKFLOW.exists(): raise FileExistsError('Preserve existing lane workflow; do not duplicate worker')
        record = {'status': 'WAITING_TRAINING', 'condition': 'Current', 'gpu': 1,
            'worker_pid': os.getpid(), 'worker_source': str(Path(__file__).resolve()),
            'worker_source_sha256': sha(__file__), 'protocol_sha256': sha(FREEZE),
            'launcher_sha256': sha(LAUNCH), 'started_unix': time.time(), 'jobs': [],
            'training_run': str(TRAIN_RUN), 'training_launcher_receipt': str(TRAIN_RECEIPT),
            'training_started_by_worker': False, 'automatic_training_resume': False,
            'gpu_policy': 'Frozen launcher occupancy check for every GPU1 inference; refuse occupied GPU',
            'active_job': None}
        save(record)
        try:
            verify_frozen(record)
            while True:
                receipt = read(TRAIN_RECEIPT)
                training = read(TRAIN_RUN) if TRAIN_RUN.exists() else {}
                record['training_launcher_status'] = receipt.get('status')
                record['training_status'] = training.get('status', 'PENDING')
                record['training_step'] = latest_step()
                if receipt.get('status') in ('FAILED', 'INTERRUPTED') or training.get('status') == 'failed':
                    raise RuntimeError('Formal Current training failed/interrupted; postprocessing stopped')
                if receipt.get('status') == 'COMPLETE':
                    assert receipt.get('exit_code') == 0
                    assert training.get('status') == 'complete' and training.get('final_step') == 2048
                    assert training['protocol_sha256'] == record['protocol_sha256']
                    record['training_run_sha256'] = sha(TRAIN_RUN)
                    record['training_launcher_sha256'] = sha(TRAIN_RECEIPT)
                    break
                record['status'] = 'WAITING_LAUNCHER_EXIT' if training.get('status') == 'complete' else 'WAITING_TRAINING'
                save(record); time.sleep(15)
            for sequence, short in [('DJI_0026_2', '0026'), ('DJI_0067_2', '0067')]:
                launch_job(record, 'current_lane_infer_' + short, 'infer.py',
                           ['--condition', 'Current', '--sequence', sequence, '--step', '2048'], gpu=1)
            launch_job(record, 'current_lane_score', 'evaluate.py', ['--condition', 'Current', '--step', '2048'])
            verify_frozen(record)
            record.update(status='COMPLETE', completed_unix=time.time(), active_job=None)
            save(record)
            print('CURRENT_LANE_COMPLETE: inference and scoring complete; global paired summary owned by root', flush=True)
        except BaseException as error:
            record.update(status='FAILED', error=repr(error), stopped_unix=time.time())
            save(record)
            raise


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise InterruptedError(f'Worker received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    main()
