"""Durable CPU-only scoring handoff after both diagnostic inference receipts complete."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

E = Path(__file__).resolve().parents[1]
SEQS = ['DJI_0026_2', 'DJI_0067_2']
RECEIPT = E / 'runtime/SCORING_WORKFLOW.json'


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def save(value):
    temporary = RECEIPT.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(RECEIPT)


def main():
    assert not RECEIPT.exists(), 'Preserve the existing workflow receipt'
    freeze = E / 'PROTOCOL_FREEZE.json'
    state = {'status': 'WAITING_DIAGNOSTICS', 'pid': os.getpid(), 'start_unix': time.time(),
        'worker_source_sha256': sha(__file__), 'freeze_sha256': sha(freeze), 'commands': []}
    save(state)
    try:
        deadline = time.monotonic() + 4 * 3600
        while True:
            if time.monotonic() > deadline: raise TimeoutError('Diagnostic handoff exceeded four hours')
            assert sha(freeze) == state['freeze_sha256'], 'Protocol freeze changed after worker startup'
            pending = []
            for sequence in SEQS:
                launcher = E / 'runtime' / ('diagnostic_' + sequence.lower() + '_resume.json')
                launch = read(launcher)
                if launch['status'] in ('FAILED', 'INTERRUPTED'):
                    raise RuntimeError('Inference launcher failed: ' + str(launcher))
                paths = [E / 'traces' / sequence / 'COMPLETE.json'] + [
                    E / 'predictions' / condition / 'step_2048' / sequence / 'COMPLETE.json'
                    for condition in ('HistoryCurrent', 'HistoryOff')]
                pending.extend(str(path) for path in paths if not path.exists() or read(path)['status'] != 'COMPLETE')
                if launch['status'] != 'COMPLETE': pending.append(str(launcher))
            state['pending'] = pending; state['last_check_unix'] = time.time(); save(state)
            if not pending: break
            time.sleep(10)
        for path, expected in read(freeze)['source_sha256'].items():
            assert sha(path) == expected, 'Frozen source changed: ' + path
        jobs = [('score_history_current', ['--condition', 'HistoryCurrent']),
                ('score_history_off', ['--condition', 'HistoryOff']),
                ('score_summary', ['--summary'])]
        for tag, args in jobs:
            command = [sys.executable, str(E / 'code/launch.py'), '--tag', tag, 'evaluate.py', *args]
            state.update(status='SCORING', current_job=tag)
            state['commands'].append(command); save(state)
            result = subprocess.run(command, cwd=E)
            if result.returncode: raise RuntimeError(f'CPU scoring failed: {tag} exit={result.returncode}')
            receipt = read(E / 'runtime' / (tag + '.json'))
            assert receipt['status'] == 'COMPLETE' and receipt['exit_code'] == 0 and receipt['gpu'] is None
        summary = E / 'evaluation/step_2048/SUMMARY_COMPLETE.json'
        assert read(summary)['status'] == 'COMPLETE'
        state.update(status='COMPLETE', end_unix=time.time(), summary=str(summary), summary_sha256=sha(summary))
        save(state)
    except BaseException as error:
        state.update(status='FAILED', error=repr(error), end_unix=time.time())
        save(state)
        raise


if __name__ == '__main__': main()
