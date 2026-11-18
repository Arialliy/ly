#!/usr/bin/env python3
"""Run CPU contracts in the existing pinned image; never launch training/GPU work."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

E = Path(__file__).resolve().parent
ROOT = E.parents[1]
IMAGE = 'sha256:eb956873870d737f2b68dff0724b8f6f56f778829d119b0938d29b7abeaa2f92'


def source_hashes():
    paths = [ROOT / 'source_relational_reducer.py', Path(__file__)]
    paths += sorted((E / 'code').glob('*.py')) + sorted((E / 'tests').glob('*.py'))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    image = subprocess.check_output(['docker', 'image', 'inspect', IMAGE, '--format', '{{.Id}}'], text=True).strip()
    if image != IMAGE:
        raise RuntimeError('Pinned image mismatch')
    hashes = source_hashes()
    code = '''import json, sys, torch, unittest
torch.set_num_threads(1)
suite = unittest.defaultTestLoader.discover('/project/experiments/m1_source_relations_v1/tests', pattern='test_*.py')
result = unittest.TextTestRunner(verbosity=2).run(suite)
print('CHECK_RESULT_JSON=' + json.dumps(dict(tests=result.testsRun, failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped), torch=torch.__version__, python=sys.version.split()[0], success=result.wasSuccessful())))
raise SystemExit(not result.wasSuccessful())
'''
    command = ['docker', 'run', '--rm', '-i', '--network', 'none',
               '--user', f'{os.getuid()}:{os.getgid()}', '--cpus', '2', '--memory', '8g',
               '-v', f'{ROOT}:/project:ro',
               '-e', 'PYTHONPATH=/project:/project/environment/site',
               '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'OMP_NUM_THREADS=1',
               '-e', 'MKL_NUM_THREADS=1', '-e', 'OPENBLAS_NUM_THREADS=1',
               '--entrypoint', 'python', image, '-']
    result = subprocess.run(command, input=code, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout, end='')
    summaries = [line.split('=', 1)[1] for line in result.stdout.splitlines() if line.startswith('CHECK_RESULT_JSON=')]
    summary = json.loads(summaries[-1]) if summaries else {'success': False, 'error': 'No unittest summary'}
    unchanged = hashes == source_hashes()
    record = dict(summary, source_unchanged_during_check=unchanged,
                  status='PASS' if result.returncode == 0 and unchanged and summary['success'] else 'FAIL',
                  source_sha256=hashes, image=image, training_started=False, GPU_used=False,
                  native_PDA_forward_verified=False, exit_code=result.returncode)
    directory = E / 'checks'
    directory.mkdir(exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    path = directory / f'CPU_{stamp}.json'
    path.with_suffix('.log').write_text(result.stdout, encoding='utf-8')
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'{record["status"]}: {path}')
    raise SystemExit(0 if record['status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
