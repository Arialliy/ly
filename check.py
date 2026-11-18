#!/usr/bin/env python3
"""Validate source copies; --tests runs existing CPU tests using only bundled project code."""
import argparse
import ast
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

BASE = Path(__file__).resolve().parent
IMAGE = 'sha256:eb956873870d737f2b68dff0724b8f6f56f778829d119b0938d29b7abeaa2f92'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tests', action='store_true', help='Run CPU imports and the existing 37 temporal tests; no training')
    parser.add_argument('--test-deps', type=Path, help='Existing pytest dependency directory, or omit to use recorded local environment')
    args = parser.parse_args()
    manifest = json.loads((BASE/'provenance/SOURCE_MANIFEST.json').read_text())
    python_count = 0
    for row in manifest['files']:
        path = BASE/row['path']
        assert path.is_file() and not path.is_symlink(), path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256'], path
        if path.suffix == '.py':
            ast.parse(path.read_bytes(), filename=str(path))
            python_count += 1
    record = dict(status='PASS', source_files=len(manifest['files']), python_files=python_count,
                  original_source_content_changed=False, utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (BASE/'validation/SOURCE_CHECK.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record), flush=True)
    if not args.tests:
        return
    root = Path(manifest['canonical_project_root'])
    q = root/'temporal_diagnosis_20261010'
    n = q/'tlfp_pilot_20261012'
    deps = args.test_deps or BASE/'environment/site'
    if not deps.is_dir():
        parser.error('pytest dependency directory missing; provide --test-deps for the existing environment')
    run = ['docker','run','--rm','--network','none','--user',f'{os.getuid()}:{os.getgid()}',
           '--cpus','2','--memory','8g',
           '-v',f'{BASE / "sources"}:{root}:ro',
           '-v',f'{BASE / "tools"}:/package_tools:ro',
           '-v',f'{BASE / "validation"}:/validation:rw',
           '-v',f'{deps.resolve()}:/test_deps:ro',
           '-e',f'PYTHONPATH=/test_deps:{n / "code"}',
           '-e','PYTHONDONTWRITEBYTECODE=1','-e','OMP_NUM_THREADS=2',
           '-e','MKL_NUM_THREADS=2','-e','OPENBLAS_NUM_THREADS=2',
           '-e','MPLCONFIGDIR=/tmp/tlfp-package-matplotlib','--entrypoint','python',IMAGE]
    jobs = [('imports', ['/package_tools/import_check.py']),
            ('temporal_cpu', ['-m','pytest','-q','-p','no:cacheprovider',str(n/'code/test_temporal_lfp.py'),
                              '--junitxml=/validation/TEMPORAL_CPU.xml'])]
    receipts = []
    for label, argv in jobs:
        result = subprocess.run(run + argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (BASE/'validation'/f'{label}.log').write_text(result.stdout)
        receipts.append(dict(job=label, command=run+argv, exit_code=result.returncode))
        (BASE/'validation/CPU_EXECUTION.json').write_text(json.dumps(receipts, indent=2)+'\n')
        print(result.stdout, flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)

if __name__ == '__main__':
    main()
