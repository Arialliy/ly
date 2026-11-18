#!/usr/bin/env python3
"""Launch CPU-only scoring in the already verified immutable local image."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent.parent
IMAGE = 'deeppro-official-eval:20261009'
IMAGE_ID = 'sha256:eb956873870d737f2b68dff0724b8f6f56f778829d119b0938d29b7abeaa2f92'
ENV_SHA = 'de8a7eb93aac154354646ae9b59c7ba4086c2ad1d8a704894c9c1de86a4d6d86'


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--condition', choices=['R0', 'R1', 'R2'])
    mode.add_argument('--summary', action='store_true')
    parser.add_argument('--sequence', choices=['DJI_0026_2', 'DJI_0067_2'])
    args = parser.parse_args()
    assert not (args.summary and args.sequence)
    spec = json.loads((ROOT/'official_reproduction/env-spec.json').read_text())
    actual_env_sha = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert actual_env_sha == ENV_SHA
    actual_image_id = subprocess.check_output(['docker', 'image', 'inspect', IMAGE, '--format', '{{.Id}}'], text=True).strip()
    assert actual_image_id == IMAGE_ID
    label = 'summary' if args.summary else args.condition + ('_' + args.sequence if args.sequence else '')
    name = 'ord_score_' + label.lower()
    directory = BASE/'evaluation'
    directory.mkdir(exist_ok=True)
    receipt = directory/(name + '_LAUNCH.json')
    assert not receipt.exists(), receipt
    command = ['docker', 'run', '-d', '--name', name, '--user', '1004:1004', '--cpus', '2', '--memory', '8g']
    for value in ['OMP_NUM_THREADS=2', 'MKL_NUM_THREADS=2', 'OPENBLAS_NUM_THREADS=2',
                  'PYTHONDONTWRITEBYTECODE=1', 'PYTHONUNBUFFERED=1', 'MPLCONFIGDIR=/tmp/ord-matplotlib']:
        command.extend(['-e', value])
    command += ['-v', f'{ROOT}:{ROOT}:ro', '-v', f'{BASE}:{BASE}', '--entrypoint', 'python', IMAGE,
                str(BASE/'evaluation_code/evaluate.py')]
    if args.summary:
        command += ['--summary']
    else:
        command += ['--condition', args.condition]
        if args.sequence:
            command += ['--sequence', args.sequence]
    result = subprocess.run(command, text=True, capture_output=True)
    receipt.write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     command=command, exit_code=result.returncode, stdout=result.stdout,
                     stderr=result.stderr, image_id=IMAGE_ID, env_sha256=ENV_SHA,
                     cpu_only=True, warm_reuse=True, old_project_mounted_read_only=True), indent=2) + '\n')
    print(result.stdout, result.stderr)
    assert result.returncode == 0


if __name__ == '__main__':
    main()
