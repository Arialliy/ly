"""Host launcher for the isolated reference package; refuses occupied GPUs."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['tests', 'witness', 'upstream', 'integration'])
    parser.add_argument('--gpu', type=int, choices=[1, 2, 3])
    parser.add_argument('--receipt', required=True, help='Unique path within this package')
    args = parser.parse_args()
    package = Path(__file__).resolve().parent
    project = package.parents[1]
    receipt_path = (package / args.receipt).resolve()
    if package not in receipt_path.parents or receipt_path.exists():
        raise SystemExit('Receipt must be a NEW file within this package.')
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    spec = json.loads((package / 'runtime/ENV_SPEC.json').read_text())
    receipt = {'mode': args.mode, 'gpu': args.gpu, 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'spec_sha256': hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest()}
    def save():
        receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    if args.gpu is not None:
        state = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
        receipt['gpu_before'] = state
        row = next(line for line in state.splitlines() if int(line.split(',')[0]) == args.gpu)
        if int(row.split(',')[1]) >= 500:
            receipt.update(status='BLOCKED_OCCUPIED_BEFORE_DOCKER', exit_code=3)
            save()
            print(json.dumps(receipt))
            return 3
    if args.mode == 'integration' and args.gpu is None:
        raise SystemExit('Integration witness requires an explicitly selected idle authorized GPU.')
    artifact = receipt_path.with_suffix('.result.json')
    device = 'cuda' if args.gpu is not None else 'cpu'
    if args.mode == 'tests':
        command = ['-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'test_haar_reference.py', 'test_adapter.py', 'test_upstream_loader.py', '--junitxml=' + str(receipt_path.with_suffix('.junit.xml'))]
    elif args.mode == 'witness':
        command = ['kernel_witness.py', '--device', device, '--output', str(artifact)]
    elif args.mode == 'upstream':
        command = ['check_upstream_lfp.py', '--source', str(package / 'upstream/NS_FPN.py'), '--device', device, '--output', str(artifact)]
    else:
        command = ['check_deeppro_integration.py', '--device', device, '--output', str(artifact)]
    docker = ['docker', 'run', '--rm', '--network', 'none', '--user', f'{os.getuid()}:{os.getgid()}', '--cpus', '4', '--memory', '20g', '--shm-size', '2g']
    if args.gpu is not None:
        docker += ['--gpus', 'all', '-e', f'CUDA_VISIBLE_DEVICES={args.gpu}']
    env = {**spec['environment'], 'PYTHONUNBUFFERED': '1', 'MPLCONFIGDIR': '/tmp', 'PYTHONPATH': str(package / 'dependencies/site') + ':' + str(package)}
    for key, value in env.items():
        docker += ['-e', f'{key}={value}']
    docker += ['-v', f'{project}:{project}:ro', '-v', f'{package}:{package}:rw', '-w', str(package), '--entrypoint', '/opt/conda/bin/python', spec['base_image'], *command]
    receipt['command'] = docker
    receipt['status'] = 'RUNNING'
    receipt['code_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(package.glob('*.py'))}
    save()
    log_path = receipt_path.with_suffix('.log')
    with log_path.open('w') as log:
        process = subprocess.Popen(docker, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end='', flush=True)
        code = process.wait()
    receipt.update(status='COMPLETE' if code == 0 else 'NONZERO_EXIT', exit_code=code,
                   finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), log=str(log_path))
    save()
    return code


if __name__ == '__main__':
    sys.exit(main())
