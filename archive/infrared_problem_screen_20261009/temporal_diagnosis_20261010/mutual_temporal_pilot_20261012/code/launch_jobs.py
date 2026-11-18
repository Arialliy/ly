#!/usr/bin/env python3
"""Launch only the preregistered three variants or their fixed diagnostics."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

N = Path(__file__).resolve().parents[1]
Q, R = N.parent, N.parent.parent
SOURCE = R / 'official_deeppro'
REF = SOURCE / 'log/sem_seg/RGB-T__2025-01-14_15-27__SoftLoUloss_DeepPro-Plus_DataL40/checkpoints/best_model.pth'
IMAGE = 'deeppro-official-eval:20261009'
IMAGE_ID = 'sha256:eb956873870d737f2b68dff0724b8f6f56f778829d119b0938d29b7abeaa2f92'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('job', choices=['train', 'infer'])
    p.add_argument('--gpu', required=True, type=int)
    p.add_argument('--variant', choices=['E0', 'E1', 'E2'], required=True)
    p.add_argument('--sequence', choices=['DJI_0026_2', 'DJI_0067_2'])
    p.add_argument('--step', type=int, choices=[128, 256, 384, 512], default=512)
    p.add_argument('--diagnostic', choices=['none', 'mass', 'current'], default='none')
    a = p.parse_args()
    assert a.job != 'train' or (a.step == 512 and a.diagnostic == 'none' and not a.sequence)
    assert a.diagnostic == 'none' or (a.variant == 'E2' and a.step == 512)
    assert a.job != 'infer' or a.sequence
    image = subprocess.check_output(['docker', 'image', 'inspect', IMAGE, '--format', '{{.Id}}'], text=True).strip()
    assert image == IMAGE_ID
    spec = json.loads((R/'official_reproduction/env-spec.json').read_text())
    spec_hash = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert spec_hash == 'de8a7eb93aac154354646ae9b59c7ba4086c2ad1d8a704894c9c1de86a4d6d86'
    frozen = json.loads((N/'PROTOCOL_FREEZE.json').read_text())
    assert frozen['protocol_sha256'] == sha(N/'PROTOCOL.md')
    for rel, digest in frozen['code_sha256'].items():
        assert sha(N/rel) == digest, rel
    for path, digest in frozen.get('external_sha256', {}).items():
        assert sha(path) == digest, path
    for rel, digest in frozen['preflight_sha256'].items():
        assert sha(N/rel) == digest, rel
        assert json.loads((N/rel).read_text())['status'] == 'PASS', rel
    used = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits', '-i', str(a.gpu)], text=True).strip())
    assert used < 500, f'GPU {a.gpu} occupied: {used} MiB'
    condition = a.variant if a.diagnostic == 'none' else 'E2_' + a.diagnostic
    if a.step != 512:
        condition += f'_step{a.step}'
    tail = '' if a.job == 'train' else '_' + a.sequence
    name = 'mutual_' + a.job + '_' + condition + tail + '_20261012'
    logdir = N / ('training' if a.job == 'train' else 'inference_logs') / condition
    if a.sequence:
        logdir /= a.sequence
    logdir.mkdir(parents=True, exist_ok=True)
    assert not (logdir/'LAUNCH.json').exists(), 'Refusing duplicate formal job'
    cmd = ['docker', 'run', '-d', '--name', name, '--gpus', 'all', '--user', '1004:1004',
           '--cpus', '4', '--memory', '20g', '--shm-size', '2g', '--network', 'none']
    for env in [f'CUDA_VISIBLE_DEVICES={a.gpu}', 'OMP_NUM_THREADS=2', 'MKL_NUM_THREADS=2',
                'OPENBLAS_NUM_THREADS=2', 'CUBLAS_WORKSPACE_CONFIG=:4096:8',
                'PYTHONDONTWRITEBYTECODE=1', 'PYTHONUNBUFFERED=1', 'MPLCONFIGDIR=/tmp/deeppro-matplotlib']:
        cmd += ['-e', env]
    # Only this experiment is writable. All prior evidence and data are read-only.
    cmd += ['-v', f'{R}:{R}:ro', '-v', f'{N}:{N}:rw', '--entrypoint', 'python', IMAGE]
    if a.job == 'train':
        cmd += [str(N/'code/train_mutual.py'), '--variant', a.variant, '--radius', '2',
                '--source', str(SOURCE), '--checkpoint', str(REF), '--dataset-root', str(Q/'stage1_goal/data'),
                '--selection', str(Q/'stage1_goal/DATA_SELECTION.json'), '--plan', str(Q/'stage1_goal/pilot/TRAIN_PLAN.json'),
                '--output', str(logdir/'run'), '--steps', '512', '--seed', '46', '--lr', '0.0001', '--device', 'cuda:0']
    else:
        cp = N/'training'/a.variant/'run'/f'snapshot_{a.step:03d}.pth'
        assert cp.is_file()
        assert json.loads((cp.parent/'RUN.json').read_text())['status'] == 'complete'
        cmd += [str(N/'code/infer_mutual.py'), '--source', str(SOURCE), '--reference-checkpoint', str(REF),
                '--checkpoint', str(cp), '--dataset-root', str(Q/'stage1_goal/data'), '--sequence', a.sequence,
                '--variant', a.variant, '--diagnostic', a.diagnostic, '--step', str(a.step),
                '--output', str(N/'predictions'/condition/a.sequence), '--verify-tiling']
        if a.variant == 'E2' and a.step == 512 and a.diagnostic == 'none':
            cmd.append('--provenance')
    record = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'command': cmd,
              'container_name': name, 'gpu': a.gpu, 'gpu_preflight_memory_mib': used,
              'image_id': image, 'environment_spec_sha256': spec_hash,
              'protocol_freeze_sha256': sha(N/'PROTOCOL_FREEZE.json'), 'condition': condition,
              'step': a.step, 'job': a.job, 'sequence': a.sequence}
    result = subprocess.run(cmd, capture_output=True, text=True)
    record.update(launch_returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    (logdir/'LAUNCH.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record), flush=True)
    result.check_returncode()


if __name__ == '__main__':
    main()
