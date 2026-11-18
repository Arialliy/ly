#!/usr/bin/env python3
"""CPU-only launch for isolated mismatch scoring or comparison tables."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

N = Path(__file__).resolve().parents[1]
R = N.parent.parent
IMAGE = 'sha256:eb956873870d737f2b68dff0724b8f6f56f778829d119b0938d29b7abeaa2f92'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary',action='store_true')
    args=parser.parse_args()
    label='summary' if args.summary else 'score'
    folder=N/'diagnostic_evaluation'
    folder.mkdir(exist_ok=True)
    receipt=folder/f'CPU_{label.upper()}_LAUNCH.json'
    if receipt.exists():
        raise FileExistsError(receipt)
    command=['docker','run','-d','--name','lre_misreference_'+label,'--network','none',
             '--user',f'{os.getuid()}:{os.getgid()}','--cpus','2','--memory','8g',
             '-e','OMP_NUM_THREADS=2','-e','MKL_NUM_THREADS=2','-e','OPENBLAS_NUM_THREADS=2',
             '-e','PYTHONDONTWRITEBYTECODE=1','-e','PYTHONUNBUFFERED=1','-e','MPLCONFIGDIR=/tmp/lre-mpl',
             '-v',f'{R}:{R}:ro','-v',f'{N}:{N}','--entrypoint','python',IMAGE,
             str(N/'diagnostic_code/score_misreference.py')]
    if args.summary:
        command.append('--summary')
    result=subprocess.run(command,text=True,capture_output=True)
    receipt.write_text(json.dumps(dict(start_unix=time.time(),command=command,cpu_only=True,
                       image=IMAGE,stdout=result.stdout,stderr=result.stderr,exit_code=result.returncode),indent=2)+'\n')
    print(result.stdout,result.stderr)
    assert result.returncode==0


if __name__=='__main__':
    main()
