#!/usr/bin/env python3
"""Run the predeclared four old development clips, never a confirmation set."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=['paired_mean','paired_max'],required=True);a=p.parse_args()
    stage=Path(__file__).resolve().parents[1];root=stage.parents[1]
    source=root/'official_deeppro'
    reference=source/'log/sem_seg/RGB-T__2025-01-14_15-27__SoftLoUloss_DeepPro-Plus_DataL40/checkpoints/best_model.pth'
    run=json.loads((stage/'runs'/a.variant/'RUN.json').read_text())
    if run['status']!='complete' or run['final_step']!=512:raise RuntimeError('Training incomplete')
    checkpoint=stage/'runs'/a.variant/'final_model.pth'
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=run['final_checkpoint_sha256']:raise RuntimeError('Weight hash changed')
    expected=json.loads((stage/'EVALUATION_FREEZE.json').read_text())['files']['pilot/infer_pilot.py']
    if hashlib.sha256((stage/'pilot/infer_pilot.py').read_bytes()).hexdigest()!=expected:raise RuntimeError('Inference source changed')
    done=[]
    for seq in ['DJI_0028_2','DJI_0028_4','DJI_0032_5','DJI_0047_3']:
        output=stage/'diagnostic_predictions'/a.variant/seq
        if (output/'COMPLETE.json').exists():raise FileExistsError('Inspect completed job rather than rerunning it')
        command=[sys.executable,str(stage/'pilot/infer_pilot.py'),'--source',str(source),
                 '--reference-checkpoint',str(reference),'--checkpoint',str(checkpoint),
                 '--dataset-root',str(stage/'data_diagnostics'),'--sequence',seq,'--output',str(output)]
        if not done:command+=['--verify-tiling']
        print(json.dumps({'starting':seq,'variant':a.variant,'command':command}),flush=True)
        subprocess.run(command,check=True)
        done.append(seq)
        status=stage/'runtime'/('DIAGNOSTIC_INFERENCE_'+a.variant+'.json')
        status.write_text(json.dumps({'variant':a.variant,'completed':done,'status':'COMPLETE' if len(done)==4 else 'RUNNING'},indent=2)+'\n')


if __name__=='__main__':main()
