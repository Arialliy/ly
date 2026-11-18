"""Run the unchanged scorer twice: fused grid, then fixed raw0.5 contexts."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

STAGE=Path(__file__).resolve().parents[2]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--variant',choices=['paired_mean','paired_max'],required=True)
    p.add_argument('--sequence',choices=['DJI_0028_2','DJI_0028_4','DJI_0032_5','DJI_0047_3'],required=True)
    p.add_argument('--expected-scoring-sha256',required=True)
    a=p.parse_args()
    score=STAGE/'pilot/score_pilot.py'
    assert hashlib.sha256(score.read_bytes()).hexdigest()==a.expected_scoring_sha256
    cmd=[sys.executable,str(score),'--data',str(STAGE/'data_diagnostics'/a.sequence),
         '--predictions',str(STAGE/'diagnostic_predictions'/a.variant/a.sequence),
         '--output',str(STAGE/'diagnostic_evaluation'/a.variant/a.sequence)]
    for suffix in [[],['--context-threshold','0.5']]:
        print(json.dumps({'command':cmd+suffix}),flush=True)
        subprocess.run(cmd+suffix,check=True)
    print(json.dumps({'status':'COMPLETE','variant':a.variant,'sequence':a.sequence,'fixed_context_threshold':0.5}),flush=True)

if __name__=='__main__':main()
