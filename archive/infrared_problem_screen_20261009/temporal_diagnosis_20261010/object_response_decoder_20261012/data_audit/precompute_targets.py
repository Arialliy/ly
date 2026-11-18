#!/usr/bin/env python3
"""Frozen-plan all-window target receipt, CPU only; no persistent dense cache."""
import hashlib
import json
from pathlib import Path
import sys
import time

N=Path(__file__).resolve().parents[1]
Q=N.parent
sys.path.insert(0,str(N/'code'))
import targets

p=Q/'stage1_goal/pilot/TRAIN_PLAN.json'
plan=json.loads(p.read_text())
started=time.monotonic()
sequences={s:targets.base.SequenceData(Q/'stage1_goal/data',s) for s in plan['train_sequences']}
assert {s:v.fingerprint for s,v in sequences.items()}==plan['data_fingerprints']
out=N/'data_audit/frozen_target_receipts'
result=targets.precompute_plan_targets(sequences,plan,out,verify_legacy=True,save_cache=False)
result.update(status='PASS',elapsed_seconds=time.monotonic()-started,plan_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),targets_source_sha256=hashlib.sha256((N/'code/targets.py').read_bytes()).hexdigest(),source_data_fingerprints=plan['data_fingerprints'])
(out/'MANIFEST.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('windows','source_data_fingerprints')},indent=2),flush=True)
