#!/usr/bin/env python3
"""Audit descriptive slot strata against the already saved per-object rows."""
import csv
import hashlib
import json
from pathlib import Path

P=Path(__file__).resolve().parents[1]
d=P/'summary/development'
rows=list(csv.DictReader((d/'OBJECTS.csv').open()))
fields=['original_fused_match','candidate_fused_match','fused_gain','fused_loss',
        'a_original_match','a_candidate_match','b_original_match','b_candidate_match',
        'protected_original','protected_lost']
groups=['all1417','strong97','suppression24','support42','stable_high502','stable_low778']
result=[]
for v in ['paired_mean','paired_max']:
    for group in groups:
        sizes=[]
        for inside in [True,False]:
            rr=[r for r in rows if r['variant']==v and (group=='all1417' or r[group]=='1')
                and all(8<=int(r['local_frame'])-int(r[s])<=31 for s in ['a_start','b_start'])==inside]
            sizes.append(len(rr))
            result.append(dict(variant=v,cohort=group,
                selected_original_pair_slot_group=('both_selected_slots_8_to_31' if inside else
                                                  'at_least_one_selected_slot_outside_8_to_31'),
                n=len(rr),**{k:sum(int(r[k]) for r in rr) for k in fields}))
        assert sum(sizes)=={'all1417':1417,'strong97':97,'suppression24':24,'support42':42,
                            'stable_high502':502,'stable_low778':778}[group]
path=d/'BOUNDARY.csv'
if path.exists():
    actual=list(csv.DictReader(path.open()))
    assert actual==[{k:str(v) for k,v in r.items()} for r in result]
else:
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
meta=dict(status='PASS',rule='Original selected pair only, both slots 8..31 vs at least one outside',
          interpretation='Descriptive, original response-dependent pair selection, no new causal claim',
          object_rows_sha256=sha(d/'OBJECTS.csv'),output_sha256=sha(path),script_sha256=sha(Path(__file__)))
with (d/'BOUNDARY_META.json').open('x') as f:json.dump(meta,f,indent=2);f.write('\n')
print(json.dumps(meta))
