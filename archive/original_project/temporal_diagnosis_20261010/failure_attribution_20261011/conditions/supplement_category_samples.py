#!/usr/bin/env python3
"""Retain frozen primary samples and add deterministic missing category strata."""
import csv
import hashlib
import json
from pathlib import Path

D=Path(__file__).resolve().parent
def read(p):
    with p.open() as f: return list(csv.DictReader(f))
def write(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
base=read(D/'VISUAL_SAMPLE_INDEX.csv'); objects=read(D/'OBJECT_CONDITIONS.csv')
old={p.name:sha(p) for p in (D/'VISUAL_SAMPLE_INDEX.csv',D/'SAMPLE_SELECTION.json')}
covered={(r['sequence'],r['category'],r['response_layer']) for r in base}
population=[r for r in objects if r['group']=='all11_failure']
strata=sorted({(r['sequence'],r['category'],r['all11_failure_response_layer']) for r in population})
extra=[]; recorded=[]
for seq,cat,layer in strata:
    if (seq,cat,layer) in covered:continue
    rr=[r for r in population if (r['sequence'],r['category'],r['all11_failure_response_layer'])==(seq,cat,layer)]
    tid=min(int(r['track_id']) for r in rr)
    candidates=sorted((r for r in rr if int(r['track_id'])==tid),key=lambda r:(int(r['source_frame']),int(r['annotation_id'])))
    r=candidates[len(candidates)//2]
    sample={k:r.get(k,'') for k in base[0]}
    sample.update(sample_id=f'{seq}_track{tid}_{layer}_ann{r["annotation_id"]}_category_supplement',
                  response_layer=layer,stratum_track_object_frames=len(candidates),representative_index=len(candidates)//2)
    extra.append(sample)
    recorded.append(dict(sequence=seq,category=cat,response_layer=layer,population_frames=len(rr),population_tracks=len({r['track_id'] for r in rr}),selected_track=tid,annotation_id=int(r['annotation_id'])))
write(D/'SUPPLEMENTAL_VISUAL_SAMPLE_INDEX.csv',extra)
write(D/'ALL_VISUAL_SAMPLE_INDEX.csv',[*base,*extra])
freeze=dict(status='FROZEN_BEFORE_SUPPLEMENTAL_IMAGE_INSPECTION',primary_selection_unchanged=old,
            reason='Primary track-ID sampling lacked some already documented sequence/category/response strata; coverage supplement requested by root before supplemental image review.',
            procedure='For each missing (sequence,category,response layer), choose lowest track_id, then source-frame sorted item floor(n/2); retain original 12 unchanged.',
            role='Exploratory category coverage only; no random prevalence or independent confirmation claim.',
            input_objects_sha256=sha(D/'OBJECT_CONDITIONS.csv'),script_sha256=sha(Path(__file__)),supplement_count=len(extra),total_count=len(base)+len(extra),selected_strata=recorded,
            outputs_sha256={n:sha(D/n) for n in ['SUPPLEMENTAL_VISUAL_SAMPLE_INDEX.csv','ALL_VISUAL_SAMPLE_INDEX.csv']})
(D/'SUPPLEMENTAL_SAMPLE_SELECTION.json').write_text(json.dumps(freeze,ensure_ascii=False,indent=2)+'\n')
assert old=={p.name:sha(p) for p in (D/'VISUAL_SAMPLE_INDEX.csv',D/'SAMPLE_SELECTION.json')}
print(json.dumps(freeze,ensure_ascii=False))
