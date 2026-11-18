#!/usr/bin/env python3
import csv,collections,hashlib,json
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent;L=OUT.parent;C=L.parent;Q=C.parent;P=Q/'stage1_goal'
SEQS=['DJI_0026_2','DJI_0067_2'];FIELDS=['gt_count','matched_count','component_count','unmatched_components','positive_pixels','positive_pixels_outside_boxes']
def readcsv(p):return list(csv.DictReader(p.open()))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
rows=readcsv(OUT/'THRESHOLD_CURVES.csv');rec=readcsv(OUT/'LOW_RECOVERY_OBJECTS.csv');old=readcsv(L/'THRESHOLD_CURVES.csv')
assert len(rows)==66
lookup={(r['group'],r['mode'],float(r['threshold'])):r for r in rows};count=0;gain=loss=lowgain=0
for seq in SEQS:
 for mode in ['O','S']:
  with np.load(OUT/f'{mode}_{seq}_scores.npz') as z:
   assert z['thresholds'].tolist()==[x/100 for x in range(40,51)]
   for ti,tau in enumerate(z['thresholds']):
    r=lookup[seq,mode,float(tau)];m=z['matched'][ti].astype(bool);small=z['small'];low=z['low'];b=z['baseline']
    assert int(r['small_matched'])==int(m[small].sum())
    assert int(r['low_cohort_recovered'])==int(m[low].sum())
    assert int(r['gained_vs_original05'])==int((small&m&~b).sum())
    assert int(r['lost_vs_original05'])==int((small&~m&b).sum())
    for j,f in enumerate(FIELDS):assert int(r[f])==int(z['frame_counts'][ti,:,j].sum());count+=1
   if mode=='S':
    aa=z['matched'][4].astype(bool);bb=z['matched'][10].astype(bool)
    gain+=int((small&aa&~bb).sum());loss+=int((small&~aa&bb).sum());lowgain+=int((low&aa&~bb).sum())
for r in rows:
 assert int(r['component_count'])-int(r['matched_count'])==int(r['unmatched_components'])
 assert int(r['U_within_budget'])==int(int(r['unmatched_components'])<=int(r['U_budget_reference']))
 if r['group']=='pooled':
  assert int(r['small_matched'])==700+int(r['gained_vs_original05'])-int(r['lost_vs_original05'])
  sub=[lookup[s,r['mode'],float(r['threshold'])] for s in SEQS]
  for f in ['small_gt','small_matched','low_cohort_gt','low_cohort_recovered','gained_vs_original05','lost_vs_original05']+FIELDS:
   assert int(r[f])==sum(int(x[f]) for x in sub);count+=1
 if float(r['threshold']) in [.4,.5]:
  oldr=next(x for x in old if x['group']==r['group'] and x['mode']==r['mode'] and x['threshold']==r['threshold'])
  for f in oldr:assert str(oldr[f])==str(r[f]);count+=1
cohort={(r['sequence'],r['frame'],r['annotation_id']) for r in readcsv(L/'LOW_COHORT.csv')};seen=set();counts=collections.Counter()
for r in rec:
 identity=(r['sequence'],r['frame'],r['annotation_id']);assert identity in cohort
 key=(r['mode'],r['threshold'])+identity;assert key not in seen;seen.add(key)
 counts[r['sequence'],r['mode'],r['threshold']]+=1
for r in rows:
 if r['group']!='pooled':assert int(r['low_cohort_recovered'])==counts[r['group'],r['mode'],r['threshold']]
s=json.loads((OUT/'SUMMARY.json').read_text())
for path,h in s['source_sha256'].items():assert sha(Path(path))==h
output=dict(status='PASS',row_count=66,array_and_table_comparisons=count,recovery_rows=len(rec),source_hashes_unchanged=len(s['source_sha256']),endpoint_all_array_equality=s['endpoint_exact_checks'],
 S044_vs_S05=dict(gained=gain,lost=loss,low_cohort_gained=lowgain),exploratory_supplement_fixed_nine_new_thresholds=True)
(OUT/'VALIDATION.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2))
