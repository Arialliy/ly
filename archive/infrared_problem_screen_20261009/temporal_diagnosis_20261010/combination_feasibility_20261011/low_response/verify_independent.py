#!/usr/bin/env python3
"""Independent CSV/accounting and nonoverlapping-box audits (stdlib only)."""
import collections,csv,hashlib,json
from pathlib import Path
OUT=Path(__file__).resolve().parent; C=OUT.parent;Q=C.parent;P=Q/'stage1_goal';V=Q/'problem_validity_20261011'
SEQS=['DJI_0026_2','DJI_0067_2']
def csvrows(p):return list(csv.DictReader(p.open()))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rows=csvrows(OUT/'THRESHOLD_CURVES.csv');cohort=csvrows(OUT/'LOW_COHORT.csv');recoveries=csvrows(OUT/'LOW_RECOVERY_OBJECTS.csv')
summary=json.loads((OUT/'SUMMARY.json').read_text())
assert len(rows)==138 and len(cohort)==4342
fields={'strict_small_gt':'small_gt','strict_small_matched':'small_matched','gt':'gt_count','matched':'matched_count','components':'component_count',
        'unmatched_components':'unmatched_components','positive_pixels':'positive_pixels','positive_pixels_outside_boxes':'positive_pixels_outside_boxes'}
checks=0
for seq in SEQS:
 for mode in ['O','S']:
  path=P/'evaluation/official'/seq/'official_fused_curves.csv' if mode=='O' else V/'simple_controls/symmetric5_probmax'/seq/'fused_curves.csv'
  source={float(r['threshold']):r for r in csvrows(path)}
  for r in rows:
   if r['group']!=seq or r['mode']!=mode:continue
   for a,b in fields.items():assert int(source[float(r['threshold'])][a])==int(r[b]);checks+=1
for r in rows:
 assert int(r['small_matched'])==700+int(r['gained_vs_original05'])-int(r['lost_vs_original05']) if r['group']=='pooled' else True
 assert int(r['unmatched_components'])==int(r['component_count'])-int(r['matched_count'])
 assert int(r['U_within_budget'])==int(int(r['unmatched_components'])<=int(r['U_budget_reference']))
 if r['group']=='pooled':
  sub=[x for x in rows if x['group'] in SEQS and x['mode']==r['mode'] and x['threshold']==r['threshold']]
  assert len(sub)==2
  for k in ['small_gt','small_matched','low_cohort_gt','low_cohort_recovered','gained_vs_original05','lost_vs_original05','gt_count','matched_count','component_count','unmatched_components','positive_pixels','positive_pixels_outside_boxes']:
   assert int(r[k])==sum(int(x[k]) for x in sub);checks+=1
ids={(r['sequence'],r['frame'],r['annotation_id']) for r in cohort};assert len(ids)==4342
recovery_counts=collections.Counter();unique=set()
for r in recoveries:
 assert (r['sequence'],r['frame'],r['annotation_id']) in ids
 key=(r['mode'],r['threshold'],r['sequence'],r['frame'],r['annotation_id'])
 assert key not in unique;unique.add(key)
 recovery_counts[r['mode'],r['threshold']]+=1
for r in rows:
 if r['group']=='pooled':assert int(r['low_cohort_recovered'])==recovery_counts[r['mode'],r['threshold']]
nonoverlap=smallcount=allcount=0
for seq in SEQS:
 meta=json.loads((P/'evaluation/official'/seq/'official_fused_COMPLETE.json').read_text())['objects'];allcount+=len(meta)
 frames=collections.defaultdict(list)
 for r in meta:frames[r['frame']].append(r)
 for frame,aa in frames.items():
  for a in aa:
   if not a['strict_small']:continue
   smallcount+=1;x,y,w,h=a['bbox']
   assert 0<w<=16 and 0<h<=16 and 0<=x<x+w<=640 and 0<=y<y+h<=512
   for b in aa:
    if a['annotation_id']==b['annotation_id']:continue
    bx,by,bw,bh=b['bbox']
    area=max(0,min(x+w,bx+bw)-max(x,bx))*max(0,min(y+h,by+bh)-max(y,by))
    assert area==0,(seq,frame,a['annotation_id'],b['annotation_id'],area)
   nonoverlap+=1
assert smallcount==nonoverlap==5200 and allcount==16297
matrix=csvrows(C/'support/MATCH_MATRIX.csv')
any_support=sum(any(int(r[f'w{s:03d}'])==1 for s in range(0,121,12)) for r in matrix)
assert len(matrix)==5200 and any_support==809
assert {(r['sequence'],r['frame'],r['annotation_id']) for r in matrix}=={(r['sequence'],r['frame'],r['annotation_id']) for r in csvrows(C/'support/OBJECT_SUPPORT.csv')}
unchanged=[]
for p,h in summary['source_sha256'].items():assert sha(Path(p))==h;unchanged.append(p)
below=[r for r in rows if r['group']=='pooled' and float(r['threshold'])<.5]
assert all(int(r['low_cohort_recovered'])>0 and not int(r['U_within_budget']) for r in below)
min_u={mode:min((r for r in below if r['mode']==mode),key=lambda r:int(r['unmatched_components'])) for mode in ['O','S']}
output=dict(status='PASS',numeric_source_curve_comparisons=checks,recovery_rows=len(recoveries),
            identity_and_aggregation_checks=True,source_hashes_unchanged=len(unchanged),
            all_valid_gt=allcount,small_gt=smallcount,small_boxes_independently_confirmed_nonoverlap_with_all_valid_gt=nonoverlap,
            matrix_row_count=len(matrix),any_window_actual_match_count=any_support,all_window_no_match_count=5200-any_support,
            match_matrix_sha256=sha(C/'support/MATCH_MATRIX.csv'),
            threshold_points_per_mode=23,no_budget_feasible_low_recovery_at_tested_points=True,
            lowest_U_low_recovery_rows=min_u,script_sha256=sha(Path(__file__)),helper_sha256=sha(Q/'analysis/analyze.py'))
(OUT/'INDEPENDENT_VALIDATION.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(output,indent=2))
