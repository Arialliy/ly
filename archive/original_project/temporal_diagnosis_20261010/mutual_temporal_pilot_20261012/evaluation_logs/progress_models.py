import sys,json
from pathlib import Path
b=Path(__file__).resolve().parents[1];sys.path.insert(0,str(b/'evaluation_code'))
import evaluate_mutual as e
available=[c for c in e.PRIMARY if all((b/'evaluation'/c/s/'COMPLETE.json').exists() for s in e.SEQS)]
records,areas,meta,hashes,frames,objects=e.collect(e.REFERENCES+available)
results=[]
for condition in available:
 rows=[]
 for pi,plan in enumerate(e.PLANS):
  rr=[r for r in objects if r['arrangement']==plan]
  m=lambda r,c=condition:bool(r[c+'_matched'])
  row=dict(condition=condition,arrangement=plan,M=sum(m(r) for r in rr),low_recovered=sum(m(r) and r['low_all_05'] for r in rr),low_tracks=len({(r['sequence'],r['track_id']) for r in rr if m(r) and r['low_all_05']}),stable_matched=sum(m(r) and r['both_fixed_success'] for r in rr))
  for ref in ['release']+[c for c in ['E0','E1','E2'] if c in available]:
   row['G_vs_'+ref]=sum(m(r) and not m(r,ref) for r in rr);row['L_vs_'+ref]=sum(not m(r) and m(r,ref) for r in rr)
  row['U']=sum(int(records[condition,s]['frame_counts'][pi,:,3].sum()) for s in e.SEQS)
  row['A']=sum(int(records[condition,s]['frame_counts'][pi,:,5].sum()) for s in e.SEQS)
  row['unmatched_component_area']=sum(int(areas[condition,s][plan+'_areas'][areas[condition,s][plan+'_unmatched'].astype(bool)].sum()) for s in e.SEQS)
  rows.append(row)
 a={(r['sequence'],r['frame'],r['annotation_id']):r for r in objects if r['arrangement']=='O' and r['both_fixed_success']}
 c={(r['sequence'],r['frame'],r['annotation_id']):r for r in objects if r['arrangement']=='S' and r['both_fixed_success']}
 joint={'both':sum(m(a[k]) and m(c[k]) for k in a),'O_only':sum(m(a[k]) and not m(c[k]) for k in a),'S_only':sum(not m(a[k]) and m(c[k]) for k in a),'neither':sum(not m(a[k]) and not m(c[k]) for k in a)}
 results.append(dict(condition=condition,rows=rows,stable646=joint))
result=dict(status='READ_ONLY_PROGRESS',main_endpoint=512,seed=46,results=results,scope='Incomplete experiment progress; no main candidate judgment until formal tables/mechanism controls complete.')
(b/'evaluation_logs/MODEL_PROGRESS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
