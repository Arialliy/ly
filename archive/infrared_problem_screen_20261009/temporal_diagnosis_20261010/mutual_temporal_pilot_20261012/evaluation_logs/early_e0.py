import sys,json
from pathlib import Path
b=Path(__file__).resolve().parents[1];sys.path.insert(0,str(b/'evaluation_code'))
import evaluate_mutual as e
records,areas,meta,hashes,frames,objects=e.collect(e.REFERENCES+['E0'])
result=[]
for pi,plan in enumerate(e.PLANS):
 rr=[r for r in objects if r['arrangement']==plan]
 m=lambda r,c='E0':bool(r[c+'_matched'])
 row=dict(condition='E0',arrangement=plan,M=sum(m(r) for r in rr),G_vs_release=sum(m(r) and not m(r,'release') for r in rr),L_vs_release=sum(not m(r) and m(r,'release') for r in rr),low_recovered=sum(m(r) and r['low_all_05'] for r in rr),low_tracks=len({(r['sequence'],r['track_id']) for r in rr if m(r) and r['low_all_05']}),stable_matched=sum(m(r) and r['both_fixed_success'] for r in rr))
 row['U']=sum(int(records['E0',s]['frame_counts'][pi,:,3].sum()) for s in e.SEQS)
 row['A']=sum(int(records['E0',s]['frame_counts'][pi,:,5].sum()) for s in e.SEQS)
 row['unmatched_component_area']=sum(int(areas['E0',s][plan+'_areas'][areas['E0',s][plan+'_unmatched'].astype(bool)].sum()) for s in e.SEQS)
 result.append(row)
a={r['sequence'],} if False else {(r['sequence'],r['frame'],r['annotation_id']):r for r in objects if r['arrangement']=='O' and r['both_fixed_success']}
c={(r['sequence'],r['frame'],r['annotation_id']):r for r in objects if r['arrangement']=='S' and r['both_fixed_success']}
joint={'both':sum(m(a[k]) and m(c[k]) for k in a),'O_only':sum(m(a[k]) and not m(c[k]) for k in a),'S_only':sum(not m(a[k]) and m(c[k]) for k in a),'neither':sum(not m(a[k]) and not m(c[k]) for k in a)}
result=dict(status='READ_ONLY_PROGRESS',main_endpoint=512,seed=46,rows=result,stable646=joint,scope='Incomplete experiment progress; no main candidate judgment from E0 alone.')
(b/'evaluation_logs/E0_PROGRESS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
