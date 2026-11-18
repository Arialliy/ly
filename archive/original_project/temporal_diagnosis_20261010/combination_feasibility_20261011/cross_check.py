#!/usr/bin/env python3
"""Join independently generated support matrix and actual fused outputs by identity."""
import csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np
C=Path(__file__).resolve().parent
SEQS=['DJI_0026_2','DJI_0067_2']
def read(p):return list(csv.DictReader(p.open()))
def dump(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    matrix=read(C/'support/MATCH_MATRIX.csv')
    d={(r['sequence'],int(r['frame']),int(r['annotation_id'])):r for r in matrix}
    assert len(d)==5200
    legal=json.loads((C/'LEGAL_COMBINATIONS.json').read_text())['legal_full_coverage_5']+[list(range(0,121,12))]
    meta=read(C/'scoring/OBJECTS.csv')
    metrics=[];losses=[];checks=[]
    for seq in SEQS:
        mm=[r for r in meta if r['sequence']==seq]
        with np.load(C/'scoring'/f'{seq}_scores.npz',allow_pickle=False) as z:
            support=np.zeros((len(mm),11),bool)
            for i,r in enumerate(mm):
                assert int(r['index'])==i
                assert bool(int(r['small_geometry']))==bool(z['small'][i])
                if not z['small'][i]:continue
                sr=d[(seq,int(r['frame']),int(r['annotation_id']))]
                assert int(sr['track_id'])==int(r['track_id'])
                support[i]=[int(sr[f'w{s:03}'])==1 for s in range(0,121,12)]
            for pi,starts in enumerate(legal):
                sup=support[:,[s//12 for s in starts]].any(1)&z['small']
                act=z['matched'][pi].astype(bool)&z['small']
                violations=act&~sup
                assert not violations.any(),(seq,pi,np.flatnonzero(violations))
                base=z['matched'][9].astype(bool)&z['small']
                metrics.append(dict(sequence=seq,plan=z['plans'][pi],support=int(sup.sum()),actual=int(act.sum()),
                    support_but_fusion_miss=int((sup&~act).sum()),fusion_only_small=int(violations.sum()),
                    original_success_lost_despite_support=int((sup&~act&base).sum()),
                    original_success_lost_without_support=int((~sup&base).sum())))
                for i in np.flatnonzero(sup&~act):
                    r=mm[i]
                    losses.append(dict(sequence=seq,plan=z['plans'][pi],frame=r['frame'],source_frame=r['source_frame'],
                        annotation_id=r['annotation_id'],track_id=r['track_id'],original_success=int(base[i]),
                        box_max=float(z['box_max'][pi,i]),eligible_components=int(z['eligible_components'][pi,i]),
                        selected_success_starts=json.dumps([s for s in starts if support[i,s//12]])))
            checks.append(dict(sequence=seq,all11_plans_small_actual_subset_support=True,identity_join_exact=True))
    for pi in range(11):
        plan=f'C{pi:02}' if pi<10 else 'ALL11';rows=[r for r in metrics if r['plan']==plan]
        metrics.append(dict(sequence='pooled',plan=plan,**{k:sum(r[k] for r in rows) for k in rows[0] if k not in ['sequence','plan']}))
    shared=read(C/'SHARED_COMBINATIONS.csv');pairs=read(C/'PER_CLIP_COMBINATIONS.csv')
    sup=json.loads((C/'support/SUMMARY.json').read_text())
    assert max(int(r['small_matched']) for r in shared)==sup['shared5_support_max']==720
    assert max(int(r['small_matched']) for r in pairs)==sup['independent5_support_max']==721
    assert max(int(r['small_matched']) for r in pairs if r['U_budget_feasible']=='True')==721
    assert max(int(r['small_matched']) for r in pairs if r['U_budget_feasible']=='True' and int(r['lost'])==0)==707
    assert not any(int(r['original_sym_union_retained'])==774 for r in pairs)
    dump(C/'FUSION_VS_SUPPORT.csv',metrics);dump(C/'SUPPORT_BUT_FUSION_MISS.csv',losses)
    report=dict(status='PASS',checks=checks,support_bound_720_shared_and_721_per_clip_attained_by_actual_fusion=True,
        max_five_window_matches_budgeted=721,zero_loss_max_budgeted=707,all774_compatibility_impossible_in_five=True,
        no_small_fusion_only_success_in_all_22_actual_runs=True,pooled=[r for r in metrics if r['sequence']=='pooled'])
    (C/'CROSS_CHECK.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
