#!/usr/bin/env python3
"""Read the exhaustive frozen result tables; no inference or detection scoring."""
import csv, hashlib, itertools, json
from pathlib import Path
D=Path(__file__).resolve().parent; C=D.parent/'combination_feasibility_20261011'
NUM=['small_matched','gained','lost','unmatched_components','positive_pixels_outside_boxes','positive_pixels','component_count','matched_count']
KEYS=['choice_0026','choice_0067']+NUM
def read(p):
    rows=list(csv.DictReader(p.open()))
    for r in rows:
        for k in NUM:r[k]=int(r[k])
    return rows
def trim(r):return {k:r[k] for k in KEYS}
def dump(name,rows):
    assert rows,name
    with (D/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def rank(r):return (-r['small_matched'],r['lost'],r['unmatched_components'],r['positive_pixels_outside_boxes'],r['choice_0026'],r['choice_0067'])
def run():
    allr={s:read(C/f) for s,f in [('shared','SHARED_COMBINATIONS.csv'),('per_clip','PER_CLIP_COMBINATIONS.csv')]}
    rr={s:[r for r in rows if r['unmatched_components']<=4015] for s,rows in allr.items()}
    assert len(allr['shared'])==10 and len(allr['per_clip'])==100
    primary=[];summary={};floorrows=[];lossrows=[];ties=[]
    minm=min(r['small_matched'] for rows in rr.values() for r in rows)
    maxm=max(r['small_matched'] for rows in rr.values() for r in rows)
    maxl=max(r['lost'] for rows in rr.values() for r in rows)
    for scope,rows in rr.items():
        high=[r for r in rows if r['small_matched']>=720]
        zero=[r for r in rows if r['lost']==0]
        primary.extend(dict(scope=scope,query='M>=720',**trim(r)) for r in high)
        primary.extend(dict(scope=scope,query='lost=0',**trim(r)) for r in zero)
        summary[scope]=dict(total=len(allr[scope]),budget_feasible=len(rows),
            high_min_loss=min(r['lost'] for r in high),high_min_loss_choice=trim(min(high,key=lambda r:(r['lost'],*rank(r)))),
            zero_loss_max_M=max(r['small_matched'] for r in zero),zero_loss_choice=trim(min(zero,key=rank)))
        for floor in range(minm,maxm+1):
            eligible=[r for r in rows if r['small_matched']>=floor]
            if not eligible:
                floorrows.append(dict(scope=scope,M_floor=floor,feasible_count=0,min_lost='',**{k:'' for k in KEYS}));continue
            ll=min(r['lost'] for r in eligible); best=[r for r in eligible if r['lost']==ll];selected=min(best,key=rank)
            floorrows.append(dict(scope=scope,M_floor=floor,feasible_count=len(eligible),min_lost=ll,**trim(selected)))
            ties.extend(dict(scope=scope,query='M_floor',value=floor,**trim(r)) for r in best)
        for limit in range(maxl+1):
            eligible=[r for r in rows if r['lost']<=limit]; mm=max(r['small_matched'] for r in eligible)
            best=[r for r in eligible if r['small_matched']==mm];selected=min(best,key=rank)
            lossrows.append(dict(scope=scope,loss_limit=limit,feasible_count=len(eligible),max_M=mm,**trim(selected)))
            ties.extend(dict(scope=scope,query='loss_limit',value=limit,**trim(r)) for r in best)
    dump('PRIMARY_QUERIES.csv',primary);dump('MATCH_FLOOR_MIN_LOSS.csv',floorrows)
    dump('LOSS_LIMIT_MAX_MATCH.csv',lossrows);dump('ALL_QUERY_TIES.csv',ties)

    comparisons=[]; optima=[]
    for base in rr['shared']:
        for kind,test in [('M_L_no_worse',lambda r:r['small_matched']>=base['small_matched'] and r['lost']<=base['lost']),
                          ('M_L_exact',lambda r:r['small_matched']==base['small_matched'] and r['lost']==base['lost']),
                          ('M_L_within_1',lambda r:abs(r['small_matched']-base['small_matched'])<=1 and abs(r['lost']-base['lost'])<=1)]:
            candidates=[r for r in rr['per_clip'] if test(r)]
            for r in candidates:
                comparisons.append(dict(reference=base['choice_0026'],comparison=kind,**trim(r),
                    **{'delta_'+k:r[k]-base[k] for k in NUM}))
            for metric in ['unmatched_components','positive_pixels_outside_boxes','positive_pixels','component_count']:
                chosen=min(candidates,key=lambda r:(r[metric],*rank(r)))
                optima.append(dict(reference=base['choice_0026'],comparison=kind,minimize=metric,
                    feasible_count=len(candidates),**trim(chosen),**{'delta_'+k:chosen[k]-base[k] for k in NUM}))
    dump('BACKGROUND_COMPARISONS.csv',comparisons);dump('BACKGROUND_OBJECTIVE_MINIMA.csv',optima)

    # Four-dimensional front, preserving opposing background changes.
    fronts=[]
    def dominates(a,b):
        a=(a['small_matched'],-a['lost'],-a['unmatched_components'],-a['positive_pixels_outside_boxes'])
        b=(b['small_matched'],-b['lost'],-b['unmatched_components'],-b['positive_pixels_outside_boxes'])
        return all(x>=y for x,y in zip(a,b)) and any(x>y for x,y in zip(a,b))
    for scope,rows in rr.items():
        fronts.extend(dict(scope=scope,**trim(r)) for r in rows if not any(dominates(a,r) for a in rows))
    dump('PARETO_MATCH_LOSS_U_OUTSIDE.csv',fronts)
    # Global exact M/L groups and all near pairs, not only selected references.
    groups={}
    for r in rr['per_clip']:groups.setdefault((r['small_matched'],r['lost']),[]).append(r)
    exact=[]
    for (m,l),rows in sorted(groups.items()):
        if len(rows)<2:continue
        exact.append(dict(M=m,lost=l,count=len(rows),choices=json.dumps([[r[k] for k in ['choice_0026','choice_0067']] for r in rows]),
            **{f'{k}_{suffix}':fun(r[k] for r in rows) for k in NUM[3:] for suffix,fun in [('min',min),('max',max)]}))
    dump('EXACT_M_L_BACKGROUND_RANGES.csv',exact)
    near=[]
    for a,b in itertools.combinations(rr['per_clip'],2):
        if abs(a['small_matched']-b['small_matched'])<=1 and abs(a['lost']-b['lost'])<=1:
            near.append(dict(a=a['choice_0026']+'/'+a['choice_0067'],b=b['choice_0026']+'/'+b['choice_0067'],
                **{f'{prefix}_{k}':r[k] for prefix,r in [('a',a),('b',b)] for k in NUM},
                **{'delta_b_minus_a_'+k:b[k]-a[k] for k in NUM}))
    dump('ALL_NEAR_M_L_PAIRS.csv',near)
    # Compress frontiers solely by identical returned metric/selection tuple.
    def compress(rows,col):
        groups=[]
        for r in rows:
            sig=tuple(r.get(k) for k in ['scope','choice_0026','choice_0067','small_matched','lost'])
            if groups and groups[-1]['signature']==sig:groups[-1]['to']=r[col]
            else:groups.append(dict(signature=sig,from_=r[col],to=r[col],choice={k:r[k] for k in KEYS}))
        return groups
    summary.update(dict(U_budget=4015,threshold=.5,match_floor_intervals=compress(floorrows,'M_floor'),
        loss_limit_intervals=compress(lossrows,'loss_limit'),exact_M_L_groups=len(exact),near_pairs=len(near),
        max_exact_M_L_U_span=max(r['unmatched_components_max']-r['unmatched_components_min'] for r in exact),
        max_exact_M_L_outside_span=max(r['positive_pixels_outside_boxes_max']-r['positive_pixels_outside_boxes_min'] for r in exact),
        max_near_U_absolute_difference=max(abs(r['delta_b_minus_a_unmatched_components']) for r in near),
        max_near_outside_absolute_difference=max(abs(r['delta_b_minus_a_positive_pixels_outside_boxes']) for r in near),
        sources={str(C/f):hashlib.sha256((C/f).read_bytes()).hexdigest() for f in ['SHARED_COMBINATIONS.csv','PER_CLIP_COMBINATIONS.csv']},
        limits=['Post hoc developer clips only; not deployable policy or independent confirmation.',
            'Exact and within-one comparisons describe existing points, not a new success threshold.',
            'No new model inference, fusion, threshold scoring, or training.']))
    (D/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':run()
