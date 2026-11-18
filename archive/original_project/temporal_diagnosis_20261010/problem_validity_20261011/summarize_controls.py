#!/usr/bin/env python3
"""Fixed controls, shared background budgets, and descriptive transfer check."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

V=Path(__file__).resolve().parent
P=V.parent/'stage1_goal'
SEQS=['DJI_0026_2','DJI_0067_2']
MODES=['official5_probmax','official5_probmean','symmetric5_probmax']
FIELDS=['frames','gt','matched','strict_small_gt','strict_small_matched','components',
        'unmatched_components','positive_pixels','positive_pixels_outside_boxes']


def load(mode,seq):
    d=P/'evaluation/official'/seq if mode==MODES[0] else V/'simple_controls'/mode/seq
    prefix='official_fused' if mode==MODES[0] else 'fused'
    meta=json.loads((d/(prefix+'_COMPLETE.json')).read_text())
    with np.load(d/(prefix+'_matches.npz'),allow_pickle=False) as z:
        a={k:z[k] for k in z.files}
    curves=[dict(threshold=float(r['threshold']),**{k:int(r[k]) for k in FIELDS})
            for r in csv.DictReader((d/(prefix+'_curves.csv')).open())]
    assert a['thresholds'].tolist()==[r['threshold'] for r in curves]==meta['thresholds']
    assert np.array_equal(a['strict_small'],[bool(o['strict_small']) for o in meta['objects']])
    for i,r in enumerate(curves):
        assert r['strict_small_matched']==int(a['matched'][i,a['strict_small']].sum())
        assert r['unmatched_components']==int(a['frame_counts'][i,:,3].sum())
    return dict(arrays=a,curves=curves,objects=meta['objects'])


def aggregate(data,seqs):
    return [dict(threshold=t,**{k:sum(data[s]['curves'][i][k] for s in seqs) for k in FIELDS})
            for i,t in enumerate(data[seqs[0]]['arrays']['thresholds'].tolist())]


def choose(curves,budget):
    feasible=[r for r in curves if r['unmatched_components']*budget['frames']<=budget['unmatched_components']*r['frames']]
    return min(feasible,key=lambda r:(-r['strict_small_matched'],r['unmatched_components'],-r['threshold'])) if feasible else None


def metrics(data,seqs,t,baseline,base_threshold):
    out={k:0 for k in FIELDS};gain=loss=0
    for s in seqs:
        d=data[s];b=baseline[s];assert d['objects']==b['objects']
        ti=d['arrays']['thresholds'].tolist().index(t);bi=b['arrays']['thresholds'].tolist().index(base_threshold)
        for k in FIELDS:out[k]+=d['curves'][ti][k]
        eligible=d['arrays']['strict_small']
        mm=d['arrays']['matched'][ti].astype(bool);bm=b['arrays']['matched'][bi].astype(bool)
        gain+=int((eligible&~bm&mm).sum());loss+=int((eligible&bm&~mm).sum())
    out.update(threshold=t,gained_small=gain,lost_small=loss,net_small=gain-loss,
               U_per_frame=out['unmatched_components']/out['frames'])
    return out


def write_csv(path,rows):
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    out=V/'control_summary';out.mkdir(exist_ok=True)
    if (out/'SUMMARY.json').exists():raise FileExistsError('Completed control summary exists')
    all_data={m:{s:load(m,s) for s in SEQS} for m in MODES}
    base=all_data[MODES[0]]
    budgets={name:next(r for r in aggregate(base,seqs) if r['threshold']==.5)
             for name,seqs in [('pooled',SEQS),('calibration',[SEQS[0]]),('transfer',[SEQS[1]])]}
    assert budgets['pooled']['unmatched_components']==4015
    fixed=[];pooled=[];transfer=[];choices={};all_curves=[]
    for m,data in all_data.items():
        for group,seqs in [('pooled',SEQS),(SEQS[0],[SEQS[0]]),(SEQS[1],[SEQS[1]])]:
            fixed.append(dict(mode=m,group=group,**metrics(data,seqs,.5,base,.5)))
        curves=aggregate(data,SEQS)
        all_curves.extend(dict(mode=m,**r) for r in curves)
        selected=choose(curves,budgets['pooled'])
        cal=choose(aggregate(data,[SEQS[0]]),budgets['calibration'])
        choices[m]=dict(pooled=selected,calibration=cal)
        if selected:
            row=metrics(data,SEQS,selected['threshold'],base,.5)
            row['within_background_budget']=int(row['unmatched_components']<=budgets['pooled']['unmatched_components'])
            pooled.append(dict(mode=m,**row))
        if cal:
            row=metrics(data,[SEQS[1]],cal['threshold'],base,.5)
            row['within_background_budget']=int(row['unmatched_components']<=budgets['transfer']['unmatched_components'])
            transfer.append(dict(mode=m,calibration_sequence=SEQS[0],transfer_sequence=SEQS[1],**row))
    # Compare against the threshold-only official control as well as the fixed-.5 deployment reference.
    pooled_ret=[];transfer_ret=[]
    for m in MODES:
        if choices[m]['pooled'] and choices[MODES[0]]['pooled']:
            pooled_ret.append(dict(mode=m,official_rethreshold=choices[MODES[0]]['pooled']['threshold'],
                **metrics(all_data[m],SEQS,choices[m]['pooled']['threshold'],base,choices[MODES[0]]['pooled']['threshold'])))
        if choices[m]['calibration'] and choices[MODES[0]]['calibration']:
            transfer_ret.append(dict(mode=m,official_rethreshold=choices[MODES[0]]['calibration']['threshold'],
                **metrics(all_data[m],[SEQS[1]],choices[m]['calibration']['threshold'],base,choices[MODES[0]]['calibration']['threshold'])))
    # Exact per-object transitions at .5, with opportunity cohort inherited from the official-only join.
    lookup={(r['sequence'],int(r['frame']),int(r['annotation_id'])):r
            for r in csv.DictReader((V/'OBJECTS.csv').open()) if r['sequence'] in SEQS}
    objects=[]
    for m,data in all_data.items():
        if m==MODES[0]:continue
        for s in SEQS:
            d=data[s];b=base[s];ti=d['arrays']['thresholds'].tolist().index(.5)
            for i,o in enumerate(d['objects']):
                rr=lookup[(s,o['frame'],o['annotation_id'])]
                old=int(b['arrays']['matched'][ti,i]);new=int(d['arrays']['matched'][ti,i])
                objects.append(dict(mode=m,sequence=s,frame=o['frame'],source_frame=o['source_frame'],
                    annotation_id=o['annotation_id'],track_id=o['track_id'],small_geometry=int(o['strict_small']),
                    original_system_case=rr['system_case'],official_matched=old,control_matched=new,
                    gain=int(not old and new),loss=int(old and not new)))
    pertrack=[];cohorts=[]
    for m in MODES[1:]:
        rr=[r for r in objects if r['mode']==m and r['small_geometry']]
        for case in sorted({r['original_system_case'] for r in rr}):
            ss=[r for r in rr if r['original_system_case']==case]
            cohorts.append(dict(mode=m,original_system_case=case,n=len(ss),
                control_matched=sum(r['control_matched'] for r in ss),gained=sum(r['gain'] for r in ss),lost=sum(r['loss'] for r in ss)))
        for s in SEQS:
            for t in sorted({r['track_id'] for r in rr if r['sequence']==s}):
                ss=[r for r in rr if r['sequence']==s and r['track_id']==t]
                pertrack.append(dict(mode=m,sequence=s,track_id=t,n=len(ss),
                    gained=sum(r['gain'] for r in ss),lost=sum(r['loss'] for r in ss),net=sum(r['gain']-r['loss'] for r in ss)))
    result=dict(status='COMPLETE',budgets=budgets,choices=choices,fixed_half=fixed,
        pooled_development=pooled,transfer_0026_to_0067=transfer,
        pooled_vs_rethresholded_official=pooled_ret,transfer_vs_rethresholded_official=transfer_ret,
        primary_half_opportunity_cohorts=cohorts,
        interpretation='Two previously used development clips; threshold transfer is not new independent confirmation; no model selection or GT-dependent per-object window choice',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    for name,rr in [('FIXED_HALF',fixed),('POOLED_DEVELOPMENT',pooled),('TRANSFER',transfer),
                    ('POOLED_VS_RETHRESHOLD',pooled_ret),('TRANSFER_VS_RETHRESHOLD',transfer_ret),
                    ('ALL_CURVES',all_curves),('OBJECTS',objects),('TRACKS',pertrack),('COHORTS',cohorts)]:
        write_csv(out/(name+'.csv'),rr)
    (out/'SUMMARY.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(fixed_half=fixed,pooled_development=pooled,transfer=transfer,cohorts=cohorts),ensure_ascii=False))


if __name__=='__main__':main()
