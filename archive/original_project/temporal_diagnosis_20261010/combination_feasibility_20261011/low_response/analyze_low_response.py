#!/usr/bin/env python3
"""Reuse frozen threshold evaluations only; do not run inference or select a new threshold."""
import csv, hashlib, itertools, json, time, zipfile
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parent
C=OUT.parent; Q=C.parent; P=Q/'stage1_goal'; V=Q/'problem_validity_20261011'
SEQS=['DJI_0026_2','DJI_0067_2']
FIELDS=['gt_count','matched_count','component_count','unmatched_components','positive_pixels','positive_pixels_outside_boxes']
EXPECTED_THRESHOLDS=[1e-6,1e-5,1e-4,1e-3,.01,.03,.05,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.99,.999,.9999,.99999,.999999,1.]
SOURCES={}
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    SOURCES[str(p)]=h.hexdigest(); return h.hexdigest()
def readjson(p):digest(p);return json.loads(p.read_text())
def readcsv(p):digest(p);return list(csv.DictReader(p.open()))
def writecsv(name,rows):
    with (OUT/name).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def writejson(name,value):
    with (OUT/name).open('x') as f:json.dump(value,f,indent=2);f.write('\n')

def main():
    began=time.perf_counter(); digest(Path(__file__));digest(C/'PROTOCOL.md');digest(C/'PROTOCOL_FREEZE.json')
    frozen=readjson(C/'LEGAL_COMBINATIONS.json')
    allfive=list(itertools.combinations(range(0,121,12),5))
    legal=[]
    for starts in allfive:
        covered=set().union(*(set(range(s,s+40)) for s in starts))
        if covered==set(range(160)):legal.append(list(starts))
    assert len(allfive)==462 and len(legal)==10 and legal==frozen['legal_full_coverage_5']
    audit=dict(all_five_subsets=len(allfive),legal_five_combinations=len(legal),per_clip_pairs=len(legal)**2,
               independent_criterion='Explicit union of integer frames equals set(range(160)).',
               legal_matches_frozen=True,legal_combinations=legal,window_frames_per_clip=200,
               all_combinations_distinct=True,pair_count=100)
    objects=readcsv(V/'OBJECTS.csv')
    lookup={(r['sequence'],int(r['frame']),int(r['annotation_id'])):r for r in objects if r['sequence'] in SEQS}
    data={};cohort_rows=[];inventory=[]
    for seq in SEQS:
        complete=readjson(P/'evaluation/official'/seq/'official_fused_COMPLETE.json')
        meta=complete['objects'];small=np.array([bool(m['strict_small']) for m in meta]);low=[]
        for m in meta:
            r=lookup[(seq,m['frame'],m['annotation_id'])]
            assert int(r['small_geometry'])==int(m['strict_small'])
            islow=bool(m['strict_small']) and float(r['context_box_max'])<.5
            low.append(islow)
            if islow:
                assert not int(r['any11_matched'])
                cohort_rows.append(dict(sequence=seq,frame=m['frame'],source_frame=m['source_frame'],annotation_id=m['annotation_id'],
                    track_id=m['track_id'],category_id=m['category_id'],context_box_max=float(r['context_box_max']),
                    official_box_max=float(r['fused_box_max'])))
        low=np.array(low); dd={'small':small,'low':low,'meta':meta}
        for mode in ['O','S']:
            path=P/'evaluation/official'/seq/'official_fused_matches.npz' if mode=='O' else V/'simple_controls/symmetric5_probmax'/seq/'fused_matches.npz'
            digest(path)
            with np.load(path,allow_pickle=False) as z:dd[mode]={k:z[k].copy() for k in z.files}
            z=dd[mode]
            assert z['thresholds'].tolist()==EXPECTED_THRESHOLDS
            assert np.array_equal(z['strict_small'],small)
            assert z['matched'].shape==(23,len(meta))
            assert np.array_equal(z['offsets'],dd['O']['offsets'])
            assert (z['box_max'][low]<.5).all()
            assert not z['matched'][11,low].any()
        data[seq]=dd
        for path in sorted((P/'predictions/official'/seq).rglob('*.npz'))+[V/'simple_controls/symmetric5_probmax'/seq/'fused.npz']:
            with zipfile.ZipFile(path) as z:names=z.namelist()
            inventory.append(dict(path=str(path),array_names=names,has_logit_named_array=any('logit' in n.lower() for n in names)))
    assert sum(d['small'].sum() for d in data.values())==5200
    assert len(cohort_rows)==4342
    assert sum(d['O']['matched'][11,d['small']].sum() for d in data.values())==700
    rows=[];recoveries=[]
    for group,seqs in [('pooled',SEQS)]+[(s,[s]) for s in SEQS]:
        for mode in ['O','S']:
            for ti,threshold in enumerate(EXPECTED_THRESHOLDS):
                counts=sum((data[s][mode]['frame_counts'][ti].sum(0) for s in seqs),np.zeros(6,np.int64))
                smalln=lown=matched=recovered=gain=loss=0
                for s in seqs:
                    d=data[s];mask=d['small']; low=d['low'];m=d[mode]['matched'][ti].astype(bool);baseline=d['O']['matched'][11].astype(bool)
                    smalln+=int(mask.sum());lown+=int(low.sum());matched+=int(m[mask].sum());recovered+=int(m[low].sum())
                    gain+=int((mask&m&~baseline).sum());loss+=int((mask&~m&baseline).sum())
                    if group=='pooled':
                        for idx in np.flatnonzero(low&m):
                            mm=d['meta'][idx]
                            recoveries.append(dict(mode=mode,threshold=threshold,sequence=s,frame=mm['frame'],source_frame=mm['source_frame'],
                                annotation_id=mm['annotation_id'],track_id=mm['track_id'],box_max=float(d[mode]['box_max'][idx]),
                                peak_x=float(d[mode]['peak_x'][ti,idx]),peak_y=float(d[mode]['peak_y'][ti,idx]),component_area=int(d[mode]['component_area'][ti,idx])))
                budget=4015 if group=='pooled' else {'DJI_0026_2':741,'DJI_0067_2':3274}[group]
                rows.append(dict(group=group,mode=mode,threshold=threshold,small_gt=smalln,small_matched=matched,
                    low_cohort_gt=lown,low_cohort_recovered=recovered,gained_vs_original05=gain,lost_vs_original05=loss,net_vs_original05=gain-loss,
                    **{k:int(v) for k,v in zip(FIELDS,counts)},U_budget_reference=budget,U_within_budget=int(counts[3]<=budget),
                    budget_role='frozen_pooled_primary' if group=='pooled' else 'per_sequence_secondary_flag'))
    writecsv('THRESHOLD_CURVES.csv',rows);writecsv('LOW_COHORT.csv',cohort_rows);writecsv('LOW_RECOVERY_OBJECTS.csv',recoveries)
    digest(P/'pilot/infer_pilot.py')
    pooled=[r for r in rows if r['group']=='pooled'];feasible=[r for r in pooled if r['U_within_budget']]
    selections={}
    for mode in ['O','S']:
        rr=[r for r in pooled if r['mode']==mode];ff=[r for r in rr if r['U_within_budget']]
        selections[mode]=dict(max_total_matches_with_budget=max(r['small_matched'] for r in ff),
            max_low_recovered_with_budget=max(r['low_cohort_recovered'] for r in ff),
            best_total_rows=[r for r in ff if r['small_matched']==max(x['small_matched'] for x in ff)],
            best_low_rows=[r for r in ff if r['low_cohort_recovered']==max(x['low_cohort_recovered'] for x in ff)],
            max_low_without_budget=max(r['low_cohort_recovered'] for r in rr),
            best_unbudgeted_low_rows=[r for r in rr if r['low_cohort_recovered']==max(x['low_cohort_recovered'] for x in rr)],
            low_recovery_thresholds_with_budget=[r['threshold'] for r in ff if r['low_cohort_recovered']>0])
    summary=dict(status='COMPLETE',sequences=SEQS,small_gt=5200,low_cohort_gt=4342,original05_matched=700,
        thresholds=EXPECTED_THRESHOLDS,pooled_U_budget=4015,selections=selections,
        raw_cache_inventory=inventory,original_raw_logits_cached=any(r['has_logit_named_array'] for r in inventory),
        scope='Existing 23 threshold points on two fixed five-window arrangements; descriptive, no threshold deployment or general separability proof.',
        diagnostic_not_paper_Pd_Fa=True,geometry_audit=audit,
        assertions=dict(small_identity_matches_existing_table=True,threshold_lists_equal_frozen=True,
            low_cohort_matches_4342=True,low_cohort_absent_at05_in_both=True,original05_matches700=True,
            curve_rows_138=len(rows)==138,all_probability_cache_keys_inspected=True),
        source_sha256=SOURCES,elapsed_seconds=time.perf_counter()-began)
    writejson('GEOMETRY_AUDIT.json',audit);writejson('SUMMARY.json',summary)
    print(json.dumps({'selections':selections,'seconds':summary['elapsed_seconds']},indent=2))

if __name__=='__main__':main()
