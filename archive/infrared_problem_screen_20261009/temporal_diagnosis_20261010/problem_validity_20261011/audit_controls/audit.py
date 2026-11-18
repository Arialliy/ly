#!/usr/bin/env python3
"""Independent CPU-only audit; imports none of the root aggregation code."""
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parent
V=OUT.parent
Q=V.parent
P=Q/'stage1_goal'
SEQS=['DJI_0026_2','DJI_0067_2']
MODES=['official5_probmax','official5_probmean','symmetric5_probmax']
GRID=[1e-6,1e-5,1e-4,1e-3,.01,.03,.05,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.99,.999,.9999,.99999,.999999,1.]
HASH={}

def record(p):
    h=hashlib.sha256(p.read_bytes()).hexdigest();HASH[str(p)]=h;return h

def js(p):
    record(p);return json.loads(p.read_text())

def load(mode,seq):
    if mode==MODES[0]:
        folder=P/'evaluation/official'/seq;stem='official_fused'
    else:
        folder=V/'simple_controls'/mode/seq;stem='fused'
    meta=js(folder/(stem+'_COMPLETE.json'))
    npz=folder/(stem+'_matches.npz');cpath=folder/(stem+'_curves.csv')
    npz_hash=record(npz);curve_hash=record(cpath)
    if mode!=MODES[0]:
        assert npz_hash==meta['matches_sha256']
        assert curve_hash==meta['curves_sha256']
    with np.load(npz,allow_pickle=False) as z:
        data={k:z[k] for k in z.files}
    curves=list(csv.DictReader(cpath.open()))
    assert data['thresholds'].tolist()==GRID==meta['thresholds']
    assert data['matched'].shape==(23,len(meta['objects']))
    assert data['strict_small'].tolist()==[bool(x['strict_small']) for x in meta['objects']]
    assert data['frame_counts'].shape==(23,160,6)
    assert len({(x['frame'],x['annotation_id']) for x in meta['objects']})==len(meta['objects'])
    rows=[]
    for i,th in enumerate(GRID):
        cnt=data['frame_counts'][i].sum(0)
        assert cnt[1]==data['matched'][i].sum()
        assert cnt[2]-cnt[1]==cnt[3]
        row=dict(threshold=th,frames=160,gt=int(cnt[0]),matched=int(cnt[1]),
            strict_small_gt=int(data['strict_small'].sum()),
            strict_small_matched=int(data['matched'][i,data['strict_small']].sum()),
            components=int(cnt[2]),unmatched_components=int(cnt[3]),
            positive_pixels=int(cnt[4]),positive_pixels_outside_boxes=int(cnt[5]))
        for k,x in row.items():assert float(curves[i][k])==x,(mode,seq,th,k)
        for t in range(160):
            a,b=data['offsets'][t:t+2]
            assert data['matched'][i,a:b].sum()==data['frame_counts'][i,t,1]
        rows.append(row)
    data['meta']=meta;data['rows']=rows
    return data

DATA={(m,s):load(m,s) for m in MODES for s in SEQS}
for seq in SEQS:
    base=DATA[MODES[0],seq]
    for mode in MODES[1:]:
        d=DATA[mode,seq]
        assert d['meta']['objects']==base['meta']['objects']
        assert np.array_equal(d['strict_small'],base['strict_small'])
        assert d['meta']['windows']==5 and d['meta']['window_frame_budget']==200
        assert d['meta']['offline_access_range']==[0,159]
        assert d['meta']['gpu_inference_performed'] is False
        assert d['meta']['source_labels_used_for_window_selection'] is False

def row(mode, seqs, th):
    i=GRID.index(th)
    rows=[DATA[mode,s]['rows'][i] for s in seqs]
    r={k:sum(x[k] for x in rows) for k in rows[0] if k!='threshold'};r['threshold']=th
    return r

def choose(mode,seqs):
    budget=row(MODES[0],seqs,.5)['unmatched_components']
    ok=[row(mode,seqs,t) for t in GRID if row(mode,seqs,t)['unmatched_components']<=budget]
    assert ok
    return min(ok,key=lambda r:(-r['strict_small_matched'],r['unmatched_components'],-r['threshold']))

def compare(mode,seqs,th,reference=.5):
    r=row(mode,seqs,th);gain=loss=0
    for seq in seqs:
        d=DATA[mode,seq];b=DATA[MODES[0],seq]
        m=d['matched'][GRID.index(th)].astype(bool);f=b['matched'][GRID.index(reference)].astype(bool)
        mask=b['strict_small']
        gain+=int((m&~f&mask).sum());loss+=int((f&~m&mask).sum())
    assert gain-loss==r['strict_small_matched']-row(MODES[0],seqs,reference)['strict_small_matched']
    return dict(mode=mode,**r,gained_small=gain,lost_small=loss,net_small=gain-loss,
        U_per_frame=r['unmatched_components']/r['frames'])

result={'status':'PASS','choices':{},'fixed_half':[],'pooled_development':[],
        'transfer_0026_to_0067':[],'pooled_vs_rethresholded_official':[],
        'transfer_vs_rethresholded_official':[],'primary_half_opportunity_cohorts':[]}
for m in MODES:
    pc=choose(m,SEQS);cc=choose(m,[SEQS[0]])
    result['choices'][m]={'pooled':pc,'calibration':cc}
    for ss,group in [(SEQS,'pooled'),([SEQS[0]],SEQS[0]),([SEQS[1]],SEQS[1])]:
        result['fixed_half'].append(dict(group=group,**compare(m,ss,.5)))
    r=compare(m,SEQS,pc['threshold']);r['within_background_budget']=int(r['unmatched_components']<=row(MODES[0],SEQS,.5)['unmatched_components'])
    result['pooled_development'].append(r)
    r=compare(m,[SEQS[1]],cc['threshold']);r['within_background_budget']=int(r['unmatched_components']<=row(MODES[0],[SEQS[1]],.5)['unmatched_components'])
    r.update(calibration_sequence=SEQS[0],transfer_sequence=SEQS[1]);result['transfer_0026_to_0067'].append(r)
    ref=choose(MODES[0],SEQS)['threshold']
    result['pooled_vs_rethresholded_official'].append(dict(official_rethreshold=ref,**compare(m,SEQS,pc['threshold'],ref)))
    ref=choose(MODES[0],[SEQS[0]])['threshold']
    result['transfer_vs_rethresholded_official'].append(dict(official_rethreshold=ref,**compare(m,[SEQS[1]],cc['threshold'],ref)))

# Reconstruct opportunity cohorts directly from every original window's NPZ.
cohort_records=[]
for seq in SEQS:
    base=DATA[MODES[0],seq];objects=base['meta']['objects']
    index={(o['frame'],o['annotation_id']):i for i,o in enumerate(objects)}
    any_win=np.zeros(len(objects),bool);any_orig=np.zeros(len(objects),bool)
    high=np.zeros(len(objects),bool)
    for start in range(0,121,12):
        folder=V/'window_scores'/seq;stem=f'window_{start:03d}'
        meta=js(folder/(stem+'_COMPLETE.json'));path=folder/(stem+'_matches.npz')
        record(path)
        with np.load(path,allow_pickle=False) as z:
            ti=z['thresholds'].tolist().index(.5)
            for j,o in enumerate(meta['objects']):
                i=index[o['frame'],o['annotation_id']];assert o==objects[i]
                hit=bool(z['matched'][ti,j]);any_win[i]|=hit
                if start in [0,36,72,108,120]:any_orig[i]|=hit
                high[i]|=bool(z['box_max'][j]>=.5)
    fused=base['matched'][GRID.index(.5)].astype(bool)
    for i,o in enumerate(objects):
        if fused[i]:case='fused_and_window_match' if any_win[i] else 'fused_only_match'
        elif any_orig[i]:case='original_window_match_lost_by_fusion'
        elif any_win[i]:case='extra_window_only_opportunity'
        else:case='no_match_but_box_response' if high[i] else 'no_match_all_boxes_low'
        for m in MODES[1:]:
            hit=bool(DATA[m,seq]['matched'][GRID.index(.5),i])
            cohort_records.append(dict(mode=m,sequence=seq,frame=o['frame'],source_frame=o['source_frame'],
                annotation_id=o['annotation_id'],track_id=o['track_id'],small_geometry=int(base['strict_small'][i]),
                original_system_case=case,official_matched=int(fused[i]),control_matched=int(hit),
                gain=int(hit and not fused[i]),loss=int(fused[i] and not hit)))
for m in MODES[1:]:
    ss=[r for r in cohort_records if r['mode']==m and r['small_geometry']]
    for case in sorted({r['original_system_case'] for r in ss}):
        rr=[r for r in ss if r['original_system_case']==case]
        result['primary_half_opportunity_cohorts'].append(dict(mode=m,original_system_case=case,n=len(rr),
            control_matched=sum(r['control_matched'] for r in rr),gained=sum(r['gain'] for r in rr),lost=sum(r['loss'] for r in rr)))

# Compare with root artifacts only after independently computing all numbers.
root=js(V/'control_summary/SUMMARY.json')
for name in ['choices','fixed_half','pooled_development','transfer_0026_to_0067',
             'pooled_vs_rethresholded_official','transfer_vs_rethresholded_official','primary_half_opportunity_cohorts']:
    assert result[name]==root[name],name
root_csv=V/'control_summary/OBJECTS.csv';record(root_csv)
csvrows=list(csv.DictReader(root_csv.open()))
def objkey(r):return r['mode'],r['sequence'],int(r['frame']),int(r['annotation_id'])
rc={objkey(r):r for r in csvrows};own={objkey(r):r for r in cohort_records}
assert set(rc)==set(own) and len(rc)==len(csvrows)
for k,r in own.items():assert all(str(v)==rc[k][c] for c,v in r.items())
for filename,field in [('FIXED_HALF.csv','fixed_half'),('POOLED_DEVELOPMENT.csv','pooled_development'),
    ('TRANSFER.csv','transfer_0026_to_0067'),('POOLED_VS_RETHRESHOLD.csv','pooled_vs_rethresholded_official'),
    ('TRANSFER_VS_RETHRESHOLD.csv','transfer_vs_rethresholded_official'),('COHORTS.csv','primary_half_opportunity_cohorts')]:
    path=V/'control_summary'/filename;record(path)
    rr=list(csv.DictReader(path.open()));assert len(rr)==len(result[field])
    for actual,expected in zip(rr,result[field]):
        assert set(actual)==set(expected)
        assert all(str(expected[k])==actual[k] for k in expected),(filename,actual,expected)
path=V/'control_summary/ALL_CURVES.csv';record(path)
rr=list(csv.DictReader(path.open()));expected=[dict(mode=m,**row(m,SEQS,t)) for m in MODES for t in GRID]
assert len(rr)==len(expected)
for a,b in zip(rr,expected):assert all(str(v)==a[k] for k,v in b.items())
tr=defaultdict(list)
for r in cohort_records:
    if r['small_geometry']:tr[r['mode'],r['sequence'],r['track_id']].append(r)
path=V/'control_summary/TRACKS.csv';record(path)
rr=list(csv.DictReader(path.open()));assert len(rr)==len(tr)
for r in rr:
    group=tr[r['mode'],r['sequence'],int(r['track_id'])]
    assert int(r['n'])==len(group)
    assert int(r['gained'])==sum(x['gain'] for x in group)
    assert int(r['lost'])==sum(x['loss'] for x in group)
    assert int(r['net'])==int(r['gained'])-int(r['lost'])

# Verify frozen protocol, adapter-check records and scoring implementation hashes.
freeze=js(V/'PROTOCOL_FREEZE.json')
assert record(V/'PROTOCOL.md')==freeze['protocol_sha256']
scoring_hash=record(V/'simple_controls/score_controls.py')
launches=js(V/'simple_controls/LAUNCHES.json')
for launch in launches:
    assert datetime.fromisoformat(launch['launch_time_utc'])>datetime.fromisoformat(freeze['revised_utc'])
    assert launch['script_sha256']==scoring_hash and not launch['gpu_requested']
    assert launch['launch_exit_code']==0
for seq in SEQS:
    check=js(V/'simple_controls'/(seq+'_BASELINE_ADAPTER_CHECK.json'))
    assert check['status']=='PASS' and check['official_fused_exactly_reconstructed']
    assert check['old_matches_sha256']==HASH[str(P/'evaluation/official'/seq/'official_fused_matches.npz')]
    for mode in MODES[1:]:
        meta=DATA[mode,seq]['meta'];assert meta['scoring_script_sha256']==scoring_hash
        assert meta['baseline_adapter_check']=='PASS' and meta['full_frame_fusion_formula_verified']
        assert meta['checkpoint_sha256']=='4fd4aa2736d9a33f2bf1459099068d8b4a7e202ff80aecdb4a3277fd13c7abcc'
result['ratio_audit']={'absolute_detection_fraction_gain':20/5200,
    'percentage_point_gain':100*20/5200,'relative_gain_among_baseline_matches':20/700,
    'gross_opportunity':109,'original_input_opportunity':5,'additional_window_opportunity':104}
result['audit_scope']='Independent NPZ/CSV aggregation, per-object matches, cohort reconstruction and operating-point selection. Adapter equality is verified from hash-bound saved records and code, not rerun full pixel-level matching.'
result['inputs_sha256']=HASH
for p,h in HASH.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
(OUT/'SUMMARY.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['status','choices','ratio_audit']},ensure_ascii=False,indent=2))
