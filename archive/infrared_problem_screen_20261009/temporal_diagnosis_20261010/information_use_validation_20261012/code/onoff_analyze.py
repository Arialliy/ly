#!/usr/bin/env python3
"""Analyze persisted true on/off logits and direct-residual region energies."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
Q=ROOT.parent
OLD=Q/'mutual_temporal_pilot_20261012'
PLANS={'O':[0,36,72,108,120],'S':[0,24,60,96,120]}
SEQUENCES=['DJI_0026_2','DJI_0067_2']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):
    with path.open() as stream:return list(csv.DictReader(stream))


def csvout(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def pixel_stats(delta):
    flat=delta.reshape(-1)
    q=np.quantile(flat,[.01,.05,.5,.95,.99]);absolute=np.abs(flat);a=np.quantile(absolute,[.5,.9,.99])
    return {'pixels':int(flat.size),'delta_mean':float(flat.mean(dtype=np.float64)),
        'delta_min':float(flat.min()),'delta_max':float(flat.max()),
        'delta_q01':float(q[0]),'delta_q05':float(q[1]),'delta_q50':float(q[2]),'delta_q95':float(q[3]),'delta_q99':float(q[4]),
        'absolute_mean':float(absolute.mean(dtype=np.float64)),'absolute_q50':float(a[0]),'absolute_q90':float(a[1]),'absolute_q99':float(a[2]),
        'exact_nonzero_pixels':int(np.count_nonzero(flat)),
        'positive_delta_pixels':int(np.count_nonzero(flat>0)),'negative_delta_pixels':int(np.count_nonzero(flat<0))}


def main():
    regions=read(ROOT/'A_REGIONS.csv');details=[];pixel_rows=[];sources={};norms=defaultdict(list);checks={}
    for sequence in SEQUENCES:
        folder=ROOT/'onoff'/sequence
        receipt=json.loads((folder/'COMPLETE.json').read_text())
        sources[str(folder/'COMPLETE.json')]=sha(folder/'COMPLETE.json')
        checks[sequence+':state_unchanged']=receipt['model_state_sha256']==receipt['unchanged_final_model_state_sha256']
        checks[sequence+':16_true_onoff_forwards_per_sequence']=len(receipt['on_cache_checks'])==8 and len(receipt['window_timing'])==8
        checks[sequence+':old_on_all_exact']=all(x['old_on_probabilities_bitwise_identical'] for x in receipt['on_cache_checks'])
        checks[sequence+':direct_residual_nonzero']=all(r['correction_squared_norm']>0 for r in receipt['window_timing'])
        for r in read(folder/'RESIDUAL_REGION_WINDOWS.csv'):norms[r['region_id']].append(r)
        sources[str(folder/'RESIDUAL_REGION_WINDOWS.csv')]=sha(folder/'RESIDUAL_REGION_WINDOWS.csv')
        for arrangement,starts in PLANS.items():
            on=np.full((160,512,640),-np.inf,dtype=np.float32);off=np.full_like(on,-np.inf)
            for start in starts:
                window=folder/'windows'/f'window_{start:03d}.npz'
                if sha(window)!=receipt['window_sha256'][window.name]:raise AssertionError('Raw logit cache changed')
                sources[str(window)]=sha(window)
                with np.load(window) as z:
                    on[start:start+40]=np.maximum(on[start:start+40],z['on_logits'])
                    off[start:start+40]=np.maximum(off[start:start+40],z['off_logits'])
            if not (np.isfinite(on).all() and np.isfinite(off).all()):raise AssertionError('Raw-logit diagnostic fusion uncovered or nonfinite')
            delta=on-off
            pixel_rows.append(dict(sequence=sequence,arrangement=arrangement,frame='all',**pixel_stats(delta),
                logit_zero_decision_changes=int(np.count_nonzero((on>=0)!=(off>=0)))))
            for frame in range(160):
                pixel_rows.append(dict(sequence=sequence,arrangement=arrangement,frame=frame,**pixel_stats(delta[frame]),
                    logit_zero_decision_changes=int(np.count_nonzero((on[frame]>=0)!=(off[frame]>=0)))))
            for r in [r for r in regions if r['sequence']==sequence and r['arrangement']==arrangement]:
                out=dict(r)
                out.update(on_max_logit='',off_max_logit='',delta_max_logit='',on_mean_logit='',off_mean_logit='',delta_mean_logit='',
                    on_peak_x_index='',on_peak_y_index='',off_peak_x_index='',off_peak_y_index='',
                    delta_at_release_peak='',delta_at_off_peak='',delta_at_on_peak='',
                    mean_absolute_delta='',positive_delta_fraction='',negative_delta_fraction='',
                    residual_covering_windows='',feature_norm_window_mean='',correction_norm_window_mean='',relative_correction_norm_window_mean='',relative_correction_norm_window_max='')
                if r['valid']=='1':
                    frame,x0,y0,x1,y1=(int(r[k]) for k in ['frame','x0','y0','x1','y1'])
                    a=on[frame,y0:y1,x0:x1];b=off[frame,y0:y1,x0:x1];d=delta[frame,y0:y1,x0:x1]
                    ay,ax=np.unravel_index(int(np.argmax(a)),a.shape);by,bx=np.unravel_index(int(np.argmax(b)),b.shape)
                    px,py=int(r['release_peak_x_index']),int(r['release_peak_y_index'])
                    nr=norms[r['region_id']]
                    expected=sum(start<=frame<start+40 for start in starts)
                    if len(nr)!=expected or {int(x['window_start']) for x in nr}!={s for s in starts if s<=frame<s+40}:raise AssertionError('Residual window coverage differs from legal arrangement')
                    for n in nr:
                        if int(n['pixels'])!=(x1-x0)*(y1-y0):raise AssertionError('Residual region area changed')
                        ratio=np.sqrt(float(n['correction_squared_norm']))/max(np.sqrt(float(n['feature_squared_norm'])),1e-12)
                        if abs(ratio-float(n['relative_correction_norm']))>1e-12:raise AssertionError('Residual ratio disagrees with raw energies')
                    out.update(on_max_logit=float(a.max()),off_max_logit=float(b.max()),delta_max_logit=float(a.max()-b.max()),
                        on_mean_logit=float(a.mean(dtype=np.float64)),off_mean_logit=float(b.mean(dtype=np.float64)),delta_mean_logit=float(d.mean(dtype=np.float64)),
                        on_peak_x_index=int(ax+x0),on_peak_y_index=int(ay+y0),off_peak_x_index=int(bx+x0),off_peak_y_index=int(by+y0),
                        delta_at_release_peak=float(delta[frame,py,px]),delta_at_off_peak=float(d[by,bx]),delta_at_on_peak=float(d[ay,ax]),
                        mean_absolute_delta=float(np.abs(d).mean(dtype=np.float64)),positive_delta_fraction=float(np.mean(d>0)),negative_delta_fraction=float(np.mean(d<0)),
                        residual_covering_windows=len(nr),feature_norm_window_mean=float(np.mean([np.sqrt(float(n['feature_squared_norm'])) for n in nr])),
                        correction_norm_window_mean=float(np.mean([np.sqrt(float(n['correction_squared_norm'])) for n in nr])),
                        relative_correction_norm_window_mean=float(np.mean([float(n['relative_correction_norm']) for n in nr])),
                        relative_correction_norm_window_max=max(float(n['relative_correction_norm']) for n in nr))
                details.append(out)
            del on,off,delta
            print(json.dumps({'logit_analysis_complete':sequence,'arrangement':arrangement}),flush=True)
    if len(details)!=31200:raise AssertionError('Region analysis dropped rows')
    csvout(ROOT/'A_LOGIT_REGIONS.csv',details);csvout(ROOT/'A_PIXEL_CHANGES.csv',pixel_rows)
    summaries=[]
    for group,sequences in [('pooled',SEQUENCES)]+[(s,[s]) for s in SEQUENCES]:
        for arrangement in PLANS:
            for cohort in ['all','low_all_05','extreme_all_001','both_fixed_success']:
                for kind in ['target','hard_local','hard_global']:
                    rr=[r for r in details if r['sequence'] in sequences and r['arrangement']==arrangement and r['region_type']==kind and (cohort=='all' or r[cohort]=='1')]
                    valid=[r for r in rr if r['valid']=='1']
                    row={'group':group,'arrangement':arrangement,'cohort':cohort,'region_type':kind,
                        'target_frames':len(rr),'valid_regions':len(valid),'missing_regions':len(rr)-len(valid),
                        'tracks':len({(r['sequence'],r['track_id']) for r in valid}),'clips':len({r['sequence'] for r in valid}),
                        'distinct_region_locations':len({(r['sequence'],r['frame'],r['x0'],r['y0'],r['x1'],r['y1']) for r in valid})}
                    for key in ['delta_max_logit','delta_mean_logit','delta_at_release_peak','delta_at_off_peak','delta_at_on_peak','mean_absolute_delta','relative_correction_norm_window_mean']:
                        values=np.array([float(r[key]) for r in valid])
                        row.update({key+'_mean':float(values.mean()),key+'_q05':float(np.quantile(values,.05)),key+'_q50':float(np.quantile(values,.5)),key+'_q95':float(np.quantile(values,.95)),key+'_positive_count':int(np.count_nonzero(values>0)),key+'_negative_count':int(np.count_nonzero(values<0))})
                        bytrack=defaultdict(list)
                        for r in valid:bytrack[r['sequence'],r['track_id']].append(float(r[key]))
                        row[key+'_track_equal_mean']=float(np.mean([np.mean(v) for v in bytrack.values()]))
                    summaries.append(row)
    csvout(ROOT/'A_LOGIT_SUMMARY.csv',summaries)
    old_audit=json.loads((OLD/'TRAINING_AUDIT.json').read_text())
    checks['prior_training_audit_pass']=old_audit['status']=='PASS'
    checks['old_E2_all67_parameters_updated']=old_audit['details']['E2']['changed_parameter_tensors']==67
    checks['all_region_rows_preserved']=len(details)==31200
    checks['missing_local_regions_preserved']=sum(r['valid']=='0' for r in details)==2
    report={'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'source_sha256':sources,
        'script_sha256':sha(__file__),'regions_sha256':sha(ROOT/'A_REGIONS.csv'),
        'quantile_scope':'Exact per-frame and per-clip/arrangement pixel quantiles; diagnostic fused raw logits=max of actual window logits, detection remains sigmoid/probability-max',
        'region_weighting':'Each frozen target-arrangement contributes one target, one local-hard and one global-hard record; repeated background locations retained and unique locations disclosed; first average residual ratios over legal windows, then target-frame or track equally',
        'norm_limits':'Actual Wo output energies captured at inference; no stored full32-channel feature tensor, so later audit can verify energies-to-ratio but cannot independently reconstruct features',
        'no_accuracy_claim_from_logit_change':True}
    (ROOT/'A_BRANCH_CHAIN_AUDIT.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':report['status'],'region_rows':len(details),'summary_rows':len(summaries),'pixel_rows':len(pixel_rows)}),flush=True)


if __name__=='__main__':main()
