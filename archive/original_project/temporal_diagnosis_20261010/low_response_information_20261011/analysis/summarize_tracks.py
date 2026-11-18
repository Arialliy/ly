#!/usr/bin/env python3
"""Track persistence from the frozen 10,400-row low-response table only."""
from __future__ import annotations
import csv,hashlib,json,math,statistics,time
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
RUN=HERE.parent
SOURCE=HERE/'LOW_RESPONSE_OBJECTS.csv'
GROUPS=['all','low_all_05','extreme_all_001','low_nonextreme','both_fixed_success','extra_only','high_unmatched']
SEQS=['DJI_0026_2','DJI_0067_2']
MOTIONS=['all','stationary','small_motion','moving']
PEAKS={'target':('target_logit_peak_x','target_logit_peak_y'),
       'neighborhood':('neighborhood_peak_x','neighborhood_peak_y'),
       'reference':('reference_peak_x','reference_peak_y')}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def savecsv(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def num(row,key):
    return None if row.get(key,'')=='' else float(row[key])

def flag(row,key):return row.get(key)=='1'

def adjacent(a,b):
    return int(b['frame'])==int(a['frame'])+1 and int(b['source_frame'])==int(a['source_frame'])+1

def runs(rows,predicate):
    lengths=[];active=0;prev=None
    for r in rows:
        if prev is not None and not adjacent(prev,r):
            if active:lengths.append(active)
            active=0
        if predicate(r):active+=1
        else:
            if active:lengths.append(active)
            active=0
        prev=r
    if active:lengths.append(active)
    return lengths

def quantiles(values,prefix,qs=(('q25',.25),('median',.5),('q75',.75))):
    vv=[float(v) for v in values if v is not None]
    return {prefix+'_'+name:float(np.quantile(vv,q)) if vv else '' for name,q in qs}

def predicate_metric(rows,name,eligible,predicate):
    valid=[r for r in rows if eligible(r)]
    events=[r for r in valid if predicate(r)]
    lens=runs(rows,lambda r:eligible(r) and predicate(r))
    return {name+'_valid_frames':len(valid),name+'_frames':len(events),
        name+'_fraction':len(events)/len(valid) if valid else '',
        name+'_longest_run':max(lens,default=0),name+'_run_count':len(lens)}

METRICS={
    'arrangement_box_lt_05':(lambda r:True,lambda r:num(r,'target_box_max_probability')<.5),
    'arrangement_box_lt_001':(lambda r:True,lambda r:num(r,'target_box_max_probability')<.001),
    'frozen_all_window_low':(lambda r:True,lambda r:flag(r,'low_all_05')),
    'frozen_all_window_extreme':(lambda r:True,lambda r:flag(r,'extreme_all_001')),
    'matched':(lambda r:True,lambda r:flag(r,'matched_at_05')),
    'missed':(lambda r:True,lambda r:not flag(r,'matched_at_05')),
    'local_advantage':(lambda r:num(r,'local_valid_blocks')>=20,lambda r:num(r,'local_margin_q95')>0),
    'global_advantage':(lambda r:num(r,'global_valid_blocks')>=20,lambda r:num(r,'global_margin_q95')>0),
    'both_advantage':(lambda r:num(r,'local_valid_blocks')>=20 and num(r,'global_valid_blocks')>=20,
                      lambda r:num(r,'local_margin_q95')>0 and num(r,'global_margin_q95')>0),
    'local_only_advantage':(lambda r:num(r,'local_valid_blocks')>=20 and num(r,'global_valid_blocks')>=20,
                      lambda r:num(r,'local_margin_q95')>0 and num(r,'global_margin_q95')<=0),
    'neither_advantage':(lambda r:num(r,'local_valid_blocks')>=20 and num(r,'global_valid_blocks')>=20,
                      lambda r:num(r,'local_margin_q95')<=0 and num(r,'global_margin_q95')<=0),
    'reference_valid':(lambda r:True,lambda r:flag(r,'reference_valid')),
    'reference_box_lt_05':(lambda r:flag(r,'reference_valid'),lambda r:num(r,'reference_box_max_probability')<.5),
    'reference_box_lt_001':(lambda r:flag(r,'reference_valid'),lambda r:num(r,'reference_box_max_probability')<.001),
    'reference_local_advantage':(lambda r:flag(r,'reference_valid') and num(r,'reference_local_valid_blocks')>=20,
                               lambda r:num(r,'reference_local_margin_q95')>0),
    'reference_global_advantage':(lambda r:flag(r,'reference_valid') and num(r,'reference_global_valid_blocks')>=20,
                               lambda r:num(r,'reference_global_margin_q95')>0),
    'target_above_reference':(lambda r:flag(r,'reference_valid'),
                             lambda r:num(r,'target_box_max_logit')>num(r,'reference_max_logit')),
}
for scope in ['local','global']:
    def eligible(r,sc=scope):
        return flag(r,'reference_valid') and num(r,sc+'_valid_blocks')>=20 and num(r,'reference_'+sc+'_valid_blocks')>=20
    METRICS['paired_'+scope+'_target_advantage']=(eligible,lambda r,sc=scope:num(r,sc+'_margin_q95')>0)
    METRICS['paired_'+scope+'_reference_advantage']=(eligible,lambda r,sc=scope:num(r,'reference_'+sc+'_margin_q95')>0)
    METRICS['paired_'+scope+'_target_only_advantage']=(eligible,lambda r,sc=scope:num(r,sc+'_margin_q95')>0 and num(r,'reference_'+sc+'_margin_q95')<=0)
    METRICS['paired_'+scope+'_reference_only_advantage']=(eligible,lambda r,sc=scope:num(r,sc+'_margin_q95')<=0 and num(r,'reference_'+sc+'_margin_q95')>0)
    METRICS['paired_'+scope+'_both_advantage']=(eligible,lambda r,sc=scope:num(r,sc+'_margin_q95')>0 and num(r,'reference_'+sc+'_margin_q95')>0)
    METRICS['paired_'+scope+'_neither_advantage']=(eligible,lambda r,sc=scope:num(r,sc+'_margin_q95')<=0 and num(r,'reference_'+sc+'_margin_q95')<=0)

def motion_values(rows):
    out={(p,m):[] for p in PEAKS for m in MOTIONS}
    previous=None
    for r in rows:
        if previous is not None and adjacent(previous,r):
            a=previous
            cx0=num(a,'bbox_x')+num(a,'bbox_w')/2;cy0=num(a,'bbox_y')+num(a,'bbox_h')/2
            cx1=num(r,'bbox_x')+num(r,'bbox_w')/2;cy1=num(r,'bbox_y')+num(r,'bbox_h')/2
            dx,dy=cx1-cx0,cy1-cy0;motion=math.hypot(dx,dy)
            label='stationary' if motion==0 else ('small_motion' if motion<1 else 'moving')
            scale=(max(num(a,'bbox_w'),num(a,'bbox_h'))+max(num(r,'bbox_w'),num(r,'bbox_h')))/2
            for name,(px,py) in PEAKS.items():
                if name=='reference' and not (flag(a,'reference_valid') and flag(r,'reference_valid')):continue
                if any(num(q,k) is None for q in [a,r] for k in [px,py]):continue
                pdx,pdy=num(r,px)-num(a,px),num(r,py)-num(a,py)
                residual=math.hypot(pdx-dx,pdy-dy)
                values={'bbox_center_displacement':motion,'peak_displacement':math.hypot(pdx,pdy),
                        'relative_position_change':residual,'normalized_relative_position_change':residual/scale}
                out[(name,'all')].append(values);out[(name,label)].append(values)
        previous=r
    return out

def summarize_one(rows,sequence,track,arrangement):
    rows=sorted(rows,key=lambda r:(int(r['frame']),int(r['source_frame'])))
    continuity=runs(rows,lambda r:True)
    gaps=sum(not adjacent(a,b) for a,b in zip(rows,rows[1:]))
    base={'sequence':sequence,'track_id':track,'arrangement':arrangement,'target_frames':len(rows),
          'clips':1,'local_frame_start':min(int(r['frame']) for r in rows),'local_frame_end':max(int(r['frame']) for r in rows),
          'source_frame_start':min(int(r['source_frame']) for r in rows),'source_frame_end':max(int(r['source_frame']) for r in rows),
          'observation_segments':len(continuity),'observation_gap_pairs':gaps,'longest_contiguous_observation':max(continuity,default=0),
          'local_missing_between_observations':sum(max(0,int(b['frame'])-int(a['frame'])-1) for a,b in zip(rows,rows[1:])),
          'source_discontinuity_pairs':sum(int(b['source_frame'])!=int(a['source_frame'])+1 for a,b in zip(rows,rows[1:])),
          'box_size_change_adjacent_pairs':sum(adjacent(a,b) and (a['crop_w'],a['crop_h'])!=(b['crop_w'],b['crop_h']) for a,b in zip(rows,rows[1:])),
          'reference_offset_dx':rows[0]['reference_offset_dx'],'reference_offset_dy':rows[0]['reference_offset_dy'],
          'reference_offset_D':rows[0]['reference_offset_D'],'reference_candidate_index':rows[0]['reference_candidate_index']}
    for name,(eligible,pred) in METRICS.items():base.update(predicate_metric(rows,name,eligible,pred))
    for field,prefix in [('target_box_max_logit','target_logit'),('local_margin_q95','local_margin_q95'),
                         ('global_margin_q95','global_margin_q95'),('global_background_peaks_stronger_than_target','H_gt'),
                         ('global_background_peaks_not_lower_than_target','H_ge')]:
        base.update(quantiles([num(r,field) for r in rows],prefix))
    diff=[num(r,'target_box_max_logit')-num(r,'reference_max_logit') for r in rows if flag(r,'reference_valid')]
    base.update(quantiles(diff,'target_minus_reference_logit'))
    motions=motion_values(rows)
    for (peak,condition),values in motions.items():
        prefix=peak+'_'+condition
        base[prefix+'_adjacent_pairs']=len(values)
        for variable in ['bbox_center_displacement','peak_displacement','relative_position_change','normalized_relative_position_change']:
            base.update(quantiles([v[variable] for v in values],prefix+'_'+variable,qs=(('median',.5),('q90',.9))))
    return base

def tracks_for(rows):
    groups=defaultdict(list)
    for r in rows:groups[(r['sequence'],r['track_id'],r['arrangement'])].append(r)
    return [summarize_one(rr,*key) for key,rr in sorted(groups.items())]

def group_summary(rows,tracks,arrangement,sequence,group):
    n=len(rows)
    counts=sorted([int(r['target_frames']) for r in tracks],reverse=True)
    base={'arrangement':arrangement,'sequence':sequence,'group':group,'target_frames':n,
        'tracks':len(tracks),'clips':len({r['sequence'] for r in rows}),
        'top1_track_frames':sum(counts[:1]),'top1_track_fraction':sum(counts[:1])/n if n else '',
        'top5_tracks_frames':sum(counts[:5]),'top5_tracks_fraction':sum(counts[:5])/n if n else '',
        'observation_segments':sum(r['observation_segments'] for r in tracks),
        'tracks_with_observation_gaps':sum(r['observation_gap_pairs']>0 for r in tracks)}
    base.update(quantiles(counts,'frames_per_track',qs=(('median',.5),('q90',.9),('max',1.))))
    for name in METRICS:
        total_valid=sum(r[name+'_valid_frames'] for r in tracks);total_event=sum(r[name+'_frames'] for r in tracks)
        available=[r for r in tracks if r[name+'_valid_frames']]
        base.update({name+'_valid_frames':total_valid,name+'_frames':total_event,
            name+'_frame_weighted_fraction':total_event/total_valid if total_valid else '',
            name+'_tracks_with_valid_frames':len(available),name+'_tracks_with_events':sum(r[name+'_frames']>0 for r in tracks),
            name+'_track_equal_fraction':float(np.mean([r[name+'_fraction'] for r in available])) if available else ''})
        base.update(quantiles([r[name+'_longest_run'] for r in available],name+'_longest_run',qs=(('median',.5),('q90',.9),('max',1.))))
    for field,prefix in [('target_box_max_logit','target_logit'),('global_background_peaks_stronger_than_target','H_gt'),
                         ('global_background_peaks_not_lower_than_target','H_ge')]:
        base.update(quantiles([num(r,field) for r in rows],prefix))
        base[prefix+'_median_track_equal_mean']=float(np.mean([r[prefix+'_median'] for r in tracks])) if tracks else ''
    paired=[r for r in rows if flag(r,'reference_valid')]
    base.update(quantiles([num(r,'target_box_max_logit')-num(r,'reference_max_logit') for r in paired],'target_minus_reference_logit'))
    motions=defaultdict(list)
    partition=defaultdict(list)
    for r in rows:partition[(r['sequence'],r['track_id'])].append(r)
    for rr in partition.values():
        rr=sorted(rr,key=lambda r:int(r['frame']))
        for key,values in motion_values(rr).items():motions[key].extend(values)
    for (peak,condition) in [(p,m) for p in PEAKS for m in MOTIONS]:
        values=motions[(peak,condition)];prefix=peak+'_'+condition
        base[prefix+'_adjacent_pairs']=len(values)
        for variable in ['bbox_center_displacement','peak_displacement','relative_position_change','normalized_relative_position_change']:
            base.update(quantiles([v[variable] for v in values],prefix+'_'+variable,qs=(('median',.5),('q90',.9))))
    return base

def main():
    started=time.time()
    with SOURCE.open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==10400
    full=tracks_for(rows);assert len(full)==260
    summaries=[]
    for arrangement in ['O','S']:
      for sequence in ['pooled']+SEQS:
       for group in GROUPS:
        subset=[r for r in rows if r['arrangement']==arrangement and (sequence=='pooled' or r['sequence']==sequence) and (group=='all' or flag(r,group))]
        partial=tracks_for(subset)
        summaries.append(group_summary(subset,partial,arrangement,sequence,group))
    assert len(summaries)==42
    for arrangement in ['O','S']:
        rr=[r for r in full if r['arrangement']==arrangement]
        assert sum(r['target_frames'] for r in rr)==5200
        assert sum(r['matched_frames'] for r in rr)==(700 if arrangement=='O' else 720)
        assert sum(r['frozen_all_window_low_frames'] for r in rr)==4342
        assert sum(r['frozen_all_window_extreme_frames'] for r in rr)==4170
    for row in full:
        for name in METRICS:
            assert row[name+'_frames']<=row[name+'_valid_frames']<=row['target_frames']
            assert row[name+'_longest_run']<=row[name+'_frames']
        assert row['target_all_adjacent_pairs']==row['target_frames']-row['observation_segments']
        for peak in PEAKS:
            assert row[peak+'_all_adjacent_pairs']==sum(row[peak+'_'+condition+'_adjacent_pairs'] for condition in MOTIONS[1:])
        assert row['reference_all_adjacent_pairs']<=row['target_all_adjacent_pairs']
    savecsv(RUN/'TRACK_PERSISTENCE.csv',full);savecsv(RUN/'TRACK_SUMMARY.csv',summaries)
    schema={'input':str(SOURCE),'input_sha256':sha(SOURCE),'protocol_sha256':sha(RUN/'PROTOCOL.md'),
        'continuity':'Both local frame and source_frame must increase exactly 1. False, missing, insufficient-background or excluded cohort rows break the corresponding run.',
        'advantages':'q95 margin strictly > 0; target/reference local/global each require at least 20 valid background blocks; paired terms require both sides valid on same frame.',
        'reference':'Constant track offset preselected by geometry. Invalid reference rows break reference runs and reference adjacent pairs.',
        'motion':'Stationary center displacement exactly 0; small_motion 0<d<1; moving d>=1 pixel. Centers are box geometry, not exact target centers.',
        'relative_position_change':'Euclidean norm of peak displacement minus bbox-center displacement. Equal to change in peak-to-box-center relative offset. Normalized by mean max side over the adjacent pair.',
        'track_equal_fraction':'Mean of within-track event/valid ratios among tracks with at least one valid denominator frame. Frame_weighted sums numerators and denominators.',
        'group_summary':'Original flags remain overlapping. Per-group track rows/runs are rederived from the group-filtered observations; removed rows break runs. Missing/invalid rows are never replaced by negative/zero observations.',
        'quantiles':'Linear interpolation. Longest-run distributions include 0 for tracks with valid frames but no event; no-valid tracks excluded.',
        'track_columns':list(full[0]),'summary_columns':list(summaries[0]),'metric_names':list(METRICS),
        'status':'PASS','track_rows':260,'summary_rows':42,'elapsed_seconds':time.time()-started}
    (HERE/'TRACK_SCHEMA.json').write_text(json.dumps(schema,indent=2)+'\n')
    print(json.dumps({k:schema[k] for k in ['status','track_rows','summary_rows','elapsed_seconds']}))

if __name__=='__main__':main()
