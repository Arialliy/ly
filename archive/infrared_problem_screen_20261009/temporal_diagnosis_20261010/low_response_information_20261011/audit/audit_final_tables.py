#!/usr/bin/env python3
"""Independent checks of final assembly, peak summary and gap-aware trajectories."""
import csv,hashlib,json,math,statistics,time
from collections import defaultdict
from pathlib import Path

OUT=Path(__file__).resolve().parent;RUN=OUT.parent
GROUPS=['all','low_all_05','extreme_all_001','low_nonextreme','both_fixed_success','extra_only','high_unmatched']
PEAKS={'target':('target_logit_peak_x','target_logit_peak_y'),'neighborhood':('neighborhood_peak_x','neighborhood_peak_y'),'reference':('reference_peak_x','reference_peak_y')}
MODES=['all','stationary','small_motion','moving']

def read(name):
    with (RUN/name).open() as f:return list(csv.DictReader(f))
def identity(r):return r['sequence'],r['track_id'],r['frame'],r['arrangement']
def f(r,k):return float(r[k]) if r[k]!='' else None
def yes(r,k):return r[k]=='1'
def q(a,p):
    if not a:return ''
    a=sorted(a);i=(len(a)-1)*p;j=int(i);g=i-j
    return a[j]*(1-g)+a[min(j+1,len(a)-1)]*g
def quantiles(a,prefix,levels=None):
    vals=[v for v in a if v is not None]
    return {prefix+'_'+k:q(vals,p) for k,p in (levels or [('q25',.25),('median',.5),('q75',.75)])}
def membership(rows):return len(rows),len({(r['sequence'],r['track_id']) for r in rows}),len({r['sequence'] for r in rows})
def adjacent(a,b):return (int(b['frame'])-int(a['frame']),int(b['source_frame'])-int(a['source_frame']))==(1,1)

def event_runs(rows):
    # Run computation sees only event rows; any missing/false/invalid observation
    # must therefore leave a gap in absolute/source frame indices.
    if not rows:return []
    lengths=[];start=0
    for i in range(1,len(rows)):
        if not adjacent(rows[i-1],rows[i]):lengths.append(i-start);start=i
    lengths.append(len(rows)-start)
    return lengths

def metrics_for_row(r):
    local=f(r,'local_valid_blocks')>=20;glob=f(r,'global_valid_blocks')>=20
    lp=f(r,'local_margin_q95');gp=f(r,'global_margin_q95')
    la=lp is not None and lp>0;ga=gp is not None and gp>0
    rv=yes(r,'reference_valid')
    rl=rv and f(r,'reference_local_valid_blocks')>=20;rg=rv and f(r,'reference_global_valid_blocks')>=20
    rlp=f(r,'reference_local_margin_q95');rgp=f(r,'reference_global_margin_q95')
    rla=rlp is not None and rlp>0;rga=rgp is not None and rgp>0
    out={
      'arrangement_box_lt_05':(True,f(r,'target_box_max_probability')<.5),
      'arrangement_box_lt_001':(True,f(r,'target_box_max_probability')<.001),
      'frozen_all_window_low':(True,yes(r,'low_all_05')),'frozen_all_window_extreme':(True,yes(r,'extreme_all_001')),
      'matched':(True,yes(r,'matched_at_05')),'missed':(True,not yes(r,'matched_at_05')),
      'local_advantage':(local,la),'global_advantage':(glob,ga),'both_advantage':(local and glob,la and ga),
      'local_only_advantage':(local and glob,la and not ga),'neither_advantage':(local and glob,not la and not ga),
      'reference_valid':(True,rv),
      'reference_box_lt_05':(rv,rv and f(r,'reference_box_max_probability')<.5),
      'reference_box_lt_001':(rv,rv and f(r,'reference_box_max_probability')<.001),
      'reference_local_advantage':(rl,rla),'reference_global_advantage':(rg,rga),
      'target_above_reference':(rv,rv and f(r,'target_box_max_logit')>f(r,'reference_max_logit'))}
    for scope,eligible,ta,ba in [('local',local and rl,la,rla),('global',glob and rg,ga,rga)]:
        for kind,event in [('target',ta),('reference',ba),('target_only',ta and not ba),('reference_only',ba and not ta),('both',ta and ba),('neither',not ta and not ba)]:
            out['paired_'+scope+'_'+kind+'_advantage']=(eligible,event)
    return out

def motion_pairs(rows):
    values={(p,m):[] for p in PEAKS for m in MODES}
    for a,b in zip(rows,rows[1:]):
        if not adjacent(a,b):continue
        displacement=[f(b,'bbox_'+d)+f(b,'bbox_'+size)/2-f(a,'bbox_'+d)-f(a,'bbox_'+size)/2 for d,size in [('x','w'),('y','h')]]
        length=math.hypot(*displacement);mode='stationary' if length==0 else ('small_motion' if length<1 else 'moving')
        scale=(max(f(a,'bbox_w'),f(a,'bbox_h'))+max(f(b,'bbox_w'),f(b,'bbox_h')))/2
        for name,columns in PEAKS.items():
            if name=='reference' and not(yes(a,'reference_valid') and yes(b,'reference_valid')):continue
            if any(f(r,c) is None for r in [a,b] for c in columns):continue
            shift=[f(b,c)-f(a,c) for c in columns];relative=math.hypot(*(u-v for u,v in zip(shift,displacement)))
            v={'bbox_center_displacement':length,'peak_displacement':math.hypot(*shift),'relative_position_change':relative,'normalized_relative_position_change':relative/scale}
            values[name,'all'].append(v);values[name,mode].append(v)
    return values

def add_motion(exp,motions):
    for (name,mode),vals in motions.items():
        prefix=name+'_'+mode;exp[prefix+'_adjacent_pairs']=len(vals)
        for field in ['bbox_center_displacement','peak_displacement','relative_position_change','normalized_relative_position_change']:
            exp.update(quantiles([r[field] for r in vals],prefix+'_'+field,[('median',.5),('q90',.9)]))

def one_track(rows):
    rows=sorted(rows,key=lambda r:(int(r['frame']),int(r['source_frame'])));first=rows[0]
    segments=event_runs(rows);pairs=list(zip(rows,rows[1:]))
    e={'sequence':first['sequence'],'track_id':first['track_id'],'arrangement':first['arrangement'],'target_frames':len(rows),'clips':1,
       'local_frame_start':int(rows[0]['frame']),'local_frame_end':int(rows[-1]['frame']),
       'source_frame_start':min(int(r['source_frame']) for r in rows),'source_frame_end':max(int(r['source_frame']) for r in rows),
       'observation_segments':len(segments),'observation_gap_pairs':len(segments)-1,'longest_contiguous_observation':max(segments),
       'local_missing_between_observations':sum(max(0,int(b['frame'])-int(a['frame'])-1) for a,b in pairs),
       'source_discontinuity_pairs':sum(int(b['source_frame'])-int(a['source_frame'])!=1 for a,b in pairs),
       'box_size_change_adjacent_pairs':sum(adjacent(a,b) and (a['crop_w'],a['crop_h'])!=(b['crop_w'],b['crop_h']) for a,b in pairs)}
    for k in ['reference_offset_dx','reference_offset_dy','reference_offset_D','reference_candidate_index']:e[k]=first[k]
    flags={identity(r):metrics_for_row(r) for r in rows};names=list(flags[identity(first)])
    for name in names:
        eligible=[r for r in rows if flags[identity(r)][name][0]]
        events=[r for r in eligible if flags[identity(r)][name][1]];lens=event_runs(events)
        e.update({name+'_valid_frames':len(eligible),name+'_frames':len(events),name+'_fraction':len(events)/len(eligible) if eligible else '',
                  name+'_longest_run':max(lens,default=0),name+'_run_count':len(lens)})
    for field,prefix in [('target_box_max_logit','target_logit'),('local_margin_q95','local_margin_q95'),('global_margin_q95','global_margin_q95'),
                         ('global_background_peaks_stronger_than_target','H_gt'),('global_background_peaks_not_lower_than_target','H_ge')]:
        e.update(quantiles([f(r,field) for r in rows],prefix))
    e.update(quantiles([f(r,'target_box_max_logit')-f(r,'reference_max_logit') for r in rows if yes(r,'reference_valid')],'target_minus_reference_logit'))
    add_motion(e,motion_pairs(rows));return e

def groups(rows):
    out=defaultdict(list)
    for r in rows:out[r['sequence'],r['track_id'],r['arrangement']].append(r)
    return {k:sorted(v,key=lambda r:int(r['frame'])) for k,v in out.items()}

def group_track_summary(rows):
    partition=groups(rows);tracks=[one_track(r) for r in partition.values()];n=len(rows);counts=sorted((len(r) for r in partition.values()),reverse=True)
    e={'target_frames':n,'tracks':len(tracks),'clips':len({r['sequence'] for r in rows}),
       'top1_track_frames':sum(counts[:1]),'top1_track_fraction':sum(counts[:1])/n if n else '',
       'top5_tracks_frames':sum(counts[:5]),'top5_tracks_fraction':sum(counts[:5])/n if n else '',
       'observation_segments':sum(t['observation_segments'] for t in tracks),'tracks_with_observation_gaps':sum(t['observation_gap_pairs']>0 for t in tracks)}
    e.update(quantiles(counts,'frames_per_track',[('median',.5),('q90',.9),('max',1.)]))
    names=list(metrics_for_row(read('analysis/LOW_RESPONSE_OBJECTS.csv')[0]))
    for name in names:
        valid=sum(t[name+'_valid_frames'] for t in tracks);events=sum(t[name+'_frames'] for t in tracks)
        available=[t for t in tracks if t[name+'_valid_frames']]
        e.update({name+'_valid_frames':valid,name+'_frames':events,name+'_frame_weighted_fraction':events/valid if valid else '',
                  name+'_tracks_with_valid_frames':len(available),'%s_tracks_with_events'%name:sum(t[name+'_frames']>0 for t in tracks),
                  name+'_track_equal_fraction':statistics.mean(t[name+'_fraction'] for t in available) if available else ''})
        e.update(quantiles([t[name+'_longest_run'] for t in available],name+'_longest_run',[('median',.5),('q90',.9),('max',1.)]))
    for field,prefix in [('target_box_max_logit','target_logit'),('global_background_peaks_stronger_than_target','H_gt'),('global_background_peaks_not_lower_than_target','H_ge')]:
        e.update(quantiles([f(r,field) for r in rows],prefix));e[prefix+'_median_track_equal_mean']=statistics.mean(t[prefix+'_median'] for t in tracks) if tracks else ''
    e.update(quantiles([f(r,'target_box_max_logit')-f(r,'reference_max_logit') for r in rows if yes(r,'reference_valid')],'target_minus_reference_logit'))
    motions={(p,m):[] for p in PEAKS for m in MODES}
    for rr in partition.values():
        for k,v in motion_pairs(rr).items():motions[k].extend(v)
    add_motion(e,motions);return e

def main():
    begin=time.time();errors=[];cells=0;largest=0
    def compare(actual,expected,where):
        nonlocal cells,largest
        assert set(actual)==set(expected),(where,'columns',set(actual)^set(expected))
        for field,value in expected.items():
            cells+=1
            if isinstance(value,(int,float)):
                diff=abs(float(actual[field])-value);largest=max(largest,diff);ok=diff<1e-8
            else:ok=actual[field]==value
            if not ok:errors.append({'where':where,'column':field,'actual':actual[field],'expected':value})
    original=read('analysis/LOW_RESPONSE_OBJECTS.csv');final=read('LOW_RESPONSE_OBJECTS.csv');reviews=read('visual/PEAK_REVIEW.csv')
    olds={identity(r):r for r in original};labels={identity(r):r for r in reviews};assert len(labels)==len(reviews)==144
    assert len(final)==len(olds)==10400
    labelmap={'T':'suspected_target','B':'background_or_other','U':'uncertain'}
    for r in final:
        k=identity(r);assert all(r[c]==v for c,v in olds[k].items()),('numeric rewrite',k)
        label=labels.get(k);expect={'visual_reviewed':str(int(label is not None)),
          'visual_peak_label':labelmap[label['review_label_code']] if label else '',
          'visual_case_ids':label['case_ids'] if label else '',
          'visual_peak_evidence':label['peak_observation'] if label else '',
          'visual_raw_readability':label.get('raw_readability','') if label else ''}
        assert {k:r[k] for k in expect}==expect
    assert (RUN/'SCORE_SEPARATION.csv').read_bytes()==(RUN/'analysis/SCORE_SEPARATION.csv').read_bytes()
    peak_table=read('PEAK_LOCATION.csv')
    for actual in peak_table:
        plan,seq,group=[actual[k] for k in ['arrangement','sequence','group']]
        rr=[r for r in final if r['arrangement']==plan and (seq=='pooled' or r['sequence']==seq) and (group=='all' or yes(r,group))]
        n,nt,nc=membership(rr);e={'arrangement':plan,'sequence':seq,'group':group,'target_frames':n,'tracks':nt,'clips':nc}
        subsets=defaultdict(list)
        for r in rr:
            own=yes(r,'neighborhood_peak_inside_target');other=bool(r['neighborhood_peak_other_gt_ids']);lab=r['visual_peak_label'];ge=f(r,'neighborhood_peak_probability')>=.5
            checks={'matched_at_05':yes(r,'matched_at_05'),'peak_inside_own_box':own,'peak_in_other_annotated_box':other,
              'peak_outside_all_annotated_boxes':not own and not other,'neighborhood_probability_ge05':f(r,'neighborhood_probability_max')>=.5,
              'neighborhood_logit_ge0':f(r,'neighborhood_peak_logit')>=0,'inside_peak_ge05':own and ge,'outside_peak_ge05':not own and ge,
              'nearly_flat':yes(r,'neighborhood_nearly_flat'),'border_clipped':yes(r,'neighborhood_border_clipped'),
              'target_logit_probability_argmax_differs':not yes(r,'target_logit_and_probability_peak_same'),
              'neighborhood_logit_probability_argmax_differs':not yes(r,'neighborhood_logit_and_probability_peak_same'),
              'visual_reviewed':yes(r,'visual_reviewed'),'visual_suspected_target':lab=='suspected_target',
              'visual_background_or_other':lab=='background_or_other','visual_uncertain':lab=='uncertain',
              'visual_suspected_target_outside':lab=='suspected_target' and not own,
              'visual_suspected_target_ge05_unmatched':lab=='suspected_target' and ge and not yes(r,'matched_at_05')}
            assert sum([own,other,not own and not other])==1
            for k,condition in checks.items():
                subsets[k] # retain empty category
                if condition:subsets[k].append(r)
        condition_names=['matched_at_05','peak_inside_own_box','peak_in_other_annotated_box','peak_outside_all_annotated_boxes','neighborhood_probability_ge05','neighborhood_logit_ge0','inside_peak_ge05','outside_peak_ge05','nearly_flat','border_clipped','target_logit_probability_argmax_differs','neighborhood_logit_probability_argmax_differs','visual_reviewed','visual_suspected_target','visual_background_or_other','visual_uncertain','visual_suspected_target_outside','visual_suspected_target_ge05_unmatched']
        for name in condition_names:
            for measure,value in zip(['frames','tracks','clips'],membership(subsets[name])):e[name+'_'+measure]=value
        for field in ['target_box_max_logit','neighborhood_peak_logit','neighborhood_peak_distance_to_box','neighborhood_peak_distance_normalized','neighborhood_max_minus_median','neighborhood_max_minus_q95']:
            e.update(quantiles([f(r,field) for r in rr],field,[('q25',.25),('median',.5),('q75',.75),('q95',.95)]))
        compare(actual,e,('peak',plan,seq,group))
    bytrack=groups(original);track_table=read('TRACK_PERSISTENCE.csv')
    for actual in track_table:
        k=actual['sequence'],actual['track_id'],actual['arrangement'];compare(actual,one_track(bytrack[k]),('track',*k))
    summary=read('TRACK_SUMMARY.csv')
    for actual in summary:
        plan,seq,group=[actual[k] for k in ['arrangement','sequence','group']]
        rr=[r for r in original if r['arrangement']==plan and (seq=='pooled' or r['sequence']==seq) and (group=='all' or yes(r,group))]
        compare(actual,{'arrangement':plan,'sequence':seq,'group':group,**group_track_summary(rr)},('track_summary',plan,seq,group))
    # Counterexample checks ensure source gaps, filtered-away false rows, and
    # invalid reference frames cannot be bridged by the audit's own run method.
    assert event_runs([{'frame':'1','source_frame':'10'},{'frame':'2','source_frame':'12'}])==[1,1]
    assert event_runs([{'frame':'1','source_frame':'10'},{'frame':'3','source_frame':'12'}])==[1,1]
    report={'status':'PASS' if not errors else 'FAIL','all_original_numeric_cells_preserved':True,'detail_rows':len(final),
      'unique_manual_rows':len(labels),'unique_manual_target_frames':len({k[:3] for k in labels}),
      'unreviewed_rows':sum(not yes(r,'visual_reviewed') for r in final),'peak_summary_rows':len(peak_table),
      'track_rows':len(track_table),'track_summary_rows':len(summary),'checked_aggregate_cells':cells,
      'largest_numeric_difference':largest,'mismatches':errors,'elapsed_seconds':time.time()-begin,
      'gap_and_denominator_checks':'Both frame clocks, excluded cohort rows, false/NA/<20 advantages and invalid references interrupt runs. Frame weighted and equal-track denominators independently recomputed; moving/statics never mixed.',
      'source_sha256':{name:hashlib.sha256((RUN/name).read_bytes()).hexdigest() for name in ['LOW_RESPONSE_OBJECTS.csv','PEAK_LOCATION.csv','SCORE_SEPARATION.csv','TRACK_PERSISTENCE.csv','TRACK_SUMMARY.csv','visual/PEAK_REVIEW.csv','analysis/summarize_tracks.py']}}
    (OUT/'FINAL_TABLES_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['status','checked_aggregate_cells','mismatches','elapsed_seconds']}))
    if errors:raise AssertionError(errors[:5])

if __name__=='__main__':main()
