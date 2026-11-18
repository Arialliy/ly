#!/usr/bin/env python3
"""CPU-only complete annotation condition census; no image/prediction-array selection."""
import collections
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

D = Path(__file__).resolve().parent
F = D.parent
Q = F.parent
V = Q / 'problem_validity_20261011'
P = Q / 'stage1_goal'
SEQS = ('DJI_0026_2', 'DJI_0067_2')
ARRANGEMENTS = {'official': (0,36,72,108,120), 'symmetric': (0,24,60,96,120), 'diagnostic': tuple(range(0,121,12))}
PRIMARY = ('gain', 'loss', 'common_success', 'all11_failure')

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def csvread(p):
    with p.open() as f: return list(csv.DictReader(f))
def csvwrite(p, rows):
    assert rows
    with p.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def js(p, v): p.write_text(json.dumps(v, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
def key(r): return r['sequence'], int(r['frame']), int(r['annotation_id'])
def quant(v, p):
    if not v: return ''
    a=sorted(v); j=(len(a)-1)*p; lo=int(j); hi=math.ceil(j)
    return a[lo]+(a[hi]-a[lo])*(j-lo)
def qstats(v):
    v=[x for x in v if x!='']
    return dict(n=len(v), q25=quant(v,.25), median=quant(v,.5), q75=quant(v,.75), p90=quant(v,.9), min=min(v) if v else '', max=max(v) if v else '')
def usable(a, im):
    x,y,w,h=map(float,a['bbox'])
    return not a.get('ignore',0) and all(math.isfinite(v) for v in (x,y,w,h)) and w>0 and h>0 and min(im['width'],x+w)>max(0,x) and min(im['height'],y+h)>max(0,y)
def center(a):
    x,y,w,h=map(float,a['bbox']); return x+w/2,y+h/2
def bin_scale(x): return '(0,4]' if x<=4 else '(4,8]' if x<=8 else '(8,16]'
def bin_frame(t): return '0..39' if t<40 else '40..119' if t<120 else '120..159'
def groups(r): return ['all_small', r['group']]

def main():
    if (D/'SUMMARY.json').exists(): raise FileExistsError(D/'SUMMARY.json')
    inputs=[V/'OBJECTS.csv', V/'WINDOWS.csv', V/'control_summary/OBJECTS.csv', F/'PROTOCOL.md']
    ih={str(p):sha(p) for p in inputs}
    objects=[r for r in csvread(inputs[0]) if r['sequence'] in SEQS and r['small_geometry']=='1']
    controls={key(r):r for r in csvread(inputs[2]) if r['mode']=='symmetric5_probmax' and r['small_geometry']=='1'}
    wins=collections.defaultdict(list)
    for r in csvread(inputs[1]):
        if r['sequence'] in SEQS and key(r) in controls: wins[key(r)].append(r)
    assert len(objects)==len(controls)==5200
    assert len({key(r) for r in objects})==5200
    data={}; annotation_sources={}
    for seq in SEQS:
        path=P/'data'/seq/'annotations.json'; manifest_path=P/'data'/seq/'FRAME_MANIFEST.json'
        raw=json.loads(path.read_text()); manifest=json.loads(manifest_path.read_text()); ims=raw['images']
        loc={im['id']:i for i,im in enumerate(ims)}
        aa=[[] for _ in ims]; track=[{} for _ in ims]; byid={}
        for a in raw['annotations']:
            t=loc[a['image_id']]; aa[t].append(a); assert a['track_id'] not in track[t]
            track[t][a['track_id']]=a; byid[a['id']]=a
        valid=[[a for a in row if usable(a,im)] for row,im in zip(aa,ims)]
        source=[int(Path(im['file_name']).stem) for im in ims]
        assert source==list(range(source[0],source[0]+160))
        data[seq]=dict(ims=ims,aa=aa,track=track,byid=byid,valid=valid,source=source,manifest=manifest)
        annotation_sources[seq]=dict(path=str(path),sha256=sha(path),manifest_path=str(manifest_path),manifest_sha256=sha(manifest_path))
    window_rows=[]; rows=[]; cache={}
    def window_condition(seq, trackid, start):
        ck=(seq,trackid,start)
        if ck in cache: return cache[ck]
        z=data[seq]; ids=list(range(start,start+40))
        present=[j for j in ids if trackid in z['track'][j]]
        valid=[j for j in present if usable(z['track'][j][trackid],z['ims'][j])]
        centers={j:center(z['track'][j][trackid]) for j in valid}
        steps=[math.dist(centers[j],centers[j-1]) for j in valid if j-1 in centers]
        counts=[len(z['valid'][j]) for j in ids]
        res=dict(annotation_presence_fraction=len(present)/40,usable_annotation_fraction=len(valid)/40,
                 consecutive_center_step_count=len(steps),center_step_median=quant(steps,.5),center_step_p90=quant(steps,.9),
                 window_valid_annotation_count_mean=statistics.mean(counts),window_valid_annotation_count_median=quant(counts,.5))
        cache[ck]=res; return res
    for obj in sorted(objects,key=key):
        seq,t,aid=key(obj); tid=int(obj['track_id']); z=data[seq]; a=z['byid'][aid]; ctl=controls[key(obj)]
        assert int(ctl['official_matched'])==int(obj['fused_matched'])
        original=int(ctl['official_matched']); sym=int(ctl['control_matched']); any11=int(obj['any11_matched'])
        group='gain' if not original and sym else 'loss' if original and not sym else 'common_success' if original and sym else 'all11_failure' if not any11 else 'both_failure_with_window_opportunity'
        assert int(ctl['gain'])==int(group=='gain') and int(ctl['loss'])==int(group=='loss')
        current_center=center(a)
        nearby=[math.dist(current_center,center(b)) for b in z['valid'][t] if b['id']!=aid]
        base=dict(group=group,sequence=seq,frame=t,source_frame=z['source'][t],annotation_id=aid,track_id=tid,
                  category=obj['category'],category_id=int(obj['category_id']),bbox_x=float(obj['bbox_x']),bbox_y=float(obj['bbox_y']),
                  bbox_w=float(obj['bbox_w']),bbox_h=float(obj['bbox_h']),max_side=float(obj['max_side']),scale_bin=bin_scale(float(obj['max_side'])),
                  frame_position_bin=bin_frame(t),official_matched=original,symmetric_matched=sym,any11_matched=any11,
                  original_system_case=obj['system_case'],all11_failure_response_layer=('high_box_ge_0.5' if float(obj['context_box_max'])>=.5 else 'low_box_lt_0.5') if not any11 else '',
                  diagnostic_box_min=float(obj['context_box_min']),diagnostic_box_max=float(obj['context_box_max']),
                  nearest_other_annotation_center_px=min(nearby) if nearby else '',frame_valid_annotation_count=len(z['valid'][t]),
                  frame_valid_annotation_density_per_megapixel=len(z['valid'][t])/(z['ims'][t]['width']*z['ims'][t]['height'])*1e6,
                  multi_context=int(obj['multi_context']),all_available_windows_match=int(not int(obj['match_disagreement']) and any11),
                  raw_frames_path=str(P/'data'/seq/'raw_frames.npy'),raw_array_frame_index=t,
                  source_file_name=z['ims'][t]['file_name'],source_image_id=z['ims'][t]['id'])
        base['strict_stable_control']=int(group=='common_success' and base['multi_context'] and base['all_available_windows_match'])
        wr={int(r['window_start']):r for r in wins[key(obj)]}
        for name,starts in ARRANGEMENTS.items():
            active=[s for s in starts if s<=t<s+40]
            expected=[s for s in wr if s in starts]
            assert sorted(active)==sorted(expected)
            window_info=[window_condition(seq,tid,s) for s in active]
            support=sorted({j for s in active for j in range(s,s+40)})
            centers={j:center(z['track'][j][tid]) for j in support if tid in z['track'][j] and usable(z['track'][j][tid],z['ims'][j])}
            steps=[math.dist(centers[j],centers[j-1]) for j in centers if j-1 in centers]
            base.update({name+'_window_starts':';'.join(map(str,active)),name+'_slots':';'.join(str(t-s) for s in active),
                         name+'_cover_count':len(active),name+'_earliest_access_source_frame':z['source'][support[0]],
                         name+'_latest_access_source_frame':z['source'][support[-1]],name+'_latest_access_relative_to_current':support[-1]-t,
                         name+'_earliest_access_relative_to_current':support[0]-t,name+'_unique_support_frames':len(support),
                         name+'_support_annotation_presence_fraction':sum(tid in z['track'][j] for j in support)/len(support),
                         name+'_support_center_step_median':quant(steps,.5),name+'_support_center_step_p90':quant(steps,.9),
                         name+'_support_consecutive_step_count':len(steps),name+'_any_window_actual_match':int(any(int(wr[s]['matched']) for s in active))})
            for field in ('annotation_presence_fraction','usable_annotation_fraction','center_step_median','center_step_p90','window_valid_annotation_count_mean'):
                values=[w[field] for w in window_info if w[field]!='']
                base[name+'_window_'+field+'_median']=quant(values,.5)
                base[name+'_window_'+field+'_min']=min(values) if values else ''
                base[name+'_window_'+field+'_max']=max(values) if values else ''
            for s in active:
                window_rows.append(dict(group=group,sequence=seq,frame=t,source_frame=z['source'][t],annotation_id=aid,track_id=tid,
                    arrangement=name,window_start=s,slot=t-s,window_source_first=z['source'][s],window_source_last=z['source'][s+39],
                    actual_match=int(wr[s]['matched']),box_max=float(wr[s]['box_max']),**window_condition(seq,tid,s)))
        base['symmetric_minus_official_latest_access_frames']=base['symmetric_latest_access_source_frame']-base['official_latest_access_source_frame']
        rows.append(base)
    counts=dict(collections.Counter(r['group'] for r in rows))
    assert counts==dict(gain=74,loss=54,common_success=646,all11_failure=4391,both_failure_with_window_opportunity=35),counts
    assert sum(not r['official_matched'] and not r['symmetric_matched'] for r in rows)==4426
    assert sum(r['strict_stable_control'] for r in rows)==422
    # Events are observation runs, broken by missing or ineligible target frames.
    events=[]; track_rows=[]
    track_groups=collections.defaultdict(list)
    for r in rows: track_groups[(r['group'],r['sequence'],r['track_id'])].append(r)
    for (group,seq,tid), rr in sorted(track_groups.items()):
        rr.sort(key=lambda x:(x['source_frame'],x['annotation_id'])); chunks=[]
        for r in rr:
            if not chunks or r['source_frame']!=chunks[-1][-1]['source_frame']+1: chunks.append([])
            chunks[-1].append(r)
        for ci,chunk in enumerate(chunks):
            events.append(dict(group=group,sequence=seq,track_id=tid,event_index=ci,first_source_frame=chunk[0]['source_frame'],
                               last_source_frame=chunk[-1]['source_frame'],object_frames=len(chunk),annotation_ids=';'.join(str(x['annotation_id']) for x in chunk)))
        track_rows.append(dict(group=group,sequence=seq,track_id=tid,category=rr[0]['category'],object_frames=len(rr),events=len(chunks),
                               first_source_frame=rr[0]['source_frame'],last_source_frame=rr[-1]['source_frame'],max_side_median=quant([r['max_side'] for r in rr],.5),
                               nearest_center_distance_median=quant([r['nearest_other_annotation_center_px'] for r in rr if r['nearest_other_annotation_center_px']!=''],.5),
                               official_support_center_step_median=quant([r['official_support_center_step_median'] for r in rr if r['official_support_center_step_median']!=''],.5)))
    summary_groups=[]
    for group in ('all_small',*PRIMARY,'both_failure_with_window_opportunity'):
        rr=rows if group=='all_small' else [r for r in rows if r['group']==group]
        summary_groups.append(dict(group=group,object_frames=len(rr),sequences=len({r['sequence'] for r in rr}),
                                   sequence_tracks=len({(r['sequence'],r['track_id']) for r in rr}),
                                   observation_events='' if group=='all_small' else sum(e['group']==group for e in events),
                                   strict_stable_controls=sum(r['strict_stable_control'] for r in rr)))
    categorical=[]
    for field in ('sequence','category','scale_bin','frame_position_bin','official_cover_count','symmetric_cover_count','all11_failure_response_layer'):
        for level in sorted({str(r[field]) for r in rows}):
            if level=='': continue
            eligible=[r for r in rows if str(r[field])==level]
            for group in PRIMARY+('both_failure_with_window_opportunity',):
                rr=[r for r in eligible if r['group']==group]
                categorical.append(dict(condition=field,level=level,group=group,object_frames=len(rr),group_total=counts[group],
                                        fraction_within_group=len(rr)/counts[group],all_small_at_condition=len(eligible),
                                        fraction_of_condition=len(rr)/len(eligible),sequence_tracks=len({(r['sequence'],r['track_id']) for r in rr})))
    continuous=[]
    fields=['max_side','nearest_other_annotation_center_px','frame_valid_annotation_count','frame_valid_annotation_density_per_megapixel',
            'symmetric_minus_official_latest_access_frames']
    fields += [name+'_'+f for name in ARRANGEMENTS for f in ('cover_count','latest_access_relative_to_current','support_annotation_presence_fraction',
               'support_center_step_median','support_center_step_p90','window_annotation_presence_fraction_median','window_annotation_presence_fraction_min',
               'window_center_step_median_median','window_center_step_p90_median','window_window_valid_annotation_count_mean_median')]
    for group in ('all_small',*PRIMARY,'both_failure_with_window_opportunity'):
        rr=rows if group=='all_small' else [r for r in rows if r['group']==group]
        for field in fields: continuous.append(dict(group=group,condition=field,**qstats([r[field] for r in rr])))
    # Outcome-independent continuous scaling; exact categorical strata, with replacement.
    match_fields=('max_side','official_support_center_step_median','official_support_center_step_p90','nearest_other_annotation_center_px')
    scales={f:(quant([r[f] for r in rows if r[f]!=''],.75)-quant([r[f] for r in rows if r[f]!=''],.25)) or 1.0 for f in match_fields}
    pool=[r for r in rows if r['strict_stable_control']]
    matched=[]
    for r in rows:
        if r['group'] not in ('gain','loss'): continue
        candidates=[c for c in pool if all(c[f]==r[f] for f in ('sequence','category','scale_bin'))]
        def distance(c):
            if any(c[f]=='' or r[f]=='' for f in match_fields): return math.inf
            return sum(abs(c[f]-r[f])/scales[f] for f in match_fields)
        ranked=sorted(candidates,key=lambda c:(distance(c),c['track_id'],c['source_frame'],c['annotation_id']))
        best=ranked[0] if ranked and math.isfinite(distance(ranked[0])) else None
        mr=dict(group=r['group'],sequence=r['sequence'],frame=r['frame'],source_frame=r['source_frame'],annotation_id=r['annotation_id'],track_id=r['track_id'],
                category=r['category'],scale_bin=r['scale_bin'],candidate_count=len(candidates),matched=int(best is not None),
                unmatched_reason='' if best else 'no exact sequence/category/scale stratum with complete motion/geometry',
                control_sequence=best['sequence'] if best else '',control_frame=best['frame'] if best else '',
                control_source_frame=best['source_frame'] if best else '',control_annotation_id=best['annotation_id'] if best else '',
                control_track_id=best['track_id'] if best else '',normalized_l1_distance=distance(best) if best else '',
                same_track=int(best['track_id']==r['track_id']) if best else '')
        for f in match_fields:
            mr['case_'+f]=r[f]; mr['control_'+f]=best[f] if best else ''; mr['absolute_difference_'+f]=abs(r[f]-best[f]) if best else ''
        matched.append(mr)
    reuse=collections.Counter((r['control_sequence'],r['control_annotation_id']) for r in matched if r['matched'])
    for r in matched: r['control_reuse_count']=reuse[(r['control_sequence'],r['control_annotation_id'])] if r['matched'] else ''
    # Freeze representative cases before any image inspection.
    sampled=[]; strata=[]
    for seq in SEQS:
        for layer in ('high_box_ge_0.5','low_box_lt_0.5'):
            rr=[r for r in rows if r['group']=='all11_failure' and r['sequence']==seq and r['all11_failure_response_layer']==layer]
            tids=sorted({r['track_id'] for r in rr}); chosen=tids[:4]
            strata.append(dict(sequence=seq,response_layer=layer,object_frames=len(rr),tracks=len(tids),selected_track_ids=chosen))
            for tid in chosen:
                candidates=sorted((r for r in rr if r['track_id']==tid),key=lambda r:(r['source_frame'],r['annotation_id']))
                r=candidates[len(candidates)//2]
                sampled.append(dict(sample_id=f'{seq}_track{tid}_{layer}_ann{r["annotation_id"]}',sequence=seq,response_layer=layer,
                                    stratum_track_object_frames=len(candidates),representative_index=len(candidates)//2,
                                    **{f:r[f] for f in ('track_id','annotation_id','frame','source_frame','category','bbox_x','bbox_y','bbox_w','bbox_h',
                                        'diagnostic_box_max','raw_frames_path','raw_array_frame_index','source_file_name','source_image_id')}))
    sampling=dict(status='FROZEN_BEFORE_IMAGE_INSPECTION',population='all 4391 geometric-small object-frames with all available diagnostic windows failing actual matching',
                  selection='per sequence x response layer: ascending track_id first min(4,n); within that track/layer source-frame-sorted item floor(n/2); no image access in this script',
                  response_layers='max bbox response across available diagnostic windows >=0.5 versus <0.5; stratification is explicit and not a random occurrence estimate',
                  all_failure_population=4391,strata=strata,samples=sampled,inputs_sha256=ih,script_sha256=sha(Path(__file__)))
    js(D/'SAMPLE_SELECTION.json',sampling)
    csvwrite(D/'VISUAL_SAMPLE_INDEX.csv',sampled)
    for name,rr in [('OBJECT_CONDITIONS.csv',rows),('WINDOW_CONDITIONS.csv',window_rows),('TRACK_CONDITIONS.csv',track_rows),('OBSERVATION_EVENTS.csv',events),
                    ('GROUP_COUNTS.csv',summary_groups),('CATEGORICAL_CONDITIONS.csv',categorical),('CONTINUOUS_CONDITIONS.csv',continuous),('STABLE_CONTROLS.csv',matched)]: csvwrite(D/name,rr)
    matched_summary=dict(candidate_pool=len(pool),case_objects=len(matched),matched=sum(r['matched'] for r in matched),unmatched=sum(not r['matched'] for r in matched),
                         unique_controls=len(reuse),max_reuse=max(reuse.values()),same_track=sum(r['same_track']==1 for r in matched),
                         exact_fields=['sequence','category','scale_bin'],continuous_fields=list(match_fields),iqr_scales=scales,
                         tie_break=['track_id','source_frame','annotation_id'],replacement=True,caliper='none; distance is descriptive, no causal matching claim')
    js(D/'STABLE_CONTROL_PROTOCOL.json',matched_summary)
    outnames=['OBJECT_CONDITIONS.csv','WINDOW_CONDITIONS.csv','TRACK_CONDITIONS.csv','OBSERVATION_EVENTS.csv','GROUP_COUNTS.csv','CATEGORICAL_CONDITIONS.csv',
              'CONTINUOUS_CONDITIONS.csv','STABLE_CONTROLS.csv','SAMPLE_SELECTION.json','VISUAL_SAMPLE_INDEX.csv','STABLE_CONTROL_PROTOCOL.json']
    assert ih=={str(p):sha(p) for p in inputs}
    summary=dict(status='COMPLETE',objects=5200,group_counts=counts,both_systems_failure=4426,group_summary=summary_groups,window_object_arrangement_rows=len(window_rows),
                 input_hashes=ih,annotation_sources=annotation_sources,script_sha256=sha(Path(__file__)),sample_cases=len(sampled),stable_control_summary=matched_summary,
                 outputs_sha256={n:sha(D/n) for n in outnames},interpretation='Full descriptive census on two reused development clips. Label presence is not visibility; image-coordinate displacement is not physical motion. Conditional outcome groups, not independent targets or causal estimates.')
    js(D/'SUMMARY.json',summary)
    print(json.dumps({k:summary[k] for k in ('status','objects','group_counts','window_object_arrangement_rows','sample_cases','stable_control_summary')},ensure_ascii=False))

if __name__=='__main__': main()
