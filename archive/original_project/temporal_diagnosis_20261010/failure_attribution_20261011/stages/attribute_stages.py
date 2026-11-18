#!/usr/bin/env python3
"""Frozen-output error-stage audit. Never runs inference or alters prior artifacts."""
import csv
import hashlib
import importlib.util
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parent
F=OUT.parent; Q=F.parent; V=Q/'problem_validity_20261011'; P=Q/'stage1_goal'
SEQS=['DJI_0026_2','DJI_0067_2']
PLANS={'official':[0,36,72,108,120],'symmetric':[0,24,60,96,120]}
SOURCES={}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    SOURCES[str(p)]=h.hexdigest();return SOURCES[str(p)]

def readcsv(p):
    sha(p)
    with p.open() as f:return list(csv.DictReader(f))

def readjson(p):sha(p);return json.loads(p.read_text())

def module(p,name):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def dumpcsv(name,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def j(v):return json.dumps(v,separators=(',',':'),ensure_ascii=False)

def key(r):return r['sequence'],int(r['frame']),int(r['annotation_id'])

def ids(x):return [int(a) for a in x.split('|') if a]

def stage(row,anymatch):
    if row['matched']:return 'final_match_with_window_match' if anymatch else 'final_match_without_window_match'
    if anymatch:
        return 'window_match_lost_at_fused_peak_extraction' if not row['eligible_components'] else 'window_match_lost_at_assignment'
    if row['box_max'] is None:return 'no_window_match_no_box_pixel_center'
    if row['box_max']<.5:return 'no_window_match_low_response'
    return 'no_window_match_high_response_no_peak' if not row['eligible_components'] else 'no_window_match_assignment_competition'

def main():
    if (OUT/'SUMMARY.json').exists():raise FileExistsError('Already complete')
    begun=time.perf_counter();sha(__file__);sha(F/'PROTOCOL.md')
    scorer=module(P/'pilot/score_pilot.py','score');helper,hpath=scorer.helper();sha(hpath)
    source=[r for r in readcsv(V/'OBJECTS.csv') if r['sequence'] in SEQS and r['small_geometry']=='1']
    assert len(source)==5200
    controls={key(r):r for r in readcsv(V/'control_summary/OBJECTS.csv') if r['mode']=='symmetric5_probmax' and r['small_geometry']=='1'}
    wins=defaultdict(dict)
    for r in readcsv(V/'WINDOWS.csv'):
        if r['sequence'] in SEQS:wins[key(r)][int(r['window_start'])]=r
    all_objects=[];all_windows=[];focus_components=[];maps=[];checks=[];final_background=[]
    for seq in SEQS:
        annpath=P/'data'/seq/'annotations.json';anno=readjson(annpath)
        gt,meta,offsets,_=scorer.gt_data(annpath,helper)
        sourceframes=[int(Path(im['file_name']).stem) for im in anno['images']]
        rows=[r for r in source if r['sequence']==seq]
        focus={key(r) for r in rows if controls[key(r)]['gain']=='1' or controls[key(r)]['loss']=='1' or r['system_case']=='original_window_match_lost_by_fusion'}
        focus_frames={k[1] for k in focus}
        # Cache each immutable array once. This avoids repeated ZIP decompression.
        probs={}
        for start in range(0,121,12):
            path=P/'predictions/official'/seq/'windows'/f'window_{start:03d}.npz';sha(path)
            with np.load(path,allow_pickle=False) as z:
                assert int(z['start'])==start and int(z['end'])==start+40
                probs[start]=z['probabilities']
        fused={}
        for plan,path in [('official',P/'predictions/official'/seq/'official_fused.npz'),('symmetric',V/'simple_controls/symmetric5_probmax'/seq/'fused.npz')]:
            sha(path)
            with np.load(path,allow_pickle=False) as z:fused[plan]=z['probabilities'];assert z['starts'].tolist()==PLANS[plan]
            recon=np.zeros_like(fused[plan])
            for s in PLANS[plan]:recon[s:s+40]=np.maximum(recon[s:s+40],probs[s])
            assert np.array_equal(fused[plan],recon)
            del recon
        byframe=defaultdict(list)
        for r in rows:byframe[int(r['frame'])].append(r)
        for frame in range(160):
            # Recompute all final matching states, not only selected extreme examples.
            direct={}; window_direct={}
            for plan in PLANS:
                rr,stats,cc=helper.match_frame(fused[plan][frame],gt[frame],.5)
                label=None
                if frame in focus_frames:_,label=helper.components(fused[plan][frame],.5)
                direct[plan]=({r['annotation_id']:r for r in rr},{c['component_id']:c for c in cc},label)
                final_background.append(dict(sequence=seq,frame=frame,source_frame=sourceframes[frame],plan=plan,**stats))
            if frame in focus_frames:
                for s in range(0,121,12):
                    if s<=frame<s+40:
                        rr,stats,cc=helper.match_frame(probs[s][frame-s],gt[frame],.5)
                        _,labels=helper.components(probs[s][frame-s],.5)
                        window_direct[s]=({r['annotation_id']:r for r in rr},{c['component_id']:c for c in cc},labels)
            for original in byframe[frame]:
                k=key(original);aid=k[2];ctrl=controls[k];ww=wins[k];rec=dict(original)
                rec.update(cohort=('gain' if int(ctrl['gain']) else 'loss' if int(ctrl['loss']) else 'both_success' if int(ctrl['official_matched']) else 'both_failure'),
                           all11_failed=int(not int(original['any11_matched'])),focus=int(k in focus))
                shared=[s for s in sorted(set(PLANS['official'])&set(PLANS['symmetric'])) if s in ww]
                rec['shared_starts']=j(shared);rec['shared_output_consistency']='same_immutable_array'
                for s,r in sorted(ww.items()):
                    wr=dict(r,window_start_source=sourceframes[s],window_end_source=sourceframes[s+39],window_end_exclusive=s+40,
                            symmetric5=int(s in PLANS['symmetric']),shared_window=int(s in shared))
                    all_windows.append(wr)
                for plan in PLANS:
                    starts=[s for s in PLANS[plan] if s in ww];success=[s for s in starts if int(ww[s]['matched'])]
                    rr,cc,labels=direct[plan];r=rr[aid]
                    expected=int(ctrl['official_matched'] if plan=='official' else ctrl['control_matched']);assert r['matched']==expected
                    if plan=='official':assert abs(r['box_max']-float(original['fused_box_max']))<1e-7
                    rec.update({plan+'_'+name:value for name,value in dict(
                        starts=j(starts),slots=j([frame-s for s in starts]),success_starts=j(success),coverage=len(starts),
                        earliest_access_local=min(starts),latest_access_local=max(starts)+39,
                        earliest_access_source=sourceframes[min(starts)],latest_access_source=sourceframes[max(starts)+39],
                        any_window_match=int(bool(success)),window_max_box_response=max(float(ww[s]['box_max']) for s in starts),
                        matched=r['matched'],fused_box_max=r['box_max'],fused_positive_components_in_box=r['intersecting_components'],
                        eligible_peak_count=r['eligible_components'],matched_component=r['component_id'],peak_x=r['peak_x'],peak_y=r['peak_y'],
                        stage=stage(r,bool(success))).items()})
                    if k in focus:
                        x0,y0,x1,y1=next(g['crop'] for g in gt[frame] if g['id']==aid)
                        intersects=[int(x) for x in np.unique(labels[y0:y1,x0:x1]) if x]
                        for cid in intersects:
                            c=cc[cid]
                            focus_components.append(dict(sequence=seq,frame=frame,source_frame=sourceframes[frame],annotation_id=aid,track_id=original['track_id'],
                                cohort=rec['cohort'],plan=plan,selected_component=int(r['component_id']==cid),**c))
                        for s in starts:
                            sw,sc,sl=window_direct[s];sr=sw[aid]
                            assert sr['matched']==int(ww[s]['matched']) and abs(sr['box_max']-float(ww[s]['box_max']))<1e-7
                            if not sr['matched']:continue
                            cid=sr['component_id'];smask=sl==cid;dest=np.unique(labels[smask]);assert len(dest)==1 and dest[0]>0
                            fid=int(dest[0]);fc=cc[fid];fc_mask=labels==fid
                            same_src=[int(c) for c in np.unique(sl[fc_mask]) if c]
                            assert cid in same_src
                            sy=int(sr['peak_y']-.5);sx=int(sr['peak_x']-.5);fy=int(fc['peak_y']-.5);fx=int(fc['peak_x']-.5)
                            bx,by,bw,bh=map(float,[original['bbox_x'],original['bbox_y'],original['bbox_w'],original['bbox_h']])
                            peakinside=int(bx<=fc['peak_x']<bx+bw and by<=fc['peak_y']<by+bh)
                            contributing=[a for a in starts if probs[a][frame-a,fy,fx]==fused[plan][frame,fy,fx]]
                            maps.append(dict(sequence=seq,frame=frame,source_frame=sourceframes[frame],annotation_id=aid,track_id=original['track_id'],
                                cohort=rec['cohort'],original_fusion_loss=int(original['system_case']=='original_window_match_lost_by_fusion'),plan=plan,window_start=s,
                                source_component=cid,source_area=sc[cid]['area'],source_peak_x=sr['peak_x'],source_peak_y=sr['peak_y'],source_peak_probability=sr['peak_probability'],
                                source_component_ids_merged=j(same_src),source_components_in_fused_component=len(same_src),
                                direct_component_merge=int(len(same_src)>1),source_component_foreground_preserved=1,
                                fused_component=fid,fused_area=fc['area'],new_pixels_connected_relative_to_source=int((fc_mask&(sl==0)).sum()),
                                fused_peak_x=fc['peak_x'],fused_peak_y=fc['peak_y'],fused_peak_probability=fc['peak_probability'],fused_peak_in_target_box=peakinside,
                                fused_peak_changed=int(fy!=sy or fx!=sx),source_score_at_fused_peak=float(probs[s][frame-s,fy,fx]),fused_score_at_source_peak=float(fused[plan][frame,sy,sx]),
                                fused_peak_tied_with_source_peak=int(fused[plan][frame,sy,sx]==fc['peak_probability']),
                                fused_peak_contributing_windows=j(contributing),fused_assigned_track_id=fc['assigned_track_id'],fused_eligible_tracks=fc['eligible_track_ids'],
                                final_matched=r['matched'],final_eligible_peak_count=r['eligible_components']))
                for plan,other in [('official','symmetric'),('symmetric','official')]:
                    success=json.loads(rec[plan+'_success_starts']);retained=[s for s in success if s in PLANS[other]]
                    rec[plan+'_success_starts_removed_by_'+other]=j([s for s in success if s not in PLANS[other]])
                    rec[plan+'_success_all_removed_by_'+other]=int(bool(success) and not retained)
                if rec['cohort']=='loss':
                    rec['transition_stage']=('symmetric_fusion_lost_available_match' if rec['symmetric_any_window_match'] else
                        'original_success_windows_removed_no_replacement' if rec['official_success_all_removed_by_symmetric'] else
                        'original_fusion_only_success_lost')
                elif rec['cohort']=='gain':
                    rec['transition_stage']=('official_fusion_lost_available_match' if rec['official_any_window_match'] else
                        'new_success_windows_added' if rec['symmetric_any_window_match'] else 'symmetric_fusion_only_success')
                else:rec['transition_stage']=rec['cohort']
                all_objects.append(rec)
            if frame%40==0:print(j(dict(sequence=seq,frame=frame,seconds=time.perf_counter()-begun)),flush=True)
        checks.append(dict(sequence=seq,small_objects=len(rows),focus_objects=len(focus),focus_frames=len(focus_frames),
            final_matching_recomputed_all_frames=160,focus_window_matching_verified=True,probability_fusion_exactly_reconstructed=True))
        del probs,fused
    assert Counter(r['cohort'] for r in all_objects)==Counter({'gain':74,'loss':54,'both_success':646,'both_failure':4426})
    assert sum(r['all11_failed'] for r in all_objects)==4391
    assert sum(r['focus'] for r in all_objects)==132
    # Consecutive source frames within a track and status define observations, not independent trials.
    events=[]
    for cohort,test in [('gain',lambda r:r['cohort']=='gain'),('loss',lambda r:r['cohort']=='loss'),
        ('both_success',lambda r:r['cohort']=='both_success'),('both_failure',lambda r:r['cohort']=='both_failure'),
        ('all11_failed',lambda r:r['all11_failed']),('original_fusion_loss',lambda r:r['system_case']=='original_window_match_lost_by_fusion')]:
        groups=defaultdict(list)
        for r in all_objects:
            if test(r):groups[(r['sequence'],r['track_id'])].append(r)
        for (seq,tid),rr in groups.items():
            rr.sort(key=lambda r:(int(r['source_frame']),int(r['annotation_id'])));runs=[]
            for r in rr:
                if not runs or int(r['source_frame'])!=int(runs[-1][-1]['source_frame'])+1:runs.append([])
                runs[-1].append(r)
            for run in runs:events.append(dict(cohort=cohort,sequence=seq,track_id=tid,source_start=int(run[0]['source_frame']),source_end=int(run[-1]['source_frame']),
                objects=len(run),annotation_ids=j([int(r['annotation_id']) for r in run]),transition_stages=j(dict(Counter(r['transition_stage'] for r in run)))))
    groups=[]
    for cohort in ['gain','loss','both_success','both_failure','all11_failed','original_fusion_loss']:
        test=(lambda r:r['all11_failed']) if cohort=='all11_failed' else (lambda r:r['system_case']=='original_window_match_lost_by_fusion') if cohort=='original_fusion_loss' else (lambda r:r['cohort']==cohort)
        rr=[r for r in all_objects if test(r)]
        groups.append(dict(cohort=cohort,objects=len(rr),tracks=len({(r['sequence'],r['track_id']) for r in rr}),sequences=len({r['sequence'] for r in rr}),
            events=sum(e['cohort']==cohort for e in events),official_stages=dict(Counter(r['official_stage'] for r in rr)),symmetric_stages=dict(Counter(r['symmetric_stage'] for r in rr)),
            transition_stages=dict(Counter(r['transition_stage'] for r in rr))))
    fusion_losses=[]
    for r in all_objects:
        for plan in PLANS:
            if r[plan+'_any_window_match'] and not r[plan+'_matched']:
                mm=[m for m in maps if m['sequence']==r['sequence'] and m['annotation_id']==int(r['annotation_id']) and m['plan']==plan]
                fusion_losses.append(dict(sequence=r['sequence'],source_frame=r['source_frame'],annotation_id=r['annotation_id'],track_id=r['track_id'],cohort=r['cohort'],plan=plan,
                    direct_trace_available=int(bool(mm)),successful_windows=len(mm),any_direct_merge=int(any(m['direct_component_merge'] for m in mm)),
                    all_destination_peaks_outside=int(bool(mm) and all(not m['fused_peak_in_target_box'] for m in mm)),eligible_peak_count=r[plan+'_eligible_peak_count'],
                    fused_box_max=r[plan+'_fused_box_max'],source_trace=j(mm)))
    # Full audit must include every such small-object loss, not silently limit to selected examples.
    assert all(r['direct_trace_available'] for r in fusion_losses)
    dumpcsv('ERROR_STAGE_TABLE.csv',all_objects);dumpcsv('WINDOW_ACCESS.csv',all_windows)
    dumpcsv('FOCUS_COMPONENTS.csv',focus_components);dumpcsv('COMPONENT_MAPPING.csv',maps)
    dumpcsv('FUSION_LOSSES.csv',fusion_losses);dumpcsv('OBSERVATION_EVENTS.csv',events);dumpcsv('FRAME_COUNTS.csv',final_background)
    report=dict(status='COMPLETE',threshold=.5,objects=5200,focus_objects=132,plans=PLANS,groups=groups,checks=checks,
        fusion_losses=len(fusion_losses),fusion_loss_direct_merge=sum(r['any_direct_merge'] for r in fusion_losses),
        fusion_loss_all_destination_peaks_outside=sum(r['all_destination_peaks_outside'] for r in fusion_losses),
        elapsed_seconds=time.perf_counter()-begun,sources=SOURCES,
        limitations=['Only two previously used development clips','Events are consecutive eligible annotated frames, not independent physical events',
                    'Image-coordinate box labels do not establish visibility, occlusion, or motion cause','All metrics are diagnostic peak-in-box matching, not official Pd/Fa'])
    (OUT/'SUMMARY.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
