#!/usr/bin/env python3
"""Fixed 800-frame training-object learnability, using original component matcher.

No new weak qualification: membership is the exact union of the old v2 cohort
objects. Raw full-frame labels are used by scoring only, never by inference.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import time

E = Path(__file__).resolve().parents[1]
ROOT = E.parents[1]
Q = ROOT / 'archive/infrared_problem_screen_20261009/temporal_diagnosis_20261010'
DATA = Q / 'stage1_goal/data'
PRIOR = ROOT / 'experiments/c_effective_update_v2/PAIR_PLAN.json'
PLAN_PATH = E / 'TRAIN_EVAL_PLAN.json'
OUT = E / 'evaluation/train'
PLANS = {'O': [0,36,72,108,120], 'S': [0,24,60,96,120]}
FIELDS = ['gt_count','matched_count','component_count','unmatched_components','positive_pixels','positive_pixels_outside_boxes']
CONDITIONS = ['release','F_init','ORIGINAL','REPLAY']


def read(path): return json.loads(Path(path).read_text())
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def csvwrite(path,rows,fields=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=fields or list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def build_plan():
    if PLAN_PATH.exists(): raise FileExistsError(PLAN_PATH)
    old=read(PRIOR); tracks=set(); objects={}; sources={str(PRIOR):sha(PRIOR)}
    for entry in old['entries'].values():
        cp=Path(entry['cohort_path']);assert sha(cp)==entry['cohort_sha256']
        c=read(cp); sources[str(cp)]=sha(cp);weak=set(c['weak_target_indices'])
        for row in c['target_objects']:
            if row['target_index'] not in weak: continue
            tracks.add((row['sequence'],int(row['video_id']),int(row['track_id'])))
            key=(row['sequence'],int(row['source_frame']),int(row['annotation_id']))
            value={k:row[k] for k in ['sequence','video_id','track_id','annotation_id','frame','source_frame','bbox']}
            if key in objects: assert value==objects[key]
            objects[key]=value
    assert len(tracks)==39 and len(objects)==2066
    sequences=sorted({x[0] for x in tracks});assert len(sequences)==5
    for sequence in sequences:
        ap=DATA/sequence/'annotations.json';rp=ap.with_name('raw_frames.npy');a=read(ap)
        assert len(a['images'])==160
        assert all(im['height']==512 and im['width']==640 for im in a['images'])
        lookup={int(x['id']):x for x in a['annotations']};frames={im['id']:(i,int(Path(im['file_name']).stem)) for i,im in enumerate(a['images'])}
        for o in objects.values():
            if o['sequence']!=sequence:continue
            ann=lookup[o['annotation_id']]
            assert int(ann['track_id'])==o['track_id'] and list(ann['bbox'])==o['bbox']
            assert frames[ann['image_id']]==(o['frame'],o['source_frame'])
        sources[str(ap)]=sha(ap);sources[str(rp)]=sha(rp)
    scorer=Q/'stage1_goal/pilot/score_pilot.py';matcher=Q/'analysis/analyze.py'
    sources.update({str(p):sha(p) for p in [scorer,matcher]})
    plan=dict(status='FROZEN_RANGE_BEFORE_NEW_TRAINING',conditions=CONDITIONS,
        sequences=sequences,frames_per_sequence=160,total_effective_frames=800,raw_shape=[160,512,640],
        plans=PLANS,diagnostic_starts=sorted(set(sum(PLANS.values(),[]))),threshold=.5,
        known_weak_tracks=[dict(sequence=s,video_id=v,track_id=t) for s,v,t in sorted(tracks)],
        confirmed_weak_object_frames=[objects[k] for k in sorted(objects)],
        confirmed_weak_object_frames_count=len(objects),known_weak_track_count=len(tracks),
        source_sha256=sources,
        selection='All 160 already-materialized frames of every source containing any of the exact 39 v2 confirmed weak tracks. No inference score or new training outcome used.',
        scopes={'confirmed_weak_frames':'2066 unique source-frame-annotation IDs confirmed weak under historical release cropped-window contexts. This does not assert full-frame release matching failure.',
            'known_weak_tracks_all_frames':'All original matching-valid annotated frames for the same 39 video-scoped tracks inside the fixed 800-frame range; continuity uses this full range, not sparse weak rows.',
            'other_valid_gt':'All other original matching-valid objects, for retention; no size filtering.',
            'all_valid_gt':'Every original matching-valid object in the same frames.'},
        continuity='Original observed consecutive local AND source frame indices. Gaps break runs. No interpolation or visibility invention.',
        background='Use all 800 full frames for every condition and arrangement. Report total U, outside-GT positive area, unmatched component area and per-frame costs; never apply the old 4015 budget.',
        comparisons=['ORIGINAL_vs_F_init','REPLAY_vs_F_init','REPLAY_vs_ORIGINAL','all_vs_release'],
        inference='Original 40-frame forward and original five-window max fusion O/S separately; complete 512x640 image via original verified tiling, no label input.',
        limits=['Training-object learnability only; these sources were already seen in F training, not held-out tests.',
            'O/S share source frames and must not be added as independent examples.',
            '39 track identities are confirmed; low-response membership is from cropped historical contexts. Full-frame baseline matches are measured afresh.',
            'Annotation visibility/completeness limitations remain. M/U are diagnostics, not official Pd/Fa.'])
    save(PLAN_PATH,plan);print({'status':'PLAN_CREATED','sources':sequences,'tracks':len(tracks),'confirmed_objects':len(objects),'frames':800},flush=True)


def verify_plan():
    plan=read(PLAN_PATH)
    for path,expected in plan['source_sha256'].items(): assert sha(path)==expected,('eval source changed',path)
    assert plan['plans']==PLANS and plan['total_effective_frames']==800
    return plan


def verify_freeze():
    for name,expected in read(E/'PROTOCOL_FREEZE.json')['files'].items():
        assert sha(E/name)==expected,('Frozen artifact changed',name)


def check_plan():
    """CPU-only range/identity check with the actual unchanged GT preparation."""
    plan=verify_plan();scorer=module('replay_training_gt_check',Q/'stage1_goal/pilot/score_pilot.py')
    helper,_=scorer.helper();weak={(r['sequence'],r['frame'],r['annotation_id']) for r in plan['confirmed_weak_object_frames']}
    tracks={(r['sequence'],r['track_id']) for r in plan['known_weak_tracks']}
    valid=set();seen_tracks=set();counts=[]
    for sequence in plan['sequences']:
        gt,meta,offsets,excluded=scorer.gt_data(DATA/sequence/'annotations.json',helper)
        assert len(gt)==160 and len(offsets)==161
        valid.update((sequence,m['frame'],m['annotation_id']) for m in meta)
        seen_tracks.update((sequence,m['track_id']) for m in meta if (sequence,m['track_id']) in tracks)
        counts.append(dict(sequence=sequence,frames=len(gt),all_valid_gt=len(meta),
            known_weak_track_frames=sum((sequence,m['track_id']) in tracks for m in meta),
            confirmed_weak_frames=sum((sequence,m['frame'],m['annotation_id']) in weak for m in meta),
            excluded_annotations=len(excluded)))
    assert weak<=valid and seen_tracks==tracks and len(weak)==2066
    rows=[dict(frame=f,source_frame=sf,hit=h) for f,sf,h in [(0,0,0),(1,1,1),(2,2,0),(3,7,0),(4,8,1),(6,10,1)]]
    assert [len(r) for r in runs(rows,'hit',False)]==[1,1,1]
    assert [len(r) for r in runs(rows,'hit',True)]==[1,1,1]
    assert contiguous(rows[0],rows[1]) and not contiguous(rows[2],rows[3]) and not contiguous(rows[4],rows[5])
    save(E/'evaluation/TRAIN_EVAL_CHECK.json',dict(status='PASS',plan_sha256=sha(PLAN_PATH),source_sha256=sha(__file__),
        all_confirmed_weak_frames_survive_original_gt_preparation=True,all_39_tracks_present=True,
        continuity_local_and_source_gaps_break_runs=True,counts=counts,total_frames=sum(r['frames'] for r in counts),
        all_valid_gt=sum(r['all_valid_gt'] for r in counts),known_weak_track_frames=sum(r['known_weak_track_frames'] for r in counts)))
    print({'status':'TRAIN_EVAL_CHECK_PASS','counts':counts},flush=True)


def score(condition,sequence):
    import numpy as np
    verify_freeze()
    plan=verify_plan();assert condition in CONDITIONS and sequence in plan['sequences']
    pred=E/'predictions/train'/condition/sequence;dst=OUT/'scored'/condition/sequence
    if (dst/'COMPLETE.json').exists():raise FileExistsError(dst)
    receipt=read(pred/'COMPLETE.json');assert receipt['status']=='COMPLETE'
    assert receipt['condition']==condition and receipt['sequence']==sequence
    assert receipt['train_eval_plan_sha256']==sha(PLAN_PATH)
    assert receipt['raw_sha256']==sha(DATA/sequence/'raw_frames.npy')
    assert receipt['official_starts']==PLANS['O'] and receipt['symmetric_starts']==PLANS['S']
    assert receipt['model_labels_access'] is False
    assert receipt['checkpoint_sha256']==sha(receipt['checkpoint'])
    assert receipt['protocol_sha256']==sha(E/'PROTOCOL_FREEZE.json')
    assert receipt['inference_source_sha256']==sha(E/'code/train_infer.py')
    assert receipt['runtime_source_sha256']==sha(E/'code/runtime.py')
    assert receipt['annotations_sha256']==sha(DATA/sequence/'annotations.json')
    assert receipt['tiling_validation']=='reuse_verified_original_tiled_logits_core128_halo16'
    assert receipt['all_model_state_unchanged_during_inference'] and receipt['bn_unchanged']
    scorer_path=Q/'stage1_goal/pilot/score_pilot.py';scorer=module('replay_training_original_gt',scorer_path)
    h,hp=scorer.helper();gt,meta,offsets,excluded=scorer.gt_data(DATA/sequence/'annotations.json',h)
    assert len(gt)==160
    n=len(meta);matched=np.zeros((2,n),np.uint8);box_max=np.zeros((2,n),np.float32)
    peak_x=np.full((2,n),np.nan,np.float32);peak_y=peak_x.copy();component_area=np.zeros((2,n),np.int32)
    counts=np.zeros((2,160,6),np.int64);coverage=np.zeros((2,160),np.int16)
    source_frames=[int(Path(im['file_name']).stem) for im in read(DATA/sequence/'annotations.json')['images']]
    arrays={};frame_rows=[];began=time.perf_counter()
    for pi,(arrangement,starts) in enumerate(PLANS.items()):
        fused=np.zeros((160,512,640),np.float32)
        for start in starts:
            wp=pred/'windows'/f'window_{start:03d}.npz';assert sha(wp)==receipt['window_sha256'][wp.name]
            with np.load(wp,allow_pickle=False) as z:
                pp=z['probabilities'];assert pp.shape==(40,512,640) and pp.dtype==np.float32 and np.isfinite(pp).all()
                assert int(z['start'])==start and int(z['end'])==start+40
                fused[start:start+40]=np.maximum(fused[start:start+40],pp);coverage[pi,start:start+40]+=1
        assert np.all(coverage[pi]>0) and int(coverage[pi].sum())==200
        if arrangement=='O':
            fp=pred/'official_fused.npz';assert sha(fp)==receipt['fused_sha256']
            with np.load(fp,allow_pickle=False) as z:
                assert np.array_equal(z['probabilities'],fused)
                assert np.array_equal(z['coverage'],coverage[pi])
        aa=[];uu=[];ao=[0]
        for t in range(160):
            rows,stats,components=h.match_frame(fused[t],gt[t],.5)
            assert len(rows)==offsets[t+1]-offsets[t]
            for j,r in enumerate(rows,int(offsets[t])):
                assert (r['annotation_id'],r['track_id'])==(meta[j]['annotation_id'],meta[j]['track_id'])
                matched[pi,j]=r['matched'];box_max[pi,j]=r['box_max']
                if r['matched']:
                    peak_x[pi,j]=r['peak_x'];peak_y[pi,j]=r['peak_y'];component_area[pi,j]=r['component_area']
            counts[pi,t]=[stats[k] for k in FIELDS]
            areas=[int(c['area']) for c in components];unmatched=[int(c['unmatched_component']) for c in components]
            assert sum(areas)==stats['positive_pixels'] and sum(unmatched)==stats['unmatched_components']
            aa.extend(areas);uu.extend(unmatched);ao.append(len(aa))
            frame_rows.append(dict(condition=condition,sequence=sequence,arrangement=arrangement,frame=t,
                source_frame=source_frames[t],
                **{k:int(stats[k]) for k in FIELDS},
                unmatched_component_area=sum(a for a,u in zip(areas,unmatched) if u)))
        arrays[arrangement+'_areas']=np.array(aa,np.int32);arrays[arrangement+'_unmatched']=np.array(uu,np.uint8);arrays[arrangement+'_offsets']=np.array(ao,np.int64)
        print(dict(condition=condition,sequence=sequence,arrangement=arrangement,M=int(matched[pi].sum()),U=int(counts[pi,:,3].sum())),flush=True)
    dst.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dst/'scores.npz',matched=matched,box_max=box_max,peak_x=peak_x,peak_y=peak_y,component_area=component_area,
        frame_counts=counts,coverage=coverage,offsets=offsets,plans=np.array(list(PLANS)),threshold=np.array(.5))
    np.savez_compressed(dst/'component_areas.npz',**arrays);csvwrite(dst/'FRAME_COUNTS.csv',frame_rows)
    save(dst/'COMPLETE.json',dict(status='COMPLETE',condition=condition,sequence=sequence,objects=meta,excluded_annotations=excluded,
        plans=PLANS,threshold=.5,total_effective_frames=160,train_eval_plan_sha256=sha(PLAN_PATH),prediction_receipt_sha256=sha(pred/'COMPLETE.json'),
        source_sha256={str(p):sha(p) for p in [Path(__file__),scorer_path,hp]},
        artifact_sha256={p.name:sha(p) for p in [dst/'scores.npz',dst/'component_areas.npz',dst/'FRAME_COUNTS.csv']},elapsed_seconds=time.perf_counter()-began))


def contiguous(left,right):
    return int(right['frame'])==int(left['frame'])+1 and int(right['source_frame'])==int(left['source_frame'])+1

def runs(rows,key,value):
    result=[];active=[];previous=None
    for row in rows:
        if previous is not None and not contiguous(previous,row):
            if active:result.append(active)
            active=[]
        if bool(row[key])==value:active.append(row)
        else:
            if active:result.append(active)
            active=[]
        previous=row
    if active:result.append(active)
    return result


def summarize():
    import numpy as np
    verify_freeze()
    plan=verify_plan();conditions=CONDITIONS;seqs=plan['sequences']
    weak={(o['sequence'],int(o['frame']),int(o['annotation_id'])) for o in plan['confirmed_weak_object_frames']}
    trackkeys={(o['sequence'],int(o['track_id'])) for o in plan['known_weak_tracks']}
    objects=[];record={};areas={};sources={str(PLAN_PATH):sha(PLAN_PATH)};allframes=[]
    for sequence in seqs:
        metadata=None
        for condition in conditions:
            dst=OUT/'scored'/condition/sequence;cp=dst/'COMPLETE.json';r=read(cp)
            assert r['status']=='COMPLETE' and r['train_eval_plan_sha256']==sha(PLAN_PATH)
            assert r['condition']==condition and r['sequence']==sequence and r['threshold']==.5 and r['plans']==PLANS
            assert r['prediction_receipt_sha256']==sha(E/'predictions/train'/condition/sequence/'COMPLETE.json')
            if metadata is not None:assert metadata==r['objects']
            metadata=r['objects']
            for name,digest in r['artifact_sha256'].items():assert sha(dst/name)==digest
            for path,digest in r['source_sha256'].items():assert sha(path)==digest
            sources[str(cp)]=sha(cp)
            with np.load(dst/'scores.npz',allow_pickle=False) as z:record[condition,sequence]={k:z[k] for k in z.files}
            assert record[condition,sequence]['plans'].tolist()==list(PLANS)
            assert float(record[condition,sequence]['threshold'])==.5
            assert record[condition,sequence]['matched'].shape==(2,len(metadata))
            assert record[condition,sequence]['frame_counts'].shape==(2,160,6)
            with np.load(dst/'component_areas.npz',allow_pickle=False) as z:areas[condition,sequence]={k:z[k] for k in z.files}
            with (dst/'FRAME_COUNTS.csv').open() as f:allframes.extend(list(csv.DictReader(f)))
        for j,m in enumerate(metadata):
            for pi,arrangement in enumerate(PLANS):
                row=dict(sequence=sequence,arrangement=arrangement,**{k:m[k] for k in ['frame','source_frame','annotation_id','track_id','category_id','strict_small']},
                    bbox=json.dumps(m['bbox'],separators=(',',':')),confirmed_weak_frame=int((sequence,m['frame'],m['annotation_id']) in weak),
                    known_weak_track=int((sequence,m['track_id']) in trackkeys))
                for c in conditions:
                    row[c+'_matched']=int(record[c,sequence]['matched'][pi,j]);row[c+'_box_max']=float(record[c,sequence]['box_max'][pi,j])
                objects.append(row)
    assert sum(r['confirmed_weak_frame'] for r in objects)==2*len(weak)
    assert {(r['sequence'],r['track_id']) for r in objects if r['known_weak_track']}==trackkeys
    summary=[];costs=[];protection=[];concentration=[]
    for pi,arrangement in enumerate(PLANS):
        for group,selected_sequences in [('pooled',seqs)]+[(s,[s]) for s in seqs]:
            subset=[r for r in objects if r['arrangement']==arrangement and r['sequence'] in selected_sequences]
            frames=160*len(selected_sequences)
            for c in conditions:
                cnt=sum((record[c,s]['frame_counts'][pi].sum(0) for s in selected_sequences),np.zeros(6,np.int64))
                aa=np.concatenate([areas[c,s][arrangement+'_areas'] for s in selected_sequences]);um=np.concatenate([areas[c,s][arrangement+'_unmatched'] for s in selected_sequences]).astype(bool)
                bg=dict(condition=c,arrangement=arrangement,group=group,effective_frames=frames,**dict(zip(FIELDS,map(int,cnt))),
                    unmatched_component_area=int(aa[um].sum()),unmatched_component_area_max=int(aa[um].max()) if um.any() else 0,
                    unmatched_component_area_mean=float(aa[um].mean()) if um.any() else 0.)
                for key in ['unmatched_components','positive_pixels_outside_boxes','unmatched_component_area']:bg[key+'_per_frame']=bg[key]/frames
                for ref in ['release','F_init','ORIGINAL']:
                    rcnt=sum((record[ref,s]['frame_counts'][pi].sum(0) for s in selected_sequences),np.zeros(6,np.int64))
                    rarea=sum(int(areas[ref,s][arrangement+'_areas'][areas[ref,s][arrangement+'_unmatched'].astype(bool)].sum()) for s in selected_sequences)
                    bg['delta_U_vs_'+ref]=int(cnt[3]-rcnt[3]);bg['delta_outside_area_vs_'+ref]=int(cnt[5]-rcnt[5]);bg['delta_unmatched_area_vs_'+ref]=int(aa[um].sum())-rarea
                costs.append(bg)
                scopes={'confirmed_weak_frames':[r for r in subset if r['confirmed_weak_frame']],
                    'known_weak_tracks_all_frames':[r for r in subset if r['known_weak_track']],
                    'other_valid_gt':[r for r in subset if not r['known_weak_track']], 'all_valid_gt':subset}
                for scope,rows in scopes.items():
                    item=dict(condition=c,arrangement=arrangement,group=group,scope=scope,effective_frames=frames,
                        target_frames=len(rows),tracks=len({(r['sequence'],r['track_id']) for r in rows}),M=sum(r[c+'_matched'] for r in rows))
                    for ref in ['release','F_init','ORIGINAL']:
                        item['G_vs_'+ref]=sum(r[c+'_matched'] and not r[ref+'_matched'] for r in rows)
                        item['L_vs_'+ref]=sum(not r[c+'_matched'] and r[ref+'_matched'] for r in rows)
                        item['net_vs_'+ref]=item['G_vs_'+ref]-item['L_vs_'+ref]
                        item[ref+'_M']=sum(r[ref+'_matched'] for r in rows)
                    summary.append(item)
                for ref in ['release','F_init']:
                    other=[r for r in subset if not r['known_weak_track'] and r[ref+'_matched']]
                    protection.append(dict(condition=c,reference=ref,arrangement=arrangement,group=group,other_original_success_objects=len(other),
                        other_original_success_lost=sum(not r[c+'_matched'] for r in other),other_original_success_retained=sum(r[c+'_matched'] for r in other)))
                if group=='pooled':
                    for scope,rows in scopes.items():
                        if scope not in ['confirmed_weak_frames','known_weak_tracks_all_frames']:continue
                        for ref in ['F_init','ORIGINAL']:
                            rr=[r for r in rows if r[c+'_matched'] and not r[ref+'_matched']];n=Counter((r['sequence'],r['track_id']) for r in rr)
                            rank=sorted(n.items(),key=lambda kv:(-kv[1],kv[0]))
                            concentration.append(dict(condition=c,reference=ref,arrangement=arrangement,scope=scope,
                                recovered_target_frames=len(rr),recovered_tracks=len(n),sources=len({k[0] for k in n}),
                                top1_frames=rank[0][1] if rank else 0,top5_frames=sum(v for _,v in rank[:5]),
                                ranked_tracks=json.dumps([dict(sequence=k[0],track_id=k[1],frames=v) for k,v in rank])))
    groups=defaultdict(list)
    for row in objects:
        if row['known_weak_track']:groups[row['arrangement'],row['sequence'],row['track_id']].append(row)
    trackrows=[];segments=[]
    for (arrangement,sequence,tid),rows in sorted(groups.items()):
        rows=sorted(rows,key=lambda r:(r['frame'],r['source_frame']))
        for c in conditions:
            hits=runs(rows,c+'_matched',True);misses=runs(rows,c+'_matched',False)
            eligible=[i for i in range(1,len(rows)-1) if contiguous(rows[i-1],rows[i]) and contiguous(rows[i],rows[i+1])]
            item=dict(condition=c,arrangement=arrangement,sequence=sequence,track_id=tid,target_frames=len(rows),
                confirmed_weak_frames=sum(r['confirmed_weak_frame'] for r in rows),M=sum(r[c+'_matched'] for r in rows),
                confirmed_weak_M=sum(r[c+'_matched'] and r['confirmed_weak_frame'] for r in rows),
                matched_run_count=len(hits),matched_runs_ge2=sum(len(x)>=2 for x in hits),longest_matched_run=max(map(len,hits),default=0),
                miss_run_count=len(misses),longest_miss_run=max(map(len,misses),default=0),
                eligible_isolation_centers=len(eligible),isolated010=sum(rows[i][c+'_matched'] and not rows[i-1][c+'_matched'] and not rows[i+1][c+'_matched'] for i in eligible))
            for ref in ['release','F_init','ORIGINAL']:
                item['G_vs_'+ref]=sum(r[c+'_matched'] and not r[ref+'_matched'] for r in rows);item['L_vs_'+ref]=sum(not r[c+'_matched'] and r[ref+'_matched'] for r in rows)
                item['net_vs_'+ref]=item['G_vs_'+ref]-item['L_vs_'+ref]
            trackrows.append(item)
        for k,seg in enumerate(runs(rows,'F_init_matched',False)):
            item=dict(arrangement=arrangement,sequence=sequence,track_id=tid,F_init_miss_segment=k,
                frame_start=seg[0]['frame'],frame_end=seg[-1]['frame'],source_frame_start=seg[0]['source_frame'],source_frame_end=seg[-1]['source_frame'],
                target_frames=len(seg),confirmed_weak_frames=sum(r['confirmed_weak_frame'] for r in seg))
            for c in conditions:
                rr=runs(seg,c+'_matched',False);item[c+'_recovered_frames']=sum(r[c+'_matched'] for r in seg)
                item[c+'_remaining_miss_longest']=max(map(len,rr),default=0);item[c+'_remaining_miss_runs']=len(rr)
            segments.append(item)
    csvwrite(OUT/'OBJECT_COMPARISON.csv',objects);csvwrite(OUT/'SUMMARY.csv',summary);csvwrite(OUT/'BACKGROUND_COST.csv',costs)
    csvwrite(OUT/'PROTECTION.csv',protection);csvwrite(OUT/'RECOVERY_CONCENTRATION.csv',concentration)
    csvwrite(OUT/'TRACK_COVERAGE.csv',trackrows);csvwrite(OUT/'F_INIT_MISS_SEGMENTS.csv',segments)
    csvwrite(OUT/'FRAME_COUNTS.csv',allframes)
    save(OUT/'COMPLETE.json',dict(status='COMPLETE',conditions=conditions,total_effective_frames=800,known_weak_tracks=39,
        confirmed_weak_target_frames=2066,plans=PLANS,threshold=.5,source_sha256=sources,
        artifact_sha256={p.name:sha(p) for p in sorted(OUT.glob('*.csv'))},summary_source_sha256=sha(__file__),
        scope='Training-object learnability only, original full-frame matching-valid GT; no old 4015 threshold applied. O/S are separate arrangements. No confirmation data.'))
    print({'status':'TRAIN_SUMMARY_COMPLETE','objects_per_arrangement':len(objects)//2,'known_weak_track_frames':sum(r['known_weak_track'] for r in objects)//2},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build-plan',action='store_true');p.add_argument('--check-plan',action='store_true');p.add_argument('--summary',action='store_true')
    p.add_argument('--condition',choices=CONDITIONS);p.add_argument('--sequence');a=p.parse_args()
    if a.build_plan:build_plan()
    elif a.check_plan:check_plan()
    elif a.summary:summarize()
    else:
        if a.condition is None:p.error('--condition required')
        for seq in [a.sequence] if a.sequence else read(PLAN_PATH)['sequences']:score(a.condition,seq)
