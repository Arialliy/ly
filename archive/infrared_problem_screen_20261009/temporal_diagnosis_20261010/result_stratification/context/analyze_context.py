#!/usr/bin/env python3
"""Deterministic context diagnosis from saved CPU outputs; no inference or oracle selection."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

Q = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SEQS = ['DJI_0028_2', 'DJI_0028_4', 'DJI_0032_5', 'DJI_0047_3']
THRESHOLDS = [.1, .5, .9]
RULES = ['earliest', 'latest', 'center']
PAIRS = [('earliest','latest'), ('earliest','center'), ('latest','center')]

def read(p):
    with p.open() as f: return list(csv.DictReader(f))

def write(name, rows):
    with (OUT/name).open('w', newline='') as f:
        if rows:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def choose(rr, rule):
    if rule=='earliest': return min(rr,key=lambda r:int(r['window_start']))
    if rule=='latest': return max(rr,key=lambda r:int(r['window_start']))
    return min(rr,key=lambda r:(abs(int(r['slot'])-19.5),int(r['window_start'])))

def scale(x):
    return '<=8' if x<=8 else '(8,12]' if x<=12 else '(12,16]' if x<=16 else '>16'

def key(r):return (int(r['local_frame']),float(r['threshold']),int(r['track_id']))

def load():
    all_rows=[]; all_context={}; metadata={}; hashes={}
    candidates=json.loads((Q.parent/'real_motion_stage2/INSPECTION_QUEUE.json').read_text())['candidates'][:4]
    candidate_byseq={r['sequence']:r for r in candidates}
    for seq in SEQS:
        c=candidate_byseq[seq]
        annpath=Q.parent/'real_motion_stage2/selected_raw/annotations.json' if seq=='DJI_0028_2' else Q/'additional_clips'/c['candidate_id']/'annotations.json'
        annotations=json.loads(annpath.read_text()); cats={r['id']:r['name'] for r in annotations['categories']}
        byid={a['id']:a for a in annotations['annotations']}
        durations=collections.Counter(a['track_id'] for a in annotations['annotations'])
        origpath=Q/'analysis'/seq/'per_track_frame.csv'
        contextpath=Q/'window_offset_experiment/analysis'/seq/'per_context_all_tracks.csv'
        summarypath=contextpath.parent/'same_gt_frame_context_summary.csv'
        for p in [annpath,origpath,contextpath,summarypath]:hashes[str(p)]=digest(p)
        baseline={key(r):r for r in read(origpath)}
        prior={key(r):r for r in read(summarypath)}
        contexts=read(contextpath); all_context[seq]=contexts
        groups=collections.defaultdict(list)
        for r in contexts:groups[key(r)].append(r)
        assert set(groups)==set(prior)==set(baseline)
        assert sorted({int(r['window_start']) for r in contexts})==list(range(0,121,12))
        for k,rr in sorted(groups.items()):
            fi,th,tid=k; first=rr[0]; a=byid[int(first['annotation_id'])]; old=prior[k]; original=baseline[k]
            assert len(rr)==int(old['context_count'])
            assert len({int(r['window_start']) for r in rr})==len(rr)
            assert all(int(r['slot'])==fi-int(r['window_start']) for r in rr)
            assert len({r['annotation_id'] for r in rr})==1
            maxside=max(float(first['bbox_w']),float(first['bbox_h']))
            states=[int(r['matched']) for r in rr]; vals=[float(r['box_max']) for r in rr]
            anyamb=any(int(r['association_ambiguous']) for r in rr)
            clean=not int(first['partly_outside']) and not int(first['label_overlap_count'])
            row=dict(sequence=seq,local_frame=fi,source_frame=int(first['source_frame']),track_id=tid,
                annotation_id=a['id'],source_tracking_id=a.get('tracking_id',''),category_id=a['category_id'],
                category=cats[a['category_id']],threshold=th,max_side=maxside,scale_bin=scale(maxside),
                annotation_frames_in_clip=durations[tid],long_track=int(durations[tid]>=40),
                selected_track=int(tid==c['track_id']),context_count=len(rr),multi=int(len(rr)>=2),
                clean=int(clean),strict=int(clean and not anyamb),any_context_ambiguity=int(anyamb),
                partly_outside=int(first['partly_outside']),label_overlap=int(first['label_overlap_count'])>0,
                point_disagreement=int(min(states)!=max(states)),boxmax_crossing=int(min(vals)<th<=max(vals)),
                context_boxmax_min=min(vals),context_boxmax_max=max(vals),
                strong_response_flip=int(min(vals)<.1 and max(vals)>=.9),
                all_contexts_point_negative=int(not any(states)),all_contexts_boxmax_low=int(max(vals)<th),
                original_point=int(original['matched']),original_response=int(float(original['box_max'])>=th),
                original_boxmax=float(original['box_max']))
            assert row['point_disagreement']==int(old['point_disagreement'])
            assert row['boxmax_crossing']==int(old['boxmax_threshold_crossing'])
            for rule in RULES:
                r=choose(rr,rule)
                row.update({f'{rule}_start':int(r['window_start']),f'{rule}_slot':int(r['slot']),
                    f'{rule}_point':int(r['matched']),f'{rule}_response':int(float(r['box_max'])>=th),
                    f'{rule}_boxmax':float(r['box_max'])})
            for boundary, pred in [('boundary',lambda s:s<=3 or s>=36),('left_boundary',lambda s:s<=3),('right_boundary',lambda s:s>=36)]:
                bb=[r for r in rr if pred(int(r['slot']))]; ii=[r for r in rr if 4<=int(r['slot'])<=35]
                hasboth=bool(bb and ii);row[f'{boundary}_has_interior']=int(hasboth)
                for side, candidates in [('edge',bb),('inside',ii)]:
                    r=choose(candidates,'center') if candidates else None
                    row.update({f'{boundary}_{side}_start':int(r['window_start']) if r else '',
                        f'{boundary}_{side}_slot':int(r['slot']) if r else '',
                        f'{boundary}_{side}_point':int(r['matched']) if r else '',
                        f'{boundary}_{side}_response':int(float(r['box_max'])>=th) if r else ''})
            all_rows.append(row)
        metadata[seq]={'annotation_path':str(annpath),'raw_annotation_frames':len(annotations['annotations']),
            'raw_tracks':len(durations),'valid_gt_frames':len(groups)//3,'valid_tracks':len({r['track_id'] for r in all_rows if r['sequence']==seq}),
            'source_tracking_ids':len({str(a.get('tracking_id','')) for a in annotations['annotations']}),
            'selected_track':c['track_id']}
    return all_rows,all_context,metadata,hashes

def aggregations(rows):
    # Overlapping scale totals are explicitly labelled; target-frame is the unit.
    groups=collections.defaultdict(list)
    for r in rows:
        if not r['multi']:continue
        subset=['all_multi']+(['clean'] if r['clean'] else [])+(['strict'] if r['strict'] else [])
        scales=['all',r['scale_bin']]+(['<=16'] if r['max_side']<=16 else [])+(['<=12'] if r['max_side']<=12 else [])
        for sub in subset:
            for sc in scales:
                for seq in ['ALL',r['sequence']]:groups[(r['threshold'],sub,seq,sc,'all','all')].append(r)
            for seq in ['ALL',r['sequence']]:
                for sc in ['all']+(['<=16'] if r['max_side']<=16 else ['>16'])+(['<=12'] if r['max_side']<=12 else []):
                    groups[(r['threshold'],sub,seq,sc,'category',r['category'])].append(r)
                    groups[(r['threshold'],sub,seq,sc,'long_track',str(r['long_track']))].append(r)
                    groups[(r['threshold'],sub,seq,sc,'selected_track',str(r['selected_track']))].append(r)
    return groups

def transitions(rr, a, b, metric):
    gains=[r for r in rr if not r[f'{a}_{metric}'] and r[f'{b}_{metric}']]
    losses=[r for r in rr if r[f'{a}_{metric}'] and not r[f'{b}_{metric}']]
    tracks=lambda zz:len({(r['sequence'],r['track_id']) for r in zz})
    return dict(gt_frames=len(rr),tracks=tracks(rr),a_positive=sum(r[f'{a}_{metric}'] for r in rr),
        b_positive=sum(r[f'{b}_{metric}'] for r in rr),gains=len(gains),losses=len(losses),net=len(gains)-len(losses),
        gain_tracks=tracks(gains),loss_tracks=tracks(losses),changed_tracks=tracks(gains+losses),
        both_positive=sum(r[f'{a}_{metric}'] and r[f'{b}_{metric}'] for r in rr),
        both_negative=sum(not r[f'{a}_{metric}'] and not r[f'{b}_{metric}'] for r in rr))

def summaries(rows):
    groups=aggregations(rows);paired=[];sensitivity=[];boundary=[]
    for k,rr in sorted(groups.items()):
        th,subset,seq,sc,kind,value=k
        header=dict(threshold=th,subset=subset,sequence=seq,scale=sc,stratum_kind=kind,stratum=value)
        tracks={(r['sequence'],r['track_id']) for r in rr}
        sensitivity.append(dict(**header,gt_frames=len(rr),tracks=len(tracks),
            point_disagreement=sum(r['point_disagreement'] for r in rr),boxmax_crossing=sum(r['boxmax_crossing'] for r in rr),
            point_sensitive_tracks=len({(r['sequence'],r['track_id']) for r in rr if r['point_disagreement']}),
            response_sensitive_tracks=len({(r['sequence'],r['track_id']) for r in rr if r['boxmax_crossing']}),
            strong_response_flip=sum(r['strong_response_flip'] for r in rr),
            strong_response_flip_tracks=len({(r['sequence'],r['track_id']) for r in rr if r['strong_response_flip']}),
            all_contexts_point_negative=sum(r['all_contexts_point_negative'] for r in rr),
            all_contexts_boxmax_low=sum(r['all_contexts_boxmax_low'] for r in rr),
            original_point_positive=sum(r['original_point'] for r in rr),
            earliest_point_positive=sum(r['earliest_point'] for r in rr),latest_point_positive=sum(r['latest_point'] for r in rr),
            center_point_positive=sum(r['center_point'] for r in rr)))
        for a,b in PAIRS+[('original',r) for r in RULES]:
            for metric in ['point','response']:
                paired.append(dict(**header,a=a,b=b,metric=metric,**transitions(rr,a,b,metric)))
        if kind!='all':continue
        for bb in ['boundary','left_boundary','right_boundary']:
            zz=[r for r in rr if r[f'{bb}_has_interior']]
            for metric in ['point','response']:
                boundary.append(dict(**header,boundary=bb,metric=metric,
                    **transitions(zz,f'{bb}_edge',f'{bb}_inside',metric)))
    write('pairwise_gain_loss.csv',paired);write('sensitivity_and_remaining_gap.csv',sensitivity);write('boundary_interior_pairs.csv',boundary)
    trackgroups=collections.defaultdict(list)
    for r in rows:
        if not r['multi']:continue
        for sub in ['all_multi']+(['clean'] if r['clean'] else [])+(['strict'] if r['strict'] else []):
            for sc in ['all']+(['<=16'] if r['max_side']<=16 else [])+(['<=12'] if r['max_side']<=12 else []):
                trackgroups[(r['threshold'],sub,sc,r['sequence'],r['track_id'])].append(r)
    trackrows=[]
    for k,rr in sorted(trackgroups.items()):
        th,sub,sc,seq,tid=k
        h=dict(threshold=th,subset=sub,scale=sc,sequence=seq,track_id=tid,source_tracking_ids=';'.join(sorted({r['source_tracking_id'] for r in rr})),
            category=';'.join(sorted({r['category'] for r in rr})),selected_track=rr[0]['selected_track'],
            annotation_frames_in_clip=rr[0]['annotation_frames_in_clip'],gt_frames=len(rr),
            point_disagreement=sum(r['point_disagreement'] for r in rr),boxmax_crossing=sum(r['boxmax_crossing'] for r in rr),
            all_contexts_point_negative=sum(r['all_contexts_point_negative'] for r in rr),all_contexts_boxmax_low=sum(r['all_contexts_boxmax_low'] for r in rr))
        for a,b in PAIRS:
            for metric in ['point','response']:
                t=transitions(rr,a,b,metric)
                for f in ['gains','losses','net']:h[f'{a}_to_{b}_{metric}_{f}']=t[f]
        trackrows.append(h)
    write('all_track_summary.csv',trackrows)
    concentration=[]
    for th in THRESHOLDS:
        for sub in ['all_multi','clean','strict']:
            for sc in ['all','<=16','<=12']:
                for seq in ['ALL']+SEQS:
                    tt=[r for r in trackrows if r['threshold']==th and r['subset']==sub and r['scale']==sc and (seq=='ALL' or r['sequence']==seq)]
                    for metric in ['point_disagreement','boxmax_crossing']:
                        tt.sort(key=lambda r:(-r[metric],r['sequence'],r['track_id']));total=sum(r[metric] for r in tt)
                        concentration.append(dict(threshold=th,subset=sub,scale=sc,sequence=seq,metric=metric,
                            eligible_tracks=len(tt),sensitive_tracks=sum(r[metric]>0 for r in tt),gt_frame_changes=total,
                            top1_changes=sum(r[metric] for r in tt[:1]),top3_changes=sum(r[metric] for r in tt[:3]),top5_changes=sum(r[metric] for r in tt[:5]),
                            top1_fraction=sum(r[metric] for r in tt[:1])/total if total else None,
                            top3_fraction=sum(r[metric] for r in tt[:3])/total if total else None,
                            top5_fraction=sum(r[metric] for r in tt[:5])/total if total else None,
                            top_tracks=';'.join(f"{r['sequence']}:{r['track_id']}:{r[metric]}" for r in tt[:5])))
    write('track_concentration.csv',concentration)
    sourcegroups=collections.defaultdict(list)
    for r in rows:
        if not r['multi']:continue
        for sub in ['all_multi']+(['clean'] if r['clean'] else [])+(['strict'] if r['strict'] else []):
            for sc in ['all']+(['<=16'] if r['max_side']<=16 else [])+(['<=12'] if r['max_side']<=12 else []):
                sourcegroups[(r['threshold'],sub,sc,r['sequence'],r['source_tracking_id'])].append(r)
    sourcerows=[]
    for k,rr in sorted(sourcegroups.items()):
        th,sub,sc,seq,sid=k
        sourcerows.append(dict(threshold=th,subset=sub,scale=sc,sequence=seq,source_tracking_id=sid,
            annotated_track_ids=';'.join(map(str,sorted({r['track_id'] for r in rr}))),category=';'.join(sorted({r['category'] for r in rr})),
            gt_frames=len(rr),point_disagreement=sum(r['point_disagreement'] for r in rr),
            boxmax_crossing=sum(r['boxmax_crossing'] for r in rr),
            all_contexts_point_negative=sum(r['all_contexts_point_negative'] for r in rr),all_contexts_boxmax_low=sum(r['all_contexts_boxmax_low'] for r in rr)))
    write('source_tracking_id_summary.csv',sourcerows)
    return dict(pairwise=paired,sensitivity=sensitivity,boundary=boundary,concentration=concentration)

def component_counts(contexts):
    import numpy as np
    from scipy import ndimage
    result=[];hashes={}
    for seq in SEQS:
        byframe=collections.defaultdict(list)
        for r in contexts[seq]:byframe[(int(r['local_frame']),int(r['window_start']),float(r['threshold']))].append(r)
        chosen=collections.defaultdict(list)
        for fi in range(160):
            starts=[s for s in range(0,121,12) if s<=fi<s+40]
            fake=[{'window_start':s,'slot':fi-s} for s in starts]
            for rule in RULES:chosen[(fi,int(choose(fake,rule)['window_start']))].append(rule)
        files=list((Q/'runs'/seq/'windows').glob('window*.npz'))+list((Q/'window_offset_experiment/runs'/seq/'windows').glob('window*.npz'))
        assert len(files)==11
        for p in sorted(files):
            hashes[str(p)]=digest(p)
            with np.load(p,allow_pickle=False) as saved:
                probs=saved['probabilities'];start=int(saved['start']);end=int(saved['end'])
            assert probs.shape==(40,512,640) and end-start==40
            for fi in range(start,end):
                if (fi,start) not in chosen:continue
                for th in THRESHOLDS:
                    rr=byframe[(fi,start,th)]; assert len({r['track_id'] for r in rr})==len(rr)
                    matched=sum(int(r['matched']) for r in rr)
                    ids=[int(r['component_id']) for r in rr if r['component_id']]
                    assert len(ids)==len(set(ids))==matched
                    _,count=ndimage.label(probs[fi-start]>=th,structure=np.ones((3,3),np.uint8))
                    assert count>=matched and all(i<=count for i in ids)
                    for rule in chosen[(fi,start)]:
                        result.append(dict(sequence=seq,local_frame=fi,source_frame=int(rr[0]['source_frame']) if rr else '',
                            threshold=th,rule=rule,window_start=start,slot=fi-start,multi=int(len([s for s in range(0,121,12) if s<=fi<s+40])>=2),
                            gt_count=len(rr),matched_count=matched,component_count=int(count),unmatched_components=int(count)-matched))
            print(f'components {seq} start={start}',flush=True)
        del probs
    write('chosen_per_frame_components.csv',sorted(result,key=lambda r:(r['sequence'],r['threshold'],r['rule'],r['local_frame'])))
    summaries=[]
    for th in THRESHOLDS:
        for seq in ['ALL']+SEQS:
            for subset in ['all160','multi_context_frames']:
                for rule in RULES:
                    rr=[r for r in result if r['threshold']==th and r['rule']==rule and (seq=='ALL' or r['sequence']==seq) and (subset=='all160' or r['multi'])]
                    sums={x:sum(r[x] for r in rr) for x in ['gt_count','matched_count','component_count','unmatched_components']}
                    summaries.append(dict(threshold=th,sequence=seq,subset=subset,rule=rule,frames=len(rr),**sums,
                        unmatched_components_per_frame=sums['unmatched_components']/len(rr)))
    write('component_summary.csv',summaries)
    (OUT/'component_input_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    return summaries

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--components',action='store_true');args=ap.parse_args()
    rows,contexts,meta,hashes=load();write('same_gt_frame_choices.csv',rows)
    write('strong_flips_strict_half_le16.csv',[r for r in rows if r['threshold']==.5 and r['multi'] and r['strict'] and r['max_side']<=16 and r['strong_response_flip']])
    result=summaries(rows)
    if args.components:result['components']=component_counts(contexts)
    elif (OUT/'component_summary.csv').exists():result['components']=read(OUT/'component_summary.csv')
    result['metadata']=meta
    result['methods']={'thresholds':THRESHOLDS,'primary_threshold':.5,'window_starts':list(range(0,121,12)),
        'window_length':40,'rules':{'earliest':'smallest available start','latest':'largest available start','center':'smallest abs(slot-19.5), tie smallest start'},
        'selection_uses_gt_or_score':False,'primary_unit':'same GT x current frame; >=2 available windows',
        'clean':'wholly in frame and no overlapping GT boxes','strict':'clean plus no matching ambiguity in ANY available context',
        'boundary':'same GT frame with both slot 0..3 or36..39 and interior4..35; each side center-nearest, tie lower start',
        'components':'8-connected components >=threshold from saved NPZ; subtract all-GT one-to-one matches from existing CSV; unmatched is not confirmed false-positive',
        'limitations':['contexts and slots change jointly','four prescreened clips; sequence is not necessarily independent scene','adjacent GT frames correlated; no framewise significance','processed and source IDs are annotation labels; source IDs may be category-specific or reused; their counts do not confirm physical object identity or fragmentation','all-context negative is finite-window diagnosis, not an oracle deployed metric','category mapping from annotation only; visibility not fully audited','original is max-fused prediction and differs in component merging, included only supplementary']}
    result['input_sha256']=hashes;result['script_sha256']=digest(Path(__file__))
    (OUT/'SUMMARY.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'rows':len(rows),'metadata':meta,'output':str(OUT)}),flush=True)

if __name__=='__main__':main()
