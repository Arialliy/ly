"""Independent threshold-.5 recheck; imports no original diagnostic code.

Connected components: explicit 8-neighbor flood fill over foreground pixel set.
Matching: independently implemented augmenting paths following published tie rule.
An exhaustive subset DP independently verifies each maximum match cardinality.
"""
from pathlib import Path
from collections import defaultdict, Counter
import csv
import hashlib
import json
import math
import numpy as np

Q=Path('/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/temporal_diagnosis_20261010')

def flood_components(p,th=.5):
    h,w=p.shape
    active=np.flatnonzero(p.ravel()>=th).tolist()
    unseen=set(active)
    components=[]
    for origin in active:
        if origin not in unseen: continue
        unseen.remove(origin); stack=[origin]; pixels=[]
        while stack:
            idx=stack.pop(); pixels.append(idx); y,x=divmod(idx,w)
            for yy in range(max(0,y-1),min(h,y+2)):
                for xx in range(max(0,x-1),min(w,x+2)):
                    other=yy*w+xx
                    if other in unseen:
                        unseen.remove(other); stack.append(other)
        ids=np.asarray(sorted(pixels),dtype=np.int64)
        values=p.ravel()[ids]
        best=float(values.max())
        peaks=ids[values==best]
        py,px=divmod(int(peaks[0]),w)
        components.append({'id':len(components)+1,'x':px+.5,'y':py+.5,'score':best,
                           'area':len(pixels),'peak_ties':len(peaks),'peak_indices':peaks})
    return components

def assignment(gts,comps):
    edges=[]
    for a in gts:
        x,y,w,h=a['bbox']; cx,cy=x+w/2,y+h/2
        choices=[i for i,c in enumerate(comps) if x<=c['x']<x+w and y<=c['y']<y+h]
        choices.sort(key=lambda i:(((comps[i]['x']-cx)/w)**2+((comps[i]['y']-cy)/h)**2,-comps[i]['score'],comps[i]['id']))
        edges.append(choices)
    owner={}
    def augment(gi,seen):
        for ci in edges[gi]:
            if ci in seen: continue
            seen.add(ci)
            if ci not in owner or augment(owner[ci],seen):
                owner[ci]=gi
                return True
        return False
    for gi in range(len(gts)): augment(gi,set())
    # Exhaustive reachable GT subsets: each component can choose zero or one GT.
    masks={0}
    for ci in range(len(comps)):
        candidates=[g for g,e in enumerate(edges) if ci in e]
        if candidates:
            expanded=set(masks)
            for old in masks:
                for g in candidates:
                    if not (old>>g)&1: expanded.add(old|(1<<g))
            masks=expanded
    max_count=max(int(m).bit_count() for m in masks)
    assert max_count==len(owner)
    return {g:c for c,g in owner.items()},edges

def unmatched_runs(records):
    result=[]; run=[]
    for frame,matched in records:
        if matched or (run and frame!=run[-1]+1):
            if run: result.append({'start_source_frame':run[0],'end_source_frame':run[-1],'frames':len(run)})
            run=[]
        if not matched: run.append(frame)
    if run: result.append({'start_source_frame':run[0],'end_source_frame':run[-1],'frames':len(run)})
    return result

def readcsv(p):
    with p.open() as f: return list(csv.DictReader(f))

all_results=[]
all_reported=json.loads((Q/'analysis/ALL_CLIPS_SUMMARY.json').read_text())
for seq in ['DJI_0028_2','DJI_0028_4','DJI_0032_5','DJI_0047_3']:
    out=Q/'analysis'/seq; summary=json.loads((out/'SUMMARY.json').read_text())
    assert summary==next(s for s in all_reported if s['sequence']==seq)
    raw=Path(summary['prediction_metadata']['input']).parent
    ann=json.loads((raw/'annotations.json').read_text())
    manifest=json.loads((raw/'FRAME_MANIFEST.json').read_text())
    assert manifest['selected_original_track_id']==summary['selected_track_id']
    with np.load(Q/'runs'/seq/'predictions.npz',allow_pickle=False) as z: probs=z['probabilities']
    ref_rows={(int(r['local_frame']),int(r['annotation_id'])):r for r in readcsv(out/'per_track_frame.csv') if float(r['threshold'])==.5}
    ref_comps={(int(r['local_frame']),int(r['component_id'])):r for r in readcsv(out/'components.csv') if float(r['threshold'])==.5}
    ref_tracks={int(r['track_id']):r for r in json.loads((out/'track_summary.json').read_text()) if r['threshold']==.5}
    byimage=defaultdict(list)
    for a in ann['annotations']: byimage[a['image_id']].append(a)
    id_counts=Counter(a['id'] for a in ann['annotations'])
    duplicate_ids={str(k):v for k,v in id_counts.items() if v>1}
    trackframe_counts=Counter((a['image_id'],a['track_id']) for a in ann['annotations'])
    duplicate_trackframes={str(k):v for k,v in trackframe_counts.items() if v>1}
    records=defaultdict(list); totals=Counter(); inconsistencies=[]; saturation=Counter(); represented=[]
    for fi,f in enumerate(manifest['frames']):
        p=probs[fi]; H,W=p.shape; gt=[]
        for a in byimage[f['source_image_id']]:
            x,y,w,h=map(float,a['bbox'])
            if a.get('ignore',0) or not np.isfinite([x,y,w,h]).all() or w<=0 or h<=0 or min(W,x+w)<=max(0,x) or min(H,y+h)<=max(0,y):
                totals['excluded']+=1; continue
            gt.append(a)
        gt.sort(key=lambda a:(int(a['track_id']),int(a['id'])))
        comps=flood_components(p)
        matches,edges=assignment(gt,comps)
        totals['gt_frames']+=len(gt);totals['matched_gt_frames']+=len(matches);totals['components']+=len(comps)
        for c in comps:
            ref=ref_comps[(fi,c['id'])]
            assert int(ref['area'])==c['area'] and float(ref['peak_x'])==c['x'] and float(ref['peak_y'])==c['y'] and float(ref['peak_probability'])==c['score']
            saturation['components_with_tied_maxima']+=int(c['peak_ties']>1)
            saturation['components_with_max_exactly_1']+=int(c['score']==1)
            saturation['components_with_saturated_plateau']+=int(c['score']==1 and c['peak_ties']>1)
        for gi,a in enumerate(gt):
            ref=ref_rows[(fi,a['id'])]; ismatch=gi in matches
            x,y,w,h=map(float,a['bbox']);tid=int(a['track_id']); frame=int(f['raw_frame'])
            partial=x<0 or y<0 or x+w>W or y+h>H
            totals['partial_boxes']+=int(partial)
            totals['negative_x_boxes']+=int(x<0)
            totals['gt_frames_with_identical_other_track_box']+=int(any(other['bbox']==a['bbox'] for j,other in enumerate(gt) if j!=gi))
            assert int(ref['partly_outside'])==partial
            ix0=max(0,math.ceil(x-.5));ix1=min(W,math.ceil(x+w-.5))
            iy0=max(0,math.ceil(y-.5));iy1=min(H,math.ceil(y+h-.5))
            boxmax=float(p[iy0:iy1,ix0:ix1].max()) if ix0<ix1 and iy0<iy1 else None
            assert boxmax is None or float(ref['box_max'])==boxmax
            if bool(int(ref['matched']))!=ismatch:
                inconsistencies.append({'local_frame':fi,'annotation_id':a['id'],'ref_match':ref['matched'],'independent_match':ismatch})
            records[tid].append((frame,ismatch))
            if not ismatch and boxmax is not None and boxmax>=.5:
                totals['box_response_positive_but_point_unmatched']+=1
                # Does a tied maximal pixel inside this GT lose row-major selection?
                tied_alternative=False
                for c in comps:
                    if c['peak_ties']<=1: continue
                    coords=c['peak_indices']; xs=coords%W+.5;ys=coords//W+.5
                    in_box=(xs>=x)&(xs<x+w)&(ys>=y)&(ys<y+h)
                    if np.any(in_box) and not(x<=c['x']<x+w and y<=c['y']<y+h): tied_alternative=True
                if tied_alternative: totals['unmatched_with_alternative_tied_peak_in_box']+=1
                if tid==summary['selected_track_id']:
                    represented.append({'source_frame':frame,'boxmax':boxmax,'tied_peak_inside_but_selected_outside':tied_alternative})
    tracks=[]
    for tid,rr in sorted(records.items()):
        runs=unmatched_runs(rr); nmatch=sum(m for _,m in rr); longest=max((r['frames'] for r in runs),default=0)
        ref=ref_tracks[tid]
        checks={'count':nmatch==ref['matched_frames'],'denominator':len(rr)==ref['gt_frames'],'longest':longest==ref['longest_unmatched_frames'],'runs':runs==ref['unmatched_runs']}
        if not all(checks.values()): inconsistencies.append({'track_id':tid,'checks':checks})
        tracks.append({'track_id':tid,'gt_frames':len(rr),'matched_frames':nmatch,'longest_unmatched_frames':longest,'unmatched_runs':runs})
    primary=next(x for x in summary['primary'] if x['threshold']==.5)
    for key in ['gt_frames','matched_gt_frames','components']:
        assert totals[key]==primary[key],(seq,key,totals[key],primary[key])
    assert totals['components']-totals['matched_gt_frames']==primary['unmatched_components_not_confirmed_fp']
    assert totals['gt_frames_with_identical_other_track_box']==primary['label_identical_box_gt_frames']
    assert not inconsistencies,inconsistencies
    selected=next(x for x in tracks if x['track_id']==summary['selected_track_id'])
    r={'sequence':seq,'status':'PASS','threshold':.5,'source_annotation_count':len(ann['annotations']),
       'duplicate_annotation_ids':duplicate_ids,'duplicate_image_track_keys':duplicate_trackframes,
       'counts':dict(totals),'saturation':dict(saturation),'selected_track':selected,
       'selected_positive_response_point_unmatched':represented,'tracks':tracks,
       'row_and_component_csv_exact':True,'maximum_cardinality_exhaustive_DP_all_frames':True,
       'all_clips_summary_copy_exact':True,
       'source_annotation_sha256':hashlib.sha256((raw/'annotations.json').read_bytes()).hexdigest()}
    all_results.append(r);print(json.dumps({k:r[k] for k in ['sequence','counts','saturation','selected_track']}),flush=True)
aggregate=Counter()
for r in all_results: aggregate.update(r['counts'])
result={'status':'PASS','threshold':.5,'results':all_results,'aggregate_counts':dict(aggregate),
        'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'Independent full recomputation from original boxes and fused predictions, no imports from analysis code. No overlap-window result audit.'}
(Q/'audit/DIAGNOSTIC_RESULT_RECHECK.json').write_text(json.dumps(result,indent=2)+'\n')
