#!/usr/bin/env python3
"""Post-primary representation sensitivity. Does not alter primary protocol or metrics."""
import collections
import csv
import json
from pathlib import Path
import sys

import numpy as np
import analyze as primary


def inside(point, box):
    x,y,w,h=box
    return x<=point[0]<x+w and y<=point[1]<y+h


def run(D):
    Q=D/'temporal_diagnosis_20261010';root=Q/'analysis'
    candidates=json.loads((D/'real_motion_stage2/INSPECTION_QUEUE.json').read_text())['candidates'][:4]
    all_summaries=[]
    for candidate in candidates:
        seq=candidate['sequence'];out=root/seq
        data=D/'real_motion_stage2/selected_raw' if seq=='DJI_0028_2' else Q/'additional_clips'/candidate['candidate_id']
        annotations=json.loads((data/'annotations.json').read_text())
        images=sorted(annotations['images'],key=lambda im:im['frame_id'])
        byimage=collections.defaultdict(list)
        for ann in annotations['annotations']:byimage[ann['image_id']].append(ann)
        with (out/'overlap_same_frame.csv').open() as f:raw=list(csv.DictReader(f))
        flips=[r for r in raw if float(r['threshold'])==.5 and int(r['unambiguous_pair']) and int(r['match_flip'])]
        grouped=collections.defaultdict(list)
        for ri,r in enumerate(flips):
            for suffix in ['a','b']:
                grouped[int(r['window_'+suffix])].append((ri,suffix,r))
        side_results={}
        files=sorted((Q/'runs'/seq/'windows').glob('window*.npz'))
        for wi,items in grouped.items():
            with np.load(files[wi],allow_pickle=False) as saved:
                wp=saved['probabilities'];start=int(saved['start'])
            cache={}
            for ri,suffix,r in items:
                fi=int(r['local_frame']);tid=int(r['track_id'])
                if fi not in cache:
                    gt,_=primary.prepare_gt(byimage[images[fi]['id']],*wp.shape[1:])
                    rr,_,cc=primary.match_frame(wp[fi-start],gt,.5)
                    cache[fi]=(gt,rr,cc)
                gt,rr,cc=cache[fi]
                ann=next(g for g in gt if g['track_id']==tid)
                row=next(t for t in rr if t['track_id']==tid)
                assert row['matched']==int(r['matched_'+suffix])
                if row['component_id'] is not None:
                    comp=cc[row['component_id']-1]
                    matched_centroid=int(inside((comp['centroid_x'],comp['centroid_y']),ann['bbox']))
                    matched_weighted=int(inside((comp['weighted_centroid_x'],comp['weighted_centroid_y']),ann['bbox']))
                    area=comp['area']
                else:
                    matched_centroid=matched_weighted=area=None
                any_centroid=int(any(inside((c['centroid_x'],c['centroid_y']),ann['bbox']) for c in cc))
                any_weighted=int(any(inside((c['weighted_centroid_x'],c['weighted_centroid_y']),ann['bbox']) for c in cc))
                x0,y0,x1,y1=ann['crop'];crop=wp[fi-start,y0:y1,x0:x1]
                side_results[ri,suffix]={
                    'matched_component_centroid_inside':matched_centroid,
                    'matched_component_weighted_centroid_inside':matched_weighted,
                    'matched_component_area':area,
                    'any_component_centroid_inside':any_centroid,
                    'any_component_weighted_centroid_inside':any_weighted,
                    'box_has_probability_exactly_1':int(np.any(crop==1.)),
                    'box_count_probability_exactly_1':int(np.count_nonzero(crop==1.)),
                    'boxmax_proxy':float(crop.max()) if crop.size else None}
        details=[]
        for ri,r in enumerate(flips):
            a,b=side_results[ri,'a'],side_results[ri,'b']
            boxa=a['boxmax_proxy']>=.5;boxb=b['boxmax_proxy']>=.5
            category='boxmax_crosses_threshold' if boxa!=boxb else 'boxmax_both_at_or_above_threshold' if boxa else 'boxmax_both_below_threshold'
            assert category!='boxmax_both_below_threshold', 'Point-in-box matched peak requires box max above threshold'
            detail=dict(r,boxmax_flip_category=category,
                geometric_centroid_presence_flip=int(a['any_component_centroid_inside']!=b['any_component_centroid_inside']),
                weighted_centroid_presence_flip=int(a['any_component_weighted_centroid_inside']!=b['any_component_weighted_centroid_inside']))
            for suffix,side in [('a',a),('b',b)]:
                detail.update({k+'_'+suffix:v for k,v in side.items()})
            details.append(detail)
        primary.save_csv(out/'SENSITIVITY_overlap_flips_0_5.csv',details)
        with (out/'per_track_frame.csv').open() as f:rows=list(csv.DictReader(f))
        proxy=[]
        for th in [.1,.5,.9]:
            rr=[r for r in rows if float(r['threshold'])==th and int(r['selected_track'])]
            proxy.append({'threshold':th,'gt_frames':len(rr),'primary_point_matches':sum(int(r['matched']) for r in rr),
                          'boxmax_at_or_above_threshold_frames':sum(bool(r['box_max']) and float(r['box_max'])>=th for r in rr),
                          'boxmax_exactly_one_frames':sum(bool(r['box_max']) and float(r['box_max'])==1. for r in rr),
                          'proxy_positive_but_point_unmatched_frames':sum(bool(r['box_max']) and float(r['box_max'])>=th and not int(r['matched']) for r in rr)})
        categories={}
        for category in ['boxmax_crosses_threshold','boxmax_both_at_or_above_threshold']:
            subset=[r for r in details if r['boxmax_flip_category']==category]
            categories[category]={'pairs':len(subset),
                'geometric_centroid_presence_flips':sum(r['geometric_centroid_presence_flip'] for r in subset),
                'weighted_centroid_presence_flips':sum(r['weighted_centroid_presence_flip'] for r in subset),
                'either_box_has_probability_exactly_one':sum(r['box_has_probability_exactly_1_a'] or r['box_has_probability_exactly_1_b'] for r in subset)}
        summary={'sequence':seq,'selected_track':candidate['track_id'],'primary_threshold':.5,
            'unambiguous_point_match_flips':len(flips),'flip_categories':categories,'selected_track_boxmax_proxies':proxy,
            'status':'SUPPLEMENTARY_REPRESENTATION_SENSITIVITY_NOT_REPLACEMENT_METRIC',
            'code_sha256':primary.digest(Path(__file__)),
            'limits':[
                'Peak rules and all original primary results remain unchanged.',
                'Centroid presence is a representation sensitivity on preselected flip cases, not a new full benchmark metric.',
                'GT boxes in the retained pairs do not overlap other GT boxes, so a point cannot be assigned to both this GT and another GT.',
                'Positive box max can originate from another response, merged blob, or annotation mismatch; it is not detection success.',
                'Probability exactly one shows sigmoid saturation at FP32 output; it does not prove saturation caused the match flip.',
                'Window contrast changes context, slots and padding together; the cause is not isolated.']}
        primary.save_json(out/'SENSITIVITY_SUMMARY.json',summary)
        all_summaries.append(summary)
        primary.save_json(out/'ARTIFACT_SHA256.json',{p.name:primary.digest(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='ARTIFACT_SHA256.json'})
        print(json.dumps(summary),flush=True)
    primary.save_json(root/'SENSITIVITY_ALL_CLIPS.json',all_summaries)


if __name__=='__main__':run(Path(sys.argv[1]))
