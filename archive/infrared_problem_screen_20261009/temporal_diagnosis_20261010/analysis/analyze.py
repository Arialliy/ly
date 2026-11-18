#!/usr/bin/env python3
"""Frozen bbox point-response diagnostics; CPU only, no model inference."""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np
import scipy
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def save_csv(path, rows):
    if not rows:
        path.write_text('')
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def crop_indices(box, height, width):
    x, y, w, h = box
    # Exactly those pixel centers in the half-open original bbox. Clamp BEFORE slicing.
    return (max(0, min(width, math.ceil(x - .5))),
            max(0, min(height, math.ceil(y - .5))),
            max(0, min(width, math.ceil(x + w - .5))),
            max(0, min(height, math.ceil(y + h - .5))))


def prepare_gt(anns, height, width):
    valid, excluded = [], []
    for ann in sorted(anns, key=lambda a: (a['track_id'], a['id'])):
        x, y, w, h = map(float, ann['bbox'])
        clip = [max(0., x), max(0., y), min(float(width), x+w), min(float(height), y+h)]
        reason = ('ignored' if ann.get('ignore', 0) else
                  'invalid_box' if w <= 0 or h <= 0 else
                  'outside_frame' if clip[2] <= clip[0] or clip[3] <= clip[1] else None)
        if reason:
            excluded.append({'annotation_id': ann['id'], 'track_id': ann['track_id'], 'reason': reason})
            continue
        a = dict(ann)
        a['bbox'] = [x, y, w, h]
        a['clipped_box_xyxy'] = clip
        a['partly_outside'] = int(x < 0 or y < 0 or x+w > width or y+h > height)
        a['crop'] = crop_indices(a['bbox'], height, width)
        valid.append(a)
    assert len({a['track_id'] for a in valid}) == len(valid), 'Duplicate track labels in one frame'
    for a in valid:
        ax,ay,aw,ah=a['bbox'];overlap=[];identical=[];ious=[]
        for b in valid:
            if a['id']==b['id']:continue
            bx,by,bw,bh=b['bbox']
            intersection=max(0.,min(ax+aw,bx+bw)-max(ax,bx))*max(0.,min(ay+ah,by+bh)-max(ay,by))
            if intersection>0:
                overlap.append(b['track_id']);ious.append(intersection/(aw*ah+bw*bh-intersection))
            if a['bbox']==b['bbox']:identical.append(b['track_id'])
        a['label_overlap_track_ids']=overlap
        a['label_identical_box_track_ids']=identical
        a['label_max_box_iou']=max(ious,default=0.)
    return valid, excluded


def components(probability, threshold):
    labels, number = ndimage.label(probability >= threshold, structure=np.ones((3,3), np.uint8))
    flat = np.flatnonzero(labels)
    if not number:
        return [], labels
    lab = labels.ravel()[flat]
    scores = probability.ravel()[flat]
    # Lexicographic label, decreasing probability, row-major index; never use scipy tie behavior.
    order = np.lexsort((flat, -scores, lab))
    chosen = order[np.r_[True, np.diff(lab[order]) != 0]]
    peaks = flat[chosen]
    assert len(peaks) == number
    width = probability.shape[1]
    ys, xs = np.divmod(flat, width)
    area = np.bincount(lab, minlength=number+1)
    cx = np.bincount(lab, weights=xs+.5, minlength=number+1)
    cy = np.bincount(lab, weights=ys+.5, minlength=number+1)
    weight = np.bincount(lab, weights=scores, minlength=number+1)
    wx = np.bincount(lab, weights=(xs+.5)*scores, minlength=number+1)
    wy = np.bincount(lab, weights=(ys+.5)*scores, minlength=number+1)
    boxes = ndimage.find_objects(labels)
    result = []
    for i, peak in enumerate(peaks, 1):
        py, px = divmod(int(peak), width)
        by, bx = boxes[i-1]
        result.append({'component_id': i, 'peak_x': px+.5, 'peak_y': py+.5,
                       'peak_probability': float(probability[py,px]), 'area': int(area[i]),
                       'centroid_x': float(cx[i]/area[i]), 'centroid_y': float(cy[i]/area[i]),
                       'weighted_centroid_x': float(wx[i]/weight[i]),
                       'weighted_centroid_y': float(wy[i]/weight[i]),
                       'bbox_left': bx.start, 'bbox_top': by.start,
                       'bbox_width': bx.stop-bx.start, 'bbox_height': by.stop-by.start})
    return result, labels


def match_frame(probability, gt, threshold):
    comps, labels = components(probability, threshold)
    edges, reverse = [], collections.defaultdict(list)
    for gi, ann in enumerate(gt):
        x,y,w,h = ann['bbox']
        eligible = [ci for ci,c in enumerate(comps)
                    if x <= c['peak_x'] < x+w and y <= c['peak_y'] < y+h]
        eligible.sort(key=lambda ci: (((comps[ci]['peak_x']-x-w/2)/w)**2 +
                                      ((comps[ci]['peak_y']-y-h/2)/h)**2,
                                     -comps[ci]['peak_probability'], comps[ci]['component_id']))
        edges.append(eligible)
        for ci in eligible:
            reverse[ci].append(gi)
    owner = {}
    def augment(gi, seen):
        for ci in edges[gi]:
            if ci in seen:
                continue
            seen.add(ci)
            if ci not in owner or augment(owner[ci], seen):
                owner[ci] = gi
                return True
        return False
    for gi in range(len(gt)):
        augment(gi, set())
    chosen = {gi:ci for ci,gi in owner.items()}
    assert len(chosen) == len(owner)
    rows = []
    gt_pixels = np.zeros(probability.shape, dtype=bool)
    for gi, ann in enumerate(gt):
        x,y,w,h = ann['bbox']
        x0,y0,x1,y1 = ann['crop']
        box = probability[y0:y1,x0:x1]
        gt_pixels[y0:y1,x0:x1] = True
        ci = chosen.get(gi)
        component = comps[ci] if ci is not None else None
        overlap_labels = np.unique(labels[y0:y1,x0:x1])
        overlapping_count = int(np.count_nonzero(overlap_labels))
        ambiguous = len(edges[gi]) > 1 or any(len(reverse[c]) > 1 for c in edges[gi])
        dx = (component['peak_x']-x-w/2)/w if component else None
        dy = (component['peak_y']-y-h/2)/h if component else None
        rows.append({'track_id': ann['track_id'], 'annotation_id': ann['id'],
                     'bbox_x': x, 'bbox_y': y, 'bbox_w': w, 'bbox_h': h,
                     'partly_outside': ann['partly_outside'], 'matched': int(ci is not None),
                     'label_overlap_count':len(ann['label_overlap_track_ids']),
                     'label_identical_box_count':len(ann['label_identical_box_track_ids']),
                     'label_max_box_iou':ann['label_max_box_iou'],
                     'label_overlap_track_ids':';'.join(map(str,ann['label_overlap_track_ids'])),
                     'label_identical_box_track_ids':';'.join(map(str,ann['label_identical_box_track_ids'])),
                     'component_id': component['component_id'] if component else None,
                     'component_area': component['area'] if component else None,
                     'peak_x': component['peak_x'] if component else None,
                     'peak_y': component['peak_y'] if component else None,
                     'peak_probability': component['peak_probability'] if component else None,
                     'normalized_dx': dx, 'normalized_dy': dy,
                     'normalized_center_error': math.hypot(dx,dy) if component else None,
                     'box_max': float(box.max()) if box.size else None,
                     'box_mean': float(box.mean()) if box.size else None,
                     'box_has_pixel_center': int(box.size > 0),
                     'eligible_components': len(edges[gi]),
                     'intersecting_components': overlapping_count,
                     'association_ambiguous': int(ambiguous),
                     'eligible_component_ids': ';'.join(str(comps[c]['component_id']) for c in edges[gi])})
    for ci, c in enumerate(comps):
        c['eligible_gt_count'] = len(reverse[ci])
        c['eligible_track_ids'] = ';'.join(str(gt[gi]['track_id']) for gi in reverse[ci])
        c['assigned_track_id'] = gt[owner[ci]]['track_id'] if ci in owner else None
        c['unmatched_component'] = int(ci not in owner)
        c['unmatched_with_edge'] = int(ci not in owner and len(reverse[ci]) > 0)
    frame = {'gt_count': len(gt), 'matched_count': len(chosen), 'component_count': len(comps),
             'unmatched_components': len(comps)-len(chosen),
             'unmatched_components_with_edge': sum(c['unmatched_with_edge'] for c in comps),
             'components_no_peak_in_gt': sum(not reverse[c] for c in range(len(comps))),
             'components_multiple_gt': sum(len(reverse[c]) > 1 for c in range(len(comps))),
             'gt_multiple_components': sum(len(e) > 1 for e in edges),
             'ambiguous_gt_count': sum(r['association_ambiguous'] for r in rows),
             'positive_pixels': int(np.count_nonzero(labels)),
             'positive_pixels_outside_boxes': int(np.count_nonzero((labels > 0) & ~gt_pixels))}
    return rows, frame, comps


def longest_run(rows):
    runs, active, last = [], [], None
    for row in sorted(rows, key=lambda r:r['local_frame']):
        if row['matched'] or (last is not None and row['local_frame'] != last+1):
            if active:
                runs.append(active)
            active=[]
        if not row['matched']:
            active.append(row)
        last=row['local_frame']
    if active:
        runs.append(active)
    return max((len(r) for r in runs), default=0), [
        {'start_source_frame':r[0]['source_frame'],'end_source_frame':r[-1]['source_frame'],
         'frames':len(r)} for r in runs]


def summarize_rows(rows):
    errors=[r['normalized_center_error'] for r in rows if r['matched']]
    boxmax=[r['box_max'] for r in rows if r['box_max'] is not None]
    longest,runs=longest_run(rows)
    return {'gt_frames':len(rows),'matched_frames':sum(r['matched'] for r in rows),
            'match_fraction':sum(r['matched'] for r in rows)/len(rows) if rows else None,
            'longest_unmatched_frames':longest,'unmatched_runs':runs,
            'ambiguous_gt_frames':sum(r['association_ambiguous'] for r in rows),
            'label_overlapping_gt_frames':sum(r['label_overlap_count']>0 for r in rows),
            'label_identical_box_gt_frames':sum(r['label_identical_box_count']>0 for r in rows),
            'partly_outside_frames':sum(r['partly_outside'] for r in rows),
            'normalized_center_error_median':float(np.median(errors)) if errors else None,
            'boxmax_min':min(boxmax) if boxmax else None,
            'boxmax_median':float(np.median(boxmax)) if boxmax else None,
            'boxmax_max':max(boxmax) if boxmax else None}


def plot_all(out, rows, selected, sequence):
    r=[r for r in rows if r['threshold']==.5]
    tracks=sorted({x['track_id'] for x in r})
    fig,axs=plt.subplots(len(tracks),1,figsize=(14,max(3,1.65*len(tracks))),sharex=True,squeeze=False)
    for ax,tid in zip(axs.ravel(),tracks):
        subset=sorted((x for x in r if x['track_id']==tid),key=lambda x:x['local_frame'])
        xx=[x['local_frame'] for x in subset]
        ax.plot(xx,[x['box_max'] for x in subset],color='tab:blue',lw=1,label='box max (proxy)')
        ax.scatter(xx,[1.06 if x['matched'] else -.06 for x in subset],c=['tab:green' if x['matched'] else 'tab:red' for x in subset],s=8,label='point match @0.5')
        for b in [40,80,120]:ax.axvline(b-.5,color='gray',alpha=.5,lw=.7)
        for left,right in [(36,40),(72,76),(108,112),(120,148)]:ax.axvspan(left-.5,right-.5,color='orange',alpha=.12)
        ax.axhline(.5,color='gray',ls=':',lw=.6)
        ax.set_ylim(-.15,1.15);ax.set_yticks([0,.5,1]);ax.set_ylabel(str(tid)+('\nselected' if tid==selected else ''),fontsize=8)
    axs[0,0].legend(loc='lower left',fontsize=8,ncol=2)
    axs[-1,0].set_xlabel('Local frame; gray lines: descriptive blocks; orange: overlapping model windows')
    fig.suptitle(sequence+' — all annotated tracks; point response diagnostic, not official Pd',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.98));fig.savefig(out/'all_tracks_response.png',dpi=160);plt.close(fig)


def plot_contacts(out,data,images,probability,rows,track_summary,selected):
    """Illustrations only: fixed 4 block-start observations, all metrics already computed."""
    eligible=[s for s in track_summary if s['threshold']==.5 and s['gt_frames']>=40]
    weakest=min(eligible,key=lambda s:(s['match_fraction'],s['track_id']))['track_id'] if eligible else selected
    tids=list(dict.fromkeys([selected,weakest]))
    fig,axs=plt.subplots(len(tids),4,figsize=(13,3.4*len(tids)),squeeze=False)
    selections=[]
    for ri,tid in enumerate(tids):
        for block in range(4):
            ax=axs[ri,block]
            subset=sorted((r for r in rows if r['threshold']==.5 and r['track_id']==tid and r['block']==block),key=lambda r:r['local_frame'])
            if not subset:
                ax.text(.5,.5,'No GT in this block',ha='center');ax.axis('off');continue
            r=subset[0];fi=r['local_frame'];im=images[fi]
            rawpath=data/'VT-Tiny-MOT/train2017'/im['file_name']
            raw=np.asarray(Image.open(rawpath))
            if raw.ndim==3:raw=raw[...,0]
            cx=r['bbox_x']+r['bbox_w']/2;cy=r['bbox_y']+r['bbox_h']/2
            x0=max(0,min(raw.shape[1]-96,int(cx)-48));y0=max(0,min(raw.shape[0]-96,int(cy)-48))
            x1=min(raw.shape[1],x0+96);y1=min(raw.shape[0],y0+96)
            ax.imshow(raw[y0:y1,x0:x1],cmap='gray',vmin=0,vmax=255,extent=[x0,x1,y1,y0],interpolation='nearest')
            heat=probability[fi,y0:y1,x0:x1]
            ax.imshow(np.ma.masked_less(heat,.5),cmap='autumn',vmin=.5,vmax=1,alpha=.65,extent=[x0,x1,y1,y0],interpolation='nearest')
            ax.add_patch(Rectangle((r['bbox_x'],r['bbox_y']),r['bbox_w'],r['bbox_h'],fill=False,edgecolor='cyan',lw=1))
            if r['matched']:ax.scatter([r['peak_x']],[r['peak_y']],s=25,c='lime',marker='+')
            ax.set_xlim(x0,x1);ax.set_ylim(y1,y0);ax.set_xticks([]);ax.set_yticks([])
            ax.set_title(f"track {tid}; source {im['frame_id']}\npoint match={r['matched']}; box max={r['box_max']:.3g}",fontsize=9)
            selections.append({'track_id':tid,'local_frame':fi,'source_frame':im['frame_id'],'raw_path':str(rawpath)})
    fig.suptitle('Illustrative crops only: cyan GT, red/yellow p>=0.5, green assigned peak',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(out/'selected_and_low_response_contacts.png',dpi=160);plt.close(fig)
    save_json(out/'CONTACT_SELECTION.json',{'role':'post-statistic illustrations, not independent primary evidence',
        'rule':'selected track plus minimum 0.5 point-match fraction among tracks with >=40 valid GT frames, tie track id; first annotated frame in each fixed 40-frame block',
        'selected_track':selected,'low_response_track':weakest,'selections':selections})


def analyze_clip(D,Q,candidate):
    seq=candidate['sequence'];run=Q/'runs'/seq;prediction=run/'predictions.npz'
    if not prediction.exists():
        return {'sequence':seq,'status':'PENDING_PREDICTION'}
    data=(D/'real_motion_stage2/selected_raw' if seq=='DJI_0028_2' else
          Q/'additional_clips'/candidate['candidate_id'])
    out=Q/'analysis'/seq;out.mkdir(parents=True,exist_ok=True)
    annotation_path=data/'annotations.json'
    annotations=json.loads(annotation_path.read_text())
    images=sorted(annotations['images'],key=lambda a:a['frame_id'])
    by_image=collections.defaultdict(list)
    for ann in annotations['annotations']:by_image[ann['image_id']].append(ann)
    with np.load(prediction,allow_pickle=False) as saved:
        probability=saved['probabilities'];windows=saved['windows'];coverage=saved['coverage']
        metadata=json.loads(str(saved['metadata_json']))
    assert probability.shape[0]==len(images)==160
    assert probability.dtype==np.float32 and np.isfinite(probability).all()
    assert probability.min()>=0 and probability.max()<=1
    thresholds=json.loads((Q/'DIAGNOSTIC_PROTOCOL.json').read_text())['thresholds']
    all_gt=[];excluded=[]
    for fi,im in enumerate(images):
        gt,ex=prepare_gt(by_image[im['id']],*probability.shape[1:]);all_gt.append(gt)
        excluded.extend(dict(source_frame=im['frame_id'],**x) for x in ex)
    frames=[];rows=[];component_rows=[]
    for fi,im in enumerate(images):
        window_info=[(wi,int(fi-s)) for wi,(s,e) in enumerate(windows) if s<=fi<e]
        for th in thresholds:
            rr,fr,cc=match_frame(probability[fi],all_gt[fi],th)
            context={'sequence':seq,'local_frame':fi,'source_frame':im['frame_id'],
                     'threshold':th,'block':fi//40,'coverage':int(coverage[fi]),
                     'windows_and_slots':';'.join(f'{w}:{s}' for w,s in window_info)}
            rows.extend(dict(context,selected_track=int(r['track_id']==candidate['track_id']),**r) for r in rr)
            frames.append(dict(context,**fr))
            component_rows.extend(dict(context,**c) for c in cc)
    save_csv(out/'per_track_frame.csv',rows);save_csv(out/'per_frame_counts.csv',frames)
    save_csv(out/'components.csv',component_rows);save_json(out/'excluded_annotations.json',excluded)
    print(json.dumps({'sequence':seq,'phase':'fused_metrics_ready','threshold_0_5':{
        'matched':sum(r['matched'] for r in rows if r['threshold']==.5),
        'gt':sum(r['threshold']==.5 for r in rows)}}),flush=True)
    reconstruction=np.zeros_like(probability);reconstructed_coverage=np.zeros(160,dtype=np.int64)
    overlap_records=collections.defaultdict(list);window_files=sorted((run/'windows').glob('window*.npz'))
    assert len(window_files)==len(windows)
    for wi,wfile in enumerate(window_files):
        with np.load(wfile,allow_pickle=False) as saved:
            wp=saved['probabilities'];start=int(saved['start']);end=int(saved['end'])
        assert [start,end]==list(windows[wi]) and wp.shape==(40,*probability.shape[1:])
        reconstruction[start:end]=np.maximum(reconstruction[start:end],wp)
        reconstructed_coverage[start:end]+=1
        for fi in range(start,end):
            if coverage[fi]<2:continue
            for th in thresholds:
                rr,_,_=match_frame(wp[fi-start],all_gt[fi],th)
                for row in rr:
                    overlap_records[fi,th,row['track_id']].append(dict(row,window=wi,slot=fi-start))
    assert np.array_equal(reconstruction,probability), 'Saved max fusion differs from window reconstruction'
    assert np.array_equal(reconstructed_coverage,coverage)
    del reconstruction
    overlap=[]
    for (fi,th,tid),pair in sorted(overlap_records.items()):
        assert len(pair)==2
        left,right=pair
        overlap.append({'sequence':seq,'local_frame':fi,'source_frame':images[fi]['frame_id'],
                        'track_id':tid,'threshold':th,'selected_track':int(tid==candidate['track_id']),
                        'window_a':left['window'],'slot_a':left['slot'],
                        'window_b':right['window'],'slot_b':right['slot'],
                        'matched_a':left['matched'],'matched_b':right['matched'],
                        'match_flip':int(left['matched']!=right['matched']),
                        'ambiguous_a':left['association_ambiguous'],'ambiguous_b':right['association_ambiguous'],
                        'label_overlap_count':left['label_overlap_count'],
                        'label_identical_box_count':left['label_identical_box_count'],
                        'unambiguous_pair':int(not(left['association_ambiguous'] or right['association_ambiguous'] or left['label_overlap_count'])),
                        'boxmax_a':left['box_max'],'boxmax_b':right['box_max'],
                        'boxmax_absolute_difference':abs(left['box_max']-right['box_max']) if left['box_max'] is not None and right['box_max'] is not None else None,
                        'normalized_center_error_a':left['normalized_center_error'],
                        'normalized_center_error_b':right['normalized_center_error']})
    save_csv(out/'overlap_same_frame.csv',overlap)
    tracks=sorted({r['track_id'] for r in rows});track_summary=[];block_summary=[]
    for th in thresholds:
        for tid in tracks:
            subset=[r for r in rows if r['track_id']==tid and r['threshold']==th]
            track_summary.append(dict(track_id=tid,threshold=th,selected_track=tid==candidate['track_id'],**summarize_rows(subset)))
            for block in range(4):
                group=[r for r in subset if r['block']==block]
                stats=summarize_rows(group);stats.pop('unmatched_runs')
                block_summary.append(dict(track_id=tid,threshold=th,block=block,
                    double_covered_gt_frames=sum(r['coverage']==2 for r in group),**stats))
    save_json(out/'track_summary.json',track_summary);save_csv(out/'block_summary.csv',block_summary)
    primary=[]
    for th in thresholds:
        fr=[r for r in frames if r['threshold']==th];rr=[r for r in rows if r['threshold']==th]
        oo=[r for r in overlap if r['threshold']==th];unambiguous=[r for r in oo if r['unambiguous_pair']]
        primary.append({'threshold':th,'gt_frames':len(rr),'matched_gt_frames':sum(r['matched'] for r in rr),
                        'point_match_fraction':sum(r['matched'] for r in rr)/len(rr) if rr else None,
                        'components':sum(r['component_count'] for r in fr),
                        'unmatched_components_not_confirmed_fp':sum(r['unmatched_components'] for r in fr),
                        'ambiguous_gt_frames':sum(r['association_ambiguous'] for r in rr),
                        'label_overlapping_gt_frames':sum(r['label_overlap_count']>0 for r in rr),
                        'label_identical_box_gt_frames':sum(r['label_identical_box_count']>0 for r in rr),
                        'overlap_pairs':len(oo),'overlap_match_flips':sum(r['match_flip'] for r in oo),
                        'unambiguous_overlap_pairs':len(unambiguous),
                        'unambiguous_overlap_match_flips':sum(r['match_flip'] for r in unambiguous),
                        'selected_track':next(s for s in track_summary if s['track_id']==candidate['track_id'] and s['threshold']==th)})
    plot_all(out,rows,candidate['track_id'],seq)
    plot_contacts(out,data,images,probability,rows,track_summary,candidate['track_id'])
    inputs=[prediction,annotation_path,Q/'DIAGNOSTIC_PROTOCOL.json',*window_files]
    result={'sequence':seq,'status':'COMPLETE','candidate_id':candidate['candidate_id'],
            'selected_track_id':candidate['track_id'],'frame_count':160,'unique_tracks':len(tracks),
            'source_annotation_count':len(annotations['annotations']),'excluded_annotations':len(excluded),
            'fusion_reconstruction_bitexact':True,'coverage_reconstruction_exact':True,
            'primary':primary,'prediction_metadata':metadata,
            'input_sha256':{str(p):digest(p) for p in inputs},'analysis_code_sha256':digest(Path(__file__)),
            'runtime':{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,
                       'matplotlib':matplotlib.__version__,'platform':platform.platform()},
            'matching_detail':'Sorted GT(track_id,annotation_id); augmenting-path maximum cardinality; each adjacency ordered by normalized squared distance, descending score, component id. Deterministic but not global minimum-distance assignment.',
            'threshold_comparison':'>=','coordinate_convention':'pixel center x+0.5,y+0.5 in original half-open xywh; all slices clipped to frame',
            'limitations':json.loads((Q/'DIAGNOSTIC_PROTOCOL.json').read_text())['limitations']+[
                'This is bbox point-response diagnostic, not official Pd/Fa/IoU/AP.',
                'Window-context disagreement does not isolate slot, context content, padding or motion effects.',
                'Ground-truth identity conclusions exclude ambiguous match graphs; low thresholds can merge components.',
                'Four annotation-prescreened development clips are not independent random samples or a new benchmark.']}
    save_json(out/'SUMMARY.json',result)
    save_json(out/'ARTIFACT_SHA256.json',{p.name:digest(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='ARTIFACT_SHA256.json'})
    return result


def self_check():
    p=np.zeros((8,8),np.float32);p[1,1]=p[1,2]=.9;p[5,5]=.8
    anns=[{'id':1,'track_id':1,'bbox':[0,0,4,4]}, {'id':2,'track_id':2,'bbox':[0,0,8,8]}]
    gt,_=prepare_gt(anns,8,8);rows,frame,cc=match_frame(p,gt,.5)
    assert frame['matched_count']==2 and cc[0]['peak_x']==1.5 and cc[0]['peak_y']==1.5
    assert cc[0]['area']==2 and frame['components_multiple_gt']==1
    gt,excluded=prepare_gt([{'id':3,'track_id':3,'bbox':[-3,-2,5,4]}, {'id':4,'track_id':4,'bbox':[-5,0,2,2]}],8,8)
    assert len(gt)==1 and gt[0]['crop']==(0,0,2,2) and len(excluded)==1
    # Explicit rerouting case: GT1 initially takes the only candidate available to GT2.
    p=np.zeros((5,8),np.float32);p[1,1]=.9;p[1,4]=.8
    gt,_=prepare_gt([{'id':1,'track_id':1,'bbox':[0,0,6,4]}, {'id':2,'track_id':2,'bbox':[0,0,3,4]}],5,8)
    _,fr,_=match_frame(p,gt,.5);assert fr['matched_count']==2


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--sequence');args=parser.parse_args()
    D=args.root;Q=D/'temporal_diagnosis_20261010';self_check()
    candidates=json.loads((D/'real_motion_stage2/INSPECTION_QUEUE.json').read_text())['candidates'][:4]
    results=[]
    for candidate in candidates:
        if args.sequence and candidate['sequence']!=args.sequence:continue
        result=analyze_clip(D,Q,candidate);results.append(result)
        print(json.dumps({'sequence':result['sequence'],'status':result['status'],
                          'primary':result.get('primary')}),flush=True)
    save_json(Q/'analysis'/('RUN_'+args.sequence+'.json' if args.sequence else 'ALL_CLIPS_SUMMARY.json'),results)


if __name__=='__main__':main()
