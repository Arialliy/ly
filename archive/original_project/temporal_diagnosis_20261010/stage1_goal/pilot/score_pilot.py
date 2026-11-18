#!/usr/bin/env python3
"""Diagnostic point matching and unmatched components, never official Pd/Fa."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

THRESHOLDS = [1e-6,1e-5,1e-4,1e-3,.01,.03,.05,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.99,.999,.9999,.99999,.999999,1.]


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(8*1024*1024),b''): h.update(part)
    return h.hexdigest()


def helper():
    path=Path(__file__).resolve().parents[2]/'analysis/analyze.py'
    spec=importlib.util.spec_from_file_location('existing_point_diagnostic',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m,path


def gt_data(path,m):
    data=json.loads(path.read_text());images=data['images']
    assert [int(Path(im['file_name']).stem) for im in images]==sorted(int(Path(im['file_name']).stem) for im in images)
    lookup={im['id']:i for i,im in enumerate(images)}
    anns=[[] for _ in images]
    for a in data['annotations']:anns[lookup[a['image_id']]].append(a)
    allgt=[];meta=[];offsets=[0];excluded=[]
    for t,(im,aa) in enumerate(zip(images,anns)):
        gt,ex=m.prepare_gt(aa,im['height'],im['width']);allgt.append(gt);excluded.extend(ex)
        for g in gt:
            meta.append({'frame':t,'source_frame':int(Path(im['file_name']).stem),
                         'image_id':im['id'],'track_id':g['track_id'],'annotation_id':g['id'],
                         'category_id':g['category_id'],'bbox':g['bbox'],
                         'strict_small':int(max(g['bbox'][2:])<=16 and not g['partly_outside'] and not g['label_overlap_track_ids'])})
        offsets.append(len(meta))
    return allgt,meta,np.array(offsets),excluded


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True,help='One sequence materialized data directory')
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--context-threshold',type=float,help='Frozen calibration threshold; omit for full fused grid')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    completion=json.loads((a.predictions/'COMPLETE.json').read_text())
    if completion.get('status')!='COMPLETE' or completion['raw_sha256']!=digest(a.data/'raw_frames.npy'):
        raise ValueError('Prediction completion or input hash mismatch')
    expected_names=[f'window_{s:03d}.npz' for s in range(0,121,12)]
    actual_names=sorted(x.name for x in (a.predictions/'windows').glob('window_*.npz'))
    if actual_names!=expected_names or completion['diagnostic_starts']!=list(range(0,121,12)):
        raise ValueError('Incomplete or unexpected diagnostic windows')
    if completion['official_starts']!=[0,36,72,108,120]:
        raise ValueError('Unexpected official fusion protocol')
    m,helper_path=helper()
    gt,meta,offsets,excluded=gt_data(a.data/'annotations.json',m)
    thresholds=THRESHOLDS if a.context_threshold is None else [a.context_threshold]
    if a.context_threshold is None:
        entries=[('official_fused',a.predictions/'official_fused.npz',0)]
    else:
        entries=[(f.stem,f,int(f.stem.split('_')[-1])) for f in sorted((a.predictions/'windows').glob('window_*.npz'))]
    started=time.perf_counter()
    for name,path,start in entries:
        if (a.output/(name+'_COMPLETE.json')).exists():
            raise FileExistsError('Scoring output complete; refuse silent replacement')
        expected_hash=(completion['fused_sha256'] if name=='official_fused' else completion['window_sha256'][path.name])
        if digest(path)!=expected_hash:raise ValueError('Prediction hash mismatch: '+str(path))
        with np.load(path,allow_pickle=False) as z:
            prob=z['probabilities']
            if name=='official_fused':
                if not np.array_equal(z['starts'],[0,36,72,108,120]) or np.any(z['coverage']<=0):
                    raise ValueError('Fusion starts/coverage mismatch')
            elif int(z['start'])!=start or int(z['end'])!=start+40:
                raise ValueError('Window metadata and filename mismatch')
        expected_shape=(160,512,640) if name=='official_fused' else (40,512,640)
        if prob.shape!=expected_shape or not np.isfinite(prob).all() or np.any((prob<0)|(prob>1)):
            raise ValueError('Invalid probability map shape/range')
        if name=='official_fused':
            reconstructed=np.zeros_like(prob)
            for s in [0,36,72,108,120]:
                wp=a.predictions/'windows'/f'window_{s:03d}.npz'
                if digest(wp)!=completion['window_sha256'][wp.name]:raise ValueError('Window hash mismatch')
                with np.load(wp,allow_pickle=False) as z:
                    reconstructed[s:s+40]=np.maximum(reconstructed[s:s+40],z['probabilities'])
            if not np.array_equal(prob,reconstructed):raise ValueError('Official fusion is not exactly the five-window maximum')
            del reconstructed
        end=start+len(prob);n=int(offsets[end]-offsets[start]);nt=len(thresholds)
        matched=np.zeros((nt,n),np.uint8)
        center=np.full((nt,n),np.nan,np.float32)
        peakx=np.full((nt,n),np.nan,np.float32);peaky=peakx.copy()
        component_area=np.zeros((nt,n),np.int32)
        boxmax=np.zeros(n,np.float32)
        frame_counts=np.zeros((nt,len(prob),6),np.int64)
        strict=np.array([r['strict_small'] for r in meta[offsets[start]:offsets[end]]],bool)
        curves=[]
        for ti,threshold in enumerate(thresholds):
            for local,frame in enumerate(range(start,end)):
                rows,stats,_=m.match_frame(prob[local],gt[frame],threshold)
                j0=int(offsets[frame]-offsets[start]);j1=int(offsets[frame+1]-offsets[start])
                assert len(rows)==j1-j0
                for j,r in enumerate(rows,j0):
                    matched[ti,j]=r['matched']
                    if ti==0:boxmax[j]=r['box_max'] if r['box_max'] is not None else np.nan
                    if r['matched']:
                        center[ti,j]=r['normalized_center_error'];peakx[ti,j]=r['peak_x'];peaky[ti,j]=r['peak_y'];component_area[ti,j]=r['component_area']
                frame_counts[ti,local]=[stats[k] for k in ['gt_count','matched_count','component_count','unmatched_components','positive_pixels','positive_pixels_outside_boxes']]
            total=frame_counts[ti].sum(0)
            curves.append({'threshold':threshold,'frames':len(prob),'gt':int(total[0]),'matched':int(total[1]),
                           'strict_small_gt':int(strict.sum()),'strict_small_matched':int(matched[ti,strict].sum()),
                           'components':int(total[2]),'unmatched_components':int(total[3]),
                           'positive_pixels':int(total[4]),'positive_pixels_outside_boxes':int(total[5])})
        with (a.output/(name+'_curves.csv')).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(curves[0]));w.writeheader();w.writerows(curves)
        np.savez_compressed(a.output/(name+'_matches.npz'),thresholds=np.array(thresholds),matched=matched,
                            center_error=center,peak_x=peakx,peak_y=peaky,component_area=component_area,
                            box_max=boxmax,frame_counts=frame_counts,strict_small=strict,
                            offsets=offsets[start:end+1]-offsets[start])
        result={'status':'COMPLETE','name':name,'start':start,'end':end,'thresholds':thresholds,
                'prediction_sha256':digest(path),'annotations_sha256':digest(a.data/'annotations.json'),
                'scoring_script_sha256':digest(__file__),'matching_helper_sha256':digest(helper_path),
                'prediction_complete_and_raw_hash_verified':True,
                'official_fusion_exactly_reconstructed':name=='official_fused',
                'frame_count_fields':['gt','matched','components','unmatched_components','positive_pixels','positive_pixels_outside_boxes'],
                'objects':meta[offsets[start]:offsets[end]],'excluded_annotations':excluded,
                'elapsed_seconds_total':time.perf_counter()-started,
                'interpretation':'Diagnostic one-to-one peak-in-box match and unmatched components; not official Pd/Fa.'}
        (a.output/(name+'_COMPLETE.json')).write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'scored':name,'frames':len(prob),'gt':n,'seconds':time.perf_counter()-started}),flush=True)


if __name__=='__main__':main()
