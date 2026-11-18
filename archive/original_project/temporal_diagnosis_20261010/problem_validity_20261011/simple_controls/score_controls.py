#!/usr/bin/env python3
"""CPU-only, fixed five-window fusion controls; preserve original matching rules."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np


OFFICIAL = [0, 36, 72, 108, 120]
BALANCED = [0, 24, 60, 96, 120]
EXPECTED_CKPT = '4fd4aa2736d9a33f2bf1459099068d8b4a7e202ff80aecdb4a3277fd13c7abcc'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(8*1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def read_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def scores(prob, gt, meta, offsets, thresholds, matcher, label):
    n = len(meta)
    nt = len(thresholds)
    strict = np.asarray([r['strict_small'] for r in meta], bool)
    result = dict(thresholds=np.asarray(thresholds, dtype=np.float64),
                  matched=np.zeros((nt,n), np.uint8),
                  center_error=np.full((nt,n), np.nan, np.float32),
                  peak_x=np.full((nt,n), np.nan, np.float32),
                  peak_y=np.full((nt,n), np.nan, np.float32),
                  component_area=np.zeros((nt,n), np.int32),
                  box_max=np.zeros(n, np.float32),
                  frame_counts=np.zeros((nt,160,6), np.int64),
                  strict_small=strict, offsets=offsets)
    curves = []
    fields = ['gt_count','matched_count','component_count','unmatched_components',
              'positive_pixels','positive_pixels_outside_boxes']
    for ti, threshold in enumerate(thresholds):
        for t in range(160):
            rows, stats, _ = matcher.match_frame(prob[t], gt[t], threshold)
            a, b = int(offsets[t]), int(offsets[t+1])
            assert len(rows) == b-a
            for j, r in enumerate(rows,a):
                assert r['annotation_id'] == meta[j]['annotation_id']
                result['matched'][ti,j] = r['matched']
                if ti == 0:
                    result['box_max'][j] = r['box_max'] if r['box_max'] is not None else np.nan
                if r['matched']:
                    for target, source in [('center_error','normalized_center_error'),
                                           ('peak_x','peak_x'),('peak_y','peak_y'),
                                           ('component_area','component_area')]:
                        result[target][ti,j] = r[source]
            result['frame_counts'][ti,t] = [stats[k] for k in fields]
        total = result['frame_counts'][ti].sum(0)
        curves.append(dict(threshold=float(threshold),frames=160,gt=int(total[0]),matched=int(total[1]),
                           strict_small_gt=int(strict.sum()),strict_small_matched=int(result['matched'][ti,strict].sum()),
                           components=int(total[2]),unmatched_components=int(total[3]),
                           positive_pixels=int(total[4]),positive_pixels_outside_boxes=int(total[5])))
        print(json.dumps(dict(control=label, threshold_complete=float(threshold))),flush=True)
    return result, curves


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--diagnosis-root',type=Path,required=True)
    p.add_argument('--sequence',choices=['DJI_0026_2','DJI_0067_2'],required=True)
    p.add_argument('--output-root',type=Path,required=True)
    a=p.parse_args()
    begin=time.perf_counter()
    stage=a.diagnosis_root/'stage1_goal'
    predictions=stage/'predictions/official'/a.sequence
    data=stage/'data'/a.sequence
    old_eval=stage/'evaluation/official'/a.sequence
    old_score_path=stage/'pilot/score_pilot.py'
    original_scorer=read_module('readonly_original_scorer',old_score_path)
    matcher,matcher_path=original_scorer.helper()
    gt,meta,offsets,excluded=original_scorer.gt_data(data/'annotations.json',matcher)
    completion=json.loads((predictions/'COMPLETE.json').read_text())
    assert completion['status']=='COMPLETE'
    assert completion['checkpoint_sha256']==completion['reference_checkpoint_sha256']==EXPECTED_CKPT
    assert digest(completion['checkpoint'])==EXPECTED_CKPT
    assert digest(completion['model_source'])==completion['model_source_sha256']
    assert completion['official_starts']==OFFICIAL
    assert completion['diagnostic_starts']==list(range(0,121,12))
    assert completion['normalization']==[85.0799,47.4845]
    assert completion['tile_core']==128 and completion['tile_halo']==16
    assert completion['precision']=='FP32'
    assert completion['receptive_radius_tyx']==[8,9,9]
    assert digest(data/'raw_frames.npy')==completion['raw_sha256']
    names=sorted(f.name for f in (predictions/'windows').glob('*.npz'))
    assert names==[f'window_{s:03d}.npz' for s in range(0,121,12)]
    selected={}
    selected_hashes={}
    for start in sorted(set(OFFICIAL+BALANCED)):
        path=predictions/'windows'/f'window_{start:03d}.npz'
        sha=digest(path)
        assert sha==completion['window_sha256'][path.name]
        with np.load(path,allow_pickle=False) as z:
            assert int(z['start'])==start and int(z['end'])==start+40
            prob=z['probabilities']
        assert prob.dtype==np.float32 and prob.shape==(40,512,640)
        assert np.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all()
        selected[start]=prob
        selected_hashes[str(start)]=sha
    official=np.zeros((160,512,640),np.float32)
    for start in OFFICIAL:
        official[start:start+40]=np.maximum(official[start:start+40],selected[start])
    official_path=predictions/'official_fused.npz'
    assert digest(official_path)==completion['fused_sha256']
    with np.load(official_path,allow_pickle=False) as z:
        assert np.array_equal(z['starts'],OFFICIAL)
        assert np.array_equal(z['probabilities'],official)
        assert np.all(z['coverage']>0)
    old_complete=json.loads((old_eval/'official_fused_COMPLETE.json').read_text())
    assert old_complete['objects']==meta
    assert old_complete['matching_helper_sha256']==digest(matcher_path)
    assert old_complete['annotations_sha256']==digest(data/'annotations.json')
    # Real full-sequence baseline at the frozen operating point validates the adapter.
    baseline_check,_=scores(official,gt,meta,offsets,[0.5],matcher,'baseline_adapter_check')
    with np.load(old_eval/'official_fused_matches.npz',allow_pickle=False) as z:
        idx=np.flatnonzero(z['thresholds']==0.5)
        assert len(idx)==1
        idx=int(idx[0])
        for key in ['matched','center_error','peak_x','peak_y','component_area','frame_counts']:
            assert np.array_equal(baseline_check[key][0],z[key][idx],equal_nan=True),key
        for key in ['box_max','strict_small','offsets']:
            assert np.array_equal(baseline_check[key],z[key],equal_nan=True),key
    a.output_root.mkdir(parents=True,exist_ok=True)
    verify_path=a.output_root/(a.sequence+'_BASELINE_ADAPTER_CHECK.json')
    if verify_path.exists():
        raise FileExistsError(verify_path)
    save_json(verify_path,dict(status='PASS',sequence=a.sequence,frames=160,threshold=0.5,
                             object_count=len(meta),official_fused_exactly_reconstructed=True,
                             fields_exactly_equal=['matched','center_error','peak_x','peak_y','component_area',
                                                   'frame_counts','box_max','strict_small','offsets'],
                             old_matches_sha256=digest(old_eval/'official_fused_matches.npz')))
    del baseline_check,official
    for name,starts in [('official5_probmean',OFFICIAL),('symmetric5_probmax',BALANCED)]:
        out=a.output_root/name/a.sequence
        if (out/'fused_COMPLETE.json').exists():
            raise FileExistsError(out/'fused_COMPLETE.json')
        out.mkdir(parents=True,exist_ok=True)
        cov=np.zeros(160,np.int32)
        fused=np.zeros((160,512,640),np.float64 if name.endswith('probmean') else np.float32)
        for start in starts:
            assert 0<=start<=120
            cov[start:start+40]+=1
            if name.endswith('probmean'):
                fused[start:start+40]+=selected[start]
            else:
                fused[start:start+40]=np.maximum(fused[start:start+40],selected[start])
        assert len(starts)==5 and int(cov.sum())==200 and np.all(cov>0)
        assert all(0<=t-s<=39 for s in starts for t in range(s,s+40))
        if name.endswith('probmean'):
            fused=(fused/cov[:,None,None]).astype(np.float32)
        # Independent per-frame formula verifies temporal indexing and fusion arithmetic.
        for t in range(160):
            eligible=[selected[s][t-s] for s in starts if s<=t<s+40]
            if name.endswith('probmean'):
                expected=np.mean(np.stack(eligible).astype(np.float64),axis=0).astype(np.float32)
            else:
                expected=np.max(np.stack(eligible),axis=0)
            assert np.array_equal(fused[t],expected),t
        predpath=out/'fused.npz'
        np.savez_compressed(predpath,probabilities=fused,coverage=cov,starts=np.asarray(starts))
        result,curves=scores(fused,gt,meta,offsets,original_scorer.THRESHOLDS,matcher,name)
        curvepath=out/'fused_curves.csv'
        with curvepath.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(curves[0]));w.writeheader();w.writerows(curves)
        matchpath=out/'fused_matches.npz'
        np.savez_compressed(matchpath,**result)
        record=dict(status='COMPLETE',name='fused',sequence=a.sequence,start=0,end=160,
                    thresholds=original_scorer.THRESHOLDS,fusion_control=name,
                    fusion_algorithm=('float64 sum ascending starts / coverage, cast float32' if name.endswith('probmean')
                                      else 'pixelwise float32 probability maximum'),
                    starts=starts,windows=5,window_length=40,window_frame_budget=200,
                    temporal_offsets=[0,39],offline_access_range=[0,159],
                    coverage_min=int(cov.min()),coverage_max=int(cov.max()),
                    model_source_sha256=completion['model_source_sha256'],
                    checkpoint_sha256=EXPECTED_CKPT,raw_sha256=completion['raw_sha256'],
                    source_windows_sha256={str(s):selected_hashes[str(s)] for s in starts},
                    source_complete_sha256=digest(predictions/'COMPLETE.json'),
                    prediction_sha256=digest(predpath),annotations_sha256=digest(data/'annotations.json'),
                    scoring_script_sha256=digest(__file__),matching_helper_sha256=digest(matcher_path),
                    source_score_helper_sha256=digest(old_score_path),
                    frame_count_fields=['gt','matched','components','unmatched_components','positive_pixels','positive_pixels_outside_boxes'],
                    objects=meta,excluded_annotations=excluded,
                    full_frame_fusion_formula_verified=True,baseline_adapter_check='PASS',
                    matches_sha256=digest(matchpath),curves_sha256=digest(curvepath),
                    gpu_inference_performed=False,source_labels_used_for_window_selection=False,
                    elapsed_seconds_total=time.perf_counter()-begin,
                    interpretation='Diagnostic one-to-one peak-in-box matches and unmatched components; not official Pd/Fa. Development controls, no independent confirmation.')
        save_json(out/'fused_COMPLETE.json',record)
        print(json.dumps(dict(status='COMPLETE',sequence=a.sequence,control=name,elapsed_seconds=time.perf_counter()-begin)),flush=True)
        del fused,result


if __name__=='__main__':
    main()
