#!/usr/bin/env python3
"""Frozen released-Plus layer1 regional means for one geometry-fixed probe.

All P0/P1/P2 primitive features are gathered from the SAME original model
forward. No feature model fitting, classification, threshold selection, or
annotation-dependent forward path is implemented. Query logits are exported
separately ONLY for the preregistered hard-background selection.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import time
import numpy as np

Q=Path('/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/temporal_diagnosis_20261010')
BASE=Path(__file__).resolve().parents[1]
RELEASE_SHA='4fd4aa2736d9a33f2bf1459099068d8b4a7e202ff80aecdb4a3277fd13c7abcc'
MODEL_SHA='fa46a034a30d8bbb0f677271d6368f033fcb389d682478d331551d8bae087668'
OLD_HELPER=Q/'top1_background_control_20261011/code/train_pilot.py'
HELPER_SHA='75e4fe9b4115af837cc1769471328a0b88fca8e546ec8d6f7db73d580f460b54'
STARTS=(0,36,72,108,120)


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(8*1024*1024),b''):result.update(data)
    return result.hexdigest()


def load_module(name,path):
    import sys
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def rectangle(value,height=512,width=640):
    raw=list(value)
    if len(raw)!=4 or any(int(v)!=v for v in raw):raise ValueError('ROIs must be integer half-open pixel rectangles')
    x0,y0,x1,y1=map(int,raw)
    if not (0<=x0<x1<=width and 0<=y0<y1<=height):raise ValueError(f'Invalid region {raw}')
    return x0,y0,x1,y1


def nearest_window(frame):
    valid=[s for s in STARTS if s<=frame<s+40]
    if not valid:raise ValueError('No frozen O window covers query')
    return min(valid,key=lambda s:(abs(frame-(s+19.5)),s))


def build_region_plan(samples,sequence):
    """Deduplicate exact regions; negative candidates get no outcome label here."""
    records=[];region_ids={};regions=[]
    def register(start,frame,rect,is_query):
        coords=rectangle(rect)
        key=(int(start),int(frame-start),*coords)
        if key not in region_ids:
            region_ids[key]=len(regions)
            regions.append(dict(start=key[0],slot=key[1],rect=list(coords),is_query=bool(is_query)))
        else:regions[region_ids[key]]['is_query']|=bool(is_query)
        return region_ids[key]
    for global_index,sample in enumerate(samples):
        if sample['sequence']!=sequence:continue
        start,frame,reference=map(int,(sample['start'],sample['frame'],sample['reference_frame']))
        if start!=nearest_window(frame) or abs(reference-frame)!=1 or not start<=reference<start+40:
            raise ValueError('Sample differs from frozen O-window and adjacent-frame rule')
        positive={k:sample[k] for k in ('q_rect_xyxy','r1_rect_xyxy','r2_rect_xyxy')}
        candidates=[(-1,positive)]+list(enumerate(sample['bg_candidates']))
        for candidate_index,candidate in candidates:
            qid=register(start,frame,candidate['q_rect_xyxy'],True)
            r1id=register(start,reference,candidate['r1_rect_xyxy'],False)
            r2id=register(start,reference,candidate['r2_rect_xyxy'],False)
            # These are output identifiers, NEVER classifier input features.
            records.append(dict(sample_index=global_index,candidate_index=candidate_index,
                q_region=qid,r1_region=r1id,r2_region=r2id))
    return records,regions


def tile_region_map(regions,start,height,width,core=128):
    mapping=defaultdict(list)
    for rid,region in enumerate(regions):
        if region['start']!=start:continue
        x0,y0,x1,y1=region['rect']
        for y in range((y0//core)*core,y1,core):
            for x in range((x0//core)*core,x1,core):
                xx0,yy0=max(x,x0),max(y,y0)
                xx1,yy1=min(x+core,x1,width),min(y+core,y1,height)
                if xx1>xx0 and yy1>yy0:mapping[y,x].append((rid,xx0,yy0,xx1,yy1))
    return dict(sorted(mapping.items()))


def self_test():
    sample=dict(sequence='toy',start=0,frame=19,reference_frame=20,
        q_rect_xyxy=[126,126,130,130],r1_rect_xyxy=[127,125,131,129],
        r2_rect_xyxy=[125,127,129,131],bg_candidates=[dict(q_rect_xyxy=[226,126,230,130],
        r1_rect_xyxy=[227,125,231,129],r2_rect_xyxy=[225,127,229,131])])
    records,regions=build_region_plan([sample],'toy')
    assert len(records)==2 and len(regions)==6
    tilemap=tile_region_map(regions,0,512,640)
    counts=np.zeros(len(regions),np.int64);sums=np.zeros((len(regions),32),np.float64)
    yy,xx=np.meshgrid(np.arange(512),np.arange(640),indexing='ij')
    # A deterministic analytic feature field, independently averaged per ROI.
    field=np.arange(32)[:,None,None]*.01+yy[None]*.1+xx[None]*.001
    for _,chunks in tilemap.items():
        for rid,x0,y0,x1,y1 in chunks:
            sums[rid]+=field[:,y0:y1,x0:x1].sum((1,2));counts[rid]+=(y1-y0)*(x1-x0)
    for rid,r in enumerate(regions):
        x0,y0,x1,y1=r['rect']
        assert counts[rid]==(x1-x0)*(y1-y0)
        np.testing.assert_allclose(sums[rid]/counts[rid],field[:,y0:y1,x0:x1].mean((1,2)),rtol=0,atol=1e-12)
    assert nearest_window(39)==36 and nearest_window(80)==72 and nearest_window(159)==120
    assert build_region_plan([sample],'absent')==([],[])
    print(json.dumps({'status':'PASS','scope':'geometry-only deduplication, cross-tile ROI means, frozen O-window selection; no model/device/data inference'}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    for name in ('samples','source','checkpoint','dataset-root','output','protocol-freeze'):
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--sequence')
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    if args.self_test:self_test();return
    if any(getattr(args,name) is None for name in ('samples','source','checkpoint','dataset_root','output','protocol_freeze','sequence')):
        parser.error('All extraction paths, protocol freeze, and sequence are required')
    import torch
    freeze=json.loads(args.protocol_freeze.read_text())
    code_sha=sha(__file__);sample_sha=sha(args.samples)
    if freeze['code_sha256']['code/extract_probe_features.py']!=code_sha or freeze['samples_sha256']!=sample_sha:
        raise AssertionError('Extractor and geometry samples must be frozen before reading data')
    if sha(args.checkpoint)!=RELEASE_SHA or sha(OLD_HELPER)!=HELPER_SHA:
        raise AssertionError('Only the original released Plus and unchanged helper are allowed')
    original_model=args.checkpoint.parent.parent/'DeepPro-Plus.py'
    if sha(original_model)!=MODEL_SHA:raise AssertionError('Original checkpoint-adjacent model source changed')
    if args.output.suffix != '.npz':raise ValueError('Feature output must use .npz suffix')
    if args.output.exists() or args.output.with_suffix('.json').exists():raise FileExistsError('Refuse replacement of feature artifacts')
    proposal=json.loads(args.samples.read_text())
    samples=proposal['samples']
    records,regions=build_region_plan(samples,args.sequence)
    # Even absent common support is explicitly recorded, never silently expanded.
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if not records:
        np.savez_compressed(args.output,q=np.zeros((0,32),np.float32),r1=np.zeros((0,32),np.float32),
            r2=np.zeros((0,32),np.float32),query_max_logit=np.zeros(0,np.float32),
            sample_index=np.zeros(0,np.int64),candidate_index=np.zeros(0,np.int32))
        args.output.with_suffix('.json').write_text(json.dumps(dict(status='COMPLETE_EMPTY_SUPPORT',sequence=args.sequence,
            rows=0,source_sha256=code_sha,samples_sha256=sample_sha,samples_path=str(args.samples.resolve()),output_path=str(args.output.resolve()),protocol_freeze_sha256=sha(args.protocol_freeze),
            checkpoint_sha256=RELEASE_SHA,output_sha256=sha(args.output),no_model_inference=True),indent=2)+'\n')
        print(json.dumps({'status':'COMPLETE_EMPTY_SUPPORT','sequence':args.sequence}));return
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    random.seed(46);np.random.seed(46);torch.manual_seed(46);torch.cuda.manual_seed_all(46)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device(args.device)
    helper=load_module('frozen_probe_original_helper',OLD_HELPER)
    model,loaded_source=helper.load_model(args.source,args.checkpoint)
    if loaded_source.resolve()!=original_model.resolve():raise AssertionError('Unexpected model import')
    if sum(p.numel() for p in model.parameters())!=70913:raise AssertionError('Not original Plus')
    if getattr(model.layer1,'_forward_hooks',{}):raise AssertionError('Original feature layer already has a hook')
    for p in model.parameters():p.requires_grad_(False)
    model.to(device).eval()
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    captured={}
    handle=model.layer1.register_forward_hook(lambda layer,inputs,out:captured.__setitem__('features',out))
    raw_path=args.dataset_root/args.sequence/'raw_frames.npy'
    raw=np.load(raw_path,mmap_mode='r',allow_pickle=False)
    if raw.shape!=(160,512,640) or raw.dtype!=np.uint8:raise AssertionError('Frozen source shape/dtype mismatch')
    sums=np.zeros((len(regions),32),np.float64);counts=np.zeros(len(regions),np.int64)
    maxima=np.full(len(regions),-np.inf,np.float64)
    core,halo=128,20;timings=[];started=time.perf_counter()
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for start in sorted({r['start'] for r in regions}):
            tilemap=tile_region_map(regions,start,512,640,core)
            window=(raw[start:start+40].astype(np.float32)-85.0799)/47.4845
            stamp=time.perf_counter()
            for (y,x),parts in tilemap.items():
                y1,x1=min(y+core,512),min(x+core,640)
                ya,yb=max(0,y-halo),min(512,y1+halo)
                xa,xb=max(0,x-halo),min(640,x1+halo)
                image=torch.from_numpy(np.ascontiguousarray(window[:,ya:yb,xa:xb]))[None,None].to(device)
                logits=model(image)[1]
                features=captured.pop('features')
                if features.shape[:3]!=(1,32,40) or logits.shape[:2]!=(1,40):raise AssertionError('Unexpected original tensor shapes')
                for rid,rx0,ry0,rx1,ry1 in parts:
                    slot=regions[rid]['slot']
                    patch=features[0,:,slot,ry0-ya:ry1-ya,rx0-xa:rx1-xa]
                    sums[rid]+=patch.sum(dim=(1,2),dtype=torch.float64).cpu().numpy()
                    counts[rid]+=(ry1-ry0)*(rx1-rx0)
                    if regions[rid]['is_query']:
                        maxima[rid]=max(maxima[rid],float(logits[0,slot,ry0-ya:ry1-ya,rx0-xa:rx1-xa].max()))
                del features,logits,image
            if device.type=='cuda':torch.cuda.synchronize(device)
            timings.append(dict(start=start,tiles=len(tilemap),seconds=time.perf_counter()-stamp))
            print(json.dumps(dict(sequence=args.sequence,start=start,tiles=len(tilemap),phase='features_extracted')),flush=True)
    handle.remove()
    if captured:raise AssertionError('Unexpected captured tensor retained')
    expected=np.asarray([(r['rect'][2]-r['rect'][0])*(r['rect'][3]-r['rect'][1]) for r in regions])
    if not np.array_equal(counts,expected):raise AssertionError('ROI core partition missed or duplicated pixels')
    means=(sums/counts[:,None]).astype(np.float32)
    arrays={name:np.stack([means[r[key]] for r in records]) for name,key in [('q','q_region'),('r1','r1_region'),('r2','r2_region')]}
    arrays.update(query_max_logit=np.asarray([maxima[r['q_region']] for r in records],np.float32),
        sample_index=np.asarray([r['sample_index'] for r in records],np.int64),
        candidate_index=np.asarray([r['candidate_index'] for r in records],np.int32))
    if not all(np.isfinite(a).all() for a in arrays.values()):raise FloatingPointError('Nonfinite extracted feature')
    if any(not torch.equal(initial[k],v.detach().cpu()) for k,v in model.state_dict().items()):raise AssertionError('Frozen weights or buffers changed')
    np.savez_compressed(args.output,**arrays)
    info=dict(status='COMPLETE',sequence=args.sequence,rows=len(records),positive_geometry_rows=sum(r['candidate_index']==-1 for r in records),
        candidate_background_rows=sum(r['candidate_index']>=0 for r in records),unique_regions=len(regions),
        feature_definition='32 channel-wise spatial means of original released layer1 output; no new aggregation module',
        features_shared_across_conditions=True,temporal_context='full original40-frame Owindow; regional references only t+-1',
        raw_logit_role='query_max_logit solely selects preregistered hard background candidates; never a readout feature',
        readout_feature_whitelist=['q','r1','r2'],metadata_forbidden_as_features=['sample_index','candidate_index','query_max_logit','GT','coordinates','pixel_count'],
        core=core,halo=halo,normalization=[85.0799,47.4845],seed=46,precision='FP32 model; float64 regional summation then float32 feature',
        parameter_count=70913,all_parameters_frozen=True,model_eval=True,state_bitwise_unchanged=True,
        checkpoint_sha256=RELEASE_SHA,model_source_sha256=MODEL_SHA,helper_sha256=HELPER_SHA,
        source_sha256=code_sha,samples_sha256=sample_sha,samples_path=str(args.samples.resolve()),output_path=str(args.output.resolve()),protocol_freeze_sha256=sha(args.protocol_freeze),raw_sha256=sha(raw_path),
        output_sha256=sha(args.output),elapsed_seconds=time.perf_counter()-started,window_timing=timings,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0,
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else 0,
        limitation='This exported oracle-region probe is not a deployable detector or whole-model single-frame comparison.')
    args.output.with_suffix('.json').write_text(json.dumps(info,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({'status':'COMPLETE','sequence':args.sequence,'rows':len(records),'regions':len(regions)}))

if __name__=='__main__':main()
