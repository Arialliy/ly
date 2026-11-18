"""Frozen release model: C1 roundtrip inputs, C2 anchor inputs and restored scores."""
import argparse
import json
import time
import numpy as np
import torch
from runtime import *
from geometry import prepare,restore,apply_fallback

def run(seq):
    assert json.loads((E/'checks/RUNTIME_EQUIVALENCE.json').read_text())['status']=='PASS'
    assert json.loads((E/'checks/TRANSFORM_TESTS.json').read_text())['status']=='PASS'
    freeze=json.loads((E/'PROTOCOL_FREEZE.json').read_text())
    for p,h in freeze['files'].items():assert sha(E/p)==h,p
    setup();model,mp=load_model()
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    rawpath=B/'data/stage1'/seq/'raw_frames.npy';raw=np.load(rawpath,mmap_mode='r')
    assert raw.shape==(160,512,640) and raw.dtype==np.uint8
    reference=json.loads((C0/seq/'COMPLETE.json').read_text())
    assert reference['raw_sha256']==sha(rawpath)
    assert reference['checkpoint_sha256']==sha(CP)
    assert reference['model_source_sha256']==sha(mp)
    meta={};fused={}
    for c in ['C1','C2']:
        dest=E/'predictions'/c/seq
        if dest.exists() and any(dest.iterdir()):raise FileExistsError(dest)
        (dest/'windows').mkdir(parents=True)
        fused[c]=np.zeros(raw.shape,np.float32)
        meta[c]={'status':'RUNNING','condition':c,'sequence':seq,'checkpoint_sha256':sha(CP),'model_source_sha256':sha(mp),
                 'raw_sha256':sha(rawpath),'O':O,'S':S,'starts_union':STARTS,'anchor_offset':19,'parameters':70913,
                 'threshold':.5,'normalization':[85.0799,47.4845],'model_labels_access':False,'training':False,
                 'precision':'FP32_no_AMP_no_TF32','tile_core':128,'tile_halo':16,'window_timing':[],
                 'protocol_freeze_sha256':sha(E/'PROTOCOL_FREEZE.json'),'torch':torch.__version__}
    coverage=np.zeros(160,np.int32)
    with torch.inference_mode():
        for start in STARTS:
            t0=time.perf_counter()
            gp=E/'motion'/seq/'windows'/f'window_{start:03d}.npz'
            with np.load(gp,allow_pickle=False) as z:
                H=z['H'];usable=bool(z['window_trusted'])
                assert H.shape==(40,3,3)
                assert int(z['start'])==start and int(z['anchor'])==start+19
                assert str(z['config_sha256'])==sha(E/'motion/CONFIG.json')
                assert usable==bool(z['trusted'].all())
            bp=C0/seq/'windows'/f'window_{start:03d}.npz'
            assert sha(bp)==reference['window_sha256'][bp.name]
            with np.load(bp,allow_pickle=False) as z:base=z['probabilities']
            if usable:c1,c2,valid=prepare(raw[start:start+40],H)
            else:c1=c2=None;valid=np.zeros(base.shape,bool)
            geom_seconds=time.perf_counter()-t0
            outmask=E/'geometry'/seq/'windows';outmask.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(outmask/bp.name,valid=valid,usable_window=usable,start=start,end=start+40)
            for c,arr in [('C1',c1),('C2',c2)]:
                seconds=0.;restore_seconds=0.;peak=0
                if usable:
                    x=torch.from_numpy((arr-85.0799)/47.4845)[None,None].cuda()
                    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.perf_counter()
                    pred=tiled_logits(model,x).sigmoid()[0].numpy();torch.cuda.synchronize()
                    seconds=time.perf_counter()-tick;peak=torch.cuda.max_memory_allocated()
                    tick=time.perf_counter()
                    if c=='C2':pred=restore(pred,H)
                    pred=apply_fallback(pred,base,valid);restore_seconds=time.perf_counter()-tick
                    del x
                else:pred=base.copy()
                assert pred.dtype==np.float32 and np.isfinite(pred).all() and pred.min()>=0 and pred.max()<=1
                assert np.array_equal(pred[~valid],base[~valid])
                dest=E/'predictions'/c/seq
                tick=time.perf_counter();np.savez_compressed(dest/'windows'/bp.name,probabilities=pred,start=start,end=start+40)
                write_seconds=time.perf_counter()-tick
                if start in O:fused[c][start:start+40]=np.maximum(fused[c][start:start+40],pred)
                row={'start':start,'usable_window':usable,'valid_fraction':float(valid.mean()),'model_seconds':seconds,
                     'restoration_fallback_seconds':restore_seconds,'shared_geometry_cache_seconds':geom_seconds,
                     'write_seconds':write_seconds,'peak_cuda_allocated_bytes':peak,'motion_sha256':sha(gp)}
                meta[c]['window_timing'].append(row);print(json.dumps({'condition':c,'sequence':seq,**row}),flush=True)
            if start in O:coverage[start:start+40]+=1
            del c1,c2,base,pred,valid
    assert all(torch.equal(v.detach().cpu(),initial[k]) for k,v in model.state_dict().items())
    for c in meta:
        dest=E/'predictions'/c/seq
        np.savez_compressed(dest/'official_fused.npz',probabilities=fused[c],coverage=coverage,starts=np.array(O))
        meta[c].update(status='COMPLETE',all_model_state_unchanged=True,window_sha256={p.name:sha(p) for p in (dest/'windows').glob('*.npz')},
            fused_sha256=sha(dest/'official_fused.npz'),O_model_seconds=sum(r['model_seconds'] for r in meta[c]['window_timing'] if r['start'] in O),
            S_model_seconds=sum(r['model_seconds'] for r in meta[c]['window_timing'] if r['start'] in S))
        save(dest/'COMPLETE.json',meta[c])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sequence',choices=SEQS,required=True);run(p.parse_args().sequence)
