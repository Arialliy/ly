"""Final512 full-frame inference; T normal/current/bypass share one loaded checkpoint."""
import argparse, gc, json, time
from pathlib import Path
import numpy as np
import torch
import train_tlfp as r
from temporal_lfp import spatially_tiled_forward
O=[0,36,72,108,120];S=[0,24,60,96,120];STARTS=sorted(set(O+S))
SOURCE=r.Q.parent/'official_deeppro'
REFERENCE=SOURCE/'log/sem_seg/RGB-T__2025-01-14_15-27__SoftLoUloss_DeepPro-Plus_DataL40/checkpoints/best_model.pth'

def tiled_logits(model,x,core=128,halo=20):
    # Whole40-frame time retained; starts+halo even in original fullimage grid.
    h,w=x.shape[-2:];output=torch.empty((x.shape[0],40,h,w),dtype=torch.float32,device='cpu')
    for y in range(0,h,core):
        y1=min(h,y+core);ya=max(0,y-halo)//2*2;yb=min(h,((y1+halo+1)//2)*2)
        for z in range(0,w,core):
            z1=min(w,z+core);za=max(0,z-halo)//2*2;zb=min(w,((z1+halo+1)//2)*2)
            result=model(x[...,ya:yb,za:zb].contiguous())[1]
            output[...,y:y1,z:z1]=result[...,y-ya:y1-ya,z-za:z1-za].cpu()
    return output

def main():
    p=argparse.ArgumentParser();p.add_argument('--condition',choices=['S','T'],required=True);p.add_argument('--sequence',choices=['DJI_0026_2','DJI_0067_2'],required=True);a=p.parse_args()
    r.deterministic_setup();variant=a.condition
    base=r.load_module('tlfp_frozen_batch',r.Q/'top1_background_control_20261011/code/train_pilot.py')
    original,modelpath=base.load_model(SOURCE,REFERENCE);original_hash=r.tensor_hash(original.state_dict().items())
    checkpoint=r.N/'training'/variant/'run/final_model.pth';cp=torch.load(checkpoint,map_location='cpu',weights_only=True)
    run=json.loads(checkpoint.with_name('RUN.json').read_text());sources=r.source_hashes(modelpath)
    assert run['status']=='complete' and cp['step']==512 and cp['condition']==variant
    assert r.sha(checkpoint)==run['final_checkpoint_sha256']
    assert cp['initial_checkpoint_sha256']==r.sha(REFERENCE)==r.EXPECTED['checkpoint']
    assert all(run[k]==v and cp[k]==v for k,v in sources.items())
    model=r.wrap_model(original,variant);model.load_state_dict(cp['model_state_dict'],strict=True);model=model.cuda().eval()
    assert original_hash==r.tensor_hash(model.original.state_dict().items())==cp['original_state_sha256']
    model_hash=r.tensor_hash(model.state_dict().items())
    raw_path=r.Q/'stage1_goal/data'/a.sequence/'raw_frames.npy';ann_path=raw_path.with_name('annotations.json')
    raw=np.load(raw_path,mmap_mode='r',allow_pickle=False);assert raw.shape==(160,512,640) and raw.dtype==np.uint8
    conditions={'S':('normal','spatial')} if variant=='S' else {'T':('normal','temporal'),'T_current':('current_only','current_only'),'T_bypass':('bypass','bypass')}
    meta={};fused={};coverage={}
    for label,(mode,diagnostic) in conditions.items():
        dest=r.N/'predictions'/label/a.sequence
        if dest.exists() and any(dest.iterdir()):raise FileExistsError(dest)
        (dest/'windows').mkdir(parents=True,exist_ok=True)
        fused[label]=np.zeros(raw.shape,dtype=np.float32);coverage[label]=np.zeros(160,dtype=np.int32)
        meta[label]={'status':'RUNNING','condition':label,'training_variant':variant,'sequence':a.sequence,'diagnostic_mode':diagnostic,'mode':mode,'checkpoint':str(checkpoint),'checkpoint_sha256':r.sha(checkpoint),'checkpoint_step':512,'main_checkpoint_step':512,'checkpoint_is_main':True,'reference_checkpoint_sha256':r.sha(REFERENCE),'original_initial_state_sha256':original_hash,'normalization':[85.0799,47.4845],'official_starts':O,'symmetric_starts':S,'diagnostic_starts':STARTS,'raw_sha256':r.sha(raw_path),'annotations_sha256':r.sha(ann_path),'model_labels_access':False,'model_config':model.config,'parameters':sum(p.numel() for p in model.parameters()),'trainable_parameters':100 if variant=='S' else 296,'tile_core':128,'tile_halo':20,'spatial_phase':'absolute even tile input starts','seed':46,'precision':'FP32_no_AMP_no_TF32','window_timing':[],'tiling_checks':[],**sources}
    begin=time.perf_counter()
    try:
        with torch.no_grad():
            probe=torch.from_numpy(np.array((raw[:40,176:272,224:352].astype(np.float32)-85.0799)/47.4845,copy=True))[None,None].cuda()
            for label,(mode,_) in conditions.items():
                model.feature_refiner.set_inference_mode(mode)
                direct=model(probe)[1].cpu();tiled=tiled_logits(model,probe,48,20)
                torch.testing.assert_close(direct,tiled,atol=2e-4,rtol=2e-5)
                disagreement=int(((direct.sigmoid()>=.5)!=(tiled.sigmoid()>=.5)).sum());assert disagreement==0
                meta[label]['tiling_checks'].append({'crop_yxhw':[176,224,96,128],'max_logit_error':float((direct-tiled).abs().max()),'max_probability_error':float((direct.sigmoid()-tiled.sigmoid()).abs().max()),'threshold05_disagreements':disagreement,'atol':2e-4,'rtol':2e-5})
            del probe,direct,tiled
            for start in STARTS:
                image=torch.from_numpy(np.array((raw[start:start+40].astype(np.float32)-85.0799)/47.4845,copy=True))[None,None].cuda()
                for label,(mode,_) in conditions.items():
                    model.feature_refiner.set_inference_mode(mode);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();stamp=time.perf_counter()
                    logits=tiled_logits(model,image);torch.cuda.synchronize();model_seconds=time.perf_counter()-stamp
                    prob=logits.sigmoid()[0].numpy();assert np.isfinite(prob).all()
                    dest=r.N/'predictions'/label/a.sequence;stamp=time.perf_counter()
                    np.savez_compressed(dest/'windows'/f'window_{start:03d}.npz',probabilities=prob,start=start,end=start+40)
                    if start in O:fused[label][start:start+40]=np.maximum(fused[label][start:start+40],prob);coverage[label][start:start+40]+=1
                    meta[label]['window_timing'].append({'start':start,'model_plus_cpu_copy_seconds':model_seconds,'window_write_seconds':time.perf_counter()-stamp,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved()})
                    print(json.dumps({'condition':label,'sequence':a.sequence,'window':start,'seconds':model_seconds}),flush=True)
                    del logits,prob
                del image
        assert original_hash==r.tensor_hash(model.original.state_dict().items()) and model_hash==r.tensor_hash(model.state_dict().items())
        assert sources==r.source_hashes(modelpath)
        for label in conditions:
            dest=r.N/'predictions'/label/a.sequence;assert np.all(coverage[label]>0)
            np.savez_compressed(dest/'official_fused.npz',probabilities=fused[label],coverage=coverage[label],starts=np.array(O))
            times=meta[label]['window_timing']
            meta[label].update(status='COMPLETE',elapsed_seconds=sum(t['model_plus_cpu_copy_seconds']+t['window_write_seconds'] for t in times),bundle_elapsed_seconds=time.perf_counter()-begin,original_final_state_sha256=original_hash,frozen_original_unchanged=True,all_model_state_unchanged=True,peak_cuda_allocated_bytes=max(t['peak_cuda_allocated_bytes'] for t in times),peak_cuda_reserved_bytes=max(t['peak_cuda_reserved_bytes'] for t in times),O_model_seconds=sum(t['model_plus_cpu_copy_seconds'] for t in times if t['start'] in O),S_model_seconds=sum(t['model_plus_cpu_copy_seconds'] for t in times if t['start'] in S),window_sha256={f.name:r.sha(f) for f in sorted((dest/'windows').glob('*.npz'))},fused_sha256=r.sha(dest/'official_fused.npz'))
            r.save_json(dest/'COMPLETE.json',meta[label])
    except BaseException as e:
        for label in conditions:r.save_json(r.N/'predictions'/label/a.sequence/'FAILED.json',{'status':'FAILED','error':repr(e),'partial_metadata':meta[label]})
        raise
if __name__=='__main__':main()
