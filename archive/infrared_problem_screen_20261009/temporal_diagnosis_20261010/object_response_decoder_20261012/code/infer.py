"""Fixed final2048: full-frame shared queries, original O/S inference access."""
import argparse,json,time
import common as c
OFFICIAL=[0,36,72,108,120];SYMMETRIC=[0,24,60,96,120];STARTS=sorted(set(OFFICIAL+SYMMETRIC))

def main():
    import numpy as np
    import torch
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=list(c.VARIANTS),required=True)
    p.add_argument('--sequence',choices=['DJI_0026_2','DJI_0067_2'],required=True);args=p.parse_args()
    output=c.N/'predictions'/args.variant/args.sequence
    if output.exists() and any(output.iterdir()):raise FileExistsError(output)
    (output/'windows').mkdir(parents=True)
    c.setup()
    base=c.load_module('ord_original_base',c.Q/'top1_background_control_20261011/code/train_pilot.py')
    model=c.create_model(base)
    original={k:v.detach().cpu().clone() for k,v in model.original.state_dict().items()}
    original_hash=c.tensor_hash(original.items())
    path=c.N/'training'/args.variant/'run/final_model.pth'
    run=json.loads((path.parent/'RUN.json').read_text());cp=torch.load(path,map_location='cpu',weights_only=True)
    assert run['status']=='complete' and run['final_step']==2048 and cp['step']==2048 and cp['condition']==args.variant
    assert cp['source_sha256']==run['source_sha256']==c.sources()
    assert run['final_checkpoint_sha256']==c.sha(path)
    assert cp['original_state_sha256']==run['original_initial_state_sha256']==run['original_final_state_sha256']==original_hash
    assert cp['model_config']==model.config
    model.load_state_dict(cp['model_state_dict'],strict=True);model=model.cuda().eval()
    c.assert_original(model,original)
    rawpath=c.Q/'stage1_goal/data'/args.sequence/'raw_frames.npy'
    annpath=rawpath.with_name('annotations.json')
    raw=np.load(rawpath,mmap_mode='r',allow_pickle=False)
    assert raw.dtype==np.uint8 and raw.shape==(160,512,640)
    fused=np.zeros(raw.shape,np.float32);coverage=np.zeros(160,np.int32)
    meta=dict(status='RUNNING',condition=args.variant,variant=args.variant,training_variant=args.variant,
        sequence=args.sequence,checkpoint=str(path),checkpoint_sha256=c.sha(path),checkpoint_step=2048,
        main_checkpoint_step=2048,checkpoint_is_main=True,reference_checkpoint_sha256=c.sha(c.CHECKPOINT),
        original_initial_state_sha256=original_hash,original_final_state_sha256=original_hash,
        source_sha256=c.sources(),script_sha256=c.sha(__file__),model_config=model.config,
        model_labels_access=False,global_queries=True,diagnostic='none',seed=46,
        raw_sha256=c.sha(rawpath),annotations_sha256=c.sha(annpath),
        normalization=[85.0799,47.4845],precision='FP32_no_AMP_no_TF32',tile_core=128,tile_halo=16,
        parameters=sum(p.numel() for p in model.parameters()),new_parameters=run['new_parameters'],
        official_starts=OFFICIAL,symmetric_starts=SYMMETRIC,diagnostic_starts=STARTS,
        plan_sha256=run['plan_sha256'],selection_sha256=run['selection_sha256'],
        output='max_query(sigmoid(presence)*sigmoid(mask)); no released-output union',
        window_timing=[],tiling_checks=[],frozen_original_unchanged=True)
    c.save_json(output/'RUNNING.json',meta);begin=time.perf_counter()
    try:
        with torch.inference_mode():
            # Predetermined geometry check, no checkpoint or hyperparameter selection.
            probe=torch.from_numpy(np.ascontiguousarray((raw[:40,176:272,224:352].astype(np.float32)-85.0799)/47.4845))[None,None].cuda()
            direct=model(probe,c.VARIANTS[args.variant],origin=(176,224))
            p_direct=model.response_map(direct['E'],direct['q'])
            tiled,z0=model.tiled_features(probe,core=48,halo=16,origin=(176,224))
            q=model.decode(tiled,c.VARIANTS[args.variant]);p_tiled=model.response_map(tiled,q)
            torch.testing.assert_close(p_tiled,p_direct,atol=1e-4,rtol=1e-4)
            disagreements=int(((p_tiled>=.5)!=(p_direct>=.5)).sum())
            assert disagreements==0
            meta['tiling_checks'].append(dict(status='passed',crop_yxhw=[176,224,96,128],
                max_probability_error=float((p_tiled-p_direct).abs().max()),threshold05_disagreements=disagreements,
                method='tile_features_then_same_global_queries',atol=1e-4,rtol=1e-4))
            del probe,direct,p_direct,tiled,z0,q,p_tiled
            torch.cuda.reset_peak_memory_stats()
            for start in STARTS:
                x=torch.from_numpy(np.ascontiguousarray((raw[start:start+40].astype(np.float32)-85.0799)/47.4845))[None,None].cuda()
                torch.cuda.synchronize();stamp=time.perf_counter()
                E,z0=model.tiled_features(x)
                q=model.decode(E,c.VARIANTS[args.variant])
                probabilities=model.response_map(E,q)[0].cpu().numpy()
                torch.cuda.synchronize();seconds=time.perf_counter()-stamp
                assert probabilities.shape==(40,512,640) and np.isfinite(probabilities).all()
                np.savez_compressed(output/'windows'/f'window_{start:03d}.npz',probabilities=probabilities,start=start,end=start+40)
                if start in OFFICIAL:
                    fused[start:start+40]=np.maximum(fused[start:start+40],probabilities);coverage[start:start+40]+=1
                meta['window_timing'].append(dict(start=start,model_plus_cpu_copy_seconds=seconds,global_query_decodes=1,
                    query_shape=list(q.shape),feature_tiles=20))
                print(json.dumps({'condition':args.variant,'sequence':args.sequence,'window':start,'seconds':seconds}),flush=True)
                del x,E,z0,q,probabilities
        assert np.all(coverage>0);assert meta['source_sha256']==c.sources()
        c.assert_original(model,original)
        np.savez_compressed(output/'official_fused.npz',probabilities=fused,coverage=coverage,starts=np.array(OFFICIAL))
        meta.update(status='COMPLETE',elapsed_seconds=time.perf_counter()-begin,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
            O_model_seconds=sum(r['model_plus_cpu_copy_seconds'] for r in meta['window_timing'] if r['start'] in OFFICIAL),
            S_model_seconds=sum(r['model_plus_cpu_copy_seconds'] for r in meta['window_timing'] if r['start'] in SYMMETRIC),
            fused_sha256=c.sha(output/'official_fused.npz'),
            window_sha256={p.name:c.sha(p) for p in sorted((output/'windows').glob('*.npz'))})
        c.save_json(output/'COMPLETE.json',meta)
    except BaseException as error:
        meta.update(status='FAILED',error=repr(error));c.save_json(output/'FAILED.json',meta);raise

if __name__=='__main__':main()
