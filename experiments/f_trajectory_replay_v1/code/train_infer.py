#!/usr/bin/env python3
"""Full-frame training-range original inference for the four fixed weight states."""
import argparse
from pathlib import Path
import time
import numpy as np
import torch
import runtime as rt
import train_evaluate as te


def verify_checkpoint(condition,path):
    if condition=='release':
        assert path==rt.CP
        return {'checkpoint_step':None,'checkpoint_is_main':False,'initial_optimizer_restored':False}
    if condition=='F_init':
        assert path==rt.INIT_CP
        cp=torch.load(path,map_location='cpu',weights_only=True)
        assert cp['step']==2304 and cp['condition']=='FRAME'
        return {'checkpoint_step':2304,'checkpoint_is_main':False,'initial_optimizer_restored':False}
    assert condition in ['ORIGINAL','REPLAY']
    run_path=path.with_name('RUN.json');run=rt.read_json(run_path)
    cp=torch.load(path,map_location='cpu',weights_only=True)
    assert run['status'].lower()=='complete' and run['condition']==cp['condition']==condition
    assert run['seed']==cp['seed']==46
    assert cp['step']==run['final_step']==2816 and cp['additional_step']==512
    assert cp['start_step']==run['start_step']==2304 and cp['additional_steps']==run['additional_steps']==512
    assert cp['initial_checkpoint_sha256']==run['initial_checkpoint_sha256']==rt.sha(rt.INIT_CP)
    assert cp['release_checkpoint_sha256']==run['release_checkpoint_sha256']==rt.sha(rt.CP)
    assert cp['protocol_sha256']==run['protocol_sha256']==rt.sha(rt.E/'PROTOCOL_FREEZE.json')
    assert cp['model_source_sha256']==run['model_source_sha256']==rt.sha(rt.MODEL_SOURCE)
    assert cp['schedule_sha256']==run['schedule_sha256']==rt.sha(rt.E/(condition+'_SCHEDULE.csv'))
    assert run['initial_optimizer_restored'] and run['frozen_frontend_unchanged'] and run['bn_unchanged']
    snapshots={int(row['step']):row for row in run['snapshots']}
    assert snapshots[2816]['sha256']==rt.sha(path)
    return {'checkpoint_step':2816,'checkpoint_is_main':True,'initial_optimizer_restored':True,
        'training_run_sha256':rt.sha(run_path),'schedule_sha256':cp['schedule_sha256']}


def run(condition,sequence,checkpoint=None):
    plan=te.verify_plan();assert sequence in plan['sequences'] and condition in te.CONDITIONS
    checkpoint=Path(checkpoint) if checkpoint else rt.CP if condition=='release' else rt.INIT_CP if condition=='F_init' else rt.E/'training'/condition/'run/snapshot_2816.pth'
    extra=verify_checkpoint(condition,checkpoint)
    freeze=rt.read_json(rt.E/'PROTOCOL_FREEZE.json')
    for name,expected in freeze['files'].items():assert rt.sha(rt.E/name)==expected,('frozen artifact changed',name)
    rt.setup();model,model_source=rt.load_model(checkpoint)
    initial=rt.tensor_hash(model.state_dict());bn=rt.tensor_hash(rt.bn_state(model));frontend=rt.tensor_hash(rt.frozen_state(model))
    rawpath=rt.DATA/sequence/'raw_frames.npy';raw=np.load(rawpath,mmap_mode='r',allow_pickle=False)
    assert raw.shape==(160,512,640) and raw.dtype==np.uint8
    assert rt.sha(rawpath)==plan['source_sha256'][str(rawpath)]
    dest=rt.E/'predictions/train'/condition/sequence
    if dest.exists() and any(dest.iterdir()):raise FileExistsError(dest)
    (dest/'windows').mkdir(parents=True,exist_ok=True)
    metadata=dict(status='RUNNING',condition=condition,sequence=sequence,checkpoint=str(checkpoint),checkpoint_sha256=rt.sha(checkpoint),
        release_checkpoint_sha256=rt.sha(rt.CP),initial_checkpoint_sha256=rt.sha(rt.INIT_CP),model_source_sha256=rt.sha(model_source),
        protocol_sha256=rt.sha(rt.E/'PROTOCOL_FREEZE.json'),train_eval_plan_sha256=rt.sha(te.PLAN_PATH),
        raw_sha256=rt.sha(rawpath),annotations_sha256=plan['source_sha256'][str(rawpath.with_name('annotations.json'))],
        official_starts=rt.O,symmetric_starts=rt.S,diagnostic_starts=rt.STARTS,normalization=[85.0799,47.4845],
        seed=46,precision='FP32_no_AMP_no_TF32',tile_core=128,tile_halo=16,spatial_phase='absolute_even_tile_input_origins',
        model_labels_access=False,auxiliary_modules_at_inference=False,original_forward_unchanged=True,
        initial_model_state_sha256=initial,initial_bn_state_sha256=bn,initial_frozen_frontend_state_sha256=frontend,
        inference_source_sha256=rt.sha(__file__),runtime_source_sha256=rt.sha(rt.E/'code/runtime.py'),torch=torch.__version__,
        window_timing=[],tiling_validation='reuse_verified_original_tiled_logits_core128_halo16',
        scope='Training-source full-frame learnability, not independent test performance.',**extra)
    fused=np.zeros(raw.shape,np.float32);coverage=np.zeros(160,np.int16);began=time.perf_counter()
    try:
        with torch.inference_mode():
            for start in rt.STARTS:
                x=torch.from_numpy((raw[start:start+40].astype(np.float32)-85.0799)/47.4845)[None,None].cuda()
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.perf_counter()
                probabilities=rt.tiled_logits(model,x).sigmoid()[0].numpy();torch.cuda.synchronize()
                seconds=time.perf_counter()-tick;assert probabilities.dtype==np.float32 and np.isfinite(probabilities).all()
                wp=dest/'windows'/f'window_{start:03d}.npz';tick=time.perf_counter()
                np.savez_compressed(wp,probabilities=probabilities,start=start,end=start+40)
                if start in rt.O:fused[start:start+40]=np.maximum(fused[start:start+40],probabilities);coverage[start:start+40]+=1
                row=dict(start=start,model_plus_cpu_copy_seconds=seconds,write_seconds=time.perf_counter()-tick,
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
                metadata['window_timing'].append(row);print({'condition':condition,'sequence':sequence,**row},flush=True)
                del x,probabilities
        assert initial==rt.tensor_hash(model.state_dict()) and bn==rt.tensor_hash(rt.bn_state(model)) and frontend==rt.tensor_hash(rt.frozen_state(model))
        assert np.all(coverage>0) and int(coverage.sum())==200
        np.savez_compressed(dest/'official_fused.npz',probabilities=fused,coverage=coverage,starts=np.array(rt.O))
        metadata.update(status='COMPLETE',elapsed_seconds=time.perf_counter()-began,all_model_state_unchanged_during_inference=True,
            frozen_frontend_unchanged=True,bn_unchanged=True,final_model_state_sha256=initial,
            window_sha256={p.name:rt.sha(p) for p in sorted((dest/'windows').glob('*.npz'))},fused_sha256=rt.sha(dest/'official_fused.npz'))
        rt.save_json(dest/'COMPLETE.json',metadata)
    except BaseException as error:
        rt.save_json(dest/'FAILED.json',dict(status='FAILED',error=repr(error),metadata=metadata));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--condition',required=True,choices=te.CONDITIONS)
    p.add_argument('--sequence');p.add_argument('--checkpoint',type=Path);a=p.parse_args()
    for seq in [a.sequence] if a.sequence else te.read(te.PLAN_PATH)['sequences']:run(a.condition,seq,a.checkpoint)
