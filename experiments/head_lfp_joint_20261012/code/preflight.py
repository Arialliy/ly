"""Train-source-only real-shape integration; disposable updates, no formal training."""
import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
import torch
import train_joint as r
from temporal_lfp import spatially_tiled_forward


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    report={'status':'RUNNING','formal_updates':0,'disposable_updates_per_variant':1,'seed':46,
            'precision':'FP32_no_AMP_no_TF32','cases':[]}
    started=time.perf_counter();r.deterministic_setup()
    source=r.Q.parent/'official_deeppro'
    checkpoint=source/'log/sem_seg/RGB-T__2025-01-14_15-27__SoftLoUloss_DeepPro-Plus_DataL40/checkpoints/best_model.pth'
    cfg=SimpleNamespace(source=source,checkpoint=checkpoint,dataset_root=r.Q/'stage1_goal/data',
                        selection=r.Q/'stage1_goal/DATA_SELECTION.json',plan=r.Q/'stage1_goal/pilot/TRAIN_PLAN.json')
    device=torch.device(args.device)
    try:
        base,plan,names,sequences,model_file=r.previous.validate_inputs(cfg)
        batch,positives,masks,specs=base.build_batch(plan['entries'][0],sequences)
        target_np,valid_np,_=r.build_targets(plan['entries'][0],sequences,background_masks=masks)
        old=json.loads(r.OLD_LOG.read_text().splitlines()[0])
        assert r.data_hash(batch,positives,masks,specs)==old['input_sha256']
        assert r.label_hash(target_np,valid_np)==old['label_sha256']
        x=torch.from_numpy(batch).to(device);target=torch.from_numpy(target_np).to(device)
        valid=torch.from_numpy(valid_np).to(device);background=torch.from_numpy(masks).to(device)
        # The official model imports ``networks`` from its source tree. Its
        # verified loader establishes that path before loading the criterion.
        original,_=base.load_model(source,checkpoint)
        criterion=r.load_module('joint_preflight_loss',model_file).SoftLoUloss().to(device)
        for variant_index,variant in enumerate(r.VARIANTS):
            if variant_index:
                original,_=base.load_model(source,checkpoint)
            model=r.wrap_model(original,variant).to(device).train()
            fixed={k:v.detach().cpu().clone() for k,v in r.frozen_original_state(model).items()}
            initial={k:p.detach().cpu().clone() for k,p in model.named_parameters()}
            if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
            with torch.no_grad():
                baseline=model.original(x);zero=model(x)
            assert all(torch.equal(a,b) for a,b in zip(baseline,zero))
            optimizer=torch.optim.Adam(model.parameter_groups(),weight_decay=1e-4)
            loss,parts=r.common_loss(model(x)[1],target,valid,background,criterion)
            loss.backward()
            grad0={k:float(p.grad.norm()) if p.grad is not None else None for k,p in model.named_parameters() if p.requires_grad}
            assert all(v is not None and np.isfinite(v) for v in grad0.values())
            if variant=='HL':
                assert grad0['feature_refiner.scale']>0
                assert all(torch.count_nonzero(p.grad)==0 for p in model.feature_refiner.operator.parameters())
            optimizer.step();r.assert_frozen(model,fixed)
            changed={k:not torch.equal(p.detach().cpu(),initial[k]) for k,p in model.named_parameters()}
            assert all(changed['original.'+k] for k in r.HEAD_NAMES)
            assert all(not changed['original.'+k] for k in fixed if k in dict(original.named_parameters()))
            model.zero_grad(set_to_none=True)
            nonzero_grads={}
            if variant=='HL':
                with torch.no_grad():
                    model.feature_refiner.scale.fill_(.1)
                    model.feature_refiner.operator.attention.weight.normal_(std=.02)
                loss,parts=r.common_loss(model(x)[1],target,valid,background,criterion)
                loss.backward()
                nonzero_grads={k:float(p.grad.norm()) for k,p in model.feature_refiner.named_parameters()}
                assert all(np.isfinite(v) and v>0 for v in nonzero_grads.values())
                assert all(p.grad is None for p in original.TPro.parameters())
                model.zero_grad(set_to_none=True)
            with torch.no_grad():
                probe=x[:1,:,:,:97,:101].contiguous()
                direct=model(probe)[1]
                tiled=spatially_tiled_forward(model,probe,core_hw=(47,49),halo=20)[1]
                torch.testing.assert_close(direct,tiled,atol=2e-4,rtol=2e-5)
                disagreement=int(((direct.sigmoid()>=.5)!=(tiled.sigmoid()>=.5)).sum())
                assert disagreement==0
                model.set_inference_mode('bypass')
                bypass=model(probe); adapted=model.original(probe)
                assert all(torch.equal(a,b) for a,b in zip(bypass,adapted))
            model.set_inference_mode('normal')
            amp={'status':'SKIPPED_NON_CUDA','formal_precision_unchanged':True,
                 'optimizer_updates':0,'purpose':'autocast integration finiteness only; no FP32 agreement claim'}
            if device.type=='cuda':
                # Select the first supervised training pixel by annotation
                # order, not by model output. Keep the complete 40-frame axis.
                nonzero=torch.nonzero(target,as_tuple=False)
                if len(nonzero):
                    wi,_,cy,cx=[int(v) for v in nonzero[0]]
                else:
                    wi,cy,cx=0,64,64
                y0=max(0,min(cy-8,x.shape[-2]-17))
                x0=max(0,min(cx-9,x.shape[-1]-19))
                amp_x=x[wi:wi+1,:,:,y0:y0+17,x0:x0+19].contiguous()
                amp_target=target[wi:wi+1,:,y0:y0+17,x0:x0+19]
                amp_valid=valid[wi:wi+1,:,y0:y0+17,x0:x0+19]
                amp_background=background[wi:wi+1,:,y0:y0+17,x0:x0+19]
                state_before_amp=r.tensor_hash(model.state_dict().items())
                model.zero_grad(set_to_none=True)
                with torch.random.fork_rng(devices=[device.index if device.index is not None else torch.cuda.current_device()]):
                    torch.manual_seed(46)
                    with torch.autocast(device_type='cuda',dtype=torch.float16):
                        amp_logits=model(amp_x)[1]
                        amp_loss,_=r.common_loss(amp_logits,amp_target,amp_valid,amp_background,criterion)
                    assert bool(torch.isfinite(amp_logits).all()) and bool(torch.isfinite(amp_loss))
                    amp_loss.backward()
                amp_gradients={k:{'present':p.grad is not None,
                                  'finite':p.grad is not None and bool(torch.isfinite(p.grad).all()),
                                  'norm':None if p.grad is None else float(p.grad.norm())}
                               for k,p in model.named_parameters() if p.requires_grad}
                assert all(v['present'] and v['finite'] for v in amp_gradients.values())
                model.zero_grad(set_to_none=True)
                assert state_before_amp==r.tensor_hash(model.state_dict().items())
                amp.update(status='PASS',dtype='torch.float16',seed=46,actual_shape=list(amp_x.shape),
                           crop_window_index=wi,crop_yx=[y0,x0],loss=float(amp_loss.detach()),
                           gradient_finiteness=amp_gradients,model_state_unchanged=True,
                           scaler_used=False)
                del amp_x,amp_target,amp_valid,amp_background,amp_logits,amp_loss,nonzero
            r.assert_frozen(model,fixed)
            report['cases'].append({'variant':variant,'actual_batch_shape':list(x.shape),'initial_outputs_exact':True,
                                   'first_gradient_norms':grad0,'nongate_LFP_gradient_norms_after_gate_nonzero':nonzero_grads,
                                   'all_four_head_tensors_changed':True,'original_nonhead_and_BN_unchanged':True,
                                   'trainable_count':sum(p.numel() for p in model.parameters() if p.requires_grad),
                                   'spatial_tiling_max_logit_error':float((direct-tiled).abs().max()),
                                   'threshold_disagreements':disagreement,'bypass_equals_adapted_original':True,
                                   'AMP_diagnostic':amp,
                                   'max_cuda_memory_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0})
            del model,original,optimizer,loss,baseline,zero,direct,tiled,bypass,adapted,probe
            gc.collect()
            if device.type=='cuda':torch.cuda.empty_cache()
        report['status']='PASS'
    except BaseException as error:
        report.update(status='FAILED',error=repr(error));raise
    finally:
        report['seconds']=time.perf_counter()-started
        r.save_json(args.output,report)
        print(json.dumps(report),flush=True)


if __name__=='__main__':main()
