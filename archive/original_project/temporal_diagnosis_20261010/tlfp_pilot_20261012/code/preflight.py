"""Finite TRAIN-only GPU checks. No optimizer update or detector selection."""
import argparse, gc, json, time
from types import SimpleNamespace
import numpy as np
import torch
import train_tlfp as r
from temporal_lfp import spatially_tiled_forward

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=__import__('pathlib').Path,required=True);args=p.parse_args()
    started=time.perf_counter(); report={'status':'RUNNING','seed':46,'formal_updates':0,'cases':[],'amp_training':False}
    r.deterministic_setup()
    source=r.Q.parent/'official_deeppro'
    checkpoint=source/'log/sem_seg/RGB-T__2025-01-14_15-27__SoftLoUloss_DeepPro-Plus_DataL40/checkpoints/best_model.pth'
    config=SimpleNamespace(source=source,checkpoint=checkpoint,dataset_root=r.Q/'stage1_goal/data',selection=r.Q/'stage1_goal/DATA_SELECTION.json',plan=r.Q/'stage1_goal/pilot/TRAIN_PLAN.json',steps=512,seed=46,lr=1e-4)
    try:
        base,plan,names,sequences,model_file=r.previous.validate_inputs(config)
        batch,positive,masks,specs=base.build_batch(plan['entries'][0],sequences)
        target_np,valid_np,_=r.build_targets(plan['entries'][0],sequences,background_masks=masks)
        old=json.loads(r.OLD_LOG.read_text().splitlines()[0])
        assert r.data_hash(batch,positive,masks,specs)==old['input_sha256']
        assert r.label_hash(target_np,valid_np)==old['label_sha256']
        x=torch.from_numpy(batch).cuda();target=torch.from_numpy(target_np).cuda();valid=torch.from_numpy(valid_np).cuda();background=torch.from_numpy(masks).cuda()
        for label in ['S','T']:
            original,_=base.load_model(source,checkpoint)
            # The original loader establishes its own networks import path.
            criterion=r.load_module('preflight_official_loss',model_file).SoftLoUloss().cuda()
            model=r.wrap_model(original,label).cuda().train()
            initial=r.tensor_hash(model.original.state_dict().items())
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                baseline=model.original(x)
                zero=model(x)
            eq=[bool(torch.equal(a,b)) for a,b in zip(baseline,zero)]
            assert all(eq),eq
            # Detection-loss gradient reaches gate at initialization.
            logits=model(x)[1]; loss,parts=r.common_loss(logits,target,valid,background,criterion);loss.backward()
            gate_grad=float(model.feature_refiner.scale.grad)
            assert np.isfinite(gate_grad) and gate_grad!=0
            assert all(p.grad is not None and torch.count_nonzero(p.grad)==0 for p in model.feature_refiner.operator.parameters())
            model.zero_grad(set_to_none=True)
            with torch.no_grad():
                model.feature_refiner.scale.fill_(.1)
                torch.manual_seed(46)
                model.feature_refiner.operator.attention.weight.normal_(std=.02)
            logits=model(x)[1]; loss,parts=r.common_loss(logits,target,valid,background,criterion);loss.backward()
            gradients={k:float(p.grad.detach().norm()) if p.grad is not None else None for k,p in model.feature_refiner.named_parameters()}
            assert all(v is not None and np.isfinite(v) and v>0 for v in gradients.values()),gradients
            wg=model.feature_refiner.operator.attention.weight.grad
            slot_gradients={'center':float(wg.norm())} if label=='S' else {name:float(wg[:,idx:idx+2].norm()) for name,idx in [('previous',0),('center',2),('next',4)]}
            assert all(v>0 and np.isfinite(v) for v in slot_gradients.values())
            assert all(p.grad is None for p in model.original.parameters())
            model.zero_grad(set_to_none=True)
            with torch.no_grad():
                # Local crop phase, identical for whole vs tiled; whole window T stays40.
                probe=x[:1,:,:,:97,:101].contiguous()
                direct=model(probe)[1]
                tiled=spatially_tiled_forward(model,probe,core_hw=(47,49),halo=20)[1]
                torch.testing.assert_close(direct,tiled,atol=2e-4,rtol=2e-5)
                disagreement=int(((direct.sigmoid()>=.5)!=(tiled.sigmoid()>=.5)).sum()); assert disagreement==0
                f=model.original.layer1(model.original.conv_in(probe))
                adapter=model.feature_refiner
                adapter.frame_chunk=40; full=adapter(f)
                chunks=[]
                for chunk in [1,8,17]:
                    adapter.frame_chunk=chunk; y=adapter(f)
                    torch.testing.assert_close(y,full,atol=2e-5,rtol=2e-5)
                    chunks.append({'chunk':chunk,'max_error':float((y-full).abs().max())})
                adapter.frame_chunk=8
                adapter.set_inference_mode('bypass')
                bypass=model(probe); release=model.original(probe)
                assert all(torch.equal(a,b) for a,b in zip(bypass,release))
                adapter.set_inference_mode('normal')
                # Diagnostic autocast only; training/inference remain FP32.
                amp=[]
                for dtype in [torch.float16,torch.bfloat16]:
                    with torch.autocast('cuda',dtype=dtype): mixed=adapter(f)
                    amp.append({'dtype':str(dtype),'finite':bool(torch.isfinite(mixed).all()),'max_abs_vs_FP32':float((mixed-full).abs().max()),'not_enabled_in_training':True})
                    assert torch.isfinite(mixed).all()
            assert initial==r.tensor_hash(model.original.state_dict().items())
            report['cases'].append({'condition':label,'zero_two_outputs_exact':eq,'initial_gate_grad':gate_grad,'nonzero_gate_parameter_gradients':gradients,'temporal_slot_gradient_norms':slot_gradients,'actual_batch_shape':list(x.shape),'spatial_tiling_max_logit_error':float((direct-tiled).abs().max()),'threshold_disagreements':disagreement,'time_chunks':chunks,'bypass_exact':True,'amp_diagnostics':amp,'original_state_unchanged':True,'peak_memory_bytes':torch.cuda.max_memory_allocated()})
            del model,original,logits,loss,baseline,zero,probe,direct,tiled,f,adapter,full,y,bypass,release,mixed
            gc.collect();torch.cuda.empty_cache()
        report['status']='PASS'
    except BaseException as e:
        report.update(status='FAILED',error=repr(e));raise
    finally:
        report['seconds']=time.perf_counter()-started;r.save_json(args.output,report);print(json.dumps(report),flush=True)
if __name__=='__main__':main()
