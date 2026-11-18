"""Finite training-only semantic/integration/learnability checks, no dev evaluation."""
import json,time
import common as c

def main():
    import numpy as np
    import torch
    c.setup()
    labels=c.load_module('ord_targets',c.N/'code/targets.py')
    losses=c.load_module('ord_losses',c.N/'code/losses.py')
    base,plan,_,sequences,_=c.previous.validate_inputs(c.default_args())
    chosen={}
    for ei,entry in enumerate(plan['entries']):
        batch,_,bg,_=base.build_batch(entry,sequences)
        targets=labels.build_entry_targets(entry,sequences,background_masks=bg)
        specs=[dict(sequence=p['sequence'],start=s,crop_yx=p['crop_yx']) for p in entry['pairs'] for s in p['starts']]
        for wi,target in enumerate(targets):
            kind='positive' if target['observations'] else 'empty'
            if kind not in chosen and (kind=='positive' or bool(np.all(target['negative_eligible']))):
                chosen[kind]=(batch[wi:wi+1].copy(),target,specs[wi],ei,wi)
        if len(chosen)==2:break
    assert len(chosen)==2
    model=c.create_model(base).cuda().train()
    original={k:v.detach().cpu().clone() for k,v in model.original.state_dict().items()}
    frozen=c.assert_original(model,original)
    optimizer=torch.optim.Adam([p for _,p in c.branches(model)],lr=1e-4,weight_decay=1e-4)
    rows=[];start=time.perf_counter()
    for step in range(65):
        optimizer.zero_grad(set_to_none=True);parts=[];total=0.
        for kind,(batch,target,spec,_,_) in chosen.items():
            out=model(torch.from_numpy(batch).cuda(),temporal_enabled=True,origin=tuple(spec['crop_yx']))
            masks=model.mask_logits(out['E'],out['q'])[0]
            loss,metrics,assignment=losses.object_response_loss(out['a'][0],masks,target,assignment='track')
            assert bool(torch.isfinite(loss))
            (loss/2).backward();total+=float(loss.detach())/2;parts.append(dict(kind=kind,**metrics))
            del out,masks,loss
        grads={name:float(p.grad.norm()) for name,p in c.branches(model) if p.grad is not None}
        assert grads and all(np.isfinite(v) for v in grads.values())
        rows.append(dict(step=step,loss=total,parts=parts,gradient_norms=grads if step in (0,64) else {}))
        if step<64:optimizer.step()
        if step%8==0:print(json.dumps({'preflight_step':step,'loss':total,'seconds':time.perf_counter()-start}),flush=True)
    assert rows[-1]['loss']<rows[0]['loss'],'Finite training-only fixed-pair objective did not decrease'
    c.assert_original(model,original)
    model.eval();geometry=[]
    with torch.inference_mode():
        batch,_,spec,_,_=chosen['positive'];probe=torch.from_numpy(batch[:,:,:,:96,:128]).cuda()
        direct=model(probe,temporal_enabled=True,origin=tuple(spec['crop_yx']))
        dense=model.mask_logits(direct['E'],direct['q'])
        chunk=model.mask_logits(direct['E'],direct['q'],spatial_chunk=1024)
        torch.testing.assert_close(chunk,dense,atol=1e-4,rtol=1e-4)
        expected=(direct['a'].sigmoid()[...,None,None]*dense.sigmoid()).amax(2)
        streamed=model.response_map(direct['E'],direct['q'])
        torch.testing.assert_close(streamed,expected,atol=1e-5,rtol=1e-5)
        geometry.append(dict(kind='mask_chunks',max_prob_error=float((streamed-expected).abs().max()),
            threshold_disagreements=int(((streamed>=.5)!=(expected>=.5)).sum())))
        tiled=model.tiled_features(probe,core=48,halo=16,origin=tuple(spec['crop_yx']))
        if isinstance(tiled,tuple):tiled=tiled[0]
        if isinstance(tiled,dict):tiled=tiled['E']
        torch.testing.assert_close(tiled,direct['E'],atol=1e-4,rtol=1e-4)
        q=model.decode(tiled,True);p=model.response_map(tiled,q)
        torch.testing.assert_close(p,streamed,atol=1e-4,rtol=1e-4)
        assert int(((p>=.5)!=(streamed>=.5)).sum())==0
        geometry.append(dict(kind='feature_tiles_then_global_query',max_feature_error=float((tiled-direct['E']).abs().max()),
            max_prob_error=float((p-streamed).abs().max()),threshold_disagreements=0))
        # Original path remains available and bitwise equal under same frozen state.
        _,z=model.original_forward(probe)
        torch.testing.assert_close(z,direct['z0'],rtol=0,atol=0)
        del probe,direct,dense,chunk,expected,streamed,tiled,q,p,z
        raw=sequences[spec['sequence']].frames[spec['start']:spec['start']+40]
        x=torch.from_numpy(np.ascontiguousarray((raw.astype(np.float32)-85.0799)/47.4845))[None,None].cuda()
        torch.cuda.reset_peak_memory_stats()
        E=model.tiled_features(x)
        if isinstance(E,tuple):E=E[0]
        if isinstance(E,dict):E=E['E']
        q=model.decode(E,True);p=model.response_map(E,q)
        assert p.shape==(1,40,512,640) and bool(torch.isfinite(p).all())
        geometry.append(dict(kind='full_train_image_shared_queries',shape=list(p.shape),query_shape=list(q.shape),
            probability_min=float(p.min()),probability_max=float(p.max()),peak_cuda_bytes=torch.cuda.max_memory_allocated()))
    c.assert_original(model,original)
    record=dict(status='PASS',training_only=True,development_used=False,updates=64,
        selected={k:dict(spec=v[2],source_entry=v[3],window=v[4],target_digest=v[1]['digest']) for k,v in chosen.items()},
        initial_loss=rows[0]['loss'],final_loss=rows[-1]['loss'],loss_decreased=True,geometry=geometry,
        original_state_sha256=frozen,frozen_original_unchanged=True,rows=rows,seconds=time.perf_counter()-start,
        source_sha256=c.sources(),model_config=model.config)
    c.save_json(c.N/'checks/GPU_PREFLIGHT.json',record)
    print(json.dumps({'status':'PASS','initial_loss':rows[0]['loss'],'final_loss':rows[-1]['loss']}),flush=True)

if __name__=='__main__':main()
