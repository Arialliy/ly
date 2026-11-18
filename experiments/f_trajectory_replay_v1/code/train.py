"""Two continuous F2304 continuations; only frozen input schedules differ."""
import argparse
import json
import time
import torch
import runtime as rt
import data
import effect_common as common
import replay_data as replay


def readout(risk,cache):
    idx=torch.as_tensor(cache['segment_indices'],device=risk['q_positive'].device).flatten()
    idx=idx[idx>=0]
    if not len(idx):return {'observations_with_multiplicity':0}
    a=risk['q_positive'][idx].detach();b=risk['q_negative'][idx].detach()
    return dict(observations_with_multiplicity=len(idx),target_logit_mean=float(a.mean()),
        background_logit_mean=float(b.mean()),logit_margin_mean=float((a-b).mean()),
        probability_margin_mean=float((a.sigmoid()-b.sigmoid()).mean()),
        target_logit_above_zero_fraction=float((a>0).float().mean()),
        background_logit_above_zero_fraction=float((b>0).float().mean()))


def compute(model,batch,contrast,frame,criterion,config):
    common.old.verify_cache(batch,contrast);common.old.verify_cache(batch,frame)
    terms,stats,features,logits=common.old.loss_terms(model,batch,contrast,criterion,config['temperature'],variants=('C',))
    risk=common.risk(logits,frame);terms['frame']=risk['loss_frame']
    # Preserve the scalar operation order of the existing F trainer.
    loss=(terms['det']+config['lambda_keep']*terms['keep']+config['lambda_temporal']*terms['C'])+config['lambda_readout']*terms['frame']
    return loss,terms,stats,risk


def verify_freeze():
    f=rt.read_json(rt.E/'PROTOCOL_FREEZE.json')
    for name,value in f['files'].items():assert rt.sha(rt.E/name)==value,name
    for name,value in f['reused_sources'].items():assert rt.sha(name)==value,name
    assert rt.sha(rt.INIT_CP)==f['initial_checkpoint_sha256']
    for name,value in f['cache_sha256'].items():assert rt.sha(name)==value,name
    return f


def train(condition):
    rt.setup();freeze=verify_freeze();state=replay.inputs()
    model,optimizer,checkpoint,criterion,config=common.initialize()
    names=rt.trainable_names(model);count=sum(p.numel() for p in model.parameters() if p.requires_grad)
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    frozen_hash=rt.tensor_hash(rt.frozen_state(model));bn_hash=rt.tensor_hash(rt.bn_state(model))
    optimizer_initial=common.optimizer_hash(optimizer.state_dict())
    assert optimizer_initial==freeze['initial_optimizer_sha256']
    assert all(g['lr']==1e-5 and g['weight_decay']==1e-4 for g in optimizer.param_groups)
    out=rt.E/'training'/condition/'run'
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    out.mkdir(parents=True)
    identity=dict(condition=condition,seed=46,start_step=2304,additional_steps=512,
        initial_checkpoint_sha256=rt.sha(rt.INIT_CP),release_checkpoint_sha256=rt.sha(rt.CP),
        model_source_sha256=rt.sha(rt.MODEL_SOURCE),protocol_sha256=rt.sha(rt.E/'PROTOCOL_FREEZE.json'),
        schedule_sha256=rt.sha(rt.E/(condition+'_SCHEDULE.csv')),
        trainable_parameter_names=names,settings=config)
    metadata=dict(identity,status='running',final_step=None,initial_optimizer_restored=True,
        initial_optimizer_sha256=optimizer_initial,initial_state_sha256=rt.tensor_hash(initial),
        initial_rng_sha256=common.rng_hash(),trainable_parameters=count,trainable_parameter_count=count,
        total_parameters=70913,new_parameters=0,snapshots=[],frozen_state_sha256=frozen_hash,bn_initial_sha256=bn_hash,
        optimizer=dict(name='Adam',lr=1e-5,weight_decay=1e-4,betas=[.9,.999],eps=1e-8,scheduler=None),
        bn_policy='eval for every module; original TPro/head affine parameters trainable; all buffers frozen',
        precision='FP32 deterministic no AMP no TF32',torch_version=str(torch.__version__))
    plan=rt.read_json(rt.E/'TRAIN_DIAGNOSTIC_PLAN.json')
    assert plan['checkpoints_additional_steps']==[0,128,256,512]
    assert all(0<=pi<256 for pi in plan['original_entry_indices'])

    def check_frozen():
        assert rt.tensor_hash(rt.frozen_state(model))==frozen_hash
        assert rt.tensor_hash(rt.bn_state(model))==bn_hash
        assert all(not module.training for module in model.modules())

    def snapshot(additional_step):
        check_frozen();step=2304+additional_step
        cp=dict(identity,step=step,additional_step=additional_step,
            model_state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
            optimizer_state_dict=optimizer.state_dict())
        path=out/f'snapshot_{step}.pth';torch.save(cp,path)
        metadata['snapshots'].append(dict(step=step,additional_step=additional_step,path=path.name,sha256=rt.sha(path)))
        rt.save_json(out/'RUN.json',metadata)

    def diagnostic(additional_step):
        before=common.rng_hash();rows=[]
        with torch.no_grad():
            for pi in plan['original_entry_indices']:
                batch=data.batch_data(pi,state);c=data.cache_entry(pi);f=common.cache(pi,batch)
                loss,terms,stats,risk=compute(model,batch,c,f,criterion,config)
                rows.append(dict(original_entry_index=pi,loss=float(loss),
                    loss_terms={k:float(v) for k,v in terms.items()},contrast=stats['C'],readout=readout(risk,f)))
        assert common.rng_hash()==before;check_frozen()
        rt.save_json(out/f'TRAIN_DIAGNOSTICS_{additional_step:03d}.json',dict(
            condition=condition,additional_step=additional_step,uses_development=False,
            plan_sha256=rt.sha(rt.E/'TRAIN_DIAGNOSTIC_PLAN.json'),rows=rows))

    started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    try:
        snapshot(0);diagnostic(0)
        with (out/'TRAIN.jsonl').open('w',buffering=1) as stream:
            for index in range(512):
                batch=replay.batch_data(index,condition,state);c,f=replay.load_caches(index,condition,batch)
                optimizer.zero_grad(set_to_none=True)
                loss,terms,stats,risk=compute(model,batch,c,f,criterion,config)
                assert bool(torch.isfinite(loss));loss.backward()
                grads={k:p.grad for k,p in model.named_parameters() if p.requires_grad}
                assert all(g is not None and bool(torch.isfinite(g).all()) for g in grads.values())
                row=dict(step=2305+index,additional_step=index+1,original_plan_index=batch['original_plan_index'],
                    condition=condition,loss=float(loss.detach()),loss_terms={k:float(v.detach()) for k,v in terms.items()},
                    contrast=stats['C'],readout=readout(risk,f),
                    grad_norm=float(torch.linalg.vector_norm(torch.stack([g.norm() for g in grads.values()]))),
                    **{k:batch[k] for k in ('input_sha256','label_sha256','auxiliary_input_sha256')})
                optimizer.step();row['elapsed_seconds']=time.perf_counter()-started
                stream.write(json.dumps(row,allow_nan=False)+'\n')
                if index<2 or (index+1)%64==0:
                    check_frozen();print(json.dumps({k:row[k] for k in ('condition','additional_step','loss','elapsed_seconds')}),flush=True)
                del loss,terms,stats,risk,grads
                if index+1 in (128,256,512):snapshot(index+1);diagnostic(index+1)
        assert all(float(s['step'])==2816 for s in optimizer.state.values())
        check_frozen();verify_freeze()
        metadata.update(status='complete',final_step=2816,frozen_frontend_unchanged=True,bn_unchanged=True,
            elapsed_seconds=time.perf_counter()-started,max_cuda_memory_allocated_bytes=torch.cuda.max_memory_allocated(),
            final_state_sha256=rt.tensor_hash(model.state_dict()),final_optimizer_sha256=common.optimizer_hash(optimizer.state_dict()),
            parameter_changes={k:float((v.detach().cpu()-initial[k]).double().norm()) for k,v in model.named_parameters() if v.requires_grad})
        rt.save_json(out/'RUN.json',metadata)
    except BaseException as error:
        metadata.update(status='failed',error=repr(error));rt.save_json(out/'RUN.json',metadata);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--condition',required=True,choices=['ORIGINAL','REPLAY'])
    train(parser.parse_args().condition)
