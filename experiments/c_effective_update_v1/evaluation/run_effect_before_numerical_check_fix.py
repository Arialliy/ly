"""Fixed teacher cohorts and independent actual-Adam single-step interventions."""
import argparse
import copy
import json
import time
import numpy as np
import torch
import runtime as rt
from data import inputs,batch_data
import effect_common as common
import effect_data


def frozen_check():
    freeze=rt.read_json(rt.E/'PROTOCOL_FREEZE.json')
    for name,expected in freeze['files'].items():assert rt.sha(rt.E/name)==expected,name
    for name,expected in freeze['reused_sources'].items():assert rt.sha(name)==expected,name
    return rt.sha(rt.E/'PROTOCOL_FREEZE.json')


def pair_list():
    return rt.read_json(rt.E/'PAIR_PLAN.json')['pairs']


def indices():
    return sorted({p[k] for p in pair_list() for k in ('update_index','observer_index')})


def cohorts(shard,shards):
    protocol=frozen_check();rt.setup();state=inputs();model,_=rt.load_model()
    original=rt.tensor_hash(model.state_dict());began=time.perf_counter();records=[]
    for index in indices()[shard::shards]:
        target=rt.E/'cohorts'/f'entry_{index:03d}.json'
        if target.exists():raise FileExistsError(target)
        batch=batch_data(index,state);c=common.cache(index,batch)
        with torch.no_grad():
            z=common.forward(model,batch);risk=common.risk(z,c)
            result=effect_data.prepare_entry(index,batch,state,z,risk,c)
        result['execution']={'protocol_sha256':protocol,'release_checkpoint_sha256':rt.sha(rt.CP),
            'parameter_updates':0,**{k:batch[k] for k in ('input_sha256','label_sha256','auxiliary_input_sha256')}}
        rt.save_json(target,result);records.append({'index':index,'sha256':rt.sha(target)})
        print(json.dumps({'cohort_entry':index,'done':len(records)}),flush=True)
        del z,risk,batch,c
    assert rt.tensor_hash(model.state_dict())==original
    rt.save_json(rt.E/'cohorts'/f'SHARD_{shard}.json',{'status':'COMPLETE','protocol_sha256':protocol,
        'records':records,'seconds':time.perf_counter()-began,'model_unchanged':True,'parameter_updates':0})


def read_cohort(index):return rt.read_json(rt.E/'cohorts'/f'entry_{index:03d}.json')


def measure(z,r,cohort,reference=None):
    result=effect_data.measure(z,r,cohort)
    peaks=result.get('reliable_peak_xy',[])
    if reference is None:
        result['reliable_peak_shift']=0. if len(peaks) else None
    else:
        before=reference.get('reliable_peak_xy',[])
        assert len(before)==len(peaks)
        distances=np.linalg.norm(np.asarray(peaks,dtype=float)-np.asarray(before,dtype=float),axis=1) if len(peaks) else []
        result['reliable_peak_shift']=float(np.mean(distances)) if len(peaks) else None
        result['reliable_peak_changed_count']=int(np.count_nonzero(distances)) if len(peaks) else 0
    return result


def forward_measure(model,batch,c,cohort,reference=None):
    with torch.no_grad():
        z=common.forward(model,batch);r=common.risk(z,c)
        result=measure(z,r,cohort,reference)
    return result


def diagnose(shard,shards,config_mode=False):
    protocol=frozen_check();rt.setup();state=inputs()
    model,optimizer,checkpoint,criterion,settings=common.initialize()
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    names=[n for n,p in named];parameters=[p for n,p in named]
    initial=rt.tensor_hash(model.state_dict());initial_opt=common.optimizer_hash(optimizer.state_dict())
    frozen_hash=rt.tensor_hash(rt.frozen_state(model));bn_hash=rt.tensor_hash(rt.bn_state(model))
    random_state=common.get_rng();rng_digest=common.rng_hash()
    theta=torch.cat([p.detach().reshape(-1) for p in parameters]).clone()
    if config_mode:
        gate=rt.read_json(rt.E/'OPERATOR_CHECK.json')
        assert gate['status']=='CONFIG_SINGLE_STEP_AUTHORIZED'
        from config_operator import combine_gradients
    began=time.perf_counter();records=[]
    for pair in pair_list()[shard::shards]:
        pair_id=int(pair['pair_id']);path=rt.E/'pairs'/f'pair_{pair_id:02d}.json'
        if path.exists() and not config_mode:raise FileExistsError(path)
        if config_mode:
            result=rt.read_json(path);assert 'config' not in result['conditions']
        batches={role:batch_data(pair[role+'_index'],state) for role in ('update','observer')}
        caches={role:common.cache(pair[role+'_index'],batches[role]) for role in batches}
        cohort={role:read_cohort(pair[role+'_index']) for role in batches}
        model.load_state_dict(checkpoint['model_state_dict'],strict=True)
        optimizer.load_state_dict(copy.deepcopy(checkpoint['optimizer_state_dict']));optimizer.zero_grad(set_to_none=True)
        common.set_rng(random_state)
        assert rt.tensor_hash(model.state_dict())==initial and common.optimizer_hash(optimizer.state_dict())==initial_opt
        terms,z,r,c=common.compute(model,batches['update'],pair['update_index'],criterion,settings)
        group_keys=('fit','weighted_keep','weighted_track')
        grouped=[common.gradients(terms[k],parameters) for k in group_keys]
        group_vectors=[common.vector(g,parameters) for g in grouped]
        intervention_grads={key:common.gradients(terms[key],parameters) for key in ('full_F','no_keep','no_track')}
        full_vector=common.vector(intervention_grads['full_F'],parameters)
        summed=torch.stack(group_vectors).sum(0)
        torch.testing.assert_close(summed,full_vector,rtol=1e-4,atol=2e-7)
        baseline={'update':measure(z.detach(),r,cohort['update']),
                  'observer':forward_measure(model,batches['observer'],caches['observer'],cohort['observer'])}
        repeats={role:forward_measure(model,batches[role],caches[role],cohort[role],baseline[role]) for role in batches}
        assert common.rng_hash()==rng_digest
        losses={key:float(value.detach()) for key,value in terms.items()}
        del terms,z,r,c
        if not config_mode:
            result={**pair,'status':'RUNNING','protocol_sha256':protocol,'initial_checkpoint_sha256':rt.sha(rt.INIT_CP),
                'initial_model_sha256':initial,'initial_optimizer_sha256':initial_opt,'initial_rng_sha256':rng_digest,
                'parameter_names_in_vector_order':names,'baseline':baseline,'repeat':repeats,'conditions':{},
                'initial_losses':losses,'cohort_sha256':{role:rt.sha(rt.E/'cohorts'/f"entry_{pair[role+'_index']:03d}.json") for role in batches},
                'gradients':{'norms':{key:common.norm(g) for key,g in zip(group_keys,group_vectors)},
                    'cosines':{a+'__'+b:common.cosine(group_vectors[i],group_vectors[j])
                               for i,a in enumerate(group_keys) for j,b in enumerate(group_keys) if i<j},
                    'full_original_norm':common.norm(full_vector),'group_sum_norm':common.norm(summed),
                    'sum_vs_original_l2_difference':common.norm(summed-full_vector),
                    'unused_parameter_names':{key:[n for n,g in zip(names,gs) if g is None] for key,gs in zip(group_keys,grouped)}}}
        else:
            # The same measurement must reproduce the already frozen single-step baseline.
            for role in baseline:
                assert baseline[role]==result['baseline'][role],(pair_id,role,'baseline drift')
        raw={'initial_parameters':theta.cpu().numpy(), 'grouped_gradients':torch.stack(group_vectors).cpu().numpy(),
             'full_original_gradient':full_vector.cpu().numpy()}
        full_delta=None
        conditions=['config'] if config_mode else ['full_F','no_keep','no_track']
        if config_mode:
            with np.load(path.with_suffix('.npz'),allow_pickle=False) as data:full_delta=torch.as_tensor(data['delta_full_F'],device=theta.device)
        for condition in conditions:
            model.load_state_dict(checkpoint['model_state_dict'],strict=True)
            optimizer.load_state_dict(copy.deepcopy(checkpoint['optimizer_state_dict']));optimizer.zero_grad(set_to_none=True)
            common.set_rng(random_state)
            assert common.optimizer_hash(optimizer.state_dict())==initial_opt and common.rng_hash()==rng_digest
            operator=None
            if condition=='config':grads,operator=combine_gradients(named,dict(zip(('fit','keep','track'),grouped)),mode='config')
            else:grads=intervention_grads[condition]
            assert all(g is None or bool(torch.isfinite(g).all()) for g in grads)
            common.assign(parameters,grads);optimizer.step()
            delta=torch.cat([p.detach().reshape(-1) for p in parameters])-theta
            assert bool(torch.isfinite(delta).all())
            if condition=='full_F':full_delta=delta.clone()
            assert full_delta is not None
            if condition=='no_track' and common.norm(group_vectors[2])==0:
                assert torch.equal(delta,full_delta),'Removing a zero track gradient changed the Adam update'
            if condition=='no_keep' and common.norm(group_vectors[1])==0:
                assert torch.equal(delta,full_delta),'Removing a zero keep gradient changed the Adam update'
            current={role:forward_measure(model,batches[role],caches[role],cohort[role],result['baseline'][role]) for role in batches}
            assert rt.tensor_hash(rt.frozen_state(model))==frozen_hash and rt.tensor_hash(rt.bn_state(model))==bn_hash
            assert common.rng_hash()==rng_digest
            used=[p for p,g in zip(parameters,grads) if g is not None]
            assert all(float(optimizer.state[p]['step'])==2305 for p in used)
            for p,g in zip(parameters,grads):
                if g is None:assert float(optimizer.state[p]['step'])==2304
            input_vector=common.vector(grads,parameters)
            record={'roles':current,'update':{'l2':common.norm(delta),'input_gradient_l2':common.norm(input_vector),
                'cosine_full':common.cosine(delta,full_delta),'norm_ratio_full':common.norm(delta)/common.norm(full_delta) if common.norm(full_delta)>0 else None,
                'cosine_negative_input':common.cosine(-input_vector,delta),
                'gradient_dot_delta':{k:float(torch.dot(g.double(),delta.double())) for k,g in zip(group_keys,group_vectors)},
                'used_parameter_count':len(used),'all_unused_parameter_names':[n for n,g in zip(names,grads) if g is None]},
                'optimizer_step_before':2304,'optimizer_step_after_used':2305,
                'optimizer_after_sha256':common.optimizer_hash(optimizer.state_dict()),'model_after_sha256':rt.tensor_hash(model.state_dict()),
                'frozen_frontend_unchanged':True,'bn_unchanged':True,'rng_unchanged':True,'operator':operator}
            result['conditions'][condition]=record;raw['delta_'+condition]=delta.cpu().numpy().copy()
        if config_mode:
            np.savez_compressed(path.with_name(path.stem+'_config.npz'),**raw)
        else:np.savez_compressed(path.with_suffix('.npz'),**raw)
        result['status']='COMPLETE';result['parameter_updates_per_copy']=1;result['copies_accumulated']=False
        rt.save_json(path,result);records.append({'pair_id':pair_id,'sha256':rt.sha(path)})
        print(json.dumps({'pair_id':pair_id,'conditions':conditions,'done':len(records),
            'full_update_l2':result['conditions']['full_F']['update']['l2']}),flush=True)
        del batches,caches,cohort,grouped,group_vectors,intervention_grads,full_vector,summed,baseline,repeats,raw
    frozen_check()
    rt.save_json(rt.E/'pairs'/f"{'CONFIG_' if config_mode else ''}SHARD_{shard}.json",{'status':'COMPLETE',
        'protocol_sha256':protocol,'records':records,'seconds':time.perf_counter()-began,
        'initial_model_sha256':initial,'initial_optimizer_sha256':initial_opt,'rng_sha256':rng_digest})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['cohorts','diagnose','config'],required=True)
    parser.add_argument('--shard',type=int,required=True);parser.add_argument('--shards',type=int,default=3)
    a=parser.parse_args()
    if a.phase=='cohorts':cohorts(a.shard,a.shards)
    else:diagnose(a.shard,a.shards,a.phase=='config')
