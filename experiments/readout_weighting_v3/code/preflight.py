"""Only new weighting checks on the existing fixed training batches; no updates."""
import json
import numpy as np
import torch
from runtime import E,V2,INIT_CP,MODEL_SOURCE,setup,load_module,read_json,save_json,sha,tensor_hash
from data import inputs,batch_data
from common import initial_model,compute,weighted_readout,scalarize,CALIBRATION_INDICES,old,optimizer_hash


def main():
    target=E/'resources/REAL_WEIGHTING_CHECK.json'
    if target.exists():raise FileExistsError(target)
    setup();state=inputs();model,optimizer,names,count,opt_hash=initial_model()
    initial=tensor_hash(model.state_dict());rng=torch.get_rng_state().clone()
    config=read_json(V2/'CALIBRATION.json')['fixed_settings']
    criterion=load_module('v3_weighting_check_original_loss',MODEL_SOURCE).SoftLoUloss().cuda()
    params=[p for p in model.parameters() if p.requires_grad]
    rows=[];nondegenerate=0
    for index in CALIBRATION_INDICES:
        batch=batch_data(index,state)
        terms,stats,f,z,r,c=compute(model,batch,index,criterion,config)
        q=(r['q_positive'],r['q_negative'])
        original=torch.autograd.grad(terms['frame'],q,retain_graph=True)
        results={mode:weighted_readout(r,c,mode) for mode in ('unit','confidence','matched_confidence')}
        unit=results['unit'];torch.testing.assert_close(unit['loss'],terms['frame'],rtol=2e-6,atol=1e-7)
        row={'index':index,'original_frame_loss':float(terms['frame'].detach()),
            'unit_loss_error':float((unit['loss']-terms['frame']).detach().abs()),'modes':{},
            **{key:batch[key] for key in ('input_sha256','label_sha256','auxiliary_input_sha256')}}
        unit_grads=None
        for mode,result in results.items():
            assert not result['negative_weight'].requires_grad
            assert bool(torch.isfinite(result['loss']))
            g=torch.autograd.grad(result['loss'],q,retain_graph=True)
            assert all(bool(torch.isfinite(v).all()) for v in g)
            torch.testing.assert_close(g[0],original[0],rtol=2e-5,atol=1e-9)
            assert float(g[0].min())<0 and float(g[1].min())>=0
            if mode=='unit':
                unit_grads=g
                torch.testing.assert_close(g[1],original[1],rtol=2e-5,atol=1e-12)
            assert unit_grads is not None
            ratio=float(g[1].double().sum()/unit_grads[1].double().sum())
            if mode=='matched_confidence':assert abs(ratio-1)<3e-5,(index,ratio)
            if mode=='confidence':assert ratio<=1+1e-7
            total_norm,_=old.grad_norm(result['loss'],params)
            neg_norm,_=old.grad_norm(result['loss_negative'],params)
            row['modes'][mode]={'positive_loss':float(result['loss_positive'].detach()),
                'negative_loss':float(result['loss_negative'].detach()),
                'positive_logit_grad_l1':float(g[0].double().abs().sum()),
                'negative_logit_grad_l1':float(g[1].double().sum()),'negative_budget_ratio_to_unit':ratio,
                'positive_gradient_max_error_to_v2':float((g[0]-original[0]).abs().max()),
                'negative_gradient_l1_difference_to_unit':float((g[1]-unit_grads[1]).double().abs().sum()),
                'total_parameter_gradient_l2':total_norm,'negative_parameter_gradient_l2':neg_norm,
                'weight_requires_grad':result['negative_weight'].requires_grad,
                'diagnostics':scalarize(result['diagnostics'])}
        neg=results['matched_confidence']['segment_negative_logits'].detach().double()
        valid=results['matched_confidence']['segment_valid']
        probabilities=neg.sigmoid();lengths=valid.sum(1)
        highest=probabilities.masked_fill(~valid,-torch.inf).amax(1)
        lowest=probabilities.masked_fill(~valid,torch.inf).amin(1)
        different=(lengths>1)&(highest>lowest)
        nondegenerate+=int(different.sum())
        row['group_degeneracy']={'segments':len(lengths),'singleton':int((lengths==1).sum()),
            'equal_background_score_groups':int(((lengths>0)&(highest==lowest)).sum()),
            'nondegenerate_groups':int(different.sum()),
            'median_probability_range':float((highest-lowest)[lengths>0].median()),
            'maximum_probability_range':float((highest-lowest)[lengths>0].max()),
            'max_absolute_N_weight_deviation_from_unit':float((results['matched_confidence']['negative_weight'][valid]-1).abs().max())}
        rows.append(row);print(json.dumps(row,allow_nan=False),flush=True)
        del terms,stats,f,z,r,c,q,results,unit,original,unit_grads,result,g,neg,valid,probabilities
    assert tensor_hash(model.state_dict())==initial and optimizer_hash(optimizer.state_dict())==opt_hash
    assert torch.equal(torch.get_rng_state(),rng),'Unexpected CPU random consumption in new weighting path'
    save_json(target,{'status':'PASS','uses_development':False,'parameter_updates':0,
        'indices':CALIBRATION_INDICES,'initial_checkpoint_sha256':sha(INIT_CP),
        'initial_model_sha256':initial,'initial_optimizer_sha256':opt_hash,
        'calibration_sha256':sha(V2/'CALIBRATION.json'),'lambda_readout_reused':config['lambda_readout'],
        'gamma':2,'group':'existing_segment8','unit_equivalence':'independent new scalar and positive/negative gradients versus original v2 FRAME',
        'budget_scope':'same snapshot, original per-segment/per-window reduction, repeated observations scatter-added',
        'nondegenerate_groups':nondegenerate,'run_N':nondegenerate>0,'rows':rows,
        'source_sha256':{str(p):sha(p) for p in [E/'code/preflight.py',E/'train_modules/readout_weighting.py',E/'code/common.py']}})

if __name__=='__main__':main()
