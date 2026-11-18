"""One deterministic training-only numerical calibration; discard pilot weights."""
import json
import statistics
import time
import numpy as np
import torch
from runtime import E,CP,MODEL_SOURCE,setup,load_model,freeze_for_training,load_module,save_json,sha
from data import inputs,batch_data,cache_entry,cache_tensors
from train import loss_terms,grad_norm,verify_cache
from region_contrast_loss import pool_regions


def choose_entries(plan):
    # First eight eligible entries, preferring unseen training sources.
    result=[];seen=set()
    for entry in plan['entries']:
        index=entry['step'];c=cache_entry(index)
        if not len(c['negative_indices']) or not bool((c['negative_indices']>=0).all(1).any()):continue
        names={p['sequence'] for p in entry['pairs']}
        if names-seen:result.append(index);seen.update(names)
        if len(seen)==len(plan['train_sequences']):break
    for index in range(512):
        if len(result)>=8:break
        c=cache_entry(index)
        if index not in result and len(c['negative_indices']) and bool((c['negative_indices']>=0).all(1).any()):
            result.append(index)
    assert len(result)==8
    return result,sorted(seen)


def main():
    path=E/'CALIBRATION.json'
    if path.exists():raise FileExistsError(path)
    setup();state=inputs();model,_=load_model();names,count=freeze_for_training(model)
    criterion=load_module('calibration_official_loss',MODEL_SOURCE).SoftLoUloss().cuda()
    params=[p for p in model.parameters() if p.requires_grad]
    indices,coverage=choose_entries(state[1]);rows=[]
    optimizer=torch.optim.Adam(params,lr=1e-5,weight_decay=1e-4)
    started=time.perf_counter()
    # Keep MSE and its gradient are exactly zero at teacher initialization.
    # Eight training-only det updates make the keep scale measurable.
    for index in indices:
        batch=batch_data(index,state);cache=cache_entry(index);verify_cache(batch,cache)
        optimizer.zero_grad(set_to_none=True)
        terms,stats,features,logits=loss_terms(model,batch,cache,criterion,.1,variants=())
        terms['det'].backward();optimizer.step()
        rows.append({'phase':'discarded_det_warmup','index':index,'loss':float(terms['det'].detach())})
        del terms,stats,features,logits
        print(json.dumps(rows[-1]),flush=True)
    spreads=[]
    with torch.no_grad():
        for index in indices:
            batch=batch_data(index,state);cache=cache_entry(index)
            terms,stats,f,z=loss_terms(model,batch,cache,criterion,.1,variants=())
            c=cache_tensors(cache,'cuda');targets=pool_regions(f,c['target_regions'])
            bg=pool_regions(f,c['background_regions']);valid=(c['negative_indices']>=0).all(1)
            negatives=c['negative_indices'][valid]
            for variant in ('B','C'):
                pairs=c['positive_pairs_'+variant][valid]
                anchors=targets[pairs[:,0]]
                pos=(anchors*targets[pairs[:,1]]).sum(-1,keepdim=True)
                neg=(anchors[:,None,:]*bg[negatives]).sum(-1)
                spreads.extend(torch.cat([pos,neg],1).std(1,unbiased=False).cpu().tolist())
            del terms,stats,f,z,c,targets,bg
    temperature=float(np.clip(2*np.median(spreads),.05,.5))
    gradient_rows=[];pathway={}
    for index in indices:
        batch=batch_data(index,state);cache=cache_entry(index)
        terms,stats,f,z=loss_terms(model,batch,cache,criterion,temperature)
        row={'index':index,'losses':{k:float(v.detach()) for k,v in terms.items()},'gradient_norms':{}}
        for key,value in terms.items():
            norm,grads=grad_norm(value,params)
            assert np.isfinite(norm) and norm>0,(key,norm)
            row['gradient_norms'][key]=norm
            if key in ('B','C') and key not in pathway:
                named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
                pathway[key]={prefix:sum(float(g.detach().double().square().sum()) for (n,p),g in zip(named,grads)
                    if n.startswith(prefix) and g is not None)**.5 for prefix in ('TPro.','conv_out1.','conv_out2.')}
                assert pathway[key]['TPro.']>0 and pathway[key]['conv_out1.']>0
                assert pathway[key]['conv_out2.']==0 # F precedes final projection: expected.
        gradient_rows.append(row);print(json.dumps(row),flush=True)
        del terms,stats,f,z,grads
    med={k:statistics.median(r['gradient_norms'][k] for r in gradient_rows) for k in ('det','keep','B','C')}
    raw_temp=.25*med['det']/max(med['B'],med['C'])
    raw_keep=.25*med['det']/med['keep']
    config={'temperature':temperature,'lambda_temporal':float(np.clip(raw_temp,1e-6,1e6)),
            'lambda_keep':float(np.clip(raw_keep,1e-6,1e6))}
    save_json(path,{'status':'complete','uses_development':False,'calibration_entry_indices':indices,
        'training_sources_covered':coverage,'seed':46,'released_checkpoint_sha256':sha(CP),
        'method':'discard8det updates; tau=clip(2median within-triplet rawcos std,.05,.5); lambdas target .25det median param-grad norm; temporal uses max of B/C medians; discard all pilot weights/optimizer',
        'lambda_safety_bounds':[1e-6,1e6],'unclipped_lambda_temporal':raw_temp,'unclipped_lambda_keep':raw_keep,
        'raw_cosine_triplet_std_median':float(np.median(spreads)),
        'gradient_norm_medians':med,'gradient_pathway':pathway,'warmup':rows,'measurements':gradient_rows,
        'fixed_settings':config,'elapsed_seconds':time.perf_counter()-started,
        'note':'one calibration shared by all formal conditions, no development-based choice; gradients of keep zero at release require discarded warmup'})

if __name__=='__main__':main()
