"""Exact C-v1 loss plus the reference risk applied to original detector logits."""
import hashlib
import json
import sys
import numpy as np
import torch
from runtime import E,V1,INIT_CP,REFERENCE,MODEL_SOURCE,load_module,load_model,freeze_for_training,sha,tensor_hash
from data import inputs,batch_data,cache_entry,cache_tensors,source_fingerprints

old = load_module('c_v1_training_terms', V1/'code/train.py')
risk_module = load_module('c_v2_readout_adapter', E/'train_modules/temporal_risk.py')
RISK_KEYS = ('target_regions','background_regions','negative_indices','segment_indices','segment_windows')
CALIBRATION_INDICES = [5,6,18,25,30,32,51,7]


def optimizer_hash(state):
    tensors={}; scalars={}
    for key,values in state['state'].items():
        for name,value in values.items():
            if torch.is_tensor(value):tensors[f'{key}/{name}']=value
            else:scalars[f'{key}/{name}']=value
    text=json.dumps({'groups':state['param_groups'],'scalars':scalars},sort_keys=True)
    return hashlib.sha256((tensor_hash(tensors)+text).encode()).hexdigest()


def initial_model():
    model,_=load_model(INIT_CP)
    names,count=freeze_for_training(model)
    checkpoint=torch.load(INIT_CP,map_location='cpu',weights_only=True)
    assert checkpoint['step']==2048 and checkpoint['condition']=='C'
    assert names==checkpoint['trainable_parameter_names'] and count==14593
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.Adam(parameters,lr=1e-5,weight_decay=1e-4,betas=(.9,.999),eps=1e-8)
    expected=optimizer_hash(checkpoint['optimizer_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    assert optimizer_hash(optimizer.state_dict())==expected
    assert len(optimizer.state)==28
    assert all(float(s['step'])==2048 for s in optimizer.state.values())
    assert all(g['lr']==1e-5 and g['weight_decay']==1e-4 for g in optimizer.param_groups)
    return model,optimizer,names,count,expected


def risk_cache(index,batch):
    with np.load(E/'cache'/f'entry_{index:03d}.npz',allow_pickle=False) as source:
        c={k:source[k] for k in source.files}
    old.verify_cache(batch,c)
    return c


def compute(model,batch,index,criterion,config):
    cache=cache_entry(index);old.verify_cache(batch,cache)
    terms,stats,features,logits=old.loss_terms(model,batch,cache,criterion,config['temperature'],variants=('C',))
    terms['base']=terms['det']+config['lambda_keep']*terms['keep']+config['lambda_temporal']*terms['C']
    c=risk_cache(index,batch)
    values={k:torch.from_numpy(c[k]).to(logits.device) for k in RISK_KEYS}
    risk=risk_module.temporal_risk_loss(logits,**values,num_windows=8)
    terms['frame']=risk['loss_frame'];terms['gap']=risk['loss_gap']
    return terms,stats,features,logits,risk,c


def readout_metrics(risk,c):
    idx=torch.as_tensor(c['segment_indices'],device=risk['q_positive'].device).flatten()
    idx=idx[idx>=0]
    if len(idx)==0:return {'observations_with_multiplicity':0}
    a=risk['q_positive'][idx].detach();b=risk['q_negative'][idx].detach()
    return {'observations_with_multiplicity':len(idx),'target_logit_mean':float(a.mean()),
        'background_logit_mean':float(b.mean()),'logit_margin_mean':float((a-b).mean()),
        'probability_margin_mean':float((a.sigmoid()-b.sigmoid()).mean()),
        'evidence_mean':float((a.sigmoid()*(-b).sigmoid()).mean()),
        'target_logit_above_zero_fraction':float((a>0).float().mean()),
        'background_logit_above_zero_fraction':float((b>0).float().mean())}


def contrast_metrics(stats):
    c=stats['C'];out=dict(c)
    if 'positive_cosine_mean' in c:out['cosine_margin_mean']=c['positive_cosine_mean']-c['negative_cosine_mean']
    return out


def fingerprints():
    files=[E/'code'/x for x in ('runtime.py','data.py','common.py','train.py','calibrate.py','prepare_segments.py')]
    files+=sorted((E/'train_modules').glob('*.py'))
    files+=[REFERENCE/x for x in ('temporal_risk.py','segments.py','test_reference.py','README.md','VALIDATION.json')]
    return {str(p):sha(p) for p in files}


def diagnostics(model,state,criterion,config):
    rows=[]
    with torch.no_grad():
        for index in CALIBRATION_INDICES:
            batch=batch_data(index,state)
            terms,stats,f,z,r,c=compute(model,batch,index,criterion,config)
            rows.append({'index':index,'losses':{k:float(v) for k,v in terms.items()},
                'contrast':contrast_metrics(stats),'readout':readout_metrics(r,c)})
    return rows
