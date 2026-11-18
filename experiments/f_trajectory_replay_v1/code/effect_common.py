"""Frozen F2304 losses and optimizer; only the single-step intervention is new."""
import copy
import hashlib
import json
import random
import numpy as np
import torch
from runtime import E,V1,V2,INIT_CP,MODEL_SOURCE,load_module,load_model,freeze_for_training,sha,tensor_hash,read_json
from data import cache_entry

old=load_module('effective_update_original_C',V1/'code/train.py')
risk_module=load_module('effective_update_original_frame',V2/'train_modules/temporal_risk.py')
RISK_KEYS=('target_regions','background_regions','negative_indices','segment_indices','segment_windows')


def optimizer_hash(state):
    tensors={};scalars={}
    for key,values in state['state'].items():
        for name,value in values.items():
            if torch.is_tensor(value):tensors[f'{key}/{name}']=value
            else:scalars[f'{key}/{name}']=value
    encoded=json.dumps({'groups':state['param_groups'],'scalars':scalars},sort_keys=True)
    return hashlib.sha256((tensor_hash(tensors)+encoded).encode()).hexdigest()


def initialize():
    checkpoint=torch.load(INIT_CP,map_location='cpu',weights_only=True)
    assert checkpoint['step']==2304 and checkpoint['condition']=='FRAME'
    model,_=load_model(INIT_CP);names,count=freeze_for_training(model)
    assert names==checkpoint['trainable_parameter_names'] and count==14593
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.Adam(parameters,lr=1e-5,weight_decay=1e-4,betas=(.9,.999),eps=1e-8)
    optimizer.load_state_dict(copy.deepcopy(checkpoint['optimizer_state_dict']))
    assert optimizer_hash(optimizer.state_dict())==optimizer_hash(checkpoint['optimizer_state_dict'])
    assert len(optimizer.state)==28 and all(float(s['step'])==2304 for s in optimizer.state.values())
    config=read_json(V2/'CALIBRATION.json')['fixed_settings']
    assert checkpoint['settings']==config
    criterion=load_module('effective_update_detector_loss',MODEL_SOURCE).SoftLoUloss().cuda()
    return model,optimizer,checkpoint,criterion,config


def cache(index,batch=None):
    with np.load(V2/'cache'/f'entry_{index:03d}.npz',allow_pickle=False) as source:
        result={key:source[key] for key in source.files}
    if batch is not None:old.verify_cache(batch,result)
    return result


def risk(logits,c):
    values={k:torch.as_tensor(c[k],device=logits.device) for k in RISK_KEYS}
    return risk_module.temporal_risk_loss(logits,**values,num_windows=8)


def compute(model,batch,index,criterion,config):
    c1=cache_entry(index);old.verify_cache(batch,c1)
    terms,stats,features,logits=old.loss_terms(model,batch,c1,criterion,config['temperature'],variants=('C',))
    c=cache(index,batch);r=risk(logits,c)
    terms['frame']=r['loss_frame']
    terms['fit']=terms['det']+config['lambda_readout']*terms['frame']
    terms['weighted_keep']=config['lambda_keep']*terms['keep']
    terms['weighted_track']=config['lambda_temporal']*terms['C']
    # Exact old F operation order, preserving its existing scalar loss.
    terms['full_F']=(terms['det']+terms['weighted_keep']+terms['weighted_track'])+config['lambda_readout']*terms['frame']
    terms['no_keep']=(terms['det']+terms['weighted_track'])+config['lambda_readout']*terms['frame']
    terms['no_track']=(terms['det']+terms['weighted_keep'])+config['lambda_readout']*terms['frame']
    return terms,logits,r,c


def forward(model,batch):
    with torch.no_grad():
        x=torch.as_tensor(batch['x'],device=next(model.parameters()).device)
        return torch.cat([model(x[i:i+2])[1] for i in range(0,8,2)])


def get_rng():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state().clone(),
                cuda=[s.clone() for s in torch.cuda.get_rng_state_all()])


def set_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])


def rng_hash():
    state=get_rng()
    h=hashlib.sha256(repr(state['python']).encode())
    h.update(repr(state['numpy'][0]).encode());h.update(state['numpy'][1].tobytes());h.update(repr(state['numpy'][2:]).encode())
    h.update(state['torch'].numpy().tobytes())
    for t in state['cuda']:h.update(t.cpu().numpy().tobytes())
    return h.hexdigest()


def vector(grads,parameters):
    return torch.cat([(torch.zeros_like(p) if g is None else g.detach()).reshape(-1) for g,p in zip(grads,parameters)])


def assign(parameters,grads):
    for p,g in zip(parameters,grads):p.grad=None if g is None else g.detach().clone()


def gradients(loss,parameters):
    return tuple(None if x is None else x.detach().clone() for x in
                 torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True))


def norm(v):return float(v.detach().double().norm())


def cosine(a,b):
    a=a.detach().double();b=b.detach().double();denom=a.norm()*b.norm()
    return float(torch.dot(a,b)/denom) if float(denom)>0 else None
