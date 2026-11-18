"""Three controlled original-model adaptations, fixed step2048 endpoint."""
import argparse
import json
import time
import numpy as np
import torch
from runtime import (E,CP,MODEL_SOURCE,setup,load_model,load_module,freeze_for_training,
                     frozen_state,bn_state,tensor_hash,sha,save_json)
from data import inputs,batch_data,cache_entry,cache_tensors,source_fingerprints,historical
from region_contrast_loss import region_contrast_loss
from reliable_keep_loss import reliable_keep_loss

STEPS=2048
SNAPSHOTS=(0,128,512,1024,2048)


def forward(model,x):
    # Fixed microbatch2 throughout. All eight raw40-frame windows are read in A/B/C.
    outputs=[model(x[i:i+2]) for i in range(0,len(x),2)]
    return torch.cat([o[0] for o in outputs]),torch.cat([o[1] for o in outputs])


def loss_terms(model,batch,cache,criterion,temperature,variants=('B','C')):
    device=next(model.parameters()).device
    x=torch.from_numpy(batch['x']).to(device)
    features,logits=forward(model,x)
    target=torch.from_numpy(batch['targets']).to(device)
    valid=torch.from_numpy(batch['valid']).to(device)
    bg=torch.from_numpy(batch['det_background']).to(device)
    det,det_stats=historical.common_loss(logits[:4],target,valid,bg,criterion)
    c=cache_tensors(cache,device)
    keep,keep_stats=reliable_keep_loss(logits,c['keep_indices'],c['keep_probabilities'],c['keep_groups'])
    terms={'det':det,'keep':keep};stats={'det':det_stats,'keep':keep_stats}
    for variant in variants:
        value,info=region_contrast_loss(features,c['target_regions'],c['positive_pairs_'+variant],
            c['background_regions'],c['negative_indices'],temperature=temperature)
        terms[variant]=value;stats[variant]=info
    return terms,stats,features,logits


def grad_norm(loss,parameters,retain_graph=True):
    grads=torch.autograd.grad(loss,parameters,retain_graph=retain_graph,allow_unused=True)
    norm=float(torch.sqrt(sum((g.detach().double().square().sum() for g in grads if g is not None),
                              torch.zeros((),device=loss.device,dtype=torch.float64))))
    return norm,grads


def verify_cache(batch,cache):
    meta=json.loads(str(cache['metadata_json']))
    for key in ('input_sha256','label_sha256','auxiliary_input_sha256'):
        assert meta[key]==batch[key],key


def train(condition):
    setup();state=inputs();model,_=load_model();names,count=freeze_for_training(model)
    assert count==14593
    out=E/'training'/condition/'run'
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    out.mkdir(parents=True)
    frozen={k:v.detach().cpu().clone() for k,v in frozen_state(model).items()}
    frozen_hash=tensor_hash(frozen);bn_hash=tensor_hash(bn_state(model))
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    parameters=[p for p in model.parameters() if p.requires_grad]
    criterion=load_module('track_official_loss',MODEL_SOURCE).SoftLoUloss().cuda()
    optimizer=torch.optim.Adam(parameters,lr=1e-5,weight_decay=1e-4,betas=(.9,.999),eps=1e-8)
    calibration=json.loads((E/'CALIBRATION.json').read_text())
    config=calibration['fixed_settings']
    freeze=json.loads((E/'PROTOCOL_FREEZE.json').read_text())
    protocol_sha=sha(E/'PROTOCOL_FREEZE.json')
    assert sha(E/'CALIBRATION.json')==freeze['calibration_sha256']
    sources=source_fingerprints();assert sources==freeze['training_sources']
    for index in range(512):
        assert sha(E/'cache'/f'entry_{index:03d}.npz')==freeze['cache_sha256'][str(index)]
    metadata=dict(status='running',condition=condition,seed=46,final_step=None,steps=STEPS,
        trainable_parameter_names=names,trainable_parameter_count=count,trainable_parameters=count,total_parameters=70913,
        new_parameters=0,initial_checkpoint_sha256=sha(CP),model_source_sha256=sha(MODEL_SOURCE),
        protocol_sha256=protocol_sha,calibration_sha256=sha(E/'CALIBRATION.json'),
        training_sources=sources,settings=config,optimizer={'name':'Adam','lr':1e-5,'weight_decay':1e-4,
        'betas':[.9,.999],'eps':1e-8},snapshots=[],initial_state_sha256=tensor_hash(initial),
        frozen_state_sha256=frozen_hash,bn_initial_sha256=bn_hash,
        bn_policy='all buffers frozen eval; affine within TPro/heads trainable',
        exposure='512 existing four-detection-window entries repeated4; 4 common auxiliary windows per entry',
        precision='FP32 deterministic no AMP no TF32',torch_version=str(torch.__version__))

    def check_frozen():
        assert tensor_hash(frozen_state(model))==frozen_hash
        assert tensor_hash(bn_state(model))==bn_hash
        assert not model.training
        assert all(not m.training for m in model.modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm))

    def snapshot(step):
        check_frozen()
        checkpoint=dict(model_state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
            optimizer_state_dict=optimizer.state_dict(),condition=condition,step=step,seed=46,
            initial_checkpoint_sha256=sha(CP),model_source_sha256=sha(MODEL_SOURCE),protocol_sha256=protocol_sha,
            trainable_parameter_names=names,settings=config)
        path=out/f'snapshot_{step:03d}.pth';torch.save(checkpoint,path)
        metadata['snapshots'].append({'step':step,'path':path.name,'sha256':sha(path)})
        save_json(out/'RUN.json',metadata)

    snapshot(0);started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    try:
        with (out/'TRAIN.jsonl').open('w',buffering=1) as stream:
            for index in range(STEPS):
                pi=index%512;batch=batch_data(pi,state);cache=cache_entry(pi);verify_cache(batch,cache)
                optimizer.zero_grad(set_to_none=True)
                terms,stats,features,logits=loss_terms(model,batch,cache,criterion,config['temperature'],
                    variants=() if condition=='A' else (condition,))
                loss=terms['det']+config['lambda_keep']*terms['keep']
                if condition!='A':loss=loss+config['lambda_temporal']*terms[condition]
                assert bool(torch.isfinite(loss));loss.backward()
                grads={k:p.grad for k,p in model.named_parameters() if p.requires_grad}
                assert all(g is not None and bool(torch.isfinite(g).all()) for g in grads.values())
                row={'step':index+1,'plan_index':pi,'condition':condition,'loss':float(loss.detach()),
                    'loss_terms':{k:float(v.detach()) for k,v in terms.items()},'loss_statistics':stats,
                    'grad_norm':float(torch.linalg.vector_norm(torch.stack([g.norm() for g in grads.values()]))),
                    **{k:batch[k] for k in ('input_sha256','label_sha256','auxiliary_input_sha256')}}
                if index<2 or (index+1)%64==0:
                    row['parameter_grad_norms']={k:float(v.norm()) for k,v in grads.items()}
                optimizer.step()
                if index<2 or (index+1)%64==0:check_frozen()
                row['elapsed_seconds']=time.perf_counter()-started
                row['max_cuda_memory_allocated_bytes']=torch.cuda.max_memory_allocated()
                stream.write(json.dumps(row,allow_nan=False)+'\n')
                if index%64==0 or index+1 in SNAPSHOTS:
                    print(json.dumps({k:row[k] for k in ('step','condition','loss','grad_norm','elapsed_seconds',
                        'max_cuda_memory_allocated_bytes')}),flush=True)
                if index+1 in SNAPSHOTS:snapshot(index+1)
                del terms,stats,features,logits,loss,grads
        check_frozen();assert source_fingerprints()==sources
        metadata.update(status='complete',final_step=STEPS,frozen_frontend_unchanged=True,bn_unchanged=True,
            elapsed_seconds=time.perf_counter()-started,max_cuda_memory_allocated_bytes=torch.cuda.max_memory_allocated(),
            final_state_sha256=tensor_hash(model.state_dict()),
            parameter_changes={k:float((v.detach().cpu()-initial[k]).double().norm())
                               for k,v in model.named_parameters() if v.requires_grad})
        save_json(out/'RUN.json',metadata)
    except BaseException as exc:
        metadata.update(status='failed',error=repr(exc));save_json(out/'RUN.json',metadata);raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--condition',required=True,choices=['A','B','C'])
    train(p.parse_args().condition)
