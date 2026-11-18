"""Two 256-step negative-weight revisions of C2048; reuse the completed F control."""
import argparse
import json
import time
import torch
from runtime import E,V1,V2,CP,INIT_CP,MODEL_SOURCE,setup,load_module,frozen_state,bn_state,tensor_hash,sha,save_json,read_json
from data import inputs,batch_data,source_fingerprints
from common import initial_model,compute,readout_metrics,contrast_metrics,fingerprints,diagnostics,optimizer_hash,weighted_readout,MODES,scalarize


def train(condition):
    setup();state=inputs();model,optimizer,names,count,opt_hash=initial_model()
    out=E/'training'/condition/'run'
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    out.mkdir(parents=True)
    frozen_hash=tensor_hash(frozen_state(model));bn_hash=tensor_hash(bn_state(model))
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    criterion=load_module('v3_original_detector_loss',MODEL_SOURCE).SoftLoUloss().cuda()
    calibration=read_json(V2/'CALIBRATION.json');config=calibration['fixed_settings']
    freeze=read_json(E/'PROTOCOL_FREEZE.json');protocol_sha=sha(E/'PROTOCOL_FREEZE.json')
    assert sha(INIT_CP)==freeze['initial_checkpoint_sha256']==calibration['initial_checkpoint_sha256']
    assert sha(V2/'CALIBRATION.json')==freeze['calibration_sha256']
    assert fingerprints()==freeze['training_sources']
    assert source_fingerprints()==read_json(V1/'PROTOCOL_FREEZE.json')['training_sources']
    for index in range(256):
        assert sha(V2/'cache'/f'entry_{index:03d}.npz')==freeze['v2_cache_sha256'][str(index)]
        assert sha(V1/'cache'/f'entry_{index:03d}.npz')==freeze['v1_cache_sha256'][str(index)]
    assert opt_hash==calibration['initial_optimizer_sha256']
    metadata=dict(status='running',condition=condition,seed=46,start_step=2048,additional_steps=256,
        final_step=None,steps=256,initial_optimizer_restored=True,initial_optimizer_sha256=opt_hash,
        trainable_parameter_names=names,trainable_parameter_count=count,trainable_parameters=count,total_parameters=70913,
        new_parameters=0,initial_checkpoint_sha256=sha(INIT_CP),release_checkpoint_sha256=sha(CP),
        model_source_sha256=sha(MODEL_SOURCE),protocol_sha256=protocol_sha,calibration_sha256=sha(V2/'CALIBRATION.json'),
        training_sources=freeze['training_sources'],settings=config,
        readout={'mode':MODES[condition],'gamma':2.,'group':'existing_segment8','F_reference':'track_contrast_v2/FRAME'},
        optimizer={'name':'Adam','lr':1e-5,'weight_decay':1e-4,'betas':[.9,.999],'eps':1e-8,'scheduler':None},
        snapshots=[],initial_state_sha256=tensor_hash(initial),frozen_state_sha256=frozen_hash,bn_initial_sha256=bn_hash,
        bn_policy='all buffers frozen eval; affine within TPro/heads trainable',
        exposure='original plan indices 0..255, original 4 detection windows + 4 common auxiliary windows',
        precision='FP32 deterministic no AMP no TF32',torch_version=str(torch.__version__))

    def check_frozen():
        assert tensor_hash(frozen_state(model))==frozen_hash and tensor_hash(bn_state(model))==bn_hash
        assert not model.training

    def snapshot(step):
        check_frozen()
        checkpoint=dict(model_state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
            optimizer_state_dict=optimizer.state_dict(),condition=condition,step=step,seed=46,
            initial_checkpoint_sha256=sha(INIT_CP),release_checkpoint_sha256=sha(CP),
            model_source_sha256=sha(MODEL_SOURCE),protocol_sha256=protocol_sha,
            trainable_parameter_names=names,settings=config)
        path=out/f'snapshot_{step}.pth';torch.save(checkpoint,path)
        metadata['snapshots'].append({'step':step,'path':path.name,'sha256':sha(path)})
        save_json(out/'RUN.json',metadata)

    snapshot(2048);started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    try:
        with (out/'TRAIN.jsonl').open('w',buffering=1) as stream:
            for index in range(256):
                batch=batch_data(index,state);optimizer.zero_grad(set_to_none=True)
                terms,stats,f,z,r,c=compute(model,batch,index,criterion,config)
                w=weighted_readout(r,c,MODES[condition])
                terms.update(weighted_frame=w['loss'],readout_positive=w['loss_positive'],readout_negative=w['loss_negative'])
                loss=terms['base']+config['lambda_readout']*terms['weighted_frame']
                assert bool(torch.isfinite(loss));loss.backward()
                grads={k:p.grad for k,p in model.named_parameters() if p.requires_grad}
                assert all(g is not None and bool(torch.isfinite(g).all()) for g in grads.values())
                row={'step':2049+index,'additional_step':index+1,'plan_index':index,'condition':condition,
                    'loss':float(loss.detach()),'loss_terms':{k:float(v.detach()) for k,v in terms.items()},
                    'contrast':contrast_metrics(stats),'readout':readout_metrics(r,c),'weighting':scalarize(w['diagnostics']),
                    'grad_norm':float(torch.linalg.vector_norm(torch.stack([g.norm() for g in grads.values()]))),
                    **{k:batch[k] for k in ('input_sha256','label_sha256','auxiliary_input_sha256')}}
                optimizer.step()
                if index<2 or (index+1)%64==0:
                    check_frozen();row['parameter_grad_norms']={k:float(v.norm()) for k,v in grads.items()}
                    print(json.dumps({k:row[k] for k in ('step','condition','loss','grad_norm')}),flush=True)
                row['elapsed_seconds']=time.perf_counter()-started
                row['max_cuda_memory_allocated_bytes']=torch.cuda.max_memory_allocated()
                stream.write(json.dumps(row,allow_nan=False)+'\n')
                if index+1 in (64,128,256):snapshot(2049+index)
                del terms,stats,f,z,r,c,w,loss,grads
        assert all(float(s['step'])==2304 for s in optimizer.state.values())
        save_json(out/'TRAIN_DIAGNOSTICS.json',{'indices':calibration['calibration_entry_indices'],
            'uses_development':False,'rows':diagnostics(model,state,criterion,config,MODES[condition])})
        check_frozen();assert fingerprints()==freeze['training_sources']
        metadata.update(status='complete',final_step=2304,frozen_frontend_unchanged=True,bn_unchanged=True,
            elapsed_seconds=time.perf_counter()-started,max_cuda_memory_allocated_bytes=torch.cuda.max_memory_allocated(),
            final_state_sha256=tensor_hash(model.state_dict()),final_optimizer_sha256=optimizer_hash(optimizer.state_dict()),
            parameter_changes={k:float((v.detach().cpu()-initial[k]).double().norm()) for k,v in model.named_parameters() if v.requires_grad})
        save_json(out/'RUN.json',metadata)
    except BaseException as exc:
        metadata.update(status='failed',error=repr(exc));save_json(out/'RUN.json',metadata);raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--condition',required=True,choices=['D','N'])
    train(parser.parse_args().condition)
