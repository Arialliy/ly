"""128 identical new exposures: old History continuation vs one state-update change."""
import argparse
import fcntl
import hashlib
import json
import time
import torch
import runtime as rt

def train(condition,device='cuda'):
    out=rt.E/'training'/condition;out.mkdir(parents=True,exist_ok=True)
    with (out/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run_path=out/'RUN.json'
        if run_path.exists():raise FileExistsError('Preserve existing paired-short run: '+str(run_path))
        prior=rt.old();sources=rt.source_hashes()
        freeze=rt.read_json(rt.E/'PROTOCOL_FREEZE.json');assert freeze['source_sha256']==sources
        model,parent=rt.load_initial(condition,device)
        checkpoint=torch.load(rt.CHECKPOINT,map_location='cpu',weights_only=True)
        assert checkpoint['step']==2048 and checkpoint['condition']=='History'
        assert prior.tensor_hash(model.state_dict())==parent['final_model_state_sha256']
        model.train()
        parameters={k:p for k,p in model.named_parameters() if p.requires_grad}
        assert list(parameters)==parent['trainable_parameter_names']
        optimizer=torch.optim.Adam(list(parameters.values()),lr=1e-5,weight_decay=1e-4,betas=(.9,.999),eps=1e-8)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        assert all(float(v['step'])==2048 for v in optimizer.state.values())
        assert len(optimizer.state)==len(parameters)==53
        training_impl=rt.old_train();training_impl.restore_rng(checkpoint['rng_state'])
        frozen=prior.frozen_f();data=frozen.data.inputs()
        assert prior.frozen_cache_manifest()==parent['cache_sha256']
        criterion=prior.load_module('_m11_loss',prior.MODEL_SOURCE).SoftLoUloss().to(device)
        run={'status':'RUNNING','condition':condition,'seed':46,'initial_step':2048,'additional_steps':128,
            'initial_checkpoint_sha256':rt.sha(rt.CHECKPOINT),'initial_model_state_sha256':parent['final_model_state_sha256'],
            'original_state_sha256':parent['original_state_sha256'],'source_sha256':sources,
            'frozen_v1_source_sha256':prior.source_hashes(),'protocol_sha256':rt.sha(rt.E/'PROTOCOL_FREEZE.json'),
            'settings':prior.SETTINGS,'cache_sha256':parent['cache_sha256'],
            'trainable_parameter_names':list(parameters),'trainable_parameters':sum(p.numel() for p in parameters.values()),
            'parameters':sum(p.numel() for p in model.parameters()),
            'optimizer':'Restore identical History2048 Adam state for all53 unchanged parameter names/shapes; lr1e-5 wd1e-4',
            'rng':'Restore identical checkpoint RNG after model construction/revision installation',
            'exposure':'Existing frozen F entries0..127 once, in order; 4 detection+4 auxiliary per entry',
            'revision':'A+TSFM(A,X)' if condition=='Revision' else 'TSFM(A,X)',
            'all_original_parameters_and_BN_frozen':True,'snapshots':[]}
        rt.save(run_path,run);prior.assert_frozen(model,run['original_state_sha256'])
        initial={k:p.detach().cpu().clone() for k,p in parameters.items()}
        torch.cuda.reset_peak_memory_stats();began=time.perf_counter();digest=hashlib.sha256()
        try:
            with (out/'TRAIN.jsonl').open('x',buffering=1) as stream:
                for index in range(rt.STEPS):
                    batch=frozen.data.batch_data(index,data)
                    optimizer.zero_grad(set_to_none=True)
                    terms,stats,features,logits,risk,cache=frozen.common.compute(model,batch,index,criterion,prior.SETTINGS)
                    loss=terms['base']+prior.SETTINGS['lambda_readout']*terms['frame']
                    assert bool(torch.isfinite(loss));loss.backward()
                    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters.values())
                    row={'additional_step':index+1,'step':2049+index,'plan_index':index,'condition':condition,
                        'loss':float(loss.detach()),'terms':{k:float(v.detach()) for k,v in terms.items() if k!='gap'},
                        'grad_norm':float(torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in parameters.values()]))),
                        **{k:batch[k] for k in ('input_sha256','label_sha256','auxiliary_input_sha256')}}
                    optimizer.step()
                    for key in ('input_sha256','label_sha256','auxiliary_input_sha256'):digest.update(row[key].encode())
                    row['elapsed_seconds']=time.perf_counter()-began
                    stream.write(json.dumps(row,allow_nan=False)+'\n')
                    if index<2 or (index+1)%16==0:print({k:row[k] for k in ('condition','step','loss','grad_norm','elapsed_seconds')},flush=True)
                    if (index+1)%16==0:
                        run['completed_additional_steps']=index+1;rt.save(run_path,run)
                    del terms,stats,features,logits,risk,cache,loss,batch
            prior.assert_frozen(model,run['original_state_sha256'])
            assert sources==rt.source_hashes()
            assert all(float(v['step'])==2176 for v in optimizer.state.values())
            path=out/'snapshot_2176.pth'
            prior.atomic_torch_save(path,{'model_state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                'optimizer_state_dict':optimizer.state_dict(),'rng_state':training_impl.rng_state(),
                'step':2176,'condition':condition,'source_sha256':sources,'protocol_sha256':run['protocol_sha256'],
                'initial_checkpoint_sha256':rt.sha(rt.CHECKPOINT)})
            run.update(status='COMPLETE',final_step=2176,completed_additional_steps=128,
                checkpoint_sha256=rt.sha(path),final_model_state_sha256=prior.tensor_hash(model.state_dict()),
                exposure_digest_sha256=digest.hexdigest(),original_unchanged=True,
                elapsed_seconds=time.perf_counter()-began,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                parameter_changes={k:float((p.detach().cpu()-initial[k]).double().norm()) for k,p in parameters.items()})
            rt.save(run_path,run)
        except BaseException as error:
            run.update(status='FAILED',error=repr(error),elapsed_seconds=time.perf_counter()-began);rt.save(run_path,run);raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--condition',required=True,choices=rt.CONDITIONS);p.add_argument('--device',default='cuda')
    a=p.parse_args();train(a.condition,a.device)
