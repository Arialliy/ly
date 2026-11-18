"""R0/R1/R2 fixed2048, original frozen, shared direct object supervision."""
import argparse,hashlib,json,os,time
from pathlib import Path
import common as c

def train(args):
    import numpy as np
    import torch
    # Local modules explicitly loaded after historical imports.
    labels=c.load_module('ord_targets',c.N/'code/targets.py')
    losses=c.load_module('ord_losses',c.N/'code/losses.py')
    base,plan,names,sequences,_=c.previous.validate_inputs(c.default_args())
    old=[json.loads(s) for s in c.OLD_LOG.read_text().splitlines()]
    assert len(old)==512
    output=c.N/'training'/args.variant/'run'
    if output.exists() and any(output.iterdir()):raise FileExistsError(output)
    output.mkdir(parents=True,exist_ok=True)
    c.setup()
    model=c.create_model(base).to('cuda:0').train()
    initial_original={k:v.detach().cpu().clone() for k,v in model.original.state_dict().items()}
    original_hash=c.assert_original(model,initial_original)
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    named=c.branches(model)
    optimizer=torch.optim.Adam([p for _,p in named],lr=1e-4,weight_decay=1e-4)
    sources=c.sources()
    meta=dict(status='running',condition=args.variant,variant=args.variant,steps=2048,seed=46,
        lr=1e-4,weight_decay=1e-4,optimizer='Adam',temporal_enabled=c.VARIANTS[args.variant],
        assignment='track' if args.variant=='R2' else 'frame',snapshots=[],
        checkpoint_path=str(c.CHECKPOINT),checkpoint_sha256=c.sha(c.CHECKPOINT),
        source_plan_sha256=c.EXPECTED['plan'],plan_sha256=c.sha(c.N/'TRAIN_PLAN.json'),
        selection_sha256=c.EXPECTED['selection'],training_sequences=names,
        original_initial_state_sha256=original_hash,initial_model_state_sha256=c.tensor_hash(initial.items()),
        new_parameters=sum(p.numel() for _,p in named),parameter_count=sum(p.numel() for p in model.parameters()),
        model_config=model.config,source_sha256=sources,model_labels_access=False,
        torch_version=str(torch.__version__),precision='FP32_no_AMP_no_TF32',
        snapshots_selection='fixed2048only',loss_weights=[1,1,1,1,1],
        microbatches=4,repeat_source_plan=4,bn_policy='entire_original_frozen_eval')
    c.save_json(output/'INITIAL_STATE.json',dict(model_state_sha256=meta['initial_model_state_sha256'],
        original_state_sha256=original_hash,trainable_names=[name for name,_ in named],
        tensor_sha256={k:c.tensor_hash([(k,v)]) for k,v in initial.items()}))
    def snapshot(step):
        frozen=c.assert_original(model,initial_original)
        cp=dict(model_state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
            optimizer_state_dict=optimizer.state_dict(),step=step,condition=args.variant,
            original_state_sha256=frozen,model_config=model.config,source_sha256=sources,
            initial_checkpoint_sha256=meta['checkpoint_sha256'],plan_sha256=meta['plan_sha256'])
        path=output/f'snapshot_{step:04d}.pth';torch.save(cp,path)
        meta['snapshots'].append(dict(step=step,path=path.name,sha256=c.sha(path),original_state_sha256=frozen))
        c.save_json(output/'RUN.json',meta)
    snapshot(0)
    start=time.perf_counter();inputchain=hashlib.sha256();labelchain=hashlib.sha256()
    cache={};torch.cuda.reset_peak_memory_stats()
    try:
        with (output/'TRAIN.jsonl').open('w',buffering=1) as log:
            for i in range(2048):
                entry=plan['entries'][i%512]
                batch,positive,bg,specs=base.build_batch(entry,sequences)
                inputs=c.data_hash(batch,positive,bg,specs)
                assert inputs==old[i%512]['input_sha256'],('exposure',i)
                if i%512 not in cache:
                    cache[i%512]=labels.build_entry_targets(entry,sequences,background_masks=bg)
                targetlist=cache[i%512]
                targethash=labels.targets_digest(targetlist)
                inputchain.update(inputs.encode());labelchain.update(targethash.encode())
                optimizer.zero_grad(set_to_none=True)
                parts=[];assignments=[];total=0.
                origins=[tuple(p['crop_yx']) for p in entry['pairs'] for _ in p['starts']]
                for wi,target in enumerate(targetlist):
                    x=torch.from_numpy(batch[wi:wi+1]).cuda()
                    out=model(x,temporal_enabled=c.VARIANTS[args.variant],origin=origins[wi],full_shape=(512,640))
                    masks=model.mask_logits(out['E'],out['q'])[0]
                    loss,metrics,assignment=losses.object_response_loss(out['a'][0],masks,target,assignment=meta['assignment'])
                    if not bool(torch.isfinite(loss)):raise FloatingPointError('loss')
                    (loss/4).backward()
                    total+=float(loss.detach())/4;parts.append(metrics);assignments.append(assignment)
                    del x,out,masks,loss
                gradients=[p.grad for _,p in named if p.grad is not None]
                assert gradients and all(bool(torch.isfinite(g).all()) for g in gradients)
                assert not any(p.grad is not None for p in model.original.parameters())
                gradnorm=float(torch.linalg.vector_norm(torch.stack([g.detach().norm() for g in gradients])))
                optimizer.step()
                row=dict(step=i+1,source_entry=i%512,cycle=i//512,condition=args.variant,loss=total,
                    parts=parts,grad_norm=gradnorm,finite=True,original_parameters_with_gradient=0,
                    parameters_with_gradient=len(gradients),input_sha256=inputs,label_sha256=targethash,
                    assignment_sha256=hashlib.sha256(json.dumps(assignments,sort_keys=True).encode()).hexdigest(),
                    elapsed_seconds=time.perf_counter()-start,max_cuda_memory_allocated_bytes=torch.cuda.max_memory_allocated())
                if i==0 or (i+1)%128==0:
                    row['gradient_norms']={name:None if p.grad is None else float(p.grad.norm()) for name,p in named}
                log.write(json.dumps(row,allow_nan=False)+'\n')
                if i==0 or (i+1)%32==0:print(json.dumps({k:row[k] for k in ['step','loss','grad_norm','elapsed_seconds','max_cuda_memory_allocated_bytes']}),flush=True)
                if i+1 in [512,1024,2048]:snapshot(i+1)
        assert c.sources()==sources
        frozen=c.assert_original(model,initial_original)
        os.link(output/'snapshot_2048.pth',output/'final_model.pth')
        differences={name:dict(changed=int(torch.count_nonzero(p.detach().cpu()-initial[name])),
            max_abs=float((p.detach().cpu()-initial[name]).abs().max())) for name,p in model.named_parameters()}
        c.save_json(output/'PARAMETER_DIFF.json',differences)
        meta.update(status='complete',final_step=2048,duration_seconds=time.perf_counter()-start,
            original_final_state_sha256=frozen,frozen_original_unchanged=True,
            final_checkpoint_sha256=c.sha(output/'final_model.pth'),
            final_model_state_sha256=c.tensor_hash(model.state_dict().items()),
            cumulative_input_sha256=inputchain.hexdigest(),cumulative_label_sha256=labelchain.hexdigest(),
            train_log_sha256=c.sha(output/'TRAIN.jsonl'),max_cuda_memory_allocated_bytes=torch.cuda.max_memory_allocated(),
            max_cuda_memory_reserved_bytes=torch.cuda.max_memory_reserved())
        c.save_json(output/'RUN.json',meta);print(json.dumps({'status':'complete','condition':args.variant}),flush=True)
    except BaseException as error:
        meta.update(status='failed',error=repr(error),duration_seconds=time.perf_counter()-start)
        c.save_json(output/'RUN.json',meta);raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=list(c.VARIANTS),required=True)
    train(p.parse_args())
