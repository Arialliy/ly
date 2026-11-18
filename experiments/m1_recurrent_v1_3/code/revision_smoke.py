"""Native CUDA source-bank checks and measured40-frame inference/training costs.

All optimized models are disposable. Data come only from frozen training inputs.
The supplied bank core, original PDA/DCN/TSFM and53 trained tensors are preserved.
No spatial M1 tiling, altered capacity, detached bank or substitute operator.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import inspect
from pathlib import Path
import time
import traceback

import numpy as np
import torch

import runtime as rt
import revision

PAIRS = sum(min(t, 4) for t in range(40))
HASH_KEYS = ('input_sha256', 'label_sha256', 'auxiliary_input_sha256')


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def begin():
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    return time.perf_counter()


def cost(start):
    torch.cuda.synchronize()
    return dict(seconds=time.perf_counter()-start,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())


def tested_sources():
    files = [Path(__file__), rt.E/'code/revision.py', rt.E/'code/runtime.py', rt.REFERENCE/'evidence_bank.py',
             rt.ROOT/'models/adapters/window_recurrent_adapter.py']
    if (rt.E/'code/trace.py').exists():files.append(rt.E/'code/trace.py')
    return {str(p):rt.sha(p) for p in files}


def check_initial(model, parent, checkpoint):
    prior = rt.old()
    state = model.state_dict()
    assert set(state) == set(checkpoint['model_state_dict'])
    assert all(torch.equal(value.cpu(), state[name].cpu()) for name,value in checkpoint['model_state_dict'].items())
    parameters = {name:p for name,p in model.named_parameters() if p.requires_grad}
    assert list(parameters) == parent['trainable_parameter_names'] and len(parameters) == 53
    assert sum(p.numel() for p in parameters.values()) == 109350
    assert all('state_update' not in key for key in state)
    assert type(model.adapter.recurrent_cell).__name__ == 'RFRCell'
    assert Path(inspect.getfile(revision.bank)).resolve() == (rt.REFERENCE/'evidence_bank.py').resolve()
    assert model.adapter.capacity == 4 and model.adapter.checkpoint_recurrence
    assert bool(torch.count_nonzero(model.adapter.out_projection.weight))
    prior.assert_frozen(model, parent['original_state_sha256'])
    # load_state_dict can share CPU scalar step tensors with its input mapping.
    # Each disposable arm must own its optimizer state, just as separate jobs do.
    optimizer, receipt = rt.make_optimizer(model, parent, deepcopy(checkpoint))
    saved = checkpoint['optimizer_state_dict']
    assert optimizer.state_dict()['param_groups'] == saved['param_groups']
    for name, identifier in zip(parameters, saved['param_groups'][0]['params']):
        expected, actual = saved['state'][identifier], optimizer.state[parameters[name]]
        assert expected.keys() == actual.keys()
        for key,value in expected.items():
            if torch.is_tensor(value):
                assert torch.equal(value.cpu(), actual[key].cpu()), (name,key)
            else:
                assert value == actual[key]
    return optimizer, dict(receipt, all_optimizer_state_tensors_exact=True,
                           all_model_keys_and_tensors_exact=True, trainable_parameters=109350,
                           total_parameters=sum(p.numel() for p in model.parameters()))


def training_source(frozen_data):
    name = sorted(frozen_data[2])[0]
    source = frozen_data[2][name]
    assert name not in rt.SEQS
    frames = np.array((source.frames[:40].astype(np.float32)-85.0799)/47.4845, copy=True)
    assert frames.shape == (40,512,640)
    return frames, dict(sequence=name, split='training', frames=[0,40], raw_file_sha256=rt.sha(source.raw_path))


@torch.no_grad()
def off_from_h(model, h):
    return torch.stack([model.base.conv_out2(model.base.conv_out1(h[:,:,t:t+1]))[:,0,0]
                        for t in range(h.shape[2])],dim=1).cpu()


@torch.no_grad()
def small_frontend_and_off(frames):
    model, parent = rt.load_initial('BankHistory')
    model.eval()
    x = torch.from_numpy(frames[:,:160,:192].copy())[None,None].cuda()
    full_f = model.base.layer1(model.base.conv_in(x))
    full_h = model.base.TPro(full_f.permute(0,3,4,1,2))
    f,h = model.frozen_frontend_tiled(x)
    errors = {'F':float((f-full_f).abs().max()), 'H':float((h-full_h).abs().max())}
    torch.testing.assert_close(f,full_f,atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(h,full_h,atol=1e-5,rtol=1e-5)
    baseline = model.base(x)[1].cpu()
    off = off_from_h(model,h)
    assert torch.equal(off,baseline)
    rt.old().assert_frozen(model,parent['original_state_sha256'])
    result = dict(status='PASS',input_shape=list(x.shape), frontend_maximum_absolute_errors=errors,
                  off_equals_original_head_and_full_baseline_exact=True,
                  meaning='Off uses only original H and original head, including no Wout bias')
    del model,x,f,h,full_f,full_h,baseline,off
    cleanup()
    return result


@torch.no_grad()
def fullframe(condition,frames):
    model,parent = rt.load_initial(condition)
    model.eval()
    initial = rt.old().tensor_hash(model.state_dict())
    x = torch.from_numpy(frames.copy())[None,None].cuda()
    start = begin()
    baseline = rt.old().frozen_f().runtime.tiled_logits(model.base,x,core=128,halo=16)
    baseline_cost = cost(start)
    f,h = model.frozen_frontend_tiled(x)
    off = off_from_h(model,h)
    torch.testing.assert_close(off,baseline,atol=1e-5,rtol=1e-5)
    off_error = float((off-baseline).abs().max())
    off_flips = int(((off>=0)!=(baseline>=0)).sum())
    assert off_flips == 0
    del f,h,off,baseline
    cleanup()
    shapes, align_shapes, fuse_shapes = [],[],[]
    handles = [model.adapter.register_forward_pre_hook(lambda _,args:shapes.append([list(v.shape) for v in args])),
        model.adapter.recurrent_cell.pda.register_forward_pre_hook(lambda _,args:align_shapes.append([list(v.shape) for v in args])),
        model.adapter.recurrent_cell.tsfm.register_forward_pre_hook(lambda _,args:fuse_shapes.append([list(v.shape) for v in args]))]
    try:
        start = begin()
        outputs = model.forward_full_spatial(x)
        measured = cost(start)
    finally:
        for handle in handles:handle.remove()
    assert shapes == [[[1,32,40,512,640],[1,32,40,512,640]]]
    assert len(align_shapes) == len(fuse_shapes) == PAIRS == 150
    assert all(pair == [[1,16,512,640],[1,16,512,640]] for pair in align_shapes+fuse_shapes)
    assert all(bool(torch.isfinite(value).all()) for value in outputs)
    assert rt.old().tensor_hash(model.state_dict()) == initial
    rt.old().assert_frozen(model,parent['original_state_sha256'])
    result = dict(status='PASS',condition=condition,mode=model.mode,input_shape=list(x.shape),
        baseline_off=baseline_cost,complete_window=measured,
        off_original_tiled_logit_absmax=off_error,off_threshold05_disagreements=off_flips,
        adapter_calls=1,pda_calls=len(align_shapes),tsfm_calls=len(fuse_shapes),
        all_pair_inputs_full_spatial=True,pair_shape=[1,16,512,640],output_shapes=[list(v.shape) for v in outputs],
        model_state_unchanged=True,execution='frontend core128/halo16 then one global bank; original head temporalchunk1; no spatial M1 tiling')
    del model,x,outputs
    cleanup()
    return result


def grouped_norm(values):
    groups = {'input_projection':'adapter.in_projection.', 'pda':'adapter.recurrent_cell.pda.',
              'tsfm':'adapter.recurrent_cell.tsfm.', 'output_projection':'adapter.out_projection.'}
    return {group:sum(v*v for name,v in values.items() if name.startswith(prefix))**.5 for group,prefix in groups.items()}


def train_two_steps(condition,data,checkpoint):
    prior = rt.old()
    model,parent = rt.load_initial(condition)
    optimizer, identity = check_initial(model,parent,checkpoint)
    frozen = prior.frozen_f()
    criterion = prior.load_module('_m13_smoke_loss',prior.MODEL_SOURCE).SoftLoUloss().cuda()
    rt.old_train().restore_rng(checkpoint['rng_state'])
    model.train()
    parameters = {name:p for name,p in model.named_parameters() if p.requires_grad}
    steps = []
    for index in (0,1):
        batch = frozen.data.batch_data(index,data)
        assert batch['x'].shape == (8,1,40,128,128)
        before = {name:p.detach().clone() for name,p in parameters.items()}
        optimizer.zero_grad(set_to_none=True)
        forward_pair_counts = dict(pda=0,tsfm=0)
        def count(name):
            def hook(module,args):forward_pair_counts[name]+=1
            return hook
        hooks = [model.adapter.recurrent_cell.pda.register_forward_pre_hook(count('pda')),
                 model.adapter.recurrent_cell.tsfm.register_forward_pre_hook(count('tsfm'))]
        start = begin()
        try:
            terms,stats,features,logits,risk,cache = frozen.common.compute(model,batch,index,criterion,prior.SETTINGS)
        finally:
            for hook in hooks:hook.remove()  # checkpoint backward recomputations are not forward work
        loss = terms['base']+prior.SETTINGS['lambda_readout']*terms['frame']
        assert bool(torch.isfinite(loss))
        loss.backward()
        gradients = {}
        for name,p in parameters.items():
            assert p.grad is not None and bool(torch.isfinite(p.grad).all()),name
            gradients[name] = float(p.grad.double().norm())
        optimizer.step()
        measured = cost(start)
        updates = {name:float((p.detach()-before[name]).double().norm()) for name,p in parameters.items()}
        assert all(v>0 for v in grouped_norm(gradients).values())
        assert all(v>0 for v in grouped_norm(updates).values())
        assert all(float(state['step'])==2049+index for state in optimizer.state.values())
        assert all(p.grad is None for p in model.base.parameters())
        prior.assert_frozen(model,parent['original_state_sha256'])
        # Four microbatches of two complete windows, each with150 pair calls.
        assert forward_pair_counts == {'pda':600,'tsfm':600},forward_pair_counts
        row = dict(additional_step=index+1,step=2049+index,plan_index=index,
            input_shape=list(batch['x'].shape),microbatch=2,
            **{key:batch[key] for key in HASH_KEYS},
            complete_input_tensor_sha256=prior.tensor_hash({'x':torch.from_numpy(batch['x'])}),
            loss=float(loss.detach()),parameter_gradient_l2=gradients,parameter_update_l2=updates,
            group_gradient_l2=grouped_norm(gradients),group_update_l2=grouped_norm(updates),
            forward_pair_calls=forward_pair_counts,original_parameters_and_BN_unchanged=True,**measured)
        steps.append(row)
        print({'training':condition, 'additional_step':index+1, 'cost':measured, 'loss':row['loss']},flush=True)
        del batch,before,terms,stats,features,logits,risk,cache,loss
    result = dict(status='PASS',condition=condition,mode=model.mode,initial_identity=identity,steps=steps,
                  optimized_state_saved=False,optimizer='Exact old53 Adam restored at2048; same settings and two frozen exposures',
                  interpretation='Nonzero loss gradients and actual parameter deltas separately recorded; weight decay alone is not counted as learning')
    del model,optimizer,parameters,criterion
    cleanup()
    return result


def native_bank_semantics(frames):
    model,parent = rt.load_initial('BankHistory')
    model.eval()
    initial = rt.old().tensor_hash(model.state_dict())
    x = torch.from_numpy(frames[:,:64,:64].copy())[None,None].cuda()
    with torch.no_grad():
        f,h = model.frozen_frontend_tiled(x)
        projected = torch.stack([model.adapter.in_projection(f[:,:,t]) for t in range(40)],2)
        before = projected.clone()
        state = revision.bank.start_window('native',4)
        records=[];readouts=[];snapshots=[]
        pda,tsfm = model.adapter.recurrent_cell.pda,model.adapter.recurrent_cell.tsfm
        counts=dict(pda=0,tsfm=0)
        def align(a,b):counts['pda']+=1;return pda(a.contiguous(),b.contiguous())
        def fuse(a,b):counts['tsfm']+=1;return tsfm(a,b)
        for t in range(40):
            value,state=revision.bank.read_then_write(projected[:,:,t],state,window_id='native',step_index=t,align=align,fuse=fuse)
            readouts.append(value)
            expected=list(range(max(0,t-3),t+1))
            assert [entry.step_index for entry in state.entries]==expected
            assert all(torch.equal(entry.feature,projected[:,:,entry.step_index]) for entry in state.entries)
            assert all(entry.feature.data_ptr()!=projected[:,:,entry.step_index].data_ptr() for entry in state.entries)
            snapshots.extend((entry,entry.feature.clone()) for entry in state.entries if entry.step_index==t)
            records.append({'step':t,'stored_source_indices':expected})
        assert counts=={'pda':150,'tsfm':150}
        assert all(torch.equal(entry.feature,old) for entry,old in snapshots)
        assert torch.equal(projected,before) and torch.equal(readouts[0],projected[:,:,0])
        readouts=torch.stack(readouts,2)
        expected=torch.stack([h[:,:,t]+model.adapter.out_projection(torch.cat((projected[:,:,t],readouts[:,:,t]),1)) for t in range(40)],2)
        actual=model.adapter(f,h,mode='history')
        assert torch.equal(actual,expected)
        chunked=model.adapter(f,h,mode='history',temporal_chunk_size=7)
        assert torch.equal(actual,chunked)
        changed=f.clone();changed[:,:,0]+=.25
        changed_h=model.adapter(changed,h,mode='history')
        normal_c=model.adapter(f,h,mode='current_only')
        changed_c=model.adapter(changed,h,mode='current_only')
        assert torch.equal(normal_c[:,:,1:],changed_c[:,:,1:])
        affected=float((actual[:,:,1:5]-changed_h[:,:,1:5]).abs().max());assert affected>0
        assert torch.equal(actual[:,:,5:],changed_h[:,:,5:])
        model.adapter(changed,h,mode='history')
        assert torch.equal(actual,model.adapter(f,h,mode='history'))
        paired_f=torch.cat((f,f+.17),0);paired_h=torch.cat((h,h-.07),0)
        paired=model.adapter(paired_f,paired_h,mode='history')
        reversed_rows=model.adapter(paired_f.flip(0),paired_h.flip(0),mode='history')
        torch.testing.assert_close(reversed_rows,paired.flip(0),atol=1e-6,rtol=1e-5)
        try:
            revision.bank.read_then_write(projected[:,:,0],state,window_id='another',step_index=40,align=align,fuse=fuse)
        except ValueError:pass
        else:raise AssertionError('Cross-window bank accepted')
    del projected,before,state,readouts,snapshots,actual,expected,chunked,changed_h,normal_c,changed_c,changed
    del paired_f,paired_h,paired,reversed_rows
    model.train()
    gradients={}
    for mode in ('history','current_only'):
        outputs=[]
        for chunk in (None,2):
            model.zero_grad(set_to_none=True)
            leaf=f[:,:,:5,:32,:32].detach().clone().requires_grad_(True)
            adapted=model.adapter(leaf,h[:,:,:5,:32,:32].detach(),mode=mode,temporal_chunk_size=chunk)
            loss=adapted[:,:,-1].square().mean()
            loss.backward()
            assert leaf.grad is not None and bool(torch.isfinite(leaf.grad).all())
            parameters={n:p.grad.detach().clone() for n,p in model.adapter.named_parameters() if p.grad is not None}
            outputs.append((adapted.detach(),leaf.grad.detach().clone(),parameters))
        torch.testing.assert_close(outputs[0][0],outputs[1][0],atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(outputs[0][1],outputs[1][1],atol=2e-6,rtol=2e-4)
        assert outputs[0][2].keys()==outputs[1][2].keys()
        for name in outputs[0][2]:torch.testing.assert_close(outputs[0][2][name],outputs[1][2][name],atol=2e-6,rtol=2e-4)
        early=float(outputs[0][1][:,:,0].double().norm())
        assert early>0 if mode=='history' else early==0
        gradients[mode]=dict(first_source_feature_gradient_l2=early,
            input_chunk_gradient_absmax=float((outputs[0][1]-outputs[1][1]).abs().max()),
            parameter_chunk_gradient_absmax=max(float((outputs[0][2][n]-outputs[1][2][n]).abs().max()) for n in outputs[0][2]))
        del outputs,leaf,adapted,loss,parameters
    assert rt.old().tensor_hash(model.state_dict())==initial
    rt.old().assert_frozen(model,parent['original_state_sha256'])
    model.zero_grad(set_to_none=True);model.eval()
    from trace import trace_window
    trace = trace_window(model,x,[[] for _ in range(40)],sequence='implementation_training_crop',start=0)
    with torch.no_grad():
        model.mode='history';normal_h=model(x)[1].cpu()
        model.mode='current_only';normal_c=model(x)[1].cpu()
        normal_off=model.base(x)[1].cpu()
    parity={mode:float((trace['logits'][mode]-value).abs().max())
            for mode,value in (('History',normal_h),('HistoryCurrent',normal_c),('HistoryOff',normal_off))}
    assert all(value==0 for value in parity.values()),parity
    assert trace['metadata']['pda_tsfm_calls_per_mode']=={'History':150,'HistoryCurrent':150,'HistoryOff':0}
    assert len(trace['slot_frame_rows'])==150
    assert trace['metadata']['maximum_identity_residual']['bank_read_mean']==0
    result=dict(status='PASS',native_core_vs_installed_adapter_exact=True,source_FIFO_history=records,
        all_cached_sources_equal_prewrite_projection=True,old_bank_entries_unchanged_after_later_native_reads=True,
        projected_inputs_not_modified=True,first_frame_read_equals_current_exact=True,pairs_in40=150,
        internal_temporal_chunk7_output_exact=True,window_isolation_exact=True,distinct_batch_row_permutation_close=True,
        history_dependency_before_eviction_absmax=affected,expired_feature_has_no_effect_after_t4_exact=True,
        current_only_future_unaffected_by_past_feature_change_exact=True,
        gradient_checks=gradients,custom_DCN_bitwise_backward_determinism_claimed=False,
        trace_standard_forward_logit_absmax=parity,trace_slot_rows=150,
        trace_source_order_read_identity_exact=True,
        gradient_atol=2e-6,gradient_rtol=2e-4,all_model_state_unchanged=True)
    del model,x,f,h,trace,normal_h,normal_c,normal_off
    cleanup()
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=rt.E/'checks/GPU_INTEGRATION.json')
    parser.add_argument('--phase',choices=('all','resources','semantics'),default='all')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    assert torch.cuda.is_available() and PAIRS==150
    prior=rt.old();prior.setup()
    report=dict(status='RUNNING',phase=args.phase,tested_source_sha256=tested_sources(),
        original_source_sha256=prior.source_hashes(),checkpoint_sha256=rt.sha(rt.CHECKPOINT),
        torch_version=str(torch.__version__),cuda_version=str(torch.version.cuda),device=torch.cuda.get_device_name(),
        preflight_only=True,optimized_models_disposable=True,spatial_M1_tiling=False,
        numerical_settings=dict(deterministic_requested=torch.are_deterministic_algorithms_enabled(),
            deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
            tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32),
        start_unix=time.time())
    rt.save(args.output,report)
    try:
        assert report['checkpoint_sha256']==rt.CHECKPOINT_SHA
        checkpoint=torch.load(rt.CHECKPOINT,map_location='cpu',weights_only=True)
        data=prior.frozen_f().data.inputs();frames,source=training_source(data);report['source']=source
        if args.phase in ('all','resources'):
            report['small_frontend_and_off']=small_frontend_and_off(frames);rt.save(args.output,report)
            report['fullframe']={}
            for condition in rt.CONDITIONS:
                report['fullframe'][condition]=fullframe(condition,frames);rt.save(args.output,report)
                print({'fullframe':condition,**report['fullframe'][condition]['complete_window']},flush=True)
            report['training']={}
            for condition in rt.CONDITIONS:
                report['training'][condition]=train_two_steps(condition,data,checkpoint);rt.save(args.output,report)
            a,b=[report['training'][name] for name in rt.CONDITIONS]
            assert a['initial_identity']==b['initial_identity']
            for left,right in zip(a['steps'],b['steps']):
                for key in (*HASH_KEYS,'complete_input_tensor_sha256'):assert left[key]==right[key]
            report['paired_initial_tensors_optimizer_and_actual_exposures_identical']=True
        if args.phase in ('all','semantics'):
            report['native_semantics']=native_bank_semantics(frames)
        assert rt.sha(rt.CHECKPOINT)==report['checkpoint_sha256']
        assert tested_sources()==report['tested_source_sha256']
        report.update(status='PASS',end_unix=time.time(),checkpoint_unchanged=True)
    except BaseException as error:
        report.update(status='FAIL',error=repr(error),traceback=traceback.format_exc(),end_unix=time.time())
        rt.save(args.output,report)
        raise
    rt.save(args.output,report)
    print({'status':report['status'],'output':str(args.output)},flush=True)


if __name__=='__main__':main()
