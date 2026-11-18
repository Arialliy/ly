#!/usr/bin/env python3
"""Pre-registered H/HL adaptation; exact existing 512-entry plan repeated four times."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

E = Path(__file__).resolve().parents[1]
BUNDLE = E.parents[1]
Q = Path('/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/temporal_diagnosis_20261010')
OLD_CODE = BUNDLE / 'sources/temporal_diagnosis_20261010/tlfp_pilot_20261012/code'
sys.path.insert(0, str(OLD_CODE))
import train_tlfp as historical
from temporal_lfp import TemporalLFPAdapter, FramewiseFeatureAdapter, SpatialLFPReference
from joint_adaptation import HeadLFPAdaptation, HEAD_NAMES

previous = historical.previous
sha, save_json, load_module = historical.sha, historical.save_json, historical.load_module
tensor_hash, data_hash, label_hash = historical.tensor_hash, historical.data_hash, historical.label_hash
EXPECTED, LABEL_CONFIG = historical.EXPECTED, historical.LABEL_CONFIG
build_targets, common_loss = historical.build_targets, historical.common_loss
LABEL_PATH, TOP1_PATH = historical.LABEL_PATH, historical.TOP1_PATH
EFFECTIVE_SUPERVISION = historical.EFFECTIVE_SUPERVISION
OLD_LOG = historical.OLD_LOG
deterministic_setup = historical.deterministic_setup
SNAPSHOTS = (0, 128, 512, 1024, 2048)
STEPS = 2048
VARIANTS = ('H', 'HL')
INIT_PATH = BUNDLE / 'results/tlfp_pilot_20261012/init/S.pth'
INIT_SHA = 'ddea6e70af7578a2be85b037554a144c1155f4e6951a5fd5842c5b4016cf6f2a'


def wrap_model(original, variant):
    import torch
    if variant not in VARIANTS:
        raise ValueError(variant)
    adapter = None
    if variant == 'HL':
        if sha(INIT_PATH) != INIT_SHA:
            raise AssertionError('Untrained spatial initialization changed')
        spatial = FramewiseFeatureAdapter(SpatialLFPReference(channels=32), channels=32, frame_chunk=8)
        adapter = TemporalLFPAdapter.from_spatial(spatial, mode='spatial', frame_chunk=8)
        adapter.load_state_dict(torch.load(INIT_PATH, map_location='cpu', weights_only=True)['adapter_state_dict'], strict=True)
        if adapter.scale.item() != 0:
            raise AssertionError('Initialization must be identity, never prior trained weights')
    model = HeadLFPAdaptation(original, adapter, variant)
    model.config = {'condition': variant, 'variant': variant, 'mode': 'spatial' if adapter is not None else 'head_only',
                    'trainable_original_names': sorted(HEAD_NAMES), 'all_bn_affine_and_buffers_frozen': True,
                    'frame_chunk': 8, 'init_file_sha256': INIT_SHA if adapter is not None else None,
                    'adapter_source': str(OLD_CODE / 'temporal_lfp.py'), 'original_nonhead_frozen': True,
                    'TPro_autograd_enabled': True, 'model_labels_access': False}
    return model


def source_hashes(model_path):
    paths = {
        'train_source_sha256': Path(__file__), 'module_source_sha256': Path(__file__).with_name('joint_adaptation.py'),
        'model_source_sha256': model_path, 'lfp_source_sha256': OLD_CODE/'temporal_lfp.py',
        'historical_train_source_sha256': OLD_CODE/'train_tlfp.py', 'label_source_sha256': LABEL_PATH,
        'top1_loss_source_sha256': TOP1_PATH, 'common_loss_source_sha256': Path(historical.historical.__file__),
        'prior_train_source_sha256': Q/'supervision_alignment_control_20261012/code/train_supervision.py',
        'spatial_reference_sha256': OLD_CODE.parents[1]/'route_closeout_20261012/haar_reference.py',
        'adapter_reference_sha256': OLD_CODE.parents[1]/'route_closeout_20261012/framewise_adapter.py',
        'batch_source_sha256': Q/'top1_background_control_20261011/code/train_pilot.py',
    }
    return {name: sha(path) for name,path in paths.items()}


def frozen_original_state(model):
    return {name: value for name,value in model.original.state_dict().items() if name not in HEAD_NAMES}


def assert_frozen(model, reference):
    import torch
    current = frozen_original_state(model)
    if set(current) != set(reference) or any(not torch.equal(current[k].detach().cpu(), v) for k,v in reference.items()):
        raise AssertionError('A non-head original parameter or buffer changed')
    for name,p in model.original.named_parameters():
        if p.requires_grad != (name in HEAD_NAMES) or (name not in HEAD_NAMES and p.grad is not None):
            raise AssertionError('Original trainable/gradient whitelist changed: '+name)
    if model.original.training or any(m.training for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)):
        raise AssertionError('Original eval / BN freeze policy changed')
    return tensor_hash(current.items())


def train(args):
    import numpy as np
    import torch
    base,plan,names,sequences,model_file = previous.validate_inputs(args)
    old_rows = [json.loads(line) for line in OLD_LOG.read_text().splitlines()]
    if len(old_rows) != 512:
        raise AssertionError('Historical exposure comparison requires all 512 rows')
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Formal output must be new and empty')
    args.output.mkdir(parents=True, exist_ok=True)
    deterministic_setup()
    original,loaded = base.load_model(args.source,args.checkpoint)
    if loaded.resolve() != model_file.resolve():
        raise AssertionError('Wrong original model source')
    model = wrap_model(original,args.variant).to(device=torch.device(args.device),dtype=torch.float32).train()
    device = torch.device(args.device)
    criterion = load_module('joint_official_loss',model_file).SoftLoUloss().to(device)
    initial = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    frozen = {k:v.detach().cpu().clone() for k,v in frozen_original_state(model).items()}
    frozen_hash = assert_frozen(model,frozen)
    bn_initial = base.bn_snapshot(model.original)
    bn_hash = base.buffer_hash(bn_initial)
    sources = source_hashes(model_file)
    optimizer = torch.optim.Adam(model.parameter_groups(args.head_lr,args.adapter_lr),weight_decay=1e-4)
    trainable = {k:p for k,p in model.named_parameters() if p.requires_grad}
    expected_count = 273 if args.variant == 'H' else 373
    if sum(p.numel() for p in trainable.values()) != expected_count:
        raise AssertionError('Wrong trainable parameter count')
    metadata = {
        'status':'running','condition':args.variant,'variant':args.variant,'steps':STEPS,'seed':46,
        'head_lr':args.head_lr,'adapter_lr':args.adapter_lr,'weight_decay':1e-4,'optimizer':'fresh Adam',
        'betas':[.9,.999],'eps':1e-8,'precision':'FP32_no_AMP_no_TF32','torch_version':str(torch.__version__),
        'bn_policy':'all_original_BN_affine_and_running_state_frozen_eval','bn_initial_sha256':bn_hash,
        'checkpoint_sha256':EXPECTED['checkpoint'],'initial_checkpoint_sha256':EXPECTED['checkpoint'],
        'reference_checkpoint_sha256':EXPECTED['checkpoint'],'plan_sha256':EXPECTED['plan'],
        'selection_sha256':EXPECTED['selection'],'old_log_sha256':sha(OLD_LOG),
        'exposure_rule':'existing annotation-only 512-entry plan repeated four times, same order; no optimizer or RNG reset at cycle boundaries',
        'historical_exposure_verification':'all 2048 input and label hashes checked against corresponding historical E0 entry modulo512',
        'initial_model_state_sha256':tensor_hash(initial.items()),'original_nonhead_initial_sha256':frozen_hash,
        'original_initial_state_sha256':tensor_hash(model.original.state_dict().items()),
        'branch_initial_sha256': None if model.feature_refiner is None else tensor_hash(model.feature_refiner.state_dict().items()),
        'label_config':LABEL_CONFIG,'effective_supervision_config':EFFECTIVE_SUPERVISION,
        'frozen_plan_config':base.CONFIG,'model_config':model.config,'training_sequences':names,'device':str(device),
        'parameter_count':sum(p.numel() for p in model.parameters()),'new_parameters':0 if args.variant=='H' else 100,
        'trainable_parameters':expected_count,'trainable_parameter_names':list(trainable),
        'source_path':str(args.source.resolve()),'checkpoint_path':str(args.checkpoint.resolve()),
        'criterion_source_path':str(model_file.resolve()),'label_source_path':str(LABEL_PATH),
        'protocol':'fixed_original_head_optional_spatial_LFP_positive_masked_official_SoftIoU_empty_old_top1pct',
        'loss_reduction':'unchanged official per positive window; empty old top1 negative; four-window equal mean',
        'checkpoint_selection':'fixed final2048; snapshots0/128/512/1024/2048; no validation selection',
        'snapshots':[], 'telemetry_rule':'every step: per-name trainable gradient/update and unchanged positive/empty loss; LFP feature stats first2/every16',
        'model_labels_access':False, **sources,
    }
    save_json(args.output/'INITIAL_STATE.json',{
        'variant':args.variant,'model_state_sha256':tensor_hash(initial.items()),
        'state_tensor_sha256':{k:tensor_hash([(k,v)]) for k,v in initial.items()},
        'trainable_parameter_names':list(trainable),'trainable_parameter_count':expected_count,
        'original_checkpoint_sha256':EXPECTED['checkpoint'],'original_nonhead_sha256':frozen_hash,
    })

    def snapshot(step):
        nonhead_hash = assert_frozen(model,frozen)
        checkpoint = {'model_state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                      'optimizer_state_dict':optimizer.state_dict(),'step':step,'variant':args.variant,
                      'condition':args.variant,'mode':model.config['mode'],'model_config':model.config,
                      'label_config':LABEL_CONFIG,'effective_supervision_config':EFFECTIVE_SUPERVISION,
                      'initial_checkpoint_sha256':EXPECTED['checkpoint'],'plan_sha256':EXPECTED['plan'],
                      'selection_sha256':EXPECTED['selection'],'original_nonhead_state_sha256':nonhead_hash,
                      'original_state_sha256':tensor_hash(model.original.state_dict().items()),
                      'protocol':metadata['protocol'],**sources}
        path=args.output/f'snapshot_{step:03d}.pth'
        torch.save(checkpoint,path)
        metadata['snapshots'].append({'step':step,'path':path.name,'sha256':sha(path),
                                     'original_nonhead_state_sha256':nonhead_hash,
                                     'model_state_sha256':tensor_hash(checkpoint['model_state_dict'].items())})
        save_json(args.output/'RUN.json',metadata)

    snapshot(0)
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
    started=time.perf_counter()
    counts=Counter(); input_chain=hashlib.sha256(); label_chain=hashlib.sha256()
    try:
        with (args.output/'TRAIN.jsonl').open('w',buffering=1) as log:
            for index in range(STEPS):
                plan_index=index%512; entry=plan['entries'][plan_index]
                if entry['step']!=plan_index: raise AssertionError('Frozen entry order changed')
                batch,positives,masks,specs=base.build_batch(entry,sequences)
                target_np,valid_np,label_stats=build_targets(entry,sequences,background_masks=masks)
                input_sha,labels_sha=data_hash(batch,positives,masks,specs),label_hash(target_np,valid_np)
                if input_sha!=old_rows[plan_index]['input_sha256'] or labels_sha!=old_rows[plan_index]['label_sha256']:
                    raise AssertionError(f'Exposure/label mismatch at step{index+1}')
                input_chain.update(input_sha.encode());label_chain.update(labels_sha.encode())
                x=torch.from_numpy(batch).to(device);targets=torch.from_numpy(target_np).to(device)
                valid=torch.from_numpy(valid_np).to(device);background=torch.from_numpy(masks).to(device)
                optimizer.zero_grad(set_to_none=True)
                logged=index<2 or (index+1)%16==0
                if model.feature_refiner is not None: model.feature_refiner.collect_stats=logged
                before={k:p.detach().clone() for k,p in trainable.items()}
                logits=model(x)[1]
                if logits.shape!=(4,40,128,128):raise AssertionError('Wrong output dimensions')
                loss,parts=common_loss(logits,targets,valid,background,criterion)
                if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite loss')
                loss.backward()
                gradients={k:p.grad for k,p in trainable.items()}
                if any(g is None or not bool(torch.isfinite(g).all()) for g in gradients.values()):
                    raise FloatingPointError('Missing or nonfinite trainable gradient')
                assert_frozen(model,frozen)
                row={'step':index+1,'plan_entry_index':plan_index,'cycle':index//512,
                     'variant':args.variant,'condition':args.variant,'loss':float(loss.detach()),
                     'grad_norm':float(torch.linalg.vector_norm(torch.stack([g.detach().norm() for g in gradients.values()]))),
                     'finite':True,'parameters_with_gradient':len(gradients),
                     'input_sha256':input_sha,'label_sha256':labels_sha,'label_stats':label_stats,**parts,
                     'parameter_gradients':{k:{'norm':float(g.detach().norm()),'nonzero':int(torch.count_nonzero(g)),
                                              'parameter_norm':float(trainable[k].detach().norm())} for k,g in gradients.items()},
                     'lr_by_group':{g['name']:g['lr'] for g in optimizer.param_groups}}
                positive=parts['positive_pixels_per_window']
                row['positive_loss_unweighted_mean']=sum(v for v,n in zip(parts['loss_per_window'],positive) if n)/max(1,sum(n>0 for n in positive))
                row['negative_loss_unweighted_mean']=sum(parts['empty_window_negative_per_window'])/max(1,parts['pure_negative_windows'])
                row['positive_loss_weighted_contribution']=sum(v for v,n in zip(parts['loss_per_window'],positive) if n)/4
                row['negative_loss_weighted_contribution']=sum(parts['empty_window_negative_per_window'])/4
                if logged and model.feature_refiner is not None:
                    row['module_state_before_update']=model.feature_refiner.parameter_stats()
                    row['feature_statistics']=model.feature_refiner.latest_stats
                optimizer.step()
                row['parameter_updates']={k:{'norm':float((p.detach()-before[k]).norm()),
                                             'max_abs':float((p.detach()-before[k]).abs().max()),
                                             'l2_from_initial':float((p.detach()-initial[k].to(device)).norm())} for k,p in trainable.items()}
                row['bn_running_sha256']=base.buffer_hash(base.bn_snapshot(model.original))
                if row['bn_running_sha256']!=bn_hash:raise AssertionError('BN changed')
                for key in ('pure_negative_windows','valid_pixels','positive_pixels'):counts[key]+=parts[key]
                row['elapsed_seconds']=time.perf_counter()-started
                row['max_cuda_memory_allocated_bytes']=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0
                row['max_cuda_memory_reserved_bytes']=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else 0
                log.write(json.dumps(row,allow_nan=False)+'\n')
                if index%64==0 or index+1 in SNAPSHOTS:
                    print(json.dumps({k:row[k] for k in ['step','variant','loss','grad_norm','elapsed_seconds','max_cuda_memory_allocated_bytes']},allow_nan=False),flush=True)
                if index+1 in SNAPSHOTS:snapshot(index+1)
                del x,targets,valid,background,logits,loss,gradients,before
        if device.type=='cuda':torch.cuda.synchronize(device)
        if source_hashes(model_file)!=sources:raise AssertionError('Source changed during training')
        final_frozen=assert_frozen(model,frozen)
        final={k:v.detach().cpu() for k,v in model.state_dict().items()}
        if set(final)!=set(initial) or not all(bool(torch.isfinite(v).all()) for v in final.values()):
            raise AssertionError('Invalid final state')
        differences={}
        for name,p in model.named_parameters():
            delta=final[name].double()-initial[name].double()
            differences[name]={'numel':delta.numel(),'trainable':p.requires_grad,
                               'changed_elements':int(torch.count_nonzero(delta)),
                               'max_absolute_difference':float(delta.abs().max()),'l2_difference':float(delta.norm())}
        save_json(args.output/'PARAMETER_DIFF.json',{'per_parameter':differences,
                    'parameter_count':metadata['parameter_count'],'new_parameter_count':metadata['new_parameters'],
                    'trainable_parameter_count':expected_count,
                    'changed_parameter_tensors':sum(v['changed_elements']>0 for v in differences.values())})
        os.link(args.output/'snapshot_2048.pth',args.output/'final_model.pth')
        metadata.update(status='complete',final_step=STEPS,duration_seconds=time.perf_counter()-started,
                        counts=dict(counts),bn_final_sha256=base.buffer_hash(base.bn_snapshot(model.original)),
                        original_nonhead_final_sha256=final_frozen,frozen_original_nonhead_unchanged=True,
                        final_checkpoint_sha256=sha(args.output/'final_model.pth'),final_model_state_sha256=tensor_hash(final.items()),
                        cumulative_input_sha256=input_chain.hexdigest(),cumulative_label_sha256=label_chain.hexdigest(),
                        train_log_sha256=sha(args.output/'TRAIN.jsonl'),parameter_diff_sha256=sha(args.output/'PARAMETER_DIFF.json'),
                        max_cuda_memory_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0,
                        max_cuda_memory_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else 0)
        save_json(args.output/'RUN.json',metadata)
        print(json.dumps({'status':'complete','variant':args.variant,'seconds':metadata['duration_seconds']}),flush=True)
    except BaseException as error:
        metadata.update(status='failed',error=repr(error),duration_seconds=time.perf_counter()-started)
        save_json(args.output/'RUN.json',metadata)
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','checkpoint','dataset-root','selection','plan','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--variant',choices=VARIANTS,required=True)
    p.add_argument('--steps',type=int,default=2048);p.add_argument('--seed',type=int,default=46)
    p.add_argument('--head-lr',type=float,default=1e-5);p.add_argument('--adapter-lr',type=float,default=1e-4)
    p.add_argument('--device',default='cuda:0')
    args=p.parse_args()
    if args.steps!=2048 or args.seed!=46 or args.head_lr!=1e-5 or args.adapter_lr!=1e-4:
        p.error('Frozen steps2048 seed46 head-lr1e-5 adapter-lr1e-4 required')
    train(args)


if __name__=='__main__':main()
