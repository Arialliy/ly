"""Read-only independent execution audit; no trainer import, inference, or updates.

The original release tensors, each saved optimizer/model state, and every logged
exposure/gradient/update are checked. A running or absent paired arm is pending,
never treated as a successful paired verification.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch

E=Path(__file__).resolve().parents[1]
B=E.parents[1]
R=B/'archive/original_project'
Q=R/'temporal_diagnosis_20261010'
HEAD=('conv_out1.0.weight','conv_out1.0.bias','conv_out2.weight','conv_out2.bias')
ADAPTER=('feature_refiner.scale','feature_refiner.operator.sigma','feature_refiner.operator.attention.weight')
NODES=(0,128,512,1024,2048)
RELEASE=R/'official_deeppro/log/sem_seg/RGB-T__2025-01-14_15-27__SoftLoUloss_DeepPro-Plus_DataL40/checkpoints/best_model.pth'
RELEASE_SHA='4fd4aa2736d9a33f2bf1459099068d8b4a7e202ff80aecdb4a3277fd13c7abcc'
S_INIT=B/'results/tlfp_pilot_20261012/init/S.pth'
S_INIT_SHA='ddea6e70af7578a2be85b037554a144c1155f4e6951a5fd5842c5b4016cf6f2a'
OLD_LOG=Q/'mutual_temporal_pilot_20261012/training/E0/run/TRAIN.jsonl'


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def tensor_hash(state):
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):
        digest.update(name.encode());digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def require(condition,message):
    if not condition:raise AssertionError(message)


def finite_tree(value,label):
    if isinstance(value,dict):
        for k,v in value.items():finite_tree(v,label+'.'+k)
    elif isinstance(value,list):
        for i,v in enumerate(value):finite_tree(v,label+f'[{i}]')
    elif isinstance(value,float):require(math.isfinite(value),'Nonfinite '+label)


def checkpoint_load(path,release=False):
    if release:
        core=np._core if hasattr(np,'_core') else np.core
        safe=[(core.multiarray.scalar,'numpy.core.multiarray.scalar'),np.dtype,type(np.dtype(np.float64))]
        with torch.serialization.safe_globals(safe):return torch.load(path,map_location='cpu',weights_only=True)
    return torch.load(path,map_location='cpu',weights_only=True)


def audit_arm(variant,release,spatial,old_rows):
    folder=E/'training'/variant/'run'
    if not (folder/'RUN.json').is_file():return {'status':'PENDING','reason':'No RUN.json'},None
    run=json.loads((folder/'RUN.json').read_text())
    if run['status']!='complete':return {'status':'PENDING','run_status':run['status'],'reason':'Arm not complete'},None
    expected_names=['original.'+n for n in HEAD]+(list(ADAPTER) if variant=='HL' else [])
    expected_lr={'original_prediction_convolutions':1e-5}
    if variant=='HL':expected_lr['spatial_lfp']=1e-4
    rows=[json.loads(line) for line in (folder/'TRAIN.jsonl').read_text().splitlines()]
    require(len(rows)==2048,f'{variant}: expected2048rows')
    require(run['final_step']==2048 and run['steps']==2048 and run['seed']==46,f'{variant}: fixedbudget')
    require(run['precision']=='FP32_no_AMP_no_TF32',f'{variant}: precision')
    require(run['head_lr']==1e-5 and run['adapter_lr']==1e-4 and run['weight_decay']==1e-4,f'{variant}: hyperparameters')
    grad_stats={name:{'steps_present':0,'nonzero_gradient_steps':0,'positive_update_steps':0,
                      'max_gradient_norm':0.0,'sum_step_update_norms':0.0} for name in expected_names}
    counts=Counter();input_chain=hashlib.sha256();label_chain=hashlib.sha256()
    for i,row in enumerate(rows):
        finite_tree(row,f'{variant}.step{i+1}')
        require(row['step']==i+1 and row['plan_entry_index']==i%512 and row['cycle']==i//512,f'{variant}: orderedcycle{i+1}')
        require(row['variant']==variant and row['condition']==variant,f'{variant}: rowcondition')
        require(row['input_sha256']==old_rows[i%512]['input_sha256'] and row['label_sha256']==old_rows[i%512]['label_sha256'],f'{variant}: exactexposure{i+1}')
        require(row['lr_by_group']==expected_lr,f'{variant}: LR{i+1}')
        require(row['bn_running_sha256']==run['bn_initial_sha256'],f'{variant}: BNlog{i+1}')
        require(set(row['parameter_gradients'])==set(expected_names) and set(row['parameter_updates'])==set(expected_names),f'{variant}: gradient/update whitelist{i+1}')
        require(row['parameters_with_gradient']==len(expected_names) and row['finite'],f'{variant}: finitegradient{i+1}')
        require(abs(row['loss']-sum(row['loss_per_window'])/4)<2e-6,f'{variant}: lossreduction{i+1}')
        require(abs(row['loss']-(row['positive_loss_weighted_contribution']+row['negative_loss_weighted_contribution']))<2e-6,f'{variant}: contribution{i+1}')
        for name in expected_names:
            gradient=row['parameter_gradients'][name];update=row['parameter_updates'][name]
            require(gradient['norm']>=0 and gradient['nonzero']>=0 and update['norm']>=0 and update['max_abs']>=0,f'{variant}: invalidnorm')
            stat=grad_stats[name];stat['steps_present']+=1
            stat['nonzero_gradient_steps']+=gradient['nonzero']>0
            stat['positive_update_steps']+=update['norm']>0
            stat['max_gradient_norm']=max(stat['max_gradient_norm'],gradient['norm'])
            stat['sum_step_update_norms']+=update['norm']
        for key in ('pure_negative_windows','valid_pixels','positive_pixels'):counts[key]+=row[key]
        input_chain.update(row['input_sha256'].encode());label_chain.update(row['label_sha256'].encode())
    require(dict(counts)==run['counts'],f'{variant}: aggregatecounts')
    require(input_chain.hexdigest()==run['cumulative_input_sha256'] and label_chain.hexdigest()==run['cumulative_label_sha256'],f'{variant}: chains')
    require(sha(folder/'TRAIN.jsonl')==run['train_log_sha256'],f'{variant}: logfilehash')
    require([v['step'] for v in run['snapshots']]==list(NODES),f'{variant}: checkpointschedule')
    expected_state={'original.'+n for n in release}|(set(ADAPTER) if variant=='HL' else set())
    snaps=[]; initial=None; final=None
    for node,record in zip(NODES,run['snapshots']):
        path=folder/f'snapshot_{node:03d}.pth'
        require(record['path']==path.name and sha(path)==record['sha256'],f'{variant}: snapshotreceipt{node}')
        cp=checkpoint_load(path);state=cp['model_state_dict']
        require(cp['step']==node and cp['variant']==variant and cp['condition']==variant,f'{variant}: cpidentity{node}')
        require(cp['initial_checkpoint_sha256']==RELEASE_SHA,f'{variant}: releasesource{node}')
        require(set(state)==expected_state,f'{variant}: statekeys{node}')
        require(all(torch.isfinite(t).all() for t in state.values()),f'{variant}: finitestate{node}')
        for name,tensor in release.items():
            if name not in HEAD or node==0:
                require(torch.equal(state['original.'+name],tensor),f'{variant}: frozenoriginaltensor{node}:{name}')
        if variant=='HL' and node==0:
            for name,tensor in spatial.items():
                require(torch.equal(state['feature_refiner.'+name],tensor),f'HL: initialadapter{name}')
        nonhead={name:state['original.'+name] for name in release if name not in HEAD}
        require(tensor_hash(nonhead)==run['original_nonhead_initial_sha256'],f'{variant}: nonheadhash{node}')
        require(tensor_hash(state)==record['model_state_sha256'],f'{variant}: statehash{node}')
        optimizer=cp['optimizer_state_dict'];groups=optimizer['param_groups']
        require([g['name'] for g in groups]==list(expected_lr),f'{variant}: optimizer groupnames{node}')
        flattened=[]
        for group in groups:
            require(group['lr']==expected_lr[group['name']] and group['weight_decay']==1e-4,f'{variant}: optimizerparams{node}')
            require(tuple(group['betas'])==(.9,.999) and group['eps']==1e-8,f'{variant}: Adamsettings{node}')
            flattened.extend(group['params'])
        require(len(flattened)==len(expected_names) and len(set(flattened))==len(flattened),f'{variant}: optimizercoverage{node}')
        require(len(groups[0]['params'])==4 and (variant=='H' or len(groups[1]['params'])==3),f'{variant}: groupsizes{node}')
        if node==0:require(not optimizer['state'],f'{variant}: freshAdam')
        else:
            require(set(optimizer['state'])==set(flattened),f'{variant}: optimizerstatecoverage{node}')
            for pid,name in zip(flattened,expected_names):
                opt=optimizer['state'][pid]
                require(int(opt['step'])==node,f'{variant}: Adam reset detected{node}:{name}')
                for slot in ('exp_avg','exp_avg_sq'):
                    require(opt[slot].shape==state[name].shape and torch.isfinite(opt[slot]).all(),f'{variant}: Adam moments{node}:{name}')
        if node==0:initial=state
        if node==2048:final=state
        snaps.append({'step':node,'sha256':sha(path),'nonhead_all_tensors_equal_release':True,
                      'model_state_sha256':tensor_hash(state),'all_Adam_step_counters':node})
    actual_trainable=sum(initial[name].numel() for name in expected_names)
    require(actual_trainable==(273 if variant=='H' else 373),f'{variant}: actualparamcount')
    difference={}
    for name in expected_names:
        delta=final[name].double()-initial[name].double()
        difference[name]={'numel':delta.numel(),'changed_elements':int(torch.count_nonzero(delta)),
                          'l2':float(delta.norm()),'max_abs':float(delta.abs().max())}
        require(abs(difference[name]['l2']-rows[-1]['parameter_updates'][name]['l2_from_initial'])<1e-5,f'{variant}: endpointupdateagainsttensor{name}')
        require(grad_stats[name]['steps_present']==2048,f'{variant}: loggedallgradsteps{name}')
    require(all(difference['original.'+name]['changed_elements']>0 for name in HEAD),f'{variant}: everyheadtensorupdated')
    if variant=='HL':require(all(difference[name]['changed_elements']>0 for name in ADAPTER),'HL: everyadaptertensorupdated')
    require(sha(folder/'final_model.pth')==sha(folder/'snapshot_2048.pth')==run['final_checkpoint_sha256'],f'{variant}: fixedfinalselection')
    return {'status':'PASS','rows':2048,'cycles':4,'unique_plan_entries':512,'not_2048_independent_examples':True,
            'input_sha256':input_chain.hexdigest(),'label_sha256':label_chain.hexdigest(),
            'actual_trainable_parameters':actual_trainable,'every_nonhead_original_tensor_frozen_at_all_saved_nodes':True,
            'all_2048_logged_BN_hashes_constant':True,'snapshots':snaps,'gradient_update_summary':grad_stats,
            'actual_endpoint_parameter_differences':difference,'logged_loss_contributions_consistent':True,
            'no_optimizer_reset_at_saved_cycle_boundaries':True,'counts':dict(counts),
            'duration_seconds':run['duration_seconds'],'log_sha256':sha(folder/'TRAIN.jsonl')},rows


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();started=time.perf_counter()
    result={'status':'RUNNING','audit_script_sha256':sha(__file__),'no_training_or_inference':True,'arms':{}}
    try:
        freeze=json.loads((E/'PROTOCOL_FREEZE.json').read_text())
        for rel,digest in freeze['files'].items():require(sha(E/rel)==digest,'Frozen file changed: '+rel)
        result['protocol_freeze_sha256']=sha(E/'PROTOCOL_FREEZE.json')
        require(sha(RELEASE)==RELEASE_SHA,'Release checkpoint fingerprint')
        require(sha(S_INIT)==S_INIT_SHA,'Spatial initialization fingerprint')
        release=checkpoint_load(RELEASE,release=True)['model_state_dict']
        spatial=checkpoint_load(S_INIT)['adapter_state_dict']
        actual_original_count=sum(v.numel() for k,v in release.items() if not k.endswith(('running_mean','running_var','num_batches_tracked')))
        require(actual_original_count==70913,'Actual original parameter count')
        old_rows=[json.loads(line) for line in OLD_LOG.read_text().splitlines()]
        require(len(old_rows)==512,'Historical exposure rows')
        result['inputs']={'release_sha256':sha(RELEASE),'initial_spatial_sha256':sha(S_INIT),
                          'historical_E0_log_sha256':sha(OLD_LOG),'actual_original_parameters':actual_original_count}
        arm_rows={}
        for variant in ('H','HL'):
            result['arms'][variant],arm_rows[variant]=audit_arm(variant,release,spatial,old_rows)
        if all(arm_rows.values()):
            require(len(arm_rows['H'])==len(arm_rows['HL'])==2048,'Pair rowcount')
            for i,(h,hl) in enumerate(zip(arm_rows['H'],arm_rows['HL'])):
                for key in ('input_sha256','label_sha256','label_stats','plan_entry_index','cycle','positive_pixels_per_window','valid_pixels_per_window','background_pixels_per_window'):
                    require(h[key]==hl[key],f'Paired exposure differs step{i+1}:{key}')
            result['paired_exact_exposure']={'status':'PASS','steps_compared':2048,'inputs_labels_and_label_support_equal':True}
            result['status']='PASS'
        else:
            result['paired_exact_exposure']={'status':'PENDING','reason':'Both arms must finish'}
            result['status']='PARTIAL_PASS'
        result['scope']='Execution integrity only. Does not establish target recovery, background-budget compliance, or independent generalization.'
    except BaseException as error:
        result.update(status='FAILED',error=repr(error));raise
    finally:
        result['seconds']=time.perf_counter()-started
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'status':result['status'],'arms':{k:v['status'] for k,v in result['arms'].items()},'output':str(args.output)}))


if __name__=='__main__':main()
