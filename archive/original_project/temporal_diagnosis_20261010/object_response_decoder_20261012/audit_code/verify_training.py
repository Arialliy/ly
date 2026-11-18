#!/usr/bin/env python3
"""Read-only independent integrity check; final receipt requires all three runs.

This verifies data/loss accounting, state tensors, snapshots, and protocol hashes.
It does not infer scientific effectiveness or rerun training/inference.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

N=Path(__file__).resolve().parents[1]
Q=N.parent
R=Q.parent
TERMS=('existence_positive','existence_negative','mask_positive','mask_instance_negative','mask_background_negative')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def digest_tensors(items):
    h=hashlib.sha256()
    for name,tensor in sorted(items):
        h.update(name.encode());h.update(str(tensor.dtype).encode());h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m


def check(value,label):
    if not value:raise AssertionError(label)


def snapshots(directory,meta,original,initial,parameter_names,partial):
    reports=[]
    listed={v['step']:v for v in meta['snapshots']}
    if not partial:check(set(listed)=={0,512,1024,2048},'exact declared snapshots')
    for step,receipt in sorted(listed.items()):
        path=directory/receipt['path'];check(sha(path)==receipt['sha256'],f'{path} hash')
        cp=torch.load(path,map_location='cpu',weights_only=True);state=cp['model_state_dict']
        check(cp['step']==step,f'{path} actual step')
        check(set(state)==set(initial),f'{path} state keys')
        for key,val in original.items():check(torch.equal(state['original.'+key],val),f'{path}: released original tensor {key}')
        for key,val in state.items():
            check(val.dtype==initial[key].dtype and val.shape==initial[key].shape,f'{path}: tensor schema {key}')
            if val.dtype.is_floating_point:check(bool(torch.isfinite(val).all()),f'{path}: finite {key}')
        check(torch.equal(state['anchors_yx'],initial['anchors_yx']),f'{path}: anchors changed')
        check(cp['original_state_sha256']==digest_tensors(original.items()),f'{path}: original hash')
        check(cp['plan_sha256']==meta['plan_sha256'],f'{path}: plan')
        opt=cp['optimizer_state_dict'];groups=opt['param_groups']
        check(len(groups)==1,'one optimizer group')
        group=groups[0]
        check(group['lr']==1e-4 and group['weight_decay']==1e-4 and tuple(group['betas'])==(.9,.999) and group['eps']==1e-8,'Adam recipe')
        check(len(group['params'])==len(parameter_names),'optimizer only new parameters')
        new_names={name for name in parameter_names if not name.startswith('original.')}
        temporal_names={name for name in new_names if '.temporal_' in name}
        frozen_temporal=[]
        if meta['condition']=='R0':
            for name in temporal_names:
                check(torch.equal(state[name],initial[name]),f'R0 bypassed temporal parameter changed {name}')
                frozen_temporal.append(name)
        if step==0:
            check(not opt['state'],'fresh zero-step Adam state')
            check(all(torch.equal(state[k],initial[k]) for k in state),'snapshot0 vs INITIAL_STATE')
        else:
            expected_active=new_names-(temporal_names if meta['condition']=='R0' else set())
            id_to_name=dict(zip(group['params'],parameter_names))
            actual_active={id_to_name[i] for i in opt['state']}
            check(actual_active==expected_active,'Adam updated exactly active branch parameters')
            for st in opt['state'].values():check(float(st['step'])==step,'each active parameter has fixed update count')
        reports.append(dict(step=step,sha256=receipt['sha256'],original_bitwise_equal=True,optimizer_updates=step,optimizer_state_entries=len(opt['state']),r0_bypassed_temporal_tensors=len(frozen_temporal)))
    return reports


def main(partial=False):
    freeze=read(N/'PROTOCOL_FREEZE.json')
    if 'resource_amendment' in freeze:
        amendment=freeze['resource_amendment'];initial_freeze=read(N/'PROTOCOL_FREEZE_INITIAL.json')
        check(sha(N/'PROTOCOL_FREEZE_INITIAL.json')==amendment['previous_freeze_sha256'],'resource amendment predecessor')
        check(amendment['allowed_gpu_ids']==[1,2,3] and amendment['training_model_loss_data_and_2048_endpoint_unchanged'],'resource-only change')
        for key,value in initial_freeze.items():
            if key=='code_sha256':
                for path,digest in value.items():
                    if path!='code/launch_gpu.py':check(freeze[key][path]==digest,'no algorithm code hash changed')
            else:check(freeze[key]==value,'resource amendment changes no scientific protocol')
    for path,value in freeze['code_sha256'].items():check(sha(N/path)==value,'frozen code '+path)
    for field,path in [('protocol_sha256','PROTOCOL.md'),('plan_sha256','TRAIN_PLAN.json'),('label_design_sha256','LOSS_DESIGN.md'),('target_manifest_sha256','data_audit/frozen_target_receipts/MANIFEST.json')]:
        check(sha(N/path)==freeze[field],'frozen '+path)
    newplan=read(N/'TRAIN_PLAN.json');oldplan=read(Q/'stage1_goal/pilot/TRAIN_PLAN.json')
    check(newplan['steps']==2048 and newplan['seed']==46 and newplan['cycles']==4,'new plan constants')
    check(sha(Q/'stage1_goal/pilot/TRAIN_PLAN.json')==newplan['source_plan_sha256'],'old plan unchanged')
    check(newplan['entries']==[dict(step=i+1,cycle=i//512,source_entry=i%512) for i in range(2048)],'four exact cycles')
    manifest=read(N/'data_audit/frozen_target_receipts/MANIFEST.json')
    check(manifest['status']=='PASS' and manifest['legacy_background_valid_and_union_exact'],'target geometry gate')
    windows=manifest['windows'];check(len(windows)==2048,'2048 target exposure receipts')
    per_entry={i:[] for i in range(512)}
    for r in windows:per_entry[r['step']].append(r)
    target_hashes=[]
    for step,rr in per_entry.items():
        rr.sort(key=lambda v:v['window']);check([r['window'] for r in rr]==[0,1,2,3],'target window order')
        specs=[dict(sequence=p['sequence'],start=s,crop_yx=p['crop_yx']) for p in oldplan['entries'][step]['pairs'] for s in p['starts']]
        for r,spec in zip(rr,specs):check(all(r[k]==v for k,v in spec.items()),'target receipt exposure identity')
        target_hashes.append(hashlib.sha256(''.join(r['semantic_sha256'] for r in rr).encode()).hexdigest())
    oldlog=[json.loads(x) for x in (Q/'mutual_temporal_pilot_20261012/training/E0/run/TRAIN.jsonl').read_text().splitlines()]
    check(len(oldlog)==512,'historical 512 input receipts')

    # Bind immutable reused helpers and original files before loading trusted checkpoint.
    previous=module('training_audit_legacy',Q/'supervision_alignment_control_20261012/code/train_supervision.py')
    original_cp=R/'official_deeppro/log/sem_seg/RGB-T__2025-01-14_15-27__SoftLoUloss_DeepPro-Plus_DataL40/checkpoints/best_model.pth'
    check(sha(original_cp)==previous.EXPECTED['checkpoint'],'released checkpoint unchanged')
    check(sha(original_cp.parent.parent/'DeepPro-Plus.py')==previous.EXPECTED['model_source'],'released source unchanged')
    check(sha(Q/'stage1_goal/DATA_SELECTION.json')==previous.EXPECTED['selection'],'selection unchanged')
    check(sha(Q/'top1_background_control_20261011/code/train_pilot.py')==previous.EXPECTED['original_train_source'],'original input helper unchanged')
    npcore=np._core if hasattr(np,'_core') else np.core
    with torch.serialization.safe_globals([(npcore.multiarray.scalar,'numpy.core.multiarray.scalar'),np.dtype,type(np.dtype(np.float64))]):
        published=torch.load(original_cp,map_location='cpu',weights_only=True)
    original=published['model_state_dict']
    check(digest_tensors(original.items()),'original state readable')
    for s,fp in oldplan['data_fingerprints'].items():
        directory=Q/'stage1_goal/data'/s
        check(sha(directory/'annotations.json')==fp['annotations_sha256'],s+' original annotations unchanged')
        check(sha(directory/'raw_frames.npy')==fp['raw_sha256'],s+' raw frames unchanged')

    metas={v:read(N/'training'/v/'run/RUN.json') for v in ('R0','R1','R2') if (N/'training'/v/'run/RUN.json').exists()}
    if not partial:check(len(metas)==3 and all(m['status']=='complete' for m in metas.values()),'Final audit requires all three completed runs')
    result=dict(status='PARTIAL_PASS' if partial else 'PASS',scope='artifact integrity; no performance claims; exact state and exposure checks',source_freeze_sha256=sha(N/'PROTOCOL_FREEZE.json'),conditions={})
    initial_states=[];initial_hashes=[];inputchains=[];labelchains=[]
    for variant,meta in metas.items():
        directory=N/'training'/variant/'run'
        check(meta['condition']==variant and meta['steps']==2048 and meta['seed']==46,'run identity')
        check(meta['new_parameters']==31194 and meta['parameter_count']==102107,'parameter counts')
        check(meta['microbatches']==4 and meta['repeat_source_plan']==4 and meta['loss_weights']==[1,1,1,1,1],'loss/exposure recipe')
        check(meta['assignment']==('track' if variant=='R2' else 'frame'),'assignment mode')
        check(meta['temporal_enabled']==(variant!='R0'),'temporal switch')
        check(meta['plan_sha256']==freeze['plan_sha256'],'run uses frozen new plan')
        check(meta['bn_policy']=='entire_original_frozen_eval','BN policy')
        check(meta['precision']=='FP32_no_AMP_no_TF32','precision')
        for path,value in meta['source_sha256'].items():check(sha(path)==value,'run source receipt')
        initreceipt=read(directory/'INITIAL_STATE.json')
        initial=torch.load(directory/'snapshot_0000.pth',map_location='cpu',weights_only=True)['model_state_dict']
        initial_states.append(initial);initial_hashes.append(digest_tensors(initial.items()))
        check(initial_hashes[-1]==meta['initial_model_state_sha256']==initreceipt['model_state_sha256'],'complete initial state hash')
        check(meta['original_initial_state_sha256']==initreceipt['original_state_sha256']==digest_tensors(original.items()),'released state initialization')
        for key,val in initial.items():check(digest_tensors([(key,val)])==initreceipt['tensor_sha256'][key],'initial per-tensor hash')
        names=initreceipt['trainable_names'];check(all(not n.startswith('original.') for n in names),'only new tensors trainable')
        check(sum(initial[n].numel() for n in names)==31194,'actual new parameter count')
        snaps=snapshots(directory,meta,original,initial,names,partial)
        # Ignore an incomplete last line only during a running read-only partial audit.
        lines=(directory/'TRAIN.jsonl').read_text().splitlines();rows=[]
        for index,line in enumerate(lines):
            try:rows.append(json.loads(line))
            except json.JSONDecodeError:
                if partial and index==len(lines)-1:break
                raise
        if not partial:check(len(rows)==2048,'exact final update count')
        check(len(rows)<=2048,'training exceeded budget')
        totals=Counter();ih=hashlib.sha256();lh=hashlib.sha256()
        for i,row in enumerate(rows):
            check(row['step']==i+1 and row['source_entry']==i%512 and row['cycle']==i//512 and row['condition']==variant,'log exposure order')
            check(row['input_sha256']==oldlog[i%512]['input_sha256'],'exact historical input '+str(i))
            check(row['label_sha256']==target_hashes[i%512],'independent manifest label '+str(i))
            ih.update(row['input_sha256'].encode());lh.update(row['label_sha256'].encode())
            check(row['finite'] and row['original_parameters_with_gradient']==0,'finite/frozen gradients')
            check(math.isfinite(row['loss']) and math.isfinite(row['grad_norm']) and row['grad_norm']>0,'nonzero finite branch gradient')
            check(len(row['parts'])==4,'four equal microbatches')
            check(math.isclose(row['loss'],sum(p['loss'] for p in row['parts'])/4,rel_tol=3e-6,abs_tol=3e-6),'outer four-window reduction')
            for part,receipt in zip(row['parts'],per_entry[i%512]):
                st=receipt['stats'];j=st['tracks'];e=st['negative_eligible_frames'];n=st.get('valid_object_frames',0)
                check(part['assignment']==meta['assignment'] and part['queries']==100 and part['tracks']==j,'part config')
                check(part['object_frames']==n and part['cost_shape']==[n,100],'all valid objects assigned')
                check(part['negative_eligible_frames']==e,'shared negative eligibility')
                check(part['existence_negative_query_frames']==e*(100-j),'shared existence negative count')
                check(part['background_negative_query_frames']==e*(100-j),'shared background negative count')
                check(part['instance_negative_object_frames']==e*j,'shared instance negative count')
                check(part['empty_background_frames']==st['empty_background_frames'],'empty background accounting')
                check(math.isclose(part['loss'],sum(part[t] for t in TERMS),rel_tol=1e-12,abs_tol=1e-12),'five separate unit-weight loss terms')
                check(all(math.isfinite(part[t]) and part[t]>=-1e-6 for t in TERMS),'finite nonnegative individual terms')
                totals.update(dict(object_frames=n,negative_eligible_frames=e,existence_negative_query_frames=e*(100-j),instance_negative_object_frames=e*j,window_exposures=1))
            if 'gradient_norms' in row:
                gr=row['gradient_norms'];check(set(gr)==set(names),'gradient telemetry names')
                for name,value in gr.items():
                    if variant=='R0' and '.temporal_' in name:check(value is None,'bypassed R0 temporal gradient absent')
                    else:check(value is not None and math.isfinite(value),'active branch receives gradient tensor')
        inputchains.append(ih.hexdigest());labelchains.append(lh.hexdigest())
        if not partial:
            check(meta['final_step']==2048 and meta['frozen_original_unchanged'],'completion state')
            check(meta['cumulative_input_sha256']==ih.hexdigest() and meta['cumulative_label_sha256']==lh.hexdigest(),'chain receipts')
            check(meta['train_log_sha256']==sha(directory/'TRAIN.jsonl'),'complete log hash')
            check(sha(directory/'final_model.pth')==meta['final_checkpoint_sha256']==sha(directory/'snapshot_2048.pth'),'fixed final checkpoint')
            check((directory/'final_model.pth').stat().st_ino==(directory/'snapshot_2048.pth').stat().st_ino,'final hardlink')
            final=torch.load(directory/'final_model.pth',map_location='cpu',weights_only=True)['model_state_dict']
            check(digest_tensors(final.items())==meta['final_model_state_sha256'],'final full tensor hash')
            diffs=read(directory/'PARAMETER_DIFF.json')
            check(sum(initial[name].numel() for name in diffs if name.startswith('original.'))==70913,'actual original parameter count')
            check(sum(initial[name].numel() for name in diffs if not name.startswith('original.'))==31194,'actual new parameter count from named parameter receipt')
            for name,diff in diffs.items():
                delta=final[name]-initial[name]
                check(int(torch.count_nonzero(delta))==diff['changed'] and float(delta.abs().max())==diff['max_abs'],'parameter difference receipt')
                if name.startswith('original.'):check(diff['changed']==0,'all original parameters unchanged')
            check(totals['object_frames']==manifest['totals_four_cycles']['valid_object_frames'],'four cycles object count')
            check(totals['negative_eligible_frames']==manifest['totals_four_cycles']['negative_eligible_frames'],'four cycles negative qualification count')
            check(totals['window_exposures']==8192,'all8192 window exposures')
        result['conditions'][variant]=dict(status=meta['status'],updates_checked=len(rows),snapshots_checked=snaps,initial_state_sha256=initial_hashes[-1],cumulative_input_sha256=ih.hexdigest(),cumulative_label_sha256=lh.hexdigest(),totals=dict(totals),original_all_tensors_and_bn_bitwise_equal=True)
    check(len(set(initial_hashes))<=1,'identical full initialization including dormant temporal parameters')
    for a,b in zip(initial_states,initial_states[1:]):check(set(a)==set(b) and all(torch.equal(a[k],b[k]) for k in a),'direct initial tensor comparison')
    if not partial:
        check(len(set(inputchains))==len(set(labelchains))==1,'same complete input and label exposure across R0/R1/R2')
        result['final_all_three_complete']=True
        path=N/'checks/TRAINING_AUDIT.json'
        if path.exists():raise FileExistsError('Final audit receipt already exists')
        result['audit_code_sha256']=sha(__file__)
        path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');args=p.parse_args()
    torch.set_num_threads(2)
    main(partial=args.partial)
