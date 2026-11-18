#!/usr/bin/env python3
"""Recompute training integrity from frozen artifacts; no forward or updates."""
from __future__ import annotations
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import torch

N=Path(__file__).resolve().parents[1]
Q=N.parent
R=Q.parent
SOURCE=R/'official_deeppro'
REF=SOURCE/'log/sem_seg/RGB-T__2025-01-14_15-27__SoftLoUloss_DeepPro-Plus_DataL40/checkpoints/best_model.pth'
VARIANTS={'N0':'current','N1':'difference','N2':'relative'}
OLD=Q/'mutual_temporal_pilot_20261012/training/E0/run'
CHECKS=[]


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_hash(items):
    digest=hashlib.sha256()
    for name,value in sorted(items):
        digest.update(name.encode());digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def check(label,condition,**details):
    CHECKS.append({'check':label,'passed':bool(condition),**details})


def rows(path): return [json.loads(line) for line in Path(path).read_text().splitlines()]


def all_finite(value):
    if torch.is_tensor(value): return bool(torch.isfinite(value).all())
    if isinstance(value,dict): return all(all_finite(v) for v in value.values())
    if isinstance(value,(list,tuple)): return all(all_finite(v) for v in value)
    if isinstance(value,float): return math.isfinite(value)
    return True


def main():
    # Refuse partial audit rather than interpreting currently incomplete work.
    pending=[]
    for variant in VARIANTS:
        path=N/'training'/variant/'run/RUN.json'
        if not path.exists() or json.loads(path.read_text()).get('status')!='complete': pending.append(variant)
    if pending:
        print(json.dumps({'status':'PENDING','variants':pending}));return 2
    import importlib.util
    spec=importlib.util.spec_from_file_location('audit_only_original_loader',Q/'top1_background_control_20261011/code/train_pilot.py')
    base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
    original,model_path=base.load_model(SOURCE,REF)
    reference={k:v.detach().cpu() for k,v in original.state_dict().items()}
    reference_hash=tensor_hash(reference.items())
    old_rows=rows(OLD/'TRAIN.jsonl')
    old_run=json.loads((OLD/'RUN.json').read_text())
    check('historical_exposure_512',len(old_rows)==512)
    freeze=json.loads((N/'PROTOCOL_FREEZE.json').read_text())
    # Verify only recognized declared source dictionaries, not arbitrary outputs.
    freeze_sources=freeze.get('code_sha256',{})
    check('frozen_protocol',freeze.get('protocol_sha256')==sha(N/'PROTOCOL.md'))
    for rel,digest in freeze_sources.items(): check('frozen_code/'+rel,sha(N/rel)==digest)
    for rel,digest in freeze.get('artifacts_sha256',{}).items(): check('frozen_artifact/'+rel,sha(N/rel)==digest)
    summaries=[]; initial_states={}; telemetry_sets={}; all_exposure={}
    for variant,mode in VARIANTS.items():
        directory=N/'training'/variant/'run'
        run=json.loads((directory/'RUN.json').read_text()); initial=json.loads((directory/'INITIAL_STATE.json').read_text())
        log=rows(directory/'TRAIN.jsonl')
        check(variant+'/final_seed_steps',run['seed']==46 and run['steps']==512 and run['final_step']==512 and run['variant']==variant and run['mode']==mode)
        check(variant+'/optimizer_recipe',run['lr']==1e-4 and run['weight_decay']==1e-4 and run['betas']==[.9,.999] and run['eps']==1e-8)
        check(variant+'/parameter_count',run['parameter_count']==72442 and run['new_parameters']==1529 and run['trainable_parameters']==1529)
        check(variant+'/init_reference_identity',run['original_initial_state_sha256']==reference_hash and run['original_final_state_sha256']==reference_hash)
        check(variant+'/bn_metadata_unchanged',run['bn_initial_sha256']==run['bn_final_sha256']==old_run['bn_initial_sha256'])
        check(variant+'/train_log_hash',sha(directory/'TRAIN.jsonl')==run['train_log_sha256'])
        check(variant+'/exact_step_order',len(log)==512 and [r['step'] for r in log]==list(range(1,513)))
        check(variant+'/finite_logged_values',all(all_finite(row) and row['finite'] for row in log))
        check(variant+'/no_original_gradients',all(row['original_parameters_with_gradient']==0 for row in log))
        check(variant+'/bn_every_update',all(row['bn_running_sha256']==run['bn_initial_sha256'] for row in log))
        pairs=[(r['input_sha256'],r['label_sha256']) for r in log]
        all_exposure[variant]=pairs
        check(variant+'/all_512_exposures_match_historical',pairs==[(r['input_sha256'],r['label_sha256']) for r in old_rows])
        for key in ['input','label']:
            chain=hashlib.sha256()
            for row in log: chain.update(row[key+'_sha256'].encode())
            check(variant+'/'+key+'_chain_recomputed',chain.hexdigest()==run['cumulative_'+key+'_sha256']==old_run['cumulative_'+key+'_sha256'])
        counts={key:sum(row[key] for row in log) for key in ['pure_negative_windows','valid_pixels','positive_pixels']}
        check(variant+'/label_exposure_counts',counts==run['counts']==old_run['counts'])
        check(variant+'/same_label_domains_and_empty_policy',all(
            new['positive_pixels_per_window']==old['positive_pixels_per_window']
            and new['valid_pixels_per_window']==old['valid_pixels_per_window']
            and new['selected_negative_pixels_per_window']==old['selected_negative_pixels_per_window']
            and new['background_pixels_per_window']==old['background_pixels_per_window']
            for new,old in zip(log,old_rows)))
        check(variant+'/empty_domains_verified',all(
            all(bg==valid for mass,bg,valid in zip(row['positive_pixels_per_window'],row['background_pixels_per_window'],row['valid_pixels_per_window']) if mass==0)
            for row in log))
        check(variant+'/per_window_loss_policy',all(
            abs(row['loss']-sum(row['loss_per_window'])/4)<=1e-6 and
            all(abs(applied-(official if positive else negative))<=1e-7
                for applied,official,positive,negative in zip(row['loss_per_window'],row['original_official_loss_per_window'],row['positive_pixels_per_window'],row['empty_window_negative_per_window']))
            for row in log))
        check(variant+'/full_snapshot_schedule',[s['step'] for s in run['snapshots']]==[0,128,256,384,512])
        first,last=None,None
        train_names=initial['trainable_parameter_names']
        check(variant+'/trainable_names_branch_only',len(train_names)==8 and all(name.startswith('evidence.') for name in train_names))
        for snapshot in run['snapshots']:
            step=snapshot['step'];path=directory/snapshot['path']
            cp=torch.load(path,map_location='cpu',weights_only=True)
            state=cp['model_state_dict']
            check(f'{variant}/snapshot{step}/hash',sha(path)==snapshot['sha256'] and tensor_hash(state.items())==snapshot['model_state_sha256'])
            check(f'{variant}/snapshot{step}/finite',all_finite(state) and all_finite(cp['optimizer_state_dict']))
            check(f'{variant}/snapshot{step}/identity',cp['step']==step and cp['variant']==variant and cp['mode']==mode and cp['initial_checkpoint_sha256']==sha(REF))
            backbone={k.removeprefix('original.'):v for k,v in state.items() if k.startswith('original.')}
            check(f'{variant}/snapshot{step}/all_original_tensors_equal',set(backbone)==set(reference) and all(torch.equal(backbone[k],v) for k,v in reference.items()))
            check(f'{variant}/snapshot{step}/original_hash',tensor_hash(backbone.items())==reference_hash==cp['original_state_sha256'])
            optimizer=cp['optimizer_state_dict'];groups=optimizer['param_groups']
            check(f'{variant}/snapshot{step}/one_new_parameter_group',len(groups)==1 and groups[0]['params']==list(range(8)) and len(train_names)==8)
            group=groups[0]
            check(f'{variant}/snapshot{step}/optimizer_hyperparameters',group['lr']==1e-4 and group['weight_decay']==1e-4 and group['betas']==(.9,.999) and group['eps']==1e-8 and not group['amsgrad'])
            if step:
                check(f'{variant}/snapshot{step}/optimizer_only_branch',set(optimizer['state'])==set(group['params']) and all(
                    tuple(optimizer['state'][pid]['exp_avg'].shape)==tuple(state[name].shape)
                    and tuple(optimizer['state'][pid]['exp_avg_sq'].shape)==tuple(state[name].shape)
                    and int(optimizer['state'][pid]['step'])==step
                    for pid,name in zip(group['params'],train_names)))
            else:
                check(variant+'/fresh_optimizer',not optimizer['state'])
                first={k:v.clone() for k,v in state.items()};initial_states[variant]=first
                check(variant+'/initial_state_metadata',tensor_hash(first.items())==initial['model_state_sha256']==run['initial_model_state_sha256'])
                check(variant+'/branch_initial_hash',tensor_hash([(k.removeprefix('evidence.'),v) for k,v in first.items() if k.startswith('evidence.')])==run['branch_initial_sha256']==initial['branch_initial_sha256'])
                check(variant+'/zero_output_projection',int(torch.count_nonzero(state['evidence.output.weight']))==0 and int(torch.count_nonzero(state['evidence.output.bias']))==0)
                check(variant+'/other_layers_nonzero',all(int(torch.count_nonzero(state[k]))>0 for k in train_names if not k.startswith('evidence.output.')))
            if step==512: last=state
        check(variant+'/final_hardlink', (directory/'final_model.pth').stat().st_ino==(directory/'snapshot_512.pth').stat().st_ino and sha(directory/'final_model.pth')==run['final_checkpoint_sha256'])
        actual_diff={}
        for name in initial['parameter_names']:
            diff=last[name].double()-first[name].double()
            actual_diff[name]={'changed_elements':int(torch.count_nonzero(diff)), 'max_absolute_difference':float(diff.abs().max()),'l2_difference':float(diff.norm())}
        recorded=json.loads((directory/'PARAMETER_DIFF.json').read_text())
        check(variant+'/parameter_difference_recomputed',all(all(abs(actual_diff[k][field]-recorded['per_parameter'][k][field])<1e-12 for field in actual_diff[k]) for k in actual_diff))
        check(variant+'/original_parameters_all_unchanged',all(v['changed_elements']==0 for k,v in actual_diff.items() if k.startswith('original.')))
        check(variant+'/all_branch_parameter_tensors_updated',all(v['changed_elements']>0 for k,v in actual_diff.items() if k.startswith('evidence.')))
        gradients1=log[0]['module_gradients'];gradients2=log[1]['module_gradients']
        output_names=[k for k in gradients1 if k.startswith('output.')]
        upstream=[k for k in gradients1 if not k.startswith('output.')]
        check(variant+'/step1_output_gradient',sum(gradients1[k]['nonzero'] for k in output_names)>0)
        check(variant+'/step1_upstream_zero',sum(gradients1[k]['nonzero'] for k in upstream)==0)
        check(variant+'/step2_upstream_gradient',sum(gradients2[k]['nonzero'] for k in upstream)>0)
        sampled=rows(directory/'EVIDENCE_SAMPLES.jsonl')
        check(variant+'/telemetry_hash',sha(directory/'EVIDENCE_SAMPLES.jsonl')==run['telemetry_sha256'])
        check(variant+'/telemetry_steps',sorted(set(row['step'] for row in sampled))==[1]+list(range(16,513,16)))
        check(variant+'/telemetry_finite',all_finite(sampled))
        for delay in [1,4]:
            field_names=['D'+str(delay),'mu'+str(delay),'s'+str(delay),'E'+str(delay),'m'+str(delay)]
            check(f'{variant}/telemetry_missing_reference{delay}',all(all(row[k]==0 for k in field_names) for row in sampled if row['slot']<delay))
            if mode=='current': check(f'{variant}/empty_temporal_channels{delay}',all(all(row[k]==0 for k in field_names) for row in sampled))
            if mode=='difference':check(f'{variant}/empty_statistic_channels{delay}',all(row['mu'+str(delay)]==row['s'+str(delay)]==row['E'+str(delay)]==0 for row in sampled))
            if mode!='current':check(f'{variant}/valid_reference_mask{delay}',all(row['m'+str(delay)]==int(row['slot']>=delay) for row in sampled))
            if mode=='relative':check(f'{variant}/relative_formula{delay}',all(
                row['s'+str(delay)]>=0 and math.isclose(row['E'+str(delay)],(row['D'+str(delay)]-row['mu'+str(delay)])/max(row['s'+str(delay)],1/47.4845),rel_tol=1e-5,abs_tol=1e-5)
                for row in sampled if row['slot']>=delay))
        telemetry_sets[variant]=sampled
        summaries.append({'variant':variant,'mode':mode,'parameters':run['parameter_count'],'trainable_parameters':run['trainable_parameters'],
            'steps':len(log),'duration_seconds':run['duration_seconds'],'peak_GiB':run['max_cuda_memory_allocated_bytes']/2**30,
            'pure_negative_windows':counts['pure_negative_windows'],'changed_branch_parameter_tensors':sum(v['changed_elements']>0 for k,v in actual_diff.items() if k.startswith('evidence.')),
            'step1_output_grad_norm_sum':sum(gradients1[k]['norm'] for k in output_names),'step1_upstream_grad_norm_sum':sum(gradients1[k]['norm'] for k in upstream),
            'step2_upstream_grad_norm_sum':sum(gradients2[k]['norm'] for k in upstream),'telemetry_rows':len(sampled),
            'run_sha256':sha(directory/'RUN.json'),'final_checkpoint_sha256':sha(directory/'final_model.pth')})
    check('cross_mode_exposures_identical',all_exposure['N0']==all_exposure['N1']==all_exposure['N2'])
    check('cross_mode_initial_tensors_identical',all(set(initial_states[v])==set(initial_states['N0']) and all(torch.equal(initial_states[v][k],value) for k,value in initial_states['N0'].items()) for v in ['N1','N2']))
    pointkeys=['step','window','kind','slot','y','x','I','original_logit']
    check('cross_mode_telemetry_queries_identical',[[tuple(row[k] for k in pointkeys) for row in telemetry_sets[v]] for v in ['N1','N2']]==[[tuple(row[k] for k in pointkeys) for row in telemetry_sets['N0']]]*2)
    check('ordinary_and_relative_raw_difference_identical',all(a['D1']==b['D1'] and a['D4']==b['D4'] for a,b in zip(telemetry_sets['N1'],telemetry_sets['N2'])))
    passed=all(row['passed'] for row in CHECKS)
    report={'status':'PASS' if passed else 'FAIL','audit_scope':'artifact-only CPU recomputation; no model forward or optimizer update; performed by runner author, not independent blind reviewer',
            'script_sha256':sha(__file__),'protocol_freeze_sha256':sha(N/'PROTOCOL_FREEZE.json'), 'checks':CHECKS,'check_count':len(CHECKS),
            'failures':[r for r in CHECKS if not r['passed']],'training':summaries,
            'caveat':'Step1 upstream loss gradient is zero; Adam weight decay can nevertheless update nonzero upstream weights. Training integrity is not evidence of detection benefit.'}
    (N/'TRAINING_AUDIT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    lines=['# 固定分支训练执行审计','',f"状态：**{report['status']}**；{len(CHECKS)}项核查。仅重新读取产物，不调用模型前向或优化器。执行者为运行器作者，属于独立数值复算，不冒称盲审。",'',
           '核查覆盖三组seed46、512步、逐步输入及标签与历史E0的一致性、五个检查点发布参数和BN缓存逐张量保持、Adam仅含8个新增参数张量，以及三组完整初始张量相同。逐项结论见机器可读报告；PASS表示全部通过。','',
           '|配置|总参数/可学参数|耗时秒|峰值分配GiB|新增更新张量|','|---|---:|---:|---:|---:|']
    for row in summaries: lines.append(f"|{row['variant']}|{row['parameters']}/{row['trainable_parameters']}|{row['duration_seconds']:.2f}|{row['peak_GiB']:.3f}|{row['changed_branch_parameter_tensors']}/8|")
    lines+=['','首步输出投影获得损失梯度，其他新增层损失梯度为0；第二步上游梯度启动。Adam的weight decay仍可能使首步上游权重改变，不能把该改变误当作损失梯度已经贯通。','',
            '同次前向遥测的固定目标像素及原模型高分背景位置在三组完全一致，N1/N2原差分一致；缺失参考及置空通道符合模式定义。遥测仅解释实现，不代表已恢复对象或机制有效。','',
            '主结果保持最终512步；0/128/256/384检查点未用于选择最佳检测成绩。']
    if not passed:lines+=['','未通过项：',*[str(row) for row in report['failures']]]
    (N/'TRAINING_AUDIT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status':report['status'],'check_count':len(CHECKS),'failures':report['failures'],'training':summaries},ensure_ascii=False))
    return 0 if passed else 1


if __name__=='__main__':sys.exit(main())
