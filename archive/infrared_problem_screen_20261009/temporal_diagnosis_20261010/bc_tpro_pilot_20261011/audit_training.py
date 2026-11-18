#!/usr/bin/env python3
"""CPU-only post-run audit; never calls forward, backward, or optimizer.step."""
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import statistics
import sys

import torch

B = Path(__file__).resolve().parent
Q = B.parent
OLD = Q / 'top1_background_control_20261011'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_hash(items):
    digest = hashlib.sha256()
    for name, value in sorted(items):
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def batch_hash(batch, positives, masks, specs):
    digest = hashlib.sha256()
    digest.update(batch.tobytes())
    digest.update(masks.tobytes())
    digest.update(json.dumps([positives, specs], sort_keys=True, separators=(',', ':')).encode())
    return digest.hexdigest()


def close(a, b):
    return math.isclose(a, b, abs_tol=1e-6, rel_tol=1e-6)


def numeric_finite(value):
    if isinstance(value, dict):
        return all(numeric_finite(v) for v in value.values())
    if isinstance(value, list):
        return all(numeric_finite(v) for v in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def summary(values):
    return {'min': min(values), 'mean': statistics.mean(values), 'median': statistics.median(values),
            'max': max(values), 'first': values[0], 'last': values[-1]}


def main():
    checks = {}
    results = {}
    runs = {}
    logs = {}
    checkpoints = {}
    sys.path.insert(0, str(B / 'code'))
    from bc_tpro import attach_bc_tpro, MODEL_CONFIG
    spec = importlib.util.spec_from_file_location('audit_legacy_training', OLD/'code/train_pilot.py')
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    old_run = read(OLD / 'training/run/RUN.json')
    old_rows = [json.loads(line) for line in (OLD/'training/run/TRAIN.jsonl').read_text().splitlines()]
    original, model_file = base.load_model(Path(old_run['source_path']), Path(old_run['checkpoint_path']))
    original_state = original.state_dict()
    original_keys = set(original_state)
    original_bn_keys = [key for key in original_state
                        if key.endswith(('running_mean', 'running_var', 'num_batches_tracked'))]
    initial_hash = None
    for variant in ['single_dynamic', 'bc_tpro']:
        directory = B / 'training' / variant
        run = runs[variant] = read(directory / 'run/RUN.json')
        rows = logs[variant] = [json.loads(line) for line in (directory/'run/TRAIN.jsonl').read_text().splitlines()]
        checkpoint = checkpoints[variant] = torch.load(directory/'run/final_model.pth', map_location='cpu', weights_only=True)
        state = checkpoint['model_state_dict']
        container = read(directory/'CONTAINER_STATE.json')
        launch = read(directory/'LAUNCH.json')
        isolation = read(directory/'run/GRADIENT_ISOLATION.json')
        prefix = variant + '/'
        def check(name, value):
            checks[prefix + name] = bool(value)
        check('container_exited_zero', container['Status']=='exited' and container['ExitCode']==0)
        check('completed512_log_plan_and_checkpoint', run['status']=='complete' and run['final_step']==512
              and checkpoint['step']==512 and [row['step'] for row in rows]==list(range(1,513)))
        check('variant_and_config_match', run['variant']==variant and checkpoint['variant']==variant
              and run['model_config']==checkpoint['model_config']==MODEL_CONFIG)
        check('current_train_source_matches_execution', sha(B/'code/train_bc.py')==run['train_source_sha256']==launch['code_sha256']['train_bc.py'])
        check('current_model_source_matches_execution', sha(B/'code/bc_tpro.py')==run['bc_tpro_source_sha256']==launch['code_sha256']['bc_tpro.py'])
        check('frozen_original_sources_unchanged', sha(OLD/'code/train_pilot.py')==run['original_train_source_sha256']
              and sha(OLD/'code/losses.py')==run['original_loss_source_sha256'] and sha(model_file)==run['model_source_sha256'])
        check('released_init_and_plan_source', sha(old_run['checkpoint_path'])==run['checkpoint_sha256']==checkpoint['initial_checkpoint_sha256']
              and run['plan_sha256']==old_run['plan_sha256']==checkpoint['plan_sha256'])
        check('fixed_training_hyperparameters', run['seed']==46 and run['steps']==512 and run['lr']==1e-4
              and run['weight_decay']==1e-4 and run['optimizer']=='fresh Adam' and run['reference_lambda']==1
              and run['effective_extra_lambda']==0 and run['precision']=='FP32_no_AMP_no_TF32')
        check('checkpoint_and_train_log_hashes', sha(directory/'run/final_model.pth')==run['final_checkpoint_sha256']
              and sha(directory/'run/TRAIN.jsonl')==run['train_log_sha256'])
        check('checkpoint_state_finite', all(bool(torch.isfinite(v).all()) for v in state.values()))
        check('old_bn_actual_buffers_unchanged', all(torch.equal(state[key], original_state[key]) for key in original_bn_keys))
        check('old_bn_logs_unchanged', run['bn_initial_sha256']==run['bn_final_sha256']==old_run['bn_initial_sha256']
              and all(row['bn_running_sha256']==run['bn_initial_sha256'] for row in rows))
        check('logged_gradient_isolation_pass', isolation['all_passed'] and all(isolation['checks'].values()))
        check('all_log_numbers_finite', all(row['finite'] and numeric_finite(row) for row in rows))
        check('detection_loss_equals_positive_plus_negative', all(close(row['detection_loss'], row['positive']+row['negative']) for row in rows))
        check('total_loss_equals_detection_plus_reference', all(close(row['loss'], row['detection_loss']+row['reference_loss'])
              and row['reference_lambda']==1 and row['effective_extra_lambda']==0 for row in rows))
        check('top1_logged_loss_agrees', all(row['negative']==row['background_top1pct_mean'] for row in rows))
        check('top1_k_rule', all(k==max(1, math.ceil(0.01*n)) for row in rows
              for n,k in zip(row['background_N_per_window'],row['background_k_per_window'])))
        check('same_background_and_positive_counts_as_legacy', all(row['background_N_per_window']==old['background_N_per_window']
              and row['base_objects_per_window']==old['base_objects_per_window'] for row,old in zip(rows,old_rows)))
        check('reference_outer_reduction', all(close(row['reference_loss'],statistics.mean(row['reference_loss_per_window'])) for row in rows))
        check('reference_relative_error_math', all(close(row['reference_relative_l1'],statistics.mean(row['reference_relative_l1_per_window']))
              and all(close(x, loss/max(scale,1e-6)) for x,loss,scale in zip(row['reference_relative_l1_per_window'],
                  row['reference_loss_per_window'],row['reference_target_abs_mean_per_window'])) for row in rows))
        check('reference_no_empty_window', all(row['reference_empty_windows']==0 and min(row['reference_valid_positions_per_window'])>0 for row in rows))
        check('gamma_chain_and_checkpoint', rows[0]['gamma']==0 and all(a['gamma_after_update']==b['gamma'] for a,b in zip(rows,rows[1:]))
              and rows[-1]['gamma_after_update']==run['gamma_final']==float(state['TPro.gamma']))
        check('gamma_live_all_steps', all(row['gamma_gradient'] is not None and row['gamma_gradient']!=0 for row in rows))
        check('association_live_after_first_step', rows[0]['gradient_norms']['association']==0
              and all(row['gradient_norms']['association']>0 for row in rows[1:]))
        check('reference_live_all_steps', all(row['gradient_norms']['reference']>0 for row in rows))
        check('memory_telemetry_agrees', max(row['max_cuda_memory_allocated_bytes'] for row in rows)==run['max_cuda_memory_allocated_bytes']
              and max(row['max_cuda_memory_reserved_bytes'] for row in rows)==run['max_cuda_memory_reserved_bytes'])
        check('new_parameter_counts', run['parameter_counts_by_group']=={'original':70913,'reference':1184,'association':320,'gamma':1})
        recreated, _ = base.load_model(Path(run['source_path']), Path(run['checkpoint_path']))
        original_recreated_state = {name:value.clone() for name,value in recreated.state_dict().items()}
        attach_bc_tpro(recreated, variant)
        recreated_state = recreated.state_dict()
        recreated_hash = tensor_hash((name,value) for name,value in recreated_state.items() if name not in original_keys)
        check('recreated_initial_weights_preserved', all(torch.equal(recreated_state[name],value) for name,value in original_recreated_state.items()))
        check('recreated_new_initial_hash', recreated_hash==run['new_module_initial_state_sha256']==checkpoint['new_module_initial_state_sha256'])
        initial_hash = recreated_hash if initial_hash is None else initial_hash
        check('pair_same_new_initial_hash', recreated_hash==initial_hash)
        group_changes = {}
        for group,names in run['parameter_names_by_gradient_group'].items():
            squares=sum(float((state[name]-recreated_state[name]).double().square().sum()) for name in names)
            group_changes[group]={'delta_l2':math.sqrt(squares),'changed_tensors':sum(not torch.equal(state[name],recreated_state[name]) for name in names),'tensor_count':len(names)}
        check('all_parameter_groups_updated', all(item['changed_tensors']>0 for item in group_changes.values()))
        corrections=[row['weighted_correction_rms']/max(row['original_pre_norm_rms'],1e-12) for row in rows]
        results[variant]={
            'duration_seconds':run['duration_seconds'],
            'peak_allocated_gib':run['max_cuda_memory_allocated_bytes']/2**30,
            'peak_reserved_gib':run['max_cuda_memory_reserved_bytes']/2**30,
            'gamma_final':run['gamma_final'],'gamma_pre_update':summary([row['gamma'] for row in rows]),
            'last_detection_loss':rows[-1]['detection_loss'],'last_positive':rows[-1]['positive'],'last_negative':rows[-1]['negative'],
            'last_reference_loss':rows[-1]['reference_loss'],'last_reference_relative_l1':rows[-1]['reference_relative_l1'],
            'reference_loss':summary([row['reference_loss'] for row in rows]),
            'reference_relative_l1':summary([row['reference_relative_l1'] for row in rows]),
            'reference_target_abs_mean':summary([statistics.mean(row['reference_target_abs_mean_per_window']) for row in rows]),
            'original_pre_norm_rms':summary([row['original_pre_norm_rms'] for row in rows]),
            'weighted_correction_rms':summary([row['weighted_correction_rms'] for row in rows]),
            'loss_and_correction_log_position':'forward before optimizer.step for each logged step; last forward uses step511 weights',
            'reference_positions_per_window':summary([n for row in rows for n in row['reference_valid_positions_per_window']]),
            'reference_empty_windows':sum(row['reference_empty_windows'] for row in rows),
            'weighted_correction_to_original_pre_norm_rms':summary(corrections),
            'ring_complete_fraction':rows[0]['ring_complete_fraction'],
            'gradient_norms':{group:summary([row['gradient_norms'][group] for row in rows]) for group in run['parameter_counts_by_group']},
            'parameter_group_changes':group_changes,'checkpoint_sha256':run['final_checkpoint_sha256'],
        }
        del recreated,recreated_state,original_recreated_state

    left,right=logs['single_dynamic'],logs['bc_tpro']
    checks['pair/all_512_input_hashes_identical']=all(a['input_sha256']==b['input_sha256'] for a,b in zip(left,right))
    checks['pair/all_reference_masks_counts_identical']=all(a['reference_valid_positions_per_window']==b['reference_valid_positions_per_window'] for a,b in zip(left,right))
    checks['pair/initial_detection_and_reference_losses_identical']=all(left[0][key]==right[0][key] for key in ['positive','negative','reference_loss','detection_loss'])
    plan=read(Q/'stage1_goal/pilot/TRAIN_PLAN.json')
    sequences={name:base.SequenceData(Q/'stage1_goal/data',name) for name in plan['train_sequences']}
    recomputed=[]
    for i,entry in enumerate(plan['entries']):
        batch,pos,masks,specs=base.build_batch(entry,sequences)
        recomputed.append(batch_hash(batch,pos,masks,specs))
        if i%128==0:print(json.dumps({'audit_reconstructed_training_batches':i+1}),flush=True)
    checks['pair/512_inputs_match_frozen_plan_reconstruction']=all(a['input_sha256']==h for a,h in zip(left,recomputed))
    cumulative=hashlib.sha256(''.join(recomputed).encode()).hexdigest()
    checks['pair/cumulative_input_hash_matches_reconstruction']=all(run['cumulative_input_sha256']==cumulative for run in runs.values())
    report={'status':'PASS' if all(checks.values()) else 'FAIL','scope':'completed training artifacts and CPU reconstruction; no detector-quality conclusion',
            'checks':checks,'check_count':len(checks),'variants':results,'cumulative_input_sha256':cumulative,
            'new_initial_state_sha256':initial_hash,
            'independence_limit':'Audit does not replay optimizer trajectory. All512 inputs are reconstructed using unchanged frozen data builder; checkpoint BN/state and arithmetic are independently checked. Gradient isolation is corroborated by saved first-step autograd gate and implementation tests.',
            'artifact_hashes':{str(p.relative_to(B)):sha(p) for variant in runs for p in [B/'training'/variant/'run/RUN.json',B/'training'/variant/'run/TRAIN.jsonl',B/'training'/variant/'run/GRADIENT_ISOLATION.json',B/'training'/variant/'CONTAINER_STATE.json',B/'training'/variant/'stdout.log',B/'training'/variant/'stderr.log']}}
    (B/'TRAINING_AUDIT.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    a,c=results['single_dynamic'],results['bc_tpro']
    lines=['# BC-TPro 已完成训练审计','',f"审计结果：**{report['status']}**，{len(checks)} 项检查。两次成功训练各512步，容器退出码均0；日志、状态和checkpoint已归档。首次工程导入失败发生在forward/update之前，未计入训练。",'',
           '| 配置 | 时间/s | 峰值allocated/GiB | 峰值reserved/GiB | 最终gamma | 最后检测损失 | 最后ref L1 | 最后ref相对L1 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,item in results.items():
        lines.append(f"| {name} | {item['duration_seconds']:.3f} | {item['peak_allocated_gib']:.3f} | {item['peak_reserved_gib']:.3f} | {item['gamma_final']:.8f} | {item['last_detection_loss']:.6f} | {item['last_reference_loss']:.6f} | {item['last_reference_relative_l1']:.6f} |")
    lines += ['', '512步输入/正框/背景掩码哈希逐项一致，并由冻结TRAIN_PLAN在CPU重新构建核对。两配置发布初始化源、新增模块初始tensor哈希、辅助mask有效数一致；原BN实际checkpoint buffers与发布权重逐tensor相同，512步日志也一致。', '',
              '两配置首步gamma为0、关联参数梯度为0；其后511步关联梯度均非零。参考预测器、原网络及gamma均有有限梯度并实际更新。首步Lref→原参数/关联/gamma、Ldet→参考参数的隔离检查通过。', '',
              '每窗检测背景top1%规则、四窗等权辅助归约、det=pos+neg、total=det+ref及相对误差算式均核对。没有空参考背景窗口。模型参数均为72,418（原70,913＋新增1,505）。', '',
              f"训练输入中只有{100*a['ring_complete_fraction']:.3f}%空间位置具有全部8方向参照，其余使用有效方向平均；该限制按预注册保留。最后一步加权修正/原归一化前特征RMS为 single {a['weighted_correction_to_original_pre_norm_rms']['last']:.6g}、BC {c['weighted_correction_to_original_pre_norm_rms']['last']:.6g}。这说明修正规模较小，不说明对最终检测无影响。", '',
              '| 配置 | 首步 ref L1/平均绝对F（逐窗归一后平均） | 末步相对L1 | 首步修正/原特征 RMS | 末步修正/原特征 RMS |','|---|---:|---:|---:|---:|',
              f"| single_dynamic | {a['reference_relative_l1']['first']:.6f} | {a['reference_relative_l1']['last']:.6f} | 0 | {100*a['weighted_correction_to_original_pre_norm_rms']['last']:.5f}% |",
              f"| bc_tpro | {c['reference_relative_l1']['first']:.6f} | {c['reference_relative_l1']['last']:.6f} | 0 | {100*c['weighted_correction_to_original_pre_norm_rms']['last']:.5f}% |", '',
              '表中损失与修正是每次optimizer.step之前的训练forward；末步对应第512次更新前的权重，不冒充最终checkpoint的额外测量。最终gamma另从保存checkpoint核实。首末不是同一个训练batch，不能把差值直接解释为受控重建改善或泛化改善。gamma非零、相对L1降低或修正规模小均不能单独判断检测有效/无效。是否优于单路控制、是否通过背景预算，须由冻结O/S评估回答。', '',
              '审计边界：没有重放优化器全轨迹；梯度隔离依赖已归档首步autograd审计与实现测试，最终参数/BN、全部输入、日志公式和哈希另行复核。完整检查与梯度/修正统计见 TRAINING_AUDIT.json。']
    if not all(checks.values()):lines+=['','失败项：'+', '.join(key for key,value in checks.items() if not value)]
    (B/'TRAINING_AUDIT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status':report['status'],'check_count':len(checks),'failed':[k for k,v in checks.items() if not v],'results':results},indent=2),flush=True)
    if not all(checks.values()):raise SystemExit(1)


if __name__=='__main__':main()
