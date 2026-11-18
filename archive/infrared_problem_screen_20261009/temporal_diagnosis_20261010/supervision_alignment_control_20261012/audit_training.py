#!/usr/bin/env python3
"""CPU audit of the fixed supervision control; no model forward or updates."""
from pathlib import Path
import hashlib
import json
import math
import statistics
import sys

import numpy as np
import torch

A = Path(__file__).resolve().parent
Q = A.parent
sys.path.insert(0, str(A / 'code'))
import train_supervision as runner
from gaussian_labels import build_targets, LABEL_CONFIG


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def finite(value):
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def close(left, right):
    return math.isclose(left, right, rel_tol=1e-6, abs_tol=1e-6)


def describe(values):
    return {'first': values[0], 'last': values[-1], 'min': min(values),
            'max': max(values), 'mean': statistics.mean(values), 'median': statistics.median(values)}


def main():
    torch.set_num_threads(1)
    checks = {}
    def check(name, value):
        checks[name] = bool(value)
    condition = 'gaussian_softiou_approx'
    directory = A / 'training' / condition
    rdir = directory / 'run'
    run = read(rdir / 'RUN.json')
    log = rows(rdir / 'TRAIN.jsonl')
    state_record = read(directory / 'CONTAINER_STATE.json')
    launch = read(directory / 'LAUNCH.json')
    frozen = read(A / 'PROTOCOL_FREEZE.json')
    preflight = read(A / 'LABEL_PREFLIGHT.json')
    runner_check = read(A / 'checks/TRAIN_RUNNER_CHECK.json')
    initial_record = read(rdir / 'INITIAL_STATE.json')
    differences = read(rdir / 'PARAMETER_DIFF.json')
    checkpoint = torch.load(rdir / 'final_model.pth', map_location='cpu', weights_only=True)
    state = checkpoint['model_state_dict']
    base = runner.load_module('supervision_audit_old_base', runner.OLD / 'code/train_pilot.py')
    original, model_file = base.load_model(Path(run['source_path']), Path(run['checkpoint_path']))
    original_state = original.state_dict()
    parameters = dict(original.named_parameters())
    bn_keys = [key for key in original_state if key.endswith(('running_mean', 'running_var', 'num_batches_tracked'))]
    original_bn_hash = base.buffer_hash(base.bn_snapshot(original))
    bc = {}
    for variant in ['single_dynamic', 'bc_tpro']:
        bc[variant] = rows(Q / 'bc_tpro_pilot_20261011/training' / variant / 'run/TRAIN.jsonl')
    check('container_exited_zero', state_record['Status'] == 'exited' and state_record['ExitCode'] == 0)
    check('container_not_oom_killed', not state_record.get('OOMKilled', False))
    check('launcher_succeeded_same_container_name', launch['launch_returncode'] == 0
          and launch['container_name'] == 'supervision_train_gaussian_softiou_approx_20261012')
    check('immutable_validated_image', launch['image_id'] == 'sha256:eb956873870d737f2b68dff0724b8f6f56f778829d119b0938d29b7abeaa2f92')
    check('launch_protocol_freeze_hash', launch['protocol_freeze_sha256'] == runner.sha(A / 'PROTOCOL_FREEZE.json'))
    check('protocol_text_hash_unchanged', frozen['protocol_sha256'] == runner.sha(A / 'PROTOCOL.md'))
    check('all_frozen_code_files_unchanged', all(runner.sha(A / name) == fingerprint for name, fingerprint in frozen['code_sha256'].items()))
    check('all_frozen_preflight_files_unchanged', all(runner.sha(A / name) == fingerprint for name, fingerprint in frozen['preflight_sha256'].items()))
    check('label_preflight_pass', preflight['status'] == 'PASS' and all(item['pass'] for item in preflight['checks']))
    check('runner_integration_preflight_pass', runner_check['status'] == 'PASS' and all(runner_check['checks'].values()))
    check('final_run_status_and_step', run['status'] == 'complete' and run['final_step'] == 512 and checkpoint['step'] == 512)
    check('exactly_512_ordered_updates', [row['step'] for row in log] == list(range(1, 513)))
    check('fixed_condition_and_variant', run['condition'] == run['variant'] == checkpoint['condition'] == checkpoint['variant'] == condition
          and all(row['condition'] == row['variant'] == condition for row in log))
    check('one_fixed_final_checkpoint_rule', run['checkpoint_selection'] == 'fixed final update512; no best checkpoint'
          and sorted(path.name for path in rdir.glob('*.pth')) == ['final_model.pth'])
    check('official_model_source_hash', runner.sha(model_file) == run['model_source_sha256'] == runner.EXPECTED['model_source'])
    check('runner_source_hash', runner.sha(A / 'code/train_supervision.py') == run['train_source_sha256'] == checkpoint['train_source_sha256'])
    check('label_source_hash', runner.sha(A / 'code/gaussian_labels.py') == run['label_source_sha256'] == checkpoint['label_source_sha256'])
    check('old_data_helper_source_hash', runner.sha(runner.OLD / 'code/train_pilot.py') == run['original_train_source_sha256'] == runner.EXPECTED['original_train_source'])
    check('released_checkpoint_hash', runner.sha(run['checkpoint_path']) == run['checkpoint_sha256'] == checkpoint['initial_checkpoint_sha256'] == runner.EXPECTED['checkpoint'])
    check('frozen_label_config_equal', run['label_config'] == checkpoint['label_config'] == frozen['label_config'] == preflight['label_config'] == LABEL_CONFIG)
    check('frozen_seed_steps_lr', run['seed'] == 46 and run['steps'] == 512 and run['lr'] == 1e-4)
    check('fresh_adam_hyperparameters', run['optimizer'] == 'fresh Adam' and run['betas'] == [0.9, 0.999]
          and run['eps'] == 1e-8 and run['weight_decay'] == 1e-4)
    check('fp32_no_amp_tf32_contract', run['precision'] == 'FP32_no_AMP_no_TF32')
    check('no_new_architecture_parameters', run['parameter_count'] == initial_record['parameter_count'] == 70913
          and run['new_parameters'] == 0 and sum(value.numel() for value in parameters.values()) == 70913)
    check('same_state_names_as_release', set(state) == set(original_state))
    check('same_state_shapes_as_release', all(state[key].shape == value.shape for key, value in original_state.items()))
    check('same_state_dtypes_as_release', all(state[key].dtype == value.dtype for key, value in original_state.items()))
    check('all_final_state_values_finite', all(bool(torch.isfinite(value).all()) for value in state.values()))
    check('initial_model_state_hash_matches_release', runner.tensor_hash(original_state.items()) == initial_record['model_state_sha256'] == run['initial_model_state_sha256'])
    check('initial_state_individual_hashes_match_release', initial_record['state_tensor_sha256'] == {key: runner.tensor_hash([(key, value)]) for key, value in original_state.items()})
    check('initial_parameter_names_match_release', initial_record['parameter_names'] == sorted(parameters))
    check('final_state_hash_matches_checkpoint', runner.tensor_hash(state.items()) == run['final_model_state_sha256'])
    check('final_checkpoint_file_hash', runner.sha(rdir / 'final_model.pth') == run['final_checkpoint_sha256'])
    check('train_log_file_hash', runner.sha(rdir / 'TRAIN.jsonl') == run['train_log_sha256'])
    check('parameter_diff_file_hash', runner.sha(rdir / 'PARAMETER_DIFF.json') == run['parameter_diff_sha256'])
    check('bn_checkpoint_buffers_bitwise_unchanged', all(torch.equal(state[key], original_state[key]) for key in bn_keys))
    check('bn_initial_final_and_step_hashes_unchanged', run['bn_initial_sha256'] == run['bn_final_sha256'] == original_bn_hash
          and all(row['bn_running_sha256'] == original_bn_hash for row in log))
    check('bn_affine_training_policy', run['bn_policy'] == 'eval_running_stats_affine_trainable')
    check('loss_implementation_declared_actual_official', run['criterion'] == 'checkpoint-adjacent unmodified SoftLoUloss; internal sigmoid; smooth=0')
    check('loss_outer_reduction_is_four_window_equal_mean', all(len(row['loss_per_window']) == 4 and close(row['loss'], statistics.mean(row['loss_per_window'])) for row in log))
    check('loss_values_in_unit_interval', all(0 <= value <= 1 for row in log for value in row['loss_per_window']))
    check('all_logged_values_finite', all(row['finite'] and finite(row) for row in log))
    check('all_gradients_logged_finite', all(math.isfinite(row['grad_norm']) and row['grad_norm'] >= 0 for row in log))
    check('all_parameters_receive_gradient_tensors', all(row['parameters_with_gradient'] == len(parameters) for row in log))
    check('all_window_valid_domains_nonempty', all(len(row['valid_pixels_per_window']) == 4 and min(row['valid_pixels_per_window']) > 0 for row in log))
    check('binary_targets_mass_equals_positive_count', all(row['target_mass_per_window'] == row['positive_pixels_per_window'] for row in log))
    check('per_window_totals_consistent', all(row['valid_pixels'] == sum(row['valid_pixels_per_window'])
          and row['positive_pixels'] == sum(row['positive_pixels_per_window']) and close(row['target_mass'], sum(row['target_mass_per_window'])) for row in log))
    check('pure_negative_count_matches_zero_mass', all(row['pure_negative_windows'] == sum(value == 0 for value in row['target_mass_per_window']) for row in log))
    check('pure_negative_windows_have_loss_one', all(loss == 1 for row in log for mass, loss in zip(row['target_mass_per_window'], row['loss_per_window']) if mass == 0))
    check('pure_negative_total_385_frozen', sum(row['pure_negative_windows'] for row in log) == run['counts']['pure_negative_windows'] == preflight['full_plan']['pure_negative_windows'] == 385)
    check('label_stats_match_loss_counts', all(row['label_stats']['target_mass_per_window'] == row['target_mass_per_window']
          and row['label_stats']['valid_pixels_per_window'] == row['valid_pixels_per_window']
          and row['label_stats']['pure_negative_windows'] == row['pure_negative_windows'] for row in log))
    check('label_stats_match_frozen_preflight_every_step', all(row['label_stats'] == expected['stats'] for row, expected in zip(log, preflight['full_plan']['per_update'])))
    check('run_aggregate_counts_match_logs', run['counts'] == {
        'pure_negative_windows': sum(row['pure_negative_windows'] for row in log),
        'valid_pixels': sum(row['valid_pixels'] for row in log),
        'positive_pixels': sum(row['positive_pixels'] for row in log)})
    check('input_hashes_match_old_single_dynamic_512', len(bc['single_dynamic']) == len(log) and all(row['input_sha256'] == old['input_sha256'] for row, old in zip(log, bc['single_dynamic'])))
    check('input_hashes_match_old_bc_tpro_512', len(bc['bc_tpro']) == len(log) and all(row['input_sha256'] == old['input_sha256'] for row, old in zip(log, bc['bc_tpro'])))
    check('gpu_memory_telemetry_matches_run', max(row['max_cuda_memory_allocated_bytes'] for row in log) == run['max_cuda_memory_allocated_bytes']
          and max(row['max_cuda_memory_reserved_bytes'] for row in log) == run['max_cuda_memory_reserved_bytes'])
    check('elapsed_times_monotonic', all(a['elapsed_seconds'] <= b['elapsed_seconds'] for a, b in zip(log, log[1:])) and log[-1]['elapsed_seconds'] <= run['duration_seconds'])

    recomputed_diffs = {}
    for name in sorted(parameters):
        delta = state[name].double() - original_state[name].double()
        recomputed_diffs[name] = {'numel': delta.numel(), 'changed_elements': int(torch.count_nonzero(delta)),
                                 'max_absolute_difference': float(delta.abs().max()), 'l2_difference': float(delta.norm())}
    check('all_recorded_parameter_differences_recomputed', differences['per_parameter'] == recomputed_diffs)
    check('parameter_difference_summary_correct', differences['new_parameter_count'] == 0 and differences['parameter_count'] == 70913
          and differences['parameter_tensors'] == len(parameters)
          and differences['changed_parameter_tensors'] == sum(value['changed_elements'] > 0 for value in recomputed_diffs.values()))
    check('model_parameters_actually_updated', any(value['changed_elements'] > 0 for value in recomputed_diffs.values()))
    affine_names = [f'{name}.{suffix}' for name, module in original.named_modules() if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
                    for suffix in ['weight', 'bias'] if getattr(module, suffix) is not None]
    check('bn_affine_parameters_actually_updated', all(recomputed_diffs[name]['changed_elements'] > 0 for name in affine_names))

    plan = read(Q / 'stage1_goal/pilot/TRAIN_PLAN.json')
    selection_path = Q / 'stage1_goal/DATA_SELECTION.json'
    check('plan_and_selection_fingerprints', runner.sha(Q / 'stage1_goal/pilot/TRAIN_PLAN.json') == run['plan_sha256'] == checkpoint['plan_sha256'] == preflight['full_plan']['plan_sha256'] == runner.EXPECTED['plan']
          and runner.sha(selection_path) == run['selection_sha256'] == preflight['full_plan']['selection_sha256'] == runner.EXPECTED['selection'])
    sequences = {name: base.SequenceData(Q / 'stage1_goal/data', name) for name in plan['train_sequences']}
    check('all_frozen_training_data_fingerprints', {name: sequence.fingerprint for name, sequence in sequences.items()} == plan['data_fingerprints'])
    check('training_sequence_names_unchanged', run['training_sequences'] == plan['train_sequences'] == preflight['full_plan']['training_sequences'])
    reconstructed_inputs, reconstructed_labels = [], []
    target_diffs, valid_diffs, stats_diffs = [], [], []
    chain = hashlib.sha256()
    # Reconstruct all actual input and label byte arrays, then bridge the two
    # different documented fingerprint schemas used by preflight and runner.
    for index, entry in enumerate(plan['entries']):
        batch, positive, old_mask, specs = base.build_batch(entry, sequences)
        target, valid, stats = build_targets(entry, sequences, background_masks=old_mask)
        step = index + 1
        expected = preflight['full_plan']['per_update'][index]
        reconstructed_inputs.append(runner.data_hash(batch, positive, old_mask, specs))
        reconstructed_labels.append(runner.label_hash(target, valid))
        if hashlib.sha256(target.tobytes()).hexdigest() != expected['target_sha256']:
            target_diffs.append(step)
        if hashlib.sha256(valid.tobytes()).hexdigest() != expected['valid_sha256']:
            valid_diffs.append(step)
        if stats != expected['stats']:
            stats_diffs.append(step)
        chain.update(step.to_bytes(4, 'little'))
        chain.update(target.tobytes())
        chain.update(valid.tobytes())
        if index % 128 == 0:
            print(json.dumps({'audit_reconstructed_updates': step}), flush=True)
    input_chain = hashlib.sha256(''.join(reconstructed_inputs).encode()).hexdigest()
    label_chain = hashlib.sha256(''.join(reconstructed_labels).encode()).hexdigest()
    check('all_512_inputs_reconstructed_from_frozen_plan', reconstructed_inputs == [row['input_sha256'] for row in log])
    check('all_512_label_hashes_reconstructed', reconstructed_labels == [row['label_sha256'] for row in log])
    check('all_512_target_byte_hashes_match_preflight', not target_diffs)
    check('all_512_valid_domain_byte_hashes_match_preflight', not valid_diffs)
    check('all_512_label_statistics_reconstructed', not stats_diffs)
    check('raw_target_and_valid_chain_matches_preflight', chain.hexdigest() == preflight['full_plan']['target_and_domain_chain_sha256'])
    check('runner_cumulative_input_hash_matches_reconstruction', input_chain == run['cumulative_input_sha256'])
    check('runner_cumulative_label_hash_matches_reconstruction', label_chain == run['cumulative_label_sha256'])

    result = {
        'duration_seconds': run['duration_seconds'],
        'peak_allocated_gib': run['max_cuda_memory_allocated_bytes'] / 2**30,
        'peak_reserved_gib': run['max_cuda_memory_reserved_bytes'] / 2**30,
        'loss': describe([row['loss'] for row in log]), 'gradient_norm': describe([row['grad_norm'] for row in log]),
        'target_mass_per_window': describe([value for row in log for value in row['target_mass_per_window']]),
        'valid_pixels_per_window': describe([value for row in log for value in row['valid_pixels_per_window']]),
        'pure_negative_windows': 385, 'window_exposures': 2048,
        'entirely_negative_updates': sum(row['pure_negative_windows'] == 4 for row in log),
        'positive_target_pixels_exposure_sum': sum(row['positive_pixels'] for row in log),
        'valid_pixels_exposure_sum': sum(row['valid_pixels'] for row in log),
        'changed_parameter_tensors': differences['changed_parameter_tensors'],
        'parameter_tensors': len(parameters), 'parameter_count': 70913,
        'parameter_delta_l2': math.sqrt(sum(value['l2_difference'] ** 2 for value in recomputed_diffs.values())),
        'bn_affine_tensor_count': len(affine_names),
        'cumulative_input_sha256': input_chain, 'cumulative_label_sha256': label_chain,
        'preflight_target_and_domain_chain_sha256': chain.hexdigest(),
        'final_checkpoint_sha256': run['final_checkpoint_sha256'],
        'loss_log_position': 'forward before optimizer.step; final logged loss uses step511 weights on the last planned batch',
    }
    artifacts = [rdir/'RUN.json', rdir/'TRAIN.jsonl', rdir/'INITIAL_STATE.json', rdir/'PARAMETER_DIFF.json',
                 directory/'LAUNCH.json', directory/'CONTAINER_STATE.json', directory/'stdout.log', directory/'stderr.log',
                 A/'PROTOCOL_FREEZE.json', A/'LABEL_PREFLIGHT.json', A/'checks/TRAIN_RUNNER_CHECK.json']
    report = {
        'status': 'PASS' if all(checks.values()) else 'FAIL', 'check_count': len(checks), 'checks': checks,
        'condition': condition, 'result': result,
        'scope': 'completed training artifacts and CPU full-plan reconstruction; no quality or causality claim',
        'independence_limit': 'No optimizer replay and no model forward. Input/label arrays, actual final BN/state, parameter differences and logged arithmetic independently checked. Official criterion gradient behavior supported by frozen CPU preflight, not re-measured per training step.',
        'artifact_hashes': {str(path.relative_to(A)): runner.sha(path) for path in artifacts},
        'audit_source_sha256': runner.sha(__file__),
    }
    runner.save_json(A/'TRAINING_AUDIT.json', report)
    lines = [
        '# 监督近似控制：已完成训练审计', '',
        f"审计结果：**{report['status']}**，共 {len(checks)} 项检查。训练完成 512 次更新，容器退出码为 0，输出固定末次 checkpoint；未追加配置或重训。", '',
        '| 更新数 | 训练秒数 | 峰值 allocated/GiB | 峰值 reserved/GiB | 首步 loss | 末步 loss |',
        '|---:|---:|---:|---:|---:|---:|',
        f"| 512 | {result['duration_seconds']:.3f} | {result['peak_allocated_gib']:.3f} | {result['peak_reserved_gib']:.3f} | {result['loss']['first']:.6f} | {result['loss']['last']:.6f} |", '',
        '全部 512 步的归一化输入、旧正框与背景 mask 指纹均与已完成的两组 BC 控制一致，并从冻结 TRAIN_PLAN 与原数据重新构建核对。这里只复用其输入证据，没有扩展 BC 模型。', '',
        '全部 target/valid 字节分别复算，与预检的 512 项明细及原始字节链一致；再按训练入口的带 dtype/shape 联合标签 hash 格式复算，与逐步训练标签及累计链一致。两种 hash 定义不同，通过同一组重建数组对接，未错误地直接比较两条链。', '',
        f"2048 个窗口暴露中有 **385 个纯负窗口**，日志中的 loss 均为 1；整个 batch 均纯负的更新数为 {result['entirely_negative_updates']}。官方 smooth=0 导致纯负窗口的损失梯度为零，这由冻结的 CPU 梯度检查验证；本审计不声称重测了每个训练窗口的梯度。Adam 动量与 weight decay 仍可更新参数。", '',
        '四窗等权归约、目标质量/正像素/有效像素计数、所有日志数值有限及梯度记录均通过。标签统计与训练前预检逐步一致，未更换有效域或目标规则。', '',
        f"模型仍为 **70,913 个参数**，state 名称、shape 和 dtype 与发布权重相同。实际最终 checkpoint 的全部 BN buffers 与发布值逐 tensor、逐位一致；{result['changed_parameter_tensors']}/{result['parameter_tensors']} 个参数 tensor 实际变化，其中 BN affine 参数也实际更新。逐参数差异由发布值与最终 checkpoint 重新计算，与保存结果一致。", '',
        '运行代码、标签生成器、原模型源码、旧数据构造器、协议与预检文件 hash 均与冻结记录一致。GPU0 在 docker 创建前因其他任务占用而被资源检查拒绝；同配置最终使用启动记录中的空闲 GPU，没有模型更新发生在被拒绝的启动尝试中。', '',
        '首末 loss 对应不同训练 batch，且每行 loss 是 optimizer.step 之前的前向值；末行并非对最终 checkpoint 的另一次测量。不能据首末下降直接推断检测性能、背景预算或泛化改善。', '',
        '审计边界：没有重放优化器全轨迹，没有模型前向或新训练。仅核对实际训练产物、日志算式和全量输入/标签；最终检测结论须由冻结 O/S 推理评价给出。这仍是 Gaussian 派生标签与掩码有效域下的官方损失近似控制，不是完整官方训练复现。', '',
        '详细检查、资源统计和各来源 hash 见 `TRAINING_AUDIT.json`；容器 stdout/stderr 与状态已存入训练目录。',
    ]
    if not all(checks.values()):
        lines += ['', '失败项：' + ', '.join(name for name, passed in checks.items() if not passed)]
    (A/'TRAINING_AUDIT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'status': report['status'], 'check_count': len(checks),
                      'failed': [key for key, value in checks.items() if not value], 'result': result}, indent=2), flush=True)
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
