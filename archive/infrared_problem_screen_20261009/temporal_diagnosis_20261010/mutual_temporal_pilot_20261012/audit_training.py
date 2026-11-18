#!/usr/bin/env python3
"""Post-training read-only audit. No forward pass, optimizer, or source edits."""
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
Q = ROOT.parent
PREVIOUS = Q / 'supervision_alignment_control_20261012'


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def tensor_hash(state):
    value = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value.update(name.encode()); value.update(str(tensor.dtype).encode()); value.update(str(tuple(tensor.shape)).encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def bn_hash(state):
    value = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        if name.endswith(('running_mean', 'running_var', 'num_batches_tracked')):
            value.update(name.encode()); value.update(str(tensor.dtype).encode())
            value.update(tensor.detach().cpu().numpy().tobytes())
    return value.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def summary(values):
    a = np.asarray(values, dtype=float)
    return {'count': int(a.size), 'mean': float(a.mean()), 'min': float(a.min()), 'max': float(a.max()),
            'median': float(np.median(a)), 'p10': float(np.percentile(a, 10)), 'p90': float(np.percentile(a, 90))}


def main():
    torch.set_num_threads(1)
    checks, details, bins, resource = {}, {}, [], []
    freeze = load_json(ROOT / 'PROTOCOL_FREEZE.json')
    for group in ['code_sha256', 'preflight_sha256']:
        for name, expected in freeze[group].items():
            checks['freeze:' + name] = sha(ROOT / name) == expected
    for name, expected in freeze['external_sha256'].items():
        checks['freeze_external:' + name] = sha(name) == expected
    checks['protocol_unchanged'] = sha(ROOT / 'PROTOCOL.md') == freeze['protocol_sha256']
    launch_files = sorted((ROOT / 'training').glob('*/LAUNCH.json'))
    launches = [load_json(path) for path in launch_files]
    checks['exactly_three_formal_training_launches'] = len(launches) == 3 and sorted(r['condition'] for r in launches) == ['E0', 'E1', 'E2']
    checks['all_launches_seed46_512_radius2'] = all(r['job'] == 'train' and r['step'] == 512 and r['launch_returncode'] == 0
        and r['command'][r['command'].index('--seed') + 1] == '46'
        and r['command'][r['command'].index('--radius') + 1] == '2' for r in launches)
    checks['all_launches_bind_protocol_freeze'] = all(r['protocol_freeze_sha256'] == sha(ROOT / 'PROTOCOL_FREEZE.json') for r in launches)
    container_receipts = load_json(ROOT / 'TRAINING_CONTAINER_RECEIPTS.json')
    checks['formal_containers_all_completed_exit_zero'] = len(container_receipts['containers']) == 3 and all(
        r['state']['Status'] == 'exited' and r['state']['ExitCode'] == 0 for r in container_receipts['containers'])
    old_rows = [json.loads(line) for line in (PREVIOUS / 'training/gaussian_softiou_approx/run/TRAIN.jsonl').read_text().splitlines()]
    old_run = load_json(PREVIOUS / 'training/gaussian_softiou_approx/run/RUN.json')
    logs, runs, initials, states0, final_states = {}, {}, {}, {}, {}
    for variant in ['E0', 'E1', 'E2']:
        folder = ROOT / 'training' / variant / 'run'
        run = runs[variant] = load_json(folder / 'RUN.json')
        rows = logs[variant] = [json.loads(line) for line in (folder / 'TRAIN.jsonl').read_text().splitlines()]
        initial = initials[variant] = load_json(folder / 'INITIAL_STATE.json')
        prefix = variant + ':'
        checks[prefix + 'completed_fixed512'] = run['status'] == 'complete' and run['final_step'] == run['steps'] == 512
        checks[prefix + 'exact_512_rows_in_order'] = len(rows) == 512 and [r['step'] for r in rows] == list(range(1, 513))
        checks[prefix + 'all_input_hashes_equal_previous'] = len(rows) == len(old_rows) and all(a['input_sha256'] == b['input_sha256'] for a,b in zip(rows, old_rows))
        checks[prefix + 'all_label_hashes_equal_previous'] = len(rows) == len(old_rows) and all(a['label_sha256'] == b['label_sha256'] for a,b in zip(rows, old_rows))
        checks[prefix + 'all_label_stats_equal_previous'] = all(a['label_stats'] == b['label_stats'] for a,b in zip(rows, old_rows))
        for key, field in [('input_sha256', 'cumulative_input_sha256'), ('label_sha256', 'cumulative_label_sha256')]:
            digest = hashlib.sha256()
            for row in rows: digest.update(row[key].encode())
            checks[prefix + field] = digest.hexdigest() == run[field] == old_run[field]
        checks[prefix + 'same_released_initialization'] = run['checkpoint_sha256'] == old_run['checkpoint_sha256'] and run['initial_original_state_sha256'] == old_run['initial_model_state_sha256']
        checks[prefix + 'same_optimizer_configuration'] = run['optimizer'] == 'fresh Adam' and run['lr'] == run['weight_decay'] == 1e-4 and run['betas'] == [.9,.999] and run['eps'] == 1e-8
        checks[prefix + 'same_frozen_bn_and_affine_policy'] = run['bn_policy'] == 'eval_running_stats_affine_trainable' and run['bn_initial_sha256'] == run['bn_final_sha256'] == old_run['bn_initial_sha256']
        checks[prefix + 'bn_log_eachstep_unchanged'] = all(row['bn_running_sha256'] == run['bn_initial_sha256'] for row in rows)
        checks[prefix + 'loss_and_gradients_finite'] = all(row['finite'] and math.isfinite(row['loss']) and math.isfinite(row['grad_norm']) and row['grad_norm'] > 0 for row in rows)
        checks[prefix + 'labels_and_background_counts_preserved'] = all(row['valid_pixels_per_window'] == prev['valid_pixels_per_window'] and row['positive_pixels_per_window'] == prev['positive_pixels_per_window'] for row,prev in zip(rows,old_rows))
        positives, negatives = [], []
        for row in rows:
            for wi, npositive in enumerate(row['positive_pixels_per_window']):
                entry = (row, wi)
                (positives if npositive else negatives).append(entry)
        checks[prefix + 'same_385_empty_windows'] = len(negatives) == 385 and run['counts']['pure_negative_windows'] == 385
        checks[prefix + 'positive_window_loss_exactly_official_logged'] = all(r['loss_per_window'][wi] == r['original_official_loss_per_window'][wi]
            and r['empty_window_negative_per_window'][wi] == 0 and r['selected_negative_pixels_per_window'][wi] == 0 for r,wi in positives)
        checks[prefix + 'empty_window_loss_exactly_top1_logged'] = all(r['loss_per_window'][wi] == r['empty_window_negative_per_window'][wi]
            and r['original_official_loss_per_window'][wi] == 1.0
            and r['selected_negative_pixels_per_window'][wi] == max(1, math.ceil(.01*r['background_pixels_per_window'][wi]))
            and r['valid_pixels_per_window'][wi] == r['background_pixels_per_window'][wi] for r,wi in negatives)
        checks[prefix + 'outer_loss_equal_four_window_mean'] = all(abs(r['loss'] - sum(r['loss_per_window'])/4) <= max(1e-7, 1e-6 * abs(r['loss'])) for r in rows)
        checks[prefix + 'frozen_loss_source_bindings'] = run['train_source_sha256'] == freeze['code_sha256']['code/train_mutual.py'] and run['module_source_sha256'] == freeze['code_sha256']['code/module.py']
        checks[prefix + 'preserved_log_and_final_hash'] = sha(folder / 'TRAIN.jsonl') == run['train_log_sha256'] and sha(folder / 'final_model.pth') == run['final_checkpoint_sha256']
        checks[prefix + 'exact_snapshot_steps'] = [x['step'] for x in run['snapshots']] == [0,128,256,384,512]
        snapshot_summaries = []
        for info in run['snapshots']:
            step = info['step']
            cp = torch.load(folder / info['path'], map_location='cpu', weights_only=True)
            state, optimizer = cp['model_state_dict'], cp['optimizer_state_dict']
            prefix_step = prefix + 'snapshot' + str(step) + ':'
            checks[prefix_step + 'file_and_actual_state_hash'] = sha(folder / info['path']) == info['sha256'] and tensor_hash(state) == info['model_state_sha256']
            checks[prefix_step + 'correct_identity'] = cp['step'] == step and cp['variant'] == variant and cp['radius'] == 2 and cp['plan_sha256'] == run['plan_sha256']
            checks[prefix_step + 'bn_actual_buffers_frozen'] = bn_hash(state) == run['bn_initial_sha256']
            group = optimizer['param_groups']
            checks[prefix_step + 'optimizer_group_fixed'] = len(group) == 1 and group[0]['lr'] == group[0]['weight_decay'] == 1e-4 and tuple(group[0]['betas']) == (.9,.999) and group[0]['eps'] == 1e-8
            optim_steps = [int(v['step']) for v in optimizer['state'].values()]
            checks[prefix_step + 'optimizer_update_count'] = (not optim_steps if step == 0 else len(optim_steps) == len(initial['parameter_names']) and set(optim_steps) == {step})
            if step == 0: states0[variant] = state
            if step == 512: final_states[variant] = state
            snapshot_summaries.append({'step': step, 'state_sha256': info['model_state_sha256'], 'optimizer_tensor_count': len(optim_steps), 'optimizer_step': None if not optim_steps else sorted(set(optim_steps))})
        checks[prefix + 'final_is512_samebytes_hardlink'] = sha(folder / 'final_model.pth') == sha(folder / 'snapshot_512.pth') and (folder / 'final_model.pth').stat().st_ino == (folder / 'snapshot_512.pth').stat().st_ino
        checks[prefix + 'initial_actual_state_matches_receipt'] = tensor_hash(states0[variant]) == initial['model_state_sha256'] == run['initial_model_state_sha256']
        checks[prefix + 'final_actual_state_matches_receipt'] = tensor_hash(final_states[variant]) == run['final_model_state_sha256']
        diff = load_json(folder / 'PARAMETER_DIFF.json')
        actual_diffs = {name: int(torch.count_nonzero(final_states[variant][name] - states0[variant][name])) for name in initial['parameter_names']}
        checks[prefix + 'parameter_changes_recomputed'] = all(count == diff['per_parameter'][name]['changed_elements'] for name,count in actual_diffs.items())
        checks[prefix + 'all_trainable_parameters_updated'] = all(count > 0 for count in actual_diffs.values())
        checks[prefix + 'all_state_finite'] = all(torch.isfinite(tensor).all().item() for tensor in final_states[variant].values())
        zero_positive_rows = [r for r in rows if r['pure_negative_windows'] == 4]
        checks[prefix + '16_all_empty_batches_now_nonzero_gradient'] = len(zero_positive_rows) == 16 and all(r['grad_norm'] > 0 for r in zero_positive_rows)
        if variant == 'E0':
            checks[prefix + 'original_architecture_no_added_keys'] = run['parameter_count'] == 70913 and run['new_parameters'] == 0 and not any(k.startswith('temporal_aggregation') for k in states0[variant])
            checks[prefix + 'no_module_telemetry'] = all(r['module_stats'] == r['module_gradients'] == {} for r in rows)
        else:
            module_names = ['temporal_aggregation.' + x for x in ['null_logit','descriptor.weight','output_projection.weight']]
            checks[prefix + 'expected_new_parameters'] = run['parameter_count'] == 72194 and run['new_parameters'] == 1281 and all(k in states0[variant] for k in module_names)
            checks[prefix + 'output_projection_initial_exact_zero'] = not bool(torch.count_nonzero(states0[variant]['temporal_aggregation.output_projection.weight']))
            first = rows[0]['module_gradients']
            checks[prefix + 'initial_output_gradient_nonzero'] = first['temporal_aggregation.output_projection.weight']['norm'] > 0
            checks[prefix + 'initial_descriptor_null_gradient_zero_expected'] = first['temporal_aggregation.descriptor.weight']['norm'] == first['temporal_aggregation.null_logit']['norm'] == 0
            checks[prefix + 'descriptor_null_gradient_start_after_first'] = all(any(r['module_gradients'][name]['norm'] > 0 for r in rows[1:]) for name in module_names[:2])
            checks[prefix + 'all_logged_module_gradients_finite'] = all(all(r['module_gradients'][name]['present'] and math.isfinite(r['module_gradients'][name]['norm']) for name in module_names) for r in rows)
            for lo, hi in [(1,128),(129,256),(257,384),(385,512)]:
                selected = rows[lo-1:hi]
                for key in ['null_probability','nonnull_mass','correction_relative_norm','raw_feature_norm']:
                    bins.append({'variant': variant, 'step_start': lo, 'step_end': hi, 'measure': key, **summary([r['module_stats'][key] for r in selected])})
        details[variant] = {'positive_windows': len(positives), 'empty_windows': len(negatives),
            'all_empty_batches': len(zero_positive_rows), 'changed_parameter_tensors': sum(count > 0 for count in actual_diffs.values()),
            'parameter_tensors': len(actual_diffs), 'bn_sha256': run['bn_final_sha256'],
            'initial_original_state_sha256': run['initial_original_state_sha256'],
            'final_checkpoint_sha256': run['final_checkpoint_sha256'], 'snapshots': snapshot_summaries,
            'loss_summary': summary([r['loss'] for r in rows]),
            'gradient_summary': summary([r['grad_norm'] for r in rows]),
            'all_empty_batch_gradient_summary': summary([r['grad_norm'] for r in zero_positive_rows]),
            'module_first': rows[0]['module_stats'], 'module_last': rows[-1]['module_stats'],
            'module_first_gradient': rows[0]['module_gradients'], 'module_second_gradient': rows[1]['module_gradients']}
        resource.append({'variant': variant, 'parameters': run['parameter_count'], 'new_parameters': run['new_parameters'],
            'duration_seconds': run['duration_seconds'], 'peak_allocated_bytes': run['max_cuda_memory_allocated_bytes'],
            'peak_reserved_bytes': run['max_cuda_memory_reserved_bytes']})
    for field in ['initial_original_state_sha256','plan_sha256','selection_sha256','cumulative_input_sha256','cumulative_label_sha256','effective_supervision_config','label_config','training_sequences']:
        checks['cross:' + field] = runs['E0'][field] == runs['E1'][field] == runs['E2'][field]
    checks['cross:E1_E2_initial_full_state_bitwise_identical'] = set(states0['E1']) == set(states0['E2']) and all(torch.equal(states0['E1'][name],states0['E2'][name]) for name in states0['E1'])
    checks['cross:E0_original_tensors_equal_both_added_models'] = all(torch.equal(tensor,states0['E1'][name]) and torch.equal(tensor,states0['E2'][name]) for name,tensor in states0['E0'].items())
    checks['cross:branch_hash_identical_and_bound'] = runs['E1']['branch_initial_sha256'] == runs['E2']['branch_initial_sha256'] == tensor_hash({k.removeprefix('temporal_aggregation.'):v for k,v in states0['E1'].items() if k.startswith('temporal_aggregation.')})
    for info in container_receipts['containers']:
        checks['container_log:' + info['name']] = sha(ROOT / info['log_path']) == info['log_sha256']
    with (ROOT / 'TRAINING_MODULE_BINS.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(bins[0]));writer.writeheader();writer.writerows(bins)
    report = {'status': 'PASS' if all(checks.values()) else 'FAIL', 'checks': checks,
        'check_count': len(checks), 'failed_checks': [k for k,v in checks.items() if not v],
        'fingerprint_rows_verified': 1536, 'window_exposures_per_variant': 2048,
        'details': details, 'resource': resource, 'module_bins': bins,
        'source_sha256': sha(__file__), 'freeze_sha256': sha(ROOT / 'PROTOCOL_FREEZE.json'),
        'container_receipt_sha256': sha(ROOT / 'TRAINING_CONTAINER_RECEIPTS.json'),
        'limits': ['Read-only audit by runner author, not an independent data regeneration.',
            'Loss and association telemetry do not establish detection improvement or reliable correspondence.',
            'Non-null mutual mass includes both forward rejection and reverse multiplication; small mass is not by itself learned null rejection.',
            'Snapshot inference is diagnostic only; the frozen main endpoint remains step512.']}
    (ROOT / 'TRAINING_AUDIT.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    lines = ['# 训练执行审计', '', f"状态：**{report['status']}**；{len(checks)}项核查；正式训练仅E0/E1/E2各一次。", '',
        '此审计只读配置、日志、检查点与容器状态，不调用模型前向或优化器，不选择开发集检查点。由运行器作者执行，区别于独立实现测试。', '',
        '## 曝光与监督', '',
        '- 三组共1536次更新的输入SHA及标签SHA逐项等于旧近似控制；每组2048次窗口曝光，385个空标签窗。',
        '- 有目标窗的实际损失等于原官方masked Soft-IoU；仅空标签窗改为已有top-1%背景负项，四窗仍等权平均。',
        '- 原先16个全空batch在本轮均获得非零总参数梯度；这证明干预发生，不证明检测取舍改善。',
        '- 所有原模型初始张量相同；E1/E2包含新增分支的完整初始状态逐张量完全相同。',
        '- BN运行缓存所有日志节点及五个实际检查点均保持发布值；所有原参数及新增参数可更新。', '',
        '## 训练资源与参数', '', '| 组别 | 参数 | 耗时秒 | 峰值分配GiB | 实际更新参数张量 |', '|---|---:|---:|---:|---:|']
    for row in resource:
        d=details[row['variant']]
        lines.append(f"| {row['variant']} | {row['parameters']} | {row['duration_seconds']:.2f} | {row['peak_allocated_bytes']/2**30:.3f} | {d['changed_parameter_tensors']}/{d['parameter_tensors']} |")
    lines += ['', '固定保存0、128、256、384、512节点；Adam实际状态步数与节点一致。`final_model.pth`与512快照为同一硬链接。主结果继续使用512步。', '',
        '## 新分支遥测', '', '下表是最后128次更新的均值；完整四个固定区间见[TRAINING_MODULE_BINS.csv](TRAINING_MODULE_BINS.csv)。', '',
        '| 组别 | 空选项概率 | 非空权重总量 | 修正/原特征范数 |', '|---|---:|---:|---:|']
    for variant in ['E1','E2']:
        b={r['measure']:r['mean'] for r in bins if r['variant']==variant and r['step_start']==385}
        lines.append(f"| {variant} | {b['null_probability']:.8f} | {b['nonnull_mass']:.8f} | {b['correction_relative_norm']:.8f} |")
    lines += ['', '第一步输出投影获得梯度，描述子与空选项梯度为0符合零输出投影设计；后续两者均获得非零梯度。', '',
        '**空选项概率与非空权重总量不是同一个量。** E2的双向乘积会自行降低权重总量，不能把较小的非空质量直接解释为学会了拒绝错误参考。检测与mass/current诊断结果需要单独判定。', '',
        '本审计不从训练loss、梯度、空选项或修正范数推断检出能力。冻结源代码、协议和历史实验文件均未修改。', '',
        '[机器可读核查](TRAINING_AUDIT.json) · [正式容器记录](TRAINING_CONTAINER_RECEIPTS.json)', '']
    (ROOT / 'TRAINING_AUDIT.md').write_text('\n'.join(lines))
    print(json.dumps({'status': report['status'], 'checks': len(checks), 'failed': report['failed_checks'], 'resource': resource}), flush=True)
    if report['status'] != 'PASS': raise SystemExit(1)


if __name__ == '__main__':
    main()
