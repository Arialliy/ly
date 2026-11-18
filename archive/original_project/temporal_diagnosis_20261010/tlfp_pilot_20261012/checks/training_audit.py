"""Independent CPU audit of completed, immutable TLFP S/T training artifacts.

Imports no project train/model/loss helpers. Does not run models or inspect dev
predictions. Requires both completed fixed endpoints, otherwise exits 2 without
emitting a misleading final audit. A negative alpha is allowed by the protocol.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

BASE = Path(__file__).resolve().parents[1]
Q = BASE.parent
STEPS = (0, 64, 128, 256, 512)
RELEASE_SHA = '4fd4aa2736d9a33f2bf1459099068d8b4a7e202ff80aecdb4a3277fd13c7abcc'
PLAN_SHA = '6f86d929ad1275bbb120456dd23a25c7ea9571517f60b4c64a9d5ce6413c975c'
PREFIX = 'feature_refiner.'
BRANCH_KEYS = {'scale', 'operator.sigma', 'operator.attention.weight'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def state_hash(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def bn_hash(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        if not key.endswith(('running_mean', 'running_var', 'num_batches_tracked')):
            continue
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def load(path):
    # Exact old official checkpoint has NumPy scalar metadata, not executable
    # classes. Preserve weights_only and explicitly allow its numeric types.
    core = np._core if hasattr(np, '_core') else np.core
    numeric = [(core.multiarray.scalar, 'numpy.core.multiarray.scalar'),
               np.dtype, type(np.dtype(np.float64))]
    with torch.serialization.safe_globals(numeric):
        return torch.load(path, map_location='cpu', weights_only=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=BASE / 'checks/TRAINING_AUDIT.json')
    args = parser.parse_args()
    paths = {v: BASE / 'training' / v / 'run' for v in ('S', 'T')}
    if any(not (p / 'RUN.json').exists() or read(p / 'RUN.json').get('status') != 'complete'
           for p in paths.values()):
        print(json.dumps({'status': 'NOT_READY', 'reason': 'Both S/T must be complete'}))
        raise SystemExit(2)
    torch.set_num_threads(2)
    checks = collections.Counter()
    issues = []

    def require(condition, category, detail):
        checks[category] += 1
        if not bool(condition):
            issues.append({'category': category, 'detail': str(detail)})

    freeze = read(BASE / 'PROTOCOL_FREEZE.json')
    require(sha(BASE / 'PROTOCOL.md') == freeze['protocol_sha256'], 'frozen_sources', 'protocol')
    for name, digest in freeze['code_sha256'].items():
        path = Path(name) if Path(name).is_absolute() else BASE / name
        require(path.is_file() and sha(path) == digest, 'frozen_sources', name)
    old_path = Q / 'mutual_temporal_pilot_20261012/training/E0/run/TRAIN.jsonl'
    old = [json.loads(line) for line in old_path.read_text().splitlines()]
    require(len(old) == 512, 'exposure', 'historical E0 row count')
    plan_path = Q / 'stage1_goal/pilot/TRAIN_PLAN.json'
    plan = read(plan_path)
    require(sha(plan_path) == PLAN_SHA, 'exposure', 'fixed plan SHA')
    require([e['step'] for e in plan['entries']] == list(range(512)), 'exposure', 'plan steps')
    init_meta = read(BASE / 'init/INITIALIZATION.json')
    initial = {}
    for name in ('COMMON_SPATIAL', 'S', 'T'):
        path = BASE / 'init' / (name + '.pth')
        require(sha(path) == init_meta['files'][path.name], 'initialization', name + ' file SHA')
        checkpoint = load(path)
        require(checkpoint['seed'] == 46, 'initialization', name + ' seed')
        initial[name] = checkpoint['adapter_state_dict']
        require(set(initial[name]) == BRANCH_KEYS, 'initialization', name + ' state keys')
    for key in BRANCH_KEYS:
        require(torch.equal(initial['S'][key], initial['COMMON_SPATIAL'][key]), 'initialization', 'S common ' + key)
    for key in ('scale', 'operator.sigma'):
        require(torch.equal(initial['S'][key], initial['T'][key]), 'initialization', 'S/T ' + key)
    require(initial['S']['scale'].item() == 0, 'initialization', 'zero residual gate')
    ks = initial['S']['operator.attention.weight']
    kt = initial['T']['operator.attention.weight']
    require(tuple(ks.shape) == (1, 2, 7, 7) and tuple(kt.shape) == (1, 6, 7, 7), 'initialization', 'kernel shapes')
    require(torch.equal(ks, kt[:, 2:4]), 'initialization', 'T center equals S')
    require(torch.count_nonzero(kt[:, :2]) == 0 and torch.count_nonzero(kt[:, 4:]) == 0, 'initialization', 'T neighbor kernels zero')

    runs = {v: read(path / 'RUN.json') for v, path in paths.items()}
    release_path = Path(runs['S']['checkpoint_path'])
    require(sha(release_path) == RELEASE_SHA, 'frozen_original', 'released checkpoint SHA')
    released = load(release_path)['model_state_dict']
    release_hash = state_hash(released)
    released_bn_hash = bn_hash(released)
    bn_keys = [k for k in released if k.endswith(('running_mean', 'running_var', 'num_batches_tracked'))]
    require(sum(v.numel() for k, v in released.items() if k not in bn_keys) == 70913,
            'frozen_original', '70913 original parameters')
    log_rows = {}
    summaries = {}
    source_keys = {
        'train_source_sha256': BASE / 'code/train_tlfp.py',
        'module_source_sha256': BASE / 'code/temporal_lfp.py',
        'label_source_sha256': Q / 'supervision_alignment_control_20261012/code/gaussian_labels.py',
        'top1_loss_source_sha256': Q / 'top1_background_control_20261011/code/losses.py',
        'common_loss_source_sha256': Q / 'mutual_temporal_pilot_20261012/code/train_mutual.py',
        'prior_train_source_sha256': Q / 'supervision_alignment_control_20261012/code/train_supervision.py',
        'batch_source_sha256': Q / 'top1_background_control_20261011/code/train_pilot.py',
        'spatial_reference_sha256': Q / 'route_closeout_20261012/haar_reference.py',
        'adapter_reference_sha256': Q / 'route_closeout_20261012/framewise_adapter.py',
    }
    telemetry_steps = {1, 2, *range(16, 513, 16)}
    for variant, path in paths.items():
        run = runs[variant]
        rows = [json.loads(line) for line in (path / 'TRAIN.jsonl').read_text().splitlines()]
        log_rows[variant] = rows
        require(len(rows) == 512 and [r['step'] for r in rows] == list(range(1, 513)), 'exposure', variant + ' 512 sequential rows')
        expected_meta = {'condition': variant, 'seed': 46, 'steps': 512, 'final_step': 512,
                         'lr': 1e-4, 'weight_decay': 1e-4, 'betas': [.9, .999], 'eps': 1e-8,
                         'precision': 'FP32_no_AMP_no_TF32', 'trainable_parameters': 100 if variant == 'S' else 296,
                         'plan_sha256': PLAN_SHA, 'checkpoint_sha256': RELEASE_SHA,
                         'training_sequences': plan['train_sequences'], 'frozen_original_unchanged': True}
        for key, value in expected_meta.items():
            require(run[key] == value, 'run_metadata', variant + '/' + key)
        require(run['old_log_sha256'] == sha(old_path), 'exposure', variant + ' old log identity')
        require(sha(path / 'TRAIN.jsonl') == run['train_log_sha256'], 'run_metadata', variant + ' log SHA')
        require(run['original_initial_state_sha256'] == run['original_final_state_sha256'] == release_hash,
                'frozen_original', variant + ' state SHA')
        require(run['bn_initial_sha256'] == run['bn_final_sha256'] == released_bn_hash,
                'frozen_original', variant + ' BN SHA')
        for key, source in source_keys.items():
            require(run[key] == sha(source), 'frozen_sources', variant + '/' + key)
        require(run['model_source_sha256'] == sha(Path(run['criterion_source_path'])), 'frozen_sources', variant + ' model source')
        inp_chain, lab_chain = hashlib.sha256(), hashlib.sha256()
        for row, reference in zip(rows, old):
            step = row['step']
            where = variant + '/' + str(step)
            for key in ('input_sha256', 'label_sha256', 'label_stats', 'positive_pixels_per_window',
                        'valid_pixels_per_window', 'background_pixels_per_window', 'pure_negative_windows'):
                require(row[key] == reference[key], 'exposure', where + '/' + key)
            inp_chain.update(row['input_sha256'].encode())
            lab_chain.update(row['label_sha256'].encode())
            require(row['finite'] and math.isfinite(row['loss']) and math.isfinite(row['grad_norm']), 'logs', where + ' finite')
            require(row['original_parameters_with_gradient'] == 0 and row['parameters_with_gradient'] == 3, 'frozen_original', where + ' gradient count')
            require(row['bn_running_sha256'] == released_bn_hash, 'frozen_original', where + ' BN log')
            require(row['lr'] == 1e-4, 'optimizer', where + ' lr')
            losses = row['loss_per_window']
            positive = row['positive_pixels_per_window']
            neg = row['empty_window_negative_per_window']
            require(len(losses) == len(positive) == len(neg) == 4, 'loss_accounting', where + ' four windows')
            require(math.isclose(sum(losses) / 4, row['loss'], abs_tol=1e-6, rel_tol=2e-6), 'loss_accounting', where + ' outer mean')
            for wi in range(4):
                if positive[wi]:
                    require(losses[wi] == row['original_official_loss_per_window'][wi] and neg[wi] == 0
                            and row['selected_negative_pixels_per_window'][wi] == 0,
                            'loss_accounting', where + ' positive window ' + str(wi))
                else:
                    expected_k = max(1, math.ceil(.01 * row['background_pixels_per_window'][wi]))
                    require(losses[wi] == neg[wi] and row['selected_negative_pixels_per_window'][wi] == expected_k
                            and row['valid_pixels_per_window'][wi] == row['background_pixels_per_window'][wi],
                            'loss_accounting', where + ' empty window ' + str(wi))
            require(math.isclose(row['positive_loss_weighted_contribution'], sum(v for v, n in zip(losses, positive) if n) / 4, abs_tol=1e-12)
                    and math.isclose(row['negative_loss_weighted_contribution'], sum(neg) / 4, abs_tol=1e-12),
                    'loss_accounting', where + ' contribution split')
            require(('module_gradients' in row) == (step in telemetry_steps), 'logs', where + ' telemetry cadence')
            if step in telemetry_steps:
                require(set(row['module_gradients']) == BRANCH_KEYS and set(row['module_updates']) == BRANCH_KEYS,
                        'gradients', where + ' all three parameters logged')
                require(all(g['present'] and math.isfinite(g['norm']) for g in row['module_gradients'].values()),
                        'gradients', where + ' finite present gradients')
        require(run['cumulative_input_sha256'] == inp_chain.hexdigest() and run['cumulative_label_sha256'] == lab_chain.hexdigest(),
                'exposure', variant + ' recomputed hash chains')
        require(sum(r['pure_negative_windows'] for r in rows) == 385 and sum(r['pure_negative_windows'] == 4 for r in rows) == 16,
                'exposure', variant + ' empty exposure totals')
        require(rows[0]['module_gradients']['scale']['nonzero'] == 1
                and rows[0]['module_gradients']['operator.sigma']['nonzero'] == 0
                and rows[0]['module_gradients']['operator.attention.weight']['nonzero'] == 0,
                'gradients', variant + ' alpha-only first direct gradient')
        activation = {}
        for key in BRANCH_KEYS:
            nonzero = [r['step'] for r in rows if r['step'] > 1 and 'module_gradients' in r
                       and r['module_gradients'][key]['nonzero'] > 0]
            activation[key] = min(nonzero) if nonzero else None
            require(bool(nonzero), 'gradients', variant + '/' + key + ' eventually nonzero after first step')
        initial_record = read(path / 'INITIAL_STATE.json')
        require(set(initial_record['trainable_parameter_names']) == {PREFIX + k for k in BRANCH_KEYS},
                'optimizer', variant + ' only branch parameter names')
        snapshots = run['snapshots']
        require([x['step'] for x in snapshots] == list(STEPS), 'snapshots', variant + ' exact saved schedule')
        snapshot_summaries = []
        branch_final = None
        for item in snapshots:
            step = item['step']
            snap_path = path / item['path']
            require(sha(snap_path) == item['sha256'], 'snapshots', variant + '/' + str(step) + ' file SHA')
            cp = load(snap_path)
            require(cp['step'] == step and cp['condition'] == variant, 'snapshots', variant + '/' + str(step) + ' identity')
            state = cp['model_state_dict']
            original = {k[len('original.'):]: v for k, v in state.items() if k.startswith('original.')}
            branch = {k[len(PREFIX):]: v for k, v in state.items() if k.startswith(PREFIX)}
            require(set(original) == set(released) and set(branch) == BRANCH_KEYS,
                    'frozen_original', variant + '/' + str(step) + ' exact state keys')
            for key, value in released.items():
                require(torch.equal(original[key], value), 'frozen_original', variant + '/' + str(step) + '/' + key)
            require(state_hash(state) == item['model_state_sha256'], 'snapshots', variant + '/' + str(step) + ' tensor hash')
            require(all(bool(torch.isfinite(v).all()) for v in state.values()), 'snapshots', variant + '/' + str(step) + ' finite tensors')
            optimizer = cp['optimizer_state_dict']
            require(len(optimizer['param_groups']) == 1 and len(optimizer['param_groups'][0]['params']) == 3,
                    'optimizer', variant + '/' + str(step) + ' three optimized tensors')
            group = optimizer['param_groups'][0]
            require(group['lr'] == group['weight_decay'] == 1e-4 and tuple(group['betas']) == (.9, .999)
                    and group['eps'] == 1e-8, 'optimizer', variant + '/' + str(step) + ' exact hyperparameters')
            require(len(optimizer['state']) == (0 if step == 0 else 3), 'optimizer', variant + '/' + str(step) + ' state count')
            require(all(int(x['step']) == step for x in optimizer['state'].values()), 'optimizer', variant + '/' + str(step) + ' Adam update count')
            if step == 0:
                for key in BRANCH_KEYS:
                    require(torch.equal(branch[key], initial[variant][key]), 'initialization', variant + ' snapshot0 ' + key)
                require(state_hash(state) == initial_record['model_state_sha256'], 'initialization', variant + ' snapshot0 recorded hash')
            for key, source in source_keys.items():
                require(cp[key] == sha(source), 'frozen_sources', variant + '/' + str(step) + '/' + key)
            weight = branch['operator.attention.weight']
            snapshot_summaries.append({'step': step, 'alpha': float(branch['scale']),
                                      'tanh_alpha': float(branch['scale'].tanh()), 'sigma': float(branch['operator.sigma']),
                                      'center_kernel_norm': float((weight if variant == 'S' else weight[:, 2:4]).norm()),
                                      'previous_kernel_norm': None if variant == 'S' else float(weight[:, :2].norm()),
                                      'next_kernel_norm': None if variant == 'S' else float(weight[:, 4:].norm()),
                                      'original_tensor_count_compared': len(released), 'BN_buffer_count_compared': len(bn_keys)})
            if step == 512:
                branch_final = branch
                require(sha(path / 'final_model.pth') == sha(snap_path) == run['final_checkpoint_sha256'],
                        'snapshots', variant + ' final is fixed512')
        if branch_final is not None:
            for key in BRANCH_KEYS:
                require(not torch.equal(branch_final[key], initial[variant][key]), 'gradients', variant + ' final changed ' + key)
            w = branch_final['operator.attention.weight']
            if variant == 'T':
                require(torch.count_nonzero(w[:, :2]) > 0 and torch.count_nonzero(w[:, 4:]) > 0,
                        'gradients', 'T both neighboring kernels updated')
        receipts = [read(p) for p in (BASE / 'runtime').glob('train_' + variant + '_*.json')]
        complete = [r for r in receipts if r.get('status') == 'COMPLETE']
        require(len(complete) == 1 and complete[0]['exit_code'] == 0, 'execution', variant + ' one completed training launch')
        if complete:
            require(complete[0]['gpu'] in [1, 2, 3], 'execution', variant + ' authorized GPU')
            require(complete[0]['image'] == 'sha256:eb956873870d737f2b68dff0724b8f6f56f778829d119b0938d29b7abeaa2f92',
                    'execution', variant + ' immutable image')
        summaries[variant] = {'updates': len(rows), 'new_parameters': run['new_parameters'],
                              'duration_seconds': run['duration_seconds'], 'peak_memory_bytes': run['max_cuda_memory_allocated_bytes'],
                              'empty_windows': 385, 'all_empty_batches': 16, 'snapshot_count': len(snapshots),
                              'first_logged_nonzero_direct_gradient_after_step1': activation,
                              'first_step_sigma_direct_gradient': rows[0]['module_gradients']['operator.sigma']['norm'],
                              'first_step_sigma_update_norm': rows[0]['module_updates']['operator.sigma']['norm'],
                              'sigma_step1_note': 'Zero direct gradient does not forbid an Adam weight-decay update; wd=1e-4 is shared and fixed.',
                              'snapshots': snapshot_summaries}
    for s, t in zip(log_rows['S'], log_rows['T']):
        require(s['input_sha256'] == t['input_sha256'] and s['label_sha256'] == t['label_sha256'],
                'paired_exposure', 'S/T step ' + str(s['step']))
    require(runs['S']['label_config'] == runs['T']['label_config']
            and runs['S']['effective_supervision_config'] == runs['T']['effective_supervision_config'],
            'paired_exposure', 'S/T exact supervision metadata')
    result = {'status': 'PASS' if not issues else 'FAIL', 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'scope': 'Independent completed-training artifact audit; no model forward, gradients recomputation, GPU or development results.',
              'audit_source_sha256': sha(__file__), 'protocol_freeze_sha256': sha(BASE / 'PROTOCOL_FREEZE.json'),
              'checks': dict(checks), 'check_count': sum(checks.values()), 'issues': issues, 'conditions': summaries,
              'all_original_states_compared_to_released_checkpoint': True,
              'same_inputs_and_labels_for_all1024updates': not any(i['category'] in ('exposure', 'paired_exposure') for i in issues),
              'interpretation': 'Negative alpha is allowed. Updating kernels and loss-gradient receipt do not establish detection benefit. Logs supply direct-gradient evidence, snapshots supply independent parameter/state evidence.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    lines = ['# S/T 训练独立审计', '', '**' + result['status'] + '**，' + str(result['check_count']) + '项检查。', '',
             '审计直接读取冻结源码哈希、原发布权重、两组全部训练日志和各5个快照，不导入训练/损失/模型实现，不读取开发预测或执行模型。', '',
             '|配置|更新|新增参数|快照|空标签窗口|训练秒数|', '|---|---:|---:|---:|---:|---:|']
    for variant, summary in summaries.items():
        lines.append('|{}|{}|{}|{}|{}|{:.3f}|'.format(variant, summary['updates'], summary['new_parameters'], summary['snapshot_count'], summary['empty_windows'], summary['duration_seconds']))
    lines += ['', '逐步输入/标签与历史E0及另一组一致；每个快照逐张量与发布权重比较原参数和全部BN；初始化核中心/邻核/标量映射、Adam步数与固定512终点独立核验。具体通过/失败以JSON的issues与check_count为准。', '',
              '首步门控之外的直接梯度为零符合初始化设计；sigma仍可能受Adam已有weight decay更新。alpha允许为负，不据此称“抑噪”；核更新及梯度存在也不等于检测有效。', '',
              '本审计没有重新计算反向传播：直接梯度证据来自保存日志，参数更新与原网络不变来自独立读取的检查点。']
    if issues:
        lines += ['', '失败项：', *['- ' + json.dumps(item, ensure_ascii=False) for item in issues]]
    args.output.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'status': result['status'], 'check_count': result['check_count'], 'issues': len(issues), 'output': str(args.output)}))
    raise SystemExit(0 if not issues else 1)


if __name__ == '__main__':
    main()
