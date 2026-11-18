#!/usr/bin/env python3
"""One frozen Gaussian-label / official-SoftLoUloss approximation control.

Reuse the released architecture, checkpoint, and exact 512-step crop plan.
Only the supervision is replaced. This is not the official full training recipe.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time

Q = Path('/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/temporal_diagnosis_20261010')
OLD = Q / 'top1_background_control_20261011'
CONDITION = 'gaussian_softiou_approx'
EXPECTED = {
    'plan': '6f86d929ad1275bbb120456dd23a25c7ea9571517f60b4c64a9d5ce6413c975c',
    'selection': '1f5e617764c0fe27a63440cc5b26d0e27a05db7dbc8c997b0bc5fd74a4b39c92',
    'checkpoint': '4fd4aa2736d9a33f2bf1459099068d8b4a7e202ff80aecdb4a3277fd13c7abcc',
    'original_train_source': '75e4fe9b4115af837cc1769471328a0b88fca8e546ec8d6f7db73d580f460b54',
    'model_source': 'fa46a034a30d8bbb0f677271d6368f033fcb389d682478d331551d8bae087668',
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def tensor_hash(items):
    digest = hashlib.sha256()
    for name, value in sorted(items):
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def data_hash(batch, positives, masks, specs):
    """Exactly the same input fingerprint as the completed top-1%/BC pilots."""
    digest = hashlib.sha256()
    digest.update(batch.tobytes())
    digest.update(masks.tobytes())
    digest.update(json.dumps([positives, specs], sort_keys=True, separators=(',', ':')).encode())
    return digest.hexdigest()


def label_hash(targets, valid):
    digest = hashlib.sha256()
    for name, value in [('targets', targets), ('valid', valid)]:
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def masked_official_loss(logits, targets, valid, criterion):
    """Gather valid pixels, call the unmodified official criterion per window.

    SoftLoUloss performs sigmoid internally and sums axes (1, 2, 3). The
    original [B,T,H,W] input therefore computes one spatiotemporal IoU per
    window. Gathering to [1,1,1,N_w] preserves that exact reduction and lets
    each window have a different number of valid pixels. No added smoothing,
    reweighting, BCE, hard mining, MIL, paired loss, or reference loss.
    """
    import torch
    if logits.ndim != 4 or targets.shape != logits.shape or valid.shape != logits.shape:
        raise ValueError('Loss expects matching [B,T,H,W] tensors')
    if valid.dtype != torch.bool:
        raise TypeError('The validity mask must be boolean')
    values = []
    valid_counts, positive_counts, target_masses = [], [], []
    for window in range(logits.shape[0]):
        window_valid = valid[window]
        count = int(window_valid.sum())
        if count == 0:
            raise ValueError('An empty valid window has no defined official SoftIoU loss')
        selected_logits = logits[window][window_valid].reshape(1, 1, 1, -1)
        selected_targets = targets[window][window_valid].reshape(1, 1, 1, -1)
        values.append(criterion(selected_logits, selected_targets))
        valid_counts.append(count)
        positive_counts.append(int(torch.count_nonzero(selected_targets)))
        target_masses.append(float(selected_targets.detach().sum()))
    loss = torch.stack(values).mean()
    return loss, {
        'loss_per_window': [float(value.detach()) for value in values],
        'valid_pixels_per_window': valid_counts,
        'positive_pixels_per_window': positive_counts,
        'target_mass_per_window': target_masses,
        'pure_negative_windows': sum(count == 0 for count in positive_counts),
        'valid_pixels': sum(valid_counts), 'positive_pixels': sum(positive_counts),
        'target_mass': sum(target_masses),
    }


def validate_inputs(args):
    original_train = OLD / 'code/train_pilot.py'
    original_model = args.checkpoint.parent.parent / 'DeepPro-Plus.py'
    for label, path in [('plan', args.plan), ('selection', args.selection),
                        ('checkpoint', args.checkpoint), ('original_train_source', original_train),
                        ('model_source', original_model)]:
        if sha(path) != EXPECTED[label]:
            raise ValueError(f'Frozen {label} hash mismatch: {path}')
    base = load_module('supervision_frozen_training_base', original_train)
    plan = json.loads(args.plan.read_text())
    selection = json.loads(args.selection.read_text())
    if (plan['config'] != base.CONFIG or plan['steps'] != 512 or plan['seed'] != 46
            or len(plan['entries']) != 512):
        raise ValueError('Frozen plan configuration mismatch')
    if plan['selection_sha256'] != EXPECTED['selection']:
        raise ValueError('Plan selection fingerprint mismatch')
    if args.plan.with_suffix(args.plan.suffix + '.sha256').read_text().strip() != EXPECTED['plan']:
        raise ValueError('Plan SHA sidecar mismatch')
    names = base.sequence_names(selection)
    if names != plan['train_sequences']:
        raise ValueError('Training sequence list changed')
    sequences = {name: base.SequenceData(args.dataset_root, name) for name in names}
    if plan['data_fingerprints'] != {name: seq.fingerprint for name, seq in sequences.items()}:
        raise ValueError('Training data fingerprint mismatch')
    return base, plan, names, sequences, original_model


def train(args):
    import numpy as np
    import torch
    from gaussian_labels import build_targets, LABEL_CONFIG

    base, plan, names, sequences, original_model_file = validate_inputs(args)
    label_source = Path(sys.modules[build_targets.__module__].__file__)
    source_hashes = {'train_source_sha256': sha(__file__),
                     'label_source_sha256': sha(label_source),
                     'model_source_sha256': sha(original_model_file)}
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('A new empty output directory is required')
    args.output.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(46)
    np.random.seed(46)
    torch.manual_seed(46)
    torch.cuda.manual_seed_all(46)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)

    model, loaded_model_file = base.load_model(args.source, args.checkpoint)
    if loaded_model_file.resolve() != original_model_file.resolve():
        raise AssertionError('Loaded model source differs from validated source')
    # Obtain the real class from the same checkpoint-adjacent source. Loading
    # this module defines classes only; no new model or parameters are created.
    official = load_module('supervision_official_criterion_source', original_model_file)
    criterion = official.SoftLoUloss()
    if list(criterion.parameters()):
        raise AssertionError('Official criterion unexpectedly has trainable parameters')
    initial_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    initial_parameter_names = sorted(dict(model.named_parameters()))
    initial_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if initial_parameter_count != 70913:
        raise AssertionError(f'Original architecture parameter count changed: {initial_parameter_count}')
    save_json(args.output / 'INITIAL_STATE.json', {
        'checkpoint_sha256': EXPECTED['checkpoint'],
        'model_source_sha256': EXPECTED['model_source'],
        'model_state_sha256': tensor_hash(initial_state.items()),
        'state_tensor_sha256': {name: tensor_hash([(name, value)]) for name, value in initial_state.items()},
        'parameter_names': initial_parameter_names, 'parameter_count': initial_parameter_count,
    })
    device = torch.device(args.device)
    model = model.to(device=device, dtype=torch.float32)
    criterion = criterion.to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
    initial_bn = base.bn_snapshot(model)
    bn_hash = base.buffer_hash(initial_bn)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    metadata = {
        'status': 'running', 'condition': CONDITION, 'variant': CONDITION, 'steps': 512, 'seed': 46,
        'lr': 1e-4, 'optimizer': 'fresh Adam', 'betas': [0.9, 0.999],
        'eps': 1e-8, 'weight_decay': 1e-4, 'precision': 'FP32_no_AMP_no_TF32',
        'frozen_plan_config': base.CONFIG, 'label_config': LABEL_CONFIG,
        'criterion': 'checkpoint-adjacent unmodified SoftLoUloss; internal sigmoid; smooth=0',
        'loss_reduction': 'boolean gather valid pixels per window; [1,1,1,N_w]; four-window equal mean',
        'pure_negative_window_policy': 'official L=1 and zero gradient; counted without replacement loss',
        'plan_sha256': EXPECTED['plan'], 'selection_sha256': EXPECTED['selection'],
        'checkpoint_sha256': EXPECTED['checkpoint'], **source_hashes,
        'original_train_source_sha256': EXPECTED['original_train_source'],
        'initial_model_state_sha256': tensor_hash(initial_state.items()),
        'bn_initial_sha256': bn_hash, 'bn_policy': 'eval_running_stats_affine_trainable',
        'parameter_count': initial_parameter_count, 'new_parameters': 0,
        'torch_version': str(torch.__version__), 'device': str(device),
        'checkpoint_path': str(args.checkpoint.resolve()), 'source_path': str(args.source.resolve()),
        'criterion_source_path': str(original_model_file.resolve()),
        'label_source_path': str(label_source.resolve()), 'training_sequences': names,
        'protocol': 'Gaussian_box_label_official_SoftIoU_approximation_not_official_training_reproduction',
        'checkpoint_selection': 'fixed final update512; no best checkpoint',
    }
    save_json(args.output / 'RUN.json', metadata)
    started = time.perf_counter()
    counts = Counter()
    cumulative_data_hash = hashlib.sha256()
    cumulative_label_hash = hashlib.sha256()
    try:
        with (args.output / 'TRAIN.jsonl').open('w', buffering=1) as log:
            for step, entry in enumerate(plan['entries']):
                if entry['step'] != step:
                    raise ValueError('Frozen step order changed')
                for module in model.modules():
                    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                        module.eval()
                batch, positives, masks, specs = base.build_batch(entry, sequences)
                targets_np, valid_np, label_stats = build_targets(entry, sequences, background_masks=masks)
                expected_shape = (4, 40, 128, 128)
                if (targets_np.shape != expected_shape or targets_np.dtype != np.float32
                        or valid_np.shape != expected_shape or valid_np.dtype != np.bool_):
                    raise ValueError('Label API must return float32 targets and bool valid [4,40,128,128]')
                if not np.isfinite(targets_np).all() or not np.all((targets_np == 0) | (targets_np == 1)):
                    raise ValueError('Frozen official-loader label approximation must be finite binary float32')
                input_hash = data_hash(batch, positives, masks, specs)
                target_hash = label_hash(targets_np, valid_np)
                cumulative_data_hash.update(input_hash.encode())
                cumulative_label_hash.update(target_hash.encode())
                x = torch.from_numpy(batch).to(device)
                targets = torch.from_numpy(targets_np).to(device)
                valid = torch.from_numpy(valid_np).to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)[1]
                if logits.shape != expected_shape:
                    raise ValueError(f'Unexpected output shape {tuple(logits.shape)}')
                loss, parts = masked_official_loss(logits, targets, valid, criterion)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f'Nonfinite official loss at step {step + 1}')
                loss.backward()
                gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
                if not gradients or not all(bool(torch.isfinite(gradient).all()) for gradient in gradients):
                    raise FloatingPointError(f'Nonfinite or missing parameter gradient at step {step + 1}')
                norm = float(torch.linalg.vector_norm(torch.stack([gradient.detach().norm() for gradient in gradients])))
                row = {
                    'step': step + 1, 'condition': CONDITION, 'variant': CONDITION, 'loss': float(loss.detach()),
                    'grad_norm': norm, 'parameters_with_gradient': len(gradients),
                    'finite': True, 'input_sha256': input_hash, 'label_sha256': target_hash,
                    'label_stats': label_stats, **parts,
                }
                optimizer.step()
                current_bn = base.bn_snapshot(model)
                if current_bn.keys() != initial_bn.keys() or any(not torch.equal(current_bn[key], value)
                                                                 for key, value in initial_bn.items()):
                    raise AssertionError('Original BN running buffers changed')
                row['bn_running_sha256'] = base.buffer_hash(current_bn)
                row['max_cuda_memory_allocated_bytes'] = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0
                row['max_cuda_memory_reserved_bytes'] = torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else 0
                counts['pure_negative_windows'] += parts['pure_negative_windows']
                counts['valid_pixels'] += parts['valid_pixels']
                counts['positive_pixels'] += parts['positive_pixels']
                row['elapsed_seconds'] = time.perf_counter() - started
                line = json.dumps(row, allow_nan=False)
                log.write(line + '\n')
                print(line, flush=True)
                del logits, loss, x, targets, valid, gradients
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        final_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
        if set(final_state) != set(initial_state) or sorted(dict(model.named_parameters())) != initial_parameter_names:
            raise AssertionError('Original model state or parameter names changed')
        if not all(bool(torch.isfinite(value).all()) for value in final_state.values()):
            raise FloatingPointError('Nonfinite final model state')
        if source_hashes != {'train_source_sha256': sha(__file__),
                             'label_source_sha256': sha(label_source),
                             'model_source_sha256': sha(original_model_file)}:
            raise AssertionError('A frozen source file changed during training')
        differences = {}
        for name in initial_parameter_names:
            delta = final_state[name].double() - initial_state[name].double()
            differences[name] = {'numel': delta.numel(), 'changed_elements': int(torch.count_nonzero(delta)),
                                 'max_absolute_difference': float(delta.abs().max()),
                                 'l2_difference': float(delta.norm())}
        save_json(args.output / 'PARAMETER_DIFF.json', {
            'parameter_count': initial_parameter_count, 'new_parameter_count': 0,
            'changed_parameter_tensors': sum(value['changed_elements'] > 0 for value in differences.values()),
            'parameter_tensors': len(differences), 'per_parameter': differences,
        })
        checkpoint = {
            'model_state_dict': final_state, 'step': 512, 'condition': CONDITION, 'variant': CONDITION,
            'label_config': LABEL_CONFIG, 'plan_sha256': EXPECTED['plan'],
            'initial_checkpoint_sha256': EXPECTED['checkpoint'], **source_hashes,
            'protocol': metadata['protocol'],
        }
        torch.save(checkpoint, args.output / 'final_model.pth')
        metadata.update(status='complete', final_step=512, duration_seconds=time.perf_counter()-started,
                        counts=dict(counts), bn_final_sha256=base.buffer_hash(base.bn_snapshot(model)),
                        final_checkpoint_sha256=sha(args.output / 'final_model.pth'),
                        final_model_state_sha256=tensor_hash(final_state.items()),
                        train_log_sha256=sha(args.output / 'TRAIN.jsonl'),
                        cumulative_input_sha256=cumulative_data_hash.hexdigest(),
                        cumulative_label_sha256=cumulative_label_hash.hexdigest(),
                        parameter_diff_sha256=sha(args.output / 'PARAMETER_DIFF.json'),
                        max_cuda_memory_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0,
                        max_cuda_memory_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else 0)
        save_json(args.output / 'RUN.json', metadata)
        print(json.dumps({'status': 'complete', 'condition': CONDITION, 'output': str(args.output),
                          'duration_seconds': metadata['duration_seconds']}), flush=True)
    except BaseException as error:
        metadata.update(status='failed', error=repr(error), duration_seconds=time.perf_counter()-started)
        save_json(args.output / 'RUN.json', metadata)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['source', 'checkpoint', 'dataset-root', 'selection', 'plan', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--steps', type=int, default=512)
    parser.add_argument('--seed', type=int, default=46)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.steps != 512 or args.seed != 46 or args.lr != 1e-4:
        parser.error('This control freezes --steps 512 --seed 46 --lr 0.0001')
    train(args)


if __name__ == '__main__':
    main()
