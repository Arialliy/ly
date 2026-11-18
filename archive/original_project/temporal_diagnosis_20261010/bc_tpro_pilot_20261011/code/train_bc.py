#!/usr/bin/env python3
"""Frozen 512-update BC-TPro pilot; reuse original data and top-1% loss code."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time

Q = Path('/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/temporal_diagnosis_20261010')
OLD = Q / 'top1_background_control_20261011'
EXPECTED = {
    'plan': '6f86d929ad1275bbb120456dd23a25c7ea9571517f60b4c64a9d5ce6413c975c',
    'selection': '1f5e617764c0fe27a63440cc5b26d0e27a05db7dbc8c997b0bc5fd74a4b39c92',
    'checkpoint': '4fd4aa2736d9a33f2bf1459099068d8b4a7e202ff80aecdb4a3277fd13c7abcc',
    'train_source': '75e4fe9b4115af837cc1769471328a0b88fca8e546ec8d6f7db73d580f460b54',
    'loss_source': '6e7021b9bd7cc362988473449596eb92212312384d51704343928af120af7ed1',
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


def grad_norm(parameters, torch):
    grads = [p.grad.detach().norm() for p in parameters if p.grad is not None]
    return float(torch.linalg.vector_norm(torch.stack(grads))) if grads else 0.0


def gradients_are_zero_or_absent(grads, torch):
    return all(g is None or bool(torch.count_nonzero(g) == 0) for g in grads)


def data_hash(batch, positives, masks, specs):
    digest = hashlib.sha256()
    digest.update(batch.tobytes())
    digest.update(masks.tobytes())
    digest.update(json.dumps([positives, specs], sort_keys=True, separators=(',', ':')).encode())
    return digest.hexdigest()


def train(args):
    import numpy as np
    import torch
    from bc_tpro import attach_bc_tpro, MODEL_CONFIG

    original_train = OLD / 'code/train_pilot.py'
    original_loss = OLD / 'code/losses.py'
    for label, path in [('plan', args.plan), ('selection', args.selection),
                        ('checkpoint', args.checkpoint), ('train_source', original_train),
                        ('loss_source', original_loss)]:
        if sha(path) != EXPECTED[label]:
            raise ValueError(f'Frozen {label} hash mismatch: {path}')
    base = load_module('bc_frozen_training_base', original_train)
    losses = load_module('bc_frozen_top1_loss', original_loss)
    # The frozen diagnostics module imports this helper by its original module
    # name. Bind exactly the already hash-validated loss module, not a search-
    # path-dependent module with the same name.
    sys.modules['losses'] = losses
    diagnostics = load_module('bc_frozen_top1_diagnostics', OLD / 'code/training_diagnostics.py')
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

    model, original_model_file = base.load_model(args.source, args.checkpoint)
    if sha(original_model_file) != EXPECTED['model_source']:
        raise ValueError('Checkpoint-adjacent model source changed')
    original_state_hashes = {name: tensor_hash([(name, value)])
                             for name, value in model.state_dict().items()}
    original_parameter_names = set(dict(model.named_parameters()))
    rng_before = torch.get_rng_state().clone()
    model = attach_bc_tpro(model, args.variant)
    if not torch.equal(rng_before, torch.get_rng_state()):
        raise AssertionError('New module initialization changed the global Torch RNG')
    # torch.manual_seed inside the CPU-only initialization fork also schedules
    # CUDA seeds; restore the original training seed before any device transfer.
    torch.cuda.manual_seed_all(46)
    state = model.state_dict()
    if any(name not in state or tensor_hash([(name, state[name])]) != fingerprint
           for name, fingerprint in original_state_hashes.items()):
        raise AssertionError('Adapter changed or renamed released weights')
    new_names = set(state) - set(original_state_hashes)
    new_initial_hash = tensor_hash((name, state[name]) for name in new_names)
    device = torch.device(args.device)
    model = model.to(device=device, dtype=torch.float32)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
    initial_bn = base.bn_snapshot(model)
    bn_hash = base.buffer_hash(initial_bn)
    named_parameters = dict(model.named_parameters())
    groups = {
        'original': sorted(original_parameter_names),
        'reference': sorted(name for name in named_parameters if name.startswith('TPro.reference.')),
        'association': sorted(name for name in named_parameters if name.startswith(
            ('TPro.association_norm.', 'TPro.query.', 'TPro.key.'))),
        'gamma': ['TPro.gamma'],
    }
    flat_group_names = [name for group in groups.values() for name in group]
    if len(flat_group_names) != len(set(flat_group_names)) or set(flat_group_names) != set(named_parameters):
        raise AssertionError('Gradient groups must cover each trainable parameter exactly once')
    parameter_groups = {name: [named_parameters[key] for key in keys] for name, keys in groups.items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    module_path = Path(sys.modules[attach_bc_tpro.__module__].__file__)
    metadata = {
        'status': 'running', 'variant': args.variant, 'model_config': MODEL_CONFIG,
        'steps': 512, 'seed': 46, 'new_module_seed': 20261011, 'lr': 1e-4,
        'optimizer': 'fresh Adam', 'betas': [0.9, 0.999], 'eps': 1e-8, 'weight_decay': 1e-4,
        'reference_lambda': 1.0, 'effective_extra_lambda': 0.0,
        'positive_coefficient': 1.0, 'negative_coefficient': 1.0,
        'detection_background_top_fraction': 0.01, 'frozen_plan_config': base.CONFIG,
        'plan_sha256': EXPECTED['plan'], 'selection_sha256': EXPECTED['selection'],
        'checkpoint_sha256': EXPECTED['checkpoint'], 'model_source_sha256': EXPECTED['model_source'],
        'original_train_source_sha256': EXPECTED['train_source'],
        'original_loss_source_sha256': EXPECTED['loss_source'],
        'train_source_sha256': sha(__file__), 'bc_tpro_source_sha256': sha(module_path),
        'new_module_initial_state_sha256': new_initial_hash,
        'initialization_preserves_original_state': True, 'new_initialization_restores_rng': True,
        'bn_initial_sha256': bn_hash, 'parameter_names_by_gradient_group': groups,
        'parameter_counts_by_group': {name: sum(p.numel() for p in values)
                                      for name, values in parameter_groups.items()},
        'torch_version': str(torch.__version__), 'precision': 'FP32_no_AMP_no_TF32',
        'device': str(device), 'checkpoint_path': str(args.checkpoint.resolve()),
        'source_path': str(args.source.resolve()), 'training_sequences': names,
        'protocol': 'weak_box_MIL_not_official_SoftIoU',
        'checkpoint_selection': 'fixed final update512; no best checkpoint',
        'reference_mask': 'original all-box-expanded2 core9 mask eroded by temporal8 spatial9; valid ring count>0',
    }
    save_json(args.output / 'RUN.json', metadata)
    started = time.perf_counter()
    counts = Counter()
    cumulative_data_hash = hashlib.sha256()
    try:
        with (args.output / 'TRAIN.jsonl').open('w', buffering=1) as log:
            for step, entry in enumerate(plan['entries']):
                if entry['step'] != step:
                    raise ValueError('Frozen step order changed')
                for module in model.modules():
                    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                        module.eval()
                batch, positives, masks, specs = base.build_batch(entry, sequences)
                input_hash = data_hash(batch, positives, masks, specs)
                cumulative_data_hash.update(input_hash.encode())
                x = torch.from_numpy(batch).to(device)
                background = torch.from_numpy(masks).to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)[1]
                if logits.shape != (4, 40, 128, 128):
                    raise ValueError(f'Unexpected output shape {tuple(logits.shape)}')
                det_loss, parts = losses.pilot_loss(logits, positives, background, specs,
                                                  'paired_mean', entry['random_pairs'])
                ref_loss, ref_stats = model.TPro.reference_loss(background)
                loss = det_loss + ref_loss
                if not all(bool(torch.isfinite(term)) for term in [det_loss, ref_loss, loss]):
                    raise FloatingPointError(f'Nonfinite loss at step {step + 1}')
                if step == 0:
                    non_ref_parameters = (parameter_groups['original'] + parameter_groups['association']
                                          + parameter_groups['gamma'])
                    forbidden_ref = torch.autograd.grad(ref_loss, non_ref_parameters,
                                                        allow_unused=True, retain_graph=True)
                    forbidden_det = torch.autograd.grad(det_loss, parameter_groups['reference'],
                                                        allow_unused=True, retain_graph=True)
                    ref_gradient = torch.autograd.grad(ref_loss, parameter_groups['reference'],
                                                       allow_unused=True, retain_graph=True)
                    checks = {
                        'ref_has_no_gradient_to_original_association_gamma': gradients_are_zero_or_absent(forbidden_ref, torch),
                        'det_has_no_gradient_to_reference': gradients_are_zero_or_absent(forbidden_det, torch),
                        'ref_has_finite_nonzero_own_gradient': all(g is None or bool(torch.isfinite(g).all()) for g in ref_gradient)
                            and any(g is not None and bool(torch.count_nonzero(g) > 0) for g in ref_gradient),
                        'gamma_initial_zero': float(model.TPro.gamma.detach()) == 0.0,
                    }
                    old_row = json.loads((OLD / 'training/run/TRAIN.jsonl').read_text().splitlines()[0])
                    checks['first_step_positive_matches_frozen_top1'] = math.isclose(float(parts['positive'].detach()), old_row['positive'], rel_tol=1e-6, abs_tol=1e-6)
                    checks['first_step_negative_matches_frozen_top1'] = math.isclose(float(parts['negative'].detach()), old_row['negative'], rel_tol=1e-6, abs_tol=1e-6)
                    isolation = {'checks': checks, 'all_passed': all(checks.values()),
                                 'first_step_positive': float(parts['positive'].detach()),
                                 'first_step_negative': float(parts['negative'].detach()),
                                 'input_sha256': input_hash}
                    save_json(args.output / 'GRADIENT_ISOLATION.json', isolation)
                    if not isolation['all_passed']:
                        raise AssertionError(f'First-step isolation or initial-output check failed: {checks}')
                    del forbidden_ref, forbidden_det, ref_gradient
                loss.backward()
                if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                    raise FloatingPointError(f'Nonfinite parameter gradient at step {step + 1}')
                norms = {name: grad_norm(values, torch) for name, values in parameter_groups.items()}
                row = {
                    'step': step + 1, 'variant': args.variant,
                    'loss': float(loss.detach()), 'detection_loss': float(det_loss.detach()),
                    'positive': float(parts['positive'].detach()), 'negative': float(parts['negative'].detach()),
                    'extra_logged_not_optimized': float(parts['extra'].detach()),
                    'weighted_positive': float(parts['positive'].detach()),
                    'weighted_negative': float(parts['negative'].detach()),
                    'weighted_reference': float(ref_loss.detach()), 'reference_lambda': 1.0,
                    'effective_extra_lambda': 0.0, 'finite': True,
                    'base_objects_per_window': parts['base_objects_per_window'],
                    'background_pixels_per_window': parts['background_pixels_per_window'],
                    'gradient_norms': norms, 'grad_norm': math.sqrt(sum(v*v for v in norms.values())),
                    'gamma_gradient': float(model.TPro.gamma.grad.detach()) if model.TPro.gamma.grad is not None else None,
                    'input_sha256': input_hash,
                }
                row.update(diagnostics.background_diagnostics(logits, background))
                row.update(model.TPro.last_stats)
                row.update(ref_stats)
                optimizer.step()
                row['gamma_after_update'] = float(model.TPro.gamma.detach())
                current_bn = base.bn_snapshot(model)
                if current_bn.keys() != initial_bn.keys() or any(not torch.equal(current_bn[key], value)
                                                                 for key, value in initial_bn.items()):
                    raise AssertionError('Original BN running buffers changed')
                row['bn_running_sha256'] = base.buffer_hash(current_bn)
                row['max_cuda_memory_allocated_bytes'] = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0
                row['max_cuda_memory_reserved_bytes'] = torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else 0
                counts['reference_empty_windows'] += int(ref_stats['reference_empty_windows'])
                counts['extra_objects_logged'] += parts['extra_objects']
                row['elapsed_seconds'] = time.perf_counter() - started
                line = json.dumps(row, allow_nan=False)
                log.write(line + '\n')
                if step % 16 == 0 or step == 511:
                    print(line, flush=True)
                del logits, det_loss, ref_loss, loss, parts, x, background
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        final_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
        if not all(bool(torch.isfinite(value).all()) for value in final_state.values()):
            raise FloatingPointError('Nonfinite final model state')
        checkpoint = {
            'model_state_dict': final_state, 'step': 512, 'variant': args.variant,
            'model_config': MODEL_CONFIG, 'reference_lambda': 1.0,
            'effective_extra_lambda': 0.0, 'background_top_fraction': 0.01,
            'plan_sha256': EXPECTED['plan'], 'initial_checkpoint_sha256': EXPECTED['checkpoint'],
            'new_module_initial_state_sha256': new_initial_hash,
            'protocol': 'weak_box_MIL_not_official_SoftIoU',
        }
        torch.save(checkpoint, args.output / 'final_model.pth')
        metadata.update(status='complete', final_step=512, duration_seconds=time.perf_counter()-started,
                        counts=dict(counts), bn_final_sha256=base.buffer_hash(base.bn_snapshot(model)),
                        final_checkpoint_sha256=sha(args.output / 'final_model.pth'),
                        train_log_sha256=sha(args.output / 'TRAIN.jsonl'),
                        cumulative_input_sha256=cumulative_data_hash.hexdigest(),
                        gamma_final=float(model.TPro.gamma.detach()),
                        max_cuda_memory_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0,
                        max_cuda_memory_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else 0)
        save_json(args.output / 'RUN.json', metadata)
        print(json.dumps({'status': 'complete', 'variant': args.variant, 'output': str(args.output),
                          'duration_seconds': metadata['duration_seconds']}), flush=True)
    except BaseException as error:
        metadata.update(status='failed', error=repr(error), duration_seconds=time.perf_counter()-started)
        save_json(args.output / 'RUN.json', metadata)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['source', 'checkpoint', 'dataset-root', 'selection', 'plan', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--variant', choices=['bc_tpro', 'single_dynamic'], required=True)
    parser.add_argument('--steps', type=int, default=512)
    parser.add_argument('--seed', type=int, default=46)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--strategy', choices=['paired_mean'], default='paired_mean')
    args = parser.parse_args()
    if args.steps != 512 or args.seed != 46 or args.lr != 1e-4:
        parser.error('This pilot freezes --steps 512 --seed 46 --lr 0.0001')
    train(args)


if __name__ == '__main__':
    main()
