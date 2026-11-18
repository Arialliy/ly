#!/usr/bin/env python3
"""Frozen E0/E1/E2 training: one shared supervision and one fixed 512-step path."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

Q = Path('/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/temporal_diagnosis_20261010')
PREVIOUS = Q / 'supervision_alignment_control_20261012'
sys.path.insert(0, str(PREVIOUS / 'code'))
import train_supervision as previous
from gaussian_labels import build_targets, LABEL_CONFIG

sha = previous.sha
save_json = previous.save_json
load_module = previous.load_module
tensor_hash = previous.tensor_hash
data_hash = previous.data_hash
label_hash = previous.label_hash
EXPECTED = previous.EXPECTED
VARIANTS = {'E0': 'off', 'E1': 'single', 'E2': 'mutual'}
LABEL_PATH = PREVIOUS / 'code/gaussian_labels.py'
TOP1_PATH = Q / 'top1_background_control_20261011/code/losses.py'
OLD_LOG = PREVIOUS / 'training/gaussian_softiou_approx/run/TRAIN.jsonl'
SNAPSHOTS = (0, 128, 256, 384, 512)
EFFECTIVE_SUPERVISION = {
    'positive_window': 'unchanged masked official SoftLoUloss, internal sigmoid, smooth0',
    'empty_window': 'replace constant1 with old top1pct softplus background loss, coefficient1',
    'empty_domain_assertion': 'valid equals old background mask',
    'outer_reduction': 'four windows equal mean',
    'label_config_scope': 'frozen historical label generation; its old empty-loss note is superseded here',
}


def common_loss(logits, targets, valid, background, criterion):
    """Official positive-window loss; only empty windows receive old top-1%.

    The official criterion receives logits exactly once. Its internal sigmoid,
    smooth=0, per-window reduction, and four-window outer mean are preserved.
    Ties in negative mining use the unchanged old implementation.
    """
    import torch
    import torch.nn.functional as functional
    if not hasattr(common_loss, '_top1'):
        common_loss._top1 = load_module('mutual_frozen_top1_loss', TOP1_PATH).background_top1pct_mean
    if (logits.ndim != 4 or targets.shape != logits.shape or valid.shape != logits.shape
            or background.shape != logits.shape or valid.dtype != torch.bool or background.dtype != torch.bool):
        raise ValueError('Expected matching logits/targets and boolean valid/background [B,T,H,W]')
    applied, original, positive_counts, valid_counts, bg_counts, selected_counts, negative_terms = [], [], [], [], [], [], []
    for wi in range(logits.shape[0]):
        vi = valid[wi]
        if not bool(vi.any()):
            raise ValueError('An empty valid window has no defined official SoftIoU loss')
        z = logits[wi][vi].reshape(1, 1, 1, -1)
        y = targets[wi][vi].reshape(1, 1, 1, -1)
        official = criterion(z, y)
        original.append(official)
        mass = int(torch.count_nonzero(y))
        positive_counts.append(mass)
        valid_counts.append(int(vi.sum()))
        bg_counts.append(int(background[wi].sum()))
        if mass:
            applied.append(official)
            negative_terms.append(0.0)
            selected_counts.append(0)
        else:
            if not torch.equal(vi, background[wi]):
                raise AssertionError('Empty-label valid domain must equal frozen old background domain')
            negative_logits = logits[wi][background[wi]]
            negative = (common_loss._top1(functional.softplus(negative_logits)) if negative_logits.numel()
                        else logits[wi].sum() * 0.0)
            applied.append(negative)
            selected_counts.append(max(1, math.ceil(.01 * negative_logits.numel())) if negative_logits.numel() else 0)
            negative_terms.append(float(negative.detach()))
    value = torch.stack(applied).mean()
    return value, {
        'loss_per_window': [float(x.detach()) for x in applied],
        'original_official_loss_per_window': [float(x.detach()) for x in original],
        'original_official_loss': float(torch.stack(original).detach().mean()),
        'empty_window_negative_per_window': negative_terms,
        'background_pixels_per_window': bg_counts,
        'selected_negative_pixels_per_window': selected_counts,
        'valid_pixels_per_window': valid_counts,
        'positive_pixels_per_window': positive_counts,
        'target_mass_per_window': [float(x) for x in positive_counts],
        'pure_negative_windows': sum(n == 0 for n in positive_counts),
        'valid_pixels': sum(valid_counts), 'positive_pixels': sum(positive_counts),
        'target_mass': sum(positive_counts),
    }


def attach_aggregation(model, variant, radius):
    """Preserve every original key and intercept layer1 before the old TPro."""
    if variant not in VARIANTS:
        raise ValueError(variant)
    if variant == 'E0':
        return None
    from module import RejectableTemporalAggregation
    aggregation = RejectableTemporalAggregation(C=32, descriptor_dim=8, radius=radius,
                                                mode=VARIANTS[variant], temporal_chunk_size=2,
                                                checkpoint_chunks=True)
    model.add_module('temporal_aggregation', aggregation)
    model.layer1.register_forward_hook(lambda layer, args, result: model.temporal_aggregation(result))
    return aggregation


def model_metadata(model, aggregation, variant, radius):
    return {'variant': variant, 'mode': VARIANTS[variant], 'radius': radius,
            'descriptor_dim': 8, 'value_projection': 'identity',
            'output_projection': 'zero_initialized_32x32_no_bias',
            'temporal_chunk_size': 2, 'checkpoint_chunks': True,
            'module_insertion': 'layer1 output forward hook, before original permute/TPro',
            'parameter_count': sum(p.numel() for p in model.parameters()),
            'new_parameters': 0 if aggregation is None else sum(p.numel() for p in aggregation.parameters()),
            'branch_initial_sha256': None if aggregation is None else tensor_hash(aggregation.state_dict().items())}


def parameter_gradient_stats(model):
    import torch
    output = {}
    for name, parameter in model.named_parameters():
        if not name.startswith('temporal_aggregation.'):
            continue
        gradient = parameter.grad
        output[name] = {'present': gradient is not None,
                        'norm': None if gradient is None else float(gradient.detach().norm()),
                        'nonzero': 0 if gradient is None else int(torch.count_nonzero(gradient)),
                        'parameter_norm': float(parameter.detach().norm())}
    return output


def train(args):
    import numpy as np
    import torch
    base, plan, names, sequences, model_file = previous.validate_inputs(args)
    old_rows = [json.loads(line) for line in OLD_LOG.read_text().splitlines()]
    if len(old_rows) != 512:
        raise AssertionError('Prior exact-exposure comparison must have 512 rows')
    source_paths = {'train_source': Path(__file__), 'label_source': LABEL_PATH,
                    'model_source': model_file, 'top1_loss_source': TOP1_PATH,
                    'module_source': Path(__file__).with_name('module.py'),
                    'prior_train_source': PREVIOUS / 'code/train_supervision.py'}
    source_hashes = {name + '_sha256': sha(path) for name, path in source_paths.items()}
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Formal training output directory must be new and empty')
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(46); np.random.seed(46); torch.manual_seed(46); torch.cuda.manual_seed_all(46)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    model, loaded_file = base.load_model(args.source, args.checkpoint)
    if loaded_file.resolve() != model_file.resolve():
        raise AssertionError('Model source mismatch')
    original_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    if sum(p.numel() for p in model.parameters()) != 70913:
        raise AssertionError('Original architecture is not the frozen Plus')
    aggregation = attach_aggregation(model, args.variant, args.radius)
    spec = model_metadata(model, aggregation, args.variant, args.radius)
    initial_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    if any(not torch.equal(value, initial_state[name]) for name, value in original_state.items()):
        raise AssertionError('Attaching a branch modified original released tensors')
    official = load_module('mutual_official_criterion', model_file)
    criterion = official.SoftLoUloss()
    device = torch.device(args.device)
    model = model.to(device=device, dtype=torch.float32)
    criterion = criterion.to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
    original_bn = base.bn_snapshot(model)
    bn_hash = base.buffer_hash(original_bn)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    metadata = {
        'status': 'running', 'condition': args.variant, 'variant': args.variant,
        'steps': 512, 'seed': 46, 'lr': 1e-4, 'weight_decay': 1e-4,
        'optimizer': 'fresh Adam', 'betas': [.9, .999], 'eps': 1e-8,
        'precision': 'FP32_no_AMP_no_TF32', 'bn_policy': 'eval_running_stats_affine_trainable',
        'bn_initial_sha256': bn_hash, 'checkpoint_sha256': EXPECTED['checkpoint'],
        'plan_sha256': EXPECTED['plan'], 'selection_sha256': EXPECTED['selection'],
        'old_log_sha256': sha(OLD_LOG), 'initial_model_state_sha256': tensor_hash(initial_state.items()),
        'initial_original_state_sha256': tensor_hash(original_state.items()),
        'original_train_source_sha256': EXPECTED['original_train_source'],
        'label_config': LABEL_CONFIG, 'effective_supervision_config': EFFECTIVE_SUPERVISION,
        'frozen_plan_config': base.CONFIG, **spec, **source_hashes,
        'training_sequences': names, 'torch_version': str(torch.__version__), 'device': str(device),
        'source_path': str(args.source.resolve()), 'checkpoint_path': str(args.checkpoint.resolve()),
        'criterion_source_path': str(model_file.resolve()), 'label_source_path': str(LABEL_PATH),
        'protocol': 'positive_masked_official_SoftIoU_empty_old_top1pct',
        'loss_reduction': 'unchanged official per positive window; empty window old top1 negative; four-window equal mean',
        'checkpoint_selection': 'fixed final update512; snapshots0/128/256/384/512 never selected by validation',
        'snapshots': [],
    }
    save_json(args.output / 'INITIAL_STATE.json', {
        **spec, 'original_checkpoint_sha256': EXPECTED['checkpoint'],
        'model_state_sha256': tensor_hash(initial_state.items()),
        'state_tensor_sha256': {name: tensor_hash([(name, value)]) for name, value in initial_state.items()},
        'parameter_names': sorted(dict(model.named_parameters())),
    })

    def snapshot(step):
        checkpoint = {'model_state_dict': {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
                      'optimizer_state_dict': optimizer.state_dict(), 'step': step,
                      'variant': args.variant, 'condition': args.variant, 'radius': args.radius,
                      'model_config': spec, 'label_config': LABEL_CONFIG,
                      'effective_supervision_config': EFFECTIVE_SUPERVISION,
                      'initial_checkpoint_sha256': EXPECTED['checkpoint'],
                      'plan_sha256': EXPECTED['plan'], 'protocol': metadata['protocol'], **source_hashes}
        destination = args.output / f'snapshot_{step:03d}.pth'
        torch.save(checkpoint, destination)
        metadata['snapshots'].append({'step': step, 'path': destination.name, 'sha256': sha(destination),
                                      'model_state_sha256': tensor_hash(checkpoint['model_state_dict'].items())})
        save_json(args.output / 'RUN.json', metadata)

    snapshot(0)
    started = time.perf_counter()
    counts = Counter()
    input_chain, label_chain = hashlib.sha256(), hashlib.sha256()
    try:
        with (args.output / 'TRAIN.jsonl').open('w', buffering=1) as log:
            for index, entry in enumerate(plan['entries']):
                if entry['step'] != index:
                    raise AssertionError('Frozen step order changed')
                batch, positives, masks, specs = base.build_batch(entry, sequences)
                target_np, valid_np, label_stats = build_targets(entry, sequences, background_masks=masks)
                input_sha = data_hash(batch, positives, masks, specs)
                labels_sha = label_hash(target_np, valid_np)
                if input_sha != old_rows[index]['input_sha256'] or labels_sha != old_rows[index]['label_sha256']:
                    raise AssertionError(f'Exact previous data/labels differ at step {index + 1}')
                input_chain.update(input_sha.encode()); label_chain.update(labels_sha.encode())
                x = torch.from_numpy(batch).to(device)
                targets = torch.from_numpy(target_np).to(device)
                valid = torch.from_numpy(valid_np).to(device)
                background = torch.from_numpy(masks).to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)[1]
                if logits.shape != (4, 40, 128, 128):
                    raise ValueError('Unexpected training output dimensions')
                loss, parts = common_loss(logits, targets, valid, background, criterion)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Nonfinite training loss')
                loss.backward()
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                if not grads or not all(bool(torch.isfinite(g).all()) for g in grads):
                    raise FloatingPointError('Missing or nonfinite parameter gradient')
                row = {'step': index + 1, 'variant': args.variant, 'condition': args.variant,
                       'loss': float(loss.detach()), 'grad_norm': float(torch.linalg.vector_norm(torch.stack([g.detach().norm() for g in grads]))),
                       'parameters_with_gradient': len(grads), 'finite': True,
                       'input_sha256': input_sha, 'label_sha256': labels_sha,
                       'label_stats': label_stats, **parts,
                       'module_stats': {} if aggregation is None else dict(aggregation.last_stats),
                       'module_gradients': parameter_gradient_stats(model)}
                optimizer.step()
                current_bn = base.bn_snapshot(model)
                if current_bn.keys() != original_bn.keys() or any(not torch.equal(current_bn[name], value) for name, value in original_bn.items()):
                    raise AssertionError('BN running statistics changed')
                row['bn_running_sha256'] = base.buffer_hash(current_bn)
                for key in ['pure_negative_windows', 'valid_pixels', 'positive_pixels']:
                    counts[key] += parts[key]
                row['elapsed_seconds'] = time.perf_counter() - started
                row['max_cuda_memory_allocated_bytes'] = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0
                row['max_cuda_memory_reserved_bytes'] = torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else 0
                encoded = json.dumps(row, allow_nan=False)
                log.write(encoded + '\n')
                if index % 16 == 0 or index == 511:
                    print(encoded, flush=True)
                if index + 1 in SNAPSHOTS:
                    snapshot(index + 1)
                del x, targets, valid, background, logits, loss, grads
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        if source_hashes != {name + '_sha256': sha(path) for name, path in source_paths.items()}:
            raise AssertionError('Frozen source file changed during training')
        final_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
        if set(initial_state) != set(final_state) or not all(bool(torch.isfinite(v).all()) for v in final_state.values()):
            raise AssertionError('Invalid final model state')
        differences = {}
        for name, _ in model.named_parameters():
            difference = final_state[name].double() - initial_state[name].double()
            differences[name] = {'numel': difference.numel(), 'changed_elements': int(torch.count_nonzero(difference)),
                                 'max_absolute_difference': float(difference.abs().max()), 'l2_difference': float(difference.norm())}
        save_json(args.output / 'PARAMETER_DIFF.json', {'parameter_count': spec['parameter_count'],
                  'new_parameter_count': spec['new_parameters'], 'parameter_tensors': len(differences),
                  'changed_parameter_tensors': sum(d['changed_elements'] > 0 for d in differences.values()),
                  'per_parameter': differences})
        # Identical serialized bytes, not a second serialization with a new archive name.
        os.link(args.output / 'snapshot_512.pth', args.output / 'final_model.pth')
        metadata.update(status='complete', final_step=512, duration_seconds=time.perf_counter() - started,
                        counts=dict(counts), bn_final_sha256=base.buffer_hash(base.bn_snapshot(model)),
                        final_checkpoint_sha256=sha(args.output / 'final_model.pth'),
                        final_model_state_sha256=tensor_hash(final_state.items()),
                        cumulative_input_sha256=input_chain.hexdigest(), cumulative_label_sha256=label_chain.hexdigest(),
                        train_log_sha256=sha(args.output / 'TRAIN.jsonl'),
                        parameter_diff_sha256=sha(args.output / 'PARAMETER_DIFF.json'),
                        max_cuda_memory_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0,
                        max_cuda_memory_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else 0)
        save_json(args.output / 'RUN.json', metadata)
        print(json.dumps({'status': 'complete', 'variant': args.variant, 'duration_seconds': metadata['duration_seconds']}), flush=True)
    except BaseException as error:
        metadata.update(status='failed', error=repr(error), duration_seconds=time.perf_counter() - started)
        save_json(args.output / 'RUN.json', metadata)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['source', 'checkpoint', 'dataset-root', 'selection', 'plan', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--radius', type=int, required=True)
    parser.add_argument('--steps', type=int, default=512)
    parser.add_argument('--seed', type=int, default=46)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.steps != 512 or args.seed != 46 or args.lr != 1e-4 or args.radius != 2:
        parser.error('Frozen 512 steps, seed46, lr1e-4, preregistered radius2 required')
    train(args)


if __name__ == '__main__':
    main()
