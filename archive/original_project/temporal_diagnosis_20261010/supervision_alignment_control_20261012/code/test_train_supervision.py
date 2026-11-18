#!/usr/bin/env python3
"""CPU-only integration and exact-official-loss checks; no model updates."""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import torch

import train_supervision as runner
from gaussian_labels import build_targets, LABEL_CONFIG


def main():
    torch.set_num_threads(1)
    torch.manual_seed(46)
    root = runner.Q / 'supervision_alignment_control_20261012'
    problem = runner.Q.parent
    checkpoint = problem / 'official_deeppro/log/sem_seg/RGB-T__2025-01-14_15-27__SoftLoUloss_DeepPro-Plus_DataL40/checkpoints/best_model.pth'
    args = argparse.Namespace(source=problem/'official_deeppro', checkpoint=checkpoint,
                              plan=runner.Q/'stage1_goal/pilot/TRAIN_PLAN.json',
                              selection=runner.Q/'stage1_goal/DATA_SELECTION.json',
                              dataset_root=runner.Q/'stage1_goal/data')
    checks = {}
    details = {}
    base, plan, names, sequences, model_file = runner.validate_inputs(args)
    checks['original_plan_selection_sources_all_train_fingerprints'] = True
    model, loaded_file = base.load_model(args.source, args.checkpoint)
    checks['checkpoint_adjacent_model_source'] = loaded_file.resolve() == model_file.resolve()
    checks['original_parameter_count_70913'] = sum(p.numel() for p in model.parameters()) == 70913
    checks['all_original_parameters_trainable'] = all(p.requires_grad for p in model.parameters())
    initial_bn = base.bn_snapshot(model)
    checks['released_bn_buffers_hash'] = base.buffer_hash(initial_bn) == '51832be8d623f18e372232ddefdf4214ad67b3e45f795fbaa2ea9daa7fd0ecab'
    official = runner.load_module('supervision_test_real_criterion', model_file)
    criterion = official.SoftLoUloss()
    checks['criterion_has_no_parameters'] = not list(criterion.parameters())
    logits = torch.randn(4, 3, 5, 7, dtype=torch.float32, requires_grad=True)
    target = (torch.rand_like(logits) > 0.72).float()
    valid = torch.ones_like(logits, dtype=torch.bool)
    full_loss, _ = runner.masked_official_loss(logits, target, valid, criterion)
    official_loss = criterion(logits, target)
    full_grad = torch.autograd.grad(full_loss, logits, retain_graph=True)[0]
    official_grad = torch.autograd.grad(official_loss, logits)[0]
    checks['all_valid_loss_matches_actual_official_4D'] = torch.allclose(full_loss, official_loss, rtol=1e-6, atol=1e-7)
    checks['all_valid_gradient_matches_actual_official_4D'] = torch.allclose(full_grad, official_grad, rtol=1e-5, atol=1e-8)
    details['all_valid_loss_max_difference'] = float((full_loss - official_loss).abs())
    details['all_valid_gradient_max_difference'] = float((full_grad - official_grad).abs().max())
    valid[0, :, :2, :] = False
    valid[1, :, 2:, :3] = False
    valid[2, :2] = False
    masked_loss, parts = runner.masked_official_loss(logits, target, valid, criterion)
    prob = logits.sigmoid() * valid
    masked_target = target * valid
    intersection = (prob * masked_target).sum((1, 2, 3))
    union = (prob + masked_target - prob * masked_target).sum((1, 2, 3))
    explicit = (1 - intersection / union).mean()
    masked_grad = torch.autograd.grad(masked_loss, logits, retain_graph=True)[0]
    explicit_grad = torch.autograd.grad(explicit, logits)[0]
    checks['partial_valid_value_equals_mathematical_mask'] = torch.allclose(masked_loss, explicit, rtol=1e-6, atol=1e-7)
    checks['partial_valid_gradient_equals_mathematical_mask'] = torch.allclose(masked_grad, explicit_grad, rtol=1e-5, atol=1e-8)
    checks['invalid_pixels_exact_zero_gradient'] = bool(torch.count_nonzero(masked_grad[~valid]) == 0)
    checks['variable_count_windows_equal_mean'] = abs(float(masked_loss) - sum(parts['loss_per_window']) / 4) < 1e-7
    checks['variable_count_windows_actually_distinct'] = len(set(parts['valid_pixels_per_window'])) == 4
    details['partial_valid_counts'] = parts['valid_pixels_per_window']
    negative_loss, negative_parts = runner.masked_official_loss(logits, torch.zeros_like(target), valid, criterion)
    negative_grad = torch.autograd.grad(negative_loss, logits)[0]
    checks['pure_negative_window_official_loss_one'] = float(negative_loss) == 1.0
    checks['pure_negative_window_exact_zero_gradient'] = bool(torch.count_nonzero(negative_grad) == 0)
    checks['pure_negative_windows_counted'] = negative_parts['pure_negative_windows'] == 4
    empty = valid.clone()
    empty[1] = False
    try:
        runner.masked_official_loss(logits, target, empty, criterion)
        checks['empty_valid_window_rejected'] = False
    except ValueError as error:
        checks['empty_valid_window_rejected'] = 'empty valid window' in str(error)
    batch, positives, old_mask, specs = base.build_batch(plan['entries'][0], sequences)
    labels, validity, stats = build_targets(plan['entries'][0], sequences, background_masks=old_mask)
    checks['true_first_batch_shape_and_dtype'] = batch.shape == (4, 1, 40, 128, 128) and batch.dtype == np.float32
    checks['true_labels_shape_and_dtype'] = labels.shape == (4, 40, 128, 128) and labels.dtype == np.float32
    checks['true_valid_shape_and_dtype'] = validity.shape == labels.shape and validity.dtype == np.bool_
    checks['true_labels_binary_and_invalid_zero'] = bool(np.isin(labels, [0, 1]).all() and not np.any(labels[~validity]))
    checks['true_background_mask_unmutated'] = np.array_equal(old_mask, base.build_batch(plan['entries'][0], sequences)[2])
    old_first = json.loads((runner.Q/'bc_tpro_pilot_20261011/training/single_dynamic/run/TRAIN.jsonl').read_text().splitlines()[0])
    details['first_input_sha256'] = runner.data_hash(batch, positives, old_mask, specs)
    checks['true_first_batch_hash_matches_completed_control'] = details['first_input_sha256'] == old_first['input_sha256']
    details['first_label_sha256'] = runner.label_hash(labels, validity)
    details['first_label_stats'] = stats
    # Exercise the actual label tensors through the real criterion; gradients
    # are only with respect to a synthetic logit tensor, never model parameters.
    synthetic_logits = torch.zeros(labels.shape, requires_grad=True)
    actual_loss, actual_stats = runner.masked_official_loss(synthetic_logits, torch.from_numpy(labels), torch.from_numpy(validity), criterion)
    actual_loss.backward()
    checks['true_label_loss_and_gradient_finite'] = bool(torch.isfinite(actual_loss) and torch.isfinite(synthetic_logits.grad).all())
    checks['true_label_mask_zero_gradient_outside_valid'] = bool(torch.count_nonzero(synthetic_logits.grad[~torch.from_numpy(validity)]) == 0)
    checks['true_label_mass_matches_loss_stats'] = actual_stats['target_mass_per_window'] == stats['target_mass_per_window']
    checks['true_valid_count_matches_loss_stats'] = actual_stats['valid_pixels_per_window'] == stats['valid_pixels_per_window']
    checks['model_has_no_gradient_or_update'] = all(p.grad is None for p in model.parameters())
    checks['bn_untouched_by_smoke'] = all(torch.equal(initial_bn[name], value) for name, value in base.bn_snapshot(model).items())
    details['synthetic_first_label_loss'] = float(actual_loss)
    details['initial_model_state_sha256'] = runner.tensor_hash(model.state_dict().items())
    output = {'status': 'PASS' if all(checks.values()) else 'FAIL', 'checks': checks, 'details': details,
              'execution': 'CPU only; real model/source/data/labels loaded; no model forward, optimizer, or updates',
              'torch_version': str(torch.__version__), 'label_config': LABEL_CONFIG,
              'train_source_sha256': runner.sha(runner.__file__),
              'test_source_sha256': runner.sha(__file__),
              'label_source_sha256': runner.sha(root/'code/gaussian_labels.py'),
              'model_source_sha256': runner.sha(model_file)}
    runner.save_json(root/'checks/TRAIN_RUNNER_CHECK.json', output)
    print(json.dumps({'status': output['status'], 'checks': len(checks), 'failed': [key for key, value in checks.items() if not value],
                      'train_source_sha256': output['train_source_sha256'], 'report': str(root/'checks/TRAIN_RUNNER_CHECK.json')}))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
