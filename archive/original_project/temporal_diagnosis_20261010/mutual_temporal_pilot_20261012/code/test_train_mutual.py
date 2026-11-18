#!/usr/bin/env python3
"""CPU checks for the single frozen empty-window supervision change."""
import json
from pathlib import Path
import sys
import torch
import train_mutual as runner


def main():
    torch.set_num_threads(1)
    torch.manual_seed(46)
    source = runner.Q.parent / 'official_deeppro'
    checkpoint = source / 'log/sem_seg/RGB-T__2025-01-14_15-27__SoftLoUloss_DeepPro-Plus_DataL40/checkpoints/best_model.pth'
    model_file = checkpoint.parent.parent / 'DeepPro-Plus.py'
    sys.path.insert(0, str(source))
    official = runner.load_module('mutual_loss_check_official', model_file)
    criterion = official.SoftLoUloss()
    checks = {}
    z = torch.randn(4, 3, 5, 7, requires_grad=True)
    y = (torch.rand_like(z) > .6).float()
    valid = torch.ones_like(z, dtype=torch.bool)
    valid[:, :, 0] = False
    y[~valid] = 0
    background = valid & (y == 0)
    old, _ = runner.previous.masked_official_loss(z, y, valid, criterion)
    new, parts = runner.common_loss(z, y, valid, background, criterion)
    old_grad = torch.autograd.grad(old, z, retain_graph=True)[0]
    new_grad = torch.autograd.grad(new, z, retain_graph=True)[0]
    checks['all_positive_exact_old_loss'] = torch.equal(old, new)
    checks['all_positive_exact_old_gradient'] = torch.equal(old_grad, new_grad)
    y[1] = 0; y[3] = 0
    background[1] = valid[1]; background[3] = valid[3]
    mixed, parts = runner.common_loss(z, y, valid, background, criterion)
    mixed_grad = torch.autograd.grad(mixed, z, retain_graph=True)[0]
    separate = []
    for wi in range(4):
        if wi in (1, 3):
            values = torch.nn.functional.softplus(z[wi][background[wi]])
            separate.append(runner.common_loss._top1(values))
        else:
            separate.append(criterion(z[wi][valid[wi]].reshape(1, 1, 1, -1), y[wi][valid[wi]].reshape(1, 1, 1, -1)))
    explicit = torch.stack(separate).mean()
    explicit_grad = torch.autograd.grad(explicit, z, retain_graph=True)[0]
    checks['mixed_value_equals_four_window_mean'] = torch.equal(mixed, explicit)
    checks['mixed_gradient_equals_four_window_mean'] = torch.equal(mixed_grad, explicit_grad)
    checks['mixed_empty_count'] = parts['pure_negative_windows'] == 2
    checks['mixed_empty_original_loss_one'] = parts['original_official_loss_per_window'][1] == parts['original_official_loss_per_window'][3] == 1.0
    checks['empty_windows_have_positive_negative_gradient'] = bool((mixed_grad[1] > 0).any() and (mixed_grad[3] > 0).any())
    checks['invalid_pixels_have_exact_zero_gradient'] = not bool(torch.count_nonzero(mixed_grad[~valid]))
    bad = background.clone(); bad[1, 0, 1, 1] = ~bad[1, 0, 1, 1]
    try:
        runner.common_loss(z, y, valid, bad, criterion)
        checks['empty_mask_disagreement_rejected'] = False
    except AssertionError:
        checks['empty_mask_disagreement_rejected'] = True
    empty_valid = valid.clone(); empty_valid[0] = False
    try:
        runner.common_loss(z, y, empty_valid, background, criterion)
        checks['empty_valid_rejected'] = False
    except ValueError:
        checks['empty_valid_rejected'] = True
    tied = torch.zeros(4, 1, 1, 201, requires_grad=True)
    zero = torch.zeros_like(tied)
    all_valid = torch.ones_like(tied, dtype=torch.bool)
    loss, tied_parts = runner.common_loss(tied, zero, all_valid, all_valid, criterion)
    loss.backward()
    checks['top1_exact_ceil_selection'] = tied_parts['selected_negative_pixels_per_window'] == [3] * 4
    checks['ties_favor_first_flat_indices'] = bool((tied.grad[..., :3] > 0).all() and (tied.grad[..., 3:] == 0).all())
    checks['all_negative_loss_softplus_zero'] = abs(float(loss) - float(torch.nn.functional.softplus(torch.tensor(0.0)))) < 1e-7
    details = {'mixed_parts': parts, 'mixed_loss': float(mixed.detach()), 'top1_tie_parts': tied_parts}
    report = {'status': 'PASS' if all(checks.values()) else 'FAIL', 'checks': checks, 'details': details,
              'execution': 'CPU synthetic logits only, no model training or new data generation',
              'train_source_sha256': runner.sha(runner.__file__), 'test_source_sha256': runner.sha(__file__),
              'official_source_sha256': runner.sha(model_file), 'label_source_sha256': runner.sha(runner.LABEL_PATH),
              'top1_loss_source_sha256': runner.sha(runner.TOP1_PATH)}
    out = Path(__file__).resolve().parents[1] / 'checks/TRAIN_LOSS_CHECK.json'
    runner.save_json(out, report)
    print(json.dumps({'status': report['status'], 'checks': len(checks), 'failed': [k for k, v in checks.items() if not v], 'report': str(out)}))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
