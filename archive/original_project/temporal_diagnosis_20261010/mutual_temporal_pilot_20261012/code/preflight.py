#!/usr/bin/env python3
"""Readiness checks only: discarded probe models, no formal training outputs."""
import argparse
import gc
import json
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
import train_mutual as runner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    root = Path(__file__).resolve().parents[1]
    problem = runner.Q.parent
    official_root = problem / 'official_deeppro'
    sys.path.insert(0, str(official_root))
    checkpoint = official_root / 'log/sem_seg/RGB-T__2025-01-14_15-27__SoftLoUloss_DeepPro-Plus_DataL40/checkpoints/best_model.pth'
    settings = argparse.Namespace(source=official_root, checkpoint=checkpoint,
        plan=runner.Q / 'stage1_goal/pilot/TRAIN_PLAN.json',
        selection=runner.Q / 'stage1_goal/DATA_SELECTION.json',
        dataset_root=runner.Q / 'stage1_goal/data')
    base, plan, names, sequences, model_file = runner.previous.validate_inputs(settings)
    tiling = runner.load_module('mutual_gpu_tiling', runner.Q / 'infer_with_windows.py')
    official = runner.load_module('mutual_gpu_criterion', model_file)
    criterion = official.SoftLoUloss().to(device)
    checks, details = {}, {}
    batch, positives, background_np, specs = base.build_batch(plan['entries'][0], sequences)
    target_np, valid_np, labelstats = runner.build_targets(plan['entries'][0], sequences, background_masks=background_np)
    old = json.loads(runner.OLD_LOG.read_text().splitlines()[0])
    checks['first_batch_frozen_input_hash'] = runner.data_hash(batch, positives, background_np, specs) == old['input_sha256']
    checks['first_batch_frozen_label_hash'] = runner.label_hash(target_np, valid_np) == old['label_sha256']
    x = torch.from_numpy(batch).to(device)
    targets = torch.from_numpy(target_np).to(device)
    valid = torch.from_numpy(valid_np).to(device)
    background = torch.from_numpy(background_np).to(device)
    probe = x[:1, :, :, :32, :48].contiguous()
    original, _ = base.load_model(official_root, checkpoint)
    original.to(device).eval()
    with torch.inference_mode(): baseline = original(probe)[1].detach().cpu()
    del original
    resource_rows = []
    branch_hashes = []
    for variant in ['E0', 'E1', 'E2']:
        random.seed(46); np.random.seed(46); torch.manual_seed(46); torch.cuda.manual_seed_all(46)
        model, _ = base.load_model(official_root, checkpoint)
        branch = runner.attach_aggregation(model, variant, 2)
        if branch is not None:
            branch_hashes.append(runner.tensor_hash(branch.state_dict().items()))
        model.to(device).eval()
        initial_bn = base.bn_snapshot(model)
        with torch.inference_mode(): output = model(probe)[1].detach().cpu()
        checks[variant + '_zero_projection_exact_original_output'] = torch.equal(output, baseline)
        details[variant + '_initial_max_output_error'] = float((output - baseline).abs().max())
        model.train()
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm): module.eval()
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device); start = time.perf_counter()
        logits = model(x)[1]
        loss, parts = runner.common_loss(logits, targets, valid, background, criterion)
        loss.backward()
        torch.cuda.synchronize(device)
        seconds = time.perf_counter() - start
        gradients = runner.parameter_gradient_stats(model)
        checks[variant + '_full_real_batch_loss_and_gradients_finite'] = bool(torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        checks[variant + '_bn_buffers_unchanged'] = all(torch.equal(value, base.bn_snapshot(model)[name]) for name, value in initial_bn.items())
        if branch is not None:
            checks[variant + '_output_projection_startup_gradient'] = gradients['temporal_aggregation.output_projection.weight']['norm'] > 0
            checks[variant + '_descriptor_initial_zero_gradient_expected'] = gradients['temporal_aggregation.descriptor.weight']['norm'] == 0
        resource_rows.append({'variant': variant, 'batch_shape': list(x.shape), 'forward_backward_seconds': seconds,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(device), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
            'parameters': sum(p.numel() for p in model.parameters()), 'loss': float(loss.detach()),
            'module_gradients': gradients, 'module_stats': {} if branch is None else dict(branch.last_stats)})
        print(json.dumps({'resource': resource_rows[-1]}), flush=True)
        del logits, loss
        model.zero_grad(set_to_none=True)
        if branch is not None:
            # Nonzero arbitrary deterministic projection, not a trained checkpoint.
            with torch.no_grad():
                branch.output_projection.weight.copy_(.01 * torch.randn_like(branch.output_projection.weight))
            second = model(probe)[1].square().mean()
            second.backward()
            gradients = runner.parameter_gradient_stats(model)
            checks[variant + '_descriptor_gradient_after_projection_nonzero'] = gradients['temporal_aggregation.descriptor.weight']['norm'] > 0
            checks[variant + '_null_gradient_after_projection_nonzero'] = gradients['temporal_aggregation.null_logit']['norm'] > 0
            model.zero_grad(set_to_none=True)
            del second
        model.eval()
        if branch is not None: branch.collect_stats = False
        # Real normalized training imagery, enlarged with the same raw scene to
        # force internal tile seams; no validation performance is calculated.
        full_probe = x[:1, :, :, :96, :112].contiguous()
        with torch.inference_mode():
            direct = model(full_probe)[1].cpu()
            stitched = tiling.tiled_logits(model, full_probe, 32, 20)
        torch.testing.assert_close(stitched, direct, atol=1e-4, rtol=1e-4)
        checks[variant + '_nonzero_projection_full_model_tiling_close'] = True
        details[variant + '_tiling'] = {'shape': list(full_probe.shape), 'core': 32, 'halo': 20,
            'max_logit_error': float((direct - stitched).abs().max()),
            'max_probability_error': float((direct.sigmoid() - stitched.sigmoid()).abs().max()),
            'threshold05_disagreements': int(((direct >= 0) != (stitched >= 0)).sum())}
        if variant == 'E2':
            raw = sequences[names[0]].frames[:40]
            native = torch.from_numpy(np.ascontiguousarray((raw.astype(np.float32) - 85.0799) / 47.4845))[None, None].to(device)
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device); native_start = time.perf_counter()
            with torch.inference_mode():
                native_output = tiling.tiled_logits(model, native, 128, 20)
            torch.cuda.synchronize(device)
            details['E2_native_tiled_inference'] = {'input_shape': list(native.shape),
                'output_shape': list(native_output.shape), 'seconds': time.perf_counter() - native_start,
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                'peak_reserved_bytes': torch.cuda.max_memory_reserved(device), 'core': 128, 'halo': 20,
                'source': names[0], 'condition': 'discarded deterministic nonzero output projection',
                'performance_evaluation': False}
            checks['E2_native_40x512x640_finite'] = bool(torch.isfinite(native_output).all())
            print(json.dumps({'native_inference': details['E2_native_tiled_inference']}), flush=True)
            del native, native_output
        del model, branch, direct, stitched, output
        gc.collect(); torch.cuda.empty_cache()
    checks['E1_E2_branch_initial_tensor_state_identical'] = len(branch_hashes) == 2 and branch_hashes[0] == branch_hashes[1]
    details['branch_initial_sha256'] = branch_hashes
    report = {'status': 'PASS' if all(checks.values()) else 'FAIL', 'checks': checks, 'details': details,
        'resource_rows': resource_rows, 'device': str(device), 'gpu_name': torch.cuda.get_device_name(device),
        'execution': 'Discarded probe models, no optimizer or saved weights; synthetic nonzero output projection only for implementation tests',
        'train_source_sha256': runner.sha(runner.__file__), 'module_source_sha256': runner.sha(root / 'code/module.py'),
        'test_source_sha256': runner.sha(__file__)}
    runner.save_json(root / 'checks/NETWORK_PREFLIGHT.json', report)
    print(json.dumps({'status': report['status'], 'checks': len(checks), 'failed': [k for k,v in checks.items() if not v]}), flush=True)
    if not all(checks.values()): raise SystemExit(1)


if __name__ == '__main__':
    main()
