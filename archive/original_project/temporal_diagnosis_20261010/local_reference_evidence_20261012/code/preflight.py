"""Necessary seeded CUDA integration gates; no formal training or dev inference."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch

N = Path(__file__).resolve().parents[1]
Q = N.parent
R = Q.parent
sys.path.insert(0, str(Q / 'supervision_alignment_control_20261012/code'))
import train_supervision as prior
FrozenEvidenceDetector = prior.load_module('lre_preflight_module', N / 'code/module.py').FrozenEvidenceDetector

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.source = R / 'official_deeppro'
    args.checkpoint = args.source / 'log/sem_seg/RGB-T__2025-01-14_15-27__SoftLoUloss_DeepPro-Plus_DataL40/checkpoints/best_model.pth'
    args.dataset_root = Q / 'stage1_goal/data'
    args.selection = Q / 'stage1_goal/DATA_SELECTION.json'
    args.plan = Q / 'stage1_goal/pilot/TRAIN_PLAN.json'
    base, plan, names, sequences, source = prior.validate_inputs(args)
    random.seed(46); np.random.seed(46); torch.manual_seed(46); torch.cuda.manual_seed_all(46)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    witness = torch.randn(8, 8, device='cuda')
    print('WITNESS_OK', float((witness @ witness).sum()), torch.cuda.get_device_name(), flush=True)
    torch.manual_seed(46); torch.cuda.manual_seed_all(46)
    original, loaded_source = base.load_model(args.source, args.checkpoint)
    assert source == loaded_source
    model = FrozenEvidenceDetector(original, 'relative').cuda().train()
    assert not model.original.training and all(not p.requires_grad for p in model.original.parameters())
    assert sum(p.numel() for p in model.evidence.parameters()) == 1529
    initial_original = prior.tensor_hash(model.original.state_dict().items())
    batch, positives, masks, specs = base.build_batch(plan['entries'][0], sequences)
    x = torch.from_numpy(batch[:1, :, :, :96, :112].copy()).cuda()
    checks = []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.no_grad():
        baseline = model.original(x)[1]
        for mode in ['current', 'difference', 'relative']:
            model.evidence.mode = mode
            out = model(x)[1]
            assert torch.equal(out, baseline), mode
            checks.append({'name': 'zero_projection_exact_' + mode, 'pass': True})
        model.evidence.mode = 'relative'
        torch.nn.init.normal_(model.evidence.output.weight, std=.02)
        model.evidence.output.bias.fill_(.03)
        direct = model(x)[1].cpu()
        assert torch.count_nonzero(direct - baseline.cpu()) > 0
        inference = prior.load_module('lre_old_tiling_helper', Q / 'infer_with_windows.py')
        tiled = inference.tiled_logits(model, x, 32, 16)
        torch.testing.assert_close(tiled, direct, atol=1e-4, rtol=1e-4)
        disagreements = int(((tiled.sigmoid() >= .5) != (direct.sigmoid() >= .5)).sum())
        assert disagreements == 0
        checks.append({'name': 'real_training_crop_nonzero_full_vs_tiled', 'pass': True,
                       'shape': list(x.shape), 'core': 32, 'halo': 16,
                       'max_logit_error': float((tiled-direct).abs().max()),
                       'max_probability_error': float((tiled.sigmoid()-direct.sigmoid()).abs().max()),
                       'threshold05_disagreements': disagreements})
        model.correction_enabled = False
        assert torch.equal(model(x)[1], baseline)
        checks.append({'name': 'disabled_correction_exact_after_nonzero_projection', 'pass': True})
    model.correction_enabled = True
    model.zero_grad(set_to_none=True)
    loss = model(x[..., :32, :40])[1].square().mean()
    loss.backward()
    gradients = {}
    for name, p in model.evidence.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        gradients[name] = float(p.grad.norm())
    assert all(p.grad is None for p in model.original.parameters())
    assert prior.tensor_hash(model.original.state_dict().items()) == initial_original
    checks.append({'name': 'real_baseline_frozen_new_branch_finite_gradients', 'pass': True, 'gradients': gradients})
    torch.cuda.synchronize()
    result = {'status': 'PASS', 'seed': 46, 'gpu': torch.cuda.get_device_name(),
              'checks': checks, 'seconds': time.perf_counter()-started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'original_state_unchanged': True, 'new_parameters': 1529,
              'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
              'formal_training_updates': 0, 'optimizer_steps_in_preflight': 0,
              'data_source': 'first frozen training-plan batch only; no development or confirmation input',
              'source_sha256': {str(p): prior.sha(p) for p in [Path(__file__), N/'code/module.py', source, args.plan]},
              'scope': 'implementation and formula gates only; no detection-performance claim'}
    prior.save_json(args.output, result)
    print(json.dumps(result), flush=True)

if __name__ == '__main__':
    main()
