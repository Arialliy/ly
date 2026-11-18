#!/usr/bin/env python3
"""CPU checks of real DeepPro-Plus compatibility and BC_TPro gradient contracts."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import torch

from bc_tpro import BC_TPro, MODEL_CONFIG, attach_bc_tpro


ROOT = Path(__file__).resolve().parents[2]
OFFICIAL = ROOT.parent / "official_deeppro"
CHECKPOINT = OFFICIAL / "log/sem_seg/RGB-T__2025-01-14_15-27__SoftLoUloss_DeepPro-Plus_DataL40/checkpoints/best_model.pth"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=OFFICIAL)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    source_paths = {"bc_tpro.py": Path(__file__).with_name("bc_tpro.py"),
                    "test_bc_tpro.py": Path(__file__).resolve()}
    source_hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                     for name, path in source_paths.items()}
    torch.set_num_threads(2)
    torch.manual_seed(714)
    torch.use_deterministic_algorithms(True)
    checks = []

    def checked(name, condition, **details):
        item = {"name": name, "pass": bool(condition), **details}
        checks.append(item)
        if not item["pass"]:
            raise AssertionError(json.dumps(item))

    loader_path = ROOT / "top1_background_control_20261011/code/train_pilot.py"
    spec = importlib.util.spec_from_file_location("bc_test_original_loader", loader_path)
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    original, _ = loader.load_model(args.source, args.checkpoint)
    original.eval()
    original_state = {k: v.clone() for k, v in original.state_dict().items()}
    rng_before = torch.random.get_rng_state().clone()
    complete = attach_bc_tpro(copy.deepcopy(original), "bc_tpro")
    checked("attachment_preserves_cpu_rng", torch.equal(rng_before, torch.random.get_rng_state()))
    single = attach_bc_tpro(copy.deepcopy(original), "single_dynamic")
    checked("only_single_TPro_attached", isinstance(complete.TPro, BC_TPro)
            and not hasattr(complete, "TPro1"))
    checked("all_original_state_names_and_values_preserved",
            all(k in complete.state_dict() and torch.equal(v, complete.state_dict()[k])
                for k, v in original_state.items()))
    checked("same_initial_state_for_two_variants",
            set(complete.state_dict()) == set(single.state_dict())
            and all(torch.equal(v, single.state_dict()[k]) for k, v in complete.state_dict().items()))
    checked("gamma_zero_query_key_nonzero", complete.TPro.gamma.item() == 0
            and bool(complete.TPro.query.weight.abs().sum() > 0)
            and bool(complete.TPro.key.weight.abs().sum() > 0))

    # Real checkpoint, actual complete network, declared T=40 and >=112 crop.
    x = torch.randn(1, 1, 40, 112, 112)
    with torch.no_grad():
        expected = original(x)
        actual = complete(x)
        actual_single = single(x)
    checked("real_model_output_tuple_contract", isinstance(actual, tuple)
            and len(actual) == 2 and actual[1].shape == (1, 40, 112, 112))
    for i in range(2):
        checked(f"gamma_zero_exact_full_output_{i}", torch.equal(actual[i], expected[i]),
                max_abs_error=float((actual[i] - expected[i]).abs().max()))
        checked(f"gamma_zero_exact_single_output_{i}", torch.equal(actual_single[i], expected[i]),
                max_abs_error=float((actual_single[i] - expected[i]).abs().max()))
    checked("eval_does_not_cache_aux_graph", complete.TPro._reference_cache is None)
    del x, expected, actual, actual_single

    module = complete.TPro
    observed = torch.randn(2, 1, 1, 40, 32)
    probe_value = torch.randn_like(observed)
    with torch.no_grad():
        equal_value = module.correction(observed, observed)
        equal_operator = module.correction(observed, observed, value=probe_value)
        single_probe = single.TPro.correction(observed, observed, value=probe_value)
    checked("equal_background_zero_relative_value", torch.count_nonzero(equal_value) == 0)
    checked("equal_associations_cancel_nonzero_probe_value", torch.count_nonzero(equal_operator) == 0)
    checked("nonzero_probe_is_informative", single_probe.abs().sum() > 0)
    checked("shared_generator_has_no_background_parameter_copy",
            not any("bg_query" in n or "bg_key" in n for n, _ in module.named_parameters()))

    # Independent explicit A on a tiny tensor checks only algebra, while the
    # production path must never materialize a full 40x40 matrix per pixel.
    with torch.no_grad():
        q, k = module.factors(observed)
        dense = torch.matmul(q, k.transpose(-1, -2)) / (40 * 2)
        factored = module.association_apply(observed, probe_value)
        dense_value = torch.matmul(dense, probe_value)
    checked("factorized_apply_matches_explicit_algebra", torch.allclose(factored, dense_value, atol=1e-7, rtol=1e-5),
            max_abs_error=float((factored-dense_value).abs().max()))

    # Center exclusion is tested after accounting for the stem receptive field:
    # a raw +/-23 patch could affect pre-TPro centers only within +/-32.
    zero = torch.zeros(1, 32, 40, 112, 112)
    changed = zero.clone()
    changed[..., 56-32:56+33, 56-32:56+33] = 1
    context_zero, count = module.reference_context(zero)
    context_changed, _ = module.reference_context(changed)
    checked("ring_no_raw_center_region_leak", torch.equal(context_zero[..., 56, 56], context_changed[..., 56, 56]))
    checked("ring_has_eight_center_samples", int(count[..., 56, 56]) == 8)
    ring_input = zero.clone()
    ring_input[..., 56, 96] = 8
    ring_context, _ = module.reference_context(ring_input)
    checked("declared_ring_source_is_used", torch.equal(ring_context[..., 56, 56], torch.ones_like(ring_context[..., 56, 56])))
    ones_context, count = module.reference_context(torch.ones_like(zero))
    checked("valid_count_normalization_at_boundaries", torch.all(ones_context[count.expand_as(ones_context) > 0] == 1))
    small_context, small_count = module.reference_context(torch.ones(1, 32, 40, 16, 16))
    checked("no_context_no_reflection_or_wrap", torch.count_nonzero(small_context) == 0 and torch.count_nonzero(small_count) == 0)
    histogram = torch.bincount(count.to(torch.int64).flatten(), minlength=9).tolist()
    del zero, changed, context_zero, context_changed, ring_input, ring_context, ones_context, small_context

    background = torch.ones(1, 40, 112, 112, dtype=torch.bool)
    mask_before = background.clone()
    background[:, 20, 56, 56] = False
    unchanged = background.clone()
    valid = module.reference_valid_mask(background, count)
    checked("auxiliary_mask_does_not_mutate_detection_mask", torch.equal(background, unchanged))
    checked("auxiliary_RF_excludes_space_and_time", not bool(valid[0, 0, 12:29, 47:66, 47:66].any()))
    checked("auxiliary_RF_keeps_outside_temporal_neighborhood", bool(valid[0, 0, 11, 56, 56]))
    checked("auxiliary_boundary_is_invalid", not bool(valid[:, :, :8].any())
            and not bool(valid[:, :, -8:].any()) and not bool(valid[..., :9, :].any()))
    del valid, unchanged, mask_before

    # Differentiable standalone TPro test with the real original submodule.
    module.train()
    for part in module.modules():
        if isinstance(part, torch.nn.modules.batchnorm._BatchNorm):
            part.eval()
    feature_input = torch.randn(1, 112, 112, 32, 40, requires_grad=True)
    prediction = module(feature_input)
    detection_loss = prediction.square().mean()
    reference_loss, reference_stats = module.reference_loss(background)
    reference_params = list(module.reference.parameters())
    association_params = list(module.association_norm.parameters()) + list(module.query.parameters()) + list(module.key.parameters())
    old_params = list(module.QK_heads.parameters()) + list(module.norm1.parameters()) + list(module.conv.parameters())
    detached_from_detection = torch.autograd.grad(detection_loss, reference_params, retain_graph=True, allow_unused=True)
    checked("detection_loss_cannot_train_reference", all(g is None for g in detached_from_detection))
    detached_from_reference = torch.autograd.grad(reference_loss, [feature_input] + old_params + association_params + [module.gamma],
                                                retain_graph=True, allow_unused=True)
    checked("reference_loss_cannot_train_backbone_or_dynamic_branch", all(g is None for g in detached_from_reference))
    initial_gradients = torch.autograd.grad(detection_loss, [module.gamma] + association_params,
                                          retain_graph=True, allow_unused=True)
    checked("gamma_has_nonzero_first_step_gradient", initial_gradients[0] is not None
            and torch.isfinite(initial_gradients[0]) and initial_gradients[0].abs() > 0,
            gradient=float(initial_gradients[0]))
    checked("association_zero_detection_gradient_at_gamma_zero",
            all(g is None or torch.count_nonzero(g) == 0 for g in initial_gradients[1:]))
    (detection_loss + reference_loss).backward()
    checked("reference_learns_at_gamma_zero", any(p.grad is not None and p.grad.abs().sum() > 0 for p in reference_params))
    checked("initial_gradients_finite", all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in module.parameters()))
    checked("reference_valid_positions_exist", all(n > 0 for n in reference_stats["reference_valid_positions_per_window"]))
    checked("reference_cache_consumed", module._reference_cache is None)
    del prediction, detection_loss, reference_loss, feature_input, detached_from_detection, detached_from_reference, initial_gradients
    module.zero_grad(set_to_none=True)
    with torch.no_grad():
        module.gamma.fill_(0.1)
    next_feature = torch.randn(1, 112, 112, 32, 40, requires_grad=True)
    next_output = module(next_feature)
    next_output.square().mean().backward()
    checked("query_key_trainable_once_gamma_opens", module.query.weight.grad is not None
            and module.key.weight.grad is not None and module.query.weight.grad.abs().sum() > 0
            and module.key.weight.grad.abs().sum() > 0)
    checked("reference_still_no_detection_gradient", all(p.grad is None for p in module.reference.parameters()))
    module._reference_cache = None
    checked("tested_sources_unchanged_during_checks",
            source_hashes == {name: hashlib.sha256(path.read_bytes()).hexdigest()
                              for name, path in source_paths.items()})

    result = {
        "status": "PASS", "device": "cpu", "torch": str(torch.__version__),
        "tested_source_sha256": source_hashes,
        "model_config": MODEL_CONFIG, "checks": checks,
        "ring_count_histogram_112": {str(i): n for i, n in enumerate(histogram)},
        "reference_stats": reference_stats,
        "original_parameter_count": sum(p.numel() for p in original.parameters()),
        "candidate_parameter_count": sum(p.numel() for p in complete.parameters()),
        "single_parameter_count": sum(p.numel() for p in single.parameters()),
        "seconds": time.perf_counter() - started,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: result[k] for k in ["status", "device", "original_parameter_count", "candidate_parameter_count", "single_parameter_count", "seconds"]}
                     | {"passed_checks": len(checks)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
