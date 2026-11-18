"""Observe the actual encoder inputs, center path and full/tiled S3 prediction entry points."""
from contextlib import ExitStack
from dataclasses import replace

import torch
from torch.nn import functional as F

from ..engine.predict import OutputWrites
from ..geometry.sampler import SampledEvidence
from ..geometry.velocity_grid import make_velocity_grid
from ..models.hypothesis_detector import HypothesisEvidenceDetector
from .check_s2a import fixture
from .s2c_prediction import fixed_threshold_sample


MODES = ("additive_only", "raw_joint", "interaction")


def _controlled(device):
    # First two queries have equal margins and opposite checkerboard residuals.
    # The third has no complete trajectory, the fourth has a single valid one.
    t = torch.arange(16, device=device).float()
    valid = torch.zeros((1, 4, 3, 3), dtype=torch.bool, device=device)
    valid[0, :2, 1, :2] = True
    valid[0, 3, 1, 1] = True
    v = torch.zeros(3, 3, device=device)
    v[1, :2] = torch.tensor([-1., 1.], device=device)
    main = (t[:, None, None] / 16 + v + 1).expand(1, 4, 9, 16, 3, 3).clone()
    cross = ((t.remainder(2) * 2 - 1)[:, None, None] * v).expand_as(main).clone()
    cross[:, 1].neg_()
    cross[:, 2:] = 0
    support = valid[:, :, None, None]
    main.masked_fill_(~support, 0)
    cross.masked_fill_(~support, 0)
    values = (main + cross).masked_fill(~support, float("nan")).requires_grad_()
    sample_valid = valid.unsqueeze(2).expand(1, 4, 16, 3, 3).clone()
    sample_valid[:, :, 0] = True  # partial observations do not define a legal column
    offsets = torch.cat((t[:8] - 8, t[8:] - 7))[None]
    grid = make_velocity_grid([-1, 0, 1], [-1, 0, 1], device=device)
    evidence = SampledEvidence(values, sample_valid, valid, offsets, torch.zeros(1, 4, 2, device=device), grid)
    center = torch.linspace(-1, 1, 8, device=device).expand(1, 4, 8).clone().requires_grad_()
    return evidence, center, main, cross


def _inspect_inputs(model, mode, device):
    evidence, center, main, cross = _controlled(device)
    encoder = model.video_aggregator.encoder
    before, after = {}, {}
    def prehook(key):
        def capture(module, inputs):
            before[key] = inputs[0].detach()
        return capture
    def posthook(key):
        def capture(module, inputs, output):
            after[key] = output.detach()
        return capture
    with ExitStack() as stack:
        for name in ("joint", "fusion"):
            stack.enter_context(getattr(encoder, name).register_forward_pre_hook(prehook(name)))
        for name in ("amplitude", "temporal", "velocity", "joint"):
            stack.enter_context(getattr(encoder, name).register_forward_hook(posthook(name)))
        output = encoder(evidence, center)
    expect = {"additive_only": main, "raw_joint": main + cross, "interaction": cross}[mode]
    torch.testing.assert_close(before["joint"], expect, atol=0, rtol=0)
    valid = evidence.velocity_valid
    observed = valid.any(dim=(-2, -1)).unsqueeze(-1)
    coverage = valid.float().mean(dim=(-2, -1)).unsqueeze(-1)
    expected_fusion = torch.cat((F.silu(after["amplitude"]) * observed, after["temporal"] * observed,
                                 after["velocity"], after["joint"], center.detach(), coverage), dim=-1)
    joined = before["fusion"]
    assert joined.shape[-1] == 49
    torch.testing.assert_close(joined, expected_fusion, atol=0, rtol=0)
    torch.testing.assert_close(joined[:, 0, :24], joined[:, 1, :24], atol=0, rtol=0)
    assert torch.count_nonzero(joined[:, 2, :40]).item() == 0
    if mode == "interaction":
        assert torch.count_nonzero(joined[:, 3, 24:40]).item() == 0
    delta = (joined[:, 0, 24:40] - joined[:, 1, 24:40]).abs().max().item()
    assert delta == 0 if mode == "additive_only" else delta > 0

    with torch.no_grad():
        repacked = replace(evidence, values=evidence.values.nan_to_num(nan=32768))
        torch.testing.assert_close(encoder(repacked, center), output, atol=0, rtol=0)
    grad, grad_center = torch.autograd.grad(output.square().sum(), (evidence.values, center))
    rejected = (~valid[:, :, None, None]).expand_as(grad)
    assert torch.isfinite(grad).all() and not torch.count_nonzero(grad[rejected]).item()
    assert torch.count_nonzero(grad[~rejected]).item() and torch.count_nonzero(grad_center[:, 2]).item()
    # Only query zero is perturbed, along a pure interaction with zero margins.
    sensitivity = (grad[:, 0] * cross[:, 0]).sum().abs().item()
    assert sensitivity < 1e-6 if mode == "additive_only" else sensitivity > 1e-8
    return {"joint_input_matches": True, "positive_negative_joint_feature_difference": delta,
            "zero_marginal_perturbation_gradient": sensitivity, "empty_center_gradient_l1": float(grad_center[:, 2].abs().sum()),
            "invalid_gradient_nonzeros": 0}


@torch.no_grad()
def _center_and_entry(models, device):
    clip = fixture(device)
    model = models["interaction"]
    def record_center_path(sample):
        rows = []
        def observe(module, inputs):
            evidence, center = inputs
            assert torch.equal(evidence.time_offsets, torch.cat((clip.time_offsets[:, :8], clip.time_offsets[:, 9:]), 1))
            assert torch.equal(evidence.velocity_valid, evidence.sample_valid.all(2))
            rows.append((evidence.values.detach(), center.detach()))
        with model.video_aggregator.encoder.register_forward_pre_hook(observe):
            output = model(sample)
        return torch.cat([r[0] for r in rows], 1), torch.cat([r[1] for r in rows], 1), output.logits
    original, center, logits = record_center_path(clip)
    changed = clip.frames.clone()
    changed[:, 8] = 1 - changed[:, 8]
    after, new_center, new_logits = record_center_path(replace(clip, frames=changed))
    assert torch.equal(original, after)
    d_center = (center - new_center).abs().max().item()
    d_logit = (logits - new_logits).abs().max().item()
    assert d_center > 0 and d_logit > 0
    entries = {}
    for mode, selected in models.items():
        pred = selected(clip)
        assert pred.output_frame_id == clip.output_frame_id and pred.logits.shape == (1, 1, 9, 13)
        assert pred.logits.dtype == torch.float32 and torch.isfinite(pred.logits).all()
        scored = fixed_threshold_sample(pred.output_frame_id[0], pred.logits[0, 0].cpu().numpy(), [(4., 5.)])
        assert scored["gt"] == 1
        entries[mode] = {"logits_hw": list(pred.logits.shape[-2:]), "frame_id": pred.output_frame_id[0]}
    return {"center_to_evidence_max_change": 0.0, "center_feature_max_change": d_center,
            "center_to_output_max_change": d_logit, "detector_and_s2c_evaluation_entries": entries}


@torch.no_grad()
def _spatial(model, device):
    height, width = 13, 43
    small = fixture(device)
    frames = torch.arange(17 * height * width, device=device).remainder(97).float().reshape(1, 17, 1, height, width) / 97
    valid = torch.ones_like(frames, dtype=torch.bool)
    valid[:, 1, :, 5:7, 12:15] = False
    meta = replace(small.scene_meta[0], original_hw=(height, width), core_xyxy=(0, 0, width, height))
    clip = replace(small, frames=frames, pixel_valid=valid, scene_meta=(meta,))
    whole = model(clip)
    results = []
    for core in ((13, 15), (8, 12)):
        planned = model.tiles((height, width), core)
        cropped_sources = sum(tile.input_hw != (height, width) for tile in planned)
        assert cropped_sources > 0
        trace = OutputWrites()
        with trace:
            tiled = model.predict_tiled(clip, core_hw=core)
        coverage = trace.coverage(tiled.logits)
        assert coverage["min_writes"] == coverage["max_writes"] == 1
        assert coverage["core_count"] == len(planned)
        torch.testing.assert_close(tiled.logits, whole.logits, atol=1e-5, rtol=1e-4)
        results.append({"core_hw": list(core), "actual_core_writes": coverage["core_count"],
                        "cropped_sources": cropped_sources,
                        "max_abs_delta": (tiled.logits - whole.logits).abs().max().item()})
    original_chunk = model.video_aggregator.query_chunk
    try:
        model.video_aggregator.query_chunk = 47
        rechunked = model(clip)
    finally:
        model.video_aggregator.query_chunk = original_chunk
    torch.testing.assert_close(rechunked.logits, whole.logits, atol=1e-5, rtol=1e-4)
    return {"pixels_per_pass": height * width, "tiled": results, "query_chunk_sizes": [original_chunk, 47],
            "query_chunk_max_abs_delta": (rechunked.logits - whole.logits).abs().max().item()}


def run_checks(device):
    models = {}
    for mode in MODES:
        torch.manual_seed(46)
        models[mode] = HypothesisEvidenceDetector(mode=mode).to(device).eval()
    reference = dict(models["interaction"].named_parameters())
    for model in models.values():
        actual = dict(model.named_parameters())
        assert actual.keys() == reference.keys()
        assert all(torch.equal(reference[k], actual[k]) for k in reference)
        assert torch.equal(model.video_aggregator.velocities_xy, models["interaction"].video_aggregator.velocities_xy)
    return {"status": "PASS", "equal_parameter_count": sum(p.numel() for p in reference.values()),
            "encoder_paths": {m: _inspect_inputs(model, m, device) for m, model in models.items()},
            "center_and_interface": _center_and_entry(models, device), "spatial": _spatial(models["interaction"], device)}
