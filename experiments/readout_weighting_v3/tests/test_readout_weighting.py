"""Tests only the v3 changes, using the unchanged v2 adapter as a comparator."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

E = Path(__file__).resolve().parents[1]
ROOT = E.parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


new = load("_v3_test_loss", E / "train_modules/readout_weighting.py")
old = load("_v3_test_v2_loss", ROOT / "experiments/track_contrast_v2/train_modules/temporal_risk.py")
official = load("_v3_test_official", E / "resources/official_kernel.py")
reference = load("_v3_test_reference", E / "reference/DeepProPlus_C_readout_v3/readout_weighting.py")


def fixture(dtype=torch.float64):
    generator = torch.Generator().manual_seed(4613)
    z = (torch.randn((8, 12, 6, 6), dtype=dtype, generator=generator)*4-3).requires_grad_()
    targets, background, neg = [], [], []
    for w, count in [(0, 12), (1, 8)]:
        for t in range(count):
            targets.append([w, t, 0, 0, 2, 2])
            neg.append([len(background), len(background)+1])
            background += [[w, t, 3, 3, 5, 5], [w+4, t, 3, 3, 5, 5]]
    # Includes overlapping segment exposure and a window with fewer segments.
    segments = [list(range(8)), list(range(4, 12)), list(range(12, 20)), [-1]*8]
    windows = [0, 0, 1, 2]
    result = old.temporal_risk_loss(z, targets, background, neg, segments, windows)
    return z, result, segments, windows


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_new_unit_independently_matches_original_v2_frame_value_and_pixel_gradient(dtype):
    z, risk, segments, windows = fixture(dtype)
    result = new.readout_weighting_loss(risk, segments, windows, mode="unit")
    assert result["loss"] is not risk["loss_frame"]
    torch.testing.assert_close(result["loss"], risk["loss_frame"], rtol=0, atol=0)
    torch.testing.assert_close(result["per_segment_total"], risk["per_segment_frame"], rtol=0, atol=0)
    ours, = torch.autograd.grad(result["loss"], z, retain_graph=True)
    prior, = torch.autograd.grad(risk["loss_frame"], z)
    torch.testing.assert_close(ours, prior, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(1, 1, 1), (1, 7, 1), (2, 11, 3)])
def test_confidence_matches_executed_official_mal_all_background_value_and_gradient(dtype, shape):
    q = torch.linspace(-20, 20, int(np.prod(shape)), dtype=dtype).reshape(shape).requires_grad_()
    q2 = q.detach().clone().requires_grad_()
    got = new.readout_terms(torch.zeros_like(q.reshape(shape[0], -1)), q.reshape(shape[0], -1), mode="confidence")["negative"].mean()
    author = official.author_all_background(q2)
    g, = torch.autograd.grad(got, q)
    ga, = torch.autograd.grad(author, q2)
    rtol, atol = (2e-6, 1e-7) if dtype == torch.float32 else (1e-12, 1e-12)
    torch.testing.assert_close(got, author, rtol=rtol, atol=atol)
    torch.testing.assert_close(g, ga, rtol=rtol, atol=atol)


@pytest.mark.parametrize("mode", ["unit", "confidence", "matched_confidence"])
def test_positive_values_and_extremely_weak_target_gradients_are_unchanged(mode):
    qp = torch.tensor([[-1000., -80., -10., 0., 5.]], requires_grad=True, dtype=torch.float64)
    qm = torch.tensor([[-1000., -80., -2., 0., 5.]], requires_grad=True, dtype=torch.float64)
    got = new.readout_terms(qp, qm, mode=mode)
    unit = new.readout_terms(qp, qm, mode="unit")
    gp, = torch.autograd.grad(got["positive"].sum(), qp, retain_graph=True)
    gu, = torch.autograd.grad(unit["positive"].sum(), qp)
    torch.testing.assert_close(got["positive"], unit["positive"], rtol=0, atol=0)
    torch.testing.assert_close(gp, gu, rtol=0, atol=0)
    assert gp[0, 0].item() == -1. and gp[0, 1].item() == -1.


@pytest.mark.parametrize("mode", ["confidence", "matched_confidence"])
def test_weights_and_group_normalization_are_detached(mode):
    q = torch.tensor([[-3., 0., 3.]], dtype=torch.float64, requires_grad=True)
    out = new.readout_terms(torch.zeros_like(q), q, mode=mode)
    for key in ("negative_weight", "negative_log_weight", "group_log_scale"):
        assert not out[key].requires_grad and out[key].grad_fn is None
    grad, = torch.autograd.grad(out["negative"].sum(), q)
    torch.testing.assert_close(grad, out["negative_weight"]*q.sigmoid(), rtol=1e-12, atol=1e-12)
    if mode == "confidence":
        other = (q.sigmoid().square()*F.softplus(q)).sum()
        focal_grad, = torch.autograd.grad(other, q)
        assert not torch.allclose(grad, focal_grad)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_N_weighted_reduction_budget_and_hard_negative_redistribution(dtype):
    q = torch.tensor([[-4., 0., 2., float("nan")], [-3., -1., 3., -9.]], dtype=dtype, requires_grad=True)
    valid = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool)
    mu = torch.tensor([[1., 4., 2., 7.], [2., 1., 5., 0.]], dtype=dtype)
    n = new.readout_terms(torch.zeros_like(q), q, negative_valid=valid, sample_weight=mu, mode="matched_confidence")
    f = new.readout_terms(torch.zeros_like(q), q, negative_valid=valid, sample_weight=mu, mode="unit")
    gn, = torch.autograd.grad((n["negative"]*mu).sum(), q, retain_graph=True)
    gf, = torch.autograd.grad((f["negative"]*mu).sum(), q)
    torch.testing.assert_close(gn.sum(1), gf.sum(1), rtol=2e-6, atol=1e-12)
    assert not torch.allclose(gn, gf)
    assert gn[0, 0] < gf[0, 0] and gn[0, 2] > gf[0, 2]
    assert torch.isfinite(gn).all() and gn[0, 3] == 0 and gn[1, 3] == 0


def test_segment_normalization_preserves_original_overlap_and_fixed_window_budget():
    z, risk, segments, windows = fixture()
    n = new.readout_weighting_loss(risk, segments, windows, mode="matched_confidence")
    f = new.readout_weighting_loss(risk, segments, windows, mode="unit")
    gn, = torch.autograd.grad(n["loss_negative"], risk["q_negative"], retain_graph=True)
    gf, = torch.autograd.grad(f["loss_negative"], risk["q_negative"])
    torch.testing.assert_close(gn.sum(), gf.sum(), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(n["diagnostics"]["unit_negative_readout_gradient_l1"], gf.abs().sum(), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(n["diagnostics"]["weighted_negative_readout_gradient_l1"], gn.abs().sum(), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(n["group_reduction_weight"], torch.tensor([1/128, 1/128, 1/64, 0.], dtype=torch.float64))
    assert not torch.allclose(gn, gf)
    torch.testing.assert_close(n["loss_positive"], f["loss_positive"], rtol=0, atol=0)
    assert n["diagnostics"]["observations_exposed"] == 24
    assert n["diagnostics"]["windows_active"] == 2
    assert torch.count_nonzero(n["per_window_total"][2:]) == 0
    torch.testing.assert_close(n["loss"], (n["per_segment_total"][:2].mean()+n["per_segment_total"][2])/8)
    # Shared observation 4 has different weights in its two overlapping groups.
    assert n["negative_weight"][0, 4] != n["negative_weight"][1, 0]


@pytest.mark.parametrize("level", [-1000., -80., -5., 0., 80.])
def test_single_valid_background_N_reduces_to_F_value_and_gradient(level):
    q = torch.tensor([[level, 1000.]], requires_grad=True)
    valid = torch.tensor([[True, False]])
    n = new.readout_terms(torch.zeros_like(q), q, negative_valid=valid, mode="matched_confidence")
    f = new.readout_terms(torch.zeros_like(q), q, negative_valid=valid, mode="unit")
    torch.testing.assert_close(n["negative_weight"], torch.tensor([[1., 0.]]), rtol=0, atol=0)
    torch.testing.assert_close(n["negative"], f["negative"], rtol=0, atol=0)
    gn, = torch.autograd.grad(n["negative"].sum(), q, retain_graph=True)
    gf, = torch.autograd.grad(f["negative"].sum(), q)
    torch.testing.assert_close(gn, gf, rtol=0, atol=0)


@pytest.mark.parametrize("level", [-100000., -10000., -1000., -80., 0., 40.])
def test_uniform_background_N_is_unit_even_when_float32_logsumexp_cancels(level):
    q = torch.full((2, 8), level, requires_grad=True)
    n = new.readout_terms(torch.zeros_like(q), q, mode="matched_confidence")
    torch.testing.assert_close(n["negative_weight"], torch.ones_like(q), rtol=0, atol=0)
    n["negative"].sum().backward()
    assert torch.isfinite(n["negative"]).all() and torch.isfinite(q.grad).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_extreme_logits_and_masked_nonfinite_padding(dtype):
    qp = torch.tensor([[-1000., 1000., float("nan")]], dtype=dtype, requires_grad=True)
    qm = torch.tensor([[-1000., 1000., float("inf")]], dtype=dtype, requires_grad=True)
    valid = torch.tensor([[True, True, False]])
    n = new.readout_terms(qp, qm, valid, valid, mode="matched_confidence")
    loss = (n["positive"]+n["negative"]).sum()
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(qp.grad).all() and torch.isfinite(qm.grad).all()
    assert qp.grad[0, 0] == -1 and qp.grad[0, 2] == 0 and qm.grad[0, 2] == 0


@pytest.mark.parametrize("mode", ["unit", "confidence", "matched_confidence"])
def test_empty_segments_keep_fixed_zero_and_original_detection_gradient(mode):
    z = torch.randn((8, 4, 6, 6), requires_grad=True)
    segments = np.empty((0, 8), dtype=int)
    risk = old.temporal_risk_loss(z, [], [], np.empty((0, 2), dtype=int), segments, [])
    got = new.readout_weighting_loss(risk, segments, [], mode=mode)
    assert got["loss"].item() == 0 and torch.isfinite(got["loss"])
    g, = torch.autograd.grad(z.square().mean()+got["loss"], z, retain_graph=True)
    expected, = torch.autograd.grad(z.square().mean(), z)
    torch.testing.assert_close(g, expected, rtol=0, atol=0)


def test_all_invalid_negative_group_keeps_positive_and_zero_negative_gradient():
    qp = torch.tensor([[-30., -20.]], requires_grad=True)
    qm = torch.tensor([[float("nan"), float("inf")]], requires_grad=True)
    got = new.readout_terms(qp, qm, negative_valid=torch.zeros_like(qm, dtype=torch.bool), mode="matched_confidence")
    (got["positive"].sum()+got["negative"].sum()).backward()
    assert (qp.grad < -.99).all() and torch.equal(qm.grad, torch.zeros_like(qm))
    assert torch.isfinite(got["group_log_scale"]).all()


def test_masks_cannot_silently_change_original_exposure_or_fixed_window_reduction():
    z, risk, segments, windows = fixture()
    changed = [row[:] for row in segments]; changed[0][0] = 1
    with pytest.raises(ValueError, match="exposure"):
        new.readout_weighting_loss(risk, changed, windows, mode="confidence")
    with pytest.raises(ValueError, match="reduction"):
        new.readout_weighting_loss(risk, segments, [0, 1, 1, 2], mode="confidence")


def test_common_case_formula_agrees_with_user_reference_and_D_only_reduces_budget():
    qp = torch.tensor([[-10., 0., 3.]], dtype=torch.float64)
    qm = torch.tensor([[-8., -1., 4.]], dtype=torch.float64, requires_grad=True)
    for mode in ("unit", "confidence", "matched_confidence"):
        ours = new.readout_terms(qp, qm, mode=mode)
        prior = reference.readout_terms(qp, qm, mode=mode)
        torch.testing.assert_close(ours["positive"], prior.positive, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(ours["negative"], prior.negative, rtol=1e-12, atol=1e-12)
    d = new.readout_terms(qp, qm, mode="confidence")
    f = new.readout_terms(qp, qm, mode="unit")
    gd, = torch.autograd.grad(d["negative"].sum(), qm, retain_graph=True)
    gf, = torch.autograd.grad(f["negative"].sum(), qm)
    assert (gd <= gf).all() and gd.sum() < gf.sum()
