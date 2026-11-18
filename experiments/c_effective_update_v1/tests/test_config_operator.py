"""Only new coordinate/ConFIG/fallback semantics; no real-model update."""
import copy
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

E = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_effective_update_config", E/"code/config_operator.py")
op = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = op
spec.loader.exec_module(op)


def parameters():
    return [("first", torch.nn.Parameter(torch.tensor([1., 2.]))),
            ("second", torch.nn.Parameter(torch.tensor([3.]))),
            ("unused", torch.nn.Parameter(torch.tensor([4.])))]


def independent_case():
    named = [("x", torch.nn.Parameter(torch.zeros(5)))]
    rows = torch.tensor([[1., .2, .1, 0., .1], [.1, 2., .3, .1, 0.], [.2, .1, 3., 0., .2]])
    return named, {key: (row,) for key, row in zip(op.GROUPS, rows)}, rows


def test_fixed_coordinates_zero_single_loss_None_but_preserve_all_loss_None():
    named = parameters()
    grads = {"fit": (torch.tensor([1., 2.]), None, None),
             "keep": (None, torch.tensor([3.]), None),
             "track": (torch.tensor([4., 5.]), None, None)}
    got, info = op.combine_gradients(named, grads, mode="sum")
    torch.testing.assert_close(got[0], torch.tensor([5., 7.]))
    torch.testing.assert_close(got[1], torch.tensor([3.]))
    assert got[2] is None and info["parameter_active_any_loss"] == [True, True, False]
    op.apply_gradients(named, got)
    assert named[2][1].grad is None


def test_normal_candidate_is_original_official_direction_at_group_sum_norm():
    named, gradients, rows = independent_case()
    got, info = op.combine_gradients(named, gradients)
    raw = op._official.ConFIG_update(rows, use_least_square=True, losses=None)
    sum_norm = ((rows[0]+rows[1])+rows[2]).double().norm()
    expected = (raw.double()*sum_norm/raw.double().norm()).float()
    assert info["used_mode"] == "config" and info["fallback_reason"] is None
    torch.testing.assert_close(got[0], expected, rtol=2e-5, atol=2e-6)
    assert abs(got[0].double().norm()/sum_norm-1) < 1e-6
    assert min(info["candidate_group_cosines"]) > 0


def test_zero_sum_retains_explicit_zero_and_adam_momentum_step():
    p = torch.nn.Parameter(torch.tensor([2., 3.]))
    optimizer = torch.optim.Adam([p], lr=.01, weight_decay=.1)
    p.grad = torch.tensor([.5, -.5]);optimizer.step();optimizer.zero_grad(set_to_none=True)
    before = p.detach().clone();step = float(optimizer.state[p]["step"])
    grads = {"fit": (torch.tensor([1., 0.]),), "keep": (torch.tensor([-1., 0.]),), "track": (torch.zeros(2),)}
    combined, info = op.combine_gradients([("p", p)], grads)
    assert info["fallback_reason"] == "zero_group_sum" and combined[0] is not None
    op.apply_gradients([("p", p)], combined);optimizer.step()
    assert float(optimizer.state[p]["step"]) == step+1
    assert not torch.equal(before, p)


def test_joint_None_preserves_adam_parameter_and_state_despite_momentum_and_decay():
    named = parameters();params = [p for _, p in named]
    optimizer = torch.optim.Adam(params, lr=.01, weight_decay=.1)
    for p in params:p.grad = torch.ones_like(p)
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    before = params[2].detach().clone();state = copy.deepcopy(optimizer.state[params[2]])
    grads = {"fit": (torch.tensor([1., 2.]), None, None),
             "keep": (None, torch.tensor([3.]), None),
             "track": (torch.tensor([.1, .2]), None, None)}
    combined, _ = op.combine_gradients(named, grads, mode="sum")
    op.apply_gradients(named, combined);optimizer.step()
    assert torch.equal(params[2], before)
    for key, value in state.items():torch.testing.assert_close(optimizer.state[params[2]][key], value, rtol=0, atol=0)


@pytest.mark.parametrize("case,reason", [("none", "all_losses_unused"), ("zero", "zero_loss_gradient"), ("rank", "rank_deficient_or_ill_conditioned")])
def test_explicit_pre_solver_fallbacks(case, reason):
    named, gradients, rows = independent_case()
    if case == "none":gradients = {k: (None,) for k in op.GROUPS}
    elif case == "zero":gradients["track"] = (torch.zeros(5),)
    else:gradients = {k: (rows[0]*(i+1),) for i, k in enumerate(op.GROUPS)}
    got, info = op.combine_gradients(named, gradients)
    baseline, _ = op.combine_gradients(named, gradients, mode="sum")
    assert info["fallback_reason"] == reason and not info["official_called"]
    if baseline[0] is None:assert got[0] is None
    else:torch.testing.assert_close(got[0], baseline[0], rtol=0, atol=0)


@pytest.mark.parametrize("case,reason", [("nan", "nonfinite_candidate"), ("zero", "zero_candidate"), ("negative", "nonpositive_or_numerically_weak_projection"), ("unequal", "unreliable_equal_projection_solution"), ("raise", "official_solver_exception")])
def test_solver_failures_fall_back_to_unchanged_sum(monkeypatch, case, reason):
    named, gradients, rows = independent_case()
    def fake(values, **kwargs):
        if case == "raise":raise RuntimeError("test solver failure")
        if case == "nan":return torch.full_like(values[0], float("nan"))
        if case == "zero":return torch.zeros_like(values[0])
        if case == "negative":return -values.sum(0)
        return values[0]
    monkeypatch.setattr(op._official, "ConFIG_update", fake)
    got, info = op.combine_gradients(named, gradients)
    baseline, _ = op.combine_gradients(named, gradients, mode="sum")
    assert info["fallback_reason"] == reason and info["official_called"]
    torch.testing.assert_close(got[0], baseline[0], rtol=0, atol=0)


@pytest.mark.parametrize("error", ["nonfinite", "shape", "length", "keys", "dtype"])
def test_malformed_original_gradients_are_errors_not_silent_fallback(error):
    named, gradients, _ = independent_case()
    if error == "nonfinite":gradients["fit"] = (torch.full((5,), float("nan")),)
    elif error == "shape":gradients["fit"] = (torch.zeros(4),)
    elif error == "length":gradients["fit"] = ()
    elif error == "keys":gradients["other"] = gradients.pop("track")
    else:gradients["fit"] = (torch.zeros(5, dtype=torch.float64),)
    with pytest.raises((ValueError, FloatingPointError)):
        op.combine_gradients(named, gradients)


def test_original_official_EqualWeight_shape_and_dtype_contract():
    rows = torch.eye(3, 5)
    weights = op._official.EqualWeight().get_weights(gradients=rows, device=rows.device)
    assert weights.shape == (3,) and weights.dtype == torch.float32
    torch.testing.assert_close(weights, torch.ones(3), rtol=0, atol=0)
    # The original EqualWeight uses torch's default dtype; a double caller
    # cannot silently assume dtype matching. Wrapper deliberately keeps FP32.
    with pytest.raises(RuntimeError):
        op._official.ConFIG_update(rows.double())
