"""Finite adapter checks; no detection result or official-module reproduction."""
import copy

import pytest
import torch
from torch import nn

from framewise_adapter import (FramewiseFeatureAdapter,
                               FrozenDeepProPlusWithSpatialReference,
                               spatially_tiled_forward)
from haar_reference import SpatialLFPReference


@pytest.mark.parametrize("shape", [(1, 3, 5, 16, 18), (2, 3, 3, 15, 17)])
@pytest.mark.parametrize("chunk", [1, 2, 8, 100])
def test_zero_gate_exact_identity(shape, chunk):
    torch.manual_seed(46)
    model = FramewiseFeatureAdapter(SpatialLFPReference(3), 3, chunk)
    x = torch.randn(shape)
    assert torch.equal(model(x), x)
    assert model(x).stride() == x.stride()


@pytest.mark.parametrize("chunk", [1, 2, 8])
def test_frame_chunks_preserve_batch_time_order(chunk):
    torch.manual_seed(46)
    full = FramewiseFeatureAdapter(SpatialLFPReference(3), 3, 100).double()
    with torch.no_grad():
        full.scale.fill_(0.25)
    blocked = copy.deepcopy(full)
    blocked.frame_chunk = chunk
    x = torch.randn(2, 3, 5, 15, 17, dtype=torch.float64)
    torch.testing.assert_close(blocked(x), full(x), atol=1e-12, rtol=1e-12)
    separate = torch.stack([
        torch.stack([full(x[b:b+1, :, t:t+1])[0, :, 0]
                     for t in range(x.shape[2])], dim=1)
        for b in range(x.shape[0])])
    torch.testing.assert_close(blocked(x), separate, atol=1e-12, rtol=1e-12)


def test_zero_gate_first_gradient_then_active_operator_gradient():
    torch.manual_seed(46)
    model = FramewiseFeatureAdapter(SpatialLFPReference(3), 3, 2)
    x = torch.randn(2, 3, 3, 16, 18)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    model(x).square().mean().backward()
    assert model.scale.grad is not None and torch.isfinite(model.scale.grad)
    assert model.scale.grad.abs().item() > 0
    operator_grads = [p.grad for p in model.operator.parameters()]
    assert operator_grads and all(g is not None for g in operator_grads)
    assert all(torch.count_nonzero(g) == 0 for g in operator_grads)
    optimizer.step()
    assert model.scale.item() != 0
    optimizer.zero_grad(set_to_none=True)
    model(x).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.operator.parameters())
    assert sum(p.grad.abs().sum().item() for p in model.operator.parameters()) > 0


def test_nonfinite_operator_is_not_hidden_by_zero_gate():
    class Nonfinite(nn.Module):
        def forward(self, x):
            return x * float("nan")
    model = FramewiseFeatureAdapter(Nonfinite(), 3)
    assert not torch.isfinite(model(torch.ones(1, 3, 1, 4, 4))).all()


@pytest.mark.parametrize("bad", [torch.ones(1, 2, 3, 4, 4),
                                 torch.ones(1, 3, 4, 4),
                                 torch.ones(1, 3, 0, 4, 4)])
def test_input_validation(bad):
    with pytest.raises(ValueError):
        FramewiseFeatureAdapter(nn.Identity(), 3)(bad)


@pytest.mark.parametrize("kwargs", [{"channels": 0}, {"frame_chunk": 0}])
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        FramewiseFeatureAdapter(nn.Identity(), **kwargs)


@pytest.mark.parametrize("shape", [(1, 3, 3, 54, 58), (2, 3, 2, 55, 59)])
def test_active_lfp_even_phase_tiles(shape):
    torch.manual_seed(46)
    adapter = FramewiseFeatureAdapter(SpatialLFPReference(3), 3, 2).double()
    with torch.no_grad():
        adapter.scale.fill_(0.25)
        adapter.operator.attention.weight.normal_(std=0.02)
    # Return a tuple, as the frozen detector does, to check the common tiler.
    class TupleModel(nn.Module):
        def forward(self, x):
            y = adapter(x)
            return y, y.sum(dim=1)
    x = torch.randn(shape, dtype=torch.float64)
    model = TupleModel()
    with torch.no_grad():
        full = model(x)
        tiled = spatially_tiled_forward(model, x, core_hw=(19, 21), halo=20,
                                       absolute_origin_yx=(2, 4))
    for a, b in zip(full, tiled):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
    with pytest.raises(ValueError, match="even Haar"):
        spatially_tiled_forward(model, x, absolute_origin_yx=(1, 0))


def test_wrapper_keeps_original_bn_frozen_during_training():
    class MockOriginal(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_in = nn.Conv3d(1, 3, 1)
            self.layer1 = nn.BatchNorm3d(3)
            self.TPro = nn.Identity()
            self.conv_out1 = nn.Identity()
            self.conv_out2 = nn.Conv3d(3, 1, 1)
    original = MockOriginal()
    wrapper = FrozenDeepProPlusWithSpatialReference(
        original, FramewiseFeatureAdapter(SpatialLFPReference(3), 3))
    wrapper.train()
    assert wrapper.training and wrapper.feature_refiner.training
    assert not wrapper.original.training
    assert all(not m.training for m in original.modules())
    assert all(not p.requires_grad for p in original.parameters())
