"""Finite implementation tests; these do not establish detection effectiveness."""
import copy
import io

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from temporal_lfp import (FramewiseFeatureAdapter, SpatialLFPReference,
                          TemporalLFPAdapter, haar_forward, haar_inverse)

torch.set_num_threads(2)


def spatial(channels=3, scale=0.0, dtype=torch.float64):
    torch.manual_seed(46)
    model = FramewiseFeatureAdapter(SpatialLFPReference(channels), channels, 8).to(dtype=dtype)
    with torch.no_grad():
        model.scale.fill_(scale)
        model.operator.sigma.fill_(0.83)
        model.operator.attention.weight.normal_(0, 0.025)
    return model


def candidate(mode="temporal", scale=0.23, chunk=8, channels=3):
    result = TemporalLFPAdapter.from_spatial(spatial(channels, scale), mode, chunk)
    if mode == "temporal":
        with torch.no_grad():
            result.operator.attention.weight[:, :2].normal_(0, 0.02)
            result.operator.attention.weight[:, 4:].normal_(0, 0.02)
    return result


def data(shape=(2, 3, 5, 13, 15)):
    generator = torch.Generator().manual_seed(239)
    return torch.randn(shape, generator=generator, dtype=torch.float64) * 0.2


@pytest.mark.parametrize("mode,count", [("spatial", 100), ("temporal", 296)])
def test_parameter_count_and_copy(mode, count):
    original = spatial(32)
    model = TemporalLFPAdapter.from_spatial(original, mode)
    assert sum(p.numel() for p in model.parameters()) == count
    assert set(dict(model.named_parameters())) == {"scale", "operator.sigma", "operator.attention.weight"}
    assert torch.equal(model.scale, original.scale)
    assert torch.equal(model.operator.sigma, original.operator.sigma)
    weight = model.operator.attention.weight
    if mode == "temporal":
        assert torch.count_nonzero(weight[:, :2]) == 0
        assert torch.count_nonzero(weight[:, 4:]) == 0
        weight = weight[:, 2:4]
    assert torch.equal(weight, original.operator.attention.weight)
    assert weight.data_ptr() != original.operator.attention.weight.data_ptr()


@pytest.mark.parametrize("mode", ["spatial", "temporal"])
@pytest.mark.parametrize("hw", [(12, 14), (13, 15), (1, 1)])
def test_zero_gate_is_exact_identity(mode, hw):
    model = TemporalLFPAdapter.from_spatial(spatial(), mode, 2)
    x = data((2, 3, 5, *hw))
    y = model(x)
    assert y.is_contiguous()
    assert torch.equal(x, y)


@pytest.mark.parametrize("scale", [-0.31, 0.27])
@pytest.mark.parametrize("hw", [(12, 14), (13, 15)])
def test_nonzero_gate_matches_verified_spatial_and_center_temporal(scale, hw):
    original = spatial(scale=scale)
    x = data((2, 3, 5, *hw))
    expected = original(x)
    for mode in ("spatial", "temporal"):
        actual = TemporalLFPAdapter.from_spatial(original, mode, 2)(x)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("chunk", [1, 8, 17])
def test_time_chunks_forward_backward_match_full_window(chunk):
    whole = candidate(chunk=40)
    chunked = copy.deepcopy(whole)
    chunked.frame_chunk = chunk
    x = data((2, 3, 19, 9, 11)).requires_grad_()
    xc = x.detach().clone().requires_grad_()
    expected = whole(x)
    actual = chunked(xc)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    probe = torch.linspace(-0.8, 1.1, x.numel(), dtype=x.dtype).reshape_as(x)
    (expected * probe).sum().backward()
    (actual * probe).sum().backward()
    torch.testing.assert_close(xc.grad, x.grad, rtol=1e-10, atol=1e-11)
    for (name, param), (_, other) in zip(whole.named_parameters(), chunked.named_parameters()):
        assert param.grad is not None, name
        torch.testing.assert_close(other.grad, param.grad, rtol=1e-10, atol=1e-11)


def test_video_batch_boundaries_never_mix():
    model = candidate(chunk=3)
    x = data((2, 3, 7, 9, 11))
    together = model(x)
    separated = torch.cat([model(x[i:i + 1]) for i in range(2)])
    torch.testing.assert_close(together, separated, rtol=1e-12, atol=1e-12)
    changed = x.clone()
    changed[1] += 100
    torch.testing.assert_close(model(changed)[0], together[0], rtol=0, atol=0)


def test_true_window_endpoints_replicate_guides():
    model = candidate(chunk=1)
    x = data((1, 3, 1, 9, 11))
    torch.testing.assert_close(model(x), model(x, mode="current_only"), rtol=0, atol=0)


def test_neighbor_guide_has_local_temporal_effect():
    model = candidate(chunk=2)
    x = data((1, 3, 5, 13, 15))
    altered = x.clone()
    altered[:, :, 2] += 2  # Changes the LL descriptions in exactly frame 2.
    baseline, result = model(x), model(altered)
    assert (result[:, :, 1] - baseline[:, :, 1]).abs().max() > 1e-7
    assert (result[:, :, 3] - baseline[:, :, 3]).abs().max() > 1e-7
    torch.testing.assert_close(result[:, :, 0], baseline[:, :, 0], rtol=0, atol=0)
    torch.testing.assert_close(result[:, :, 4], baseline[:, :, 4], rtol=0, atol=0)


def test_neighbor_high_bands_are_not_aggregated():
    model = candidate(chunk=2)
    x = data((1, 3, 3, 12, 14))
    altered = x.clone()
    low, high, hw = haar_forward(x[:, :, 0])
    altered[:, :, 0] = haar_inverse(low, high + 1.0, hw)
    # Neighbor high coefficients change while its LL description is retained.
    torch.testing.assert_close(model(altered)[:, :, 1], model(x)[:, :, 1], rtol=1e-12, atol=1e-12)


def test_current_only_equals_time_sum_spatial_kernel():
    temporal = candidate(chunk=2)
    source = spatial(scale=float(temporal.scale.detach()))
    with torch.no_grad():
        weight = temporal.operator.attention.weight
        source.operator.attention.weight.copy_(weight[:, :2] + weight[:, 2:4] + weight[:, 4:])
    x = data()
    torch.testing.assert_close(temporal(x, mode="current_only"), source(x), rtol=1e-12, atol=1e-12)
    temporal.set_inference_mode("current_only")
    torch.testing.assert_close(temporal(x), source(x), rtol=1e-12, atol=1e-12)


def test_slot_convolution_matches_explicit_conv3d_forward_and_gradients():
    """The implemented six channels exactly map a 2-input, 3-time kernel.

    Temporal endpoints replicate; the 7x7 spatial kernel uses zero padding.
    Both input and kernel gradients are checked with independently stored
    parameters, including endpoints of two distinct videos.
    """
    generator = torch.Generator().manual_seed(731)
    guide3d = torch.randn((2, 2, 5, 9, 11), generator=generator,
                         dtype=torch.float64).requires_grad_()
    weight3d = (torch.randn((1, 2, 3, 7, 7), generator=generator,
                           dtype=torch.float64) * 0.05).requires_grad_()
    guide2d = guide3d.detach().clone().requires_grad_()
    weight2d = weight3d.detach().permute(0, 2, 1, 3, 4).reshape(1, 6, 7, 7).clone().requires_grad_()
    temporal_padded = F.pad(guide3d, (0, 0, 0, 0, 1, 1), mode="replicate")
    expected = F.conv3d(temporal_padded, weight3d, padding=(0, 3, 3))
    previous = torch.cat((guide2d[:, :, :1], guide2d[:, :, :-1]), dim=2)
    following = torch.cat((guide2d[:, :, 1:], guide2d[:, :, -1:]), dim=2)
    slots = torch.cat((previous, guide2d, following), dim=1)
    flat_slots = slots.permute(0, 2, 1, 3, 4).reshape(10, 6, 9, 11)
    actual = F.conv2d(flat_slots, weight2d, padding=3).reshape(2, 5, 1, 9, 11).permute(0, 2, 1, 3, 4)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    probe = torch.randn(expected.shape, generator=generator, dtype=torch.float64)
    (expected * probe).sum().backward()
    (actual * probe).sum().backward()
    torch.testing.assert_close(guide2d.grad, guide3d.grad, rtol=1e-11, atol=1e-11)
    mapped_weight_grad = weight2d.grad.reshape(1, 3, 2, 7, 7).permute(0, 2, 1, 3, 4)
    torch.testing.assert_close(mapped_weight_grad, weight3d.grad, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("mode", ["spatial", "temporal"])
def test_gate_bootstrap_then_operator_gradients(mode):
    model = TemporalLFPAdapter.from_spatial(spatial(), mode, 2)
    x = data((1, 3, 5, 13, 15))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    model(x).square().mean().backward()
    assert model.scale.grad.abs() > 1e-8
    assert torch.count_nonzero(model.operator.attention.weight.grad) == 0
    assert model.operator.sigma.grad == 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model(x).square().mean().backward()
    assert model.operator.attention.weight.grad.norm() > 1e-10
    assert model.operator.sigma.grad.abs() > 1e-10
    if mode == "temporal":
        assert model.operator.attention.weight.grad[:, :2].norm() > 1e-10
        assert model.operator.attention.weight.grad[:, 4:].norm() > 1e-10


def test_frozen_downstream_still_backpropagates_and_remains_unchanged():
    model = candidate()
    downstream = nn.Sequential(nn.Conv3d(3, 2, 1), nn.BatchNorm3d(2), nn.ReLU(),
                               nn.Conv3d(2, 1, 1)).double().eval().requires_grad_(False)
    before = {name: value.clone() for name, value in downstream.state_dict().items()}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    downstream(model(data())).square().mean().backward()
    assert model.operator.attention.weight.grad.norm() > 0
    assert model.scale.grad.abs() > 0
    optimizer.step()
    assert all(torch.equal(value, before[name]) for name, value in downstream.state_dict().items())
    assert all(p.grad is None for p in downstream.parameters())


@pytest.mark.parametrize("mode", ["spatial", "temporal"])
def test_state_roundtrip(mode):
    model = candidate(mode)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = candidate(mode)
    restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
    assert torch.equal(restored(data()), model(data()))


def test_bypass_is_exact_and_does_not_evaluate_operator():
    model = candidate()
    with torch.no_grad():
        model.operator.attention.weight.fill_(float("nan"))
    x = data()
    assert model(x, mode="bypass") is x
    model.set_inference_mode("bypass")
    assert model(x) is x


@pytest.mark.parametrize("hw", [(42, 44), (43, 45)])
def test_even_phase_spatial_tiles_match_whole_nonzero_temporal(hw):
    model = candidate(chunk=2)
    x = data((1, 3, 5, *hw))
    expected = model(x)
    result = torch.empty_like(expected)
    h, w = hw
    for y0 in range(0, h, 13):
        for x0 in range(0, w, 15):
            y1, x1 = min(h, y0 + 13), min(w, x0 + 15)
            iy0, ix0 = max(0, y0 - 10) // 2 * 2, max(0, x0 - 10) // 2 * 2
            iy1, ix1 = min(h, (y1 + 11) // 2 * 2), min(w, (x1 + 11) // 2 * 2)
            tile = model(x[..., iy0:iy1, ix0:ix1])
            result[..., y0:y1, x0:x1] = tile[..., y0 - iy0:y1 - iy0, x0 - ix0:x1 - ix0]
    torch.testing.assert_close(result, expected, rtol=1e-12, atol=1e-12)


def test_telemetry_does_not_change_forward_or_gradients():
    model = candidate()
    x = data().requires_grad_()
    expected = model(x)
    expected.sum().backward()
    before = {name: p.grad.clone() for name, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    model.collect_stats = True
    actual = model(x)
    actual.sum().backward()
    assert torch.equal(actual, expected)
    for name, p in model.named_parameters():
        assert torch.equal(p.grad, before[name])
    stats = model.latest_stats
    assert 0 <= stats["gate_saturation_fraction"] <= 1
    assert stats["correction_to_input_norm"] > 0
    assert stats["previous_kernel_norm"] > 0


@pytest.mark.parametrize("kind", ["rank", "channels", "empty", "integer", "mode", "chunk", "source"])
def test_invalid_inputs_rejected(kind):
    model = candidate()
    with pytest.raises((ValueError, TypeError)):
        if kind == "rank":
            model(torch.zeros(1, 3, 5, 7))
        elif kind == "channels":
            model(torch.zeros(1, 4, 5, 7, 9))
        elif kind == "empty":
            model(torch.zeros(1, 3, 0, 7, 9))
        elif kind == "integer":
            model(torch.zeros(1, 3, 5, 7, 9, dtype=torch.long))
        elif kind == "mode":
            model(data(), mode="spatial")
        elif kind == "chunk":
            TemporalLFPAdapter(frame_chunk=0)
        else:
            source = spatial()
            source.register_parameter("unexpected", nn.Parameter(torch.zeros(())))
            TemporalLFPAdapter.from_spatial(source)
