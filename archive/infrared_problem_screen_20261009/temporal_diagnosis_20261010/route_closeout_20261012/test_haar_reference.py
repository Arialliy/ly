"""Finite numerical/structural checks; these are not detector effectiveness tests."""

import pytest
import torch
import torch.nn.functional as F

from haar_reference import SpatialLFPReference, haar_forward, haar_inverse


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(2, 3, 8, 10), (2, 5, 7, 9), (1, 32, 1, 1), (1, 2, 1, 7), (1, 3, 8, 1)])
def test_roundtrip_and_energy(shape, dtype):
    x = torch.randn(shape, generator=torch.Generator().manual_seed(46), dtype=dtype)
    low, high, hw = haar_forward(x)
    assert high.shape == (shape[0], shape[1], 3, (shape[2] + 1) // 2, (shape[3] + 1) // 2)
    tolerance = 2e-6 if dtype == torch.float32 else 1e-14
    torch.testing.assert_close(haar_inverse(low, high, hw), x, atol=tolerance, rtol=tolerance)
    padded = F.pad(x, (0, shape[3] % 2, 0, shape[2] % 2), mode="replicate")
    torch.testing.assert_close(low.square().sum() + high.square().sum(), padded.square().sum(), atol=tolerance, rtol=tolerance)


def test_impulse_subband_sign_and_channel_order():
    # Separate impulses cover every input coordinate and multiple channels.
    x = torch.eye(4, dtype=torch.float64).reshape(4, 1, 2, 2).repeat(1, 3, 1, 1)
    x *= torch.tensor([1.0, 2.0, 7.0], dtype=x.dtype)[None, :, None, None]
    low, high, hw = haar_forward(x)
    signs = torch.tensor([[-1, -1, 1], [-1, 1, -1], [1, -1, -1], [1, 1, 1]], dtype=x.dtype)
    scales = torch.tensor([1.0, 2.0, 7.0], dtype=x.dtype)
    torch.testing.assert_close(low[..., 0, 0], scales[None].expand(4, -1) / 2)
    torch.testing.assert_close(high[..., 0, 0], signs[:, None, :] * scales[None, :, None] / 2)
    torch.testing.assert_close(haar_inverse(low, high, hw), x, atol=0, rtol=0)


@pytest.mark.parametrize("band", [0, 1, 2])
def test_inverse_does_not_mix_subbands_or_channels(band):
    low = torch.zeros(1, 5, 2, 3, dtype=torch.float64)
    high = torch.zeros(1, 5, 3, 2, 3, dtype=torch.float64)
    high[0, 3, band, 1, 2] = 2
    reconstruction = haar_inverse(low, high, (4, 6))
    new_low, new_high, _ = haar_forward(reconstruction)
    torch.testing.assert_close(new_low, low, atol=0, rtol=0)
    torch.testing.assert_close(new_high, high, atol=0, rtol=0)
    assert reconstruction[:, [0, 1, 2, 4]].count_nonzero() == 0


def test_haar_gradcheck_odd_shape():
    x = torch.randn(1, 2, 3, 5, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda v: haar_inverse(*haar_forward(v)), (x,))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(2, 3, 12, 14), (2, 3, 11, 15), (1, 3, 1, 1)])
def test_lfp_size_dtype_and_finiteness(shape, dtype):
    model = SpatialLFPReference(channels=3).to(dtype=dtype)
    x = torch.randn(shape, dtype=dtype)
    y = model(x)
    assert y.shape == x.shape and y.dtype == dtype and torch.isfinite(y).all()


def test_lfp_initially_nonidentity_and_constant_preserving():
    model = SpatialLFPReference(3).double()
    x = torch.randn(2, 3, 9, 11, dtype=torch.float64)
    assert (model(x) - x).abs().max() > 1e-3
    constant = torch.full_like(x, 3.0)
    torch.testing.assert_close(model(constant), constant, atol=0, rtol=0)


def test_lfp_gradients_reach_input_attention_and_sigma():
    torch.manual_seed(46)
    model = SpatialLFPReference(3).double()
    x = (torch.randn(2, 3, 9, 11, dtype=torch.float64) * 0.05).requires_grad_()
    weights = torch.randn_like(x)
    (model(x) * weights).sum().backward()
    for gradient in (x.grad, model.attention.weight.grad, model.sigma.grad):
        assert gradient is not None and torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0


@pytest.mark.parametrize("sigma", [0.0, 1e-12, -1e-12, 1e12, -1e12])
def test_extreme_sigma_is_finite(sigma):
    model = SpatialLFPReference(2).double()
    with torch.no_grad():
        model.sigma.fill_(sigma)
    x = torch.randn(1, 2, 7, 9, dtype=torch.float64, requires_grad=True)
    y = model(x)
    y.square().mean().backward()
    assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()
    assert torch.isfinite(model.sigma.grad).all()
    kernel = model.gaussian_kernel(x)
    torch.testing.assert_close(kernel.sum(), torch.tensor(1.0, dtype=x.dtype))


def test_frame_batching_independent_even_with_nonzero_attention():
    torch.manual_seed(46)
    model = SpatialLFPReference(3).double()
    with torch.no_grad():
        model.attention.weight.normal_(0, 0.05)
    x = torch.randn(7, 3, 13, 15, dtype=torch.float64)
    whole = model(x)
    chunks = torch.cat([model(part) for part in x.split(2)])
    singles = torch.cat([model(frame) for frame in x.split(1)])
    torch.testing.assert_close(chunks, whole, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(singles, whole, rtol=1e-13, atol=1e-13)


def test_even_phase_halo_matches_whole_image():
    # Nonzero attention activates the longest, LL -> attention -> Gaussian path.
    torch.manual_seed(46)
    model = SpatialLFPReference(2).double()
    with torch.no_grad():
        model.attention.weight.normal_(0, 0.05)
    x = torch.randn(2, 2, 51, 55, dtype=torch.float64) * 0.03
    whole = model(x)
    result = torch.empty_like(x)
    for top in range(0, x.shape[-2], 16):
        for left in range(0, x.shape[-1], 18):
            bottom, right = min(top + 16, x.shape[-2]), min(left + 18, x.shape[-1])
            y0, x0 = max(0, top - 10), max(0, left - 10)
            y1, x1 = min(x.shape[-2], bottom + 10), min(x.shape[-1], right + 10)
            assert y0 % 2 == 0 and x0 % 2 == 0
            part = model(x[..., y0:y1, x0:x1])
            result[..., top:bottom, left:right] = part[..., top-y0:bottom-y0, left-x0:right-x0]
    torch.testing.assert_close(result, whole, rtol=1e-13, atol=1e-13)


def test_invalid_shapes_and_types_fail_explicitly():
    with pytest.raises(ValueError):
        haar_forward(torch.zeros(2, 3, 4))
    with pytest.raises(TypeError):
        haar_forward(torch.zeros(1, 1, 2, 2, dtype=torch.int64))
    with pytest.raises(ValueError):
        haar_forward(torch.zeros(1, 1, 0, 2))
    low, high, _ = haar_forward(torch.zeros(1, 2, 4, 6))
    with pytest.raises(ValueError):
        haar_inverse(low, high.flatten(1, 2), (4, 6))
    with pytest.raises(ValueError):
        haar_inverse(low, high, (2, 6))
    with pytest.raises(ValueError):
        SpatialLFPReference(3)(torch.zeros(1, 2, 4, 6))
    with pytest.raises(ValueError):
        SpatialLFPReference(0)
