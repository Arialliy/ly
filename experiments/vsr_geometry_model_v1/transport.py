"""Transport observation residuals to the reference LR lattice for x2 VSR."""
import math

import torch
import torch.nn.functional as F


def _grid(height, width, like):
    y, x = torch.meshgrid(torch.arange(height, device=like.device, dtype=like.dtype),
                          torch.arange(width, device=like.device, dtype=like.dtype), indexing="ij")
    return torch.stack((x, y), dim=-1)[None]


def _sample(values, coordinates):
    height, width = values.shape[-2:]
    scale = coordinates.new_tensor((width, height))
    normalized = (coordinates + 0.5) * (2 / scale) - 1
    return F.grid_sample(values, normalized, mode="bilinear", padding_mode="border", align_corners=False)


def _in_bounds(coordinates, height, width):
    maximum = coordinates.new_tensor((width - 1, height - 1))
    return ((coordinates >= 0) & (coordinates <= maximum)).all(dim=-1).unsqueeze(1)


def _gaussian_blur(values, sigma, radius):
    # The spatially invariant acquisition kernel lives in the observing frame.
    x = torch.arange(-radius, radius + 1, device=values.device, dtype=values.dtype)
    kernel = torch.softmax(-0.5 * (x / sigma).square(), dim=0)
    channels = values.shape[1]
    horizontal = kernel.reshape(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = kernel.reshape(1, 1, -1, 1).expand(channels, 1, -1, 1)
    values = F.conv2d(F.pad(values, (radius, radius, 0, 0), mode="replicate"), horizontal, groups=channels)
    return F.conv2d(F.pad(values, (0, 0, radius, radius), mode="replicate"), vertical, groups=channels)


def _project(initial_hr, backward, cells, blur_sigma):
    """On neighbor coordinates u, warp HR first, blur there, then take even HR centers."""
    n, _, h, w = backward.shape
    if blur_sigma == 0:
        source = 2 * (cells + backward.permute(0, 2, 3, 1))
        return _sample(initial_hr, source), _in_bounds(source, 2*h, 2*w)
    neighbors = _grid(2*h, 2*w, backward).expand(n, -1, -1, -1)
    # LR flow is defined at even HR centers. Border extension covers the last half-cell.
    backward_hr = _sample(backward, neighbors / 2)
    source = neighbors + 2 * backward_hr.permute(0, 2, 3, 1)
    warped = _sample(initial_hr, source)
    valid = _in_bounds(source, 2*h, 2*w)
    radius = math.ceil(3 * blur_sigma)
    blurred = _gaussian_blur(warped, blur_sigma, radius)
    # Discard observations if any tap of their warp footprint crossed the input boundary.
    invalid = F.max_pool2d((~valid).to(warped.dtype), 2*radius+1, stride=1, padding=radius)
    return blurred[:, :, ::2, ::2], (invalid[:, :, ::2, ::2] == 0)


def transport_evidence(lr, initial_hr, ref_to_frame, frame_to_ref, blur_sigma=0.0, consistency_sigma=1.0):
    """Project HR, measure LR residuals, and return evidence on the reference grid.

    lr is B,T,C,H,W. initial_hr is B,C,2H,2W. Both flow tensors are
    B,T,2,H,W in LR pixels, channels (dx,dy). Pixel (x,y) of the LR lattice
    coincides with (2x,2y) of the HR lattice. The forward flow at reference
    site p maps it to neighbor p+f(p). The reverse flow at neighbor u maps
    it to reference u+b(u). Displacements support fractional values.

    predicted_lr is sampled on the neighbor lattice. aligned_lr and
    aligned_residual lie on the reference lattice. valid is a boolean
    B,T,1,H,W tensor. It combines support for both warps. confidence equals
    valid * exp(-||f(p) + b(p+f(p))||² / (2 consistency_sigma²)).
    A supplied pair of zero center-frame flows thus receives unit weight.
    sigma for acquisition blur is in HR pixels and sigma for flow consistency
    is in LR pixels. All sampling uses bilinear interpolation. Working output
    precision is float32, or float64 when any input uses float64.
    """
    if lr.ndim != 5:
        raise ValueError("Expected LR B,T,C,H,W")
    batch, frames, channels, height, width = lr.shape
    flow_shape = (batch, frames, 2, height, width)
    if (min(lr.shape) < 1 or initial_hr.shape != (batch, channels, 2*height, 2*width)
            or ref_to_frame.shape != flow_shape or frame_to_ref.shape != flow_shape):
        raise ValueError("Inconsistent LR, x2 HR, or bidirectional flow shapes")
    if not math.isfinite(blur_sigma) or blur_sigma < 0 or not math.isfinite(consistency_sigma) or consistency_sigma <= 0:
        raise ValueError("Expected a nonnegative blur and a positive consistency scale")
    tensors = (lr, initial_hr, ref_to_frame, frame_to_ref)
    if any(t.device != lr.device or not t.is_floating_point() for t in tensors):
        raise ValueError("All inputs must be floating-point tensors on the same device")
    dtype = torch.float64 if any(t.dtype == torch.float64 for t in tensors) else torch.float32
    n = batch * frames
    with torch.autocast(device_type=lr.device.type, enabled=False):
        observed = lr.to(dtype).reshape(n, channels, height, width)
        hr = initial_hr.to(dtype)[:, None].expand(-1, frames, -1, -1, -1).reshape(n, channels, 2*height, 2*width)
        forward = ref_to_frame.to(dtype).reshape(n, 2, height, width)
        backward = frame_to_ref.to(dtype).reshape(n, 2, height, width)
        cells = _grid(height, width, observed)
        pred, observed_valid = _project(hr, backward, cells, blur_sigma)
        to_neighbor = cells + forward.permute(0, 2, 3, 1)
        residual = _sample(observed - pred, to_neighbor)
        aligned = _sample(observed, to_neighbor)
        back_at_neighbor = _sample(backward, to_neighbor)
        footprint = _sample(observed_valid.to(dtype), to_neighbor)
        valid = _in_bounds(to_neighbor, height, width) & (footprint >= 1 - 8*torch.finfo(dtype).eps)
        closure = (forward + back_at_neighbor).square().sum(dim=1, keepdim=True)
        confidence = valid.to(dtype) * torch.exp(-closure / (2 * consistency_sigma**2))
    def unpack(t):
        return t.reshape(batch, frames, *t.shape[1:])
    return {"aligned_residual": unpack(residual * valid), "aligned_lr": unpack(aligned * valid),
            "predicted_lr": unpack(pred), "valid": unpack(valid), "confidence": unpack(confidence)}


def selfcheck():
    """Check subpixel sign, return transport, boundary mask, PSF, confidence and gradients."""
    h, w = 12, 16
    grid = _grid(2*h, 2*w, torch.zeros((), dtype=torch.float64))
    hr = (grid[..., 0] + 3*grid[..., 1]).unsqueeze(1)
    offsets = torch.tensor([[0, 0], [.25, -.5], [-1., 0.]], dtype=hr.dtype)
    forward = offsets[None, :, :, None, None].expand(1, 3, 2, h, w).clone()
    backward = -forward
    lr = hr[..., ::2, ::2][:, None] - 2*(offsets[:, 0] + 3*offsets[:, 1])[None, :, None, None, None]
    # Analytic neighbor values are valid for this affine target in the shared interior.
    for sigma in (0., .65):
        out = transport_evidence(lr, hr, forward, backward, sigma)
        mid = (..., slice(3, -3), slice(3, -3))
        torch.testing.assert_close(out["predicted_lr"][mid], lr[mid])
        torch.testing.assert_close(out["aligned_residual"][mid], torch.zeros_like(lr)[mid])
        torch.testing.assert_close(out["aligned_lr"][mid], lr[:, :1].expand_as(lr)[mid])
        assert out["valid"][:, 0].all() and (out["confidence"][:, 0] == 1).all()
        assert not out["valid"][0, 2, 0, h//2, 0]
    inconsistent = transport_evidence(lr, hr, forward, backward*0)
    expected = math.exp(-(.25**2+.5**2)/2)
    assert abs(float(inconsistent["confidence"][0,1,0,5,5]) - expected) < 1e-12
    marks = torch.zeros_like(lr)
    marks[0,2,0,5,5] = 1
    out = transport_evidence(marks, torch.zeros_like(hr), forward, backward)
    assert out["aligned_residual"][0,2,0,5,6] == 1
    impulse = torch.zeros_like(hr)
    impulse[0,0,10,12] = 1
    sigma = .65
    out = transport_evidence(torch.zeros_like(lr), impulse, forward*0, backward*0, sigma)
    taps = torch.arange(-2,3,dtype=hr.dtype)
    kernel = torch.softmax(-.5*(taps/sigma)**2,dim=0)
    torch.testing.assert_close(out["predicted_lr"][0,0,0,5,6],kernel[2]**2)
    discrepancy = torch.arange(w,dtype=lr.dtype).sin()/10
    params = [t.detach().requires_grad_() for t in (lr+discrepancy,hr,forward,backward)]
    out = transport_evidence(*params, blur_sigma=.65)
    out["aligned_residual"][..., 3:-3, 3:-3].sum().backward()
    assert all(t.grad is not None and t.grad.isfinite().all() and t.grad.abs().max()>0 for t in params)
    return {"translation": "pass", "psf": "pass", "confidence": "pass", "gradients": "pass"}


if __name__ == "__main__":
    print(selfcheck())
