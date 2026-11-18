"""A local, real-valued x2 polyphase observation model for gated corrections.

Each LR sample is bilinearly interpolated on a repeating 2x2 HR cell.
Integer LR translations therefore do not change its phase row; the actual
integer motion and the neighboring cell's content must be handled by alignment.
Reflection of either displacement component gives the same row in this local
approximation. It distinguishes horizontal from vertical coverage, but does
not encode the full signed phase or PSF of a Fourier image-formation model.
"""
import math

import torch


def phase_rows(offsets):
    """Return sample weights [B,T,4,H,W] for LR offsets [B,T,2,H,W] in (x,y).

    The four HR cell phases are (y,x)=(0,0),(0,1),(1,0),(1,1), the same
    order as torch.nn.functional.pixel_shuffle. Offsets use the convention
    'sample at reference position + displacement'. Values are continuous and
    piecewise differentiable; cell-boundary derivatives are subgradients.
    """
    if offsets.ndim != 5 or offsets.shape[2] != 2 or not offsets.is_floating_point():
        raise ValueError("offsets must be floating point with shape [B,T,2,H,W]")
    work = offsets if offsets.dtype == torch.float64 else offsets.float()
    # On a periodic two-site HR axis, the odd site's bilinear weight is the
    # triangle wave with weights 0,1,0 at LR offsets 0,1/2,1 respectively.
    odd = 1 - (2 * torch.remainder(work, 1) - 1).abs()
    wx, wy = odd.unbind(dim=2)
    return torch.stack(((1-wx)*(1-wy), wx*(1-wy), (1-wx)*wy, wx*wy), dim=2)


def mix_polyphase(observation, prior, offsets, reliability, tau=0.1, chunk_size=16384):
    """Mix two HR corrections in packed [B,4*C,H,W] form, returning that shape.

    At each LR site G = A.T diag(q) A / T. The output is
    prior + (G + tau*I)^(-1) G (observation - prior), applied to each color.
    Reliability has shape [B,T,1,H,W] or [B,T,H,W]. Confidences saturate at
    0 and 1. Dividing by T preserves absolute observation mass. Empty support
    returns the prior exactly. Solve and Gram products use at least FP32 even
    inside AMP. Different spatial sites are independent and solved in chunks.
    """
    if observation.ndim != 4 or observation.shape != prior.shape or observation.shape[1] % 4:
        raise ValueError("observation and prior must have identical packed [B,4*C,H,W] shapes")
    if not observation.is_floating_point() or observation.dtype != prior.dtype:
        raise ValueError("observation and prior must use the same floating-point dtype")
    if offsets.ndim != 5 or offsets.shape[2] != 2:
        raise ValueError("offsets must have shape [B,T,2,H,W]")
    b, t, _, h, w = offsets.shape
    if min(b, t, h, w, observation.shape[1]) <= 0 or observation.shape[0] != b or observation.shape[-2:] != (h,w):
        raise ValueError("correction and offset grids must have matching, nonempty spatial and batch dimensions")
    if reliability.ndim == 5 and reliability.shape[2] == 1:
        reliability = reliability.squeeze(2)
    if reliability.shape != (b,t,h,w):
        raise ValueError("reliability must have shape [B,T,1,H,W] or [B,T,H,W]")
    if not all(x.device == observation.device for x in (prior, offsets, reliability)):
        raise ValueError("corrections, offsets and reliabilities must be on one device")
    if not isinstance(chunk_size,int) or chunk_size < 1 or not math.isfinite(tau) or tau <= 0:
        raise ValueError("chunk_size must be a positive integer and tau must be finite and positive")

    c = observation.shape[1] // 4
    dtype = torch.float64 if torch.float64 in (observation.dtype, offsets.dtype, reliability.dtype) else torch.float32
    with torch.autocast(device_type=observation.device.type, enabled=False):
        rows = phase_rows(offsets).to(dtype).permute(0,3,4,1,2).reshape(-1,t,4)
        q = reliability.to(dtype).clamp(0,1).permute(0,2,3,1).reshape(-1,t,1)
        delta = observation.to(dtype) - prior.to(dtype)
        delta = delta.reshape(b,c,4,h,w).permute(0,3,4,2,1).reshape(-1,4,c)
        eye = torch.eye(4, device=observation.device, dtype=dtype)
        updates = []
        for start in range(0, rows.shape[0], chunk_size):
            region = slice(start,start+chunk_size)
            a = rows[region]
            gram = a.mT @ (q[region] * a) / t
            # G is positive semidefinite; strictly positive tau makes the
            # system nonsingular even when every frame is masked out.
            rhs = gram @ delta[region]
            update = torch.linalg.solve_ex(gram + tau*eye, rhs, check_errors=False).result
            updates.append(update)
        update = torch.cat(updates).reshape(b,h,w,4,c).permute(0,4,3,1,2).reshape_as(prior)
        return (prior.to(dtype) + update).to(prior.dtype)
