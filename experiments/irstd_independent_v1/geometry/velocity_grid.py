"""Fixed velocity hypotheses in native-image pixels per original frame interval."""
from collections.abc import Sequence

import torch


def make_velocity_grid(
    vx: Sequence[float] | torch.Tensor,
    vy: Sequence[float] | torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return ``[My, Mx, 2]`` with the last dimension ordered ``(vx, vy)``.

    Each axis must be nonempty, finite, strictly increasing, and contain an
    exact zero. The grid is fixed geometry, not a learned proposal or top-k.
    Tensor axes requiring gradients are rejected rather than silently detached.
    """
    if dtype not in (torch.float32, torch.float64):
        raise TypeError("Velocity grids support float32 or float64")

    def axis(values, name):
        raw = torch.as_tensor(values, device=device)
        if raw.requires_grad:
            raise TypeError(f"{name} is fixed geometry in S1; only feature gradients are supported")
        if raw.is_complex() or raw.dtype == torch.bool:
            raise TypeError(f"{name} must contain real numeric velocities")
        # Convert the original values directly: a Python float sequence must
        # not first lose precision through the default float32 inference.
        out = torch.as_tensor(values, device=device, dtype=dtype)
        if out.ndim != 1 or out.numel() == 0:
            raise ValueError(f"{name} must be a nonempty one-dimensional sequence")
        if not bool(torch.isfinite(out).all()):
            raise ValueError(f"{name} must contain only finite velocities")
        if out.numel() > 1 and not bool((out[1:] > out[:-1]).all()):
            raise ValueError(f"{name} must be strictly increasing")
        if not bool((out == 0).any()):
            raise ValueError(f"{name} must contain zero velocity")
        return out

    x = axis(vx, "vx")
    y = axis(vy, "vy")
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((grid_x, grid_y), dim=-1)
