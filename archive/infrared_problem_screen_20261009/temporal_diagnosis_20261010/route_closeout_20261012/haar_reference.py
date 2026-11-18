"""Self-written, spatial low-frequency-guided reference; not NS-FPN code.

Haar coefficients keep a distinct [N, C, 3, h, w] high-band axis.  For a
2 x 2 input block [[a, b], [c, d]], the subband order is vertical, horizontal,
diagonal: (-a-b+c+d)/2, (-a+b-c+d)/2, (a-b-c+d)/2.  LL is
(a+b+c+d)/2.  These orthonormal formulas define this implementation's signs
and order; checkpoint compatibility with another implementation is not implied.

Odd inputs replicate the last row/column on the bottom/right before analysis.
Synthesis crops that padding.  The attention convolution zero-pads in coefficient
space; Gaussian filtering replicate-pads in coefficient space.  There are no
temporal operations, batch statistics, or stochastic forward operations.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _check_image(x: Tensor) -> None:
    if x.ndim != 4:
        raise ValueError("Expected [N,C,H,W].")
    if not x.is_floating_point():
        raise TypeError("Haar input must have a floating-point dtype.")
    if any(size <= 0 for size in x.shape):
        raise ValueError("Every image dimension must be positive.")


def haar_forward(x: Tensor) -> Tuple[Tensor, Tensor, Tuple[int, int]]:
    """Return LL, explicit high bands, and the unpadded (height, width)."""
    _check_image(x)
    height, width = x.shape[-2:]
    if height % 2 or width % 2:
        x = F.pad(x, (0, width % 2, 0, height % 2), mode="replicate")
    a = x[..., 0::2, 0::2]
    b = x[..., 0::2, 1::2]
    c = x[..., 1::2, 0::2]
    d = x[..., 1::2, 1::2]
    low = (a + b + c + d) * 0.5
    high = torch.stack(
        ((-a - b + c + d) * 0.5,
         (-a + b - c + d) * 0.5,
         (a - b - c + d) * 0.5),
        dim=2,
    )
    return low, high, (height, width)


def haar_inverse(low: Tensor, high: Tensor, original_hw: Tuple[int, int]) -> Tensor:
    """Invert this module's explicit subband convention, then crop padding."""
    _check_image(low)
    if high.shape != (low.shape[0], low.shape[1], 3, *low.shape[-2:]):
        raise ValueError("High bands must have shape [N,C,3,h,w] matching LL.")
    if high.dtype != low.dtype or high.device != low.device:
        raise ValueError("Low and high bands must share dtype and device.")
    if len(original_hw) != 2 or any(not isinstance(v, int) for v in original_hw):
        raise ValueError("original_hw must contain two integer dimensions.")
    height, width = original_hw
    if height not in (2 * low.shape[-2] - 1, 2 * low.shape[-2]) or width not in (
        2 * low.shape[-1] - 1, 2 * low.shape[-1]
    ):
        raise ValueError("original_hw is inconsistent with one-level Haar padding.")
    vertical, horizontal, diagonal = high.unbind(dim=2)
    a = (low - vertical - horizontal + diagonal) * 0.5
    b = (low - vertical + horizontal - diagonal) * 0.5
    c = (low + vertical - horizontal - diagonal) * 0.5
    d = (low + vertical + horizontal + diagonal) * 0.5
    top = torch.stack((a, b), dim=-1).flatten(-2, -1)
    bottom = torch.stack((c, d), dim=-1).flatten(-2, -1)
    result = torch.stack((top, bottom), dim=-2).flatten(-3, -2)
    return result[..., :height, :width]


class SpatialLFPReference(nn.Module):
    """A local, independently written reference for the spatial LFP idea.

    LL channel mean/max condition a 7 x 7 attention convolution.  Its sigmoid
    modulates all high bands; values with absolute magnitude below the fixed
    gate are replaced by a 3 x 3 Gaussian-filtered value.  The scalar sigma is
    learned, and sigma squared has a numerical floor.  No author weights load.

    Zero attention weights give initial attention 0.5, deliberately making P
    nonidentity for nonconstant inputs.  An enclosing zero residual gate can
    therefore receive a first-step gradient.  This is an initialization choice,
    not a detection-performance or background-suppression guarantee.

    Dependency: Gaussian radius 1 plus attention radius 3 = 4 coefficient cells.
    Each coefficient/output cell is a 2 x 2 input block, giving a worst-case
    input-pixel distance of 9.  A 10-pixel halo is sufficient for an even-phase
    interior tile.  Tile starts must follow the full image's even grid and
    tiles touching image boundaries must use the full image boundary rules.
    This radius covers only P; upstream/downstream detector layers add halos.
    """

    wavelet_phase_period = 2
    coefficient_dependency_radius = 4
    input_dependency_radius = 9
    aligned_input_halo = 10

    def __init__(
        self,
        channels: int,
        gaussian_gate: float = 0.5,
        sigma_squared_floor: float = 1e-6,
    ) -> None:
        super().__init__()
        if not isinstance(channels, int) or channels <= 0:
            raise ValueError("channels must be a positive integer.")
        if not math.isfinite(gaussian_gate) or gaussian_gate <= 0:
            raise ValueError("gaussian_gate must be finite and positive.")
        if not math.isfinite(sigma_squared_floor) or sigma_squared_floor <= 0:
            raise ValueError("sigma_squared_floor must be finite and positive.")
        self.channels = channels
        self.gaussian_gate = float(gaussian_gate)
        self.sigma_squared_floor = float(sigma_squared_floor)
        self.attention = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        nn.init.zeros_(self.attention.weight)
        self.sigma = nn.Parameter(torch.tensor(1.0))

    def gaussian_kernel(self, like: Tensor) -> Tensor:
        """Normalized 3 x 3 kernel on the input device/dtype; differentiable."""
        coord = torch.arange(-1, 2, device=like.device, dtype=like.dtype)
        radius_squared = coord[:, None].square() + coord[None, :].square()
        sigma_squared = self.sigma.to(dtype=like.dtype).square().clamp_min(
            self.sigma_squared_floor
        )
        kernel = torch.exp(-radius_squared / (2.0 * sigma_squared))
        return kernel / kernel.sum()

    def forward(self, x: Tensor) -> Tensor:
        _check_image(x)
        if x.shape[1] != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {x.shape[1]}.")
        low, high, original_hw = haar_forward(x)
        attention_input = torch.cat(
            (low.mean(dim=1, keepdim=True), low.amax(dim=1, keepdim=True)), dim=1
        )
        attention = self.attention(attention_input).sigmoid()
        modulated = high * attention.unsqueeze(2)
        flat_high = modulated.flatten(1, 2)
        kernel = self.gaussian_kernel(flat_high).view(1, 1, 3, 3)
        blurred = F.conv2d(
            F.pad(flat_high, (1, 1, 1, 1), mode="replicate"),
            kernel.expand(3 * self.channels, 1, 3, 3),
            groups=3 * self.channels,
        ).reshape_as(modulated)
        filtered = torch.where(modulated.abs() < self.gaussian_gate, blurred, modulated)
        return haar_inverse(low, filtered, original_hw)
