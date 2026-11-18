"""Two-way decomposition on the common set of observed velocities at each query."""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EvidenceParts:
    mean: torch.Tensor             # [B,N,C]
    time_effect: torch.Tensor      # [B,N,C,T]
    velocity_effect: torch.Tensor  # [B,N,C,My,Mx]
    interaction: torch.Tensor      # [B,N,C,T,My,Mx]


def decompose_evidence(values: torch.Tensor, velocity_valid: torch.Tensor) -> EvidenceParts:
    """Split native S1 responses using velocities observed at every context time.

    Input values have shape [B,N,C,T,My,Mx]. The common velocity mask has
    shape [B,N,My,Mx]. The caller retains this mask for all subsequent layers.
    Reductions and outputs use FP32, including when the input is lower precision.
    """
    if values.ndim != 6 or not values.is_floating_point():
        raise ValueError("Expected floating evidence with shape [B,N,C,T,My,Mx]")
    batch, queries, channels, times, rows, cols = values.shape
    if min(channels, times, rows, cols) <= 0:
        raise ValueError("Evidence requires nonempty channel, time and velocity axes")
    if velocity_valid.shape != (batch, queries, rows, cols) or velocity_valid.dtype != torch.bool:
        raise ValueError("Expected a common bool velocity mask with shape [B,N,My,Mx]")
    if velocity_valid.device != values.device:
        raise ValueError("Velocity mask and evidence must share a device")

    support = velocity_valid[:, :, None, None]
    observed = values.float().masked_fill(~support, 0)
    count = velocity_valid.sum(dim=(-2, -1)).clamp_min(1).float()
    row_mean = observed.sum(dim=(-2, -1)) / count[:, :, None, None]
    mean = row_mean.mean(dim=3)
    time_effect = row_mean - mean.unsqueeze(3)

    # Removing the row mean first makes the single-velocity residual exactly
    # zero and avoids subtracting two large marginal means from the raw value.
    within_time = (observed - row_mean[..., None, None]).masked_fill(~support, 0)
    velocity_effect = within_time.mean(dim=3)
    interaction = within_time - velocity_effect.unsqueeze(3)
    return EvidenceParts(mean, time_effect, velocity_effect, interaction)
