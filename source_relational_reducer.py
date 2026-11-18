"""Research prototype for M1-U, NOT a complete M1 or a trained detector.

Inputs must already have been aligned to the current coordinate system by
an independently validated aligner (for example, RFR's PDA).
This module owns no memory and does not select sources or change time windows.

Two modes share exactly the same learned layers:
  relations:    form pair relations before pooling across source frames.
  early_merge:  pool source values first, then apply the same relation network.

The early_merge implementation deliberately repeats pair computation. This
keeps the relation-network call count matched to the candidate during pilots.
It is a mean-information control, not an optimized inference implementation
or a claim of equal effective capacity: its difference inputs are always zero.
"""
from __future__ import annotations

from itertools import combinations
from numbers import Integral
from typing import Literal

import torch
from torch import Tensor, nn

ReductionMode = Literal["relations", "early_merge"]


class SourceRelationalReducer(nn.Module):
    """Dense, order-invariant source relation summary.

    current:  [B, C, H, W], used only as the no-history fallback.
    aligned:  [B, K, C, H, W], one tensor per distinct source frame.
    valid:    [B, K] boolean source-availability mask. This is NOT a target
              mask, geometric match-confidence map, or learned reliability.
    returns:  [B, 2*value_channels, H, W], with current's dtype and device.
              Defaults give 16 output channels and 672 learned parameters.

    Output halves are a unary content summary and a learned source-relation
    summary, not target probability and not calibrated uncertainty.
    With fewer than two valid sources, the relation summary is exactly zero.
    Source identities and temporal order are not encoded. Even source count
    is not generally identifiable: any n >= 2 identical sources have the
    same summary. The caller is responsible for providing distinct sources.
    """

    def __init__(
        self,
        in_channels: int = 16,
        value_channels: int = 8,
        hidden_channels: int = 16,
    ) -> None:
        super().__init__()
        channels = (in_channels, value_channels, hidden_channels)
        if any(not isinstance(n, Integral) or isinstance(n, bool) for n in channels):
            raise TypeError("Channel counts must be integers.")
        if min(channels) <= 0:
            raise ValueError("Channel counts must be positive.")
        self.in_channels = in_channels
        self.value_channels = value_channels
        self.value = nn.Sequential(
            nn.Conv2d(in_channels, value_channels, kernel_size=1),
            nn.Tanh(),
        )
        self.relation = nn.Sequential(
            nn.Conv2d(3 * value_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, value_channels, kernel_size=1),
        )

    def forward(
        self,
        current: Tensor,
        aligned: Tensor,
        valid: Tensor | None = None,
        *,
        mode: ReductionMode = "relations",
    ) -> Tensor:
        if mode not in ("relations", "early_merge"):
            raise ValueError(f"Unknown reduction mode: {mode!r}")
        if not isinstance(current, Tensor) or not isinstance(aligned, Tensor):
            raise TypeError("Current and sources must be tensors.")
        if current.ndim != 4 or aligned.ndim != 5:
            raise ValueError("Expected current [B,C,H,W] and aligned [B,K,C,H,W].")
        b, c, h, w = current.shape
        if c != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {c}.")
        if aligned.shape[0] != b or tuple(aligned.shape[2:]) != (c, h, w):
            raise ValueError("Current and source dimensions do not match.")
        if h == 0 or w == 0:
            raise ValueError("Feature spatial dimensions must be non-empty.")
        if current.device != aligned.device or current.dtype != aligned.dtype:
            raise ValueError("Current and sources must share device and dtype.")
        if not current.is_floating_point() or not aligned.is_floating_point():
            raise TypeError("Features must be floating-point tensors.")
        k = aligned.shape[1]
        if valid is None:
            valid = torch.ones((b, k), dtype=torch.bool, device=current.device)
        if not isinstance(valid, Tensor):
            raise TypeError("valid must be a tensor or None.")
        if valid.shape != (b, k) or valid.dtype != torch.bool:
            raise ValueError("valid must be a boolean tensor of shape [B,K].")
        if valid.device != current.device:
            raise ValueError("valid must be on the feature device.")

        fallback = self.value(current)
        if k == 0:
            return torch.cat((fallback, torch.zeros_like(fallback)), dim=1).to(current.dtype)

        # Padded invalid entries are sanitized BEFORE learned operations.
        slot_mask = valid[:, :, None, None, None]
        safe_aligned = torch.where(slot_mask, aligned, torch.zeros_like(aligned))
        values = self.value(safe_aligned.reshape(b * k, c, h, w))
        values = values.reshape(b, k, self.value_channels, h, w)

        # Pool in at least float32; descriptors and learned layers retain
        # the projection dtype selected by the model/autocast context.
        accum_dtype = torch.float64 if values.dtype == torch.float64 else torch.float32
        weights = slot_mask.to(dtype=accum_dtype)
        count = weights.sum(dim=1)
        mean = (values.to(accum_dtype) * weights).sum(dim=1) / count.clamp_min(1)
        mean = torch.where(count > 0, mean, fallback.to(accum_dtype))
        mean_for_layers = mean.to(values.dtype)

        relation_sum = torch.zeros_like(mean)
        pair_count = torch.zeros((b, 1, 1, 1), dtype=accum_dtype, device=current.device)
        for j, l in combinations(range(k), 2):
            if mode == "relations":
                a, z = values[:, j], values[:, l]
            else:
                a, z = mean_for_layers, mean_for_layers
            # Symmetric descriptors: source order has no semantics in this prototype.
            descriptor = torch.cat((a + z, (a - z).abs(), a * z), dim=1)
            pair_features = self.relation(descriptor).to(accum_dtype)
            pair_valid = (valid[:, j] & valid[:, l])[:, None, None, None]
            pair_weight = pair_valid.to(accum_dtype)
            relation_sum = relation_sum + pair_features * pair_weight
            pair_count = pair_count + pair_weight

        relation_mean = relation_sum / pair_count.clamp_min(1)
        return torch.cat((mean, relation_mean), dim=1).to(current.dtype)
