"""Independent M1 state-update reference; not upstream MatAnyone/RFR code.

Inputs have already been aligned by the caller's existing PDA. This file
changes only how the aligned state and existing TSFM candidate are combined.
It does not implement alignment, the video loop, detection, masks or losses.

Inspiration: MatAnyone (CVPR 2025), matanyone/model/matanyone.py read_memory:
region-dependent interpolation between a new readout and a previous value.
Our feature-only gate and RFR candidate inputs are task adaptations, NOT an
implementation of its mask-guided/uncertainty-supervised method.
"""
from typing import Literal, Tuple
import torch
from torch import Tensor, nn

Scope = Literal['original', 'frame', 'pixel']


def _check_inputs(aligned: Tensor, candidate: Tensor, current: Tensor) -> None:
    if aligned.ndim != 4 or candidate.shape != aligned.shape or current.shape != aligned.shape:
        raise ValueError('aligned, candidate and current must have identical [B,C,H,W] shapes')
    if any(s <= 0 for s in aligned.shape):
        raise ValueError('empty tensor dimensions are not supported')
    if not all(torch.is_floating_point(x) for x in (aligned, candidate, current)):
        raise TypeError('all inputs must be floating point tensors')
    if candidate.device != aligned.device or current.device != aligned.device:
        raise ValueError('input devices must match')
    if candidate.dtype != aligned.dtype or current.dtype != aligned.dtype:
        raise ValueError('input dtypes must match')


def convex_state_update(aligned: Tensor, candidate: Tensor, retention: Tensor) -> Tensor:
    """Blend already-aligned old state and TSFM candidate.

    retention=0 returns candidate; retention=1 returns aligned.
    retention must be in [0,1] (provided by sigmoid in the module).
    No semantic, detector-monotonicity or recurrent-stability guarantee follows.
    """
    if aligned.ndim != 4 or candidate.shape != aligned.shape:
        raise ValueError('state tensors must have equal [B,C,H,W] shapes')
    if retention.shape != (aligned.shape[0], 1, aligned.shape[2], aligned.shape[3]):
        raise ValueError('retention must have shape [B,1,H,W]')
    gate = retention.to(dtype=candidate.dtype)
    return (1.0 - gate) * candidate + gate * aligned


class BoundedStateUpdate(nn.Module):
    """Feature-conditioned interpolation, intended as ONE replacement in M1.

    scope='original': original RFR-style M_t=candidate; no gate is used.
    scope='frame': compute the same gate network but broadcast its spatial mean.
    scope='pixel': use the gate's full spatial map.

    'frame' and 'pixel' have IDENTICAL learned parameters and initialization.
    Output layer is initialized to zero, hence both start with retention=0.5.
    Neither starts equivalent to a trained original M_t=candidate model.
    Frozen-baseline equivalence only follows separately if the outer residual
    projection is zero; do not reset a trained projection during continuation.

    Two padded 3x3 convolutions imply radius-two feature halo for gate-only
    spatial tiling. Frame mode needs a full-frame gate mean; computing separate
    tile means is not equivalent. This excludes upstream PDA receptive fields.
    """
    def __init__(self, channels: int = 16, hidden: int = 16, scope: Scope = 'pixel') -> None:
        super().__init__()
        if not isinstance(channels, int) or channels <= 0:
            raise ValueError('channels must be a positive integer')
        if not isinstance(hidden, int) or hidden <= 0:
            raise ValueError('hidden must be a positive integer')
        if scope not in ('original', 'frame', 'pixel'):
            raise ValueError('scope must be original, frame or pixel')
        self.channels = channels
        self.scope = scope
        mid = max(1, hidden // 2)
        # Independent compact predictor. No running normalization statistics.
        self.gate_net = nn.Sequential(
            nn.Conv2d(3 * channels, hidden, 1, bias=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden, mid, 3, padding=1, bias=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(mid, 1, 3, padding=1, bias=True),
        )
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.zeros_(self.gate_net[-1].bias)

    def gate_map(self, aligned: Tensor, candidate: Tensor, current: Tensor) -> Tensor:
        _check_inputs(aligned, candidate, current)
        if aligned.shape[1] != self.channels:
            raise ValueError(f'expected {self.channels} channels')
        # Crucially, no detection-score threshold, foreground mask, test box,
        # track id, or last predicted mask is required to query the gate.
        descriptor = torch.cat((aligned, current, candidate - aligned), dim=1)
        return torch.sigmoid(self.gate_net(descriptor))

    def forward(self, aligned: Tensor, candidate: Tensor, current: Tensor,
                *, scope: Scope | None = None) -> Tuple[Tensor, Tensor]:
        _check_inputs(aligned, candidate, current)
        if aligned.shape[1] != self.channels:
            raise ValueError(f'expected {self.channels} channels')
        selected = self.scope if scope is None else scope
        if selected not in ('original', 'frame', 'pixel'):
            raise ValueError('invalid scope')
        if selected == 'original':
            gate = candidate.new_zeros((candidate.shape[0], 1, *candidate.shape[-2:]))
            return candidate, gate
        gate = self.gate_map(aligned, candidate, current)
        if selected == 'frame':
            gate = gate.mean(dim=(-2, -1), keepdim=True).expand_as(gate)
        return convex_state_update(aligned, candidate, gate), gate
