"""Frozen DeepPro-Plus with explicit local-reference evidence.

Only the new branch is trainable. Input is the existing normalized image tensor,
not a target crop or a feature-derived estimate of background. Spatial reference
uses replicate padding at image boundaries; missing temporal references are zero.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

DELAYS = (1, 4)
INNER_RADIUS = 5
OUTER_RADIUS = 7
ANNULUS_OFFSETS = tuple(
    (dy, dx)
    for dy in range(-OUTER_RADIUS, OUTER_RADIUS + 1)
    for dx in range(-OUTER_RADIUS, OUTER_RADIUS + 1)
    if max(abs(dy), abs(dx)) > INNER_RADIUS
)
ANNULUS_COUNT = len(ANNULUS_OFFSETS)
S_MIN = 1.0 / 47.4845
CHANNEL_NAMES = ('I', 'D1', 'mu1', 's1', 'E1', 'm1', 'D4', 'mu4', 's4', 'E4', 'm4')
MODES = ('current', 'difference', 'relative')
BRANCH_SPATIAL_RADIUS = OUTER_RADIUS + 2


def exact_even_median(values: torch.Tensor) -> torch.Tensor:
    """Average both central order statistics, unlike torch.median's lower one.

    The sample dimension is last. Values are never sorted approximately. The
    function is separate for its independent numerical tests.
    """
    count = values.shape[-1]
    if count < 1:
        raise ValueError('Median requires at least one sample')
    # CUDA kthvalue is forbidden by torch deterministic algorithms even when
    # gradients are disabled. Stable exact sorting retains the frozen strict
    # deterministic execution contract and the same mathematical median.
    ordered = torch.sort(values, dim=-1, stable=True).values
    lower = ordered[..., (count - 1) // 2]
    if count % 2:
        return lower
    upper = ordered[..., count // 2]
    return (lower + upper) * 0.5


class LocalReferenceEvidence(nn.Module):
    def __init__(self, mode: str, *, statistics_chunk_frames: int = 8, s_min: float = S_MIN):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f'Unknown evidence mode: {mode}')
        if statistics_chunk_frames < 1 or s_min <= 0:
            raise ValueError('Chunk size and s_min must be positive')
        self.mode = mode
        self.statistics_chunk_frames = int(statistics_chunk_frames)
        self.s_min = float(s_min)
        self.encoder = nn.Sequential(
            nn.Conv2d(11, 8, 3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 8, 3, padding=1), nn.ReLU(),
        )
        self.joint = nn.Sequential(nn.Conv2d(16, 8, 1), nn.ReLU())
        self.output = nn.Conv2d(8, 1, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.last_input = None
        self.last_statistics = {}
        self.last_delta_z = None

    @property
    def config(self):
        return dict(mode=self.mode, delays=list(DELAYS), inner_radius=INNER_RADIUS,
                    outer_radius=OUTER_RADIUS, annulus_count=ANNULUS_COUNT,
                    annulus_offsets=[list(v) for v in ANNULUS_OFFSETS],
                    s_min=self.s_min, channels=list(CHANNEL_NAMES),
                    spatial_padding='replicate', temporal_missing='all five channels zero',
                    median='average of both central order statistics',
                    mad='median(abs(neighbor difference - same local median))',
                    statistics_chunk_frames=self.statistics_chunk_frames,
                    branch_spatial_radius=BRANCH_SPATIAL_RADIUS,
                    encoder='Conv2d(11,8,3,p1),ReLU,Conv2d(8,8,3,p1),ReLU',
                    joint='Conv2d(16,8,1),ReLU', output='Conv2d(8,1,1)',
                    zero_initialized=['output.weight', 'output.bias'],
                    temporal_statistic_input='normalized raw image, detached',
                    clipping=None)

    @torch.no_grad()
    def statistics(self, difference: torch.Tensor):
        """Exact local median and MAD for arbitrary leading frame dimensions.

        A chunk materializes N*H*W*104 elements. It is reused in-place for MAD;
        no input gradient is needed or retained. Chunking only controls memory.
        """
        if difference.ndim < 3:
            raise ValueError('Expected [..., H, W] with at least one leading dimension')
        h, w = difference.shape[-2:]
        flat = difference.detach().reshape(-1, h, w)
        mu = torch.empty_like(flat)
        scale = torch.empty_like(flat)
        for lo in range(0, len(flat), self.statistics_chunk_frames):
            hi = min(lo + self.statistics_chunk_frames, len(flat))
            padded = F.pad(flat[lo:hi, None], (OUTER_RADIUS,) * 4, mode='replicate')[:, 0]
            values = torch.stack([
                padded[:, OUTER_RADIUS + dy:OUTER_RADIUS + dy + h,
                       OUTER_RADIUS + dx:OUTER_RADIUS + dx + w]
                for dy, dx in ANNULUS_OFFSETS
            ], dim=-1)
            local_mu = exact_even_median(values)
            mu[lo:hi] = local_mu
            values.sub_(local_mu[..., None]).abs_()
            scale[lo:hi] = exact_even_median(values)
        return mu.reshape(difference.shape), scale.reshape(difference.shape)

    @torch.no_grad()
    def construct_evidence(self, image: torch.Tensor):
        """Return [B,11,T,H,W]; each absent-reference five-tuple is exactly zero."""
        if image.ndim != 5 or image.shape[1] != 1:
            raise ValueError(f'Expected [B,1,T,H,W], got {tuple(image.shape)}')
        raw = image.detach()[:, 0]
        b, t, h, w = raw.shape
        evidence = raw.new_zeros((b, t, 11, h, w))
        evidence[:, :, 0] = raw
        if self.mode != 'current':
            for delay_index, delay in enumerate(DELAYS):
                if t <= delay:
                    continue
                start = 1 + 5 * delay_index
                difference = raw[:, delay:] - raw[:, :-delay]
                evidence[:, delay:, start] = difference
                evidence[:, delay:, start + 4] = 1
                if self.mode == 'relative':
                    mu, scale = self.statistics(difference)
                    evidence[:, delay:, start + 1] = mu
                    evidence[:, delay:, start + 2] = scale
                    evidence[:, delay:, start + 3] = (difference - mu) / scale.clamp_min(self.s_min)
        self.last_input = evidence.detach()
        self.last_statistics = {name: self.last_input[:, :, i] for i, name in enumerate(CHANNEL_NAMES)}
        return evidence.permute(0, 2, 1, 3, 4)

    def forward(self, image: torch.Tensor, original_features: torch.Tensor):
        b, _, t, h, w = image.shape
        if tuple(original_features.shape) != (b, 8, t, h, w):
            raise ValueError(f'Expected original features [B,8,T,H,W], got {tuple(original_features.shape)}')
        evidence = self.construct_evidence(image)
        flat_input = evidence.permute(0, 2, 1, 3, 4).reshape(b * t, 11, h, w)
        encoded = self.encoder(flat_input)
        original = original_features.detach().permute(0, 2, 1, 3, 4).reshape(b * t, 8, h, w)
        delta = self.output(self.joint(torch.cat((original, encoded), dim=1)))
        delta = delta.reshape(b, t, h, w)
        self.last_delta_z = delta.detach()
        return delta


class FrozenEvidenceDetector(nn.Module):
    """Preserve original forward interface and state keys below `original`."""
    def __init__(self, original: nn.Module, mode: str, *, statistics_chunk_frames: int = 8,
                 s_min: float = S_MIN):
        super().__init__()
        self.original = original
        self.original.requires_grad_(False)
        self.original.eval()
        self.evidence = LocalReferenceEvidence(mode, statistics_chunk_frames=statistics_chunk_frames,
                                               s_min=s_min)
        self.correction_enabled = True
        self.last_original_logits = None
        self.last_delta = None

    @property
    def config(self):
        return dict(original_frozen=True, original_bn='eval',
                    readout='z0 + delta_z', evidence=self.evidence.config)

    def train(self, mode: bool = True):
        super().train(mode)
        self.original.eval()
        return self

    def forward(self, image: torch.Tensor):
        self.original.eval()
        with torch.no_grad():
            features, original_logits = self.original(image)
        if original_logits.shape != image[:, 0].shape:
            raise ValueError(f'Expected original logits [B,T,H,W], got {tuple(original_logits.shape)}')
        self.last_original_logits = original_logits.detach()
        if self.correction_enabled:
            delta = self.evidence(image, features)
        else:
            delta = torch.zeros_like(original_logits)
            self.evidence.last_input = None
            self.evidence.last_statistics = {}
            self.evidence.last_delta_z = delta
        self.last_delta = delta.detach()
        return features, original_logits + delta
