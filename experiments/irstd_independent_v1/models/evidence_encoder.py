"""Encode shared main effects and one controlled choice of time–velocity evidence."""
import torch
from torch import nn
from torch.nn import functional as F

from ..contracts import CENTER_INDEX
from ..geometry.sampler import SampledEvidence
from .evidence_parts import decompose_evidence


class TemporalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Conv1d(10, 8, 3, padding=1)
        self.second = nn.Conv1d(8, 8, 3, padding=1)

    def forward(self, time_effect, offsets):
        batch, queries, _, times = time_effect.shape
        coord = offsets[:, None, None, :].expand(batch, queries, 1, times)
        features = torch.cat((time_effect, coord), dim=2).flatten(0, 1)
        hidden = F.silu(self.first(features))
        return F.silu(self.second(hidden)).mean(dim=-1).reshape(batch, queries, 8)


class VelocityEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Conv2d(12, 8, 3, padding=1)
        self.second = nn.Conv2d(8, 8, 3, padding=1)

    def forward(self, velocity_effect, grid, velocity_valid):
        batch, queries, _, height, width = velocity_effect.shape
        valid = velocity_valid.flatten(0, 1).unsqueeze(1)
        coords = grid.permute(2, 0, 1).expand(batch * queries, 2, height, width)
        features = torch.cat((velocity_effect.flatten(0, 1), coords, valid.to(coords.dtype)), dim=1)
        hidden = F.silu(self.first(features.masked_fill(~valid, 0))).masked_fill(~valid, 0)
        hidden = F.silu(self.second(hidden)).masked_fill(~valid, 0)
        mean = hidden.sum(dim=(-2, -1)) / valid.sum(dim=(-2, -1)).clamp_min(1)
        return mean.reshape(batch, queries, 8)


class AxialResidualBlock(nn.Module):
    """Time filtering and velocity-plane filtering retain all cells until readout."""

    def __init__(self, dilation):
        super().__init__()
        self.time = nn.Conv3d(16, 16, (5, 1, 1), groups=16, bias=False,
                              dilation=(dilation, 1, 1), padding=(2 * dilation, 0, 0))
        self.velocity = nn.Conv3d(16, 16, (1, 3, 3), groups=16, bias=False, padding=(0, 1, 1))
        self.mix = nn.Conv3d(16, 16, 1, bias=False)

    def forward(self, hidden, valid):
        temporal = F.silu(self.time(hidden))
        # The temporal convolution is per velocity. Invalid columns enter it as zero.
        spatial = F.silu(self.velocity(temporal)).masked_fill(~valid, 0)
        return (hidden + self.mix(spatial)).masked_fill(~valid, 0)


class JointEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Conv3d(9, 16, 1, bias=False)
        self.coordinates = nn.Conv3d(3, 16, 1, bias=False)
        self.blocks = nn.ModuleList([AxialResidualBlock(1), AxialResidualBlock(3)])
        self.score = nn.Conv3d(16, 1, 1, bias=False)

    def forward(self, observations, offsets, grid, velocity_valid):
        batch, queries, _, times, height, width = observations.shape
        valid = velocity_valid.flatten(0, 1)[:, None, None, :, :]
        time = offsets[:, None, :, None, None].expand(batch, 1, times, height, width)
        velocity = grid.permute(2, 0, 1)[None, :, None].expand(batch, 2, times, height, width)
        # Coordinate modulation preserves a zero joint input, including singleton-support J.
        gain = 1 + self.coordinates(torch.cat((time, velocity), dim=1)).tanh()
        hidden = self.project(observations.flatten(0, 1)).unflatten(0, (batch, queries))
        hidden = F.silu(hidden * gain[:, None]).flatten(0, 1).masked_fill(~valid, 0)
        for block in self.blocks:
            hidden = block(hidden, valid)
        scores = self.score(hidden).float()
        mask = valid.expand_as(scores).flatten(2)
        # A finite floor keeps softmax defined for empty support. Masking again gives
        # exactly zero weight to every cell on an empty row.
        scores = scores.flatten(2).masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = scores.softmax(dim=-1).masked_fill(~mask, 0)
        return (hidden.flatten(2) * weights).sum(dim=-1).reshape(batch, queries, 16)


class EvidenceEncoder(nn.Module):
    """Fuse magnitudes, temporal and velocity main effects, joint evidence and the center."""

    def __init__(self, mode="interaction"):
        super().__init__()
        if mode not in ("additive_only", "raw_joint", "interaction"):
            raise ValueError(f"Unknown evidence mode: {mode}")
        self.mode = mode
        self.amplitude = nn.Linear(9, 8)
        self.temporal = TemporalEncoder()
        self.velocity = VelocityEncoder()
        self.joint = JointEncoder()
        self.fusion = nn.Linear(49, 32)

    def forward(self, evidence: SampledEvidence, center: torch.Tensor) -> torch.Tensor:
        values = evidence.values
        batch, queries, channels, times, height, width = values.shape
        if (channels != 9 or center.shape != (batch, queries, 8)
                or evidence.time_offsets.shape != (batch, times)
                or evidence.velocities_xy.shape != (height, width, 2)):
            raise ValueError("Expected nine evidence and eight center channels on aligned query coordinates")
        valid = evidence.velocity_valid
        parts = decompose_evidence(values, valid)
        offsets = evidence.time_offsets.float() / CENTER_INDEX
        grid = evidence.velocities_xy.float()
        observed = valid.any(dim=(-2, -1)).unsqueeze(-1)
        amplitude = F.silu(self.amplitude(parts.mean)).masked_fill(~observed, 0)
        temporal = self.temporal(parts.time_effect, offsets).masked_fill(~observed, 0)
        velocity = self.velocity(parts.velocity_effect, grid, valid)
        if self.mode == "additive_only":
            joint_input = (parts.mean[..., None, None, None] + parts.time_effect[..., None, None]
                           + parts.velocity_effect.unsqueeze(3))
        elif self.mode == "raw_joint":
            joint_input = values.float()
        else:
            joint_input = parts.interaction
        joint_input = joint_input.masked_fill(~valid[:, :, None, None], 0)
        joint = self.joint(joint_input, offsets, grid, valid)
        coverage = valid.float().mean(dim=(-2, -1)).unsqueeze(-1)
        return F.silu(self.fusion(torch.cat((amplitude, temporal, velocity, joint, center, coverage), dim=-1)))
