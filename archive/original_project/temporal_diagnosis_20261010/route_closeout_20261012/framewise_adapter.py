"""Explicitly spatial, per-frame reference adapter for frozen DeepPro-Plus.

This is our interface/control code, not an NS-FPN implementation or an author
checkpoint-compatible replacement. The scalar gate starts at zero. The operator
must be deterministic per image, with no statistics across the flattened batch.
"""
from __future__ import annotations

import torch
from torch import nn


class FramewiseFeatureAdapter(nn.Module):
    """Apply F + tanh(scale) * (P(F) - F) independently to B*T frames.

    P must preserve BCHW shape, device and dtype. Zero scale is exactly identity
    for finite P(F); the branch is still evaluated so the gate can get a gradient.
    P's direct gradients are zero on that first backward pass. This is expected,
    and differs from a dead branch whose gate gradient is also zero.
    """

    def __init__(self, operator: nn.Module, channels: int = 32,
                 frame_chunk: int = 8):
        super().__init__()
        if channels < 1 or frame_chunk < 1:
            raise ValueError("channels and frame_chunk must be positive integers")
        self.operator = operator
        self.channels = int(channels)
        self.frame_chunk = int(frame_chunk)
        self.scale = nn.Parameter(torch.zeros(()))  # One shared scalar, not C gates.

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 5 or features.shape[1] != self.channels:
            raise ValueError(f"Expected [B,{self.channels},T,H,W], got {tuple(features.shape)}")
        b, c, t, h, w = features.shape
        if min(b, t, h, w) < 1:
            raise ValueError("Empty inputs are not supported")
        frames = features.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        pieces = []
        for start in range(0, b * t, self.frame_chunk):
            x = frames[start:start + self.frame_chunk]
            transformed = self.operator(x)
            if transformed.shape != x.shape:
                raise ValueError("Spatial operator changed BCHW shape")
            if transformed.dtype != x.dtype or transformed.device != x.device:
                raise ValueError("Spatial operator changed device or dtype")
            pieces.append(x + self.scale.tanh() * (transformed - x))
        result = torch.cat(pieces, dim=0).reshape(b, t, c, h, w)
        # Match the original layer1's contiguous BCTHW layout. Returning only
        # the permuted view changes TPro's time-linear input strides, and can
        # change its floating-point kernel path even with a zero residual.
        return result.permute(0, 2, 1, 3, 4).contiguous()


class FrozenDeepProPlusWithSpatialReference(nn.Module):
    """Insert only after layer1 and preserve the frozen Plus return contract.

    `original` must already have loaded its released checkpoint strictly. The
    original parameter tensors and BN buffers remain frozen; autograd through
    TPro and both original output heads remains enabled.
    """

    def __init__(self, original: nn.Module, feature_refiner: FramewiseFeatureAdapter):
        super().__init__()
        for name in ("conv_in", "layer1", "TPro", "conv_out1", "conv_out2"):
            if not hasattr(original, name):
                raise ValueError(f"Frozen DeepPro-Plus lacks required module {name}")
        if any(hasattr(original, name) for name in ("TPro1", "TPro2", "TPro3")):
            raise ValueError("This wrapper is for the single-TPro Plus snapshot")
        self.original = original
        self.feature_refiner = feature_refiner
        self.original.requires_grad_(False)
        self.original.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.original.eval()  # Also freeze BN when the new branch trains.
        return self

    def forward(self, seq_imgs: torch.Tensor):
        features = self.original.layer1(self.original.conv_in(seq_imgs))
        features = self.feature_refiner(features)
        features = self.original.TPro(features.permute(0, 3, 4, 1, 2))
        features = self.original.conv_out1(features)
        logits = self.original.conv_out2(features).squeeze(dim=1)
        return features, logits


def spatially_tiled_forward(model: nn.Module, x: torch.Tensor,
                           core_hw=(47, 49), halo: int = 20,
                           absolute_origin_yx=(0, 0)):
    """Finite-check tiler for this local reference, keeping Haar's even grid.

    At the current insertion, the original front end has radius 9 and the
    reference operator has maximum spatial reach 9; the later frozen layers
    are pointwise spatially. Halo 20 conservatively covers their sum. Tile
    *input* starts and non-global ends are rounded to the same even grid as the
    whole input. True odd image boundaries retain replicate/crop semantics.

    The whole input origin must be even in its parent image. A generic cropped
    image with an odd origin needs a phase-aware Haar implementation, which is
    outside this small reference helper. T remains intact on every tile.
    This inference helper returns both original outputs; it is not a trainer.
    """
    if x.ndim != 5 or min(core_hw) < 1 or halo < 18:
        raise ValueError("Expected BCTHW, positive core sizes, and halo >= 18")
    if any(int(v) % 2 for v in absolute_origin_yx):
        raise ValueError("Whole input origin must lie on the absolute even Haar grid")
    h, w = x.shape[-2:]
    outputs = None
    for y0 in range(0, h, core_hw[0]):
        y1 = min(h, y0 + core_hw[0])
        for x0 in range(0, w, core_hw[1]):
            x1 = min(w, x0 + core_hw[1])
            iy0 = max(0, y0 - halo) // 2 * 2
            ix0 = max(0, x0 - halo) // 2 * 2
            iy1 = min(h, ((y1 + halo + 1) // 2) * 2)
            ix1 = min(w, ((x1 + halo + 1) // 2) * 2)
            tile_outputs = model(x[..., iy0:iy1, ix0:ix1])
            if outputs is None:
                outputs = [o.new_empty((*o.shape[:-2], h, w)) for o in tile_outputs]
            for destination, tile in zip(outputs, tile_outputs):
                destination[..., y0:y1, x0:x1] = tile[
                    ..., y0-iy0:y1-iy0, x0-ix0:x1-ix0]
    return tuple(outputs)
