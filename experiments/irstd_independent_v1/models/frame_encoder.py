"""Shared spatial features and masked five-by-five contrast in native frame coordinates."""
import torch
from torch import nn
from torch.nn import functional as F


class FrameEncoder(nn.Module):
    radius = 2

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Conv2d(1, 8, 3, padding=1),
                                     nn.Conv2d(8, 8, 3, padding=1)])

    def forward(self, frames: torch.Tensor, pixel_valid: torch.Tensor) -> torch.Tensor:
        batch, times = frames.shape[:2]
        valid = pixel_valid.flatten(0, 1)
        features = frames.flatten(0, 1).masked_fill(~valid, 0)
        for layer in self.layers:
            features = F.silu(layer(features)).masked_fill(~valid, 0)
        return features.unflatten(0, (batch, times))


class ContrastChannel(nn.Module):
    radius = 2
    epsilon = (1.0 / 255.0) ** 2

    def forward(self, frames: torch.Tensor, pixel_valid: torch.Tensor) -> torch.Tensor:
        """Centre-pixel contrast; image edges and missing neighbours contribute no samples."""
        batch, times = frames.shape[:2]
        valid = pixel_valid.flatten(0, 1)
        pixels = frames.flatten(0, 1).float().masked_fill(~valid, 0)
        radius = self.radius
        kernel = 2 * radius + 1
        height, width = pixels.shape[-2:]
        border = (radius,) * 4
        padded = F.pad(pixels, border)
        support = F.pad(valid, border, value=False)
        count = F.avg_pool2d(valid.float(), kernel, stride=1, padding=radius, divisor_override=1)

        # If d_i = neighbour_i - centre, the z-score numerator times n is -sum(d_i).
        # Centre first to retain one-level variations on bright, nearly constant backgrounds.
        # For a valid centre, one d_i is zero, so n*sum(d_i²)-sum(d_i)² >= sum(d_i²).
        # Stream the offsets to avoid materialising a 25-plane unfolded neighbourhood.
        total = torch.zeros_like(pixels)
        squares = torch.zeros_like(pixels)
        for y in range(kernel):
            for x in range(kernel):
                if y == radius and x == radius:
                    continue
                neighbour = padded[..., y:y + height, x:x + width]
                present = support[..., y:y + height, x:x + width]
                delta = (neighbour - pixels).masked_fill(~present, 0)
                total = total + delta
                squares = torch.addcmul(squares, delta, delta)
        scatter = (count * squares - total.square()).clamp_min(0)
        contrast = -total * torch.rsqrt(scatter + self.epsilon * count.clamp_min(1).square())
        return contrast.masked_fill(~valid, 0).unflatten(0, (batch, times))
