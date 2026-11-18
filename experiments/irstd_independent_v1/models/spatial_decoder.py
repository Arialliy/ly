"""The native-resolution decoder shared by M0 and future video aggregators."""
import torch
from torch import nn
from torch.nn import functional as F


class SpatialDecoder(nn.Module):
    radius = 2

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Conv2d(32, 32, 3, padding=1),
                                     nn.Conv2d(32, 32, 3, padding=1)])

    def forward(self, features: torch.Tensor, center_valid: torch.Tensor) -> torch.Tensor:
        hidden = features.masked_fill(~center_valid, 0)
        for layer in self.layers:
            hidden = F.silu(layer(hidden)).masked_fill(~center_valid, 0)
        return hidden
