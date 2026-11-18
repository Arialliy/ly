"""Shared frozen DUF ×2 initialization for five unique low-resolution frames."""
import hashlib
from pathlib import Path

import torch
from torch import nn

if __package__:
    from .third_party.duf_arch import DUF
else:
    from third_party.duf_arch import DUF


DEFAULT_WEIGHTS = Path(__file__).with_name('third_party') / 'DUF_x2_16L_official-39537cb9.pth'
WEIGHTS_SHA256 = '39537cb9b3e899a8ed2b5c274f4ba372df4eec55558ebf9e86e0c595e61a5b08'


class FiveFrameDUF(nn.Module):
    """Route the five distinct observations into the seven published input slots.

    The boundary-repeat sequence (0,0,1,2,3,4,4) keeps the reference in slot 3.
    Input is floating RGB [B,5,3,H,W] in [0,1]. Output is raw model RGB at ×2.
    """

    def __init__(self, network):
        super().__init__()
        self.network = network.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        # Parent model training must preserve the published BatchNorm statistics.
        return super().train(False)

    def forward(self, frames):
        if frames.ndim != 5 or frames.shape[1:3] != (5, 3) or not frames.is_floating_point():
            raise ValueError('Expected floating-point RGB [B,5,3,H,W].')
        seven = torch.cat((frames[:, :1], frames, frames[:, -1:]), dim=1)
        return self.network(seven)


def load_backbone(device='cpu'):
    """Use the BasicSR conversion of the original DUF-16L ×2 checkpoint."""
    if hashlib.sha256(DEFAULT_WEIGHTS.read_bytes()).hexdigest() != WEIGHTS_SHA256:
        raise ValueError('Unexpected DUF checkpoint SHA-256.')
    state = torch.load(DEFAULT_WEIGHTS, map_location='cpu', weights_only=True)
    network = DUF(scale=2, num_layer=16, adapt_official_weights=True)
    network.load_state_dict(state['params'], strict=True)
    return FiveFrameDUF(network).to(device)
