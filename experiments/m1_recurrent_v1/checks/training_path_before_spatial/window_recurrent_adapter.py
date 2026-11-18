"""Window-local RFR feature propagation beside the original DeepPro-Plus TPro.

This is an adaptation, not the original RFR detector.  In particular, the first
state is X[0] (no PDA/TSFM), and a zero-initialized projection injects [X, M] into
DeepPro-Plus.  The default recurrent cell uses the pinned official PDA and TSFM
and requires their real DCN extension.  There is no substitute runtime operator.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class WindowRecurrentAdapter(nn.Module):
    """Map two [B,32,T,H,W] tensors to adapted TPro features.

    A forward call is exactly one window per batch element. State lives only in
    local variables. ``temporal_chunk_size`` groups traversal of already computed
    features; it never resets or detaches state and does not chunk the original
    temporal frontend. ``current_only`` still calls the same cell T-1 times.

    ``recurrent_cell`` exists for dependency injection in unit tests; production
    leaves it unset and loads the pinned RFR implementation.
    """

    MODES = ("history", "current_only")

    def __init__(self, state_channels=16, recurrent_cell=None,
                 temporal_chunk_size=None, checkpoint_recurrence=False):
        super().__init__()
        if state_channels <= 0:
            raise ValueError("state_channels must be positive")
        self._validate_chunk_size(temporal_chunk_size)
        self.state_channels = state_channels
        self.temporal_chunk_size = temporal_chunk_size
        self.checkpoint_recurrence = bool(checkpoint_recurrence)
        self.in_projection = nn.Conv2d(32, state_channels, 1)
        if recurrent_cell is None:
            from experiments.m1_recurrent_v1.upstream.rfr_cell import RFRCell
            recurrent_cell = RFRCell(channels=state_channels)
        self.recurrent_cell = recurrent_cell
        self.out_projection = nn.Conv2d(2 * state_channels, 32, 1)
        nn.init.zeros_(self.out_projection.weight)
        nn.init.zeros_(self.out_projection.bias)

    @staticmethod
    def _validate_chunk_size(value):
        if value is not None and (isinstance(value, bool) or
                                  not isinstance(value, int) or value <= 0):
            raise ValueError("temporal_chunk_size must be a positive integer or None")

    def forward(self, pre_tpro_features, original_tpro_features, mode="history",
                temporal_chunk_size=None):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        f, h = pre_tpro_features, original_tpro_features
        if f.ndim != 5 or f.shape[1] != 32 or f.shape != h.shape:
            raise ValueError("F and H must have identical [B,32,T,H,W] shapes")
        if min(f.shape[0], f.shape[2], f.shape[3], f.shape[4]) <= 0:
            raise ValueError("batch, time and spatial dimensions must be nonempty")
        if f.device != h.device or f.dtype != h.dtype:
            raise ValueError("F and H must share dtype and device")
        chunk = (self.temporal_chunk_size if temporal_chunk_size is None
                 else temporal_chunk_size)
        self._validate_chunk_size(chunk)
        chunk = f.shape[2] if chunk is None else chunk
        state = None
        outputs = []
        for start in range(0, f.shape[2], chunk):
            for t in range(start, min(start + chunk, f.shape[2])):
                current = self.in_projection(f[:, :, t])
                if state is None:
                    state = current
                else:
                    previous = state if mode == "history" else current
                    if self.checkpoint_recurrence and self.training and torch.is_grad_enabled():
                        state = checkpoint(self.recurrent_cell, previous, current,
                                           use_reentrant=False)
                    else:
                        state = self.recurrent_cell(previous, current)
                update = self.out_projection(torch.cat((current, state), dim=1))
                outputs.append(h[:, :, t] + update)
        return torch.stack(outputs, dim=2)


class FrozenDeepProWithAdapter(nn.Module):
    """Preserve the original head outputs and freeze every baseline parameter/BN.

    Freezing does not disable differentiation through the prediction head: the
    loss must still reach the adapter. ``base`` is the unmodified 40->40 detector.
    """

    def __init__(self, base, adapter, mode="history"):
        super().__init__()
        if mode not in WindowRecurrentAdapter.MODES:
            raise ValueError(f"invalid mode: {mode!r}")
        self.base = base.requires_grad_(False).eval()
        self.adapter = adapter
        self.mode = mode

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, seq_imgs):
        with torch.no_grad():
            f = self.base.layer1(self.base.conv_in(seq_imgs))
            h = self.base.TPro(f.permute(0, 3, 4, 1, 2))
        adapted = self.adapter(f, h, mode=self.mode)
        features = self.base.conv_out1(adapted)
        logits = self.base.conv_out2(features).squeeze(dim=1)
        return features, logits
