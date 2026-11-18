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

    @torch.no_grad()
    def frozen_frontend_tiled(self, seq_imgs, core=128, halo=16):
        """Stitch full F/H from the frozen, spatially local baseline frontend.

        Every tile reads all 40 input frames. The official frontend's spatial
        receptive radius is 9; halo16 keeps each interior core away from tile
        padding. TPro mixes time at each spatial position, and frozen BN is
        pointwise. This method never applies PDA or TSFM to a spatial tile.
        """
        if self.training or self.base.training or self.adapter.training:
            raise RuntimeError("Spatial frontend execution is inference-only; call eval()")
        if isinstance(core, bool) or not isinstance(core, int) or core <= 0 or core % 4:
            raise ValueError("frontend core must be a positive multiple of four")
        if isinstance(halo, bool) or not isinstance(halo, int) or halo < 16 or halo % 2:
            raise ValueError("frontend halo must be even and at least 16")
        if seq_imgs.ndim != 5 or seq_imgs.shape[1:3] != (1, 40):
            raise ValueError("Expected [B,1,40,H,W] input")
        batch, _, time, height, width = seq_imgs.shape
        if height % 4 or width % 4:
            raise ValueError("Full spatial dimensions must be divisible by four")
        f = seq_imgs.new_empty((batch, 32, time, height, width))
        h = torch.empty_like(f)
        for y in range(0, height, core):
            y1 = min(height, y + core)
            ya, yb = max(0, y-halo), min(height, y1+halo)
            for x in range(0, width, core):
                x1 = min(width, x + core)
                xa, xb = max(0, x-halo), min(width, x1+halo)
                tile = seq_imgs[..., ya:yb, xa:xb].contiguous()
                tile_f = self.base.layer1(self.base.conv_in(tile))
                tile_h = self.base.TPro(tile_f.permute(0, 3, 4, 1, 2))
                f[..., y:y1, x:x1] = tile_f[..., y-ya:y1-ya, x-xa:x1-xa]
                h[..., y:y1, x:x1] = tile_h[..., y-ya:y1-ya, x-xa:x1-xa]
        return f, h

    @torch.no_grad()
    def forward_full_spatial(self, seq_imgs, frontend_core=128,
                             frontend_halo=16, head_temporal_chunk=1):
        """Bound baseline kernels while running M1 on the complete spatial map.

        The full 40-frame F/H tensors reside on the input device. There is one
        adapter call, with its normal within-window state. Only the original
        frozen frontend is spatially tiled. The original pointwise prediction
        head is evaluated in time chunks to bound each Conv3d invocation.
        Returned features and logits retain the normal shape/device contract.
        """
        WindowRecurrentAdapter._validate_chunk_size(head_temporal_chunk)
        if head_temporal_chunk is None:
            raise ValueError("head_temporal_chunk must be a positive integer")
        for module in (self.base.conv_out1, self.base.conv_out2):
            for layer in module.modules():
                if isinstance(layer, nn.Conv3d) and (
                    layer.kernel_size != (1, 1, 1) or layer.stride != (1, 1, 1)
                    or layer.padding != (0, 0, 0) or layer.dilation != (1, 1, 1)
                ):
                    raise ValueError("Time-chunked inference requires the original pointwise head")
        f, h = self.frozen_frontend_tiled(seq_imgs, frontend_core, frontend_halo)
        adapted = self.adapter(f, h, mode=self.mode)
        del f, h
        features = adapted.new_empty((adapted.shape[0], 8, adapted.shape[2],
                                      adapted.shape[3], adapted.shape[4]))
        logits = adapted.new_empty((adapted.shape[0], adapted.shape[2],
                                    adapted.shape[3], adapted.shape[4]))
        for start in range(0, adapted.shape[2], head_temporal_chunk):
            end = min(start + head_temporal_chunk, adapted.shape[2])
            part = self.base.conv_out1(adapted[:, :, start:end].contiguous())
            features[:, :, start:end] = part
            logits[:, start:end] = self.base.conv_out2(part).squeeze(dim=1)
        return features, logits
