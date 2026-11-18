"""Independent object response readout on frozen DeepPro-Plus features.

Only local feature extraction may be tiled. Decode one shared query set from
the complete feature field; chunking attention frames or final masks changes
memory use, not the spatial domain accessible to a query.
"""
from __future__ import annotations

import math
from typing import Iterator

import torch
from torch import nn
from torch.nn import functional as F

D_MODEL = 32
NUM_QUERIES = 100
NUM_HEADS = 4
NUM_LAYERS = 2
TOKEN_STRIDE = 4
DEFAULT_FULL_SHAPE = (512, 640)


def xy_encoding(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Encode normalized ORIGINAL-image coordinates; final axis is d=32.

    Eight fixed Fourier frequencies 1,2,...,128 per axis, with sine and cosine.
    y and x must broadcast; no tile- or crop-relative coordinate normalization.
    """
    y, x = torch.broadcast_tensors(y, x)
    frequency = 2.0 ** torch.arange(8, device=y.device, dtype=y.dtype)
    phase_x = (2.0 * math.pi) * x[..., None] * frequency
    phase_y = (2.0 * math.pi) * y[..., None] * frequency
    return torch.cat((phase_x.sin(), phase_x.cos(),
                      phase_y.sin(), phase_y.cos()), dim=-1)


def time_encoding(length: int, *, device, dtype) -> torch.Tensor:
    position = torch.arange(length, device=device, dtype=dtype)[:, None]
    frequency = torch.exp(-math.log(10000.0) *
                          torch.arange(0, D_MODEL, 2, device=device, dtype=dtype) / D_MODEL)
    phase = position * frequency[None]
    return torch.stack((phase.sin(), phase.cos()), dim=-1).flatten(-2)


class Attention(nn.Module):
    """Explicit FP32 attention math, without dropout or fused CUDA dispatch."""

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(D_MODEL, D_MODEL)
        self.k_proj = nn.Linear(D_MODEL, D_MODEL)
        self.v_proj = nn.Linear(D_MODEL, D_MODEL)
        self.out_proj = nn.Linear(D_MODEL, D_MODEL)

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        batch, length, _ = query.shape
        head_dim = D_MODEL // NUM_HEADS
        q = self.q_proj(query).reshape(batch, length, NUM_HEADS, head_dim).transpose(1, 2)
        k = self.k_proj(memory).reshape(batch, -1, NUM_HEADS, head_dim).transpose(1, 2)
        v = self.v_proj(memory).reshape(batch, -1, NUM_HEADS, head_dim).transpose(1, 2)
        score = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
        attended = torch.matmul(score.softmax(dim=-1), v)
        return self.out_proj(attended.transpose(1, 2).reshape(batch, length, D_MODEL))


class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial_norm = nn.LayerNorm(D_MODEL)
        self.spatial_attention = Attention()
        self.temporal_norm = nn.LayerNorm(D_MODEL)
        self.temporal_attention = Attention()
        self.ffn_norm = nn.LayerNorm(D_MODEL)
        self.ffn = nn.Sequential(nn.Linear(D_MODEL, 64), nn.ReLU(), nn.Linear(64, D_MODEL))

    def forward(self, q: torch.Tensor, tokens: torch.Tensor, *,
                temporal_enabled: bool, frame_chunk: int) -> torch.Tensor:
        b, t, k, d = q.shape
        flat = q.reshape(b * t, k, d)
        memory = tokens.reshape(b * t, -1, d)
        spatial = []
        for lo in range(0, b * t, frame_chunk):
            hi = min(lo + frame_chunk, b * t)
            spatial.append(self.spatial_attention(self.spatial_norm(flat[lo:hi]), memory[lo:hi]))
        q = (flat + torch.cat(spatial, dim=0)).reshape(b, t, k, d)
        if temporal_enabled:
            temporal = q.permute(0, 2, 1, 3).reshape(b * k, t, d)
            normalized = self.temporal_norm(temporal)
            temporal = temporal + self.temporal_attention(normalized, normalized)
            q = temporal.reshape(b, k, t, d).permute(0, 2, 1, 3)
        return q + self.ffn(self.ffn_norm(q))


class ObjectResponseDecoder(nn.Module):
    def __init__(self, original: nn.Module, *, spatial_frame_chunk: int = 4):
        super().__init__()
        if spatial_frame_chunk < 1:
            raise ValueError('spatial_frame_chunk must be positive')
        self.original = original
        self.original.requires_grad_(False)
        self.original.eval()
        self.spatial_frame_chunk = int(spatial_frame_chunk)
        self.raw_encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 8, 3, padding=1), nn.ReLU(),
        )
        self.project = nn.Conv2d(16, D_MODEL, 1)
        self.query_embed = nn.Parameter(torch.randn(NUM_QUERIES, D_MODEL) * 0.02)
        self.layers = nn.ModuleList(DecoderLayer() for _ in range(NUM_LAYERS))
        self.presence_head = nn.Linear(D_MODEL, 1)
        self.mask_head = nn.Linear(D_MODEL, D_MODEL + 1)
        axis = (torch.arange(10, dtype=torch.float32) + 0.5) / 10.0
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        self.register_buffer('anchors_yx', torch.stack((yy, xx), dim=-1).reshape(NUM_QUERIES, 2))

    @property
    def config(self):
        return dict(original_frozen=True, original_bn='eval', d_model=D_MODEL,
                    num_queries=NUM_QUERIES, anchor_grid=[10, 10], anchor_coordinates='full-image normalized pixel locations',
                    num_heads=NUM_HEADS, decoder_layers=NUM_LAYERS, ffn_hidden=64,
                    dropout=0.0, token_pool='average', token_stride=TOKEN_STRIDE,
                    token_pool_ceil_mode=True, token_pool_count_include_pad=False,
                    position_encoding='sin/cos of original-image normalized xy, Fourier frequencies 2**arange(8)',
                    time_encoding='standard sinusoidal positions within current legal window',
                    raw_encoder='Conv2d(1,8,3,p1),ReLU,Conv2d(8,8,3,p1),ReLU',
                    projection='Conv2d(16,32,1)', spatial_frame_chunk=self.spatial_frame_chunk,
                    readout='max_query(sigmoid(presence) * sigmoid(dynamic_mask))',
                    mask_logit='dynamic_weight dot E + dynamic_bias',
                    normalization='pre-LayerNorm residual attention and FFN',
                    temporal_disabled='bypass temporal attention and its pre-LayerNorm only')

    def train(self, mode: bool = True):
        super().train(mode)
        self.original.eval()
        return self

    def original_forward(self, raw: torch.Tensor):
        self.original.eval()
        with torch.no_grad():
            return self.original(raw)

    def features(self, raw: torch.Tensor, H0: torch.Tensor, *,
                 origin=(0, 0), full_shape=DEFAULT_FULL_SHAPE) -> torch.Tensor:
        if raw.ndim != 5 or raw.shape[1] != 1:
            raise ValueError('raw must be [B,1,T,H,W]')
        b, _, t, h, w = raw.shape
        if H0.shape != (b, 8, t, h, w):
            raise ValueError('H0 must be [B,8,T,H,W] at input resolution')
        full_h, full_w = map(int, full_shape)
        y0, x0 = map(int, origin)
        if full_h <= 0 or full_w <= 0 or min(y0, x0) < 0 or y0 + h > full_h or x0 + w > full_w:
            raise ValueError('crop must be inside the declared full image')
        image = raw.permute(0, 2, 1, 3, 4).reshape(b * t, 1, h, w)
        original = H0.detach().permute(0, 2, 1, 3, 4).reshape(b * t, 8, h, w)
        E = self.project(torch.cat((original, self.raw_encoder(image)), dim=1))
        yy = (torch.arange(h, device=E.device, dtype=E.dtype) + y0 + 0.5) / full_h
        xx = (torch.arange(w, device=E.device, dtype=E.dtype) + x0 + 0.5) / full_w
        pe = xy_encoding(yy[:, None], xx[None, :]).permute(2, 0, 1)
        return (E + pe[None]).reshape(b, t, D_MODEL, h, w)

    def decode(self, E: torch.Tensor, temporal_enabled: bool) -> torch.Tensor:
        if E.ndim != 5 or E.shape[2] != D_MODEL:
            raise ValueError('E must be [B,T,32,H,W]')
        b, t, d, h, w = E.shape
        flat = E.reshape(b * t, d, h, w)
        pooled = F.avg_pool2d(flat, TOKEN_STRIDE, TOKEN_STRIDE,
                              ceil_mode=True, count_include_pad=False)
        tokens = pooled.flatten(2).transpose(1, 2).reshape(b, t, -1, d)
        anchors = self.anchors_yx.to(dtype=E.dtype)
        anchor_pe = xy_encoding(anchors[:, 0], anchors[:, 1])
        time_pe = time_encoding(t, device=E.device, dtype=E.dtype)
        q = (self.query_embed[None, None] + anchor_pe[None, None] + time_pe[None, :, None]).expand(b, -1, -1, -1)
        for layer in self.layers:
            q = layer(q, tokens, temporal_enabled=temporal_enabled,
                      frame_chunk=self.spatial_frame_chunk)
        return q

    def presence(self, q: torch.Tensor) -> torch.Tensor:
        return self.presence_head(q).squeeze(-1)

    def dynamic_parameters(self, q: torch.Tensor):
        parameters = self.mask_head(q)
        return parameters[..., :D_MODEL], parameters[..., D_MODEL]

    def mask_logits(self, E: torch.Tensor, q: torch.Tensor, *, spatial_chunk: int | None = None) -> torch.Tensor:
        """Dense masks for supervision; output allocation remains B*T*K*H*W."""
        b, t, d, h, w = E.shape
        weight, bias = self.dynamic_parameters(q)
        flat = E.flatten(-2)
        chunk = h * w if spatial_chunk is None else int(spatial_chunk)
        if chunk < 1:
            raise ValueError('spatial_chunk must be positive')
        pieces = [torch.matmul(weight, flat[..., lo:lo + chunk]) + bias[..., None]
                  for lo in range(0, h * w, chunk)]
        return torch.cat(pieces, dim=-1).reshape(b, t, q.shape[2], h, w)

    def response_map(self, E: torch.Tensor, q: torch.Tensor, *,
                     query_chunk: int = 8, spatial_chunk: int = 16384,
                     frame_chunk: int = 1) -> torch.Tensor:
        """Memory-bounded exact max reduction, using one GLOBAL query set."""
        if min(query_chunk, spatial_chunk, frame_chunk) < 1:
            raise ValueError('all chunk sizes must be positive')
        b, t, d, h, w = E.shape
        weight, bias = self.dynamic_parameters(q)
        presence = self.presence(q).sigmoid()
        flat = E.reshape(b * t, d, h * w)
        weight = weight.reshape(b * t, q.shape[2], d)
        bias = bias.reshape(b * t, q.shape[2])
        presence = presence.reshape(b * t, q.shape[2])
        frames = []
        for flo in range(0, b * t, frame_chunk):
            fhi = min(flo + frame_chunk, b * t)
            fields = []
            for slo in range(0, h * w, spatial_chunk):
                shi = min(slo + spatial_chunk, h * w)
                best = None
                for qlo in range(0, q.shape[2], query_chunk):
                    qhi = min(qlo + query_chunk, q.shape[2])
                    logits = torch.matmul(weight[flo:fhi, qlo:qhi], flat[flo:fhi, :, slo:shi])
                    logits = logits + bias[flo:fhi, qlo:qhi, None]
                    probability = logits.sigmoid() * presence[flo:fhi, qlo:qhi, None]
                    current = probability.max(dim=1).values
                    best = current if best is None else torch.maximum(best, current)
                fields.append(best)
            frames.append(torch.cat(fields, dim=-1))
        return torch.cat(frames, dim=0).reshape(b, t, h, w)

    @torch.no_grad()
    def tiled_features(self, raw: torch.Tensor, *, core: int = 128, halo: int = 16,
                       origin=(0, 0), full_shape=DEFAULT_FULL_SHAPE):
        """Inference-only extraction. NO query decoding occurs inside tiles."""
        if core < 1 or halo < 0:
            raise ValueError('invalid core/halo')
        b, _, t, h, w = raw.shape
        E = raw.new_empty((b, t, D_MODEL, h, w))
        z0 = raw.new_empty((b, t, h, w))
        for y in range(0, h, core):
            ye = min(y + core, h)
            ys, yse = max(y - halo, 0), min(ye + halo, h)
            for x in range(0, w, core):
                xe = min(x + core, w)
                xs, xse = max(x - halo, 0), min(xe + halo, w)
                tile = raw[..., ys:yse, xs:xse]
                H0_tile, z0_tile = self.original_forward(tile)
                E_tile = self.features(tile, H0_tile,
                                       origin=(int(origin[0]) + ys, int(origin[1]) + xs),
                                       full_shape=full_shape)
                sy, sx = y - ys, x - xs
                E[..., y:ye, x:xe] = E_tile[..., sy:sy + ye - y, sx:sx + xe - x]
                z0[..., y:ye, x:xe] = z0_tile[..., sy:sy + ye - y, sx:sx + xe - x]
        return E, z0

    def forward(self, raw: torch.Tensor, temporal_enabled: bool, *,
                origin=(0, 0), full_shape=DEFAULT_FULL_SHAPE):
        H0, z0 = self.original_forward(raw)
        E = self.features(raw, H0, origin=origin, full_shape=full_shape)
        q = self.decode(E, temporal_enabled=temporal_enabled)
        return dict(a=self.presence(q), q=q, E=E, z0=z0)
