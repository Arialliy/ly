"""Rejectable local temporal aggregation, one implementation for E1 and E2.

Input/output: B,C,T,H,W.  One shared descriptor, identity value projection,
zero-initialized output projection. All spatial neighborhoods include the center.
The learned null score competes with every valid spatial candidate separately in
both directions. No temporal wrapping or non-null re-normalization is performed.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

MODES = frozenset(("off", "single", "mutual", "mass", "current"))


def _slices(dy: int, dx: int, height: int, width: int):
    """Query and reference slices for reference(x)=query(x)+(dy,dx)."""
    y0, y1 = max(0, -dy), min(height, height - dy)
    x0, x1 = max(0, -dx), min(width, width - dx)
    if y1 <= y0 or x1 <= x0:
        return None
    return (slice(y0, y1), slice(x0, x1),
            slice(y0 + dy, y1 + dy), slice(x0 + dx, x1 + dx))


def _shift_map(x: torch.Tensor, dy: int, dx: int, fill: float = 0.):
    """At output pixel x, read input pixel x+(dy,dx); fill invalid pixels."""
    height, width = x.shape[-2:]
    sl = _slices(dy, dx, height, width)
    if sl is None:
        return x.new_full(x.shape, fill)
    qy, qx, ry, rx = sl
    return F.pad(x[..., ry, rx],
                 (qx.start, width-qx.stop, qy.start, height-qy.stop), value=fill)


def bidirectional_probabilities(scores: torch.Tensor, radius: int,
                                 null_logit: torch.Tensor):
    """Normalize both directions of shared pairwise scores independently.

    scores[n,k,x] is q_left(x) dot q_right(x+offset[k]) / temperature.
    Invalid edges MUST have score -inf. Return forward/reverse spatial
    probabilities, their null probabilities, and reverse probability for each
    forward edge. This public helper supports independent artificial-score tests.
    """
    offsets = [(dy, dx) for dy in range(-radius, radius+1)
               for dx in range(-radius, radius+1)]
    if scores.ndim != 4 or scores.shape[1] != len(offsets):
        raise ValueError("scores must have shape N,(2r+1)^2,H,W")
    reverse_scores = torch.stack([
        _shift_map(scores[:, len(offsets)-1-k], dy, dx, -math.inf)
        for k, (dy, dx) in enumerate(offsets)], dim=1)
    null = null_logit.to(dtype=scores.dtype).reshape(1, 1, 1, 1).expand(
        scores.shape[0], 1, scores.shape[2], scores.shape[3])
    forward_all = torch.softmax(torch.cat((scores, null), dim=1), dim=1)
    reverse_all = torch.softmax(torch.cat((reverse_scores, null), dim=1), dim=1)
    forward, reverse = forward_all[:, :-1], reverse_all[:, :-1]
    reverse_on_forward = torch.stack([
        _shift_map(reverse[:, len(offsets)-1-k], dy, dx)
        for k, (dy, dx) in enumerate(offsets)], dim=1)
    return (forward, reverse, forward_all[:, -1], reverse_all[:, -1],
            reverse_on_forward)


class RejectableTemporalAggregation(nn.Module):
    """One state-compatible implementation for single/mutual aggregation.

    ``mass`` redistributes each query's mutual non-null mass according to its
    forward probabilities. ``current`` keeps mutual weighting but replaces only
    this module's explicit reference frames by the respective current frame.
    ``off`` bypasses the module. Diagnostics do not add parameters.
    """
    def __init__(self, C: int = 32, descriptor_dim: int = 8, radius: int = 2,
                 mode: str = "mutual", temporal_chunk_size: int = 2,
                 checkpoint_chunks: bool = True):
        super().__init__()
        if radius < 0 or int(radius) != radius:
            raise ValueError("radius must be a nonnegative integer")
        if temporal_chunk_size < 1:
            raise ValueError("temporal_chunk_size must be positive")
        self.C = int(C)
        self.descriptor_dim = int(descriptor_dim)
        self.radius = int(radius)
        self.temperature = 0.1
        self.temporal_chunk_size = int(temporal_chunk_size)
        self.checkpoint_chunks = bool(checkpoint_chunks)
        self.descriptor = nn.Conv3d(C, descriptor_dim, 1, bias=False)
        self.null_logit = nn.Parameter(torch.zeros(()))
        self.output_projection = nn.Conv3d(C, C, 1, bias=False)
        nn.init.zeros_(self.output_projection.weight)
        self.set_mode(mode)
        self.collect_stats = True
        self.last_stats = {}
        self.offsets = [(dy, dx) for dy in range(-radius, radius+1)
                        for dx in range(-radius, radius+1)]

    def set_mode(self, mode: str):
        if mode not in MODES:
            raise ValueError(f"unknown aggregation mode {mode!r}")
        self.mode = mode
        return self

    def _scores(self, left: torch.Tensor, right: torch.Tensor):
        height, width = left.shape[-2:]
        scores = []
        for dy, dx in self.offsets:
            sl = _slices(dy, dx, height, width)
            if sl is None:
                scores.append(left.new_full((left.shape[0], height, width), -math.inf))
                continue
            qy, qx, ry, rx = sl
            score = (left[..., qy, qx] * right[..., ry, rx]).sum(1) / self.temperature
            scores.append(F.pad(score,
                (qx.start, width-qx.stop, qy.start, height-qy.stop), value=-math.inf))
        return torch.stack(scores, dim=1)

    def _weighted_difference(self, query, reference, weights):
        """Loop over K, avoiding materialization of an N,C,K,H,W value tensor."""
        height, width = query.shape[-2:]
        result = torch.zeros_like(query)
        for k, (dy, dx) in enumerate(self.offsets):
            sl = _slices(dy, dx, height, width)
            if sl is None:
                continue
            qy, qx, ry, rx = sl
            weighted = weights[:, k:k+1, qy, qx] * reference[..., ry, rx]
            result = result + F.pad(weighted,
                (qx.start, width-qx.stop, qy.start, height-qy.stop))
        return result - query * weights.sum(1, keepdim=True)

    def _pair(self, left, right, left_desc, right_desc, mode):
        scores = self._scores(left_desc, right_desc)
        forward, reverse, null_f, null_r, reverse_edges = bidirectional_probabilities(
            scores, self.radius, self.null_logit)
        if mode == "single":
            wf, wr = forward, reverse
        else:
            wf = forward * reverse_edges
            # Every mutual edge has the same product viewed from either endpoint.
            wr = torch.stack([
                _shift_map(wf[:, len(self.offsets)-1-k], dy, dx)
                for k, (dy, dx) in enumerate(self.offsets)], dim=1)
            if mode == "mass":
                tiny = torch.finfo(forward.dtype).tiny
                wf = forward * (wf.sum(1, keepdim=True) /
                                forward.sum(1, keepdim=True).clamp_min(tiny))
                wr = reverse * (wr.sum(1, keepdim=True) /
                                reverse.sum(1, keepdim=True).clamp_min(tiny))
        dl = self._weighted_difference(left, right, wf)
        dr = self._weighted_difference(right, left, wr)
        # Kept as a detached tensor until the end of forward, avoiding chunkwise sync.
        stats = torch.stack((null_f.detach().sum() + null_r.detach().sum(),
                             wf.detach().sum() + wr.detach().sum(),
                             scores.new_tensor(null_f.numel()+null_r.numel())))
        return dl, dr, stats

    def _pair_dispatch(self, left, right, ld, rd, mode):
        if mode != "current":
            return self._pair(left, right, ld, rd, mode)
        # Only the new branch replaces reference frames, with the original
        # number of available temporal references retained at window boundaries.
        dl, _, sl = self._pair(left, left, ld, ld, "mutual")
        dr, _, sr = self._pair(right, right, rd, rd, "mutual")
        return dl, dr, (sl + sr) * 0.5

    @torch.no_grad()
    def sparse_query_weights(self, features, queries, chunk_size=128):
        """Detached E2 provenance without affecting normal aggregation.

        Queries may be dicts with batch (default 0), slot, y, x, or rows
        [batch, slot, y, x]. Returned dictionaries retain query metadata and
        contain one ``references`` entry for each legal t-1/t+1 frame. Each
        entry includes full forward and correctly reverse-normalized edge
        probabilities, mutual weights, validity, and forward null probability.
        Coordinates refer to the supplied feature tile, including its halo.
        """
        if torch.is_tensor(queries):
            queries = queries.detach().cpu().tolist()
        batch, channels, frames, height, width = features.shape
        if channels != self.C or chunk_size < 1:
            raise ValueError("invalid feature channels or sparse chunk size")
        rows, directed = [], []
        for index, query in enumerate(queries):
            row = dict(query) if isinstance(query, dict) else dict(zip(
                ("batch", "slot", "y", "x"), map(int, query)))
            b, t, y, x = (int(row.get("batch", 0)), int(row["slot"]),
                          int(row["y"]), int(row["x"]))
            if not (0 <= b < batch and 0 <= t < frames and
                    0 <= y < height and 0 <= x < width):
                raise ValueError(f"invalid sparse query: {row}")
            row.update(batch=b, slot=t, y=y, x=x,
                       query_index=index, references=[])
            rows.append(row)
            for ref in (t-1, t+1):
                if 0 <= ref < frames:
                    directed.append((index, b, t, y, x, ref))
        if not directed:
            return rows
        descriptor = F.normalize(self.descriptor(features), p=2, dim=1,
                                 eps=1e-6).permute(0, 2, 3, 4, 1)
        offsets = torch.tensor(self.offsets, device=features.device,
                               dtype=torch.long)
        K = len(self.offsets)
        for start in range(0, len(directed), chunk_size):
            chunk = directed[start:start+chunk_size]
            coords = torch.tensor(chunk, device=features.device, dtype=torch.long)
            _, b, t, y, x, s = coords.unbind(1)
            ry, rx = y[:, None]+offsets[:, 0], x[:, None]+offsets[:, 1]
            valid = (ry >= 0) & (ry < height) & (rx >= 0) & (rx < width)
            q = descriptor[b, t, y, x]
            refs = descriptor[b[:, None], s[:, None], ry.clamp(0, height-1),
                              rx.clamp(0, width-1)]
            scores = (q[:, None] * refs).sum(-1) / self.temperature
            scores = scores.masked_fill(~valid, -math.inf)
            null = self.null_logit.to(scores.dtype).expand(len(chunk), 1)
            forward_all = torch.softmax(torch.cat((scores, null), dim=1), dim=1)
            # Each reference candidate y has its OWN local reverse denominator.
            # It may need query-frame features up to 2r away from original x.
            back_y = ry[:, :, None] + offsets[None, None, :, 0]
            back_x = rx[:, :, None] + offsets[None, None, :, 1]
            back_valid = ((back_y >= 0) & (back_y < height) &
                          (back_x >= 0) & (back_x < width) & valid[:, :, None])
            back = descriptor[b[:, None, None], t[:, None, None],
                              back_y.clamp(0, height-1), back_x.clamp(0, width-1)]
            reverse_scores = (refs[:, :, None] * back).sum(-1) / self.temperature
            reverse_scores = reverse_scores.masked_fill(~back_valid, -math.inf)
            reverse_null = self.null_logit.to(scores.dtype).expand(len(chunk), K, 1)
            reverse_all = torch.softmax(torch.cat((reverse_scores, reverse_null), dim=2), dim=2)
            indices = (K-1-torch.arange(K, device=features.device)).reshape(1, K, 1)
            reverse_edges = reverse_all.gather(2, indices.expand(len(chunk), -1, -1)).squeeze(-1)
            forward = forward_all[:, :-1]
            mutual = forward * reverse_edges
            packed = torch.stack((forward, reverse_edges, mutual), dim=1).cpu().tolist()
            null_list = forward_all[:, -1].cpu().tolist()
            valid_list = valid.cpu().tolist()
            for j, (index, _, _, _, _, ref) in enumerate(chunk):
                rows[index]["references"].append(dict(
                    ref_slot=ref, offsets=[list(o) for o in self.offsets],
                    valid=valid_list[j], forward=packed[j][0],
                    reverse_on_forward=packed[j][1], mutual=packed[j][2],
                    null_probability=null_list[j]))
        return rows

    def forward(self, features: torch.Tensor):
        if features.ndim != 5 or features.shape[1] != self.C:
            raise ValueError(f"expected B,{self.C},T,H,W")
        batch, channels, frames, height, width = features.shape
        if self.mode == "off" or frames < 2:
            if self.collect_stats:
                self.last_stats = {"null_probability": 0.0, "nonnull_mass": 0.0,
                                   "correction_relative_norm": 0.0,
                                   "raw_feature_norm": float(features.detach().norm()),
                                   "available_directed_frame_pairs": 0,
                                   "mode": self.mode}
            return features
        descriptors = F.normalize(self.descriptor(features), p=2, dim=1, eps=1e-6)
        left_parts, right_parts = [], []
        stats_total = features.new_zeros(3)
        mode = self.mode
        for start in range(0, frames-1, self.temporal_chunk_size):
            stop = min(start+self.temporal_chunk_size, frames-1)
            length = stop-start
            def pack(x, shift):
                return x[:, :, start+shift:stop+shift].permute(0, 2, 1, 3, 4).reshape(
                    batch*length, x.shape[1], height, width)
            args = (pack(features, 0), pack(features, 1),
                    pack(descriptors, 0), pack(descriptors, 1), mode)
            if self.checkpoint_chunks and self.training and torch.is_grad_enabled():
                dl, dr, stats = checkpoint(self._pair_dispatch, *args, use_reentrant=False)
            else:
                dl, dr, stats = self._pair_dispatch(*args)
            left_parts.append(dl.reshape(batch, length, channels, height, width).permute(0, 2, 1, 3, 4))
            right_parts.append(dr.reshape(batch, length, channels, height, width).permute(0, 2, 1, 3, 4))
            stats_total = stats_total + stats
        dl, dr = torch.cat(left_parts, dim=2), torch.cat(right_parts, dim=2)
        delta = F.pad(dl, (0, 0, 0, 0, 0, 1)) + F.pad(dr, (0, 0, 0, 0, 1, 0))
        counts = features.new_full((1, 1, frames, 1, 1), 2.)
        counts[:, :, 0] = 1.
        counts[:, :, -1] = 1.
        delta = delta / counts
        correction = self.output_projection(delta)
        if self.collect_stats:
            with torch.no_grad():
                raw_norm = features.detach().norm()
                stats = torch.stack((stats_total[0] / stats_total[2].clamp_min(1),
                                     stats_total[1] / stats_total[2].clamp_min(1),
                                     correction.detach().norm() / raw_norm.clamp_min(1e-12),
                                     raw_norm)).cpu().tolist()
                self.last_stats = dict(zip(("null_probability", "nonnull_mass",
                                            "correction_relative_norm", "raw_feature_norm"), stats))
                self.last_stats.update(mode=mode,
                    available_directed_frame_pairs=2*(frames-1))
        return features + correction
