"""Training-only bbox descriptors and shared, teacher-mined background negatives.

Region coordinates are integer half-open [batch, frame, x0, y0, x1, y1]
indices in the original conv_out1 feature map. No projection or trainable
parameters are introduced. Callers own identity pairing and annotation masks.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _regions(regions, shape):
    if isinstance(regions, torch.Tensor):
        regions = regions.detach().cpu().numpy()
    arr = np.asarray(regions)
    if arr.size == 0:
        return np.empty((0, 6), dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] != 6 or not np.isfinite(arr).all():
        raise ValueError("regions must be finite [N,6] coordinates")
    if not np.equal(arr, np.floor(arr)).all():
        raise ValueError("ROI coordinates must already be integer pixel indices")
    arr = arr.astype(np.int64, copy=False)
    b, t, h, w = shape
    ok = ((arr[:, 0] >= 0) & (arr[:, 0] < b) & (arr[:, 1] >= 0)
          & (arr[:, 1] < t) & (arr[:, 2] >= 0) & (arr[:, 3] >= 0)
          & (arr[:, 4] <= w) & (arr[:, 5] <= h)
          & (arr[:, 4] > arr[:, 2]) & (arr[:, 5] > arr[:, 3]))
    if not ok.all():
        raise ValueError("ROI outside its window/frame or has nonpositive area")
    return arr


def pool_regions(features, regions, eps=1e-8):
    """BBox mean then L2 normalization, preserving gradients into original F."""
    if features.ndim != 5 or eps <= 0:
        raise ValueError("features must be [B,C,T,H,W] and eps positive")
    b, c, t, h, w = features.shape
    regions = _regions(regions, (b, t, h, w))
    if len(regions) == 0:
        return features.new_empty((0, c))
    # One gather keeps backward from allocating a full feature gradient for
    # every individual ROI slice. Repeated pixels deliberately contribute to
    # every ROI containing them; this remains the exact bbox arithmetic mean.
    pixel_indices, lengths = [], []
    for bi, ti, x0, y0, x1, y1 in regions.tolist():
        yy, xx = np.mgrid[y0:y1, x0:x1]
        pixel_indices.append((((bi*t+ti)*h+yy)*w+xx).ravel())
        lengths.append((y1-y0)*(x1-x0))
    indices = torch.as_tensor(np.concatenate(pixel_indices), device=features.device)
    pixels = features.permute(0, 2, 3, 4, 1).reshape(-1, c).index_select(0, indices)
    means = torch.stack([chunk.mean(0) for chunk in pixels.split(lengths)])
    return F.normalize(means, p=2, dim=-1, eps=eps)


def region_contrast_loss(features, target_regions, positive_pairs,
                         background_regions, negative_indices,
                         temperature=0.1):
    """One true positive plus shared background candidates per directed pair.

    Both local and global negatives are required. Pairs missing either give
    zero while the denominator stays the full requested pair count. -1 marks
    unavailable negatives; other targets are never implicit negatives.
The detector loss is not passed to, masked by, or reduced by this function.
"""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    pairs = torch.as_tensor(positive_pairs, dtype=torch.long, device=features.device)
    neg = torch.as_tensor(negative_indices, dtype=torch.long, device=features.device)
    if pairs.numel() == 0:
        pairs = pairs.reshape(0, 2)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("positive_pairs must be [P,2]")
    if neg.ndim != 2 or neg.shape[0] != len(pairs):
        raise ValueError("negative_indices must be [P,K]")
    stats = {"pairs_requested": len(pairs), "pairs_used": 0,
             "valid_negatives": 0, "reduction": "sum_complete_pairs_divided_by_all_requested_pairs"}
    if not len(pairs):
        return features.sum() * 0., stats
    if neg.shape[1] != 2:
        raise ValueError("exactly two negative slots are required: local and global")
    targets = pool_regions(features, target_regions)
    backgrounds = pool_regions(features, background_regions)
    if (pairs < 0).any() or (pairs >= len(targets)).any():
        raise ValueError("positive pair index outside target descriptor table")
    if (pairs[:, 0] == pairs[:, 1]).any():
        raise ValueError("positive pairs must contain distinct observations")
    if (neg < -1).any() or (neg >= len(backgrounds)).any():
        raise ValueError("negative index outside background descriptor table")
    valid_neg = neg >= 0
    denominator = len(pairs)
    keep = valid_neg.all(dim=1)
    if not keep.any():
        return features.sum() * 0., stats
    pairs, neg, valid_neg = pairs[keep], neg[keep], valid_neg[keep]
    anchors, positives = targets[pairs[:, 0]], targets[pairs[:, 1]]
    negative_features = backgrounds[neg.clamp_min(0)]
    positive_logits = (anchors * positives).sum(-1, keepdim=True) / temperature
    negative_logits = torch.einsum("pc,pkc->pk", anchors, negative_features) / temperature
    negative_logits = negative_logits.masked_fill(~valid_neg, -torch.inf)
    logits = torch.cat((positive_logits, negative_logits), dim=1)
    loss = F.cross_entropy(logits, torch.zeros(len(pairs), dtype=torch.long,
                                            device=features.device), reduction="sum")/denominator
    stats.update(pairs_used=int(keep.sum()), valid_negatives=int(valid_neg.sum()),
                 positive_cosine_mean=float((positive_logits.detach()*temperature).mean()),
                 negative_cosine_mean=float((negative_logits.detach()[valid_neg]*temperature).mean()))
    return loss, stats


def _box_sum(image, height, width):
    integral = np.pad(np.asarray(image, np.float64), ((1, 0), (1, 0)))
    integral = integral.cumsum(0).cumsum(1)
    return (integral[height:, width:] - integral[:-height, width:]
            - integral[height:, :-width] + integral[:-height, :-width])


def mine_background_regions(teacher_probs, valid_bg, anchor_regions,
                            global_window_indices, local_radius=32):
    """Mine same-sized local and global background ROIs from frozen teacher.

Input arrays are [B,T,H,W]; valid_bg must already exclude *all* task targets,
ignore/incomplete regions and invalid crop margins. Every pixel of a chosen
ROI must be valid. Local centers are within Euclidean local_radius pixels
of the anchor center; global ROIs use the supplied same-source global crop
and the same frame slot. Highest mean teacher probability wins, ties use
row-major order. Output regions are deduplicated; indices are [P,2].
"""
    if isinstance(teacher_probs, torch.Tensor):
        teacher_probs = teacher_probs.detach().cpu().numpy()
    if isinstance(valid_bg, torch.Tensor):
        valid_bg = valid_bg.detach().cpu().numpy()
    probs = np.asarray(teacher_probs)
    valid = np.asarray(valid_bg, dtype=bool)
    if probs.ndim != 4 or valid.shape != probs.shape or not np.isfinite(probs).all():
        raise ValueError("teacher probabilities and background masks must be finite [B,T,H,W]")
    if probs.min(initial=0) < 0 or probs.max(initial=1) > 1 or local_radius < 0:
        raise ValueError("invalid probability range or radius")
    anchors = _regions(anchor_regions, probs.shape)
    globals_ = np.asarray(global_window_indices, dtype=np.int64).reshape(-1)
    if len(globals_) == probs.shape[0] and len(globals_) != len(anchors):
        globals_ = globals_[anchors[:, 0]]
    if len(globals_) != len(anchors) or (globals_ < 0).any() or (globals_ >= len(probs)).any():
        raise ValueError("global_window_indices must select a global window for every anchor")
    table, lookup, cache = [], {}, {}
    indices = np.full((len(anchors), 2), -1, np.int64)
    for i, (bi, ti, x0, y0, x1, y1) in enumerate(anchors.tolist()):
        height, width = y1-y0, x1-x0
        for branch, window in enumerate((bi, int(globals_[i]))):
            key = (window, ti, height, width)
            if key not in cache:
                score = _box_sum(probs[window, ti], height, width)/(height*width)
                legal = _box_sum(~valid[window, ti], height, width) == 0
                cache[key] = score, legal
            score, legal = cache[key]
            possible = legal.copy()
            if branch == 0:
                yy, xx = np.ogrid[:possible.shape[0], :possible.shape[1]]
                possible &= (xx-x0)**2 + (yy-y0)**2 <= local_radius**2
            if not possible.any():
                continue
            flat = int(np.argmax(np.where(possible, score, -np.inf)))
            yy, xx = np.unravel_index(flat, score.shape)
            roi = (window, ti, int(xx), int(yy), int(xx+width), int(yy+height))
            if roi not in lookup:
                lookup[roi] = len(table)
                table.append(roi)
            indices[i, branch] = lookup[roi]
    return np.asarray(table, dtype=np.int64).reshape(-1, 6), indices
