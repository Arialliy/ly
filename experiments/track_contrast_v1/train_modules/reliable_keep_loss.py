"""Sparse release-teacher preservation on reliable targets and true background.

Cache construction is training-only. It uses actual component-to-GT matches;
missed or unreliable annotations and ignored regions cannot be distilled.
"""
from __future__ import annotations

import math
import numpy as np
import torch


def _paint(mask, frame, bbox, margin=0):
    x, y, w, h = map(float, bbox)
    height, width = mask.shape[1:]
    x0 = max(0, min(width, math.ceil(x-margin-.5)))
    y0 = max(0, min(height, math.ceil(y-margin-.5)))
    x1 = max(0, min(width, math.ceil(x+w+margin-.5)))
    y1 = max(0, min(height, math.ceil(y+h+margin-.5)))
    if w > 0 and h > 0 and x1 > x0 and y1 > y0:
        mask[frame, y0:y1, x0:x1] = True


def actual_reliable_boxes(teacher_prob, frame_annotations, matcher_module,
                          threshold=.5, core_margin=9, exclusion_margin=2):
    """Use supplied frozen prepare_gt/match_frame, never bbox-max detection.

Annotations use the frozen protocol's local xywh bbox, id and track_id.
Matched objects must also have unambiguous association, no label overlap,
and lie entirely in the core. All annotations failing this reliability test
(including missed targets), plus ignored/crowd objects, are forbidden.
"""
    probs = np.asarray(teacher_prob)
    if probs.ndim != 3 or len(frame_annotations) != len(probs):
        raise ValueError("teacher_prob [T,H,W] and per-frame annotations required")
    if not np.isfinite(probs).all():
        raise ValueError("nonfinite teacher probabilities")
    reliable = np.zeros(probs.shape, bool)
    forbidden = np.zeros(probs.shape, bool)
    kept_ids = []
    matched_count = 0
    invalid_slots = []
    height, width = probs.shape[1:]
    for ti, anns in enumerate(frame_annotations):
        # A spatially unlocatable target/ignore cannot safely be subtracted
        # from a background mask. Treat the entire slot as unavailable.
        try:
            boxes = np.asarray([a['bbox'] for a in anns], dtype=float).reshape(-1, 4)
            invalid = (not np.isfinite(boxes).all() or (boxes[:, 2:] <= 0).any()
                       or any('id' not in a or 'track_id' not in a for a in anns))
        except (ValueError, KeyError, TypeError):
            invalid = True
        if invalid:
            forbidden[ti] = True
            kept_ids.append([])
            invalid_slots.append(ti)
            continue
        gt, _ = matcher_module.prepare_gt(anns, height, width)
        rows, _, _ = matcher_module.match_frame(probs[ti], gt, threshold)
        by_id = {r['annotation_id']: r for r in rows}
        frame_kept = []
        for ann in anns:
            r = by_id.get(ann['id'])
            x, y, w, h = map(float, ann['bbox'])
            matched_count += int(bool(r and r['matched']))
            ok = (r is not None and bool(r['matched'])
                  and not r['association_ambiguous'] and not r['label_overlap_count']
                  and not r['partly_outside'] and not ann.get('ignore', 0)
                  and not ann.get('iscrowd', 0)
                  and x >= core_margin and y >= core_margin
                  and x+w <= width-core_margin and y+h <= height-core_margin)
            if ok:
                _paint(reliable, ti, ann['bbox'])
                frame_kept.append(ann['id'])
            else:
                _paint(forbidden, ti, ann['bbox'], exclusion_margin)
        kept_ids.append(frame_kept)
    reliable &= ~forbidden
    return {'reliable_target_mask': reliable, 'forbidden_mask': forbidden,
            'matched_annotation_ids': kept_ids,
            'stats': {'actual_matched_annotations': matched_count,
                      'reliable_annotations_before_forbidden_overlap': sum(map(len, kept_ids)),
                      'reliable_pixels': int(reliable.sum()),
                      'forbidden_pixels': int(forbidden.sum()),
                      'unlocatable_annotation_slots': invalid_slots}}


def build_keep_cache(teacher_prob, reliable_target_mask, valid_background_mask,
                     forbidden_mask, max_background=1024,
                     background_max_probability=.1, seed=46):
    """Cache all reliable-target pixels and up to 1024 background pixels/window.

Background selection is half largest teacher probabilities <= .1, then half
uniform without replacement from the remaining eligible pixels. Selection
depends only on the frozen teacher, training masks and fixed seed.
"""
    probs = np.asarray(teacher_prob, dtype=np.float32)
    target = np.asarray(reliable_target_mask, bool)
    bg = np.asarray(valid_background_mask, bool)
    forbidden = np.asarray(forbidden_mask, bool)
    if probs.ndim != 3 or any(a.shape != probs.shape for a in (target, bg, forbidden)):
        raise ValueError("cache masks/probabilities must share [T,H,W]")
    if not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any():
        raise ValueError("teacher probabilities must be finite in [0,1]")
    if max_background < 0 or not 0 <= background_max_probability <= 1:
        raise ValueError("invalid background cache configuration")
    # Enforce exclusions again at the cache boundary, including overlapping GT.
    target = target & ~forbidden
    eligible_bg = bg & ~forbidden & ~target & (probs <= background_max_probability)
    target_flat = np.flatnonzero(target)
    bg_flat = np.flatnonzero(eligible_bg)
    count = min(max_background, len(bg_flat))
    hard_n = count//2
    if count:
        order = np.lexsort((bg_flat, -probs.ravel()[bg_flat]))
        hard = bg_flat[order[:hard_n]]
        remaining = bg_flat[order[hard_n:]]
        uniform = np.random.default_rng(seed).choice(remaining, count-hard_n, replace=False)
        chosen_bg = np.r_[hard, uniform]
    else:
        chosen_bg = np.empty(0, np.int64)
    flat = np.r_[target_flat, chosen_bg]
    coordinates = np.column_stack(np.unravel_index(flat, probs.shape)).astype(np.int64)
    groups = np.r_[np.zeros(len(target_flat), np.int64), np.ones(len(chosen_bg), np.int64)]
    assert len(set(flat)) == len(flat) and not forbidden.ravel()[flat].any()
    return {'coordinates': coordinates, 'teacher_probabilities': probs.ravel()[flat].copy(),
            'groups': groups,
            'stats': {'target_pixels': len(target_flat), 'background_pixels': len(chosen_bg),
                      'background_hard_pixels': hard_n,
                      'background_uniform_pixels': len(chosen_bg)-hard_n,
                      'reliable_background_candidates': len(bg_flat), 'seed': int(seed)}}


def reliable_keep_loss(student_logits, indices, teacher_probabilities, groups):
    """Fixed .5(target MSE or0 + background MSE or0), then all-window mean."""
    if student_logits.ndim != 4:
        raise ValueError("student logits must be [B,T,H,W]")
    device = student_logits.device
    indices = torch.as_tensor(indices, dtype=torch.long, device=device)
    if not indices.numel():
        indices = indices.reshape(0, 4)
    teacher = torch.as_tensor(teacher_probabilities, dtype=student_logits.dtype,
                              device=device).detach()
    groups = torch.as_tensor(groups, dtype=torch.long, device=device)
    if indices.ndim != 2 or indices.shape[1] != 4 or teacher.shape != (len(indices),) or groups.shape != teacher.shape:
        raise ValueError("indices[N,4], teacher[N], groups[N] required")
    if not torch.isfinite(teacher).all() or ((teacher < 0) | (teacher > 1)).any():
        raise ValueError("invalid teacher probabilities")
    if ((groups != 0) & (groups != 1)).any():
        raise ValueError("keep groups are 0=reliable GT, 1=valid background")
    stats = {'target_pixels': int((groups == 0).sum()),
             'background_pixels': int((groups == 1).sum()),
             'windows': student_logits.shape[0],
             'reduction': 'mean_all_windows_of_half_targetmean_plus_half_bgmean'}
    if not len(indices):
        return student_logits.sum()*0., stats
    limits = torch.tensor(student_logits.shape, device=device)
    if (indices < 0).any() or (indices >= limits).any():
        raise ValueError("keep coordinate outside logits")
    selected = student_logits[tuple(indices.T)].sigmoid()
    squared = (selected-teacher).square()
    parts = []
    for bi in range(student_logits.shape[0]):
        for group in (0, 1):
            select = (indices[:, 0] == bi) & (groups == group)
            parts.append(squared[select].mean() if select.any() else squared.sum()*0.)
    return torch.stack(parts).sum()/(2*student_logits.shape[0]), stats
