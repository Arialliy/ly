"""Training-only temporal miss-risk surrogate for a DeepPro-Plus revision.

This is an original reference implementation, NOT SlotContrast, DEIM, TrackMAE,
or a detector implementation. Inputs must be pooled from the actual detector
logits; the caller owns annotations, valid background selection and grouping.
The score below is a surrogate, not a calibrated detection probability.
"""
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass
class TemporalRiskResult:
    loss: Tensor                    # [N] for reduction='none', scalar otherwise
    per_segment: Tensor             # [N]; inactive segments contain zero
    active: Tensor                  # [N], bool
    log_evidence: Tensor            # [N, T], -inf at invalid positions


def temporal_miss_risk(
    target_logits: Tensor,
    background_logits: Tensor,
    valid: Tensor,
    *,
    reduction: Literal['none', 'mean', 'sum'] = 'none',
) -> TemporalRiskResult:
    """Penalize segments with no target/background-separated detection evidence.

    For a valid frame, r = sigmoid(target_logit) * sigmoid(-background_logit).
    Segment loss is -log(1 - product_t(1-r_t)), evaluated without forming this
    subtraction. A log-space first-success recurrence retains usable gradients
    for very negative target logits. Temporal independence is NOT assumed as a
    claim about actual videos: this expression is only a training surrogate.

    Args:
        target_logits: floating [N, T], from a fixed, documented target sampler.
        background_logits: same shape, from valid non-target regions; exclude
            all labeled task objects, not only the small-target evaluation class.
        valid: boolean [N, T]. False entries are padding ONLY. Split occlusion,
            missing annotations and temporal discontinuities BEFORE this call.
        reduction: 'none' by default: integrate per-segment values using the
            existing per-track/per-window outer reduction, rather than silently
            replacing it. 'mean' averages active segments for isolated tests.

    This objective complements, never replaces, per-frame detection/negative
    supervision and existing keep constraints. It can be satisfied by isolated
    detections; final recovery, longest gaps, false positives and retention must
    still be evaluated independently.
    """
    if target_logits.ndim != 2 or target_logits.shape[1] == 0:
        raise ValueError('target_logits must have shape [N,T] with T > 0')
    if background_logits.shape != target_logits.shape or valid.shape != target_logits.shape:
        raise ValueError('all inputs must have identical [N,T] shape')
    if not target_logits.is_floating_point() or not background_logits.is_floating_point():
        raise TypeError('logits must be floating tensors')
    if target_logits.dtype != background_logits.dtype:
        raise TypeError('target/background dtypes must match')
    if valid.dtype != torch.bool:
        raise TypeError('valid must be boolean')
    if valid.device != target_logits.device or background_logits.device != target_logits.device:
        raise ValueError('all inputs must be on the same device')
    if reduction not in ('none', 'mean', 'sum'):
        raise ValueError('unknown reduction')

    # Avoid fragile fp16 log computations while preserving the autograd graph.
    dtype = torch.float64 if target_logits.dtype == torch.float64 else torch.float32
    a, b = target_logits.to(dtype), background_logits.to(dtype)
    # Valid positions must contain finite logits. Mask arbitrary padding first.
    a = torch.where(valid, a, torch.zeros_like(a))
    b = torch.where(valid, b, torch.zeros_like(b))
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise ValueError('non-finite logits at a valid position')

    log_pa = F.logsigmoid(a)
    log_r = log_pa + F.logsigmoid(-b)
    # 1 - sigmoid(a)*sigmoid(-b) = sigmoid(-a) + sigmoid(a)*sigmoid(b)
    log_not_r = torch.logaddexp(F.logsigmoid(-a), log_pa + F.logsigmoid(b))

    n, t = a.shape
    log_hit = a.new_full((n,), -torch.inf)
    log_no_hit = a.new_zeros(n)
    for k in range(t):
        proposed = torch.logaddexp(log_hit, log_no_hit + log_r[:, k])
        log_hit = torch.where(valid[:, k], proposed, log_hit)
        log_no_hit = log_no_hit + torch.where(valid[:, k], log_not_r[:, k], 0.0)

    active = valid.any(dim=1)
    safe_log_hit = torch.where(active, log_hit, torch.zeros_like(log_hit))
    # Tiny positive log probabilities may occur from roundoff at near-certain hit.
    per_segment = -safe_log_hit.clamp(max=0)
    # Attach a zero-gradient path, including for completely empty batches.
    per_segment = per_segment + (a.sum(dim=1) + b.sum(dim=1)) * 0.0
    if reduction == 'none':
        loss = per_segment
    elif reduction == 'sum':
        loss = per_segment.sum()
    else:
        loss = per_segment.sum() / active.sum().clamp(min=1)
    log_evidence = torch.where(valid, log_r, a.new_full((), -torch.inf))
    return TemporalRiskResult(loss, per_segment, active, log_evidence)
