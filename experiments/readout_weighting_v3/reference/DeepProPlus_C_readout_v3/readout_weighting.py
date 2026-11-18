"""Independent, partial adaptation of DEIM's detached negative-confidence weight.

Reference: Intellindust-AI-Lab/DEIM, engine/deim/deim_criterion.py,
DEIMCriterion.loss_labels_mal, specifically the negative weight and BCE call.

This is NOT full MAL: no DETR matching, box IoU, soft positive quality targets,
DEIM normalization, or DEIM training schedule is reproduced.
The 'matched_confidence' rescaling is our adaptation, not an author method.
Returns unreduced terms: the caller must retain its existing window reduction.
No model, labels, sampling, optimizer, or inference thresholds are changed here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import math

import torch
from torch import Tensor
import torch.nn.functional as F

Mode = Literal['unit', 'confidence', 'matched_confidence']


@dataclass(frozen=True)
class ReadoutTerms:
    positive: Tensor
    negative: Tensor
    positive_valid: Tensor
    negative_valid: Tensor
    negative_weight: Tensor
    negative_probability: Tensor


def _prepare(z: Tensor, valid: Tensor | None, name: str) -> tuple[Tensor, Tensor]:
    if z.ndim != 2 or not z.is_floating_point():
        raise ValueError(f'{name} must be a floating tensor [groups, samples].')
    if valid is None:
        valid = torch.ones_like(z, dtype=torch.bool)
    if valid.dtype != torch.bool or valid.shape != z.shape or valid.device != z.device:
        raise ValueError(f'{name} validity must be a same-device Boolean mask of the same shape.')
    if not torch.isfinite(z.detach()[valid]).all():
        raise ValueError(f'{name} contains non-finite VALID entries.')
    dtype = torch.float64 if z.dtype == torch.float64 else torch.float32
    # Invalid padding may contain NaN/Inf. Remove it before any differentiable op.
    safe = torch.where(valid, z.to(dtype), torch.zeros_like(z, dtype=dtype))
    return safe, valid


def readout_terms(
    positive_logits: Tensor,
    negative_logits: Tensor,
    *,
    mode: Mode = 'unit',
    gamma: float = 2.0,
    positive_valid: Tensor | None = None,
    negative_valid: Tensor | None = None,
) -> ReadoutTerms:
    """Return the original positive BCE and selected negative BCE weighting.

    Args:
        positive_logits, negative_logits: [G, P] and [G, N]; G is an EXISTING
            reduction group (e.g. window). Do not silently regroup by track.
        mode: unit = original C-frame; confidence = detached sigmoid(z)^gamma;
            matched_confidence = confidence weights rescaled per group to match unit mode's
            sum of absolute NEGATIVE-LOGIT gradients at the SAME logits
            with equal per-sample reduction. It does not match parameter gradients
            or the budgets of two independently evolving models.
        gamma: Fixed proposal value 2.0, as in the inspected DEIM default.

    The budget rescaling is our adaptation, not a DEIM component. If original
    per-background weights are unequal, the caller must extend budget matching
    to those weights; this implementation intentionally does not guess them.
    """
    if mode not in ('unit', 'confidence', 'matched_confidence'):
        raise ValueError(f'Unknown mode: {mode}')
    if not math.isfinite(gamma) or gamma < 0:
        raise ValueError('gamma must be finite and nonnegative.')
    if positive_logits.device != negative_logits.device:
        raise ValueError('Positive and negative logits must be on the same device.')
    pos, pm = _prepare(positive_logits, positive_valid, 'positive_logits')
    neg, nm = _prepare(negative_logits, negative_valid, 'negative_logits')
    if pos.shape[0] != neg.shape[0]:
        raise ValueError('Positive and negative logits must have the same number of groups.')

    with torch.no_grad():
        p = neg.sigmoid()
        confidence = p.pow(gamma)
        if mode == 'unit':
            weight = torch.ones_like(neg)
        elif mode == 'confidence':
            weight = confidence
        else:
            # Match sum_i d[softplus(z_i)]/dz_i to the UNIT-weight branch.
            # alpha = sum(p) / sum(p ** (gamma+1)), weight_i = alpha*p_i**gamma.
            # Compute in log space to avoid underflow for extreme weak scores.
            logp = F.logsigmoid(neg)
            masked_logp = torch.where(nm, logp, torch.full_like(logp, -torch.inf))
            exists = nm.any(dim=1, keepdim=True)
            log_num = torch.logsumexp(masked_logp, dim=1, keepdim=True)
            log_den = torch.logsumexp((gamma + 1) * masked_logp, dim=1, keepdim=True)
            log_num = torch.where(exists, log_num, torch.zeros_like(log_num))
            log_den = torch.where(exists, log_den, torch.zeros_like(log_den))
            log_weight = gamma * logp + log_num - log_den
            log_weight = torch.where(nm, log_weight, torch.zeros_like(log_weight))
            weight = log_weight.exp()
        weight = torch.where(nm, weight, torch.zeros_like(weight))
        p = torch.where(nm, p, torch.zeros_like(p))

    positive = torch.where(pm, F.softplus(-pos), torch.zeros_like(pos))
    negative = torch.where(nm, weight * F.softplus(neg), torch.zeros_like(neg))
    return ReadoutTerms(positive, negative, pm, nm, weight, p)


def equal_mean_per_group(terms: Tensor, valid: Tensor) -> Tensor:
    """OPTIONAL reference reducer, only when it matches the local protocol.

    An empty group contributes connected zero. The caller remains responsible
    for the ORIGINAL outer/window weights and DDP normalization.
    """
    if terms.ndim != 2 or valid.shape != terms.shape or valid.dtype != torch.bool:
        raise ValueError('Expected terms [G,N] and a matching Boolean mask.')
    n = valid.sum(dim=1).clamp_min(1).to(terms.dtype)
    return torch.where(valid, terms, torch.zeros_like(terms)).sum(dim=1) / n
