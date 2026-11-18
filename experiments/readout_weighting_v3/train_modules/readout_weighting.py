"""New negative-readout weighting only; the C-v2 sampler and readouts stay fixed.

The unit mode is independently evaluated for equivalence tests. Existing FRAME
training is reused; only confidence (D) and matched_confidence (N) are new runs.
The N normalization group is the LAST axis: one original eight-frame segment,
before the unchanged segment/window/fixed-eight-window reduction.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


MODES = ("unit", "confidence", "matched_confidence")


def _mask(value, like, name):
    if value is None:
        return torch.ones_like(like, dtype=torch.bool)
    value = torch.as_tensor(value, device=like.device)
    if value.dtype != torch.bool or value.shape != like.shape:
        raise ValueError(name + " must be a bool tensor matching logits")
    return value


def readout_terms(q_positive, q_negative, positive_valid=None,
                  negative_valid=None, mode="unit", gamma=2.0,
                  sample_weight=None):
    """Return unreduced positive/negative terms and detached negative weights.

Each last-axis row is one normalization group. sample_weight is an OPTIONAL
fixed original negative-reduction weight: it enters BOTH sums defining N but
does not itself multiply returned terms. The caller retains its own reducer.
No such additional weight is used in the registered equal-frame C-v3 groups.

For N, log_a = logsumexp(log(mu)+log(p)) -
logsumexp(log(mu)+(gamma+1)*log(p)). Combined log weights are evaluated before
exponentiation, so p^3 underflow never creates a 0/0 normalization. No epsilon,
cap, or probability clipping is used. Empty/zero-weight groups have zero terms.
"""
    if mode not in MODES:
        raise ValueError("unknown weighting mode")
    if gamma != 2.0:
        raise ValueError("the registered gamma is fixed at 2")
    if (not isinstance(q_positive, torch.Tensor)
            or not isinstance(q_negative, torch.Tensor)
            or not q_positive.is_floating_point() or not q_negative.is_floating_point()
            or q_positive.shape != q_negative.shape or q_positive.ndim < 1
            or q_positive.device != q_negative.device):
        raise ValueError("positive/negative logits must be same-shaped floating tensors")
    dtype = torch.float64 if (q_positive.dtype == torch.float64 or q_negative.dtype == torch.float64) else torch.float32
    qp, qm = q_positive.to(dtype), q_negative.to(dtype)
    pv = _mask(positive_valid, qp, "positive_valid")
    nv = _mask(negative_valid, qm, "negative_valid")
    if not bool(torch.isfinite(qp[pv]).all()) or not bool(torch.isfinite(qm[nv]).all()):
        raise ValueError("valid logits must be finite")
    # Invalid padding must not inject NaN into a masked backward operation.
    qp = torch.where(pv, qp, 0.)
    qm = torch.where(nv, qm, 0.)
    if sample_weight is None:
        mu = torch.ones_like(qm, dtype=torch.float64)
    else:
        if isinstance(sample_weight, torch.Tensor) and sample_weight.requires_grad:
            raise ValueError("original sample weights must be fixed")
        mu = torch.as_tensor(sample_weight, dtype=torch.float64, device=qm.device)
        if mu.shape != qm.shape or not bool(torch.isfinite(mu).all()) or bool((mu < 0).any()):
            raise ValueError("sample_weight must be finite, nonnegative and match logits")
    negative_active = nv & (mu > 0)
    with torch.no_grad():
        logp = F.logsigmoid(qm.detach().double())
        if mode == "unit":
            logw = torch.zeros_like(logp)
            loga = torch.zeros_like(logp[..., :1])
        elif mode == "confidence":
            logw = gamma * logp
            loga = torch.zeros_like(logp[..., :1])
        else:
            logmu = torch.where(negative_active, mu.log(), -torch.inf)
            numerator = torch.logsumexp(logmu + logp, dim=-1, keepdim=True)
            denominator = torch.logsumexp(logmu + (gamma + 1.) * logp, dim=-1, keepdim=True)
            active_group = negative_active.any(dim=-1, keepdim=True)
            # Avoid (-inf)-(-inf) even in an inactive diagnostic branch.
            numerator = torch.where(active_group, numerator, 0.)
            denominator = torch.where(active_group, denominator, 0.)
            loga = numerator - denominator
            logw = loga + gamma * logp
        logw = torch.where(negative_active, logw, -torch.inf)
        # D follows the author's direct sigmoid(...).detach().pow(2) arithmetic.
        # N needs the combined log weight because its separate scale can overflow.
        weight = (qm.detach().sigmoid().pow(gamma) if mode == "confidence"
                  else logw.exp().to(dtype))
        weight = torch.where(negative_active, weight, 0.)
        if not bool(torch.isfinite(weight).all()):
            raise ValueError("negative weight is not representable in the loss dtype")
        p = logp.exp()
        unit_budget = torch.where(negative_active, mu*p, 0.).sum(dim=-1)
        # Reflect the actual cast weight used by the differentiable loss.
        weighted_budget = torch.where(negative_active, mu*weight.double()*p, 0.).sum(dim=-1)
    positive = torch.where(pv, -F.logsigmoid(qp), 0.)
    negative = torch.where(negative_active, weight * -F.logsigmoid(-qm), 0.)
    return {"positive": positive, "negative": negative,
            "positive_valid": pv, "negative_valid": nv,
            "negative_active": negative_active,
            "negative_weight": weight, "negative_log_weight": logw,
            "group_log_scale": loga.squeeze(-1),
            "sample_weight": mu, "unit_group_negative_gradient_budget": unit_budget,
            "weighted_group_negative_gradient_budget": weighted_budget}


def _index_tensor(values, device, name, ndim):
    raw = torch.as_tensor(values, device=device)
    if raw.ndim != ndim or raw.dtype == torch.bool:
        raise ValueError(name + " has invalid dimensions or type")
    if raw.is_floating_point() and (not bool(torch.isfinite(raw).all()) or not bool((raw == raw.floor()).all())):
        raise ValueError(name + " must contain integer indices")
    return raw.long()


def readout_weighting_loss(risk_result, segment_indices, segment_windows,
                           num_windows=8, mode="unit"):
    """Apply new terms to existing C-v2 readouts without changing exposure.

risk_result is the unchanged temporal_risk_loss dictionary. It supplies pooled
actual logits and verified background availability. D and N use the same
segment padding and within-segment mean, active-segment mean per window, and
fixed num_windows mean as the earlier FRAME control. Overlapping observations
remain repeated segment exposures and their gradients accumulate naturally.
"""
    if mode not in MODES or not isinstance(num_windows, int) or num_windows < 1:
        raise ValueError("invalid mode or fixed window count")
    qp, qm = risk_result["q_positive"], risk_result["q_negative"]
    if qp.ndim != 1 or qm.shape != qp.shape:
        raise ValueError("existing pooled logits must be equal-length vectors")
    seg = _index_tensor(segment_indices, qp.device, "segment_indices", 2)
    windows = _index_tensor(segment_windows, qp.device, "segment_windows", 1)
    if seg.shape[1] == 0 or windows.shape != (len(seg),):
        raise ValueError("segment/window shape mismatch")
    if bool((seg < -1).any()) or bool((seg >= len(qp)).any()):
        raise ValueError("invalid segment observation index")
    if bool((windows < 0).any()) or bool((windows >= num_windows).any()):
        raise ValueError("segment outside fixed windows")
    valid = seg >= 0
    if seg.shape[1] > 1 and bool(((~valid[:, :-1]) & valid[:, 1:]).any()):
        raise ValueError("segment padding must be suffix-only")
    exposure = seg[valid]
    if not torch.equal(exposure, risk_result["exposure_indices"]):
        raise ValueError("segment exposure differs from frozen v2 readout")
    lengths = valid.sum(dim=1)
    active = lengths > 0
    counts = torch.stack([((windows == wi) & active).sum() for wi in range(num_windows)])
    if (not torch.equal(lengths, risk_result["segment_lengths"])
            or not torch.equal(active, risk_result["segment_active"])
            or not torch.equal(counts, risk_result["window_segment_counts"])):
        raise ValueError("segment reduction differs from frozen v2 readout")
    if exposure.numel() and not bool(risk_result["valid_observations"][exposure].all()):
        raise ValueError("an exposed observation has no valid background")
    # This zero retains the exact already-connected graph for empty windows.
    zero = risk_result["loss_frame"] * 0.
    safe = torch.where(valid, seg, len(qp))
    a = torch.cat((qp, zero.reshape(1)))[safe]
    b = torch.cat((qm, zero.reshape(1)))[safe]
    terms = readout_terms(a, b, valid, valid, mode=mode)
    denominator = lengths.clamp(min=1)
    per_positive = terms["positive"].sum(1) / denominator
    per_negative = terms["negative"].sum(1) / denominator
    # Sum the per-frame terms before reduction, preserving the original FRAME
    # floating-point ordering as well as its mathematical weighting.
    per_total = (terms["positive"] + terms["negative"]).sum(1) / denominator
    window_parts = {key: [] for key in ("positive", "negative", "total")}
    for wi in range(num_windows):
        selected = (windows == wi) & active
        for key, values in (("positive", per_positive), ("negative", per_negative), ("total", per_total)):
            window_parts[key].append(values[selected].mean() if bool(selected.any()) else zero)
    window_parts = {key: torch.stack(values) for key, values in window_parts.items()}
    budget_f = terms["unit_group_negative_gradient_budget"]
    budget_v = terms["weighted_group_negative_gradient_budget"]
    group_reduction_weight = (active.double() / (num_windows
        * counts[windows].clamp(min=1).double() * lengths.clamp(min=1).double()))
    relative = torch.where(budget_f > 0, (budget_v-budget_f).abs()/budget_f, 0.)
    def average(x):
        return x.mean() if x.numel() else zero.detach()
    diagnostics = {
        "segments_active": active.sum(), "observations_exposed": valid.sum(),
        "windows_active": (counts > 0).sum(),
        "positive_loss": window_parts["positive"].mean().detach(),
        "negative_loss": window_parts["negative"].mean().detach(),
        "negative_weight_mean": average(terms["negative_weight"][valid]),
        "negative_weight_min": terms["negative_weight"][valid].min() if bool(valid.any()) else zero.detach(),
        "negative_weight_max": terms["negative_weight"][valid].max() if bool(valid.any()) else zero.detach(),
        "group_log_scale_mean": average(terms["group_log_scale"][active]),
        "group_budget_relative_error_max": relative[active].max() if bool(active.any()) else zero.detach(),
        "raw_segment_unit_budget_sum": budget_f.sum(),
        "raw_segment_weighted_budget_sum": budget_v.sum(),
        "unit_negative_readout_gradient_l1": (budget_f*group_reduction_weight).sum(),
        "weighted_negative_readout_gradient_l1": (budget_v*group_reduction_weight).sum(),
    }
    return {"loss": window_parts["total"].mean(),
            "loss_positive": window_parts["positive"].mean(),
            "loss_negative": window_parts["negative"].mean(),
            "per_segment_positive": per_positive, "per_segment_negative": per_negative,
            "per_segment_total": per_total,
            "per_window_positive": window_parts["positive"],
            "per_window_negative": window_parts["negative"],
            "per_window_total": window_parts["total"],
            "segment_positive_logits": a, "segment_negative_logits": b,
            "segment_valid": valid, "segment_windows": windows,
            "negative_weight": terms["negative_weight"],
            "negative_log_weight": terms["negative_log_weight"],
            "group_log_scale": terms["group_log_scale"],
            "group_reduction_weight": group_reduction_weight,
            "unit_group_negative_gradient_budget": budget_f,
            "weighted_group_negative_gradient_budget": budget_v,
            "diagnostics": diagnostics}
