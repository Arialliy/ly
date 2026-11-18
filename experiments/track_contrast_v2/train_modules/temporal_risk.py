"""Actual-logit ROI adapter for the supplied, unchanged C-v2 reference core.

The score is a training surrogate, not component-match success probability.
Only training annotations define regions/segments. The sampler must split
missing annotations, identity uncertainty, frame discontinuities and missing
background before calling this adapter. No inference module is introduced.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch


REFERENCE_PATH = Path(__file__).resolve().parents[3] / "deeppro_c_v2_reference/temporal_risk.py"
_REFERENCE_NAME = "_deep_pro_c_v2_supplied_temporal_risk"
if _REFERENCE_NAME not in sys.modules:
    _spec = importlib.util.spec_from_file_location(_REFERENCE_NAME, REFERENCE_PATH)
    _reference = importlib.util.module_from_spec(_spec)
    # dataclasses consult sys.modules while executing their class definition.
    sys.modules[_REFERENCE_NAME] = _reference
    _spec.loader.exec_module(_reference)
temporal_miss_risk = sys.modules[_REFERENCE_NAME].temporal_miss_risk


def _integers(values, columns=None, label="indices"):
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    values = np.asarray(values)
    if not values.size and columns is not None:
        values = values.reshape(0, columns)
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise ValueError(label + " must contain finite integer indices")
    if columns is not None and (values.ndim != 2 or values.shape[1] != columns):
        raise ValueError(label + " has incorrect column count")
    return values.astype(np.int64, copy=False)


def _regions(values, shape):
    values = _integers(values, 6, "regions")
    b, t, h, w = shape
    ok = ((values[:, 0] >= 0) & (values[:, 0] < b)
          & (values[:, 1] >= 0) & (values[:, 1] < t)
          & (values[:, 2] >= 0) & (values[:, 3] >= 0)
          & (values[:, 4] <= w) & (values[:, 5] <= h)
          & (values[:, 4] > values[:, 2]) & (values[:, 5] > values[:, 3]))
    if not ok.all():
        raise ValueError("ROI is outside its actual-logit window or has zero area")
    return values


def pool_logit_regions(logits, regions):
    """Arithmetic mean of actual logits over each integer half-open bbox.

One gather has one gradient scatter, including exactly additive contributions
from overlapping ROIs. No sigmoid, peak selection or learned projection occurs
before averaging. Half/bfloat16 inputs compute the mean in float32.
"""
    if logits.ndim != 4 or not logits.is_floating_point():
        raise ValueError("actual logits must be floating [B,T,H,W]")
    regions = _regions(regions, logits.shape)
    dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    values = logits.to(dtype)
    if not len(regions):
        return values.reshape(-1)[:0]
    _, t, h, w = logits.shape
    indices, lengths = [], []
    for bi, ti, x0, y0, x1, y1 in regions.tolist():
        yy, xx = np.mgrid[y0:y1, x0:x1]
        indices.append((((bi*t+ti)*h+yy)*w+xx).ravel())
        lengths.append((y1-y0)*(x1-x0))
    index = torch.as_tensor(np.concatenate(indices), device=logits.device)
    pixels = values.reshape(-1).index_select(0, index)
    if not bool(torch.isfinite(pixels).all()):
        raise ValueError("nonfinite actual logits in a sampled ROI")
    return torch.stack([part.mean() for part in pixels.split(lengths)])


def _mean_or_zero(values, zero):
    return values.mean() if values.numel() else zero


def temporal_risk_loss(logits, target_regions, background_regions,
                       negative_indices, segment_indices, segment_windows,
                       num_windows=8):
    """Return gap and frame losses with the same segment/window exposure.

Input region rows are [window, slot, x0, y0, x1, y1]. Negative indices [N,2]
refer to frozen local/global candidates, with -1 for an unavailable branch.
Each background ROI is read by its mean actual logit; the higher valid mean is
q_negative. A valid global candidate suffices when local background is absent.

segment_indices [S,L] contain target indices, followed only by -1 padding.
Targets may be shared across overlapping segments. A referenced target must
have a background candidate: removing a middle observation here would join
across a temporal break, so that condition is rejected for sampler correction.

For frame control, mean(-log r) is computed within each active segment. Both
losses then mean active segments in each window, and mean all num_windows
windows. Empty rows are inactive; empty windows contribute differentiable zero
without renormalizing the fixed window denominator. All diagnostics use the
segment-index exposure multiset unless their name explicitly says unique.
"""
    if logits.ndim != 4 or not logits.is_floating_point():
        raise ValueError("actual logits must be floating [B,T,H,W]")
    if not isinstance(num_windows, int) or num_windows <= 0 or logits.shape[0] != num_windows:
        raise ValueError("num_windows must equal the actual fixed batch size")
    target_regions = _regions(target_regions, logits.shape)
    background_regions = _regions(background_regions, logits.shape)
    neg = _integers(negative_indices, 2, "negative_indices")
    if len(neg) != len(target_regions) or (neg < -1).any() or (neg >= len(background_regions)).any():
        raise ValueError("negative indices must be [N,2] and refer to background ROIs or -1")
    for ti, row in enumerate(neg):
        target = target_regions[ti]
        for branch, ni in enumerate(row):
            if ni < 0:
                continue
            background = background_regions[ni]
            if background[1] != target[1] or not np.array_equal(background[4:6]-background[2:4], target[4:6]-target[2:4]):
                raise ValueError("background readout must use the same slot and ROI size as its target")
            if branch == 0 and background[0] != target[0]:
                raise ValueError("local background must lie in the target window")
    seg = _integers(segment_indices, label="segment_indices")
    if seg.ndim != 2 or seg.shape[1] == 0 or (seg < -1).any() or (seg >= len(target_regions)).any():
        raise ValueError("segment indices must be [S,L>0] and refer to target ROIs or -1")
    windows = _integers(segment_windows, label="segment_windows")
    if windows.shape != (len(seg),) or (windows < 0).any() or (windows >= num_windows).any():
        raise ValueError("segment_windows must map every segment into the fixed batch")
    valid_np = seg >= 0
    if (np.diff(valid_np.astype(np.int8), axis=1) > 0).any():
        raise ValueError("only suffix padding is allowed; split temporal breaks before the loss")
    for row, wi in zip(seg, windows):
        observations = row[row >= 0]
        if len(set(observations.tolist())) != len(observations):
            raise ValueError("an observation is repeated within one segment")
        if observations.size and (target_regions[observations, 0] != wi).any():
            raise ValueError("a segment crosses input windows")
        if observations.size and not (neg[observations] >= 0).any(axis=1).all():
            raise ValueError("referenced target has no valid background; split/rebuild this segment")

    qp = pool_logit_regions(logits, target_regions)
    background = pool_logit_regions(logits, background_regions)
    # Multiplication by zero preserves the actual detection graph when empty.
    zero = logits.to(qp.dtype).sum() * 0.0
    device = logits.device
    neg_t = torch.as_tensor(neg, device=device)
    valid_negative = neg_t >= 0
    safe_negative = torch.where(valid_negative, neg_t, len(background))
    candidate = torch.cat((background, zero.reshape(1)))[safe_negative]
    candidate = candidate.masked_fill(~valid_negative, -torch.inf)
    valid_observation = valid_negative.any(1)
    qm = torch.where(valid_observation, candidate.max(1).values, zero)
    valid = torch.as_tensor(valid_np, device=device)
    seg_t = torch.as_tensor(seg, device=device)
    safe_seg = torch.where(valid, seg_t, len(qp))
    a = torch.cat((qp, zero.reshape(1)))[safe_seg]
    b = torch.cat((qm, zero.reshape(1)))[safe_seg]
    core = temporal_miss_risk(a, b, valid, reduction="none")
    count = valid.sum(1)
    per_frame = -torch.where(valid, core.log_evidence, 0.).sum(1) / count.clamp(min=1)
    per_gap = core.per_segment
    window_t = torch.as_tensor(windows, device=device)
    gap_windows, frame_windows, window_counts = [], [], []
    for wi in range(num_windows):
        selected = (window_t == wi) & core.active
        gap_windows.append(_mean_or_zero(per_gap[selected], zero))
        frame_windows.append(_mean_or_zero(per_frame[selected], zero))
        window_counts.append(selected.sum())
    per_window_gap = torch.stack(gap_windows)
    per_window_frame = torch.stack(frame_windows)
    log_r = torch.nn.functional.logsigmoid(qp) + torch.nn.functional.logsigmoid(-qm)
    r = log_r.exp()
    exposure = seg_t[valid]
    positive = qp[exposure]
    negative = qm[exposure]
    diagnostics = {
        "segments_active": core.active.sum(),
        "observations_exposed": valid.sum(),
        "observations_unique_exposed": torch.tensor(len(np.unique(seg[valid_np])), device=device),
        "windows_active": (torch.stack(window_counts) > 0).sum(),
        "q_positive_mean": _mean_or_zero(positive, zero),
        "q_negative_mean": _mean_or_zero(negative, zero),
        "logit_margin_mean": _mean_or_zero(positive-negative, zero),
        "score_margin_mean": _mean_or_zero(positive.sigmoid()-negative.sigmoid(), zero),
        "target_score_mean": _mean_or_zero(positive.sigmoid(), zero),
        "background_score_mean": _mean_or_zero(negative.sigmoid(), zero),
        "log_r_mean": _mean_or_zero(log_r[exposure], zero),
        "r_mean": _mean_or_zero(r[exposure], zero),
        "local_missing_exposures": (neg_t[exposure, 0] < 0).sum(),
        "global_missing_exposures": (neg_t[exposure, 1] < 0).sum(),
    }
    return {
        "loss_gap": per_window_gap.mean(), "loss_frame": per_window_frame.mean(),
        "per_segment_gap": per_gap, "per_segment_frame": per_frame,
        "per_window_gap": per_window_gap, "per_window_frame": per_window_frame,
        "q_positive": qp, "q_negative": qm, "q_background_regions": background,
        "log_r": log_r, "r": r, "valid_observations": valid_observation,
        "segment_active": core.active, "segment_lengths": count,
        "window_segment_counts": torch.stack(window_counts),
        "segment_log_r": core.log_evidence, "exposure_indices": exposure,
        "diagnostics": diagnostics,
    }


def readout_gradient_diagnostics(result, loss_key="loss_gap", retain_graph=True):
    """Explicit optional derivative probe; never called implicitly by training.

Gradients are with respect to unique pooled observations after accumulation
from overlapping segments, not separately counted copies of the same tensor.
The caller separately checks native TPro/conv_out1/conv_out2 parameter gradients.
"""
    if loss_key not in ("loss_gap", "loss_frame"):
        raise ValueError("loss_key must select gap or frame")
    values = (result["q_positive"], result["q_negative"])
    if not all(v.requires_grad for v in values):
        raise ValueError("a gradient-enabled original-logit forward is required")
    derivatives = torch.autograd.grad(result[loss_key], values,
                                     retain_graph=retain_graph, allow_unused=True)
    gradients = [torch.zeros_like(v) if g is None else g for v, g in zip(values, derivatives)]
    return {"target_readout_gradient": gradients[0],
            "background_readout_gradient": gradients[1],
            "target_readout_gradient_l2": gradients[0].double().norm(),
            "background_readout_gradient_l2": gradients[1].double().norm(),
            "target_nonzero_gradients": torch.count_nonzero(gradients[0]),
            "background_nonzero_gradients": torch.count_nonzero(gradients[1])}
