"""Fixed-coordinate wrapper around unmodified official ConFIG_update.

Only the gradient-combination rule is adapted. Existing weighted losses and
Adam remain owned by the caller. A successful candidate is matched to the
ordinary grouped-sum L2 norm; numerical fallback uses that sum without skipping
the optimizer step. Parameters unused by every loss retain grad=None.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import torch

E = Path(__file__).resolve().parents[1]
OFFICIAL = E / "resources/config_official"
manifest = json.loads((OFFICIAL / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
assert manifest["commit"] == "94862437f451f175673bce9c85f3e14bd9182c21"
for item in manifest["files"]:
    if item["path"].startswith("conflictfree/"):
        assert hashlib.sha256((OFFICIAL/item["path"]).read_bytes()).hexdigest() == item["sha256"]
sys.path.insert(0, str(OFFICIAL))
import conflictfree.grad_operator as _official
assert Path(_official.__file__).resolve() == (OFFICIAL/"conflictfree/grad_operator.py").resolve()

GROUPS = ("fit", "keep", "track")
RANK_RELATIVE_MIN = 1e-6
MIN_POSITIVE_COSINE = 1e-7
EQUAL_PROJECTION_RELATIVE_TOLERANCE = 1e-4
NORM_RELATIVE_TOLERANCE = 1e-6


def _layout(named_parameters):
    named = tuple(named_parameters)
    if not named or len({name for name, _ in named}) != len(named):
        raise ValueError("a nonempty unique fixed named-parameter list is required")
    parameters = [p for _, p in named]
    if len({id(p) for p in parameters}) != len(parameters):
        raise ValueError("a parameter appears twice in the fixed coordinate list")
    device = parameters[0].device
    if any(not p.requires_grad or p.dtype != torch.float32 or p.device != device for p in parameters):
        raise ValueError("fixed coordinates must be trainable FP32 parameters on one device")
    return named, parameters, device


def combine_gradients(named_parameters, grouped_grads, mode="config"):
    """Return (ordered per-parameter gradient-or-None, JSON-safe diagnostics).

grouped_grads maps the exact keys fit/keep/track to autograd.grad sequences in
the SAME fixed parameter order. Coefficients have already been included in
the three scalar losses. No loss-dependent or tunable direction weight is used.
The original full scalar gradient can be recorded separately by the caller;
the norm control here uses ((g_fit + g_keep) + g_track), as preregistered.
"""
    named, parameters, device = _layout(named_parameters)
    if mode not in ("sum", "config") or set(grouped_grads) != set(GROUPS):
        raise ValueError("expected mode sum/config and exactly fit/keep/track gradients")
    sequences = [tuple(grouped_grads[key]) for key in GROUPS]
    if any(len(values) != len(parameters) for values in sequences):
        raise ValueError("gradient sequence length does not match fixed coordinates")
    active = [any(values[i] is not None for values in sequences) for i in range(len(parameters))]
    rows = []
    for values in sequences:
        pieces = []
        for p, g in zip(parameters, values):
            if g is None:
                pieces.append(torch.zeros_like(p).reshape(-1))
            else:
                if g.shape != p.shape or g.dtype != p.dtype or g.device != device or g.is_sparse:
                    raise ValueError("gradient does not match its fixed parameter coordinates")
                if not bool(torch.isfinite(g).all()):
                    raise FloatingPointError("nonfinite original loss gradient; no numerical fallback")
                pieces.append(g.detach().reshape(-1))
        rows.append(torch.cat(pieces))
    matrix = torch.stack(rows)
    coordinate_active = torch.cat([torch.full((p.numel(),), used, dtype=torch.bool, device=device)
                                   for p, used in zip(parameters, active)])
    ordinary = (matrix[0] + matrix[1]) + matrix[2]
    if not bool(torch.isfinite(ordinary).all()):
        raise FloatingPointError("nonfinite grouped sum; no finite baseline for fallback")
    ordinary_norm = ordinary.double().norm()
    row_norms = matrix.double().norm(dim=1)
    diagnostics = {
        "requested_mode": mode, "used_mode": "sum", "official_called": False,
        "fallback_reason": None, "groups": list(GROUPS),
        "parameter_names": [name for name, _ in named],
        "parameter_shapes": [list(p.shape) for p in parameters],
        "parameter_active_any_loss": active,
        "parameter_used_by_loss": {key: [g is not None for g in seq] for key, seq in zip(GROUPS, sequences)},
        "fixed_parameter_count": len(parameters), "fixed_coordinate_count": ordinary.numel(),
        "active_coordinate_count": int(coordinate_active.sum()),
        "group_gradient_l2": row_norms.tolist(), "group_sum_l2": float(ordinary_norm),
        "solver": "official ConFIG_update, FP32, EqualWeight, ProjectionLength, use_least_square=True, losses=None",
        "solver_device": str(device), "norm_control": "match grouped ordinary sum L2 before Adam",
        "optimizer_step_is_not_skipped": True,
    }
    chosen = ordinary
    if mode == "config":
        reason = None
        if not any(active):
            reason = "all_losses_unused"
        elif ordinary_norm == 0:
            reason = "zero_group_sum"
        elif bool((row_norms == 0).any()):
            reason = "zero_loss_gradient"
        else:
            compact = matrix[:, coordinate_active]
            # Numerical diagnosis uses float64 but never replaces the official
            # FP32 solver or its original EqualWeight/ProjectionLength objects.
            units64 = compact.double() / row_norms[:, None]
            singular = torch.linalg.svdvals(units64)
            ratio = float(singular[-1]/singular[0]) if len(singular) == 3 else 0.
            diagnostics["normalized_singular_values"] = singular.tolist()
            diagnostics["normalized_min_max_singular_ratio"] = ratio
            diagnostics["group_cosines"] = (units64@units64.T).tolist()
            if ratio <= RANK_RELATIVE_MIN:
                reason = "rank_deficient_or_ill_conditioned"
            else:
                try:
                    diagnostics["official_called"] = True
                    candidate = _official.ConFIG_update(compact, use_least_square=True, losses=None)
                except RuntimeError as exc:
                    reason = "official_solver_exception"
                    diagnostics["official_exception"] = repr(exc)
                if reason is None:
                    if not bool(torch.isfinite(candidate).all()):
                        reason = "nonfinite_candidate"
                    else:
                        candidate_norm = candidate.double().norm()
                        diagnostics["official_candidate_l2"] = float(candidate_norm)
                        if candidate_norm == 0:
                            reason = "zero_candidate"
                        else:
                            cosines = units64@(candidate.double()/candidate_norm)
                            diagnostics["candidate_group_cosines"] = cosines.tolist()
                            mean_projection = cosines.mean()
                            if bool((cosines <= MIN_POSITIVE_COSINE).any()):
                                reason = "nonpositive_or_numerically_weak_projection"
                            else:
                                spread = float((cosines-mean_projection).abs().max()/mean_projection.abs())
                                diagnostics["equal_projection_relative_spread"] = spread
                                if spread > EQUAL_PROJECTION_RELATIVE_TOLERANCE:
                                    reason = "unreliable_equal_projection_solution"
                            if reason is None:
                                scaled = (candidate.double()*(ordinary_norm/candidate_norm)).to(torch.float32)
                                norm_error = float((scaled.double().norm()-ordinary_norm).abs()/ordinary_norm)
                                diagnostics["norm_match_relative_error"] = norm_error
                                if not bool(torch.isfinite(scaled).all()) or norm_error > NORM_RELATIVE_TOLERANCE:
                                    reason = "unreliable_norm_match"
                                else:
                                    chosen = torch.zeros_like(ordinary)
                                    chosen[coordinate_active] = scaled
                                    diagnostics["used_mode"] = "config"
        diagnostics["fallback_reason"] = reason
    diagnostics["chosen_gradient_l2"] = float(chosen.double().norm())
    output = []
    cursor = 0
    for p, used in zip(parameters, active):
        part = chosen[cursor:cursor+p.numel()].reshape_as(p)
        output.append(part.detach().clone() if used else None)
        cursor += p.numel()
    return tuple(output), diagnostics


def apply_gradients(named_parameters, gradients):
    """Assign only the fixed trainable coordinates; preserve genuine None."""
    named, parameters, _ = _layout(named_parameters)
    gradients = tuple(gradients)
    if len(gradients) != len(parameters):
        raise ValueError("gradient assignment length mismatch")
    with torch.no_grad():
        for p, g in zip(parameters, gradients):
            if g is None:
                p.grad = None
            else:
                if g.shape != p.shape or g.dtype != p.dtype or g.device != p.device or not bool(torch.isfinite(g).all()):
                    raise ValueError("invalid gradient assignment")
                p.grad = g.detach().clone()
