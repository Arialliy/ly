"""Compare S3's differentiable decomposition with independent FP64 references."""
import numpy as np
import torch

from ..models.evidence_parts import decompose_evidence


def _reference(values, valid, weights):
    """Direct means on compact valid matrices and the adjoint of two-way centering."""
    batch, queries, channels, times, rows, cols = values.shape
    mean, temporal, velocity, residual = [np.zeros(w.shape, np.float64) for w in weights]
    input_gradient = np.zeros_like(values, dtype=np.float64)
    for b in range(batch):
        for n in range(queries):
            ys, xs = np.nonzero(valid[b, n])
            k = len(ys)
            if k == 0:
                continue
            # The last index enumerates only valid velocities for this query.
            x = values[b, n][..., ys, xs].astype(np.float64)
            grand = x.mean(axis=(1, 2))
            by_time = x.mean(axis=2) - grand[:, None]
            by_speed = x.mean(axis=1) - grand[:, None]
            mean[b, n] = grand
            temporal[b, n] = by_time
            velocity[b, n][..., ys, xs] = by_speed
            residual[b, n][..., ys, xs] = x - grand[:, None, None] - by_time[:, :, None] - by_speed[:, None, :]

            # The two centering matrices are symmetric, so the joint adjoint
            # projects the cotangent on both axes. Other terms use the mean adjoints.
            ct = np.eye(times) - np.ones((times, times)) / times
            cv = np.eye(k) - np.ones((k, k)) / k
            w0, wt, wv, wj = (w[b, n] for w in weights)
            dmean = w0[:, None, None] / (times * k)
            dtime = (wt @ ct)[:, :, None] / k
            dvelocity = (wv[..., ys, xs] @ cv)[:, None, :] / times
            djoint = ct @ wj[..., ys, xs] @ cv
            input_gradient[b, n][..., ys, xs] = dmean + dtime + dvelocity + djoint
    return (mean, temporal, velocity, residual), input_gradient


def _checkerboards(device):
    patterns = torch.tensor([[[1, 0], [0, 1]], [[0, 1], [1, 0]]], device=device, dtype=torch.float32)
    values = patterns.reshape(1, 2, 1, 2, 1, 2)
    p = decompose_evidence(values, torch.ones((1, 2, 1, 2), device=device, dtype=torch.bool))
    assert torch.equal(p.mean, torch.full_like(p.mean, 0.5))
    assert not torch.count_nonzero(p.time_effect) and not torch.count_nonzero(p.velocity_effect)
    torch.testing.assert_close(p.interaction, values - 0.5, rtol=0, atol=0)
    return float((p.interaction[:, 0] - p.interaction[:, 1]).abs().max())


def run_checks(device):
    rng = np.random.default_rng(46)
    shape = (2, 4, 3, 16, 3, 3)
    b, n, c, t, y, x = shape
    mask = rng.random((b, n, y, x)) >= 0.4
    mask[:, 0] = False
    mask[:, 1] = False
    mask[:, 1, 1, 2] = True
    mask[:, 2] = True
    valid = torch.tensor(mask, device=device)
    scores = rng.normal(size=shape).astype(np.float32)
    scores = np.where(mask[:, :, None, None], scores, np.nan)
    weights = [rng.normal(size=s).astype(np.float32).astype(np.float64) for s in
               ((b, n, c), (b, n, c, t), (b, n, c, y, x), shape)]
    expected, expected_grad = _reference(scores, mask, weights)
    values = torch.tensor(scores, device=device, requires_grad=True)
    parts = decompose_evidence(values, valid)
    outputs = (parts.mean, parts.time_effect, parts.velocity_effect, parts.interaction)
    errors = {}
    for key, result, target in zip(("mean", "time", "velocity", "interaction"), outputs, expected):
        assert result.dtype == torch.float32
        actual = result.detach().cpu().numpy()
        np.testing.assert_allclose(actual, target, atol=1e-5, rtol=1e-4)
        errors[key] = float(np.max(np.abs(actual - target)))
    output_gradient = [torch.tensor(w, dtype=torch.float32, device=device) for w in weights]
    gradient, = torch.autograd.grad(outputs, values, grad_outputs=output_gradient)
    grad = gradient.detach().cpu().numpy()
    np.testing.assert_allclose(grad, expected_grad, atol=1e-5, rtol=1e-4)
    bad = (~valid[:, :, None, None]).expand_as(values)
    assert torch.count_nonzero(gradient.masked_select(bad)).item() == 0
    assert all(torch.count_nonzero(v[:, 0]).item() == 0 for v in outputs)
    assert torch.count_nonzero(parts.velocity_effect[:, 1]).item() == 0
    assert torch.count_nonzero(parts.interaction[:, 1]).item() == 0
    assert torch.count_nonzero(parts.time_effect[:, 1]).item() > 0

    interaction = parts.interaction.detach()
    row_marginal = interaction.sum(dim=(-2, -1)) / valid.sum((-2, -1)).clamp_min(1)[:, :, None, None]
    col_marginal = interaction.mean(dim=3)
    row_error = float(row_marginal.abs().max())
    col_error = float(col_marginal.abs().max())
    assert max(row_error, col_error) <= 1e-5
    reconstruction = (parts.mean[..., None, None, None] + parts.time_effect[..., None, None] +
                      parts.velocity_effect.unsqueeze(3) + parts.interaction)
    difference = (reconstruction.detach() - values.detach()).masked_select(~bad)
    reconstruction_error = float(difference.abs().max())
    assert reconstruction_error <= 1e-5

    lower = values.detach().to(torch.float16).requires_grad_()
    low = decompose_evidence(lower, valid)
    fp32 = decompose_evidence(lower.detach().float(), valid)
    for key in ("mean", "time_effect", "velocity_effect", "interaction"):
        assert getattr(low, key).dtype == torch.float32 and torch.equal(getattr(low, key), getattr(fp32, key))
    low_grad, = torch.autograd.grad(low.mean.sum(), lower)
    assert torch.isfinite(low_grad).all() and torch.count_nonzero(low_grad.masked_select(~bad)).item()
    assert not torch.count_nonzero(low_grad.masked_select(bad)).item()

    return {"status": "PASS", "reference": "compact FP64 means and analytical centered-matrix adjoint",
            "max_abs_error_by_component": errors, "input_gradient_max_abs_error": float(abs(grad - expected_grad).max()),
            "reconstruction_max_abs_error": reconstruction_error, "row_margin_max_abs": row_error,
            "column_margin_max_abs": col_error, "velocity_counts": mask.sum(axis=(-2, -1)).tolist(),
            "invalid_gradient_nonzeros": 0, "empty_support_all_zero": True, "singleton_interaction_exact_zero": True,
            "fp16_promoted_and_differentiable": True, "checkerboard_joint_difference": _checkerboards(device)}
