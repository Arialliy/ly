"""Passive detached background-loss diagnostics. Never used to compute gradients."""
import math
import torch
import torch.nn.functional as F
from losses import background_top1pct_mean

@torch.no_grad()
def background_diagnostics(logits, masks):
    means, tops, counts, selected, fractions = [], [], [], [], []
    for x, mask in zip(logits.detach(), masks):
        values = F.softplus(x[mask])
        n = values.numel()
        k = max(1, math.ceil(.01 * n)) if n else 0
        counts.append(n); selected.append(k); fractions.append(k / n if n else None)
        means.append(values.mean() if n else x.sum() * 0.0)
        tops.append(background_top1pct_mean(values) if n else x.sum() * 0.0)
    return {
        'background_full_mean': float(torch.stack(means).mean()),
        'background_top1pct_mean': float(torch.stack(tops).mean()),
        'background_full_mean_per_window': [float(x) for x in means],
        'background_top1pct_mean_per_window': [float(x) for x in tops],
        'background_N_per_window': counts,
        'background_k_per_window': selected,
        'background_selected_fraction_per_window': fractions,
        'background_selected_fraction_pooled': sum(selected) / sum(counts) if sum(counts) else None,
    }
