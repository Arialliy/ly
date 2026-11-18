"""Background-conditioned correction inside the released DeepPro-Plus TPro.

Only ``attach_bc_tpro`` mutates the caller's model.  Original TPro parameter
names and the detector output contract are retained.  The reference predictor
has an auxiliary-only gradient path; inference never receives annotations.
"""
from __future__ import annotations

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F


MODEL_CONFIG = {
    "version": 1,
    "channels": 32,
    "rank": 4,
    "length": 40,
    "new_module_seed": 20261011,
    "reference_offset": 40,
    "reference_directions": [[-1, -1], [-1, 0], [-1, 1], [0, -1],
                             [0, 1], [1, -1], [1, 0], [1, 1]],
    "stem_spatial_radius": 9,
    "stem_temporal_radius": 8,
    "raw_center_exclusion_radius": 23,
    "recommended_inference_halo": 64,
    "reference_temporal_kernel": 3,
    "association": "shared_LayerNorm32_linear_QK_tanh_QKt_over_T_sqrt_r",
    "value": "(F-stopgrad(B))/sqrt(T)",
    "gamma_initial": 0.0,
    "reference_lambda": 1.0,
    "reference_reduction": "channel_pixel_mean_per_window_then_equal_window_mean",
    "zero_context_correction": "zero",
}


def _rms(x):
    return float(torch.linalg.vector_norm(x.detach()) / math.sqrt(x.numel()))


class BC_TPro(nn.Module):
    def __init__(self, original, variant="bc_tpro", config=None):
        super().__init__()
        if variant not in {"bc_tpro", "single_dynamic"}:
            raise ValueError(f"Unknown variant {variant!r}")
        self.variant = variant
        self.config = copy.deepcopy(MODEL_CONFIG)
        if config is not None:
            unknown = set(config) - set(self.config)
            if unknown:
                raise ValueError(f"Unknown model configuration fields: {sorted(unknown)}")
            self.config.update(copy.deepcopy(config))
        cfg = self.config
        if cfg["rank"] != 4 or cfg["channels"] != 32 or cfg["length"] != 40:
            raise ValueError("This frozen pilot requires 32 channels, rank 4, and length 40")
        if cfg["reference_temporal_kernel"] != 3:
            raise ValueError("This pilot fixes the reference temporal kernel to 3")
        if cfg["reference_offset"] <= cfg["raw_center_exclusion_radius"] + cfg["stem_spatial_radius"]:
            raise ValueError("Reference samples could read the excluded raw center region")
        self.seqlen = original.seqlen
        self.out_len = original.out_len
        self.num_head = original.num_head
        self.hidden_dim = original.hidden_dim
        if (self.seqlen, self.out_len, self.num_head * self.hidden_dim) != (40, 40, 32):
            raise ValueError("Expected the released 40-to-40, 32-channel DeepPro-Plus TPro")

        # Preserve every original state_dict name beneath model.TPro.
        self.QK_heads = original.QK_heads
        self.relu = original.relu
        self.norm1 = original.norm1
        self.conv = original.conv
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(cfg["new_module_seed"])
            self.reference = nn.Sequential(
                nn.Conv3d(32, 32, kernel_size=(3, 1, 1), padding=(1, 0, 0), groups=32),
                nn.SiLU(),
                nn.Conv3d(32, 32, kernel_size=1),
            )
            self.association_norm = nn.LayerNorm(32)
            self.query = nn.Linear(32, 4, bias=False)
            self.key = nn.Linear(32, 4, bias=False)
            self.gamma = nn.Parameter(torch.tensor(float(cfg["gamma_initial"])))
        self.last_stats = {}
        self._reference_cache = None

    def reference_context(self, features):
        """Eight shifted feature samples; no padding, reflection, or center path.

        features: [B,C,T,H,W]. The source feature center must be at least nine
        pixels from each spatial crop boundary, where its stem RF is complete.
        Geometry is independent of image content, annotations, and detections.
        """
        if features.ndim != 5:
            raise ValueError("reference_context expects [B,C,T,H,W]")
        source = features.detach()
        height, width = source.shape[-2:]
        margin = self.config["stem_spatial_radius"]
        offset = self.config["reference_offset"]
        summed = torch.zeros_like(source)
        count = source.new_zeros((1, 1, 1, height, width))
        for dy_unit, dx_unit in self.config["reference_directions"]:
            dy, dx = dy_unit * offset, dx_unit * offset
            # destination y reads source y+dy. Source RF stays inside the crop.
            y0, y1 = max(0, margin - dy), min(height, height - margin - dy)
            x0, x1 = max(0, margin - dx), min(width, width - margin - dx)
            if y0 >= y1 or x0 >= x1:
                continue
            summed[..., y0:y1, x0:x1] += source[..., y0+dy:y1+dy, x0+dx:x1+dx]
            count[..., y0:y1, x0:x1] += 1
        return summed / count.clamp_min(1), count

    def factors(self, z):
        """Shared generator for BOTH branches, z=[B,H,W,T,32]."""
        normalized = self.association_norm(z)
        return self.query(normalized).tanh(), self.key(normalized).tanh()

    def association_apply(self, z, value):
        """Apply Q(K^T V)/(T sqrt(r)); never allocate a T-by-T matrix."""
        query, key = self.factors(z)
        compressed_value = torch.einsum("...tr,...tc->...rc", key, value)
        output = torch.einsum("...tr,...rc->...tc", query, compressed_value)
        return output / (z.shape[-2] * math.sqrt(self.config["rank"]))

    def correction(self, observed, background, value=None):
        """Observed/background [B,H,W,T,C]; optional nonzero value aids testing."""
        background = background.detach()
        if value is None:
            value = (observed - background) / math.sqrt(self.seqlen)
        correction = self.association_apply(observed, value)
        if self.variant == "bc_tpro":
            correction = correction - self.association_apply(background, value)
        return correction

    def forward(self, input):
        if input.ndim != 5 or input.shape[-2:] != (32, 40):
            raise ValueError(f"Expected [B,H,W,32,40], received {tuple(input.shape)}")
        bs, height, width, channels, length = input.shape
        features = input.permute(0, 3, 4, 1, 2)
        context, count = self.reference_context(features)
        background = self.reference(context)

        # Exact operation order of original TPro up to its first normalization.
        value = input / (self.seqlen ** 0.5)
        value = value.view(bs, height, width, self.num_head, self.hidden_dim, length)
        qkv = input.new_zeros((bs, height, width, self.num_head, self.hidden_dim, self.out_len))
        for i in range(self.num_head):
            qkv[:, :, :, i, :, :] = self.QK_heads[i](value[:, :, :, i, :, :])
        qkv = qkv.view(bs, height, width, -1, self.out_len).permute(0, 3, 4, 1, 2)

        observed_tc = input.transpose(-1, -2)
        background_tc = background.detach().permute(0, 3, 4, 2, 1)
        delta = self.correction(observed_tc, background_tc).permute(0, 4, 3, 1, 2)
        delta = delta * (count > 0).to(delta.dtype)
        corrected = qkv + self.gamma * delta
        output = self.conv(self.relu(self.norm1(corrected)))

        if self.training:
            self._reference_cache = (background, features.detach(), count.detach())
        else:
            self._reference_cache = None
        with torch.no_grad():
            histogram = torch.bincount(count.to(torch.int64).flatten(), minlength=9).cpu().tolist()
            gamma_value = float(self.gamma.detach())
            delta_rms = _rms(delta)
            self.last_stats = {
                "variant": self.variant,
                "gamma": gamma_value,
                "original_pre_norm_rms": _rms(qkv),
                "correction_rms": delta_rms,
                "weighted_correction_rms": abs(gamma_value) * delta_rms,
                "reference_prediction_rms": _rms(background),
                "reference_target_rms": _rms(features),
                "ring_count_histogram_per_window": {str(i): int(n) for i, n in enumerate(histogram)},
                "ring_available_fraction": float((count > 0).float().mean()),
                "ring_complete_fraction": float((count == 8).float().mean()),
            }
        return output

    def reference_valid_mask(self, background_masks, count):
        """Erode only the auxiliary background mask by the original stem RF.

        Input is the existing detection background mask [B,T,H,W], whose False
        sites already include every annotated box and the original core border.
        Temporal/crop boundaries are invalid auxiliary targets. Separable max
        pooling equals rectangular 17x19x19 dilation without its huge cost.
        """
        if background_masks.ndim != 4:
            raise ValueError("background_masks must be [B,T,H,W]")
        invalid = (~background_masks.bool()).float().unsqueeze(1)
        rt = self.config["stem_temporal_radius"]
        rs = self.config["stem_spatial_radius"]
        invalid = F.max_pool3d(F.pad(invalid, (0, 0, 0, 0, rt, rt), value=1),
                               (2*rt+1, 1, 1), stride=1)
        invalid = F.max_pool3d(F.pad(invalid, (0, 0, rs, rs, 0, 0), value=1),
                               (1, 2*rs+1, 1), stride=1)
        invalid = F.max_pool3d(F.pad(invalid, (rs, rs, 0, 0, 0, 0), value=1),
                               (1, 1, 2*rs+1), stride=1)
        return (invalid == 0) & (count > 0)

    def reference_loss(self, background_masks):
        if self._reference_cache is None:
            raise RuntimeError("reference_loss requires one fresh train-mode forward")
        prediction, target, count = self._reference_cache
        if tuple(background_masks.shape) != tuple(target.shape[:1] + target.shape[2:]):
            raise ValueError("Reference mask does not match the preceding forward")
        background_masks = background_masks.to(device=target.device)
        valid = self.reference_valid_mask(background_masks, count)
        per_position = (prediction - target).abs().mean(dim=1, keepdim=True)
        numerator = (per_position * valid).sum(dim=(1, 2, 3, 4))
        denominator = valid.sum(dim=(1, 2, 3, 4))
        # Empty windows contribute differentiable zero, preserving the outer
        # equal-window reduction rather than reweighting non-empty windows.
        per_window = numerator / denominator.clamp_min(1)
        loss = per_window.mean()
        target_abs = (target.abs().mean(dim=1, keepdim=True) * valid).sum(dim=(1, 2, 3, 4))
        target_abs = target_abs / denominator.clamp_min(1)
        relative_l1 = per_window.detach() / target_abs.clamp_min(1e-6)
        stats = {
            "reference_loss": float(loss.detach()),
            "reference_loss_per_window": per_window.detach().cpu().tolist(),
            "reference_valid_positions_per_window": denominator.detach().cpu().tolist(),
            "reference_empty_windows": int((denominator == 0).sum()),
            "reference_valid_fraction": float(valid.float().mean()),
            "reference_target_abs_mean_per_window": target_abs.detach().cpu().tolist(),
            "reference_relative_l1_per_window": relative_l1.cpu().tolist(),
            "reference_relative_l1": float(relative_l1.mean()),
        }
        self.last_stats.update(stats)
        self._reference_cache = None
        return loss, stats


def attach_bc_tpro(model, variant="bc_tpro", config=None):
    """Replace the single frozen-architecture TPro AFTER original weight load."""
    if not hasattr(model, "TPro") or any(hasattr(model, key) for key in ("TPro1", "TPro2", "TPro3")):
        raise ValueError("This adapter targets the single-TPro DeepPro-Plus architecture")
    if isinstance(model.TPro, BC_TPro):
        raise ValueError("BC_TPro is already attached")
    old = model.TPro
    bn_modes = [(module, module.training) for module in old.modules()
                if isinstance(module, nn.modules.batchnorm._BatchNorm)]
    device, dtype = next(old.parameters()).device, next(old.parameters()).dtype
    replacement = BC_TPro(old, variant=variant, config=config).to(device=device, dtype=dtype)
    replacement.train(old.training)
    # train(mode) would also alter original frozen BN flags; restore them.
    for module, mode in bn_modes:
        module.train(mode)
    model.TPro = replacement
    return model
