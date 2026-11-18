"""Temporal low-frequency conditioning, derived from the verified local reference.

This file does not load author NS-FPN code or weights.  The frozen local Haar
implementation remains authoritative.  Only the attention input changes: the
temporal mode concatenates previous/current/next LL mean/max descriptions.
The high bands always belong to the current frame.  Spatial padding, Gaussian
selection, sigma stabilization, and signed residual gate follow the reference.

Attention weights are [1, 2, 7, 7] for S and [1, 6, 7, 7] for T.  The six
channels are (previous mean, previous max, current mean, current max, next mean,
next max).  This is exactly a length-three temporal convolution expressed as a
2-D convolution on concatenated descriptions; no temporal high-band mixing.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch
from torch import Tensor, nn
import torch.nn.functional as F


REFERENCE_DIR = Path(__file__).resolve().parents[2] / "route_closeout_20261012"


def _load_reference(filename: str, module_name: str):
    """Load the explicitly identified, unchanged local reference source."""
    path = REFERENCE_DIR / filename
    existing = sys.modules.get(module_name)
    if existing is not None:
        if Path(existing.__file__).resolve() != path.resolve():
            raise ImportError(f"Conflicting reference module {module_name}")
        return existing
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_haar = _load_reference("haar_reference.py", "_tlfp_verified_haar_reference")
_adapter = _load_reference("framewise_adapter.py", "_tlfp_verified_framewise_adapter")
SpatialLFPReference = _haar.SpatialLFPReference
haar_forward = _haar.haar_forward
haar_inverse = _haar.haar_inverse
FramewiseFeatureAdapter = _adapter.FramewiseFeatureAdapter
FrozenDeepProPlusWithSpatialReference = _adapter.FrozenDeepProPlusWithSpatialReference
spatially_tiled_forward = _adapter.spatially_tiled_forward


class _ConditionedOperator(SpatialLFPReference):
    def __init__(self, channels: int, mode: str, gaussian_gate: float,
                 sigma_squared_floor: float):
        super().__init__(channels, gaussian_gate, sigma_squared_floor)
        if mode == "temporal":
            self.attention = nn.Conv2d(6, 1, 7, padding=3, bias=False)
            nn.init.zeros_(self.attention.weight)

    def apply_coefficients(self, low: Tensor, high: Tensor, original_hw,
                           guide: Tensor):
        attention = self.attention(guide).sigmoid()
        modulated = high * attention.unsqueeze(2)
        flat_high = modulated.flatten(1, 2)
        kernel = self.gaussian_kernel(flat_high).view(1, 1, 3, 3)
        blurred = F.conv2d(
            F.pad(flat_high, (1, 1, 1, 1), mode="replicate"),
            kernel.expand(3 * self.channels, 1, 3, 3),
            groups=3 * self.channels,
        ).reshape_as(modulated)
        filtered = torch.where(modulated.abs() < self.gaussian_gate, blurred, modulated)
        return haar_inverse(low, filtered, original_hw), attention


class TemporalLFPAdapter(nn.Module):
    """S/T control with exact zero gate, actual window endpoints and chunk halo.

    `from_spatial` explicitly maps the verified local adapter's three parameter
    tensors.  T's center kernel copies S and both neighboring kernels start at
    zero.  `current_only` replaces each new guide slot by the current guide;
    `bypass` returns the original features without evaluating the branch.

    Temporal chunks are scheduling only: each carries a one-frame guide halo
    from its own video and the same legal input window.  Replication occurs
    only at true window endpoints.  No cross-video or out-of-window reads.
    """

    def __init__(self, channels=32, mode="spatial", frame_chunk=8,
                 gaussian_gate=0.5, sigma_squared_floor=1e-6):
        super().__init__()
        if mode not in ("spatial", "temporal"):
            raise ValueError("Construction mode must be spatial or temporal")
        if not isinstance(frame_chunk, int) or isinstance(frame_chunk, bool) or frame_chunk < 1:
            raise ValueError("frame_chunk must be a positive integer")
        self.mode = mode
        self.channels = channels
        self.frame_chunk = frame_chunk
        self.scale = nn.Parameter(torch.zeros(()))
        self.operator = _ConditionedOperator(channels, mode, gaussian_gate,
                                             sigma_squared_floor)
        self.runtime_mode = "normal"
        self.collect_stats = False
        self.latest_stats = {}

    @classmethod
    def from_spatial(cls, spatial_init, mode="spatial", frame_chunk=8):
        required = {"scale", "operator.sigma", "operator.attention.weight"}
        if not hasattr(spatial_init, "operator") or not hasattr(spatial_init, "channels"):
            raise TypeError("Expected the verified local FramewiseFeatureAdapter interface")
        state = spatial_init.state_dict()
        if set(state) != required:
            raise ValueError(f"Unsupported spatial state keys: {sorted(state)}")
        op = spatial_init.operator
        channels = spatial_init.channels
        if op.channels != channels or tuple(state["operator.attention.weight"].shape) != (1, 2, 7, 7):
            raise ValueError("Expected the verified two-channel 7x7 spatial attention")
        if op.attention.bias is not None or op.attention.padding != (3, 3):
            raise ValueError("Expected unbiased spatial attention with padding 3")
        if state["scale"].ndim != 0 or state["operator.sigma"].ndim != 0:
            raise ValueError("The verified reference uses scalar scale and sigma")
        for name in ("gaussian_gate", "sigma_squared_floor"):
            if not hasattr(op, name):
                raise ValueError(f"Missing explicit reference configuration: {name}")
        result = cls(channels, mode, frame_chunk, op.gaussian_gate,
                     op.sigma_squared_floor).to(device=state["scale"].device,
                                                dtype=state["scale"].dtype)
        with torch.no_grad():
            result.scale.copy_(state["scale"])
            result.operator.sigma.copy_(state["operator.sigma"])
            if mode == "spatial":
                result.operator.attention.weight.copy_(state["operator.attention.weight"])
            else:
                result.operator.attention.weight.zero_()
                result.operator.attention.weight[:, 2:4].copy_(state["operator.attention.weight"])
        return result

    def set_inference_mode(self, mode="normal"):
        self._checked_mode(mode)
        self.runtime_mode = mode
        return self

    def _checked_mode(self, mode):
        if mode == self.mode:
            return "normal"
        if mode not in ("normal", "current_only", "bypass"):
            raise ValueError(f"Invalid mode {mode!r} for constructed {self.mode!r} adapter")
        return mode

    @staticmethod
    def _guide(low):
        return torch.cat((low.mean(dim=1, keepdim=True),
                          low.amax(dim=1, keepdim=True)), dim=1)

    def parameter_stats(self):
        weight = self.operator.attention.weight.detach()
        center = weight if self.mode == "spatial" else weight[:, 2:4]
        zero = weight.new_zeros(())
        previous = zero if self.mode == "spatial" else weight[:, :2].norm()
        following = zero if self.mode == "spatial" else weight[:, 4:].norm()
        return {
            "alpha": float(self.scale.detach()),
            "tanh_alpha": float(self.scale.detach().tanh()),
            "sigma": float(self.operator.sigma.detach()),
            "center_kernel_norm": float(center.norm()),
            "previous_kernel_norm": float(previous),
            "next_kernel_norm": float(following),
        }

    def forward(self, features: Tensor, mode=None):
        if features.ndim != 5 or features.shape[1] != self.channels:
            raise ValueError(f"Expected [B,{self.channels},T,H,W], got {tuple(features.shape)}")
        if not features.is_floating_point():
            raise TypeError("Features must have a floating-point dtype")
        b, c, t, h, w = features.shape
        if min(b, t, h, w) < 1:
            raise ValueError("Empty features are not supported")
        active = self._checked_mode(self.runtime_mode if mode is None else mode)
        if active == "bypass":
            if self.collect_stats:
                self.latest_stats = {**self.parameter_stats(), "mode": "bypass"}
            return features
        pieces = []
        stat_sums = None
        gate_count = 0
        for start in range(0, t, self.frame_chunk):
            end = min(t, start + self.frame_chunk)
            count = end - start
            current = features[:, :, start:end].permute(0, 2, 1, 3, 4).reshape(b * count, c, h, w)
            low, high, original_hw = haar_forward(current)
            guide = self._guide(low)
            if self.mode == "temporal":
                by_time = guide.reshape(b, count, 2, *guide.shape[-2:])
                if active == "current_only":
                    slot_guides = (by_time, by_time, by_time)
                else:
                    if start == 0:
                        previous_halo = by_time[:, :1]
                    else:
                        previous_low, _, _ = haar_forward(features[:, :, start - 1])
                        previous_halo = self._guide(previous_low).unsqueeze(1)
                    if end == t:
                        next_halo = by_time[:, -1:]
                    else:
                        next_low, _, _ = haar_forward(features[:, :, end])
                        next_halo = self._guide(next_low).unsqueeze(1)
                    slot_guides = (
                        torch.cat((previous_halo, by_time[:, :-1]), dim=1),
                        by_time,
                        torch.cat((by_time[:, 1:], next_halo), dim=1),
                    )
                guide = torch.cat(slot_guides, dim=2).reshape(b * count, 6, *guide.shape[-2:])
            transformed, attention = self.operator.apply_coefficients(low, high, original_hw, guide)
            correction = transformed - current
            updated = current + self.scale.tanh() * correction
            pieces.append(updated.reshape(b, count, c, h, w))
            if self.collect_stats:
                # Detached device-side accumulation; transfer only once per forward.
                with torch.no_grad():
                    values = torch.stack((current.square().sum(), correction.square().sum(),
                                          (attention < 0.01).sum().to(current.dtype),
                                          (attention > 0.99).sum().to(current.dtype)))
                    stat_sums = values if stat_sums is None else stat_sums + values
                    gate_count += attention.numel()
        if self.collect_stats:
            inp_sq, corr_sq, sat_low, sat_high = stat_sums.double().cpu().tolist()
            ratio = (corr_sq / max(inp_sq, 1e-30)) ** 0.5
            self.latest_stats = {
                **self.parameter_stats(), "mode": active,
                "gate_below_001_fraction": sat_low / gate_count,
                "gate_above_099_fraction": sat_high / gate_count,
                "gate_saturation_fraction": (sat_low + sat_high) / gate_count,
                "correction_to_input_norm": ratio,
                "applied_correction_to_input_norm": abs(float(self.scale.detach().tanh())) * ratio,
            }
        # Preserve the frozen original TPro's contiguous BCTHW input layout.
        return torch.cat(pieces, dim=1).permute(0, 2, 1, 3, 4).contiguous()
