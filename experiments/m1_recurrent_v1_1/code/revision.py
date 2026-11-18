"""One registered change to M1: retain the PDA-aligned state around TSFM.

The frozen v1 implementation and all parameter names/shapes are retained.
Install after loading History-2048, before constructing/restoring the optimizer.
The caller restores the checkpoint RNG after this module's construction.
"""
from __future__ import annotations

import torch

from experiments.m1_recurrent_v1.upstream.rfr_cell import _load_dcn

_load_dcn()
from experiments.m1_recurrent_v1.upstream.compat.RFR_framework import TSFM


class AlignedResidualTSFM(TSFM):
    """Identical TSFM parameters; one additional, unscaled aligned-state sum."""

    def forward(self, aligned_feat, curr_feat):
        return aligned_feat + super().forward(aligned_feat, curr_feat)


def install_revision(model):
    """Replace TSFM without changing any model tensor or parameter ordering.

    M0=X0 and the history/current-only traversal remain owned by the unchanged
    WindowRecurrentAdapter. No added activation, gate, scale or parameter is
    introduced. The signed PDA output is intentionally not clipped.
    """
    cell = model.adapter.recurrent_cell
    original = cell.tsfm
    if type(original) is not TSFM:
        raise TypeError("Expected the original pinned TSFM, installed exactly once")
    channels = original.feat_fusion.out_channels
    if channels != 16:
        raise ValueError("M1 v1.1 retains the registered 16 state channels")
    before_parameters = list(model.named_parameters())
    parameter_contract = [(name, tuple(p.shape), p.dtype, p.device, p.requires_grad)
                          for name, p in before_parameters]
    before_state = {name: value.detach().clone()
                    for name, value in model.state_dict().items()}
    device_dtypes = {(p.device, p.dtype) for p in original.parameters()}
    if len(device_dtypes) != 1:
        raise ValueError("Expected one device/dtype for the original TSFM")
    device, dtype = next(iter(device_dtypes))
    replacement = AlignedResidualTSFM(num_feat=channels).to(device=device, dtype=dtype)
    replacement.load_state_dict(original.state_dict(), strict=True)
    replacement.train(original.training)
    original_parameters = dict(original.named_parameters())
    for name, parameter in replacement.named_parameters():
        parameter.requires_grad_(original_parameters[name].requires_grad)
    original_modules = dict(original.named_modules())
    for name, module in replacement.named_modules():
        module.training = original_modules[name].training
    cell.tsfm = replacement
    after_parameters = list(model.named_parameters())
    assert parameter_contract == [
        (name, tuple(p.shape), p.dtype, p.device, p.requires_grad)
        for name, p in after_parameters]
    after_state = model.state_dict()
    assert list(before_state) == list(after_state)
    assert all(torch.equal(before_state[name], value)
               for name, value in after_state.items())
    return {
        'revision': 'M_t = A_t + TSFM(A_t, X_t), A_t = PDA(M_(t-1), X_t), t>0',
        'first_frame': 'unchanged M0=X0',
        'state_channels': channels,
        'state_dict_keys_shapes_values_identical': True,
        'parameter_names_order_shapes_dtype_device_requires_grad_identical': True,
        'parameter_count': sum(p.numel() for _, p in after_parameters),
        'trainable_parameter_count': sum(p.numel() for _, p in after_parameters if p.requires_grad),
        'added_parameters': 0,
        'caller_must_restore_checkpoint_rng_after_construction': True,
    }
