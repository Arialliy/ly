"""Validate the actual compiled operator against an independent bilinear oracle.

The oracle is test-only. M1 always executes the pinned compiled RFR kernel.
"""
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from experiments.m1_recurrent_v1.upstream.rfr_cell import RFRCell, _load_dcn


def oracle(x, offset, mask, weight, bias):
    # Test uses 4 channels, 4 deformable groups, stride1/pad1/dilation1.
    b, c, h, w = x.shape
    yy, xx = torch.meshgrid(torch.arange(h, device=x.device, dtype=x.dtype),
                            torch.arange(w, device=x.device, dtype=x.dtype), indexing="ij")
    channels = []
    for ch in range(c):
        samples = []
        for ky in range(3):
            for kx in range(3):
                k = ky * 3 + kx
                sy = yy + ky - 1 + offset[:, ch * 18 + 2 * k]
                sx = xx + kx - 1 + offset[:, ch * 18 + 2 * k + 1]
                grid = torch.stack((2 * sx / (w - 1) - 1, 2 * sy / (h - 1) - 1), -1)
                value = F.grid_sample(x[:, ch:ch+1], grid, mode="bilinear",
                                      padding_mode="zeros", align_corners=True)[:, 0]
                samples.append(value * mask[:, ch * 9 + k])
        channels.append(torch.stack(samples, 1))
    values = torch.stack(channels, 1)
    return torch.einsum("bckhw,ock->bohw", values, weight.flatten(2)) + bias[None, :, None, None]


def main():
    torch.manual_seed(46)
    torch.set_num_threads(2)
    _load_dcn()
    from experiments.m1_recurrent_v1.upstream.compat.dcn.functions.modulated_deform_conv_func import ModulatedDeformConvFunction
    args = [torch.randn(1, 4, 4, 5, device="cuda", dtype=torch.float64),
            torch.randn(1, 72, 4, 5, device="cuda", dtype=torch.float64) * 1.7,
            torch.rand(1, 36, 4, 5, device="cuda", dtype=torch.float64),
            torch.randn(3, 4, 3, 3, device="cuda", dtype=torch.float64),
            torch.randn(3, device="cuda", dtype=torch.float64)]
    for x in args:
        x.requires_grad_()
    native = ModulatedDeformConvFunction.apply(*args, (1, 1), (1, 1), (1, 1), 1, 4, 128)
    expected = oracle(*args)
    upstream = torch.randn_like(native)
    grad_native = torch.autograd.grad(native, args, upstream)
    grad_expected = torch.autograd.grad(expected, args, upstream)
    names = ["input", "offset", "mask", "weight", "bias"]
    errors = {name: float((a-b).abs().max()) for name, a, b in zip(names, grad_native, grad_expected)}
    error = float((native - expected).abs().max())
    assert error < 1e-10, error
    assert all(v < 1e-9 for v in errors.values()), errors

    cell = RFRCell().cuda()
    previous = torch.randn(2, 16, 16, 20, device="cuda", requires_grad=True)
    current = torch.randn_like(previous, requires_grad=True)
    output = cell(previous, current)
    output.square().mean().backward()
    finite = bool(torch.isfinite(output).all() and torch.isfinite(previous.grad).all() and torch.isfinite(current.grad).all())
    assert output.shape == current.shape and finite
    grad_names = [name for name, p in cell.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    assert any(name.startswith("pda.") for name in grad_names)
    assert any(name.startswith("tsfm.") for name in grad_names)
    record = {"status": "PASS", "torch": torch.__version__, "cuda": torch.version.cuda,
              "device": torch.cuda.get_device_name(), "native_dcn_forward_oracle_max_abs": error,
              "native_dcn_backward_oracle_max_abs": errors, "cell_output_shape": list(output.shape),
              "cell_finite_forward_backward": finite, "cell_parameter_count": sum(p.numel() for p in cell.parameters()),
              "nonzero_gradient_parameter_names": grad_names,
              "oracle_note": "Test-only grid_sample bilinear zero extension; randomized offsets include samples inside, less than one pixel outside, and farther outside boundaries. Runtime uses RFR native DCN2."}
    path = Path(__file__).with_name("DCN_SMOKE.json")
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
