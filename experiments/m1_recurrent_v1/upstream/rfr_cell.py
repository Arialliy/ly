"""Load the pinned RFR PDA -> TSFM unit with its real, locally built DCNv2.

The official source checkout is in RFR/; compat/ changes import paths and
removed PyTorch C++ APIs only. See PROVENANCE.json and COMPATIBILITY.md.
No alternate convolution or fake operator is used at runtime.
"""
from importlib import machinery, util
from pathlib import Path
import sys

import torch
from torch import nn


def _load_dcn():
    directory = Path(__file__).resolve().parent / "compat" / "dcn"
    candidates = [directory / ("DCN2" + suffix) for suffix in machinery.EXTENSION_SUFFIXES]
    binary = next((p for p in candidates if p.is_file()), None)
    if binary is None:
        raise RuntimeError(
            "The official RFR DCN2 extension is not built for this Python ABI. "
            "Build upstream/compat/dcn/setup.py with the recorded Docker image; "
            "see upstream/COMPATIBILITY.md. No substitute operator is enabled."
        )
    if "DCN2" in sys.modules:
        loaded = Path(sys.modules["DCN2"].__file__).resolve()
        if loaded != binary.resolve():
            raise RuntimeError(f"Another DCN2 extension is already loaded: {loaded}")
        return sys.modules["DCN2"]
    spec = util.spec_from_file_location("DCN2", binary)
    try:
        module = util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except (ImportError, OSError) as exc:
        raise RuntimeError(f"Cannot load pinned RFR DCN2 binary {binary}: {exc}") from exc
    sys.modules["DCN2"] = module
    return module


class RFRCell(nn.Module):
    """Official 16-channel PDA followed by TSFM; no hidden persistent state.

    Caller owns the first-frame rule and all window boundaries. For this M1
    experiment the caller uses M0=X0, then this cell at t>0 in both controls.
    """

    def __init__(self, channels=16):
        super().__init__()
        if channels != 16:
            raise ValueError("M1 v1 fixes RFR state channels at 16")
        _load_dcn()
        from .compat.RFR_framework import PDA, TSFM
        self.channels = channels
        self.pda = PDA(num_feat=channels, deformable_groups=4)
        self.tsfm = TSFM(num_feat=channels)

    def forward(self, previous, current):
        if previous.shape != current.shape or current.ndim != 4:
            raise ValueError("previous and current must have the same [B,16,H,W] shape")
        if current.shape[1] != self.channels:
            raise ValueError("RFRCell requires 16 channels")
        if current.shape[-2] % 4 or current.shape[-1] % 4:
            raise ValueError("Official PDA requires height and width divisible by 4")
        if previous.device != current.device or previous.dtype != current.dtype:
            raise ValueError("previous and current must share dtype and device")
        if not current.is_cuda:
            raise RuntimeError("The official RFR DCNv2 path requires CUDA")
        if current.dtype not in (torch.float32, torch.float64):
            raise TypeError("Official RFR DCNv2 supports float32/float64; do not enable AMP")
        aligned = self.pda(previous.contiguous(), current.contiguous())
        return self.tsfm(aligned, current)
