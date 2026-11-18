# RFR source and native DCN compatibility

RFR is pinned to commit `fd71ccdf17f056f80e0a5ea8726b87c82c27fb69` from
<https://github.com/XinyiYing/RFR>. `RFR/` is the unmodified checkout, including
source copyright notices. `PROVENANCE.json` records every tracked file SHA-256.
The checkout has no root LICENSE; no project-wide license has been inferred.
Some DCN files contain Microsoft MIT license notices, retained verbatim.

`compat/` is a separate copy of the needed framework, utility and DCN files.
`COMPATIBILITY.patch` records all changes from these upstream source files:

- Python imports use a private relative package, avoiding the project's own
  `model` or `models` names. The unused whole-detector model wildcard is removed.
- Removed THC header names are replaced by current ATen/c10 headers.
- `THCudaCheck` becomes `C10_CUDA_CHECK`; `THCCeilDiv` becomes `at::ceil_div`.
- Deprecated tensor accessors become `is_cuda()`, `scalar_type()` and
  `data_ptr<T>()`. All sampling loops, coordinate order, offset scaling,
  initialization and PDA/TSFM operations are retained.

This is the native DCNv2 source shipped with RFR, compiled locally. It is not
replaced with torchvision, grid_sample, regular convolution or another DCN.
The independent grid_sample implementation in `smoke_dcn.py` is only a test oracle.

## Recorded runtime

The old project image `sha256:eb956873...` is absent on this host. No dependency
was upgraded. Build and smoke test use the already installed image
`sha256:8470a68886ff2f480d76858d930fb72d0e680e69276fc1d068fe8b6625b4cb9f`
(`nvcr.io/nvidia/pytorch:24.01-py3`): Python 3.10.12,
PyTorch `2.2.0a0+81ea7a4`, CUDA 12.3, RTX 3090 compute capability 8.6.
This runtime change from the historical image must be recorded in paired M1
experiments; both controls use the same image.

Build command inside that image, with a free authorized GPU exposed:

```bash
cd /workspace/experiments/m1_recurrent_v1/upstream/compat/dcn
MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.6 python setup.py build_ext --inplace
```

This intentionally uses `build_ext --inplace`, without installing packages into
the immutable image. The official setup compiles CUDA only if
`torch.cuda.is_available()`. The build used 2 CPUs and a 12 GiB host-memory bound.
The built Python 3.10 extension and its SHA-256 are recorded in provenance.
`build_dcn.log` preserves the actual compilation output.

For the real operator smoke test, set the project root on `PYTHONPATH` and run:

```bash
python -m experiments.m1_recurrent_v1.upstream.smoke_dcn
```

`DCN_SMOKE.json` contains observed values. The native double-precision operator
matches an independent bilinear oracle, including fractional sampling just
outside and beyond spatial boundaries: maximum forward error 1.56e-15 and
maximum first-order gradient error 2.45e-15. Real PDA+TSFM forward/backward is
finite at `[2,16,16,20]`. These are operator checks, not target-recovery results.

## Adapter contract

```python
from experiments.m1_recurrent_v1.upstream.rfr_cell import RFRCell
cell = RFRCell(channels=16)
next_state = cell(previous, current)  # TSFM(PDA(previous,current),current)
```

The wrapper owns no hidden state. The caller initializes and resets window state.
It uses `M0=X0` to honor the user's M1 protocol. This differs from the full RFR
forward methods, which also apply TSFM at the first frame. Both M1 controls must
share the user-specified first-frame rule.

Inputs must be matching `[B,16,H,W]` CUDA float32/float64 tensors. The wrapper makes
only a contiguous copy if needed. Half precision and AMP are unsupported by the
official CUDA dispatch. Height and width must be divisible by 4 because PDA uses
two stride-2 reductions and scale-factor-2 bilinear interpolation. There is no
implicit padding, resizing or sampler replacement. DCN uses 4 deformable groups,
3x3 kernels, stride 1, dilation 1 and padding 1. Its offset predictor starts at
zero, producing zero offsets and modulation masks 0.5.

At 16 channels, TSFM includes spatial 3x3 attention and frequency attention with
adaptive average pooling to 56x56, fixed top16 DCT frequencies, and a 16→1→16
channel MLP. Thus spatial cropping changes the frequency attention for the whole
crop. Spatially tiled inference is not mathematically equivalent to whole-frame
inference, even with a local halo. Temporal chunking is a separate question and
must preserve the within-window recursive state.
