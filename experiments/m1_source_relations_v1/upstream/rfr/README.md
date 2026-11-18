# Independently vendored RFR alignment

Source: [XinyiYing/RFR](https://github.com/XinyiYing/RFR), commit
`fd71ccdf17f056f80e0a5ea8726b87c82c27fb69`. `PROVENANCE.json` records verified
upstream and compatibility-source SHA-256 hashes. The files in `original/`
preserve the selected original source bytes. Runtime `pda.py` contains only
the official PDA class, with an identical class AST and local DCN import.
The RFR detector, TSFM, previous M1 modules, weights, binary extensions and
old build directories are not used by this package.

`dcn/` keeps the native CUDA sampling and autograd implementation. Changes
are ATen/c10 compatibility updates, local extension imports, and a build
script that permits compilation with an explicit architecture without a GPU.
There is no torchvision or grid-sampling substitute. CPU construction works;
the upstream native forward/backward CPU functions deliberately raise an
unsupported-operation error. Use float32/float64 CUDA features for this
upstream operator; it does not implement half/bfloat16 dispatch.

## Build

From the project root:

```bash
python3 experiments/m1_source_relations_v1/upstream/rfr/build_native.py
```

This invokes immutable image
`sha256:eb956873870d737f2b68dff0724b8f6f56f778829d119b0938d29b7abeaa2f92`
(Python 3.11, Torch 2.7, CUDA 12.8). It queries compute capability on GPUs
1-3, then compiles with no GPU devices exposed, networking disabled, two
CPU cores, eight GiB memory, and at most two compiler jobs. No dependencies
are installed or upgraded. All generated binaries, build intermediates,
`build.log`, and `BUILD_REPORT.json` stay in this directory.

Compilation and import validation are separate from a CUDA numerical test.
Do not infer detection accuracy or successful CUDA forward/backward solely
from a compiled binary. The constructor in `../../code/pda.py` is lazy and
returns a fresh `PDA(num_feat=16, deformable_groups=4)` each time. No weight
loading or native compilation occurs on import.

The official PDA uses factor-two pyramid upsampling; callers must pad input
height and width to multiples of four if their feature sizes require it,
then crop the result. This package preserves the original PDA implementation.

## Source notices

The pinned upstream checkout has no root LICENSE file. Existing embedded
notices are preserved verbatim, including the Microsoft/MIT notice in
`dcn/src/cuda/deform_psroi_pooling_cuda.cu`. This is a factual record of
the vendored source; no project-wide license is inferred or assigned here.
