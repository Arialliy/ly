"""Build vendored native RFR DCN for the current Python/Torch ABI.

Derived from original/dcn/setup.py. An explicit architecture permits nvcc
cross-compilation with no GPU devices exposed to the build container.
"""
import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

ROOT = Path(__file__).resolve().parent
if CUDA_HOME is None:
    raise RuntimeError("The pinned CUDA toolkit is required to build native RFR DCN.")
if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
    raise RuntimeError("Set TORCH_CUDA_ARCH_LIST explicitly; build_native.py detects GPUs 1-3.")

sources = sorted((ROOT / "src").glob("*.cpp"))
sources += sorted((ROOT / "src" / "cpu").glob("*.cpp"))
sources += sorted((ROOT / "src" / "cuda").glob("*.cu"))

setup(
    name="deeppro-m1-source-relations-rfr-dcn",
    version="0.1",
    author="xvjiarui (upstream DCNv2)",
    url="https://github.com/XinyiYing/RFR",
    description="Independent native RFR alignment dependency; no pretrained weights",
    ext_modules=[CUDAExtension(
        "_rfr_dcn",
        sources=[str(path) for path in sources],
        include_dirs=[str(ROOT / "src")],
        define_macros=[("WITH_CUDA", None)],
        extra_compile_args={
            "cxx": ["-O2", "-g0"],
            "nvcc": [
                "-O2", "-DCUDA_HAS_FP16=1", "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__", "-D__CUDA_NO_HALF2_OPERATORS__",
            ],
        },
    )],
    cmdclass={"build_ext": BuildExtension},
)
