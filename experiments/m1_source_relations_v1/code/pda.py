"""Lazy constructor for this experiment's independently vendored RFR PDA.

Importing this module neither imports Torch/native extensions nor modifies
sys.path. Calling build_pda imports the local native operator and initializes
a new 16-channel PDA. No checkpoint or previous M1 implementation is loaded.
"""
from importlib import import_module, util
from pathlib import Path
import sys


def build_pda():
    """Return a freshly initialized official PDA(16, deformable_groups=4).

    Native CUDA forward/backward is required. The upstream CPU operator is
    intentionally not emulated with torchvision or grid_sample. Constructing
    the module on CPU is supported once the native extension has been built.
    """
    source = Path(__file__).resolve().parents[1] / "upstream" / "rfr"
    package_name = "_deeppro_m1_source_relations_rfr"
    existing = sys.modules.get(package_name)
    if existing is not None and Path(existing.__file__).resolve() != source / "__init__.py":
        raise RuntimeError("The private RFR module name is already bound to another source tree.")
    if existing is None:
        spec = util.spec_from_file_location(
            package_name,
            source / "__init__.py",
            submodule_search_locations=[str(source)],
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("Cannot locate this experiment's vendored RFR package.")
        package = util.module_from_spec(spec)
        sys.modules[package_name] = package
        try:
            spec.loader.exec_module(package)
        except Exception:
            sys.modules.pop(package_name, None)
            raise
    try:
        module = import_module(f"{package_name}.pda")
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "This experiment's native RFR DCN extension is unavailable or ABI-incompatible. "
            "From the project root run: python3 "
            "experiments/m1_source_relations_v1/upstream/rfr/build_native.py . "
            "Use the pinned image documented in upstream/rfr/README.md; no substitute "
            "alignment backend or historical binary is loaded."
        ) from exc
    return module.PDA(num_feat=16, deformable_groups=4)
