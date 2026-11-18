"""Load only the unchanged LFP definitions from a trusted local NS-FPN source.

This is an import-isolation convenience, not a security sandbox.  The selected
class bodies/decorators are Python code and execute with normal permissions.
No SFS module, repository-level import, or rewritten author class is executed.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from types import ModuleType
from typing import Tuple


CLASS_NAMES = (
    "ConvDWT",
    "ConvIDWT",
    "SpatialAttention",
    "LearnableGaussianFilterBank",
    "wav_Enhance",
)
DEFAULT_SOURCE = Path(__file__).resolve().parent / "upstream" / "NS_FPN.py"


def selected_class_tree(source: str, filename: str = "<official NS_FPN.py>") -> ast.Module:
    """Select the actual, unmodified top-level class AST nodes in source order."""
    tree = ast.parse(source, filename=filename)
    classes = [node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name in CLASS_NAMES]
    counts = {name: sum(node.name == name for node in classes) for name in CLASS_NAMES}
    if any(count != 1 for count in counts.values()):
        raise ValueError(f"Expected exactly one definition of each LFP class: {counts}")
    return ast.Module(body=classes, type_ignores=[])


def load_official_lfp(source_path: str | Path = DEFAULT_SOURCE,
                      expected_sha256: str | None = None) -> ModuleType:
    """Return a module exposing the selected author classes and source provenance."""
    path = Path(source_path).resolve()
    raw = path.read_bytes()
    source_sha = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and source_sha != expected_sha256:
        raise ValueError(f"Official source SHA mismatch: {source_sha} != {expected_sha256}")
    selected = selected_class_tree(raw.decode("utf-8"), filename=str(path))

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from pytorch_wavelets import DWTForward, DWTInverse

    module = ModuleType("ns_fpn_official_lfp")
    module.__file__ = str(path)
    module.__dict__.update(torch=torch, nn=nn, F=F, Tensor=torch.Tensor,
                           Tuple=Tuple, DWTForward=DWTForward, DWTInverse=DWTInverse)
    # AST nodes are selected, never edited; their original locations are retained.
    exec(compile(selected, filename=str(path), mode="exec"), module.__dict__)
    module.SOURCE_SHA256 = source_sha
    module.SELECTED_AST_SHA256 = hashlib.sha256(
        ast.dump(selected, include_attributes=False).encode("utf-8")).hexdigest()
    module.SELECTED_CLASS_NAMES = CLASS_NAMES
    module.__all__ = list(CLASS_NAMES)
    return module
