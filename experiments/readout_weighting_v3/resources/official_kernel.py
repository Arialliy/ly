"""Execute reviewed original DEIM methods without importing its unrelated tree."""
import ast
import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "experiments/track_contrast_v2/resources/DEIM_deim_criterion.py"
SOURCE_SHA256 = "380a7d86b71ba9a3cbbdb670b4c11a3f2bfd2e367ffe87db7ef8e9ba30806dd2"
COMMIT = "09d35d53d39ee3145a1e61e3a989b28b9468d1dd"


def author_kernel(num_classes=1):
    raw = SOURCE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SOURCE_SHA256
    tree = ast.parse(raw.decode("utf-8"))
    cls, = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DEIMCriterion"]
    wanted = {"loss_labels_mal", "_get_src_permutation_idx"}
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    assert len(methods) == 2
    namespace = {"torch": torch, "F": F}
    for method in methods:
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), "exec"), namespace)
    kernel = type("OriginalDEIMKernel", (), {name: namespace[name] for name in wanted})()
    kernel.gamma, kernel.mal_alpha, kernel.num_classes = 2.0, None, num_classes
    return kernel


def author_all_background(logits):
    """Actual MAL, empty match sets, default gamma/alpha; mean over all logits."""
    batch, queries, classes = logits.shape
    empty = torch.empty(0, dtype=torch.long, device=logits.device)
    indices = [(empty.clone(), empty.clone()) for _ in range(batch)]
    targets = [{"labels": empty.clone()} for _ in range(batch)]
    outputs = {"pred_logits": logits, "pred_boxes": logits.new_zeros((batch, queries, 4))}
    return author_kernel(classes).loss_labels_mal(
        outputs, targets, indices, num_boxes=logits.numel(), values=logits.new_empty(0)
    )["loss_mal"]
