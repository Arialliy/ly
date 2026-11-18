"""S4 native-core loss with per-update hard-background selection."""
import torch
from torch.nn import functional as F

from ..contracts import Prediction, Targets


def s4_detection_loss(prediction: Prediction, targets: Targets, center_valid: torch.Tensor, core_xyxy):
    """Return one differentiable loss and detached, JSON-ready component diagnostics.

    Labels and validity cover the original image. A square with Chebyshev radius 3
    around each valid foreground pixel protects hard mining even across a core edge.
    All valid background pixels still contribute to the ordinary background BCE.
    """
    if not isinstance(prediction, Prediction) or not isinstance(targets, Targets) or targets.label_type != "mask":
        raise TypeError("S4 expects a native mask target and a Prediction")
    y = targets.target_map
    if (y.ndim != 4 or tuple(y.shape[:2]) != (1, 1) or y.dtype != torch.float32
            or y.requires_grad or len(prediction.output_frame_id) != 1):
        raise ValueError("S4 requires one original center-frame binary mask")
    z = prediction.logits
    for mask in (center_valid, targets.ignore_map):
        if mask.shape != y.shape or mask.device != y.device or mask.dtype != torch.bool:
            raise ValueError("Native validity and ignore masks must match the full label")
    if (len(core_xyxy) != 4 or any(type(coord) is not int for coord in core_xyxy)
            or z.device != y.device or z.dtype != torch.float32):
        raise ValueError("Expected four integral native coordinates and FP32 core logits")
    x0, y0, x1, y1 = core_xyxy
    if not (0 <= x0 < x1 <= y.shape[-1] and 0 <= y0 < y1 <= y.shape[-2]
            and tuple(z.shape) == (1, 1, y1 - y0, x1 - x0)):
        raise ValueError("Core output does not match its rectangle inside the native label")
    native_valid = center_valid & ~targets.ignore_map
    if not ((y == 0) | (y == 1) | ~native_valid).all():
        raise ValueError("Non-binary supervision in a valid native pixel")
    native_fg = (y == 1) & native_valid
    protected = F.max_pool2d(native_fg.float(), 7, stride=1, padding=3) > 0
    valid = native_valid[..., y0:y1, x0:x1]
    foreground = native_fg[..., y0:y1, x0:x1]
    background = valid & ~foreground
    eligible = background & ~protected[..., y0:y1, x0:x1]
    if not valid.any() or not torch.isfinite(z[valid]).all():
        raise ValueError("Expected finite logits and a nonempty supervision region")

    positive = z[foreground]
    negative = z[background]
    candidates = z[eligible]
    n = candidates.numel()
    k = min(n, max(16, (n + 9) // 10))
    fg_loss = F.softplus(-positive).mean() if positive.numel() else None
    all_bg = F.softplus(negative).mean() if negative.numel() else None
    hard = None
    if k:
        # Boolean indexing enumerates rows then columns. Within one native core
        # this is precisely the global (y,x) order, including score ties.
        order = torch.argsort(candidates.detach(), descending=True, stable=True)
        hard = F.softplus(candidates[order[:k]]).mean()
    bg_loss = (0.75 * all_bg + 0.25 * hard) if hard is not None else all_bg
    existing = [v for v in (fg_loss, bg_loss) if v is not None]
    total = sum(existing) / len(existing)
    dice = None
    if positive.numel():
        p = z[valid].sigmoid()
        intersection = p[foreground[valid]].sum()
        dice = 1 - (2 * intersection + 1) / (p.sum() + positive.numel() + 1)
        total = total + 0.5 * dice
    fields = {"total": total, "foreground_bce": fg_loss, "background_bce_all": all_bg,
              "background_bce_hard": hard, "background_bce": bg_loss, "dice_loss": dice}
    values = {name: None if value is None else value.detach().item() for name, value in fields.items()}
    return total, {**values, "foreground_pixels": positive.numel(), "background_pixels": negative.numel(),
                   "eligible_hard_pixels": n, "selected_hard_pixels": k}
