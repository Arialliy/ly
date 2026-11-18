"""Mask-supervised S2 loss with equal frame weights and explicit empty-frame behavior."""
import torch
import torch.nn.functional as F

from ..contracts import Prediction, Targets


def detection_loss(prediction: Prediction, targets: Targets, center_valid: torch.Tensor) -> torch.Tensor:
    """Mean of per-frame balanced BCE plus 0.5 soft Dice for nonempty target masks.

    Valid foreground and background receive equal class weight when both exist.
    S0 returns labels in the same batch order as its center IDs. Targets has no ID
    field. Identity is checked at the clip boundary and maintained by its collator.
    """
    if not isinstance(prediction, Prediction) or not isinstance(targets, Targets):
        raise TypeError("detection_loss requires a Prediction and S0 Targets")
    if targets.label_type != "mask":
        raise ValueError("S2 requires native binary mask supervision")
    logits = prediction.logits
    if not isinstance(logits, torch.Tensor) or not logits.is_floating_point():
        raise TypeError("Prediction logits must be floating point")
    if logits.ndim != 4 or logits.shape[1] != 1 or min(logits.shape) <= 0:
        raise ValueError("Prediction logits must have nonempty shape [B,1,H,W]")
    if len(prediction.output_frame_id) != logits.shape[0]:
        raise ValueError("Prediction center ID count differs from its batch size")
    for name, tensor, dtype in (("target_map", targets.target_map, torch.float32),
                                ("ignore_map", targets.ignore_map, torch.bool),
                                ("center_valid", center_valid, torch.bool)):
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != dtype or tensor.requires_grad:
            raise TypeError(f"{name} must be a fixed {dtype} tensor")
        if tensor.shape != logits.shape or tensor.device != logits.device:
            raise ValueError(f"{name} must have the same shape and device as the predicted logits")

    valid = center_valid & ~targets.ignore_map
    spatial = (1, 2, 3)
    if not valid.any(dim=spatial).all():
        raise ValueError("Every supervised frame requires at least one valid, nonignored pixel")
    # Sanitize ignored storage before any nonlinear operation, including sigmoid.
    z = torch.where(valid, logits.float(), 0.0)
    y = torch.where(valid, targets.target_map, 0.0)
    if not torch.isfinite(z).all():
        raise ValueError("Nonfinite logit in the valid supervision region")
    if not ((y == 0) | (y == 1)).all():
        raise ValueError("Expected finite binary mask values on every supervised pixel")

    foreground = valid & (y == 1)
    background = valid & (y == 0)
    n_fg, n_bg = foreground.sum(dim=spatial), background.sum(dim=spatial)
    bce = F.binary_cross_entropy_with_logits(z, y, reduction="none")
    fg_bce = torch.where(foreground, bce, 0.0).sum(dim=spatial) / n_fg.clamp_min(1)
    bg_bce = torch.where(background, bce, 0.0).sum(dim=spatial) / n_bg.clamp_min(1)
    class_count = (n_fg > 0).to(torch.float32) + (n_bg > 0).to(torch.float32)
    balanced_bce = (fg_bce + bg_bce) / class_count

    probability = torch.where(valid, z.sigmoid(), 0.0)
    intersection = (probability * y).sum(dim=spatial)
    dice_loss = 1.0 - (2.0 * intersection + 1.0) / (probability.sum(dim=spatial) + n_fg + 1.0)
    return (balanced_bce + 0.5 * torch.where(n_fg > 0, dice_loss, 0.0)).mean()
