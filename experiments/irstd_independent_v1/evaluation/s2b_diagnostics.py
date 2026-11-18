"""Fixed-center diagnostics for S2b, evaluated at initialization and every 50 updates."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..losses.detection import detection_loss
from .s2b_froc import evaluate_froc


def _frame_loss(pred, target, center_valid):
    valid = center_valid & ~target.ignore_map
    z = pred.logits[valid].float()
    y = target.target_map[valid]
    fg = y == 1
    bg = y == 0
    classes = []
    fg_bce = bg_bce = dice = None
    if fg.any():
        fg_bce = F.softplus(-z[fg]).mean().item()
        classes.append(fg_bce)
        probability = z.sigmoid()
        dice = (1 - (2 * probability[fg].sum() + 1) / (probability.sum() + fg.sum() + 1)).item()
    if bg.any():
        bg_bce = F.softplus(z[bg]).mean().item()
        classes.append(bg_bce)
    total = detection_loss(pred, target, center_valid).item()
    reconstructed = np.mean(classes) + 0.5 * (dice if dice is not None else 0.0)
    if not np.isfinite(total) or not np.isclose(total, reconstructed, rtol=1e-5, atol=1e-6):
        raise ValueError("Independent decomposition disagrees with the S2a detection loss")
    return {"total": total, "foreground_bce": fg_bce, "background_bce": bg_bce, "dice_loss": dice}


@torch.inference_mode()
def evaluate_snapshot(model, samples, device, step, run_dir: Path):
    """Evaluate the same full, native images; mean loss is the unweighted mean over frames."""
    per_frame, metric_inputs = {}, []
    keep_maps = step in (0, 400)
    for sample_id, (clip, labels) in samples.items():
        if clip.output_frame_id != (sample_id,):
            raise ValueError(f"Expected one center {sample_id}, got {clip.output_frame_id}")
        clip = replace(clip, frames=clip.frames.to(device), pixel_valid=clip.pixel_valid.to(device),
                       time_offsets=clip.time_offsets.to(device))
        labels = replace(labels, target_map=labels.target_map.to(device), ignore_map=labels.ignore_map.to(device))
        pred = model(clip)
        native_hw = clip.scene_meta[0].original_hw
        valid = clip.pixel_valid[:, clip.center_index]
        if (pred.output_frame_id != (sample_id,) or tuple(pred.logits.shape) != (1, 1, *native_hw)
                or not bool(valid.all()) or bool(labels.ignore_map.any())):
            raise ValueError("S2b diagnostics require one fully valid native center image per frame")
        per_frame[sample_id] = _frame_loss(pred, labels, valid)
        scores = pred.logits[0, 0].detach().cpu().numpy()
        objects = [obj.centroid_xy for obj in labels.objects[0]]
        metric_inputs.append((sample_id, scores, objects))
    if len(metric_inputs) != 12 or sum(len(objects) for _, _, objects in metric_inputs) != 8:
        raise ValueError("Fixed S2b diagnostic population changed")
    if keep_maps:
        # Stable, explicit identifiers permit an independent endpoint audit without rerunning training.
        np.savez_compressed(run_dir / f"logits_{step:03d}.npz",
                            **{name: scores for name, scores, _ in metric_inputs})
    def average(key):
        return float(np.mean([r[key] for r in per_frame.values() if r[key] is not None]))
    losses = {"total": average("total"), "foreground_bce": average("foreground_bce"),
              "background_bce": average("background_bce"), "dice_loss_positive": average("dice_loss"),
              "empty_frame_total": float(np.mean([r["total"] for r in per_frame.values()
                                                  if r["foreground_bce"] is None]))}
    froc = evaluate_froc(metric_inputs, run_dir / f"froc_{step:03d}.npz")
    return {"step": step, "loss": losses, "froc": froc, "per_frame_loss": per_frame}
