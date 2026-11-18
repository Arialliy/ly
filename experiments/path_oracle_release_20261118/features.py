"""Frozen DeepPro+ inference on original-window, full-resolution features."""
from hashlib import sha256
import importlib.util
from pathlib import Path

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[2]
ARCHIVE = PROJECT / "archive" / "original_project"
SOURCE = ARCHIVE / "official_deeppro"
LOADER = (ARCHIVE / "temporal_diagnosis_20261010" / "top1_background_control_20261011"
          / "code" / "train_pilot.py")
CHECKPOINT = (SOURCE / "log" / "sem_seg"
              / "RGB-T__2025-01-14_15-27__SoftLoUloss_DeepPro-Plus_DataL40"
              / "checkpoints" / "best_model.pth")
CHECKPOINT_SHA256 = "4fd4aa2736d9a33f2bf1459099068d8b4a7e202ff80aecdb4a3277fd13c7abcc"
MODEL_SHA256 = "fa46a034a30d8bbb0f677271d6368f033fcb389d682478d331551d8bae087668"
O_STARTS = (0, 36, 72, 108, 120)
WINDOW = 40
CHANNELS = 32
HALO = 20


def load_release(device="cuda"):
    """Load the checkpoint-adjacent architecture with the historical strict loader."""
    if sha256(CHECKPOINT.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
        raise ValueError("Release checkpoint digest mismatch")
    spec = importlib.util.spec_from_file_location("oracle_frozen_release_loader", LOADER)
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    model, model_path = loader.load_model(SOURCE, CHECKPOINT)
    if sha256(model_path.read_bytes()).hexdigest() != MODEL_SHA256:
        raise ValueError("Release model source digest mismatch")
    return model.requires_grad_(False).eval().to(device=device, dtype=torch.float32)


def _halo_bounds(lo, hi, limit):
    # Match the global tile phase used for release inference.
    return max(0, lo - HALO) // 2 * 2, min(limit, ((hi + HALO + 1) // 2) * 2)


@torch.inference_mode()
def extract_features(model, raw, start, roi):
    """Compute [1,32,40,h,w] before TPro. roi is global half-open (x0,y0,x1,y1).

    raw is the memory-mapped uint8 [T,H,W] sequence. All 40 original frames
    participate in the backbone, with true padding only at the image border.
    """
    if model.training or raw.ndim != 3 or raw.dtype != np.uint8:
        raise ValueError("Expected an eval model and a uint8 [T,H,W] raw sequence")
    frames, height, width = raw.shape
    x0, y0, x1, y1 = roi
    if not (0 <= start <= frames - WINDOW and 0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"Invalid window start {start} or ROI {roi} for {raw.shape}")
    xa, xb = _halo_bounds(x0, x1, width)
    ya, yb = _halo_bounds(y0, y1, height)
    crop = raw[start:start + WINDOW, ya:yb, xa:xb].astype(np.float32)
    crop -= np.float32(85.0799)
    crop /= np.float32(47.4845)
    device = next(model.parameters()).device
    x = torch.from_numpy(crop)[None, None].to(device)
    features = model.layer1(model.conv_in(x))
    if features.shape != (1, CHANNELS, WINDOW, yb - ya, xb - xa):
        raise ValueError(f"Unexpected original-feature shape {features.shape}")
    return features[..., y0-ya:y1-ya, x0-xa:x1-xa].contiguous()


@torch.inference_mode()
def project_logits(model, features, slot):
    """Run the unmodified release TPro and heads. Return [B,h,w] FP32 logits.

    Build one [B,32,40,h,w] batch for the same-shaped conditions. Only the
    requested endpoint is returned. TPro receives the entire 40-slot history.
    """
    if model.training or not 0 <= slot < WINDOW:
        raise ValueError(f"Expected eval mode and a valid local window slot, got {slot}")
    if features.ndim != 5 or features.shape[1:3] != (CHANNELS, WINDOW) or features.dtype != torch.float32:
        raise ValueError("Expected FP32 [B,32,40,h,w] pre-TPro features")
    projected = model.TPro(features.permute(0, 3, 4, 1, 2))
    # Both remaining convolutions are pointwise. Eval-mode BatchNorm is fixed.
    endpoint = projected[:, :, slot:slot+1].contiguous()
    return model.conv_out2(model.conv_out1(endpoint))[:, 0, 0]
