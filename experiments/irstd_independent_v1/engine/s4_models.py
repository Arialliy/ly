"""Construct the four frozen S4 variants and evaluate arbitrary scheduled native cores."""
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import random

import numpy as np
import torch

from ..contracts import CENTER_INDEX, Prediction
from ..geometry.tiles import Tile, _expand
from ..models.hypothesis_detector import HypothesisEvidenceDetector
from ..models.input_checks import validate_clip
from ..models.video_baseline import VideoBaseline
from .checkpoints import configure_determinism, read_checkpoint, require_exact


MODES = {"M0": None, "Msep": "additive_only", "Mraw": "raw_joint", "Mint": "interaction"}


def make_model(mode):
    if mode not in MODES:
        raise ValueError(f"Unknown S4 model {mode!r}")
    if mode == "M0":
        model = VideoBaseline()
    else:
        model = HypothesisEvidenceDetector(mode=MODES[mode], query_chunk=64,
                                           velocity_chunk=8, checkpoint_chunks=True)
    if (not all(p.requires_grad for p in model.parameters()) or
            sum(p.numel() for p in model.parameters()) != (22921 if mode == "M0" else 23945)):
        raise ValueError(f"Unexpected trainable layers for {mode}")
    return model.to(device="cpu", dtype=torch.float32)


def initial_model(run_dir, protocol, mode, seed, device):
    """Load the paired initialization recorded before S4, with its per-tensor witness."""
    run_dir = Path(run_dir)
    if (protocol["stage"] != "S4" or protocol["run_id"] != run_dir.name or
            mode not in protocol["modes"] or protocol["mode_to_evidence"] != MODES or
            seed not in protocol["seeds"]):
        raise ValueError("Model/seed is outside this S4 run")
    init = protocol["initialization"][str(seed)]
    state = read_checkpoint(run_dir / init["file"], init["sha256"])
    raw = (run_dir / init["tensors"]).read_bytes()
    if sha256(raw).hexdigest() != init["tensors_sha256"]:
        raise ValueError("Initialization tensor manifest SHA-256 mismatch")
    witness = json.loads(raw)
    if (state["format_version"] != 1 or state["seed"] != seed or witness["seed"] != seed or
            state["mode_to_evidence"] != MODES or set(state["states"]) != set(MODES)):
        raise ValueError("Mismatched paired initialization metadata")
    weights = state["states"][mode]
    hashes = {k: {"shape": list(v.shape), "dtype": str(v.dtype),
                  "sha256": sha256(v.contiguous().numpy().tobytes()).hexdigest()} for k,v in weights.items()}
    if hashes != witness["tensors"][mode]:
        raise ValueError(f"Initialization tensor identity mismatch for {mode}")
    configure_determinism()
    model = make_model(mode)
    model.load_state_dict(weights, strict=True)
    require_exact(weights, model.state_dict(), "paired initialization")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return model.to(device).train()


def _check_core(clip, box):
    if len(clip.scene_meta) != 1 or len(box) != 4 or any(type(n) is not int for n in box):
        raise ValueError("Expected one native clip and four integer xyxy coordinates")
    x, y, ex, ey = box
    hw = clip.scene_meta[0].original_hw
    if not (0 <= x < ex <= hw[1] and 0 <= y < ey <= hw[0] and (ex-x, ey-y) == (64, 64)):
        raise ValueError(f"Expected a 64x64 core within the native frame, got {box} at {hw}")
    return hw


def core_prediction(model, clip, box):
    hw = _check_core(clip, box)
    core = tuple(box)
    decoder = model.spatial_decoder.radius
    query = _expand(core, decoder, decoder, hw)
    local = max(model.frame_encoder.radius, model.contrast_channel.radius)
    if type(model) is HypothesisEvidenceDetector:
        # The frozen S1 grid reaches 2 native pixels per frame in both axes.
        halo = 2 * CENTER_INDEX + local + 1
        tile = Tile(core, query, _expand(query, halo, halo, hw), hw)
        pred = model.forward_core(clip, tile)
    elif type(model) is VideoBaseline:
        validate_clip(clip)
        tile = Tile(core, query, _expand(query, local, local, hw), hw)
        source = HypothesisEvidenceDetector._crop(clip, tile, 0)
        rows, cols = tile.core_slice_in_query
        pred = Prediction(model._region_logits(source, tile)[..., rows, cols], clip.output_frame_id)
    else:
        raise TypeError("Expected an S4 detector")
    if (pred.output_frame_id != clip.output_frame_id or pred.logits.shape != (1, 1, 64, 64)
            or pred.logits.dtype != torch.float32):
        raise ValueError("Core prediction differs from the registered native logit contract")
    return pred


def core_labels(target, clip, box):
    hw = _check_core(clip, box)
    if (target.label_type != "mask" or target.target_map.shape != (1, 1, *hw) or
            target.ignore_map.shape != target.target_map.shape or len(target.objects) != 1):
        raise ValueError("Core supervision must be a single native mask")
    x, y, ex, ey = box
    local_objects = []
    for obj in target.objects[0]:
        x0, y0, x1, y1 = obj.bbox_xyxy
        if x1 <= x or y1 <= y or x0 >= ex or y0 >= ey:
            continue
        if x0 < x or y0 < y or x1 > ex or y1 > ey:
            raise ValueError("Scheduled core truncates a labeled object")
        cx, cy = obj.centroid_xy
        local_objects.append(replace(obj, bbox_xyxy=(x0-x, y0-y, x1-x, y1-y), centroid_xy=(cx-x, cy-y)))
    labels = replace(target, target_map=target.target_map[..., y:ey, x:ex],
                     ignore_map=target.ignore_map[..., y:ey, x:ex], objects=(tuple(local_objects),))
    return labels, clip.pixel_valid[:, CENTER_INDEX, :, y:ey, x:ex]
