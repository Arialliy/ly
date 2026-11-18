"""Native-resolution prediction, with an audit of the model's actual tiled output writes."""
from dataclasses import replace
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.overrides import TorchFunctionMode


class OutputWrites(TorchFunctionMode):
    """Observe successful Tensor.__setitem__ calls; only the returned buffer counts."""

    def __init__(self):
        super().__init__()
        self.destinations = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        if func is torch.Tensor.__setitem__ and args[0].ndim == 4:
            self.destinations.append((args[0], args[1]))
        return result

    def coverage(self, logits):
        _, _, height, width = logits.shape
        counts = np.zeros((height, width), dtype=np.int32)
        cores = []
        for target, key in self.destinations:
            if target is not logits:
                continue
            if len(key) != 4 or not all(isinstance(part, slice) for part in key):
                raise ValueError("Unexpected index used to write the final tiled output")
            batch, channel, rows, cols = (k.indices(n) for k, n in zip(key, logits.shape))
            if batch != (0, 1, 1) or channel != (0, 1, 1) or rows[2] != 1 or cols[2] != 1:
                raise ValueError("Expected a dense, single-frame core assignment")
            x0, x1, _ = cols
            y0, y1, _ = rows
            if x0 >= x1 or y0 >= y1:
                raise ValueError("Empty core assignment")
            counts[y0:y1, x0:x1] += 1
            cores.append([x0, y0, x1, y1])
        return {"core_boxes_xyxy": cores, "core_count": len(cores),
                "unwritten_pixels": int((counts == 0).sum()), "repeated_pixels": int((counts > 1).sum()),
                "min_writes": int(counts.min()), "max_writes": int(counts.max())}


def compare_logits(full, tiled, atol, rtol):
    """Compare against full-frame FP32 logits. Retain evidence even if the gate fails."""
    expected = torch.from_numpy(full)
    observed = torch.from_numpy(tiled)
    matched = torch.isclose(observed, expected, atol=atol, rtol=rtol)
    error = (observed - expected).abs()
    allowed = atol + rtol * expected.abs()
    ratio = error / allowed

    def example(which):
        flat = int(which.argmax())
        y, x = divmod(flat, full.shape[1])
        return {"xy": [x, y], "full": float(expected[y, x]), "tiled": float(observed[y, x]),
                "abs_error": float(error[y, x]), "allowed": float(allowed[y, x]),
                "tolerance_ratio": float(ratio[y, x])}

    return {"outside_tolerance_pixels": int((~matched).sum()),
            "max_abs_error_location": example(error), "max_tolerance_ratio_location": example(ratio)}


def _save_logits(run_dir, mode, sample_id, logits):
    path = Path("predictions") / mode / (sample_id + ".npy")
    destination = run_dir / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        np.save(handle, logits, allow_pickle=False)
    return {"file": path.as_posix(), "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}


@torch.inference_mode()
def predict_native(model, dataset, sample_indices, run_dir, device, core_hw, atol, rtol):
    """Run full and tiled prediction without any supervised dataset access."""
    if model.training:
        raise ValueError("Prediction requires eval mode")
    rows = []
    for sample_id, index in sample_indices.items():
        clip = dataset.inference_item(index)
        shape = clip.scene_meta[0].original_hw
        if clip.output_frame_id != (sample_id,) or not bool(clip.pixel_valid.all()):
            raise ValueError(f"Expected a fully valid native clip for {sample_id}")
        clip = replace(clip, frames=clip.frames.to(device), pixel_valid=clip.pixel_valid.to(device),
                       time_offsets=clip.time_offsets.to(device))
        full = model(clip)
        trace = OutputWrites()
        with trace:
            tiled = model.predict_tiled(clip, core_hw=core_hw)
        maps, artifacts = {}, {}
        for name, pred in (("full", full), ("tiled", tiled)):
            if pred.output_frame_id != (sample_id,) or pred.logits.shape != (1, 1, *shape):
                raise ValueError(f"{name}: incorrect native output shape or ID for {sample_id}")
            scores = pred.logits[0, 0].cpu().numpy()
            if scores.dtype != np.float32 or not np.isfinite(scores).all():
                raise ValueError(f"{name}: invalid raw FP32 scores for {sample_id}")
            maps[name] = scores
            artifacts[name] = _save_logits(run_dir, name, sample_id, scores)
        rows.append({"id": sample_id, "native_hw": list(shape), "native_pixels": int(maps["full"].size),
                     "coverage": trace.coverage(tiled.logits), "logits": artifacts,
                     "agreement": compare_logits(maps["full"], maps["tiled"], atol, rtol)})
        if len(rows) % 42 == 0 or len(rows) == len(sample_indices):
            print(f"S2c prediction: {len(rows)}/{len(sample_indices)} full and tiled pairs", flush=True)
    return rows
