"""S2b fixed-subset fit, starting from M0's seeded random initialization."""
from collections import Counter
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch

from ..checks.check_s0 import atomic_text
from ..contracts import CENTER_INDEX
from ..data.dataset import MaskClipDataset
from ..losses.detection import detection_loss
from ..models.video_baseline import VideoBaseline


STEPS = 400
SEED = 46
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DIAGNOSTIC_INTERVAL = 50


def _json(path, data):
    atomic_text(path, json.dumps(data, indent=2, allow_nan=False) + "\n")


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sample_on_device(sample, device):
    """Move tensors only; retain the original scene and object metadata."""
    clip, targets = sample
    return (replace(clip, frames=clip.frames.to(device), time_offsets=clip.time_offsets.to(device),
                    pixel_valid=clip.pixel_valid.to(device)),
            replace(targets, target_map=targets.target_map.to(device), ignore_map=targets.ignore_map.to(device)))


def _load_samples(project_root, experiment, subset):
    full = experiment / "manifests/nudt_mirsdt_mask17.json"
    data = MaskClipDataset(json.loads(full.read_bytes()), project_root, "train")
    lookup = {center: i for i, center in enumerate(data.centers)}
    samples = {}
    for row in subset["samples"]:
        sample_id = row["sample_id"]
        clip, targets = data[lookup[row["source"], row["center"]]]
        if sample_id in samples or clip.output_frame_id != (sample_id,):
            raise ValueError(f"Duplicate or incorrect diagnostic sample: {sample_id}")
        foreground = int(targets.target_map.sum())
        if (list(clip.scene_meta[0].original_hw) != row["native_hw"]
                or foreground != row["foreground_pixels"] or bool(foreground) != row["positive"]):
            raise ValueError(f"Frozen polarity, native size or foreground count disagrees: {sample_id}")
        if clip.frames.shape[0] != 1 or tuple(clip.frames.shape[-2:]) != clip.scene_meta[0].original_hw:
            raise ValueError(f"Expected one native, full-resolution observation: {sample_id}")
        samples[sample_id] = (clip, targets)
    if len(samples) != 12 or sum(row["positive"] for row in subset["samples"]) != 8:
        raise ValueError("Expected eight positive and four empty distinct training centers")
    return samples


def _parameter_groups(model):
    modules = {"frame_encoder": model.frame_encoder, "time_convolutions": model.video_aggregator.temporal,
               "fusion": model.video_aggregator.fusion, "spatial_decoder": model.spatial_decoder,
               "detection_head": model.detection_head}
    groups = {name: tuple(module.parameters()) for name, module in modules.items()}
    ids = [id(p) for params in groups.values() for p in params]
    if len(set(ids)) != len(ids) or set(ids) != {id(p) for p in model.parameters()}:
        raise ValueError("Diagnostic module groups must partition the full model")
    return groups


def _vectors(groups):
    return {name: torch.cat([p.detach().reshape(-1) for p in params])
            for name, params in groups.items()}


def _delta(current, initial, previous):
    return {name: {"parameters": vector.numel(),
                   "delta_l2_from_init": float(torch.linalg.vector_norm(vector.double() - initial[name])),
                   "delta_l2_from_previous_snapshot": float(torch.linalg.vector_norm(vector.double() - previous[name]))}
            for name, vector in current.items()}


def _checkpoint(path, model, optimizer, subset, dependencies, device, config):
    algo, keys, pos, has_gauss, cached = np.random.get_state()
    state = {
        "format_version": 1, "stage": "S2b", "global_step": STEPS,
        "model_config": {"class": "VideoBaseline", "trainable_parameters": sum(p.numel() for p in model.parameters())},
        "training_config": config, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "schedule": subset["schedule"], "next_sample_index": STEPS, "dependencies": dependencies,
        "rng": {"python": random.getstate(), "numpy": (algo, keys.tolist(), pos, has_gauss, cached),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else []},
    }
    with tempfile.NamedTemporaryFile(prefix=".s2b-checkpoint-", dir=path.parent, delete=False) as out:
        tmp = Path(out.name)
        try:
            torch.save(state, out)
            out.flush()
            os.fsync(out.fileno())
            out.close()
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
    return {"path": path.name, "sha256": _sha256(path), "global_step": STEPS}


def run_short_fit(project_root: Path, run_dir: Path, subset: dict, device: torch.device) -> dict:
    """Run the preregistered 400 updates exactly once in a new output directory."""
    from ..evaluation.s2b_diagnostics import evaluate_snapshot

    project_root, run_dir, device = Path(project_root), Path(run_dir), torch.device(device)
    experiment = project_root / "experiments/irstd_independent_v1"
    subset_path = experiment / "manifests/s2b_train12.json"
    if json.loads(subset_path.read_bytes()) != subset:
        raise ValueError("The training subset differs from the frozen on-disk manifest")
    samples = _load_samples(project_root, experiment, subset)
    schedule = subset["schedule"]
    if subset["seed"] != SEED or len(schedule) != STEPS or set(schedule) != set(samples):
        raise ValueError("Invalid fixed training schedule")
    for start in range(0, STEPS, len(samples)):
        epoch = schedule[start:start + len(samples)]
        if len(set(epoch)) != len(epoch):
            raise ValueError(f"Repeated training example in epoch starting at update {start + 1}")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    model = VideoBaseline().to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, foreach=False)
    groups = _parameter_groups(model)
    parameters = tuple(model.parameters())
    initial = _vectors(groups)
    previous = initial
    config = {"seed": SEED, "precision": "float32", "batch_size": 1, "steps": STEPS,
              "diagnostic_interval": DIAGNOSTIC_INTERVAL, "optimizer": "AdamW", "lr": LEARNING_RATE,
              "weight_decay": WEIGHT_DECAY, "betas": [0.9, 0.999], "eps": 1e-8,
              "augmentation": False, "scheduler": None, "amp": False, "accumulation_steps": 1,
              "deterministic_algorithms": True, "cudnn_benchmark": False, "tf32": False}
    bound = ["manifests/s2b_train12.json", "manifests/nudt_mirsdt_mask17.json", "reports/S0.json",
             "reports/S1.json", "reports/S2a.json", "configs/task.yaml", "engine/s2b_train.py",
             "evaluation/s2b_diagnostics.py", "evaluation/s2b_froc.py", "data/s2b_subset.py"]
    dependencies = {path: _sha256(experiment / path) for path in bound}
    run_dir.mkdir(parents=True, exist_ok=True)
    progress = {"phase": "diagnostics", "completed_steps": 0, "last_completed_diagnostic": None}
    snapshots = []
    with (run_dir / "updates.jsonl").open("x", encoding="utf-8") as history:
        _json(run_dir / "progress.json", progress)
        for step in range(STEPS + 1):
            if step:
                model.train()
                sample_id = schedule[step - 1]
                clip, targets = sample_on_device(samples[sample_id], device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(clip)
                if prediction.output_frame_id != (sample_id,):
                    raise ValueError(f"Prediction ID differs from scheduled sample: {sample_id}")
                loss = detection_loss(prediction, targets, clip.pixel_valid[:, CENTER_INDEX])
                if not torch.isfinite(loss).item():
                    raise FloatingPointError(f"Nonfinite training loss at step {step} ({sample_id})")
                loss.backward()
                if any(p.grad is None for p in parameters):
                    raise FloatingPointError(f"Disconnected trainable parameter at step {step}")
                grad = torch.cat([p.grad.detach().flatten() for p in parameters])
                grad_norm = torch.linalg.vector_norm(grad.double()).item()
                if not np.isfinite(grad_norm):
                    raise FloatingPointError(f"Nonfinite gradient at step {step} ({sample_id})")
                optimizer.step()
                # Check all model weights and both Adam moments after the completed update.
                tracked = [v.reshape(-1) for p in parameters for v in
                           (p.detach(), optimizer.state[p]["exp_avg"], optimizer.state[p]["exp_avg_sq"])]
                if not torch.isfinite(torch.cat(tracked)).all().item():
                    raise FloatingPointError(f"Nonfinite parameter or AdamW moment at step {step} ({sample_id})")
                row = {"step": step, "sample_id": sample_id, "loss_before_update": loss.item(),
                       "gradient_l2": grad_norm, "finite_after_update": True}
                history.write(json.dumps(row, allow_nan=False) + "\n")
                history.flush()
                os.fsync(history.fileno())
                progress.update(completed_steps=step, phase="training", last_update=row)
                _json(run_dir / "progress.json", progress)
                if step % 25 == 0:
                    print(f"S2b update {step}/{STEPS}: sample={sample_id}, loss={row['loss_before_update']:.6f}",
                          flush=True)
                del clip, targets, prediction, loss, grad, tracked
            if step % DIAGNOSTIC_INTERVAL == 0:
                progress["phase"] = "diagnostics"
                _json(run_dir / "progress.json", progress)
                model.eval()
                with torch.inference_mode():
                    snapshot = evaluate_snapshot(model, samples, device, step, run_dir)
                current = _vectors(groups)
                snapshot["step"] = step
                snapshot["module_updates"] = _delta(current, initial, previous)
                previous = current
                snapshots.append(snapshot)
                _json(run_dir / f"snapshot_{step:03d}.json", snapshot)
                progress.update(last_completed_diagnostic=step, phase="training")
                _json(run_dir / "progress.json", progress)
                print(f"S2b diagnostic {step}: mean_loss={snapshot['loss']['total']:.6f}", flush=True)

    checkpoint = _checkpoint(run_dir / "final.pt", model, optimizer, subset, dependencies, device, config)
    progress.update(phase="complete", checkpoint=checkpoint)
    _json(run_dir / "progress.json", progress)
    return {"steps_completed": STEPS, "snapshots": snapshots, "all_finite": True, "training_config": config,
            "exposure_by_sample": dict(Counter(schedule)), "history": "updates.jsonl", "checkpoint": checkpoint}
