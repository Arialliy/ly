"""Read and write restartable M0/AdamW states at optimizer update boundaries."""
from hashlib import sha256
from io import BytesIO
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch

from ..models.video_baseline import VideoBaseline


def configure_determinism():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def capture_rng(device):
    algo, keys, pos, gauss, cache = np.random.get_state()
    return {"python": random.getstate(), "numpy": (algo, keys.tolist(), pos, gauss, cache),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.device(device).type == "cuda" else []}


def restore_rng(saved, device):
    cuda_count = torch.cuda.device_count() if torch.device(device).type == "cuda" else 0
    if len(saved["torch_cuda"]) != cuda_count:
        raise ValueError(f"Archive has {len(saved['torch_cuda'])} CUDA RNG states, runtime has {cuda_count}")
    random.setstate(saved["python"])
    algo, keys, pos, gauss, cache = saved["numpy"]
    np.random.set_state((algo, np.asarray(keys, dtype=np.uint32), pos, gauss, cache))
    torch.set_rng_state(saved["torch_cpu"])
    if cuda_count:
        torch.cuda.set_rng_state_all(saved["torch_cuda"])


def require_exact(reference, actual, field):
    """Compare dtype, dimensions and raw tensor bits; name the first divergent state."""
    if isinstance(reference, torch.Tensor) and isinstance(actual, torch.Tensor):
        equal = (reference.shape == actual.shape and reference.dtype == actual.dtype and
                 torch.equal(reference.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                             actual.detach().cpu().contiguous().reshape(-1).view(torch.uint8)))
    elif isinstance(reference, dict) and isinstance(actual, dict):
        equal = reference.keys() == actual.keys()
        if equal:
            for k in reference:
                require_exact(reference[k], actual[k], f"{field}.{k}")
    elif isinstance(reference, (tuple, list)) and type(reference) is type(actual):
        equal = len(reference) == len(actual)
        if equal:
            for i, (left, right) in enumerate(zip(reference, actual)):
                require_exact(left, right, f"{field}[{i}]")
    else:
        equal = type(reference) is type(actual) and reference == actual
    if not equal:
        raise ValueError(f"State mismatch at {field}")


def read_checkpoint(path, expected_sha256):
    raw = Path(path).read_bytes()
    if sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"Checkpoint SHA-256 mismatch: {path}")
    return torch.load(BytesIO(raw), map_location="cpu", weights_only=True)


def restore_training(path, expected_sha256, experiment, device):
    """Restore an M0 and its optimizer first, then restore and verify all RNGs."""
    configure_determinism()
    state = read_checkpoint(path, expected_sha256)
    if state["format_version"] != 1 or state["stage"] not in {"S2b", "S2c"}:
        raise ValueError("Unrecognized M0 training checkpoint")
    if not (0 <= state["next_sample_index"] <= len(state["schedule"]) and state["global_step"] >= 0):
        raise ValueError("Invalid checkpoint cursor or global step")
    if not state["dependencies"]:
        raise ValueError("Missing bound training dependencies")
    for relative, expected in state["dependencies"].items():
        if sha256((Path(experiment) / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Bound training dependency changed: {relative}")
    cfg = state["training_config"]
    if (cfg["optimizer"], cfg["precision"], cfg["amp"], cfg["scheduler"],
        cfg["deterministic_algorithms"], cfg["cudnn_benchmark"], cfg["tf32"]) != (
            "AdamW", "float32", False, None, True, False, False):
        raise ValueError("Unsupported training configuration for M0 restart")
    model = VideoBaseline().to(device=device, dtype=torch.float32)
    if state["model_config"] != {"class": "VideoBaseline",
                                  "trainable_parameters": sum(p.numel() for p in model.parameters())}:
        raise ValueError("M0 parameter contract changed")
    model.load_state_dict(state["model"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=tuple(cfg["betas"]),
                                  eps=cfg["eps"], weight_decay=cfg["weight_decay"], foreach=False)
    optimizer.load_state_dict(state["optimizer"])
    model.train()
    require_exact(state["model"], model.state_dict(), "model")
    require_exact(state["optimizer"], optimizer.state_dict(), "optimizer")
    for group in optimizer.param_groups:
        if (group["lr"], group["weight_decay"], group["betas"], group["eps"], group["foreach"]) != (
                cfg["lr"], cfg["weight_decay"], tuple(cfg["betas"]), cfg["eps"], False):
            raise ValueError("Restored AdamW parameter group differs from the bound training configuration")
    for param in model.parameters():
        buffers = optimizer.state[param]
        if int(buffers["step"].item()) != state["global_step"]:
            raise ValueError("Adam parameter step differs from checkpoint step")
    restore_rng(state["rng"], device)
    require_exact(state["rng"], capture_rng(device), "restored_rng")
    return model, optimizer, state


def save_training(path, model, optimizer, *, device, global_step, next_sample_index, schedule,
                  training_config, dependencies, parent_sha256, audit_branch):
    """Persist a new audit checkpoint without overwriting any archived state."""
    if global_step < 0 or not 0 <= next_sample_index <= len(schedule):
        raise ValueError("Invalid output checkpoint cursor or step")
    if not audit_branch or len(parent_sha256) != 64:
        raise ValueError("Audit checkpoints require a branch name and a parent SHA-256")
    path = Path(path)
    state = {"format_version": 1, "stage": "S2c", "audit_branch": audit_branch,
             "parent_checkpoint_sha256": parent_sha256, "global_step": global_step,
             "next_sample_index": next_sample_index, "schedule": list(schedule),
             "model_config": {"class": type(model).__name__, "trainable_parameters": sum(
                 p.numel() for p in model.parameters() if p.requires_grad)},
             "training_config": training_config, "dependencies": dependencies,
             "model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": capture_rng(device)}
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".checkpoint-", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return sha256(path.read_bytes()).hexdigest()
