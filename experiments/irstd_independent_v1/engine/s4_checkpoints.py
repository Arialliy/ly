"""Immutable S4 AdamW checkpoints with a mode-bound model and an explicit update cursor."""
import hashlib
import json
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile

import torch

from ..contracts import CENTER_INDEX
from ..models.hypothesis_detector import HypothesisEvidenceDetector
from ..models.video_baseline import VideoBaseline
from .checkpoints import capture_rng, configure_determinism, read_checkpoint, require_exact, restore_rng
from .s4_models import MODES, make_model, training_chunks


FORMAT = 1
LR_STAGES = ((0, 1e-3), (8000, 3e-4), (16000, 1e-4))


def lr_for_update(next_step):
    if type(next_step) is not int or not 1 <= next_step <= 36000:
        raise ValueError("Expected a one-based S4 update number in [1, 36000]")
    return next(lr for completed, lr in reversed(LR_STAGES) if next_step > completed)


def make_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=lr_for_update(1), betas=(0.9, 0.999),
                             eps=1e-8, weight_decay=1e-4, foreach=False)


def _sha_string(digest):
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Expected a lowercase SHA-256 digest")
    return digest


def identity(protocol, mode, seed, *, protocol_sha256, execution_sha256):
    if (protocol["stage"] != "S4" or mode not in MODES or protocol["mode_to_evidence"] != MODES or
            type(seed) is not int or seed not in protocol["seeds"]):
        raise ValueError("Model or seed does not belong to the frozen S4 run")
    proposal = protocol["proposal"]
    cfg = proposal["models"][mode]
    runtime = proposal["evidence_runtime"]
    training_chunks(runtime)
    opt = proposal["optimization"]
    stages = tuple((r["start_completed_updates"], r["lr"]) for r in opt["lr_by_completed_updates"])
    settings = (opt["optimizer"], tuple(opt["betas"]), opt["eps"], opt["weight_decay"], opt["foreach"],
                opt["precision"], opt["amp"], opt["tf32"], opt["deterministic_algorithms"], opt["cudnn_benchmark"])
    if (stages != LR_STAGES or settings != ("AdamW", (0.9, 0.999), 1e-8, 0.0001, False, "float32", False,
                                             False, True, False) or cfg["mode"] != MODES[mode]):
        raise ValueError("Unrecognized model, optimizer or arithmetic contract")
    axis = runtime["velocity_axis"]
    if axis != [v / 2 for v in range(-4, 5)]:
        raise ValueError("Unexpected S4 velocity coordinates")
    velocities = [[[vx, vy] for vx in axis] for vy in axis] if mode != "M0" else []
    components = {"prerequisites": protocol["prerequisite_sha256"], "protocol_code": protocol["code_sha256"]}
    return {"run_id": protocol["run_id"], "mode": mode, "seed": seed, "model_config": cfg,
            "protocol_sha256": _sha_string(protocol_sha256), "execution_sha256": _sha_string(execution_sha256),
            "dependencies_sha256": hashlib.sha256(json.dumps(components, sort_keys=True).encode()).hexdigest(),
            "schedule_sha256": protocol["schedule_sha256"][str(seed)], "diagnostics_sha256": protocol["diagnostics_sha256"],
            "data_manifest_sha256": proposal["data"]["manifest_sha256"], "initialization": protocol["initialization"][str(seed)],
            "grid_config_sha256": runtime["grid_config_sha256"], "velocities_xy": velocities,
            "evidence_runtime": runtime if mode != "M0" else None,
            "precision_and_optimizer": {k: opt[k] for k in ("optimizer", "betas", "eps", "weight_decay", "foreach",
                                                           "precision", "amp", "tf32", "deterministic_algorithms",
                                                           "cudnn_benchmark")},
            "layout_version": "S3-native-xy-v1", "time_offsets": list(range(-CENTER_INDEX, CENTER_INDEX + 1))}


def _runtime(device):
    device = torch.device(device)
    result = {"device_type": device.type, "torch": str(torch.__version__),
              "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
              "cudnn_deterministic": torch.backends.cudnn.deterministic,
              "cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_tf32": torch.backends.cudnn.allow_tf32,
              "matmul_tf32": torch.backends.cuda.matmul.allow_tf32}
    if (not result["deterministic_algorithms"] or not result["cudnn_deterministic"] or
            any(result[k] for k in ("cudnn_benchmark", "cudnn_tf32", "matmul_tf32"))):
        raise ValueError("Runtime violates the frozen deterministic FP32 contract")
    if device.type == "cuda":
        result.update(cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                      capability=torch.cuda.get_device_capability(device), device_name=torch.cuda.get_device_name(device))
    else:
        result["num_threads"] = torch.get_num_threads()
    return result


def _check_model(model, bound):
    mode = bound["mode"]
    cls = VideoBaseline if mode == "M0" else HypothesisEvidenceDetector
    if type(model) is not cls or type(model).__name__ != bound["model_config"]["class"]:
        raise ValueError("S4 checkpoint model class mismatch")
    if not all(p.dtype == torch.float32 and p.requires_grad for p in model.parameters()):
        raise ValueError("Expected trainable FP32 parameters")
    if mode != "M0":
        ag = model.video_aggregator
        if bound["evidence_runtime"] is None:
            raise ValueError("Missing bound runtime for the evidence detector")
        query, velocity = training_chunks(bound["evidence_runtime"])
        if ((ag.encoder.mode, ag.query_chunk, ag.velocity_chunk, ag.checkpoint_chunks) !=
                (MODES[mode], query, velocity, True) or ag.velocities_xy.detach().cpu().tolist() != bound["velocities_xy"]):
            raise ValueError("S4 checkpoint evidence mode, coordinates or chunk configuration mismatch")


def _scheduler(step):
    if type(step) is not int or not 0 <= step <= 36000:
        raise ValueError("Invalid completed S4 update count")
    return {"completed_updates": step, "stages": LR_STAGES, "last_lr": lr_for_update(max(step, 1)),
            "next_lr": None if step == 36000 else lr_for_update(step + 1)}


def _check_optimizer(model, optimizer, step):
    sched = _scheduler(step)
    if type(optimizer) is not torch.optim.AdamW or len(optimizer.param_groups) != 1:
        raise ValueError("Expected single-group AdamW")
    group = optimizer.param_groups[0]
    frozen = {"lr": sched["last_lr"], "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 1e-4,
              "foreach": False, "amsgrad": False, "maximize": False, "capturable": False,
              "differentiable": False, "fused": None}
    if (any(group.get(k) != v for k, v in frozen.items()) or
            list(map(id, group["params"])) != list(map(id, model.parameters()))):
        raise ValueError("AdamW parameters or learning-rate boundary changed")
    if step == 0:
        if optimizer.state:
            raise ValueError("Nonempty optimizer at initialization")
        return
    if set(optimizer.state) != set(model.parameters()):
        raise ValueError("Checkpoint missing AdamW moments")
    for param, buffers in optimizer.state.items():
        if (set(buffers) != {"step", "exp_avg", "exp_avg_sq"} or buffers["step"].item() != step or
                any(buffers[name].dtype != torch.float32 or buffers[name].shape != param.shape
                    for name in ("exp_avg", "exp_avg_sq"))):
            raise ValueError("AdamW moments or step do not match the model update boundary")


def save_checkpoint(path, model, optimizer, *, identity, step, device, parent_sha256, extra=None):
    """Publish an immutable state by hard-link after flush, without overwriting a concurrent writer."""
    _sha_string(parent_sha256)
    if step == 0 and parent_sha256 != identity["initialization"]["sha256"]:
        raise ValueError("Step-zero parent must be the frozen paired initialization")
    _check_model(model, identity)
    _check_optimizer(model, optimizer, step)
    payload = {"stage": "S4", "format_version": FORMAT, "identity": identity,
               "mode": identity["mode"], "seed": identity["seed"], "step": step, "next_sample_index": step,
               "parent_checkpoint_sha256": parent_sha256, "runtime": _runtime(device), "scheduler": _scheduler(step),
               "model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": capture_rng(device), "extra": extra}
    path = Path(path)
    with NamedTemporaryFile(dir=path.parent, prefix=".s4-checkpoint-", delete=False) as output:
        temp = Path(output.name)
        try:
            torch.save(payload, output)
            output.flush()
            os.fsync(output.fileno())
            digest = hashlib.sha256(temp.read_bytes()).hexdigest()
            os.link(temp, path)
            dirfd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dirfd)
            finally:
                os.close(dirfd)
        finally:
            temp.unlink(missing_ok=True)
    return digest


def _bound_state(path, expected_sha256, bound):
    state = read_checkpoint(path, _sha_string(expected_sha256))
    if (state.get("stage") != "S4" or state.get("format_version") != FORMAT or
            state.get("mode") != bound["mode"] or state.get("seed") != bound["seed"] or state.get("identity") != bound):
        raise ValueError("S4 checkpoint identity, mode, seed or format mismatch")
    _sha_string(state["parent_checkpoint_sha256"])
    step = state["step"]
    if (state["next_sample_index"] != step or state["scheduler"] != _scheduler(step) or
            (step == 0 and state["parent_checkpoint_sha256"] != bound["initialization"]["sha256"])):
        raise ValueError("S4 checkpoint cursor, learning rate or parent mismatch")
    return state


def _model_from_state(state, device, *, bound=None):
    bound = state["identity"] if bound is None else bound
    model = make_model(bound["mode"], evidence_runtime=bound["evidence_runtime"])
    model.load_state_dict(state["model"], strict=True)
    require_exact(state["model"], model.state_dict(), "S4 model")
    _check_model(model, bound)
    return model.to(device)


def _restore_training_state(state, bound, device):
    if state["runtime"] != _runtime(device):
        raise ValueError("Training runtime differs from the saved optimizer boundary")
    model = _model_from_state(state, device, bound=bound).train()
    opt = make_optimizer(model)
    opt.load_state_dict(state["optimizer"])
    require_exact(state["optimizer"], opt.state_dict(), "S4 AdamW")
    _check_optimizer(model, opt, state["step"])
    restore_rng(state["rng"], device)
    require_exact(state["rng"], capture_rng(device), "S4 RNG")
    return model, opt, state


def restore_checkpoint(path, expected_sha256, *, identity, device):
    configure_determinism()
    state = _bound_state(path, expected_sha256, identity)
    return _restore_training_state(state, identity, device)


def restore_chunk_boundary(path, expected_sha256, *, source_identity, target_identity, expected_step, device):
    """Restore a committed parent into a new run's chunk schedule without taking an update.

    The returned state retains the source identity, cursor and journal. Publish the
    new checkpoint separately with the target identity and expected_sha256 as parent.
    """
    if type(expected_step) is not int or not 0 < expected_step <= 36000:
        raise ValueError("Migration requires a committed nonzero optimizer boundary")
    # The new run and its lock are necessarily different. Scientific input,
    # schedule, optimization, coordinate layout and initialized tensors are pinned.
    lineage_keys = {"run_id", "protocol_sha256", "execution_sha256", "dependencies_sha256", "evidence_runtime"}
    left = {k:v for k,v in source_identity.items() if k not in lineage_keys}
    right = {k:v for k,v in target_identity.items() if k not in lineage_keys}
    require_exact(left, right, "S4 migration scientific identity")
    if any(source_identity[k] == target_identity[k] for k in ("run_id", "protocol_sha256", "execution_sha256")):
        raise ValueError("A scheduling migration requires separate run, protocol and execution bindings")
    if source_identity["mode"] == "M0":
        raise ValueError("M0 has no evidence chunk schedule to migrate")
    old_runtime, new_runtime = source_identity["evidence_runtime"], target_identity["evidence_runtime"]
    if old_runtime is None or new_runtime is None or training_chunks(old_runtime) != (64, 8) or training_chunks(new_runtime) != (256, 81):
        raise ValueError("Expected the registered 64/8 to 256/81 evidence migration")
    expected_runtime = dict(old_runtime, query_chunk=256, velocity_chunk=81)
    require_exact(expected_runtime, new_runtime, "S4 migration evidence runtime")
    configure_determinism()
    state = _bound_state(path, expected_sha256, source_identity)
    if state["step"] != expected_step or state["next_sample_index"] != expected_step:
        raise ValueError("Parent checkpoint does not lie at the registered migration step")
    # Validate under the original identity (including the coordinate buffers),
    # then materialize and compare the target model and Adam moments separately.
    original, original_opt, _ = _restore_training_state(state, source_identity, device)
    target, target_opt, _ = _restore_training_state(state, target_identity, device)
    require_exact(original.state_dict(), target.state_dict(), "S4 migration model")
    require_exact(original_opt.state_dict(), target_opt.state_dict(), "S4 migration AdamW")
    return target, target_opt, state


def evaluation_model(path, expected_sha256, *, identity, device):
    """Check the bound model identity on inference without restoring training-process RNGs."""
    return _model_from_state(_bound_state(path, expected_sha256, identity), device).eval()
