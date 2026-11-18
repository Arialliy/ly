"""Train one frozen S4 model on paired native cores and resume at committed update boundaries."""
import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
from statistics import mean
import tempfile
from time import perf_counter

import numpy as np
import torch

from ..checks.check_s0 import atomic_text
from ..data.dataset import MaskClipDataset
from ..losses.s4_detection import s4_detection_loss
from .checkpoints import capture_rng, configure_determinism, require_exact
from .s2b_train import sample_on_device
from .s4_checkpoints import evaluation_model, identity, lr_for_update, make_optimizer, restore_checkpoint, save_checkpoint
from .s4_models import core_prediction, initial_model


EXPERIMENT = Path(__file__).resolve().parents[1]
PROJECT = EXPERIMENT.parents[1]
RECOVERY_INTERVAL = 300
MIB = 1024 ** 2
ROLES = ("positive", "random_background", "empty_frame")


def _check(condition, detail):
    if not condition:
        raise ValueError(detail)


def _read(path):
    return json.loads(path.read_bytes())


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")


def _stamp():
    return datetime.now(timezone.utc).isoformat()


def _configure(seed):
    configure_determinism()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _group_parameters(model):
    groups = defaultdict(list)
    for name, param in model.named_parameters():
        _check(param.requires_grad and param.dtype == torch.float32, f"Expected trainable FP32 tensor {name}")
        parts = name.split(".")
        if parts[0] == "video_aggregator":
            depth = 3 if parts[1] == "encoder" else 2
            groups[".".join(parts[:depth])].append(param)
        else:
            groups[parts[0]].append(param)
    return groups


def _apply_row(model, dataset, lookup, row, device):
    key = (row["source"], row["center"])
    clip, targets = sample_on_device(dataset[lookup[key]], device)
    box = tuple(row["core_xyxy"])
    x0,y0,x1,y1 = box
    _check(x1-x0 == y1-y0 == 64 and clip.output_frame_id == (f"{key[0]}/{key[1]:05d}",),
           f"Wrong native crop or output identity at {key}")
    role = row["sample_role"]
    _check(role in ROLES and role == row["kind"], "Unknown scheduled exposure")
    boxes = [obj.bbox_xyxy for obj in targets.objects[0]]
    if role == "positive":
        _check(boxes and all(x0 <= x and y0 <= y and u <= x1 and v <= y1 for x,y,u,v in boxes),
               f"Scheduled positive misses part of its native target: {key}")
    if role == "empty_frame":
        _check(not boxes, f"Nominal empty frame contains foreground: {key}")
    prediction = core_prediction(model, clip, box)
    _check(prediction.output_frame_id == clip.output_frame_id, "Supervision and prediction centers disagree")
    loss, parts = s4_detection_loss(prediction, targets, clip.pixel_valid[:, clip.center_index], box)
    _check(bool(parts["foreground_pixels"]) == (role == "positive"), f"Core label contradicts role at {key}")
    if not math.isfinite(parts["total"]):
        raise FloatingPointError(f"Nonfinite core loss for {key}")
    return loss, parts


def _optimize(model, optimizer, groups, dataset, lookup, row, device):
    start = perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    step = row["step"]
    lr = lr_for_update(step)
    for group in optimizer.param_groups:
        group["lr"] = lr
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, parts = _apply_row(model, dataset, lookup, row, device)
    loss.backward()
    snapshots = {}
    grad_norms = {}
    for module, parameters in groups.items():
        if any(p.grad is None for p in parameters):
            raise FloatingPointError(f"Disconnected gradient at step {step}, {module}")
        grads = torch.cat([p.grad.detach().reshape(-1) for p in parameters])
        grad_norms[module] = torch.linalg.vector_norm(grads.double()).item()
        if not math.isfinite(grad_norms[module]):
            raise FloatingPointError(f"Nonfinite gradient at step {step}, {module}")
        snapshots[module] = torch.cat([p.detach().reshape(-1) for p in parameters])
    optimizer.step()
    delta_norms = {}
    for module, parameters in groups.items():
        new = torch.cat([p.detach().reshape(-1) for p in parameters])
        delta_norms[module] = torch.linalg.vector_norm(new.double() - snapshots[module]).item()
        moments = torch.cat([optimizer.state[p][key].reshape(-1)
                             for p in parameters for key in ("exp_avg","exp_avg_sq")])
        if not math.isfinite(delta_norms[module]) or not torch.isfinite(moments).all().item():
            raise FloatingPointError(f"Nonfinite parameter or AdamW state at step {step}, {module}")
    peak = torch.cuda.max_memory_allocated(device) / MIB if device.type == "cuda" else None
    if peak is not None and peak > 8192:
        raise MemoryError(f"Allocated {peak:.1f} MiB in update {step}, above the 8192 MiB limit")
    return {**row, "learning_rate": lr, **parts, "gradient_l2_by_module": grad_norms,
            "parameter_update_l2_by_module": delta_norms, "finite_after_update": True,
            "peak_allocated_mib": peak, "seconds": perf_counter()-start}


@torch.inference_mode()
def _diagnostics(model, dataset, lookup, rows, device):
    start = perf_counter()
    model.eval()
    by_core = []
    for row in rows:
        _, detail = _apply_row(model, dataset, lookup, row, device)
        by_core.append({"source": row["source"], "center": row["center"], "core_xyxy": row["core_xyxy"],
                        "role": row["sample_role"], **detail})
    terms = {}
    for k in ("foreground_bce", "background_bce_all", "background_bce_hard", "background_bce", "dice_loss"):
        values = [r[k] for r in by_core if r[k] is not None]
        terms[k] = mean(values) if values else None
    return {"loss": mean(r["total"] for r in by_core), **terms, "by_core": by_core,
            "cores": len(by_core), "finite": True, "seconds": perf_counter()-start}


def _state_path(folder, step):
    return folder / "checkpoints" / f"step_{step:06d}.pt"


def _preserve_orphan(path):
    """Keep interrupted bytes/checkpoints for inspection without accepting them as committed history."""
    backup = path.parent / "interrupted"
    backup.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return backup / f"{path.name}.{stamp}"


class UpdateJournal:
    """The checkpoint binds the exact committed prefix; keep a crash tail before replay."""
    def __init__(self, folder, checkpoint_extra, cursor, schedule):
        self.path = folder / "updates.jsonl"
        self.digest = hashlib.sha256()
        self.counts = Counter()
        self.exposures = {role: Counter() for role in ROLES}
        self.edges = 0
        self.peak = None
        self.last_update = None
        self.scheduled = schedule
        self.seconds = 0.
        self.file = self.path.open("a+b")
        try:
            self.file.seek(0)
            for _ in range(cursor):
                data = self.file.readline()
                _check(data.endswith(b"\n"), "Committed update log is incomplete")
                self._track(json.loads(data), data)
            offset = self.file.tell()
            if cursor or checkpoint_extra:
                _check(self.prefix() == checkpoint_extra["journal"], "Log diverges from committed checkpoint")
            if self.file.seek(0, os.SEEK_END) != offset:
                self.file.seek(offset)
                with _preserve_orphan(self.path).open("xb") as orphan:
                    for block in iter(lambda: self.file.read(1 << 20), b""):
                        orphan.write(block)
                    orphan.flush()
                    os.fsync(orphan.fileno())
                self.file.truncate(offset)
                self.sync()
            self.file.seek(0, os.SEEK_END)
        except BaseException:
            self.file.close()
            raise

    def _track(self, row, data):
        step = self.last_update["step"] + 1 if self.last_update else 1
        expected = self.scheduled[step-1]
        _check(expected["step"] == step and all(row.get(k) == v for k,v in expected.items())
               and row["finite_after_update"] is True, f"Unpaired update journal at step {step}")
        self.digest.update(data)
        role = row["sample_role"]
        self.counts[role] += 1
        self.exposures[role][f"{row['source']}/{row['center']:05d}"] += 1
        self.edges += int(row["edge_priority"])
        self.seconds += row["seconds"]
        peak = row["peak_allocated_mib"]
        if peak is not None:
            self.peak = max(peak, self.peak or 0)
        self.last_update = row

    def append(self, row):
        line = (json.dumps(row, allow_nan=False, separators=(",",":")) + "\n").encode()
        self._track(row, line)
        self.file.write(line)
        self.file.flush()

    def prefix(self):
        return {"updates": self.last_update["step"] if self.last_update else 0, "sha256": self.digest.hexdigest()}

    def sync(self):
        self.file.flush()
        os.fsync(self.file.fileno())

    def exposure(self):
        return {"role_counts": dict(self.counts), "center_exposures": self.exposures, "edge_priority_slots": self.edges}


@contextmanager
def _journal(folder, extra, step, schedule):
    log = UpdateJournal(folder, extra, step, schedule)
    try:
        yield log
    finally:
        log.file.close()


def _save_state(folder, model, optimizer, context, journal, step, device, parent_sha):
    path = _state_path(folder, step)
    journal.sync()
    if path.exists() and not path.with_suffix(".json").exists():
        path.rename(_preserve_orphan(path))
    checksum = save_checkpoint(path, model, optimizer, identity=context, step=step, device=device,
                               parent_sha256=parent_sha, extra={"journal": journal.prefix()})
    _write(path.with_suffix(".json"), {"step": step, "checkpoint_sha256": checksum,
                                      "parent_checkpoint_sha256": parent_sha, "identity": context})
    return checksum


def _latest(folder, context, target, initial_sha):
    json_context = json.loads(json.dumps(context))
    previous_sha = initial_sha
    latest = None
    for sidecar in sorted((folder/"checkpoints").glob("step_*.json")):
        saved = _read(sidecar)
        n = saved["step"]
        _check(sidecar == _state_path(folder,n).with_suffix(".json") and saved["identity"] == json_context
               and saved["parent_checkpoint_sha256"] == previous_sha, f"Checkpoint chain diverged at {sidecar}")
        if latest is None:
            _check(n == 0, "Missing initial checkpoint in archive")
        else:
            previous_step = latest["step"]
            _check(n == (8 if previous_step == 0 else (previous_step//RECOVERY_INTERVAL+1)*RECOVERY_INTERVAL),
                   f"Skipped or unordered checkpoint between {previous_step} and {n}")
        previous_sha = saved["checkpoint_sha256"]
        latest = saved
        _check(n <= target, "Requested endpoint precedes the newest checkpoint")
    return latest


def _snapshot(run, p, model, mode, seed, step, checksum, folder, dataset, lookup, device, journal):
    if step % 3000:
        return
    record = {"step": step, "mode": mode, "seed": seed, "protocol_sha256": _sha(run/"protocol.json"),
              "checkpoint_sha256": checksum, "diagnostic_sha256": p["diagnostics_sha256"]["train_diagnostics.json"]}
    path = folder/f"diagnostics_{step}.json"
    if path.exists():
        existing = _read(path)
        _check(all(existing.get(k) == v for k,v in record.items()) and existing["fit"]["cores"] == 144,
               "Existing training diagnostic differs from frozen input")
    else:
        _write(folder/"progress.json", {**record, "phase":"train_diagnostics", "utc":_stamp()})
        fit = _diagnostics(model, dataset, lookup, p["fit_centers"], device)
        _check(fit["cores"] == 144, "Incorrect fixed training panel")
        last = journal.last_update
        _write(path, {**record, "fit": fit, "last_update_gradient_l2_by_module":
                      last["gradient_l2_by_module"] if last else None,
                      "last_parameter_update_l2_by_module": last["parameter_update_l2_by_module"] if last else None})
    if step in (6000,18000,30000,12000,24000,36000):
        from ..evaluation.s4_predict import evaluate_checkpoint
        population = "development" if step % 12000 == 0 else "development_monitor"
        _write(folder/"progress.json", {**record, "phase":population, "utc":_stamp()})
        evaluate_checkpoint(run, p, model, mode, seed, step, population=population, checkpoint_sha256=checksum)


def _verify_resume(model, optimizer, ckpt, checksum, context, device, step, prefix):
    before_rng = capture_rng(device)
    restored, opt, state = restore_checkpoint(ckpt, checksum, identity=context, device=device)
    require_exact(model.state_dict(), restored.state_dict(), "model")
    require_exact(optimizer.state_dict(), opt.state_dict(), "optimizer")
    require_exact(before_rng, capture_rng(device), "rng")
    _check(state["step"] == state["next_sample_index"] == step and state["extra"]["journal"] == prefix,
           "Round-trip update cursor or journal mismatch")
    return restored, opt


def _receipt(run, p, folder, mode, seed, step, checksum, context, journal, schedule, resumed):
    exposure = journal.exposure()
    if str(step) in schedule["prefix_exposure"]:
        _check(exposure == schedule["prefix_exposure"][str(step)], "Actual training exposure differs from design")
    receipt = {"step": step, "mode": mode, "seed": seed, "protocol_sha256": _sha(run/"protocol.json"),
               "execution_binding_sha256":_sha(run/"execution_binding.json"), "checkpoint_sha256": checksum,
               "schedule_sha256": p["schedule_sha256"][str(seed)], "identity": context,
               "journal": journal.prefix(), "role_counts": exposure["role_counts"], "gradient_finite": True,
               "peak_allocated_mib": journal.peak, "update_seconds_total": journal.seconds,
               "resume_verified": resumed, "host_gpu_index": os.environ.get("S2_GPU_INDEX"),
               "gpu_uuid": os.environ.get("S2_GPU_UUID"), "image": os.environ.get("S2_IMAGE_ID")}
    _write(folder/f"exposure_{step}.json", exposure)
    receipt["actual_exposure_sha256"] = _sha(folder/f"exposure_{step}.json")
    _write(folder/f"training_{step}.json", receipt)
    return receipt


def train(run, p, folder, mode, seed, target, context, device):
    """Load only training images for updates and fixed-core diagnostics; development is explicit at milestones."""
    folder.joinpath("checkpoints").mkdir(exist_ok=True)
    _check(target in (8,24000,36000) and target <= p["max_steps"], "Unknown training endpoint")
    completed = folder/f"training_{target}.json"
    if completed.exists():
        receipt = _read(completed)
        _check((receipt["step"],receipt["mode"],receipt["seed"]) == (target,mode,seed)
               and receipt["identity"] == json.loads(json.dumps(context)) and receipt["resume_verified"] is True
               and receipt["checkpoint_sha256"] == _sha(_state_path(folder,target))
               and receipt["actual_exposure_sha256"] == _sha(folder/f"exposure_{target}.json"),
               f"Training receipt is inconsistent: {completed}")
        return
    if target == 36000:
        budget = _read(run/"budget_decision.json")
        _check(budget["selected_step"] == 36000 and budget["protocol_sha256"] == _sha(run/"protocol.json"),
               "Missing common 36k extension decision")
    schedule_path = run/f"schedule_{seed}.json"
    _check(_sha(schedule_path) == p["schedule_sha256"][str(seed)], "Changed paired schedule")
    schedule = _read(schedule_path)
    _check(schedule["seed"] == seed and len(schedule["updates"]) == schedule["steps"] == p["max_steps"],
           "Incomplete frozen update schedule")
    manifest = EXPERIMENT/p["proposal"]["data"]["manifest"]
    _check(_sha(manifest) == p["proposal"]["data"]["manifest_sha256"], "Changed S0 manifest")
    dataset = MaskClipDataset(_read(manifest), PROJECT, "train")
    lookup = {v:i for i,v in enumerate(dataset.centers)}
    _check(len(lookup) == len(dataset) == 6048, "Unexpected S4 training population")
    last_sha = p["initialization"][str(seed)]["sha256"]
    latest = _latest(folder,context,target,last_sha)
    extra = None
    if latest:
        cursor, last_sha = latest["step"], latest["checkpoint_sha256"]
        model, optimizer, state = restore_checkpoint(_state_path(folder,cursor), last_sha, identity=context, device=device)
        _check(state["step"] == state["next_sample_index"] == cursor, "Wrong restored update cursor")
        _check(state["parent_checkpoint_sha256"] == latest["parent_checkpoint_sha256"], "Wrong checkpoint parent")
        extra = state["extra"]
    else:
        cursor = 0
        model = initial_model(run,p,mode,seed,device)
        optimizer = make_optimizer(model)
    with _journal(folder, extra, cursor, schedule["updates"]) as journal:
        if latest is None:
            last_sha = _save_state(folder,model,optimizer,context,journal,0,device,last_sha)
        groups = _group_parameters(model)
        for step in range(cursor, target+1):
            if step > cursor:
                row = _optimize(model,optimizer,groups,dataset,lookup,schedule["updates"][step-1],device)
                journal.append(row)
                if step <= 8 or step % 25 == 0:
                    _write(folder/"progress.json", {"phase":"training", "mode":mode, "seed":seed,"step":step,
                           "target":target, "utc":_stamp(), "last_update":row})
            checkpoint_due = step == 8 or step % RECOVERY_INTERVAL == 0
            if checkpoint_due and step > cursor:
                last_sha = _save_state(folder,model,optimizer,context,journal,step,device,last_sha)
            if checkpoint_due:
                _snapshot(run,p,model,mode,seed,step,last_sha,folder,dataset,lookup,device,journal)
            if step == target or (step == 8 and not (folder/"training_8.json").exists()):
                model, optimizer = _verify_resume(model,optimizer,_state_path(folder,step),last_sha,context,device,
                                                  step,journal.prefix())
                groups = _group_parameters(model)
                _receipt(run,p,folder,mode,seed,step,last_sha,context,journal,schedule,True)
                print(f"S4 {mode} seed {seed}: committed {step} paired updates, state round-trip verified",flush=True)
        _write(folder/"progress.json", {"phase":"training_target_complete","mode":mode,"seed":seed,"step":target,
                                       "checkpoint_sha256":last_sha,"utc":_stamp()})


def worker(run, action, mode, seed, step):
    from ..checks.s4_binding import verify_execution
    run = Path(run).resolve()
    p, binding = verify_execution(run)
    _check(p["seeds"] == [46] and seed == 46 and mode in p["modes"] and run.name == p["run_id"],
           "Job is outside the fixed-seed research protocol")
    _check(torch.cuda.is_available() and torch.cuda.device_count() == 1 and
           os.environ.get("S2_GPU_INDEX") in ("1","2","3") and
           os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", "Expected one authorized CUDA device")
    _configure(seed)
    device = torch.device("cuda:0")
    context = identity(p,mode,seed,protocol_sha256=binding["common_protocol_sha256"],
                       execution_sha256=_sha(run/"execution_binding.json"))
    folder = run/mode/f"seed{seed}"
    folder.mkdir(exist_ok=True,parents=True)
    with (folder/"worker.lock").open("a+b") as handle:
        fcntl.flock(handle,fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if action == "train":
                train(run,p,folder,mode,seed,step,context,device)
            else:
                from ..evaluation.s4_predict import evaluate_checkpoint
                lock = _read(run/"historical_lock.json")
                _check(step == lock["selected_step"] and lock["protocol_sha256"] == binding["common_protocol_sha256"],
                       "Historical access requires all four common endpoints and development thresholds")
                checksum = lock["runs"][mode][str(seed)]["checkpoint_sha256"]
                _check(checksum == _read(folder/f"training_{step}.json")["checkpoint_sha256"], "Endpoint changed")
                model = evaluation_model(_state_path(folder,step), checksum, identity=context, device=device)
                evaluate_checkpoint(run,p,model,mode,seed,step,population="historical_test",checkpoint_sha256=checksum)
        except BaseException as exc:
            with tempfile.NamedTemporaryFile(prefix="failure_",suffix=".json",dir=folder,delete=False) as handle:
                path = Path(handle.name)
            _write(path, {"utc":_stamp(), "action":action, "mode":mode,"seed":seed,"target":step,
                          "error_type":type(exc).__name__, "error":str(exc)})
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir",type=Path)
    parser.add_argument("action",choices=("train","historical"))
    parser.add_argument("mode",choices=("M0","Msep","Mraw","Mint"))
    parser.add_argument("seed",type=int,choices=(46,))
    parser.add_argument("step",type=int,choices=(8,24000,36000))
    args = parser.parse_args()
    worker(args.run_dir,args.action,args.mode,args.seed,args.step)
