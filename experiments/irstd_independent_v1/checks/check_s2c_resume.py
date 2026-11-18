"""Verify the step-400 M0 and compare continuous versus fresh-process updates at step 402."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import torch

EXPERIMENT = Path(__file__).resolve().parents[1]
PROJECT = EXPERIMENT.parents[1]
sys.path.insert(0, str(PROJECT))

from experiments.irstd_independent_v1.checks.check_s2b import write
from experiments.irstd_independent_v1.contracts import CENTER_INDEX
from experiments.irstd_independent_v1.data.s2b_subset import hash_order
from experiments.irstd_independent_v1.engine.checkpoints import (
    capture_rng, read_checkpoint, require_exact, restore_training, save_training,
)
from experiments.irstd_independent_v1.engine.s2b_train import _load_samples, sample_on_device
from experiments.irstd_independent_v1.losses.detection import detection_loss


FROZEN_SHA = "a960e500f7f00da69de273ade47ecc021de386afb7835a9957a1dd7b90cb37de"


def read_json(path):
    return json.loads(path.read_bytes())


def audit_schedule(subset):
    ids = [row["sample_id"] for row in subset["samples"]]
    extended = []
    for epoch in range((402 + len(ids) - 1) // len(ids)):
        extended.extend(sorted(ids, key=lambda sample: (hash_order("epoch", epoch, sample), sample)))
    extended = extended[:402]
    if (subset["seed"] != 46 or extended[:400] != subset["schedule"]
            or extended[400:] != ["Sequence83/00020", "Sequence77/00046"]):
        raise ValueError("Reconstructed audit schedule differs from the frozen protocol")
    return extended


@torch.inference_mode()
def output(model, sample, device, name):
    model.eval()
    clip, _ = sample_on_device(sample, device)
    pred = model(clip)
    if pred.output_frame_id != (name,):
        raise ValueError(f"Output ID mismatch at {name}")
    scores = pred.logits[0, 0].cpu().numpy()
    if (scores.shape != clip.scene_meta[0].original_hw or scores.dtype != np.float32
            or not np.isfinite(scores).all()):
        raise ValueError(f"Invalid native score map at {name}")
    return scores


def check_archived_output(model, samples, device, archive, expected_digest):
    digest = sha256(archive.read_bytes()).hexdigest()
    if digest != expected_digest:
        raise ValueError("Archived logits digest differs from independent endpoint audit")
    sizes = {}
    with np.load(archive, allow_pickle=False) as saved:
        if sorted(saved.files) != sorted(samples):
            raise ValueError("Archived logit IDs differ from the diagnostic subset")
        for name, sample in samples.items():
            actual = output(model, sample, device, name)
            original = saved[name]
            if (original.shape != actual.shape or original.dtype != actual.dtype
                    or original.tobytes() != actual.tobytes()):
                raise ValueError(f"Step-400 native FP32 logits did not reproduce exactly at {name}")
            sizes[name] = actual.size
    return {"archive_sha256": digest, "bitwise_equal": True, "frames": len(sizes),
            "pixels": sum(sizes.values()), "pixels_by_frame": sizes, "max_abs_difference": 0.0}


def advance(model, optimizer, samples, schedule, cursor, device):
    name = schedule[cursor]
    model.train()
    clip, target = sample_on_device(samples[name], device)
    optimizer.zero_grad(set_to_none=True)
    pred = model(clip)
    if pred.output_frame_id != (name,):
        raise ValueError(f"Wrong sample for restart update {cursor + 1}")
    loss = detection_loss(pred, target, clip.pixel_valid[:, CENTER_INDEX])
    if not torch.isfinite(loss).item():
        raise FloatingPointError("Nonfinite restart loss")
    loss.backward()
    gradients = [p.grad for p in model.parameters()]
    if any(g is None for g in gradients) or not torch.isfinite(torch.cat([g.flatten() for g in gradients])).all():
        raise FloatingPointError("Missing or nonfinite restart gradients")
    optimizer.step()
    for param in model.parameters():
        moment = optimizer.state[param]
        if (moment["step"].item() != cursor + 1 or not all(torch.isfinite(x).all().item()
                for x in (param, moment["exp_avg"], moment["exp_avg_sq"]))):
            raise FloatingPointError("Invalid post-update AdamW state")
    return {"logical_step": cursor + 1, "sample_id": name, "loss": loss.item()}


def save_boundary(path, model, optimizer, state, schedule, cursor, parent, device):
    return save_training(path, model, optimizer, device=device, global_step=cursor,
                         next_sample_index=cursor, schedule=schedule, training_config=state["training_config"],
                         dependencies=state["dependencies"], parent_sha256=parent, audit_branch="resume_check")


def finish_branch(folder, label, model, optimizer, state, samples, schedule, parent, device):
    # A continues with its in-memory model and moments. Only B reloads the boundary.
    row = advance(model, optimizer, samples, schedule, 401, device)
    name = row["sample_id"]
    scores = output(model, samples[name], device, name)
    with (folder / f"{label}_402_logits.npy").open("xb") as stream:
        np.save(stream, scores, allow_pickle=False)
    digest = save_boundary(folder / f"{label}_402.pt", model, optimizer, state, schedule, 402, parent, device)
    return {**row, "pid": os.getpid(), "checkpoint_sha256": digest}


def branch_b(folder, device):
    protocol = read_json(folder / "protocol.json")
    subset = read_json(EXPERIMENT / "manifests/s2b_train12.json")
    samples = _load_samples(PROJECT, EXPERIMENT, subset)
    # Perturb every RNG before loading. An omitted restore must fail even with deterministic M0.
    random.seed(402)
    np.random.seed(402)
    torch.manual_seed(402)
    before = capture_rng(device)
    middle = read_json(folder / "boundary_401.json")
    model, optimizer, state = restore_training(folder / "boundary_401.pt", middle["sha256"], EXPERIMENT, device)
    if ((state["global_step"], state["next_sample_index"], state["parent_checkpoint_sha256"]) != (401, 401, FROZEN_SHA)
            or state["schedule"] != protocol["audit_schedule"] or state["audit_branch"] != "resume_check"):
        raise ValueError("Invalid branch-B restart boundary")
    restored = capture_rng(device)
    require_exact(state["rng"], restored, "process_B.live_rng_after_restore")
    domains = {"python": before["python"] != restored["python"], "numpy": before["numpy"] != restored["numpy"],
               "torch_cpu": not torch.equal(before["torch_cpu"], restored["torch_cpu"]),
               "torch_cuda_devices": [not torch.equal(a, b) for a, b in zip(before["torch_cuda"], restored["torch_cuda"])]}
    if not (domains["python"] and domains["numpy"] and domains["torch_cpu"] and all(domains["torch_cuda_devices"])):
        raise ValueError("The RNG negative control did not distinguish all generators")
    row = finish_branch(folder, "B", model, optimizer, state, samples, state["schedule"], middle["sha256"], device)
    write(folder / "branch_B.json", {**row, "rng_negative_control": domains, "rng_restored_exactly": True})


def run_resume(run_dir, device):
    device = torch.device(device)
    folder = Path(run_dir).resolve() / "resume"
    folder.mkdir(exist_ok=False)
    subset = read_json(EXPERIMENT / "manifests/s2b_train12.json")
    prior = read_json(EXPERIMENT / "reports/S2b.json")
    archive = EXPERIMENT / prior["run_dir"]
    if (prior["status"], prior["steps_completed"], prior["checkpoint"]["sha256"]) != ("PASS", 400, FROZEN_SHA):
        raise ValueError("S2b did not pass with the frozen step-400 checkpoint")
    schedule = audit_schedule(subset)
    protocol = {"frozen_sha256": FROZEN_SHA, "main_fit_updates": 400, "audit_schedule": schedule,
                "audit_samples": schedule[400:], "planned_updates": {"A": 2, "B": 1},
                "restart_comparison": "exact native FP32, all model/AdamW state, RNG, schedule, cursor"}
    write(folder / "protocol.json", protocol)
    samples = _load_samples(PROJECT, EXPERIMENT, subset)
    model, optimizer, state = restore_training(archive / prior["checkpoint"]["path"], FROZEN_SHA, EXPERIMENT, device)
    if ((state["global_step"], state["next_sample_index"], state["model_config"]["trainable_parameters"]) != (400, 400, 22921)
            or state["schedule"] != schedule[:400] or state["training_config"] != prior["training_config"]):
        raise ValueError("Restored step-400 state disagrees with its report")
    archived = read_json(archive / "endpoint_audit.json")
    if archived["status"] != "PASS":
        raise ValueError("Missing independent verification of step-400 logits")
    original_logits = check_archived_output(model, samples, device, archive / "logits_400.npz",
                                             archived["steps"]["400"]["logits_sha256"])
    step401 = advance(model, optimizer, samples, schedule, state["next_sample_index"], device)
    middle = save_boundary(folder / "boundary_401.pt", model, optimizer, state, schedule, 401, FROZEN_SHA, device)
    write(folder / "boundary_401.json", {**step401, "sha256": middle})
    a = finish_branch(folder, "A", model, optimizer, state, samples, schedule, middle, device)
    write(folder / "branch_A.json", a)
    del model, optimizer, state, samples
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "--run-dir", str(folder.parent),
                    "--device", str(device), "--branch-b"], check=True)
    b = read_json(folder / "branch_B.json")
    if a["pid"] == b["pid"]:
        raise ValueError("Branch B did not execute in a fresh process")
    for key in ("logical_step", "sample_id", "loss"):
        require_exact(a[key], b[key], f"402.{key}")
    saved_a = read_checkpoint(folder / "A_402.pt", a["checkpoint_sha256"])
    saved_b = read_checkpoint(folder / "B_402.pt", b["checkpoint_sha256"])
    require_exact(saved_a, saved_b, "402.checkpoint")
    a_scores = np.load(folder / "A_402_logits.npy", allow_pickle=False)
    b_scores = np.load(folder / "B_402_logits.npy", allow_pickle=False)
    if (a_scores.dtype != b_scores.dtype or a_scores.shape != b_scores.shape or a_scores.tobytes() != b_scores.tobytes()):
        raise ValueError("A/B updated-model outputs differ")
    report = {"status": "PASS", "frozen_checkpoint_sha256": FROZEN_SHA, "original_logits": original_logits,
              "step_401": {**step401, "checkpoint_sha256": middle}, "step_402": {"A": a, "B": b},
              "logical_end_step": 402, "next_sample_index": saved_a["next_sample_index"], "audit_updates": 3,
              "model_state_tensors": len(saved_a["model"]), "adam_parameter_states": len(saved_a["optimizer"]["state"]),
              "all_model_optimizer_rng_metadata_exact": True, "step_402_eval_logits_exact": True,
              "step_402_pixels_compared": int(a_scores.size), "step_402_loss_exact": True}
    write(folder / "result.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--branch-b", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.branch_b:
        branch_b(args.run_dir / "resume", torch.device(args.device))
    else:
        result = run_resume(args.run_dir, args.device)
        print(f"S2c-1 {result['status']}; audit updates {result['audit_updates']}", flush=True)


if __name__ == "__main__":
    main()
