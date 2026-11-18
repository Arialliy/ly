"""Gate the S3 representation, then run one predeclared four-update training check."""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
from time import perf_counter
import traceback

import numpy as np
import torch
import yaml

EXPERIMENT = Path(__file__).resolve().parents[1]
PROJECT = EXPERIMENT.parents[1]
sys.path.insert(0, str(PROJECT))

from experiments.irstd_independent_v1.checks.check_s0 import repository_commit
from experiments.irstd_independent_v1.checks.check_s2b import write
from experiments.irstd_independent_v1.checks import s3_parts_check, s3_structure_check
from experiments.irstd_independent_v1.contracts import CENTER_INDEX
from experiments.irstd_independent_v1.data.dataset import MaskClipDataset, _components
from experiments.irstd_independent_v1.engine.checkpoints import configure_determinism
from experiments.irstd_independent_v1.engine.s2b_train import sample_on_device
from experiments.irstd_independent_v1.losses.detection import detection_loss
from experiments.irstd_independent_v1.models.hypothesis_detector import HypothesisEvidenceDetector

REPORTS = EXPERIMENT / "reports"
SEED = 46
# x and y are native image coordinates. The positive cores contain the entire
# native annotation; the empty cores test an interior and a corner region.
SCHEDULE = (
    ("Sequence82", 43, (64, 64, 128, 128), True),
    ("Sequence17", 9, (64, 64, 128, 128), False),
    ("Sequence6", 59, (128, 64, 192, 128), True),
    ("Sequence17", 78, (0, 0, 64, 64), False),
)
CODE = ("models/evidence_parts.py", "models/evidence_encoder.py", "models/hypothesis_detector.py",
        "checks/check_s3.py", "checks/s3_parts_check.py", "checks/s3_structure_check.py")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def prerequisites():
    s2 = read(REPORTS / "S2.json")
    require((s2["status"], s2["next_allowed_stage"]) == ("PASS", "S3"), "S2 prerequisite failed")
    bound = {}
    for entry in s2["substage_reports"].values():
        path = EXPERIMENT / entry["path"]
        require(read(path)["status"] == "PASS" and digest(path) == entry["sha256"], f"Stale report {path}")
        bound[entry["path"]] = entry["sha256"]
    require(digest(EXPERIMENT / s2["checkpoint"]) == s2["checkpoint_sha256"], "S2 fit checkpoint mismatch")
    for name in ("S0", "S1", "S2a", "S2b"):
        previous = read(REPORTS / f"{name}.json")
        for file, expected in previous.get("code_sha256", previous.get("code_and_config_sha256")).items():
            require(digest(EXPERIMENT / file) == expected, f"Prior implementation changed: {file}")
    config_file = EXPERIMENT / "configs/s1_formal.yaml"
    config = yaml.safe_load(config_file.read_bytes())
    bound["configs/s1_formal.yaml"] = digest(config_file)
    bound["reports/S2.json"] = digest(REPORTS / "S2.json")
    manifest_path = EXPERIMENT / "manifests/nudt_mirsdt_mask17.json"
    bound["manifests/nudt_mirsdt_mask17.json"] = digest(manifest_path)
    require(bound["manifests/nudt_mirsdt_mask17.json"] == read(REPORTS / "S0.json")["data_manifest_hash"],
            "Dataset manifest mismatch")
    return bound, config


def seed_model():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    configure_determinism()


def prepare_samples(device):
    manifest = read(EXPERIMENT / "manifests/nudt_mirsdt_mask17.json")
    data = MaskClipDataset(manifest, PROJECT, "train")
    using = {row[0] for row in SCHEDULE}
    for seq in (s for s in manifest["sequences"] if s["id"] in using):
        require(seq["split"] == "train", "Diagnostic sample outside training split")
        for role in ("image", "mask"):
            h = hashlib.sha256()
            for frame in seq["frames"]:
                h.update((digest(PROJECT / frame[role]) + "\n").encode("ascii"))
            require(h.hexdigest() == seq[role + "_files_sha256"], f"{seq['id']} {role} files changed")
    for name, number, box, positive in SCHEDULE:
        clip, targets = data[data.centers.index((name, number))]
        x0, y0, x1, y1 = box
        native_mask = targets.target_map[..., y0:y1, x0:x1].contiguous()
        fg = int(native_mask.sum())
        require(bool(fg) == positive and fg == int(targets.target_map.sum()), "Fixed core label changed")
        objects = _components(native_mask[0, 0].numpy().astype(np.uint8))
        targets = replace(targets, target_map=native_mask, ignore_map=targets.ignore_map[..., y0:y1, x0:x1],
                          objects=(objects,))
        clip, targets = sample_on_device((clip, targets), device)
        yield clip, targets, box, fg


def parameter_groups(model):
    e = model.video_aggregator.encoder
    modules = {"frame_encoder": model.frame_encoder, "amplitude": e.amplitude, "temporal": e.temporal,
               "velocity": e.velocity, "joint": e.joint, "fusion": e.fusion,
               "spatial_decoder": model.spatial_decoder, "detection_head": model.detection_head}
    groups = {k: tuple(m.parameters()) for k, m in modules.items()}
    grouped = [id(p) for ps in groups.values() for p in ps]
    require(len(grouped) == len(set(grouped)) and set(grouped) == {id(p) for p in model.parameters()},
            "Gradient diagnostics do not cover exactly all trainable parameters")
    return groups


def vector(parameters):
    return torch.cat([p.detach().flatten() for p in parameters])


def real_chain(folder, device, config):
    seed_model()
    model = HypothesisEvidenceDetector(mode="interaction").to(device).train()
    groups = parameter_groups(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001, foreach=False)
    budget = config["resource"]["max_peak_cuda_allocated_mib"]
    records = []
    for step, (clip, targets, box, fg) in enumerate(prepare_samples(device), 1):
        tile = next(t for t in model.tiles(clip.scene_meta[0].original_hw, (64, 64)) if t.core_xyxy == box)
        x0, y0, x1, y1 = box
        qx0, qy0, qx1, qy1 = tile.query_xyxy
        qhw = [qy1 - qy0, qx1 - qx0]
        if step == 2:
            require(qhw == config["resource"]["query_hw"], "Primary measured core needs full decoder halo")
        optimizer.zero_grad(set_to_none=True)
        before = {k: vector(ps).cpu() for k, ps in groups.items()}
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start = perf_counter()
        prediction = model.forward_core(clip, tile)
        require(prediction.output_frame_id == clip.output_frame_id, "Core prediction ID mismatch")
        valid = clip.pixel_valid[:, CENTER_INDEX, :, y0:y1, x0:x1]
        loss = detection_loss(prediction, targets, valid)
        torch.cuda.synchronize(device)
        forward = perf_counter()
        loss.backward()
        torch.cuda.synchronize(device)
        backward = perf_counter()
        optimizer.step()
        torch.cuda.synchronize(device)
        end = perf_counter()
        peak = torch.cuda.max_memory_allocated(device) / 2**20
        reserved = torch.cuda.max_memory_reserved(device) / 2**20
        metrics = {}
        for key, params in groups.items():
            require(all(p.grad is not None for p in params), f"Disconnected {key}")
            grad = torch.cat([p.grad.detach().flatten() for p in params])
            norm = float(torch.linalg.vector_norm(grad.double()))
            delta = float(torch.linalg.vector_norm(vector(params).cpu().double() - before[key].double()))
            require(math.isfinite(norm) and math.isfinite(delta), f"Nonfinite state for {key}")
            metrics[key] = {"gradient_l2": norm, "update_l2": delta}
        for param in model.parameters():
            adam = optimizer.state[param]
            require(int(adam["step"].item()) == step and torch.isfinite(adam["exp_avg"]).all()
                    and torch.isfinite(adam["exp_avg_sq"]).all(), "Invalid AdamW state")
        require(math.isfinite(loss.item()), "Nonfinite detection loss")
        row = {"step": step, "frame_id": clip.output_frame_id[0], "core_xyxy": box, "query_hw": qhw,
               "input_xyxy": tile.input_xyxy, "foreground_pixels": fg, "loss": loss.item(), "modules": metrics,
               "peak_allocated_mib": peak, "peak_reserved_mib": reserved,
               "seconds": {"forward_and_loss": forward-start, "backward": backward-forward,
                           "optimizer": end-backward, "total": end-start}}
        records.append(row)
        write(folder / "updates.json", records)
        print(f"S3 update {step}/{len(SCHEDULE)}: {row['frame_id']} loss={row['loss']:.6f} "
              f"peak={peak:.1f} MiB elapsed={end-start:.2f}s", flush=True)
        require(peak <= budget, f"Measured full update exceeds {budget} MiB")
        del prediction, loss, clip, targets, grad
    for group in groups:
        require(any(r["foreground_pixels"] and r["modules"][group]["gradient_l2"] > 0 and
                    r["modules"][group]["update_l2"] > 0 for r in records), f"No positive-example update: {group}")
    with (folder / "interaction_step4.pt").open("xb") as file:
        torch.save({"mode": "interaction", "seed": SEED, "optimizer_updates": len(records),
                    "model": model.state_dict()}, file)
    return {"optimizer_updates": len(records), "parameters": sum(p.numel() for p in model.parameters()),
            "updates": records, "checkpoint_sha256": digest(folder / "interaction_step4.pt"),
            "primary_measured_step": 2, "peak_over_all_updates_mib": max(r["peak_allocated_mib"] for r in records),
            "allocated_budget_mib": budget}


def execute(folder, device):
    bound, config = prerequisites()
    folder.mkdir(parents=True, exist_ok=False)
    protocol = {"seed": SEED, "mode": "interaction", "schedule": SCHEDULE, "updates": 4,
                "optimizer": "AdamW", "lr": 0.001, "weight_decay": 0.0001, "precision": "FP32",
                "batch": 1, "query_chunk": 64, "velocity_chunk": 8, "checkpoint_chunks": True,
                "core_hw": [64, 64], "primary_resource_step": 2, "allocated_budget_mib": config["resource"]["max_peak_cuda_allocated_mib"],
                "atol": 1e-5, "rtol": 1e-4, "time_limit": None, "stage_order": ["decomposition", "structure", "training"]}
    state = {"stage":"S3", "status":"RUNNING", "next_allowed_stage":None, "start_utc":datetime.now(timezone.utc).isoformat(),
             "run_dir":str(folder.relative_to(EXPERIMENT)), "protocol":protocol, "prerequisites":bound,
             "code_sha256":{name:digest(EXPERIMENT/name) for name in CODE}, "base_commit":repository_commit(PROJECT),
             "environment":{"image":os.environ["S2_IMAGE_ID"], "gpu":torch.cuda.get_device_name(device),
                            "gpu_uuid":os.environ["S2_GPU_UUID"], "gpu_index":os.environ["S2_GPU_INDEX"],
                            "torch":str(torch.__version__), "cuda":torch.version.cuda}}
    write(folder / "protocol.json", state)
    write(REPORTS / "S3.json", state)
    try:
        seed_model()
        for key, module in [("decomposition", s3_parts_check), ("structure", s3_structure_check)]:
            state[key] = module.run_checks(device)
            write(folder / (key + ".json"), state[key])
            print(f"S3 {key} checks passed", flush=True)
        torch.cuda.empty_cache()
        state["training"] = real_chain(folder, device, config)
        state.update(status="PASS", next_allowed_stage="S4")
    except Exception as exc:
        state.update(status="FAIL", error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        state["finish_utc"] = datetime.now(timezone.utc).isoformat()
        write(folder / "result.json", state)
        write(REPORTS / "S3.json", state)
    print(f"S3 {state['status']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not args.run_id or args.run_id in (".", "..") or Path(args.run_id).name != args.run_id:
        parser.error("expected a new run directory name")
    lock = os.open(REPORTS, os.O_RDONLY)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(REPORTS / "s3_runs" / args.run_id, torch.device(args.device))
    finally:
        os.close(lock)
