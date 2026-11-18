"""Run and record the S2a forward/gradient checks against the frozen input contracts."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import traceback

import numpy as np
import torch

from .check_s0 import atomic_text, repository_commit
from .s1_runtime import verify_s0


EXPERIMENT = Path(__file__).resolve().parents[1]
PROJECT = EXPERIMENT.parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_inputs():
    previous = verify_s0(EXPERIMENT)
    reports = EXPERIMENT / "reports"
    s0 = json.loads((reports / "S0.json").read_bytes())
    s1 = json.loads((reports / "S1.json").read_bytes())
    if (s1["status"], s1["next_allowed_stage"], s1["pending_requirements"]) != ("PASS", "S2", []):
        raise ValueError("S1 is incomplete")
    evidence = s1["formal_coverage"]["evidence"]
    bound_files = {**s1["code_and_config_sha256"], evidence["path"]: evidence["sha256"],
                   evidence["protocol_path"]: evidence["protocol_sha256"]}
    for name, expected in bound_files.items():
        if sha256(EXPERIMENT / name) != expected:
            raise ValueError(f"S1 dependency changed: {name}")

    manifest = json.loads((EXPERIMENT / previous["data_manifest"]).read_bytes())
    train = {row["id"]: row for row in manifest["sequences"] if row["split"] == "train"}
    sample_ids = s0["loader"]["sample_ids"]
    if sample_ids != ["Sequence1/00009", "Sequence17/00009", "Sequence2/00009"]:
        raise ValueError("Fixed S2a examples differ from the S0 loader audit")
    files = []
    for sample in sample_ids:
        sequence, physical = sample.split("/")
        center = int(physical)
        frames = {row["number"]: row for row in train[sequence]["frames"]}
        if center not in train[sequence]["centers"]:
            raise ValueError(f"Training center is not eligible: {sample}")
        files.extend(frames[i]["image"] for i in range(center - 8, center + 9))
        files.append(frames[center]["mask"])
    archive = s0["pixel_audit"]["provenance"]
    raw = (PROJECT / archive["inventory"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != archive["inventory_sha256"]:
        raise ValueError("Source inventory changed")
    members = {row["member"]: row["sha256"] for row in json.loads(raw)["files"]}
    for name in files:
        if sha256(PROJECT / name) != members[Path(name).relative_to("data/public").as_posix()]:
            raise ValueError(f"Fixed training input differs from archive: {name}")
    return {"S0_sha256": sha256(reports / "S0.json"), "S1_sha256": sha256(reports / "S1.json"),
            "data_manifest_sha256": previous["data_manifest_sha256"], "train_sample_ids": sample_ids,
            "release_files_verified": len(files)}


def run(device, output):
    state = {"stage": "S2a", "status": "RUNNING", "next_allowed_stage": None, "seed": 46,
             "started_utc": datetime.now(timezone.utc).isoformat(), "short_fit_steps": 0}
    atomic_text(output, json.dumps(state, indent=2) + "\n")
    try:
        state["base_commit"] = repository_commit(PROJECT)
        state["prerequisites"] = check_inputs()
        design = PROJECT / "1/S2_model_design.md"
        state["design_sha256"] = sha256(design) if design.is_file() else None
        files = [Path(__file__), EXPERIMENT / "checks/check_s2a.py", *sorted((EXPERIMENT / "models").glob("*.py")),
                 *sorted((EXPERIMENT / "losses").glob("*.py"))]
        state["code_sha256"] = {str(path.relative_to(EXPERIMENT)): sha256(path) for path in files}
        random.seed(46)
        np.random.seed(46)
        torch.manual_seed(46)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        state["environment"] = {"python": platform.python_version(), "torch": torch.__version__,
                                "device": str(device), "image": os.environ.get("S2_IMAGE_ID"),
                                "precision": "float32", "cudnn_deterministic": True, "tf32": False}
        if device.type == "cuda":
            state["environment"]["gpu"] = torch.cuda.get_device_name(device)

        from .check_s2a import verify_s2a
        state["checks"] = verify_s2a(PROJECT, device)
        state.update(status="PASS", next_allowed_stage="S2b")
    except Exception as exc:
        state.update(status="FAIL", error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        state["finished_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_text(output, json.dumps(state, indent=2, allow_nan=False) + "\n")
    print(f"S2a PASS. Report: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    reports = EXPERIMENT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    lock = os.open(reports, os.O_RDONLY)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX)
        run(device, reports / "S2a.json")
    finally:
        os.close(lock)
