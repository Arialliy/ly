"""Launch the frozen S2b fit and decide the stage from its step-0 and step-400 diagnostics."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import traceback

import torch

EXPERIMENT = Path(__file__).resolve().parents[1]
PROJECT = EXPERIMENT.parents[1]
sys.path.insert(0, str(PROJECT))

from experiments.irstd_independent_v1.checks.check_s0 import atomic_text, repository_commit
from experiments.irstd_independent_v1.engine.s2b_train import run_short_fit
from experiments.irstd_independent_v1.evaluation.s2b_froc import check_froc_fixtures


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def verify_dependencies(subset):
    files = {"reports/S2a.json": "s2a_sha256", "reports/S0.json": "s0_sha256",
             "manifests/nudt_mirsdt_mask17.json": "data_manifest_sha256", "data/s2b_subset.py": "selector_sha256"}
    for name, key in files.items():
        if sha(EXPERIMENT / name) != subset["dependencies"][key]:
            raise ValueError(f"Frozen subset dependency changed: {name}")
    s2a = json.loads((EXPERIMENT / "reports/S2a.json").read_bytes())
    if (s2a["status"], s2a["next_allowed_stage"]) != ("PASS", "S2b"):
        raise ValueError("S2a did not pass")
    for name, expected in s2a["code_sha256"].items():
        if sha(EXPERIMENT / name) != expected:
            raise ValueError(f"S2a code changed: {name}")
    for sample in subset["samples"]:
        for row in [*sample["images"], sample["center_mask"]]:
            if sha(PROJECT / row["path"]) != row["sha256"]:
                raise ValueError(f"Frozen raw data changed: {row['path']}")


def decide(result):
    snapshots = result["snapshots"]
    if [s["step"] for s in snapshots] != list(range(0, 401, 50)):
        raise ValueError("Missing a fixed diagnostic point")
    a, b = snapshots[0], snapshots[-1]
    reduction = 1 - b["loss"]["total"] / a["loss"]["total"]
    criteria = {
        "400_finite_updates": result["steps_completed"] == 400 and result["all_finite"],
        "loss_decrease_at_least_20_percent": reduction >= 0.20,
        "positive_frame_foreground_bce_improves": b["loss"]["foreground_bce"] < a["loss"]["foreground_bce"],
        "pd_at_one_fp_per_frame_improves": b["froc"]["budgets"]["1"]["pd"] > a["froc"]["budgets"]["1"]["pd"],
    }
    return criteria, reduction


def execute(device, folder, summary):
    subset_path = EXPERIMENT / "manifests/s2b_train12.json"
    subset = json.loads(subset_path.read_bytes())
    verify_dependencies(subset)
    folder.mkdir(parents=True, exist_ok=False)
    state = {"stage": "S2b", "status": "RUNNING", "next_allowed_stage": None,
             "start_utc": datetime.now(timezone.utc).isoformat(), "seed": 46,
             "run_dir": str(folder.relative_to(EXPERIMENT)), "subset_sha256": sha(subset_path),
             "base_commit": repository_commit(PROJECT), "subset": "manifests/s2b_train12.json",
             "decision_protocol": {"updates": 400, "finite_losses_and_gradients": True,
                 "minimum_fractional_loss_decrease": 0.20, "foreground_bce_strict_decrease": True,
                 "pd_at_one_fp_per_frame_strict_increase": True, "evaluation_population": "same 12 training centers"},
             "environment": {"device": str(device), "gpu": torch.cuda.get_device_name(device),
                 "host_gpu_index": os.environ.get("S2_GPU_INDEX"), "gpu_uuid": os.environ.get("S2_GPU_UUID"),
                 "image": os.environ.get("S2_IMAGE_ID"), "python": platform.python_version(), "torch": torch.__version__}}
    files = ["checks/check_s2b.py", "engine/s2b_train.py", "evaluation/s2b_diagnostics.py",
             "evaluation/s2b_froc.py", "data/s2b_subset.py", "configs/task.yaml"]
    state["code_sha256"] = {name: sha(EXPERIMENT / name) for name in files}
    write(folder / "protocol.json", state)
    write(summary, state)
    try:
        state["froc_fixtures"] = check_froc_fixtures()
        result = run_short_fit(PROJECT, folder, subset, device)
        state.update(result)
        criteria, reduction = decide(result)
        state.update(criteria=criteria, relative_loss_decrease=reduction,
                     status="PASS" if all(criteria.values()) else "FAIL",
                     next_allowed_stage="S2c" if all(criteria.values()) else None)
    except Exception as error:
        state.update(status="FAIL", error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        state["finish_utc"] = datetime.now(timezone.utc).isoformat()
        write(folder / "result.json", state)
        write(summary, state)
    print(f'S2b {state["status"]}: {state["steps_completed"]} updates, criteria {state["criteria"]}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-id", required=True, help="New output directory name")
    args = parser.parse_args()
    if not args.run_id or Path(args.run_id).name != args.run_id or args.run_id in (".", ".."):
        parser.error("run-id must be a single directory name")
    reports = EXPERIMENT / "reports"
    lock = os.open(reports, os.O_RDONLY)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(torch.device(args.device), reports / "s2b_runs" / args.run_id, reports / "S2b.json")
    finally:
        os.close(lock)


if __name__ == "__main__":
    main()
