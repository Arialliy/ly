"""Run the three S2c engineering checks in order, binding the frozen S2b inputs."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys

import torch

EXPERIMENT = Path(__file__).resolve().parents[1]
PROJECT = EXPERIMENT.parents[1]
sys.path.insert(0, str(PROJECT))
from experiments.irstd_independent_v1.checks.check_s0 import atomic_text, repository_commit
from experiments.irstd_independent_v1.checks.check_s2b import verify_dependencies

REPORTS = EXPERIMENT / "reports"
ARCHIVE = REPORTS / "s2b_runs/seed46_400_20261118_01"
WEIGHTS = "a960e500f7f00da69de273ade47ecc021de386afb7835a9957a1dd7b90cb37de"
STAGES = {
    "resume": ("check_s2c_resume", "resume/result.json", ["engine/checkpoints.py"]),
    "prediction": ("s2c_prediction", "prediction.json", ["engine/checkpoints.py", "engine/predict.py"]),
    "resources": ("s2c_resources", "resources.json", ["engine/checkpoints.py"]),
}


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def preflight():
    a, b = (read(REPORTS / name) for name in ("S2a.json", "S2b.json"))
    subset_path = EXPERIMENT / "manifests/s2b_train12.json"
    subset = read(subset_path)
    require((a["status"], a["next_allowed_stage"], b["status"], b["next_allowed_stage"])
            == ("PASS", "S2b", "PASS", "S2c"), "S2a/S2b prerequisites not satisfied")
    verify_dependencies(subset)
    require(all(sha(EXPERIMENT / path) == digest for path, digest in b["code_sha256"].items()),
            "S2b code differs from the training snapshot")
    require(b["subset_sha256"] == sha(subset_path), "training subset digest mismatch")
    require(sha(REPORTS / "S2b.json") == sha(ARCHIVE / "result.json"), "S2b reports differ")
    history = [json.loads(line) for line in (ARCHIVE / b["history"]).read_bytes().splitlines()]
    require(len(history) == b["steps_completed"] == len(subset["schedule"]) == 400 and
            all(r["step"] == n and r["sample_id"] == key and r["finite_after_update"]
                and math.isfinite(r["loss_before_update"]) and math.isfinite(r["gradient_l2"])
                for n, (r, key) in enumerate(zip(history, subset["schedule"]), 1)), "training log mismatch")
    require(sha(ARCHIVE / "final.pt") == b["checkpoint"]["sha256"] == WEIGHTS and
            b["checkpoint"]["global_step"] == 400, "incorrect formal checkpoint")
    audit = read(ARCHIVE / "endpoint_audit.json")
    require(audit["status"] == "PASS" and audit["steps"]["400"]["logits_sha256"] ==
            sha(ARCHIVE / "logits_400.npz"), "archived native logits mismatch")
    snapshot = read(ARCHIVE / "snapshot_400.json")
    require(snapshot == b["snapshots"][-1] and snapshot["step"] == 400 and
            all(pt["threshold"] == 59.03249740600586 for pt in snapshot["froc"]["budgets"].values()),
            "frozen threshold or step-400 snapshot mismatch")
    files = [REPORTS / "S2a.json", REPORTS / "S2b.json", subset_path, ARCHIVE / "final.pt",
             ARCHIVE / "logits_400.npz", ARCHIVE / "endpoint_audit.json", ARCHIVE / b["history"],
             EXPERIMENT / "manifests/nudt_mirsdt_mask17.json", PROJECT / "1/S2c_execution_plan.md"]
    return {str(path.relative_to(PROJECT)): sha(path) for path in files}


def execute(run_dir, stage, device):
    frozen = preflight()
    summary_path = run_dir / "result.json"
    canonical = REPORTS / "S2c.json"
    if stage == "resume":
        run_dir.mkdir(parents=True, exist_ok=False)
        state = {"stage": "S2c", "status": "RUNNING", "next_allowed_stage": None, "start_utc": now(),
                 "run_dir": str(run_dir.relative_to(EXPERIMENT)), "base_commit": repository_commit(PROJECT),
                 "main_fit_optimizer_updates": 400, "checkpoint_sha256": WEIGHTS, "inputs": frozen,
                 "controller_sha256": sha(Path(__file__)), "stages": {}}
        write(run_dir / "protocol.json", dict(state, stage_order=list(STAGES),
              resume_samples=["Sequence83/00020", "Sequence77/00046"], prediction_core_hw=[64, 64],
              prediction_sources=["Sequence1", "Sequence17"], centers_inclusive=[9, 92],
              atol=1e-5, rtol=1e-4, threshold=59.03249740600586, resource_sample_id="Sequence32/00009",
              allocated_budget_mib=8192))
    else:
        state = read(summary_path)
        require(state["status"] == "RUNNING" and state["inputs"] == frozen and
                state["controller_sha256"] == sha(Path(__file__)), "run or inputs changed")
    expected = list(STAGES)[:list(STAGES).index(stage)]
    require(list(state["stages"]) == expected, "stage already attempted or predecessor missing")
    for name in expected:
        prev = state["stages"][name]
        require(prev["status"] == "PASS" and prev["sha256"] == sha(run_dir / prev["result"])
                and all(sha(EXPERIMENT / key) == val for key, val in prev["code_sha256"].items()),
                f"prior result or code changed for {name}")
    entry, result_path, helpers = STAGES[stage]
    sources = [f"checks/{entry}.py", *helpers]
    job = {"start_utc": now(), "status": "RUNNING", "result": result_path,
           "code_sha256": {name: sha(EXPERIMENT / name) for name in sources},
           "environment": {"device": device, "gpu": torch.cuda.get_device_name(device),
               "gpu_index": os.getenv("S2_GPU_INDEX"), "gpu_uuid": os.getenv("S2_GPU_UUID"),
               "image": os.getenv("S2_IMAGE_ID"), "python": platform.python_version(), "torch": str(torch.__version__)}}
    state["stages"][stage] = job
    write(summary_path, state)
    write(canonical, state)
    try:
        cmd = [sys.executable, "-m", "experiments.irstd_independent_v1.checks." + entry,
               "--run-dir", str(run_dir), "--device", device]
        proc = subprocess.run(cmd, cwd=PROJECT, check=False)
        evidence = run_dir / result_path
        if evidence.exists():
            job["sha256"] = sha(evidence)
            job["status"] = read(evidence)["status"]
        require(proc.returncode == 0 and job["status"] == "PASS", f"{stage} failed; see {result_path}")
        require(frozen == preflight(), "frozen artifacts changed during the check")
        if stage == "resources":
            state.update(status="PASS", next_allowed_stage="S3", finish_utc=now())
    except Exception as exc:
        job.update(status="FAIL", error=str(exc))
        state.update(status="FAIL", finish_utc=now())
        raise
    finally:
        job["finish_utc"] = now()
        write(summary_path, state)
        write(canonical, state)
    if state["status"] == "PASS":
        names = ("S2a", "S2b", "S2c")
        write(REPORTS / "S2.json", {"stage": "S2", "status": "PASS", "next_allowed_stage": "S3",
              "main_fit_optimizer_updates": 400, "s2a_engineering_updates": 1,
              "s2c_resume_audit_updates": 3, "s2c_resource_updates": 2, "checkpoint_sha256": WEIGHTS,
              "checkpoint": str((ARCHIVE / "final.pt").relative_to(EXPERIMENT)),
              "substage_reports": {k: {"path": f"reports/{k}.json", "sha256": sha(REPORTS / f"{k}.json")}
                                   for k in names}, "finish_utc": now()})
    print(f"S2c {stage} PASS", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not args.run_id or args.run_id in (".", "..") or Path(args.run_id).name != args.run_id:
        parser.error("use a single run directory name")
    lock = os.open(REPORTS, os.O_RDONLY)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(REPORTS / "s2c_runs" / args.run_id, args.stage, args.device)
    finally:
        os.close(lock)
