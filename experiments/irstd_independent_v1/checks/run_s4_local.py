"""Run the four configurations of the frozen, fixed-seed S4 comparison."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import NamedTemporaryFile
from threading import Event


EXP = Path(__file__).resolve().parents[1]
PROJECT = EXP.parents[1]
sys.path.insert(0, str(PROJECT))
IMAGE = "sha256:eb956873870d737f2b68dff0724b8f6f56f778829d119b0938d29b7abeaa2f92"
MODES = ("M0", "Msep", "Mraw", "Mint")
LANES = {1: ("Msep", "M0"), 2: ("Mraw",), 3: ("Mint",)}
SEED = 46
WORKER = "experiments.irstd_independent_v1.engine.s4_train"


def stamp():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def publish(path, value):
    with NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf8") as f:
        tmp = Path(f.name)
        try:
            json.dump(value, f, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)


def cmd(args):
    return subprocess.check_output(args, text=True).strip()


def check(condition, detail):
    if not condition:
        raise ValueError(detail)


def result_valid(run, protocol, bindings, action, mode, seed, step):
    folder = run / mode / f"seed{seed}"
    path = folder / f"training_{step}.json"
    if not path.exists():
        return False
    entry = read(path)
    check((entry["mode"], entry["seed"], entry["step"]) == (mode, seed, step), f"Wrong job identity {path}")
    data_sha = bindings["protocol_sha256"]
    checkpoint = folder / "checkpoints" / f"step_{step:06d}.pt"
    check(entry["protocol_sha256"] == data_sha and entry["checkpoint_sha256"] == sha(checkpoint),
          f"Training receipt mismatch {path}")
    if step == 8:
        check(entry.get("resume_verified") is True, f"Fresh-model restore has not passed at {path}")
    if step >= 24000:
        migration = protocol.get("migration")
        if migration and mode in migration["modes"]:
            first = migration["common_step"]+1
            receipt = read(folder/"migration_first_update.json")
            check(receipt["step"] == first and receipt["mode"] == mode and receipt["seed"] == seed
                  and receipt["run_id"] == run.name and receipt["protocol_sha256"] == data_sha
                  and receipt["source_run_id"] == migration["parent_run_id"]
                  and receipt["source_checkpoint_sha256"] == migration["modes"][mode]["checkpoint_sha256"]
                  and receipt["root_checkpoint_sha256"] == sha(folder/"checkpoints"/f"step_{first-1:06d}.pt")
                  and receipt["checkpoint_sha256"] == sha(folder/"checkpoints"/f"step_{first:06d}.pt"),
                  "Missing or inconsistent first optimizer update after the parent boundary")
        for observed in range(0, step + 1, 3000):
            diag_folder, diag_protocol = folder, data_sha
            if migration and mode in migration["modes"] and observed <= migration["common_step"]:
                from experiments.irstd_independent_v1.checks.s4_lineage import diagnostic_origin
                ancestor, diag_protocol = diagnostic_origin(run,protocol,mode,observed)
                diag_folder = ancestor/mode/f"seed{seed}"
            diag = read(diag_folder / f"diagnostics_{observed}.json")
            check(diag["step"] == observed and diag["protocol_sha256"] == diag_protocol
                  and isinstance(diag["fit"]["loss"], (float, int)),
                  f"Missing training diagnostic {folder}/{observed}")
        for observed in range(12000, step + 1, 12000):
            out = folder / f"dev_{observed}"
            verify_metrics(out, protocol["dev_ids"])
            receipt = read(out / "binding.json")
            check((receipt["step"], receipt["mode"], receipt["seed"]) == (observed, mode, seed)
                  and receipt["protocol_sha256"] == data_sha and receipt["metrics_sha256"] == sha(out / "metrics.json")
                  and receipt["checkpoint_sha256"] == sha(folder / "checkpoints" / f"step_{observed:06d}.pt")
                  and set(receipt["frame_sha256"]) == set(protocol["dev_ids"]), f"Unbound development endpoint {out}")
    if action == "train":
        return True
    out = folder / f"test_{step}"
    receipt = out / "binding.json"
    if not receipt.exists():
        return False
    record = read(receipt)
    check((record["mode"], record["seed"], record["step"]) == (mode, seed, step)
          and record["protocol_sha256"] == data_sha and record["checkpoint_sha256"] == entry["checkpoint_sha256"]
          and record["historical_lock_sha256"] == sha(run / "historical_lock.json")
          and record["metrics_sha256"] == sha(out / "metrics.json"), f"Invalid historical receipt {receipt}")
    verify_metrics(out, protocol["test_ids"])
    return True


def verify_metrics(folder, ids):
    metrics = read(folder / "metrics.json")
    froc = metrics["descriptive"]
    check(froc["frames"] == len(ids) and set(froc["budgets"]["1"]["per_frame"]) == set(ids)
          and froc["curve_sha256"] == sha(folder / froc["curve_file"]), f"Invalid FROC at {folder}")
    return metrics


def historical_lock(run, protocol, bindings, step):
    result = {"stage": "S4", "state": "HISTORICAL_LOCKED", "run_id": run.name, "selected_step": step,
              "protocol_sha256": bindings["protocol_sha256"], "execution_binding_sha256": bindings["execution_binding_sha256"],
              "budget_decision_sha256": sha(run / "budget_decision.json"), "runs": {}}
    for mode in MODES:
        check(result_valid(run, protocol, bindings, "train", mode, SEED, step), f"Missing endpoint for {mode}")
        folder = run / mode / f"seed{SEED}"
        dev = folder / f"dev_{step}" / "metrics.json"
        points = read(dev)["descriptive"]["budgets"]
        result["runs"][mode] = {str(SEED): {"mode": mode, "seed": SEED, "step": step,
              "checkpoint_sha256": read(folder / f"training_{step}.json")["checkpoint_sha256"],
              "dev_metrics_sha256": sha(dev), "thresholds": {b: points[b]["threshold"] for b in ("0.5", "1", "5")}}}
    path = run / "historical_lock.json"
    if path.exists():
        check(read(path) == result, "Historical model or threshold binding changed")
    else:
        publish(path, result)


def find_container(name, labels):
    ids = cmd(["docker", "ps", "-a", "--no-trunc", "-q", "--filter", f"name=^/{name}$"]).splitlines()
    if not ids:
        return None
    check(len(ids) == 1, f"Ambiguous container {name}")
    info = json.loads(cmd(["docker", "inspect", ids[0]]))[0]
    check(info["Image"] == IMAGE and all(info["Config"]["Labels"].get(k) == v for k, v in labels.items()),
          f"Existing container {name} has a different owner or frozen protocol")
    return info


def idle_device(index):
    check(index in LANES, "GPU outside authorized set")
    spec = cmd(["nvidia-smi", f"--id={index}", "--query-gpu=index,uuid,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits"]).split(",")
    idx, uuid, used, util = (part.strip() for part in spec)
    check(int(idx) == index and uuid.startswith("GPU-"), "Invalid host-GPU identity")
    processes = cmd(["nvidia-smi", f"--id={uuid}", "--query-compute-apps=pid", "--format=csv,noheader,nounits"])
    return uuid if not processes and int(used) < 50 and int(util) < 5 else None


def launch_args(run, name, labels, index, uuid, action, mode, seed, step):
    container_folder = Path("/project") / run.relative_to(PROJECT)
    args = ["docker", "run", "--detach", "--name", name, "--network", "none",
            "--user", f"{os.getuid()}:{os.getgid()}", "--cpus", "4", "--memory", "20g",
            "--gpus", f"device={uuid}", "--security-opt", "no-new-privileges",
            "-v", f"{PROJECT}:/project:ro", "-v", f"{run}:{container_folder}:rw"]
    for k, v in labels.items():
        args.extend(["--label", f"{k}={v}"])
    env = {"S2_IMAGE_ID": IMAGE, "S2_GPU_INDEX": str(index), "S2_GPU_UUID": uuid,
           "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
           "PYTHONPATH": "/project/environment/site", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1"}
    for k, v in env.items():
        args.extend(["-e", f"{k}={v}"])
    return args + ["-w", "/project", "--entrypoint", "/opt/conda/bin/python", IMAGE, "-m", WORKER,
                   str(container_folder), action, mode, str(seed), str(step)]


def run_job(run, protocol, bindings, stop, index, action, mode, seed, step):
    driver = run / "driver"
    key = f"{action}_{mode}_{seed}_{step}"
    name = f"s4-{run.name}-{key}"
    lane_path = driver / f"lane_gpu{index}.json"
    labels = {"s4.job": key, "s4.run": run.name, "s4.protocol": bindings["protocol_sha256"],
              "s4.binding": bindings["execution_binding_sha256"], "s4.gpu": str(index)}
    old = find_container(name, labels)
    if result_valid(run, protocol, bindings, action, mode, seed, step):
        if old and old["State"]["Status"] == "exited":
            cmd(["docker", "rm", old["Id"]])
        publish(lane_path, {"status": "VERIFIED", "job": key, "utc": stamp()})
        return
    if old and old["State"]["Status"] in ("exited", "dead"):
        # A previous interrupted controller may have left a stopped container.
        with (driver / f"{key}.log").open("ab") as log:
            subprocess.run(["docker", "logs", old["Id"]], stdout=log, stderr=subprocess.STDOUT, check=True)
        cmd(["docker", "rm", old["Id"]])
        old = None
    if old:
        check(old["State"]["Status"] == "running", f"Unexpected state for {name}: {old['State']['Status']}")
        identifier = old["Id"]
    else:
        publish(lane_path, {"status": "WAITING_FOR_GPU", "job": key, "gpu_index": index, "utc": stamp()})
        while not stop.is_set():
            uuid = idle_device(index)
            if uuid:
                break
            stop.wait(30)
        if stop.is_set():
            return
        identifier = cmd(launch_args(run, name, labels, index, uuid, action, mode, seed, step))
    print(f"{stamp()} started {key} ({identifier[:12]})", flush=True)
    publish(lane_path, {"status": "RUNNING", "job": key, "container": identifier, "gpu_index": index,
                        "started_or_reattached_utc": stamp()})
    log_path = driver / f"{key}.log"
    with log_path.open("ab", buffering=0) as log:
        log.write(f"\n{stamp()} container {identifier}\n".encode())
        subprocess.run(["docker", "logs", "--follow", identifier], stdout=log, stderr=subprocess.STDOUT, check=True)
    status = int(cmd(["docker", "wait", identifier]))
    if status:
        raise RuntimeError(f"{key} exited {status}; see {log_path}")
    check(result_valid(run, protocol, bindings, action, mode, seed, step), f"Missing successful receipt for {key}")
    cmd(["docker", "rm", identifier])
    publish(lane_path, {"status": "VERIFIED", "job": key, "utc": stamp()})
    print(f"{stamp()} verified {key}", flush=True)


def run_phase(run, protocol, bindings, action, step):
    stop = Event()
    def lane(index, modes):
        for mode in modes:
            if stop.is_set():
                return
            try:
                run_job(run, protocol, bindings, stop, index, action, mode, SEED, step)
            except Exception as error:
                stop.set()
                failure = {"status": "FAILED", "action": action, "seed": SEED, "mode": mode, "step": step,
                           "utc": stamp(), "error": str(error)}
                publish(run / "driver" / f"lane_gpu{index}.json", failure)
                publish(run / "driver/state.json", failure)
                raise
    with ThreadPoolExecutor(max_workers=len(LANES)) as pool:
        futures = [pool.submit(lane, index, modes[:1] if step == 8 else modes) for index, modes in LANES.items()]
        for future in as_completed(futures):
            future.result()


def compare(run, step=None):
    folder = Path("/project") / run.relative_to(PROJECT)
    args = ["docker", "run", "--rm", "--network", "none", "--user", f"{os.getuid()}:{os.getgid()}",
            "--cpus", "4", "--memory", "20g", "--security-opt", "no-new-privileges",
            "-v", f"{PROJECT}:/project:ro", "-v", f"{run}:{folder}:rw",
            "-e", "PYTHONPATH=/project/environment/site", "-e", "PYTHONDONTWRITEBYTECODE=1",
            "-e", "OPENBLAS_NUM_THREADS=1", "-w", "/project", "--entrypoint", "/opt/conda/bin/python",
            IMAGE, "-m", "experiments.irstd_independent_v1.checks.s4_compare", str(folder)]
    if step is not None:
        args.extend(["--step", str(step)])
    log = run / "driver" / ("budget_decision.log" if step is None else "summary.log")
    with log.open("ab", buffering=0) as out:
        subprocess.run(args, check=True, stdout=out, stderr=subprocess.STDOUT)


def run_protocol(run):
    protocol = read(run / "protocol.json")
    if (protocol.get("initial_steps"), protocol.get("max_steps")) != (24000, 36000):
        raise ValueError("Obsolete S4 budget; the November 18 research protocol requires 24000/36000 updates")
    check(tuple(protocol["modes"]) == MODES and tuple(protocol["seeds"]) == (SEED,)
          and protocol["run_id"] == run.name, "Only the authorized four fixed-seed-46 runs may start")
    from experiments.irstd_independent_v1.checks.s4_binding import verify_execution
    verified_protocol, locked = verify_execution(run)
    check(verified_protocol == protocol, "Frozen protocol changed during driver startup")
    lock_path = run / "execution_binding.json"
    digest = sha(run / "protocol.json")
    check(locked["can_launch_training"] is True and locked["run_id"] == run.name
          and locked["common_protocol_sha256"] == digest
          and sha(lock_path) == sha(EXP / "reports/S4_protocol_lock.json"), "Missing formal execution binding")
    for path, expected in protocol["proposal_sha256"].items():
        check(sha(PROJECT / path) == expected, f"Changed user protocol {path}")
    check(sha(run / "schedule_46.json") == protocol["schedule_sha256"]["46"], "Seed 46 schedule digest mismatch")
    driver = run / "driver"
    driver.mkdir(exist_ok=True)
    bindings = {"protocol_sha256": digest, "image": IMAGE, "execution_binding_sha256": sha(lock_path)}
    inputs = driver / "inputs.json"
    if inputs.exists():
        check(read(inputs) == bindings, "Live driver/worker protocol does not match the previous invocation")
    else:
        publish(inputs, bindings)
    check(os.getuid() != 0 and cmd(["docker", "image", "inspect", "--format", "{{.Id}}", IMAGE]) == IMAGE,
          "A non-root user and the fixed local Docker image are required")
    state_path = driver / "state.json"
    def update(status, **more):
        publish(state_path, {"status": status, "pid": os.getpid(), "utc": stamp(), **more})
    try:
        migration = protocol.get("migration")
        if migration:
            check(set(migration["modes"]) == {"Msep","Mraw","Mint"}
                  and 0 < migration["common_step"] < 3000 and migration["common_step"] % 300 == 0,
                  "Expected a shared pre-diagnostic migration boundary for the three evidence models")
            update("MIGRATING", step=migration["common_step"], parent_run_id=migration["parent_run_id"], seed=SEED)
        else:
            update("TRAINING_PREFIX", step=8, seed=SEED)
            run_phase(run, protocol, bindings, "train", 8)
        update("TRAINING", step=24000, seed=SEED)
        run_phase(run, protocol, bindings, "train", 24000)
        compare(run)
        decision = read(run / "budget_decision.json")
        step = decision["selected_step"]
        check(step in (24000, 36000), "Unexpected selected training budget")
        if step == 36000:
            update("TRAINING", step=step, seed=SEED)
            run_phase(run, protocol, bindings, "train", step)
        historical_lock(run, protocol, bindings, step)
        update("HISTORICAL_EVALUATION", step=step, seed=SEED)
        run_phase(run, protocol, bindings, "historical", step)
        compare(run, step)
        final = read(run / "S4_summary.json")
        check(final["execution_status"] == "COMPLETE" and final["step"] == step and
              final["protocol_sha256"] == bindings["protocol_sha256"], "Unexpected final S4 summary")
        for source, target in (("S4_summary.json", "S4.json"), ("S4_research_report.md", "S4_research_report.md"),
                               ("S4_comparison.csv", "S4_comparison.csv"), ("S4_paired_targets.csv", "S4_paired_targets.csv")):
            with NamedTemporaryFile(dir=EXP / "reports", delete=False) as out:
                temp = Path(out.name)
                try:
                    out.write((run / source).read_bytes())
                    out.flush()
                    os.fsync(out.fileno())
                    os.replace(temp, EXP / "reports" / target)
                finally:
                    temp.unlink(missing_ok=True)
        update("COMPLETE", step=step)
    except Exception as error:
        update("FAILED", error=str(error))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--first-seed-only", action="store_true", help="Fixed seed 46 is always enforced")
    args = parser.parse_args()
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_id) is None:
        parser.error("expected a single run ID")
    folder = EXP / "reports/s4_runs" / args.run_id
    if not (folder / "protocol.json").is_file():
        parser.error(f"freeze the protocol first with s4_protocol.py --run-id {args.run_id}")
    fd = os.open(folder, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_protocol(folder)
    finally:
        os.close(fd)
