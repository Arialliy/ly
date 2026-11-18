"""Durable host-only History postprocessing; never starts or retries training."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

RUNTIME = Path(__file__).resolve().parent
E = RUNTIME.parent
ROOT = E.parents[1]
WORKFLOW = RUNTIME / "History_WORKFLOW.json"
FREEZE = E / "PROTOCOL_FREEZE.json"
TRAINING = E / "training/History/run/RUN.json"
TRAIN_LAUNCH = RUNTIME / "train_history.json"
LAUNCH = E / "code/launch.py"
GPU = 2
SEQUENCES = ("DJI_0026_2", "DJI_0067_2")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else None


def save(record):
    record["heartbeat_unix"] = time.time()
    temporary = WORKFLOW.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(WORKFLOW)


def gpu_free():
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                                      "--format=csv,noheader,nounits"], text=True)
    rows = {int(row.split(",")[0]): [int(x.strip()) for x in row.split(",")[1:]]
            for row in output.splitlines()}
    return GPU in rows and rows[GPU][0] < 500 and rows[GPU][1] == 0, rows


def verify_sources(record):
    assert sha(FREEZE) == record["protocol_freeze_sha256"], "Frozen protocol changed"
    frozen = read(FREEZE)
    assert frozen["status"] == "FROZEN" and frozen["main_step"] == 2048
    assert sha(E / "PROTOCOL.md") == frozen["protocol_document_sha256"]
    for path, expected in frozen["source_sha256"].items():
        assert sha(path) == expected, ("Frozen source changed", path)


def wait_training(record):
    while True:
        training, launcher = read(TRAINING), read(TRAIN_LAUNCH)
        record.update(status="WAIT_TRAINING", training_status=None if training is None else training.get("status"),
                      training_launcher_status=None if launcher is None else launcher.get("status"))
        if training and training.get("status", "").lower() in ("failed", "interrupted", "error"):
            raise RuntimeError("History training failed; no retry is authorized by this worker")
        if launcher and launcher.get("status") in ("FAILED", "INTERRUPTED", "BLOCKED_OCCUPIED"):
            raise RuntimeError("History training launcher failed; no automatic retry")
        if training and launcher and training.get("status") == "complete" and launcher.get("status") == "COMPLETE":
            assert training["condition"] == "History" and training["final_step"] == 2048
            assert launcher["exit_code"] == 0
            verify_sources(record)
            record.update(training_run_sha256=sha(TRAINING), training_launcher_sha256=sha(TRAIN_LAUNCH),
                          training_completed_unix=time.time())
            save(record)
            return
        save(record)
        time.sleep(30)


def run_job(record, tag, script, arguments, gpu=None):
    receipt = RUNTIME / (tag + ".json")
    if receipt.exists() or receipt.with_suffix(".log").exists():
        raise FileExistsError("Refusing to overwrite/retry existing History postprocess job: " + tag)
    command = [sys.executable, str(LAUNCH)]
    if gpu is not None:
        command += ["--gpu", str(gpu)]
    command += ["--tag", tag, script, *arguments]
    job = {"tag": tag, "script": script, "command": command,
           "invoked_source_sha256": {str(path): sha(path) for path in
                                     (LAUNCH, E / "code" / script, E / "code/runtime.py")},
           "status": "PENDING", "gpu": gpu}
    record["jobs"].append(job)
    while True:
        verify_sources(record)
        if gpu is not None:
            free, rows = gpu_free()
            record["gpu_observation"] = rows
            if not free:
                record.update(status="WAIT_GPU", current_job=tag)
                job["status"] = "WAIT_GPU"
                save(record)
                time.sleep(30)
                continue
        record.update(status="RUNNING_POSTPROCESS", current_job=tag)
        job.update(status="RUNNING", start_unix=time.time())
        process = subprocess.Popen(command, cwd=ROOT)
        job["launcher_pid"] = process.pid
        save(record)
        while process.poll() is None:
            time.sleep(5)
            save(record)
        job.update(exit_code=process.returncode, end_unix=time.time())
        # launch.py rejects a newly occupied GPU before reserving any receipt.
        # Waiting after that race does not retry an inference workload.
        if process.returncode and gpu is not None and not receipt.exists() and not receipt.with_suffix(".log").exists():
            free, rows = gpu_free()
            record["gpu_observation"] = rows
            if not free:
                job["occupancy_races"] = job.get("occupancy_races", 0) + 1
                time.sleep(30)
                continue
        outcome = read(receipt)
        if process.returncode != 0 or not outcome or outcome.get("status") != "COMPLETE" or outcome.get("exit_code") != 0:
            job["status"] = "FAILED"
            raise RuntimeError("History postprocess failed without retry: " + tag)
        job.update(status="COMPLETE", launcher_receipt_sha256=sha(receipt))
        save(record)
        return


def main():
    with (RUNTIME / "History_WORKFLOW.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if WORKFLOW.exists():
            raise FileExistsError("History workflow already exists; do not restart automatically")
        record = {"status": "STARTING", "condition": "History", "gpu": GPU, "pid": os.getpid(),
                  "worker": str(Path(__file__).resolve()), "worker_sha256": sha(__file__),
                  "protocol_freeze_sha256": sha(FREEZE), "start_unix": time.time(), "jobs": [],
                  "main_step": 2048, "train_retry_policy": "never", "inference_retry_policy": "never after a workload launches",
                  "scope": "Wait existing History training and launcher; infer two fixed development clips on GPU2; CPU score History only"}
        (RUNTIME / "History_WORKFLOW.pid").write_text(str(os.getpid()) + "\n")
        save(record)
        try:
            verify_sources(record)
            wait_training(record)
            for sequence in SEQUENCES:
                run_job(record, "history_infer_" + sequence.lower(), "infer.py",
                        ["--condition", "History", "--sequence", sequence, "--step", "2048"], GPU)
            run_job(record, "history_evaluate", "evaluate.py", ["--condition", "History", "--step", "2048"])
            record.update(status="COMPLETE", current_job=None, end_unix=time.time())
        except BaseException as error:
            record.update(status="FAILED", error=repr(error), traceback=traceback.format_exc(), end_unix=time.time())
            save(record)
            raise
        save(record)


if __name__ == "__main__":
    main()
