"""Run a same-weight M1 diagnostic in the existing pinned environment."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
E = Path(__file__).resolve().parents[1]
IMAGE = "sha256:8470a68886ff2f480d76858d930fb72d0e680e69276fc1d068fe8b6625b4cb9f"
OLD = "/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009"


def save(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, choices=[1, 2, 3])
    p.add_argument("--tag", required=True)
    p.add_argument("script")
    p.add_argument("args", nargs=argparse.REMAINDER)
    a = p.parse_args()
    if not a.tag.replace("_", "").replace("-", "").isalnum():
        p.error("tag must contain only letters, digits, hyphens and underscores")
    script = (E / "code" / a.script).resolve()
    if script.parent != E / "code" or not script.is_file():
        p.error("script must name an existing file in this experiment's code directory")
    runtime = E / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    receipt = runtime / (a.tag + ".json")
    log = receipt.with_suffix(".log")
    if receipt.exists() or log.exists():
        raise FileExistsError(f"Will not overwrite job receipt/log: {receipt}")
    image_id = subprocess.check_output(
        ["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"], text=True).strip()
    if image_id != IMAGE:
        raise RuntimeError("Pinned image mismatch")
    record = dict(status="PREPARING", image=image_id, gpu=a.gpu, start_unix=time.time())
    if a.gpu is not None:
        raw = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits"], text=True)
        rows = {int(row.split(",")[0]): [int(v.strip()) for v in row.split(",")[1:]]
                for row in raw.splitlines()}
        record["gpu_before"] = rows
        if a.gpu not in rows or rows[a.gpu][0] >= 500 or rows[a.gpu][1] > 0:
            raise RuntimeError("Requested GPU is unavailable/occupied; no job launched")
    cmd = ["docker", "run", "--rm", "--name", "m1_history_effect_v1_" + a.tag,
           "--network", "none", "--user", f"{os.getuid()}:{os.getgid()}",
           "--cpus", "4", "--memory", "20g", "--shm-size", "2g"]
    if a.gpu is not None:
        cmd += ["--gpus", f"device={a.gpu}", "-e", "CUDA_VISIBLE_DEVICES=0"]
    env = {"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
           "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
           "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
           "PYTHONPATH": str(ROOT) + ":" + str(E / "code")}
    for key, value in env.items():
        cmd += ["-e", f"{key}={value}"]
    cmd += ["-v", f"{ROOT}:{ROOT}:ro",
            "-v", f"{ROOT}/archive/infrared_problem_screen_20261009:{OLD}:ro",
            "-v", f"{E}:{E}:rw", "-w", str(E), "--entrypoint", "python",
            image_id, str(script), *a.args]
    record.update(status="RUNNING", command=cmd)
    # Exclusive creation reserves the receipt even if another caller races.
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
    try:
        with log.open("x", encoding="utf-8") as stream:
            result = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT)
        record.update(status="COMPLETE" if result.returncode == 0 else "FAILED",
                      exit_code=result.returncode, end_unix=time.time())
    except BaseException as exc:
        record.update(status="INTERRUPTED", error=repr(exc), end_unix=time.time())
        save(receipt, record)
        raise
    save(receipt, record)
    print(record["status"], log, flush=True)
    raise SystemExit(record["exit_code"])


if __name__ == "__main__":
    main()
