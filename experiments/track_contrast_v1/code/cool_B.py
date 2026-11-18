"""Audited resource scheduling for this experiment's own formal-B container.

Default: one authorized 15-second pause followed by 30 seconds of observation.
--loop must be separately authorized; it pauses only on GPU1 thermal slowdown
with SM <1000 MHz, then allows at least 30 uninterrupted seconds after resume.
No model, optimizer, GPU configuration, or unrelated container is changed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import threading
import time

from telemetry import read_gpus, utc_now

E = Path(__file__).resolve().parents[1]
NAME = "track_contrast_formal_B"
EVENTS = E / "runtime/COOLING_B.jsonl"
STOP = threading.Event()


def docker(*args):
    return subprocess.run(["docker", *args], text=True, capture_output=True,
                          timeout=15, check=True).stdout


def inspect():
    try:
        value = json.loads(docker("inspect", NAME))[0]
    except subprocess.CalledProcessError:
        return None
    return value


def training():
    path = E / "training/B/run/TRAIN.jsonl"
    rows = []
    if path.exists():
        for line in path.read_text().splitlines()[-32:]:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    if not rows:
        return {"step": 0, "elapsed_seconds": None}
    out = {k: rows[-1][k] for k in ("step", "elapsed_seconds")}
    if len(rows) > 1:
        dt = rows[-1]["elapsed_seconds"] - rows[0]["elapsed_seconds"]
        out["recent_steps_per_second"] = (rows[-1]["step"] - rows[0]["step"]) / dt
    return out


def sample():
    gpu = next(g for g in read_gpus() if g["index"] == 1)
    return {"utc": utc_now(), "monotonic": time.monotonic(),
            "training": training(), "gpu": gpu}


def emit(event, **kwargs):
    value = {"utc": utc_now(), "event": event, "container": NAME, **kwargs}
    with EVENTS.open("a") as out:
        out.write(json.dumps(value, ensure_ascii=False) + "\n")
    print(json.dumps(value, ensure_ascii=False), flush=True)


def verify():
    receipt_path = E / "runtime/formal_B.json"
    receipt = json.loads(receipt_path.read_text())
    command = receipt["command"]
    assert command[command.index("--name") + 1] == NAME
    assert receipt["gpu"] == 1
    assert command[-3:] == [str(E / "code/pipeline.py"), "--condition", "B"]
    value = inspect()
    assert value and value["State"]["Running"] and not value["State"]["Paused"]
    assert value["Name"] == "/" + NAME
    assert value["Config"]["Cmd"] == command[-3:]
    assert value["Config"]["WorkingDir"] == str(E)
    devices = value["HostConfig"]["DeviceRequests"]
    assert len(devices) == 1 and devices[0]["DeviceIDs"] == ["1"]
    assert any(m["Source"] == str(E) and m["Destination"] == str(E) and m["RW"]
               for m in value["Mounts"])
    return {"container_id": value["Id"],
            "receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
            "gpu": 1, "command": value["Config"]["Cmd"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: STOP.set())
    proof = verify()
    emit("verified_owned_container", proof=proof, mode="loop" if args.loop else "single")
    paused_by_us = False
    try:
        while not STOP.is_set():
            state = inspect()
            if not state or not state["State"]["Running"]:
                emit("container_finished")
                break
            if (E / "runtime/STOP_COOLING_B").exists():
                emit("stop_marker")
                break
            before = sample()
            if args.loop and not (before["gpu"]["thermal_active"]
                                  and before["gpu"]["sm_clock_mhz"] < 1000):
                STOP.wait(5)
                continue
            emit("before_pause", sample=before, pause_seconds=15)
            docker("pause", NAME)
            paused_by_us = True
            emit("paused", sample=sample())
            STOP.wait(15)
            docker("unpause", NAME)
            paused_by_us = False
            resumed = sample()
            emit("resumed", sample=resumed)
            after = resumed
            # At least 30 seconds without another pause, sampled every 5 sec.
            for _ in range(6):
                if STOP.wait(5):
                    break
                after = sample()
                emit("post_resume_sample", sample=after)
            elapsed = after["monotonic"] - resumed["monotonic"]
            progressed = after["training"]["step"] - resumed["training"]["step"]
            emit("cycle_summary", pre_pause=before, resumed=resumed, after=after,
                 measured_post_resume_seconds=elapsed, post_resume_steps=progressed,
                 post_resume_steps_per_second=progressed / elapsed if elapsed else 0)
            if not args.loop:
                break
    finally:
        if paused_by_us:
            state = inspect()
            if state and state["State"]["Paused"]:
                docker("unpause", NAME)
                emit("finally_unpaused")
        emit("controller_finished")


if __name__ == "__main__":
    main()
