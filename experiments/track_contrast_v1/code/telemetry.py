"""Read-only 30-second NVIDIA telemetry; independent of experiment training code.

No GPU allocation, configuration, or process management is performed. Stop by
creating runtime/STOP_GPU_TELEMETRY or runtime/TASK_COMPLETE, or sending SIGTERM.
The two files are markers only; this script never creates them.
"""
import argparse
import csv
import datetime as dt
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time


FIELDS = (
    "index", "uuid", "temperature.gpu", "clocks.current.sm",
    "utilization.gpu", "memory.used", "memory.total", "power.draw",
    "clocks_throttle_reasons.active",
    "clocks_throttle_reasons.sw_thermal_slowdown",
    "clocks_throttle_reasons.hw_thermal_slowdown",
    "clocks_throttle_reasons.hw_slowdown",
    "clocks_throttle_reasons.sw_power_cap",
)
NAMES = (
    "index", "uuid", "temperature_c", "sm_clock_mhz", "utilization_percent",
    "memory_used_mib", "memory_total_mib", "power_w", "throttle_active_hex",
    "sw_thermal_slowdown", "hw_thermal_slowdown", "hw_slowdown", "sw_power_cap",
)


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def read_gpus():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=" + ",".join(FIELDS),
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10, check=True,
    )
    gpus = []
    for row in csv.reader(io.StringIO(result.stdout), skipinitialspace=True):
        if len(row) != len(NAMES):
            raise ValueError("Unexpected NVIDIA CSV field count: " + repr(row))
        gpu = dict(zip(NAMES, (v.strip() for v in row)))
        for key in ("index", "temperature_c", "sm_clock_mhz", "utilization_percent",
                    "memory_used_mib", "memory_total_mib"):
            try:
                gpu[key] = int(gpu[key])
            except ValueError:
                pass  # Preserve explicit N/A or unsupported readings.
        try:
            gpu["power_w"] = float(gpu["power_w"])
        except ValueError:
            pass
        gpu["thermal_active"] = any(
            gpu[key].lower() == "active"
            for key in ("sw_thermal_slowdown", "hw_thermal_slowdown")
        )
        # Active mask can also contain ordinary idle or power-cap reasons.
        # Do not interpret a nonzero mask as evidence of thermal slowdown.
        gpus.append(gpu)
    return gpus


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path,
                        default=Path(__file__).resolve().parents[1] / "runtime")
    parser.add_argument("--interval", type=float, default=30)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("interval must be positive")
    args.runtime.mkdir(parents=True, exist_ok=True)
    pid_path = args.runtime / "GPU_TELEMETRY.pid"
    status_path = args.runtime / "GPU_TELEMETRY_STATUS.json"
    output_path = args.runtime / "GPU_TELEMETRY.jsonl"
    alert_path = args.runtime / "GPU_THERMAL_ALERTS.jsonl"
    markers = [args.runtime / "STOP_GPU_TELEMETRY", args.runtime / "TASK_COMPLETE"]
    stop_event = threading.Event()
    stop_reason = ""

    def signal_stop(signum, _frame):
        nonlocal stop_reason
        stop_reason = "signal:" + str(signum)
        stop_event.set()

    signal.signal(signal.SIGTERM, signal_stop)
    signal.signal(signal.SIGINT, signal_stop)
    pid_path.write_text(str(os.getpid()) + "\n")
    start = utc_now()
    sample_count = 0
    thermal_sample_count = 0
    status = {"pid": os.getpid(), "status": "RUNNING", "started_utc": start,
              "interval_seconds": args.interval, "read_only": True,
              "output": str(output_path), "stop_markers": [str(p) for p in markers]}
    atomic_json(status_path, status)
    try:
        with output_path.open("a", buffering=1) as output:
            while not stop_event.is_set():
                found = next((p for p in markers if p.exists()), None)
                if found is not None:
                    stop_reason = "marker:" + str(found)
                    break
                started = time.monotonic()
                record = {"utc": utc_now(), "sample": sample_count, "pid": os.getpid()}
                try:
                    record["gpus"] = read_gpus()
                    hot = [gpu for gpu in record["gpus"] if gpu["thermal_active"]]
                    if hot:
                        thermal_sample_count += 1
                        with alert_path.open("a") as alerts:
                            alerts.write(json.dumps({"utc": record["utc"],
                                "sample": sample_count, "gpus": hot}) + "\n")
                except Exception as error:
                    record["error"] = type(error).__name__ + ": " + str(error)
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                sample_count += 1
                status.update(last_sample=record, sample_count=sample_count,
                              thermal_sample_count=thermal_sample_count)
                atomic_json(status_path, status)
                if args.once:
                    stop_reason = "one_sample_requested"
                    break
                stop_event.wait(max(0, args.interval - (time.monotonic() - started)))
    finally:
        status.update(status="STOPPED", ended_utc=utc_now(),
                      stop_reason=stop_reason or "stop_event", sample_count=sample_count,
                      thermal_sample_count=thermal_sample_count)
        atomic_json(status_path, status)
        if pid_path.exists() and pid_path.read_text().strip() == str(os.getpid()):
            pid_path.unlink()


if __name__ == "__main__":
    main()
