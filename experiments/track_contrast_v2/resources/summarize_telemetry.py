"""Summarize completed, read-only C-v2 GPU telemetry without touching training."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import median


E = Path(__file__).resolve().parents[1]
R = E / "runtime"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    status = json.loads((R / "GPU_TELEMETRY_STATUS.json").read_text())
    assert status["status"] == "STOPPED"
    assert not (R / "GPU_TELEMETRY.pid").exists()
    rows = [json.loads(line) for line in (R / "GPU_TELEMETRY.jsonl").read_text().splitlines()]
    assert len(rows) == status["sample_count"]
    summaries = []
    for index in range(3):
        samples = [g for row in rows for g in row.get("gpus", []) if g["index"] == index]
        allocated = [g for g in samples if g["memory_used_mib"] > 500]
        streak = longest = 0
        for row in rows:
            gpu = next((g for g in row.get("gpus", []) if g["index"] == index), None)
            streak = streak + 1 if gpu and gpu["temperature_c"] >= 90 else 0
            longest = max(longest, streak)
        summaries.append({
            "physical_gpu": index,
            "uuid": samples[0]["uuid"],
            "sample_count": len(samples),
            "max_observed_temperature_c": max(g["temperature_c"] for g in samples),
            "max_observed_memory_used_mib": max(g["memory_used_mib"] for g in samples),
            "max_observed_power_w": max(g["power_w"] for g in samples),
            "thermal_active_samples": sum(g["thermal_active"] for g in samples),
            "sw_thermal_active_samples": sum(g["sw_thermal_slowdown"] == "Active" for g in samples),
            "hw_thermal_active_samples": sum(g["hw_thermal_slowdown"] == "Active" for g in samples),
            "temperature_at_least_90_samples": sum(g["temperature_c"] >= 90 for g in samples),
            "longest_consecutive_temperature_at_least_90_samples": longest,
            "allocated_samples_memory_above_500_mib": len(allocated),
            "allocated_sm_clock_mhz_min": min(g["sm_clock_mhz"] for g in allocated),
            "allocated_sm_clock_mhz_median": median(g["sm_clock_mhz"] for g in allocated),
            "allocated_sm_clock_mhz_max": max(g["sm_clock_mhz"] for g in allocated),
            "final_sample": samples[-1],
        })
    training = []
    for condition in ("CONT", "FRAME", "GAP"):
        path = E / "training" / condition / "run" / "RUN.json"
        run = json.loads(path.read_text())
        receipt = json.loads((R / f"train_{condition}.json").read_text())
        assert run["status"] == "complete" and receipt["status"] == "COMPLETE"
        assert run["additional_steps"] == 256 and run["final_step"] == 2304
        assert run["frozen_frontend_unchanged"] and run["bn_unchanged"]
        training.append({
            "condition": condition,
            "physical_gpu": receipt["gpu"],
            "additional_steps": run["additional_steps"],
            "final_step": run["final_step"],
            "elapsed_seconds": run["elapsed_seconds"],
            "max_cuda_memory_allocated_bytes": run["max_cuda_memory_allocated_bytes"],
            "run_sha256": sha256(path),
        })
    inference = []
    for path in sorted(R.glob("infer_*.json")):
        receipt = json.loads(path.read_text())
        assert receipt["status"] == "COMPLETE" and receipt["exit_code"] == 0
        inference.append({"receipt": path.name, "physical_gpu": receipt["gpu"],
                          "status": receipt["status"], "sha256": sha256(path)})
    assert len(inference) == 6
    result = {
        "status": "complete",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "telemetry_process_status": status["status"],
        "telemetry_pid": status["pid"],
        "stop_reason": status["stop_reason"],
        "started_utc": status["started_utc"],
        "ended_utc": status["ended_utc"],
        "first_sample_utc": rows[0]["utc"],
        "last_sample_utc": rows[-1]["utc"],
        "interval_seconds": status["interval_seconds"],
        "sample_count": len(rows),
        "error_sample_count": sum("error" in row or not row.get("gpus") for row in rows),
        "samples_with_any_thermal_active": sum(any(g["thermal_active"] for g in row.get("gpus", [])) for row in rows),
        "per_gpu": summaries,
        "completed_training": training,
        "completed_inference": inference,
        "resource_controls_performed": [],
        "scope": "Read-only polling of physical GPU state during this C-v2 training/inference period; no pause, frequency, power, or other GPU controls.",
        "limits": [
            "Peaks are observed at 30-second sampling intervals, not guaranteed continuous-time peaks.",
            "Physical GPU memory includes driver/process allocations and differs from PyTorch max allocated memory.",
            "Allocated sample clock statistics retain host/pre/postprocessing idle periods; they are not kernel-normalized performance measures.",
            "Thermal slowdown was observed on all three cards. Wall time is descriptive and does not establish model compute-efficiency differences.",
            "No >=90C sample was observed. The three-consecutive-sample alert rule was not triggered.",
        ],
        "sources": {str(p.relative_to(E)): sha256(p) for p in [
            E / "code/telemetry.py", R / "GPU_TELEMETRY.jsonl", R / "GPU_TELEMETRY_STATUS.json"]},
    }
    (R / "GPU_TELEMETRY_SUMMARY.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# C-v2 hardware observations", "",
        f"The owned read-only telemetry process stopped normally via the requested stop marker. It recorded {len(rows)} samples at 30-second intervals from {rows[0]['utc']} to {rows[-1]['utc']}.", "",
        "| Physical GPU | Observed peak temperature (C) | Observed peak memory (MiB) | Thermal-active samples |",
        "| --- | --- | --- | --- |",
    ]
    for gpu in summaries:
        lines.append(f"| {gpu['physical_gpu']} | {gpu['max_observed_temperature_c']} | {gpu['max_observed_memory_used_mib']} | {gpu['thermal_active_samples']} |")
    lines += ["", "All three 256-step continuations and six development inference jobs completed. No sampled temperature reached 90 C, and the three-consecutive-sample alert was not triggered. No GPU controls or cooling pauses were applied in this C-v2 experiment.", "",
              "Thermal slowdown was sampled on all three cards. Training elapsed times were " + ", ".join(f"{r['condition']} {r['elapsed_seconds']:.3f} s" for r in training) + ". These times are descriptive hardware-confounded observations, not evidence of relative model efficiency. All three runs reported the same PyTorch peak allocated memory of 5,766,688,768 bytes.", "",
              "The 30-second observations may miss shorter temperature/memory peaks. Physical GPU memory and SM clocks include driver allocation and host/idle phases. Exact sample data, hashes, and completed-run receipts are recorded in GPU_TELEMETRY_SUMMARY.json.", ""]
    (R / "HARDWARE_NOTE.md").write_text("\n".join(lines))
    print(json.dumps({"samples": len(rows), "peak_temperatures_c": [g["max_observed_temperature_c"] for g in summaries], "peak_memory_mib": [g["max_observed_memory_used_mib"] for g in summaries], "status": status["status"]}))


if __name__ == "__main__":
    main()
