"""Finish the authorized paired endpoint after both durable lane workers finish.

This host-side worker starts no training. It invokes the already frozen scorer,
then renders a report from measured tables. It has no external notifications.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

E = Path(__file__).resolve().parents[1]
STATUS = E / "runtime/WORKFLOW.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(value):
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATUS)


def verify():
    freeze = read(E / "PROTOCOL_FREEZE.json")
    assert freeze["status"] == "FROZEN" and freeze["steps"] == 2048
    for path, expected in freeze["source_sha256"].items():
        assert sha(Path(path)) == expected, f"Frozen source changed: {path}"
    assert sha(E / "PROTOCOL.md") == freeze["protocol_document_sha256"]
    assert (E / "runtime/render_report.py").is_file()
    return freeze


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    verify()
    if args.check_only:
        print("Frozen sources and report entrypoint verified", flush=True)
        return
    with (E / "runtime/summary_worker.lock").open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if STATUS.exists():
            raise FileExistsError("Workflow already exists; inspect before restarting")
        report_script = E / "runtime/render_report.py"
        record = {"status": "WAITING_FOR_LANES", "pid": os.getpid(), "start_unix": time.time(),
                  "worker_sha256": sha(Path(__file__)), "report_script_sha256": sha(report_script),
                  "protocol_freeze_sha256": sha(E / "PROTOCOL_FREEZE.json"),
                  "main_step": 2048, "conditions": ["Current", "History"], "actions": [],
                  "new_training_after_endpoints": False, "next_module_automatic": False}
        save(record)
        try:
            while True:
                states = {}
                for condition in record["conditions"]:
                    path = E / "runtime" / f"{condition}_WORKFLOW.json"
                    states[condition] = read(path) if path.exists() else {"status": "STARTING"}
                record["lanes"] = {key: value["status"] for key, value in states.items()}
                record["last_checked_unix"] = time.time()
                save(record)
                if any(value["status"].startswith(("FAIL", "ERROR", "STOP")) for value in states.values()):
                    raise RuntimeError(f"Lane failed; inspect lane receipt: {record['lanes']}")
                if all(value["status"] == "COMPLETE" for value in states.values()):
                    break
                time.sleep(30)
            verify()
            assert sha(report_script) == record["report_script_sha256"]
            assert read(E / "checks/PAIRED_TRAINING.json")["status"] == "PASS"
            actions = [
                [sys.executable, str(E / "code/launch.py"), "--tag", "summarize_final",
                 "evaluate.py", "--summary"],
                [sys.executable, str(report_script)],
            ]
            for name, command in zip(("SUMMARIZING", "RENDERING_REPORT"), actions):
                record["status"] = name
                action = {"command": command, "start_unix": time.time()}
                record["actions"].append(action)
                save(record)
                result = subprocess.run(command, cwd=E)
                action.update(exit_code=result.returncode, end_unix=time.time())
                save(record)
                if result.returncode:
                    raise RuntimeError(f"{name} failed with exit code {result.returncode}")
            outputs = [E / "RESULTS.md", E / "RESULTS.json",
                       E / "evaluation/step_2048/SUMMARY_COMPLETE.json"]
            record.update(status="COMPLETE", end_unix=time.time(),
                          outputs_sha256={str(path.relative_to(E)): sha(path) for path in outputs})
            save(record)
            print("M1 paired2048 training, inference, evaluation and report COMPLETE", flush=True)
        except BaseException as exc:
            record.update(status="FAILED", error=repr(exc), traceback=traceback.format_exc(),
                          end_unix=time.time())
            save(record)
            raise


if __name__ == "__main__":
    main()
