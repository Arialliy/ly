"""Bind the materialized S4 inputs to a recoverable snapshot of the executed code."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


EXP = Path(__file__).resolve().parents[1]
PROJECT = EXP.parents[1]
CANONICAL = EXP / "reports/S4_protocol_lock.json"
CONFIG = EXP / "configs/s4_protocol.yaml"
WORKER_FILES = (
    "checks/run_s4_local.py", "checks/s4_binding.py", "checks/s4_compare.py",
    "engine/checkpoints.py", "engine/s4_models.py", "engine/s4_checkpoints.py",
    "engine/s4_train.py", "engine/predict.py", "checks/s2c_prediction.py",
    "evaluation/s4_predict.py", "evaluation/s4_metrics.py", "evaluation/s2b_froc.py",
    "losses/s4_detection.py",
)
AUTHOR_FILES = ("ShootingRules.py", "test.py", "write_results.py", "data_utils/TestDataLoader.py")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def check(condition, message):
    if not condition:
        raise ValueError(message)


def atomic(path, data):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        tmp = Path(stream.name)
        try:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)


def check_files(base, expected):
    for name, digest in expected.items():
        if sha(base / name) != digest:
            raise ValueError(f"S4 dependency changed: {base / name}")


def materialized_inputs(run, protocol):
    """The historical population is identified from its manifest, never opened here."""
    check(run.name == protocol["run_id"] and protocol["seeds"] == [46]
          and protocol["modes"] == ["M0", "Msep", "Mraw", "Mint"]
          and (protocol["initial_steps"], protocol["max_steps"]) == (24000, 36000),
          "Unknown S4 design or seed")
    check_files(PROJECT, protocol["proposal_sha256"])
    check_files(EXP, protocol["prerequisite_sha256"])
    check_files(EXP, protocol["code_sha256"])
    init = protocol["initialization"]["46"]
    artifacts = dict(protocol["diagnostics_sha256"], **{
        "protocol.json": sha(run / "protocol.json"),
        "schedule_46.json": protocol["schedule_sha256"]["46"],
        "data_verification.json": protocol["data_verification_sha256"],
        init["file"]: init["sha256"], init["tensors"]: init["tensors_sha256"],
    })
    audit = read(run / "schedule_verification.json")
    audited_protocol = artifacts["protocol.json"]
    if "migration" in protocol:
        from .s4_migrate import verify_parent
        audited_protocol = verify_parent(run, protocol)
    check(audit["status"] == "PASS" and audit["protocol_sha256"] == audited_protocol
          and audit["schedule_sha256"] == artifacts["schedule_46.json"], "Invalid schedule audit")
    artifacts["schedule_verification.json"] = sha(run / "schedule_verification.json")
    check_files(run, artifacts)
    return artifacts


def verify_execution(run_dir):
    run = Path(run_dir)
    protocol = read(run / "protocol.json")
    lock_file = run / "execution_binding.json"
    check(sha(lock_file) == sha(CANONICAL), "Run and project execution locks disagree")
    lock = read(lock_file)
    check(lock["status"] == "FROZEN" and lock["can_launch_training"] is True
          and lock["run_id"] == run.name and lock["seeds"] == [46]
          and lock["common_protocol_sha256"] == sha(run / "protocol.json"), "Incorrect execution lock")
    check(sha(CONFIG) == sha(run / "effective_config.yaml") == lock["config_sha256"],
          "The effective S4 configuration changed")
    check(materialized_inputs(run, protocol) == lock["input_sha256"], "S4 input binding mismatch")
    check_files(PROJECT, lock["code_sha256"])
    return protocol, lock


def source_paths(protocol):
    local = set(WORKER_FILES) | set(protocol["code_sha256"])
    if "migration" in protocol:
        local.update(("checks/s4_migrate.py", "checks/s4_lineage.py", "checks/s4_milestones.py"))
    local.update(k for k in protocol["prerequisite_sha256"] if k.endswith((".py", ".yaml")))
    # Bind the whole small production module sets, including imported utilities.
    for directory in ("models", "geometry", "data"):
        local.update(str(p.relative_to(EXP)) for p in (EXP / directory).glob("*.py"))
    local.add("contracts.py")
    files = {str((EXP / name).relative_to(PROJECT)) for name in local}
    files.update(str(Path("sources/official_deeppro") / name) for name in AUTHOR_FILES)
    files.update(protocol["proposal_sha256"])
    return sorted(files)


def git_snapshot(base, run, files):
    """An experiment ref pins immutable Git blobs without touching HEAD or the user's index."""
    def git(*args, env=None):
        return subprocess.check_output(["git", *args], cwd=PROJECT, env=env, text=True).strip()

    ref = f"refs/experiments/s4/{run.name}"
    with tempfile.TemporaryDirectory(dir=run) as tmp:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(tmp) / "index"))
        git("read-tree", base, env=env)
        git("add", "--", *files, env=env)
        tree = git("write-tree", env=env)
    old = subprocess.run(["git", "rev-parse", "--verify", "--quiet", ref],
                         cwd=PROJECT, capture_output=True, text=True)
    if old.returncode == 0:
        commit = old.stdout.strip()
        check(git("rev-parse", f"{commit}^{{tree}}") == tree, "A different S4 source snapshot already exists")
    else:
        commit = git("commit-tree", tree, "-p", base, "-m", f"Freeze S4 execution sources for {run.name}")
        git("update-ref", ref, commit, "0" * 40)
    blobs = git("ls-tree", "-r", commit, "--", *files).splitlines()
    contents = {entry.split("\t", 1)[1]: entry.split()[2] for entry in blobs}
    for name in files:
        data = (PROJECT / name).read_bytes()
        git_blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        check(contents.get(name) == git_blob, f"Executable differs from archived Git source: {name}")
    return ref, commit


def bind(run):
    import yaml

    check(run.resolve().parent == EXP / "reports/s4_runs", "Unexpected S4 run directory")
    fd = os.open(run, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if read(CANONICAL).get("can_launch_training"):
            return verify_execution(run)[1]
        protocol = read(run / "protocol.json")
        old = read(CANONICAL)
        inputs = materialized_inputs(run, protocol)
        check(old["run_id"] == run.name and old["common_protocol_sha256"] == inputs["protocol.json"]
              and old["status"] == "AWAITING_EXECUTION_BINDING", "Unexpected preparation lock")
        initial_config = run / "prepared_config.yaml"
        if not initial_config.exists():
            check(sha(CONFIG) == old["config_sha256"], "Prepared configuration changed")
            atomic(initial_config, CONFIG.read_bytes())
        check(sha(initial_config) == old["config_sha256"], "Preparation archive mismatch")
        paths = source_paths(protocol)
        code = {name: sha(PROJECT / name) for name in paths}
        ref, commit = git_snapshot(protocol["base_commit"], run, paths)
        config = yaml.safe_load(initial_config.read_bytes())
        config.update(status="FROZEN", can_launch_training=True)
        config["lock_artifacts"].update(implementation_commit=commit, implementation_ref=ref)
        config_bytes = yaml.safe_dump(config, allow_unicode=True, sort_keys=False).encode()
        lock = dict(old, status="FROZEN", can_launch_training=True,
                    prepared_config_sha256=old["config_sha256"],
                    config_sha256=hashlib.sha256(config_bytes).hexdigest(), code_sha256=code,
                    input_sha256=inputs, implementation_commit=commit, implementation_ref=ref)
        raw = (json.dumps(lock, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
        atomic(run / "effective_config.yaml", config_bytes)
        atomic(CONFIG, config_bytes)
        atomic(run / "execution_binding.json", raw)
        atomic(CANONICAL, raw)
        return verify_execution(run)[1]
    finally:
        os.close(fd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    lock = bind(args.run_dir.resolve()) if args.freeze else verify_execution(args.run_dir)[1]
    print(f"{lock['run_id']} {lock['status']} {lock['implementation_commit']}")
