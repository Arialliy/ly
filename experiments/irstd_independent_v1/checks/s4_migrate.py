"""Materialize a versioned S4 run from an authentic common optimizer boundary."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory

from . import s4_binding as b


RUNS = b.EXP / "reports/s4_runs"
EVIDENCE = ("Msep", "Mraw", "Mint")
NEW_CHUNKS = {"query_chunk": 256, "velocity_chunk": 81}


def name(value):
    b.check(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value), "Invalid run ID")
    return value


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def clock():
    return datetime.now(timezone.utc).isoformat()


def prefix(folder, step):
    with (folder / "updates.jsonl").open("rb") as stream:
        rows = []
        for n in range(1, step + 1):
            raw = stream.readline()
            b.check(raw.endswith(b"\n"), f"Missing complete parent update at {n}")
            item = json.loads(raw)
            b.check(item["step"] == n and item["finite_after_update"] is True, f"Invalid parent update {n}")
            rows.append(raw)
    return b"".join(rows)


def checkpoint_steps(folder):
    return {int(p.stem.split("_")[1]) for p in (folder / "checkpoints").glob("step_*.json")
            if p.with_suffix(".pt").is_file()}


def mode_record(parent, mode, step, protocol_sha, lock_sha):
    folder = parent / mode / "seed46"
    side = folder / "checkpoints" / f"step_{step:06d}.json"
    z = b.read(side)
    actual = b.sha(side.with_suffix(".pt"))
    ident = z["identity"]
    b.check((z["step"], z["checkpoint_sha256"], ident["mode"], ident["seed"],
             ident["protocol_sha256"], ident["execution_sha256"]) ==
            (step, actual, mode, 46, protocol_sha, lock_sha), f"Wrong upstream boundary: {side}")
    previous = b.read(parent / "protocol.json")["initialization"]["46"]["sha256"]
    for n in sorted(k for k in checkpoint_steps(folder) if k <= step):
        p = side.parent / f"step_{n:06d}.json"
        row = b.read(p)
        b.check(row["identity"] == ident and row["step"] == n and row["parent_checkpoint_sha256"] == previous
                and row["checkpoint_sha256"] == b.sha(p.with_suffix(".pt")), f"Broken upstream chain {p}")
        previous = row["checkpoint_sha256"]
    diagnostics = {str(n): b.sha(folder / f"diagnostics_{n}.json") for n in range(0, step + 1, 3000)}
    development = {str(n): b.sha(folder / f"dev_{n}" / "binding.json") for n in range(6000, step + 1, 6000)}
    return {"checkpoint_sha256": actual, "sidecar_sha256": b.sha(side),
            "journal_prefix_sha256": hashlib.sha256(prefix(folder, step)).hexdigest(),
            "diagnostics": diagnostics, "development": development}


def verify_parent(run, protocol):
    """Verify inherited inputs independently of which run owns the live global config."""
    mig = protocol["migration"]
    parent = RUNS / name(mig["parent_run_id"])
    old = b.read(parent / "protocol.json")
    source_lock = b.read(parent / "execution_binding.json")
    ps, es = b.sha(parent / "protocol.json"), b.sha(parent / "execution_binding.json")
    b.check(parent.name != run.name and mig["parent_protocol_sha256"] == ps == source_lock["common_protocol_sha256"]
            and mig["parent_execution_binding_sha256"] == es and source_lock["run_id"] == parent.name
            and b.sha(parent / "effective_config.yaml") == source_lock["config_sha256"], "Upstream protocol/lock changed")
    old_spec = deepcopy(old["proposal"])
    runtime = old_spec["evidence_runtime"]
    b.check((runtime["query_chunk"], runtime["velocity_chunk"], runtime["checkpoint_chunks"],
             runtime["preserve_rng_state"]) == (64, 8, True, False), "Unexpected upstream runtime")
    runtime.update(NEW_CHUNKS)
    b.check(old_spec == protocol["proposal"], "The child proposal changes more than the two chunk sizes")
    meta = {"run_id", "run_dir", "frozen_utc", "code_sha256", "proposal"}
    b.check({k:v for k,v in old.items() if k not in meta}
            == {k:v for k,v in protocol.items() if k not in meta | {"migration"}}, "The research protocol changed")
    b.check(set(protocol["code_sha256"]) == set(old["code_sha256"]) and
            all(protocol["code_sha256"][k] == h for k,h in old["code_sha256"].items()
                if k != "checks/s4_protocol.py"), "Changed schedule materialization logic")
    b.check(set(mig["modes"]) == set(EVIDENCE) and type(mig["common_step"]) is int and mig["common_step"] > 0,
            "Invalid common migration root")
    for mode, entry in mig["modes"].items():
        b.check(mode_record(parent, mode, mig["common_step"], ps, es) == entry, f"Parent evidence changed for {mode}")
    # The schedule and its independent verification are the original bytes.
    for fn, h in source_lock["input_sha256"].items():
        if fn != "protocol.json":
            b.check(b.sha(run / fn) == h, f"Inherited input changed: {fn}")
    b.check_files(b.PROJECT, mig["benchmark_sha256"])
    return ps


def materialize(parent, dest, benchmark):
    import yaml
    old = b.read(parent / "protocol.json")
    old_lock = b.read(parent / "execution_binding.json")
    ps, es = b.sha(parent / "protocol.json"), b.sha(parent / "execution_binding.json")
    b.check(es == b.sha(b.CANONICAL) and b.sha(b.CONFIG) == old_lock["config_sha256"],
            "Parent is not the currently bound run")
    b.check(old["seeds"] == [46] and "migration" not in old and not (parent / "M0/seed46/checkpoints").exists(),
            "This migration expects the original three active evidence lanes and a queued M0")
    b.check_files(parent, old_lock["input_sha256"])
    step = max(set.intersection(*(checkpoint_steps(parent / m / "seed46") for m in EVIDENCE)))
    b.check(step > 0, "No trained common checkpoint")
    modes = {m:mode_record(parent,m,step,ps,es) for m in EVIDENCE}
    report = b.read(benchmark / "verification.json")
    b.check(set(report["results"]) == set(EVIDENCE) and all(d["reference_matches_official_steps"] == 10
            for d in report["results"].values()), "Benchmark lacks paired validation on the official prefix")
    files = [benchmark / "REPORT.md", benchmark / "verification.json", benchmark / "launch.json",
             benchmark / "measurements/inputs.json", benchmark / "measurements/summary.json",
             *(benchmark / "measurements" / m / "comparison.json" for m in EVIDENCE)]
    mig = {"parent_run_id":parent.name, "parent_protocol_sha256":ps, "parent_execution_binding_sha256":es,
           "common_step":step, "first_new_step":step+1, "modes":modes,
           "previous_chunks":[64,8], "new_chunks":[256,81],
           "benchmark_sha256":{str(f.relative_to(b.PROJECT)):b.sha(f) for f in files}}
    p = deepcopy(old)
    p.update(run_id=dest.name, run_dir=str(dest.relative_to(b.EXP)), migration=mig, frozen_utc=clock())
    p["proposal"]["evidence_runtime"].update(NEW_CHUNKS)
    p["code_sha256"]["checks/s4_protocol.py"] = b.sha(b.EXP / "checks/s4_protocol.py")
    with TemporaryDirectory(dir=RUNS, prefix=".migrate-") as tmp:
        child = Path(tmp) / dest.name
        child.mkdir()
        for fn in old_lock["input_sha256"]:
            if fn != "protocol.json":
                shutil.copyfile(parent / fn, child / fn)
        b.atomic(child / "protocol.json", json_bytes(p))
        for mode in EVIDENCE:
            folder = child / mode / "seed46"
            folder.mkdir(parents=True)
            raw = prefix(parent / mode / "seed46", step)
            b.check(hashlib.sha256(raw).hexdigest() == modes[mode]["journal_prefix_sha256"], "Unstable upstream prefix")
            b.atomic(folder / "updates.jsonl", raw)
        inputs = b.materialized_inputs(child, p)
        paths = b.source_paths(p)
        code = {path:b.sha(b.PROJECT / path) for path in paths}
        ref, commit = b.git_snapshot(old_lock["implementation_commit"], child, paths)
        cfg = yaml.safe_load((parent / "prepared_config.yaml").read_bytes())
        cfg["evidence_runtime"].update(NEW_CHUNKS)
        cfg["migration"] = {k:mig[k] for k in ("parent_run_id","common_step","first_new_step","previous_chunks","new_chunks")}
        cfg["lock_artifacts"].update(common_protocol_sha256=inputs["protocol.json"],prerequisite_code_sha256=p["code_sha256"])
        def dump(): return yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False).encode()
        b.atomic(child / "prepared_config.yaml", dump())
        cfg.update(status="FROZEN",can_launch_training=True)
        cfg["lock_artifacts"].update(implementation_commit=commit,implementation_ref=ref)
        b.atomic(child / "effective_config.yaml", dump())
        lock = dict(old_lock,run_id=dest.name,run_dir=p["run_dir"],common_protocol_sha256=inputs["protocol.json"],
                    protocol_code_sha256=p["code_sha256"],prepared_config_sha256=b.sha(child/"prepared_config.yaml"),
                    config_sha256=b.sha(child/"effective_config.yaml"),code_sha256=code,input_sha256=inputs,
                    implementation_commit=commit,implementation_ref=ref,migration=mig)
        b.atomic(child / "execution_binding.json", json_bytes(lock))
        b.check(not dest.exists(), "Run directory already exists")
        child.rename(dest)
    return p


def activate(parent, dest):
    lock = b.read(dest / "execution_binding.json")
    p = b.read(dest / "protocol.json")
    b.check(p["migration"]["parent_run_id"] == parent.name and p["run_id"] == dest.name
            and lock["common_protocol_sha256"] == b.sha(dest/"protocol.json"), "Wrong migration destination")
    # Recover safely if activation was interrupted between the two atomic replacements.
    b.check(b.sha(b.CANONICAL) in (p["migration"]["parent_execution_binding_sha256"], b.sha(dest/"execution_binding.json")),
            "A different S4 run already owns the canonical lock")
    b.check(b.sha(b.CONFIG) in (b.sha(parent/"effective_config.yaml"), b.sha(dest/"effective_config.yaml")),
            "Unexpected live configuration")
    b.check(b.materialized_inputs(dest,p) == lock["input_sha256"] and b.sha(dest/"effective_config.yaml") == lock["config_sha256"],
            "Child input binding is incomplete")
    b.check_files(b.PROJECT,lock["code_sha256"])
    b.atomic(b.CONFIG,(dest/"effective_config.yaml").read_bytes())
    b.atomic(b.CANONICAL,(dest/"execution_binding.json").read_bytes())
    b.verify_execution(dest)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("parent_run_id")
    ap.add_argument("child_run_id")
    ap.add_argument("benchmark", type=Path)
    args = ap.parse_args()
    parent, dest = RUNS/name(args.parent_run_id), RUNS/name(args.child_run_id)
    fd = os.open(parent, os.O_RDONLY)
    try:
        # The existing S4 driver takes the same lock throughout its ownership of this run.
        fcntl.flock(fd,fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not dest.exists():
            materialize(parent,dest,args.benchmark.resolve())
        activate(parent,dest)
        print(dest.name, "FROZEN", "common step", b.read(dest/"protocol.json")["migration"]["common_step"])
    finally:
        os.close(fd)
