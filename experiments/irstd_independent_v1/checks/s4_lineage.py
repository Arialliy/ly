"""Resolve frozen S4 runtime settings and the origin of inherited diagnostics."""
import hashlib
import json
from pathlib import Path


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runtime_chunks(protocol, mode):
    if mode == "M0":
        return None, None
    runtime = protocol["proposal"]["evidence_runtime"]
    chunks = (runtime["query_chunk"], runtime["velocity_chunk"])
    if any(type(value) != int or value <= 0 for value in chunks):
        raise ValueError("Invalid evidence chunk dimensions in the frozen protocol")
    return chunks


def parent_run(run_dir, protocol):
    """Return the parent only when the child's two frozen parent digests and run identity agree."""
    migration = protocol["migration"]
    name = migration["parent_run_id"]
    if not isinstance(name, str) or name in ("", ".", "..") or Path(name).name != name or name == Path(run_dir).name:
        raise ValueError("Invalid S4 parent run identifier")
    parent = Path(run_dir).parent / name
    if (_sha(parent / "protocol.json") != migration["parent_protocol_sha256"]
            or _sha(parent / "execution_binding.json") != migration["parent_execution_binding_sha256"]):
        raise ValueError("Inherited parent protocol or execution binding changed")
    lock = json.loads((parent / "execution_binding.json").read_bytes())
    if lock["run_id"] != name or lock["common_protocol_sha256"] != migration["parent_protocol_sha256"]:
        raise ValueError("Parent execution binding belongs to a different run or protocol")
    return parent


def diagnostic_origin(run_dir, protocol, mode, step):
    """Return (source run, source protocol digest); pin the original bytes of inherited records."""
    run = Path(run_dir)
    migration = protocol.get("migration")
    if not migration or mode not in migration["modes"] or step > migration["common_step"]:
        return run, _sha(run / "protocol.json")
    parent = parent_run(run, protocol)
    key = str(step)
    hashes = migration["modes"][mode]["diagnostics"]
    path = parent / mode / "seed46" / f"diagnostics_{step}.json"
    if key not in hashes or _sha(path) != hashes[key]:
        raise ValueError(f"Missing or changed inherited training diagnostic at {mode} step {step}")
    return parent, migration["parent_protocol_sha256"]
