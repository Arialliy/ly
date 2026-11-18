"""Freeze the S4 training draws, evaluation populations and decisions before any model update."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory


EXPERIMENT = Path(__file__).resolve().parents[1]
PROJECT = EXPERIMENT.parents[1]
sys.path.insert(0, str(PROJECT))

import torch
import yaml

from experiments.irstd_independent_v1.data.s4_core_schedule import diagnostic_cores, make_schedule, training_index
from experiments.irstd_independent_v1.models.hypothesis_detector import HypothesisEvidenceDetector
from experiments.irstd_independent_v1.models.video_baseline import VideoBaseline


SEEDS = (46,)
MODES = {"M0": None, "Msep": "additive_only", "Mraw": "raw_joint", "Mint": "interaction"}
DESIGN = PROJECT / "1/S4_research_protocol.md"
PROPOSAL = PROJECT / "1/s4_protocol_proposal.yaml"


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bound_prerequisites():
    reports = EXPERIMENT / "reports"
    stages = {name: read(reports / (name + ".json"))
              for name in ("S0", "S1", "S2a", "S2b", "S2c", "S2", "S3")}
    require(all(stage["status"] == "PASS" for stage in stages.values()), "Previous stage has not passed")
    s2, s3 = stages["S2"], stages["S3"]
    require(s3["next_allowed_stage"] == "S4", "S3 has not released formal comparison")
    files = {f"reports/{name}.json": sha(reports / (name + ".json")) for name in stages}
    for name in ("S0", "S1", "S2a", "S2b", "S3"):
        row = stages[name]
        for key, digest in row.get("code_sha256", row.get("code_and_config_sha256")).items():
            require(sha(EXPERIMENT / key) == digest, f"Previous implementation changed: {name}/{key}")
            files[key] = digest
    checks = {item["path"]: item["sha256"] for item in s2["substage_reports"].values()}
    checks.update(s3["prerequisites"])
    checks[s2["checkpoint"]] = s2["checkpoint_sha256"]
    checks[s3["run_dir"] + "/interaction_step4.pt"] = s3["training"]["checkpoint_sha256"]
    checks["manifests/s2b_train12.json"] = stages["S2b"]["subset_sha256"]
    checks["manifests/nudt_mirsdt_mask17.json"] = stages["S0"]["data_manifest_hash"]
    for relative, digest in checks.items():
        require(sha(EXPERIMENT / relative) == digest, f"Previous artifact mismatch: {relative}")
        files[relative] = digest
    return files, stages["S0"]


def tensor_hashes(state):
    return {key: {"shape": list(t.shape), "dtype": str(t.dtype),
                  "sha256": hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()}
            for key, t in state.items()}


def prepare_initial(seed, folder):
    torch.manual_seed(seed)
    template = HypothesisEvidenceDetector(mode="interaction").cpu()
    common = ("frame_encoder.", "spatial_decoder.", "detection_head.")
    evidence_state = template.state_dict()
    states = {}
    for name, mode in MODES.items():
        torch.manual_seed(seed)
        model = VideoBaseline() if mode is None else HypothesisEvidenceDetector(mode=mode)
        original = model.state_dict()
        expected = 22921 if mode is None else 23945
        require(all(p.requires_grad for p in model.parameters())
                and sum(p.numel() for p in model.parameters()) == expected, "Unexpected S4 parameter contract")
        for key in original:
            if mode is not None or key.startswith(common):
                original[key] = evidence_state[key].detach().clone()
        model.load_state_dict(original, strict=True)
        states[name] = {key: val.detach().cpu().clone() for key,val in model.state_dict().items()}
    hashes = {k: tensor_hashes(v) for k,v in states.items()}
    require(hashes["Mraw"] == hashes["Msep"] == hashes["Mint"], "Evidence initial tensors are not paired")
    common_names = sorted(k for k in states["M0"] if k.startswith(common))
    require(common_names and all(hashes["M0"][k] == hashes["Mint"][k] for k in common_names),
            "M0 common layers are not paired")
    payload = {"format_version": 1, "seed": seed, "mode_to_evidence": MODES, "states": states}
    path = folder / f"init_{seed}.pt"
    with path.open("xb") as out:
        torch.save(payload, out)
    check = torch.load(path, map_location="cpu", weights_only=True)
    require(check["seed"] == seed and all(tensor_hashes(state) == hashes[name] for name,state in check["states"].items()),
            "Round-trip read of paired initialization failed")
    detail = folder / f"init_{seed}_tensors.json"
    write(detail, {"seed": seed, "tensors": hashes, "identical_M0_common_layers": common_names,
                   "all_evidence_mode_tensors_identical": True})
    return {"file": path.name, "sha256": sha(path), "tensors": detail.name, "tensors_sha256": sha(detail)}


def freeze(run_id):
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id) is None:
        raise ValueError("Expected a single new run-directory name")
    proposal = yaml.safe_load(PROPOSAL.read_bytes())
    require(proposal["initialization"]["seeds"] == [46, 47, 48]
            and list(proposal["models"]) == list(MODES)
            and all(proposal["models"][k]["mode"] == m for k,m in MODES.items()),
            "S4 document and model/seed implementation disagree")
    # The user's later explicit single-seed directive supersedes the suggested
    # three-seed design. Apply it before building any executable configuration.
    override = {"authority":"user", "original_proposal_seeds":[46,47,48], "formal_seeds":[46],
                "scope":"train the four models at seed 46 only; between-seed stability is unevaluated"}
    proposal["initialization"]["seeds"] = list(SEEDS)
    proposal["optimization"]["extension"].update(
        decision_after="all_four_seed46_models_finish_24000", apply_to="all_four_seed46_models")
    research = proposal["research"]
    for field in ("minimum_mean_gain_percentage_points", "positive_direction_required_in_all_three_seeds",
                  "report_seed_mean_std_min_and_each_seed", "downgrade_stability_if_leave_one_source_out_flips_mean_sign"):
        research.pop(field)
    research.update(minimum_single_seed_gain_percentage_points=1.0, report_single_seed_paired_deltas=True,
                    between_seed_stability="not_evaluated_single_seed", leave_one_source_out_sign_changes_reported=True)
    proposal.update(status="MATERIALIZED_AWAITING_EXECUTION_BINDING",can_launch_training=False,
                    user_override=override,design_document=str(DESIGN.relative_to(PROJECT)))
    runtime = proposal["evidence_runtime"]
    require(runtime["query_chunk"] == 64 and runtime["velocity_chunk"] == 8
            and runtime["checkpoint_chunks"] and not runtime["preserve_rng_state"], "Runtime settings changed")
    require(proposal["sampling"]["schedule_generated_through_step"] == 36000
            and proposal["optimization"]["default_updates"] == 24000
            and proposal["optimization"]["extended_updates"] == 36000, "Unexpected budget")
    bound,s0 = bound_prerequisites()
    require(bound["configs/s1_formal.yaml"] == runtime["grid_config_sha256"]
            and bound["manifests/nudt_mirsdt_mask17.json"] == proposal["data"]["manifest_sha256"],
            "Proposal grid/data binding differs from S0/S3")
    reports = EXPERIMENT / "reports"
    runs = reports / "s4_runs"
    runs.mkdir(exist_ok=True)
    dest = runs / run_id
    config = EXPERIMENT / "configs/s4_protocol.yaml"
    lock_path = reports / "S4_protocol_lock.json"
    if any(p.exists() for p in (dest,config,lock_path)):
        raise FileExistsError("Existing S4 run/config/lock; existing freeze will never be replaced")
    manifest = read(EXPERIMENT / "manifests/nudt_mirsdt_mask17.json")
    records, train_sources, verified = training_index(PROJECT, manifest, s0)
    dev = [s for s in manifest["sequences"] if s["split"] == "dev"]
    test = [s for s in manifest["sequences"] if s["split"] == "test"]
    def center_ids(rows):
        return [f"{s['id']}/{n:05d}" for s in rows for n in s["centers"]]
    dev_ids, test_ids = center_ids(dev), center_ids(test)
    monitor_sources = [f"Sequence{n}" for n in proposal["monitoring"]["development_monitor_sources"]]
    monitor = [f"{name}/{n:05d}" for name in monitor_sources for n in proposal["monitoring"]["development_monitor_center_ids"]]
    require(len(dev_ids) == len(set(dev_ids)) == 672 and len(test_ids) == len(set(test_ids)) == 1680
            and len(set(monitor).intersection(dev_ids)) == 32 and set(monitor_sources) == {s["id"] for s in dev},
            "Development or historical test manifest does not match the proposal")
    diag = diagnostic_cores(records, train_sources)
    require(len(diag) == 144 and len(train_sources) == 72 and len(dev) == 8 and len(test) == 20,
            "Incorrect S4 populations")
    require(s0["pixel_audit"]["splits"]["dev"]["empty_centers"] == 0
            and s0["pixel_audit"]["splits"]["test"]["empty_centers"] == 214, "S0 population changed")
    specs = {str(p.relative_to(PROJECT)): sha(p) for p in (DESIGN, PROPOSAL)}
    implementation = {str(p.relative_to(EXPERIMENT)): sha(p) for p in (
        Path(__file__), EXPERIMENT/"data/s4_core_schedule.py")}
    version = proposal["base_commit"]
    with TemporaryDirectory(prefix=".materialize-", dir=runs) as td:
        folder = Path(td) / "run"
        folder.mkdir()
        write(folder / "data_verification.json", {"status":"PASS","verified_source_counts":{"train":72,"dev":8},
              "opened_official_test_files":0, "source_aggregates":verified,
              "method":"SHA256 of ordered raw PNG hex digests plus newlines, relative to the frozen S0 manifest."})
        write(folder / "train_diagnostics.json", diag)
        write(folder / "evaluation_centers.json", {"development":dev_ids,"development_monitor":monitor,
                                                   "historical_test_manifest_ids_only":test_ids})
        diag_hashes = {n:sha(folder/n) for n in ("train_diagnostics.json","evaluation_centers.json")}
        init = {}
        schedules = {}
        for seed in SEEDS:
            payload = make_schedule(records, seed)
            require(payload["role_cycle"] == proposal["sampling"]["role_cycle"], "Incorrect role ordering")
            schedule_path = folder / f"schedule_{seed}.json"
            with schedule_path.open("x") as out:
                json.dump(payload, out, ensure_ascii=False, separators=(",",":"), allow_nan=False)
                out.write("\n")
            schedules[str(seed)] = sha(schedule_path)
            init[str(seed)] = prepare_initial(seed, folder)
        require(set(init) == set(schedules) == {"46"}, "Only seed 46 may be materialized")
        jobs = [f"{name}/seed{seed}" for name in MODES for seed in SEEDS]
        protocol = {"stage":"S4", "state":"MATERIALIZED", "can_launch_training":False,
                    "frozen_utc":datetime.now(timezone.utc).isoformat(), "run_id":run_id,
                    "run_dir":str(dest.relative_to(EXPERIMENT)), "base_commit":version,
                    "proposal_sha256":specs, "code_sha256":implementation, "prerequisite_sha256":bound,
                    "modes":list(MODES), "mode_to_evidence":MODES, "seeds":list(SEEDS), "user_override":override,
                    "fit_centers":diag, "dev_ids":dev_ids, "test_ids":test_ids, "diagnostic_dev_ids":monitor,
                    "dev_sources":monitor_sources, "test_sources":[s["id"] for s in test],
                    "initial_steps":24000, "max_steps":36000, "schedule_sha256":schedules,
                    "diagnostics_sha256":diag_hashes, "initialization":init, "formal_run_ids":jobs,
                    "data_verification_sha256":sha(folder/"data_verification.json"),
                    "proposal":proposal}
        write(folder/"protocol.json",protocol)
        materialized = dict(proposal)
        materialized.update(status="MATERIALIZED_AWAITING_EXECUTION_BINDING", can_launch_training=False,
                            design_document=str(DESIGN.relative_to(PROJECT)), design_sources_sha256=specs,
                            launch_authority="reports/S4_protocol_lock.json and execution binding")
        materialized["sampling"] = dict(proposal["sampling"], schedule_hashes=schedules)
        materialized["monitoring"] = dict(proposal["monitoring"], diagnostic_manifest_hashes=diag_hashes)
        materialized["lock_artifacts"] = {"implementation_base_commit":version, "prerequisite_code_sha256":implementation,
              "common_protocol_sha256":sha(folder/"protocol.json"), "schedules_by_seed":schedules,
              "diagnostics_sha256":diag_hashes, "initial_states_by_seed":init, "formal_run_ids":jobs}
        config_bytes = yaml.safe_dump(materialized,allow_unicode=True,sort_keys=False).encode()
        lock = {"stage":"S4", "status":"AWAITING_EXECUTION_BINDING", "can_launch_training":False,
                "run_id":run_id,"run_dir":protocol["run_dir"],"config":str(config.relative_to(EXPERIMENT)),
                "config_sha256":hashlib.sha256(config_bytes).hexdigest(),"common_protocol_sha256":sha(folder/"protocol.json"),
                "specification_sha256":specs, "schedules_by_seed":schedules,"diagnostics_sha256":diag_hashes,
                "initial_states_by_seed":init, "prerequisites":bound, "protocol_code_sha256":implementation,
                "seeds":list(SEEDS), "user_override":override,
                "formal_run_ids":jobs}
        fd = os.open(runs, os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            if any(p.exists() for p in (dest,config,lock_path)):
                raise FileExistsError("Concurrent or prior formal S4 run")
            folder.rename(dest)
            with config.open("xb") as out:
                out.write(config_bytes)
            write(lock_path,lock)
        finally:
            os.close(fd)
    return dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id",required=True)
    args = parser.parse_args()
    path = freeze(args.run_id)
    print(f"Materialized {path} protocol_sha256={sha(path/'protocol.json')}",flush=True)


