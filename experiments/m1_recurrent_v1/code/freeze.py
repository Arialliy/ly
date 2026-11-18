"""Freeze the complete paired M1 identity after actual engineering checks pass."""
from pathlib import Path
import xml.etree.ElementTree as ET
import torch
import runtime as rt


def main():
    path = rt.E / "PROTOCOL_FREEZE.json"
    if path.exists():
        raise FileExistsError("Protocol already frozen; do not silently replace it")
    preflight_path = rt.E / "checks/GPU_PREFLIGHT.json"
    preflight = rt.read_json(preflight_path)
    assert preflight["status"] == "PASS", "Actual GPU integration must pass first"
    assert preflight["source_sha256"] == rt.source_hashes(), "Preflight source identity changed"
    dcn_path = rt.E / "upstream/DCN_SMOKE.json"
    assert rt.read_json(dcn_path)["status"] == "PASS"
    unit_path = rt.E / "checks/CPU_UNIT.xml"
    suites = ET.parse(unit_path).getroot()
    testcases = suites.findall(".//testcase")
    assert testcases and not suites.findall(".//failure") and not suites.findall(".//error")
    assert not suites.findall(".//skipped"), "Required unit tests must actually execute"
    rt.ensure_initialization()
    initial = torch.load(rt.INITIAL, map_location="cpu", weights_only=True)
    identity = dict(
        status="FROZEN", date="2026-10-14", preflight_status="PASS",
        steps=rt.STEPS, main_step=rt.STEPS, seed=46, conditions=rt.MODES,
        exposure_plan={"entries": 256, "repetitions": 8},
        source_sha256=rt.source_hashes(), protocol_document_sha256=rt.sha(rt.E / "PROTOCOL.md"),
        initial_checkpoint_sha256=rt.sha(rt.INITIAL),
        initial_model_state_sha256=initial["model_state_sha256"],
        original_state_sha256=initial["original_state_sha256"],
        release_checkpoint_sha256=rt.sha(rt.CP),
        model_settings=rt.MODEL_SETTINGS, settings=rt.SETTINGS, runtime_settings=rt.RUNTIME_SETTINGS,
        cache_sha256=rt.frozen_cache_manifest(),
        checks_sha256={str(p): rt.sha(p) for p in (preflight_path, dcn_path, unit_path)},
        optimizer={"name": "fresh_Adam", "lr": 1e-5, "weight_decay": 1e-4,
                   "betas": [.9, .999], "eps": 1e-8},
        image="sha256:8470a68886ff2f480d76858d930fb72d0e680e69276fc1d068fe8b6625b4cb9f",
        state_boundary="independent40-frame windows, M0=X0, one forward direction",
        inference_spatial_policy="frozen frontend core128 halo16; global512x640 M1; pointwise head temporal chunk1",
        evaluation={"plans": rt.PLANS, "threshold": .5, "common_U_budget": 4015,
                    "strict": 5200, "low": 4342, "extreme": 4170, "protected": 646,
                    "confirmation_clips_used": False})
    rt.save_json(path, identity)
    print(path, flush=True)


if __name__ == "__main__":
    main()
