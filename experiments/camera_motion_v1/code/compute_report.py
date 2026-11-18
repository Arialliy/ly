#!/usr/bin/env python3
"""Report measured compute components without inventing deployment timings.

Run after C1/C2 COMPLETE.json and each development motion/windows.csv exist.
O/S rows attribute existing union measurements to five-window arrangements;
they are not additional executions. UNION8 reports actual shared cache work.
Only direct source-to-anchor matches belong to deployment. Adjacent-pair
diagnostic measurements and original training-source measurements are excluded.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
OUTPUT = BASE / "evaluation/COMPUTE.csv"
SEQUENCES = ["DJI_0026_2", "DJI_0067_2"]
PLANS = {"O": [0, 36, 72, 108, 120], "S": [0, 24, 60, 96, 120]}
UNION = sorted(set(PLANS["O"] + PLANS["S"]))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_records(sequence):
    records = {}
    provenance = {}
    for condition in ["C1", "C2"]:
        path = BASE / "predictions" / condition / sequence / "COMPLETE.json"
        receipt = read_json(path)
        assert receipt["status"] == "COMPLETE" and receipt["condition"] == condition
        assert receipt["sequence"] == sequence and receipt["all_model_state_unchanged"]
        assert receipt["training"] is False and receipt["model_labels_access"] is False
        assert receipt["parameters"] == 70913
        assert receipt["O"] == PLANS["O"] and receipt["S"] == PLANS["S"]
        assert receipt["starts_union"] == UNION
        by_start = {int(row["start"]): row for row in receipt["window_timing"]}
        assert sorted(by_start) == UNION and len(receipt["window_timing"]) == 8
        for row in by_start.values():
            assert type(row["usable_window"]) is bool
            for key in ["model_seconds", "restoration_fallback_seconds", "shared_geometry_cache_seconds", "write_seconds"]:
                assert math.isfinite(row[key]) and row[key] >= 0, (sequence, condition, key)
            if not row["usable_window"]:
                assert row["model_seconds"] == 0 and row["peak_cuda_allocated_bytes"] == 0
        records[condition] = by_start
        provenance[str(path.relative_to(BASE))] = digest(path)
    for start in UNION:
        left, right = records["C1"][start], records["C2"][start]
        for field in ["usable_window", "valid_fraction", "shared_geometry_cache_seconds", "motion_sha256"]:
            assert left[field] == right[field], (sequence, start, field)
    path = BASE / "motion" / sequence / "windows.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        motion = list(csv.DictReader(stream))
    assert len(motion) == 8 * 40
    pairs = {}
    for start in UNION:
        rows = [r for r in motion if int(r["window_start"]) == start]
        assert len(rows) == 40
        assert {int(r["source_frame"]) for r in rows} == set(range(start, start + 40))
        assert all(int(r["destination_frame"]) == start + 19 for r in rows)
        direct = [r for r in rows if int(r["source_frame"]) != int(r["destination_frame"])]
        assert len(direct) == 39
        for row in direct:
            seconds = float(row["seconds"])
            assert math.isfinite(seconds) and seconds >= 0
        pairs[start] = direct
    unique = {(int(r["source_frame"]), int(r["destination_frame"])) for rows in pairs.values() for r in rows}
    assert len(unique) == 312
    provenance[str(path.relative_to(BASE))] = digest(path)
    return records, pairs, provenance


def make_row(sequence, condition, plan, starts, records, pairs, witness, provenance, weight_bytes):
    conditions = ["C1", "C2"] if condition == "C1+C2" else [condition]
    candidates = [records[c][s] for c in conditions if c != "C0" for s in starts]
    shared = [records["C1"][s] for s in starts] if condition != "C0" else []
    direct = [row for s in starts for row in pairs[s]] if condition != "C0" else []
    trusted = sum(int(r["usable_window"]) for r in shared)
    candidate_passes = sum(int(r["usable_window"]) for r in candidates)
    model_seconds = sum(r["model_seconds"] for r in candidates)
    peak = max((r["peak_cuda_allocated_bytes"] for r in candidates), default=0)
    return dict(
        sequence=sequence, condition=condition, arrangement=plan,
        measurement_scope="actual_union_cache_work" if plan == "UNION8" else "five_window_attribution_from_union_cache",
        starts=";".join(map(str, starts)), unique_windows=len(starts), frames=160,
        candidate_conditions=len([c for c in conditions if c != "C0"]),
        trusted_geometry_windows=trusted if condition != "C0" else "not_applicable",
        whole_window_fallbacks=len(starts) - trusted if condition != "C0" else 0,
        whole_window_fallback_condition_instances=sum(not r["usable_window"] for r in candidates),
        geometric_valid_fraction_mean=sum(r["valid_fraction"] for r in shared) / len(shared) if shared else "not_applicable",
        actual_candidate_detector_passes=candidate_passes,
        actual_original_detector_passes_during_controls=0,
        reused_unique_C0_windows=len(starts),
        required_original_detector_passes_for_deployment=len(starts),
        required_total_detector_passes_for_deployment=len(starts) + candidate_passes,
        original_path_cost_status="required_in_deployment_even_when_C0_cached_offline",
        candidate_model_and_output_transfer_seconds=model_seconds,
        candidate_model_timer_scope="tiled_network_CPU_output_transfers_and_sigmoid; excludes_input_normalization_and_H2D",
        GIM_direct_pair_equivalents=len(direct),
        GIM_unique_direct_pairs=len({(int(r["source_frame"]), int(r["destination_frame"])) for r in direct}),
        GIM_match_and_fit_seconds=sum(float(r["seconds"]) for r in direct),
        GIM_timer_scope="matching_input_transfers_output_correspondences_and_geometry_fit_validation; excludes_model_load_and_pair_file_write",
        adjacent_diagnostic_pairs_charged_to_deployment=0,
        shared_geometry_and_C0_cache_io_seconds=sum(r["shared_geometry_cache_seconds"] for r in shared),
        shared_geometry_timer_scope="once_per_window_for_both_C1_C2; includes_motion_and_C0_cache_hash_load_and_prepare; excludes_validity_mask_write",
        restoration_and_valid_pixel_fallback_seconds=sum(r["restoration_fallback_seconds"] for r in candidates),
        per_window_prediction_compression_write_seconds=sum(r["write_seconds"] for r in candidates),
        candidate_peak_cuda_allocated_bytes=peak,
        peak_memory_scope="detector_maximum_not_sum; GIM_peak_not_recorded",
        original_window0_current_environment_seconds=witness["seconds"],
        original_window0_witness_count=1,
        original_full_arrangement_seconds="unmeasured",
        total_deployment_wall_seconds="unknown",
        deployment_speedup_vs_C0="not_estimated",
        missing_total_timing_components="full_original_arrangement; initialization; candidate_input_normalization_H2D; rejected_window_base_copy; mask_write; fused_write; hashing; fusion; scheduling",
        GIM_pretrained_weight_bytes=weight_bytes if condition != "C0" else 0,
        detector_parameters=70913,
        allocation_rule="GIM_and_geometry_shared_between_C1_C2; use_C1+C2_UNION8_for_joint_offline_totals; O_plus_S_duplicates_starts0_and120",
        provenance_sha256=json.dumps(provenance, separators=(",", ":"), sort_keys=True),
    )


def main():
    # Validate all expected receipts before replacing a previous usable report.
    runtime_path = BASE / "checks/RUNTIME_EQUIVALENCE.json"
    runtime = read_json(runtime_path)
    assert runtime["status"] == "PASS"
    witnesses = {r["sequence"]: r for r in runtime["records"]}
    assert set(witnesses) == set(SEQUENCES)
    assert all(w["window"] == 0 and w["seconds"] >= 0 for w in witnesses.values())
    weight_path = BASE / "resources/gim_dkm_100h.ckpt"
    weight_bytes = weight_path.stat().st_size
    rows = []
    for sequence in SEQUENCES:
        records, pairs, provenance = load_records(sequence)
        provenance[str(runtime_path.relative_to(BASE))] = digest(runtime_path)
        provenance["code/compute_report.py"] = digest(Path(__file__))
        for plan, starts in {**PLANS, "UNION8": UNION}.items():
            for condition in ["C0", "C1", "C2"]:
                rows.append(make_row(sequence, condition, plan, starts, records, pairs,
                                     witnesses[sequence], provenance, weight_bytes))
        rows.append(make_row(sequence, "C1+C2", "UNION8", UNION, records, pairs,
                             witnesses[sequence], provenance, weight_bytes))
    assert len(rows) == 20
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temp = OUTPUT.with_suffix(".csv.tmp")
    try:
        with temp.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temp.replace(OUTPUT)
    finally:
        temp.unlink(missing_ok=True)
    print(json.dumps({"status": "complete", "path": str(OUTPUT), "rows": len(rows),
                      "sha256": digest(OUTPUT), "total_deployment_time": "unknown_no_full_baseline_measurement"}))


if __name__ == "__main__":
    main()
