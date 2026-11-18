"""Read-only review of new D/N execution, without repeating historical audits."""
import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics

E = Path(__file__).resolve().parents[1]
V2 = E.parent / "track_contrast_v2"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return True


def extent(values):
    return {"min": min(values), "median": statistics.median(values), "max": max(values)}


def main():
    freeze = read(E / "PROTOCOL_FREEZE.json")
    calibration = read(V2 / "CALIBRATION.json")
    reuse = read(E / "resources/F_REUSE_AUDIT.json")
    f_run = read(V2 / "training/FRAME/run/RUN.json")
    f_rows = [json.loads(line) for line in (V2 / "training/FRAME/run/TRAIN.jsonl").read_text().splitlines()]
    f_diagnostics = read(V2 / "training/FRAME/run/TRAIN_DIAGNOSTICS.json")
    config = calibration["fixed_settings"]
    # Small source hashes only; historical data/cache audits are reused.
    for path, expected in freeze["training_sources"].items():
        assert digest(Path(path)) == expected, path
    reviews = []
    evidence = [E/"PROTOCOL_FREEZE.json", E/"resources/F_REUSE_AUDIT.json", E/"resources/REAL_WEIGHTING_CHECK.json"]
    for condition in ("D", "N"):
        folder = E / "training" / condition / "run"
        run = read(folder / "RUN.json")
        receipt = read(E / "runtime" / f"train_{condition}.json")
        rows = [json.loads(line) for line in (folder / "TRAIN.jsonl").read_text().splitlines()]
        diagnostics = read(folder / "TRAIN_DIAGNOSTICS.json")
        assert run["status"] == "complete" and receipt["status"] == "COMPLETE" and receipt["exit_code"] == 0
        assert (run["start_step"], run["additional_steps"], run["final_step"]) == (2048, 256, 2304)
        assert run["initial_optimizer_restored"] and run["frozen_frontend_unchanged"] and run["bn_unchanged"]
        assert run["trainable_parameter_count"] == 14593 and run["total_parameters"] == 70913 and run["new_parameters"] == 0
        assert run["trainable_parameter_names"] == f_run["trainable_parameter_names"]
        for key in ("initial_checkpoint_sha256", "initial_optimizer_sha256", "initial_state_sha256",
                    "frozen_state_sha256", "bn_initial_sha256", "model_source_sha256", "settings", "optimizer", "seed", "precision"):
            assert run[key] == f_run[key], (condition, key)
        assert run["initial_checkpoint_sha256"] == freeze["initial_checkpoint_sha256"]
        assert run["initial_optimizer_sha256"] == calibration["initial_optimizer_sha256"]
        assert run["protocol_sha256"] == digest(E / "PROTOCOL_FREEZE.json")
        assert run["training_sources"] == freeze["training_sources"]
        assert run["readout"] == {"mode": {"D": "confidence", "N": "matched_confidence"}[condition],
                                  "gamma": 2., "group": "existing_segment8", "F_reference": "track_contrast_v2/FRAME"}
        assert [r["step"] for r in rows] == list(range(2049, 2305))
        assert [r["plan_index"] for r in rows] == list(range(256))
        assert [r["additional_step"] for r in rows] == list(range(1, 257))
        assert all(r["condition"] == condition and finite(r) for r in rows)
        formula_error, base_error, frame_error, budget_ratios = [], [], [], []
        empty = []
        for index, (row, prior) in enumerate(zip(rows, f_rows)):
            for key in ("input_sha256", "label_sha256", "auxiliary_input_sha256"):
                assert row[key] == prior[key], (condition, index, key)
            terms, w = row["loss_terms"], row["weighting"]
            formula_error.append(abs(row["loss"]-(terms["base"]+config["lambda_readout"]*terms["weighted_frame"])))
            base_error.append(abs(terms["base"]-(terms["det"]+config["lambda_keep"]*terms["keep"]+config["lambda_temporal"]*terms["C"])))
            frame_error.append(abs(terms["weighted_frame"]-terms["readout_positive"]-terms["readout_negative"]))
            assert terms["readout_positive"] == w["positive_loss"] and terms["readout_negative"] == w["negative_loss"]
            assert row["readout"]["observations_with_multiplicity"] == prior["readout"]["observations_with_multiplicity"] == w["observations_exposed"]
            unit_budget, budget = w["unit_negative_readout_gradient_l1"], w["weighted_negative_readout_gradient_l1"]
            assert min(unit_budget, budget) >= 0 and row["grad_norm"] >= 0
            if unit_budget > 0:
                ratio = budget/unit_budget
                budget_ratios.append(ratio)
                assert abs(ratio-1) < 1e-6 if condition == "N" else 0 <= ratio <= 1.000001
            if w["segments_active"] == 0:
                empty.append(index)
                assert terms["weighted_frame"] == terms["readout_positive"] == terms["readout_negative"] == 0
        assert max(formula_error) < 2e-6 and max(base_error) < 2e-6 and max(frame_error) < 4e-6
        assert len(empty) == 10
        assert [x["step"] for x in run["snapshots"]] == [2048, 2112, 2176, 2304]
        for snapshot in run["snapshots"]:
            assert digest(folder / snapshot["path"]) == snapshot["sha256"]
        assert diagnostics["indices"] == calibration["calibration_entry_indices"] and not diagnostics["uses_development"]
        assert [r["index"] for r in diagnostics["rows"]] == diagnostics["indices"] and finite(diagnostics)
        assert [r["readout"]["observations_with_multiplicity"] for r in diagnostics["rows"]] == [r["readout"]["observations_with_multiplicity"] for r in f_diagnostics["rows"]]
        changes = {prefix: math.sqrt(sum(v*v for k,v in run["parameter_changes"].items() if k.startswith(prefix)))
                   for prefix in ("TPro.", "conv_out1.", "conv_out2.")}
        assert all(v > 0 and math.isfinite(v) for v in changes.values())
        reviews.append({"condition": condition, "status": "PASS", "steps": len(rows), "main_step": run["final_step"],
            "all256_exposure_hashes_match_F": True, "all_logs_finite": True,
            "empty_segment_entries": empty, "common_lambda_readout": config["lambda_readout"],
            "loss_reconstruction_max_absolute_error": max(formula_error),
            "base_reconstruction_max_absolute_error": max(base_error),
            "weighted_frame_positive_negative_reconstruction_max_absolute_error": max(frame_error),
            "same_snapshot_negative_logit_budget_ratio_to_unit": extent(budget_ratios),
            "nonempty_entries": len(budget_ratios), "native_module_parameter_change_l2": changes,
            "initial_state_sha256": run["initial_state_sha256"], "final_state_sha256": run["final_state_sha256"],
            "final_state_equals_archived_F": run["final_state_sha256"] == f_run["final_state_sha256"],
            "elapsed_seconds": run["elapsed_seconds"], "max_cuda_memory_allocated_bytes": run["max_cuda_memory_allocated_bytes"],
            "frozen_frontend_unchanged": True, "bn_unchanged": True})
        evidence += [folder/"RUN.json", folder/"TRAIN.jsonl", folder/"TRAIN_DIAGNOSTICS.json", E/"runtime"/f"train_{condition}.json"]
    monitor_path = E / "evaluation/TRAINING_MONITOR_SAMPLES.json"
    monitor = read(monitor_path)
    gpu_summaries = []
    for index, condition in [(0, "D"), (1, "N")]:
        samples = [gpu for row in monitor["samples"] if row["conditions"][condition]["status"] == "running"
                   for gpu in row["gpus"] if gpu["index"] == index]
        if samples:
            gpu_summaries.append({"gpu": index, "condition": condition, "samples_while_running": len(samples),
                "peak_observed_temperature_c": max(g["temperature_c"] for g in samples),
                "peak_observed_memory_mib": max(g["memory_used_mib"] for g in samples),
                "sw_thermal_active_samples": sum(g["sw_thermal_slowdown"] == "Active" for g in samples),
                "hw_thermal_active_samples": sum(g["hw_thermal_slowdown"] == "Active" for g in samples)})
    evidence.append(monitor_path)
    result = {"status": "PASS", "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": "New D/N training execution and weighting diagnostics only; no historical complete audit rerun, no source changes or GPU interventions.",
        "conditions": reviews, "gpu_observations": gpu_summaries, "monitor_alarms": monitor["alarms"],
        "limitations": ["N budget ratios describe each branch's current snapshot versus its own unit-negative readout, not equality between evolved D/N/F models.",
            "Budget diagnostics are formula-based with actual cast weights; true autograd budget agreement was independently tested in the fixed-eight-batch preflight.",
            "Positive-term identity and detach were tested at common snapshots; endpoint models naturally have different positive logits.",
            "GPU observations begin during training, are sampled every30seconds and do not bound continuous-time peaks. Thermal variation prevents interpreting wall-time differences as model efficiency.",
            "Training losses/readout proxies do not establish actual detection recovery or background cost; those are evaluated separately."],
        "source_sha256": {str(p.relative_to(E)): digest(p) for p in evidence}}
    (E / "evaluation/TRAINING_LOG_REVIEW.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "conditions": reviews, "gpu_observations": gpu_summaries}, ensure_ascii=False))


if __name__ == "__main__":
    main()
