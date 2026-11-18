"""Join preserved real-training checks with the verified inference execution path.

No training is repeated and no failed check is relabelled. The initial direct
full-frame CUDA failure stays in GPU_PREFLIGHT_INITIAL.json. PASS applies to the
unchanged training forward and the separately verified frontend-tiled/global-M1
inference method only.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import runtime as rt


def ast_digest(node):
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def existing_class_members_unchanged(before_path, after_path):
    before = ast.parse(before_path.read_text(encoding="utf-8"))
    after = ast.parse(after_path.read_text(encoding="utf-8"))
    after_classes = {node.name: node for node in after.body if isinstance(node, ast.ClassDef)}
    rows = {}
    for old_class in (node for node in before.body if isinstance(node, ast.ClassDef)):
        new_class = after_classes[old_class.name]
        assert [ast_digest(node) for node in old_class.bases] == [ast_digest(node) for node in new_class.bases]
        assert [ast_digest(node) for node in old_class.decorator_list] == [ast_digest(node) for node in new_class.decorator_list]
        for old_node in old_class.body:
            digest = ast_digest(old_node)
            if isinstance(old_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                new_node = next(node for node in new_class.body
                                if isinstance(node, type(old_node)) and node.name == old_node.name)
                assert ast_digest(new_node) == digest, (old_class.name, old_node.name)
                rows[f"{old_class.name}.{old_node.name}"] = digest
            else:
                assert digest in {ast_digest(node) for node in new_class.body}, old_class.name
    # Imports and all original top-level non-class statements must be retained.
    for node in before.body:
        if not isinstance(node, ast.ClassDef):
            assert ast_digest(node) in {ast_digest(value) for value in after.body}
    assert "WindowRecurrentAdapter.forward" in rows and "FrozenDeepProWithAdapter.forward" in rows
    return rows


def validate_training(initial):
    assert initial["status"] == "FAIL", "Original full-frame failure must stay visible"
    assert initial["paired_initialization_and_actual_exposure_identical"]
    assert initial["fullframe"]["Current"]["status"] == "FAIL"
    assert "illegal memory access" in initial["fullframe"]["Current"]["error"]
    assert not initial["fullframe"]["Current"]["cuda_out_of_memory"]
    assert rt.sha(rt.INITIAL) == initial["initial_checkpoint_sha256"]
    assert rt.sha(rt.CP) == initial["release_checkpoint_sha256"]
    for condition in rt.CONDITIONS:
        result = initial["training"][condition]
        assert result["status"] == "PASS"
        assert result["zero_initialization"]["feature_and_logit_exact"] == [True, True]
        assert result["original_parameters_and_bn_unchanged"] and result["initial_state_file_unmodified"]
        assert result["optimized_state_saved"] is False
        assert [row["plan_index"] for row in result["steps"]] == [0, 1]
        assert result["steps"][0]["group_gradient_norms"]["output_projection"] > 0
        assert all(value == 0 for name, value in result["steps"][0]["group_gradient_norms"].items()
                   if name != "output_projection")
        assert all(value > 0 for value in result["steps"][1]["group_gradient_norms"].values())
        assert all(value > 0 for value in result["steps"][1]["group_update_norms"].values())
    left, right = (initial["training"][condition] for condition in rt.CONDITIONS)
    assert left["initial_model_state_sha256"] == right["initial_model_state_sha256"]
    for a, b in zip(left["steps"], right["steps"]):
        for key in ("input_sha256", "label_sha256", "auxiliary_input_sha256", "complete_input_tensor_sha256"):
            assert a[key] == b[key]


def validate_spatial(spatial):
    assert spatial["status"] == "PASS"
    assert spatial["source_sha256"] == spatial["source_sha256_final"]
    assert spatial["source_split"] == "training"
    assert spatial["global_m1_spatial_tiling"] is False
    assert (spatial["frontend_core"], spatial["frontend_halo"], spatial["head_temporal_chunk"]) == (128, 16, 1)
    for path, digest in spatial["source_sha256_final"].items():
        assert rt.sha(path) == digest, ("Source changed after spatial test", path)
    for condition in rt.CONDITIONS:
        small = spatial["small_grid"][condition]
        assert small["status"] == "PASS" and small["active_Wout"]
        assert small["logit_change_from_zero_Wout"] > 1e-4
        assert small["original_parameters_and_bn_unchanged"]
        large = spatial["fullframe"][condition]
        assert large["status"] == "PASS" and large["global_m1_single_call"]
        assert large["input_shape"] == [1, 1, 40, 512, 640]
        assert large["threshold05_disagreements"] == 0
        assert large["recurrent_call_count"] == 39
        assert all(row == [[1, 16, 512, 640], [1, 16, 512, 640]] for row in large["recurrent_call_shapes"])
        assert large["outputs_finite"] and large["all_model_state_unchanged"]


def main():
    checks = rt.E / "checks"
    destination = checks / "GPU_PREFLIGHT.json"
    preserved = checks / "GPU_PREFLIGHT_INITIAL.json"
    if not preserved.exists():
        original_bytes = destination.read_bytes()
        original = json.loads(original_bytes)
        assert original["status"] == "FAIL" and "training" in original
        with preserved.open("xb") as stream:
            stream.write(original_bytes)
    else:
        assert destination.read_bytes() == preserved.read_bytes(), "An aggregate already exists; do not overwrite it"
    initial = rt.read_json(preserved)
    validate_training(initial)
    spatial_path = checks / "SPATIAL_PREFLIGHT.json"
    spatial = rt.read_json(spatial_path)
    validate_spatial(spatial)
    unit_path = checks / "CPU_UNIT.xml"
    suite = ET.parse(unit_path).getroot()
    assert len(suite.findall(".//testcase")) == 36
    for kind in ("failure", "error", "skipped"):
        assert not suite.findall(f".//{kind}"), kind
    unit_metadata_path = checks / "CPU_UNIT_FINAL_METADATA.json"
    unit = rt.read_json(unit_metadata_path)
    assert unit["status"] == "PASS" and unit["tests"] == 36
    for path, digest in unit["source_sha256"].items():
        assert rt.sha(path) == digest, ("Source changed after CPU tests", path)

    snapshots = checks / "training_path_before_spatial"
    adapter_path = rt.ROOT / "models/adapters/window_recurrent_adapter.py"
    continuity = {}
    for path in (rt.E / "code/runtime.py", rt.E / "code/train.py", adapter_path):
        old_path = snapshots / path.name
        old_digest = rt.sha(old_path)
        assert old_digest == initial["source_sha256"][str(path)], path
        continuity[str(path)] = {"tested_source_sha256": old_digest, "final_source_sha256": rt.sha(path),
                                 "preserved_source": str(old_path)}
        if path != adapter_path:
            assert old_digest == rt.sha(path), ("Training source changed", path)
    continuity[str(adapter_path)]["original_member_ast_sha256"] = existing_class_members_unchanged(
        snapshots / adapter_path.name, adapter_path)
    unchanged_training_sources = {}
    auxiliary_changes = {}
    for path_string, digest in initial["source_sha256"].items():
        path = Path(path_string)
        current = rt.sha(path)
        if path == adapter_path:
            continue
        if path.parent == rt.E / "code" and path.name not in ("runtime.py", "train.py"):
            if current != digest:
                auxiliary_changes[path_string] = {"before": digest, "after": current}
            continue
        assert current == digest, ("Frozen training dependency changed", path)
        unchanged_training_sources[path_string] = digest
    dcn_path = rt.E / "upstream/DCN_SMOKE.json"
    assert rt.read_json(dcn_path)["status"] == "PASS"

    source_manifest = rt.source_hashes()
    result = {
        "status": "PASS", "kind": "validated_training_plus_registered_inference_execution",
        "finalized_unix": time.time(), "source_sha256": source_manifest,
        "training": initial["training"], "training_validation_origin": str(preserved),
        "paired_initialization_and_actual_exposure_identical": True,
        "initial_checkpoint_sha256": initial["initial_checkpoint_sha256"],
        "release_checkpoint_sha256": initial["release_checkpoint_sha256"],
        "model_settings": rt.MODEL_SETTINGS, "runtime_settings": rt.RUNTIME_SETTINGS,
        "supervision_settings": rt.SETTINGS,
        "original_training_methods_unchanged": True, "source_continuity": continuity,
        "unchanged_training_dependency_sha256": unchanged_training_sources,
        "nontraining_source_changes": auxiliary_changes,
        "inference": spatial,
        "direct_fullframe_execution": {
            "status": "FAIL_PRESERVED_NOT_USED", "origin": str(preserved),
            "recorded_failure": initial["fullframe"],
            "interpretation": "CUDA illegal memory access occurred in released frontend before M1; not classified as OOM",
            "registered_replacement_execution": "frozen frontend core128/halo16; one global M1 call; pointwise head temporalchunk1",
            "operator_replacement": False,
        },
        "cpu_tests": {"status": "PASS", "executed": 36, "skipped": 0,
                      "scope": "Explicit test-only recurrence for adapter state/wrapper contracts"},
        "learning_evidence_scope": "At step2 each of P/PDA/TSFM/Wout has nonzero loss gradient and actual update; not every parameter. Step1 upstream deltas can be weight decay only.",
        "custom_dcn_bitwise_determinism_claimed": False,
        "engineering_gate_only": True, "weak_target_recovery_demonstrated": False,
        "evidence_sha256": {str(path): rt.sha(path) for path in
                            (preserved, spatial_path, unit_path, unit_metadata_path, dcn_path)},
    }
    rt.save_json(destination, result)
    print(json.dumps({"status": result["status"], "output": str(destination),
                      "initial_failed_check_preserved": str(preserved)}), flush=True)


if __name__ == "__main__":
    main()
