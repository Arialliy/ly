#!/usr/bin/env python3
"""Finite frozen-Plus implementation witness; not detector training/evaluation.

Uses the first pre-existing TRAIN plan entry, first window, with its crop origin
rounded down to an even absolute Haar phase and a fixed 97x101 crop. No dev
labels, matching, threshold, checkpoint selection or formal optimization runs.
One SGD update exists solely to check gate-to-operator gradient activation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

from framewise_adapter import (FramewiseFeatureAdapter,
                               FrozenDeepProPlusWithSpatialReference,
                               spatially_tiled_forward)
from haar_reference import SpatialLFPReference

HERE = Path(__file__).resolve().parent
Q = HERE.parent
ROOT = Q.parent
SOURCE = ROOT / "official_deeppro"
SNAPSHOT = SOURCE / "log/sem_seg/RGB-T__2025-01-14_15-27__SoftLoUloss_DeepPro-Plus_DataL40"
CHECKPOINT = SNAPSHOT / "checkpoints/best_model.pth"
EXPECTED_SOURCE = "fa46a034a30d8bbb0f677271d6368f033fcb389d682478d331551d8bae087668"
EXPECTED_CHECKPOINT = "4fd4aa2736d9a33f2bf1459099068d8b4a7e202ff80aecdb4a3277fd13c7abcc"
# Locked implementation-check tolerances, not task/metric acceptance thresholds.
ATOL = 2e-4
RTOL = 2e-5


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def comparison(a, b):
    result = {"shape": list(a.shape), "other_shape": list(b.shape),
              "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
              "bitwise_value_equal": bool(torch.equal(a, b)),
              "max_abs_error": float((a-b).abs().max()),
              "atol": ATOL, "rtol": RTOL}
    result["passed"] = result["finite"] and bool(torch.allclose(a, b, atol=ATOL, rtol=RTOL))
    return result


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def run(args, result):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(46)
    np.random.seed(46)
    torch.manual_seed(46)
    torch.cuda.manual_seed_all(46)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    source_hash = sha(SNAPSHOT / "DeepPro-Plus.py")
    checkpoint_hash = sha(CHECKPOINT)
    assert source_hash == EXPECTED_SOURCE, "Frozen Plus source hash changed"
    assert checkpoint_hash == EXPECTED_CHECKPOINT, "Frozen release checkpoint hash changed"
    helper = Q / "top1_background_control_20261011/code/train_pilot.py"
    spec = importlib.util.spec_from_file_location("reference_original_loader", helper)
    helper_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper_module)
    original, loaded_source = helper_module.load_model(SOURCE, CHECKPOINT)
    assert loaded_source == SNAPSHOT / "DeepPro-Plus.py"
    original = original.to(device).eval()
    initial = {name: value.detach().cpu().clone() for name, value in original.state_dict().items()}
    assert sum(p.numel() for p in original.parameters()) == 70913
    plan_path = Q / "stage1_goal/pilot/TRAIN_PLAN.json"
    plan = json.loads(plan_path.read_text())
    entry = plan["entries"][0]["pairs"][0]
    assert entry["sequence"] in plan["train_sequences"]
    raw_path = Q / "stage1_goal/data" / entry["sequence"] / "raw_frames.npy"
    raw_hash = sha(raw_path)
    assert raw_hash == plan["data_fingerprints"][entry["sequence"]]["raw_sha256"]
    frames = np.load(raw_path, mmap_mode="r", allow_pickle=False)
    t0 = entry["starts"][0]
    cy, cx = [v // 2 * 2 for v in entry["crop_yx"]]
    raw = np.array(frames[t0:t0+40, cy:cy+97, cx:cx+101], copy=True)
    assert raw.shape == (40, 97, 101) and raw.dtype == np.uint8
    x = torch.from_numpy((raw.astype(np.float32)-85.0799)/47.4845)[None, None].to(device)
    result.update({
        "device": str(device), "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda, "seed": 46,
        "dtype": "float32", "amp": False, "tf32": False,
        "checkpoint_sha256": checkpoint_hash, "model_source_sha256": source_hash,
        "loader_sha256": sha(helper), "plan_sha256": sha(plan_path),
        "original_state_initial_sha256": state_hash(initial),
        "source_modules": {str(path): sha(path) for path in [
            SOURCE / "networks/layers/basic.py", SOURCE / "networks/layers/TPro.py"]},
        "code_sha256": {path.name: sha(path) for path in [
            HERE / "haar_reference.py", HERE / "framewise_adapter.py", Path(__file__)]},
        "input": {"split": "TRAIN", "sequence": entry["sequence"],
                  "window_start_index": t0, "window_length": 40,
                  "original_plan_crop_yx": entry["crop_yx"],
                  "actual_crop_yx": [cy, cx], "crop_hw": [97, 101],
                  "raw_sha256": raw_hash, "normalized_tensor_sha256": state_hash({"x": x})},
        "no_dev_data": True, "detector_training_run": False,
        "detector_metrics_evaluated": False, "formal_checkpoint_written": False,
    })
    with torch.no_grad():
        expected = original(x)
    adapter = FramewiseFeatureAdapter(SpatialLFPReference(32), 32, 8).to(device)
    model = FrozenDeepProPlusWithSpatialReference(original, adapter).to(device)
    model.train()
    assert not original.training and all(not p.requires_grad for p in original.parameters())
    with torch.no_grad():
        before_adapter = original.layer1(original.conv_in(x))
        after_adapter = adapter(before_adapter)
        result['zero_gate_feature_layout'] = {
            'before_stride': list(before_adapter.stride()),
            'after_stride': list(after_adapter.stride()),
            'same_values': bool(torch.equal(before_adapter, after_adapter)),
            'same_stride': before_adapter.stride() == after_adapter.stride(),
        }
        assert result['zero_gate_feature_layout']['same_values']
        assert result['zero_gate_feature_layout']['same_stride']
        at_zero = model(x)
    zero = [comparison(a, b) for a, b in zip(expected, at_zero)]
    result["zero_gate_original_outputs"] = zero
    assert all(row["passed"] and row["bitwise_value_equal"] for row in zero)
    result["original_parameters"] = 70913
    result["new_parameters"] = sum(p.numel() for p in adapter.parameters())

    # One finite derivative witness, deliberately unrelated to detection labels.
    optimizer = torch.optim.SGD(adapter.parameters(), lr=1e-4, weight_decay=0)
    optimizer.zero_grad(set_to_none=True)
    objective = model(x)[1].square().mean()
    objective.backward()
    gate_grad = adapter.scale.grad
    assert gate_grad is not None and torch.isfinite(gate_grad) and gate_grad.abs() > 0
    first_grads = {name: {"finite": bool(p.grad is not None and torch.isfinite(p.grad).all()),
                          "abs_sum": float(p.grad.abs().sum()) if p.grad is not None else None}
                   for name, p in adapter.operator.named_parameters()}
    assert first_grads and all(row["finite"] and row["abs_sum"] == 0 for row in first_grads.values())
    result["first_backward"] = {"objective": "mean(raw_logits ** 2); gradient witness only",
                                "value": float(objective.detach()),
                                "scale_grad": float(gate_grad), "operator": first_grads}
    optimizer.step()
    scale_after = float(adapter.scale.detach())
    assert scale_after != 0 and torch.isfinite(adapter.scale)
    optimizer.zero_grad(set_to_none=True)
    second_objective = model(x)[1].square().mean()
    second_objective.backward()
    active_grads = {name: {"finite": bool(p.grad is not None and torch.isfinite(p.grad).all()),
                           "abs_sum": float(p.grad.abs().sum()) if p.grad is not None else None}
                    for name, p in adapter.operator.named_parameters()}
    assert all(row["finite"] for row in active_grads.values())
    assert sum(row["abs_sum"] for row in active_grads.values()) > 0
    assert all(p.grad is None for p in original.parameters())
    result["one_sgd_update_gradient_witness"] = {
        "steps": 1, "lr": 1e-4, "weight_decay": 0,
        "scale_after": scale_after, "second_backward_operator": active_grads,
        "original_gradients_absent": True, "not_a_detection_training_result": True}
    optimizer.zero_grad(set_to_none=True)

    # Deliberately nonzero gate AND attention weights exercise the full RF.
    # These are fixed diagnostic perturbations, not learned detector weights.
    with torch.no_grad():
        adapter.scale.fill_(0.25)
        generator = torch.Generator(device="cpu").manual_seed(46)
        attention_probe = torch.randn(adapter.operator.attention.weight.shape,
                                      generator=generator) * 0.02
        adapter.operator.attention.weight.copy_(attention_probe.to(device))
        active_full = model(x)
        layer1 = original.layer1(original.conv_in(x))
        frame_checks = []
        adapter.frame_chunk = 40
        reference_frames = adapter(layer1)
        for chunk in [1, 8, 17]:
            adapter.frame_chunk = chunk
            record = comparison(reference_frames, adapter(layer1))
            record["frame_chunk"] = chunk
            frame_checks.append(record)
        adapter.frame_chunk = 8
        active_tiled = spatially_tiled_forward(model, x, core_hw=(47, 49), halo=20,
                                              absolute_origin_yx=(cy, cx))
    result["active_gate_frame_chunk_checks"] = frame_checks
    assert all(row["passed"] for row in frame_checks)
    active_effect = [comparison(a, b) for a, b in zip(expected, active_full)]
    assert any(row["max_abs_error"] > 0 for row in active_effect)
    result["active_gate_changes_output"] = {
        "scale": 0.25, "gate_tanh": float(torch.tensor(0.25).tanh()),
        "attention_probe": "seed46 CPU normal std0.02; diagnostic, not learned",
        "attention_probe_sha256": state_hash({"attention": attention_probe}),
        "max_abs_errors_vs_original": [row["max_abs_error"] for row in active_effect]}
    tile_checks = [comparison(a, b) for a, b in zip(active_full, active_tiled)]
    result["active_gate_spatial_tile_checks"] = {
        "core_hw": [47, 49], "halo": 20, "input_crop_hw": [97, 101],
        "absolute_origin_yx": [cy, cx], "even_input_tile_starts": True,
        "right_bottom_odd_global_boundary_preserved": True,
        "original_frontend_radius": 9, "reference_maximum_reach": 9,
        "output_checks": tile_checks}
    assert all(row["passed"] for row in tile_checks)
    final = {name: value.detach().cpu() for name, value in original.state_dict().items()}
    assert final.keys() == initial.keys()
    assert all(torch.equal(value, initial[name]) for name, value in final.items())
    assert all(not module.training for module in original.modules())
    result["original_state_final_sha256"] = state_hash(final)
    result["original_parameters_and_all_bn_buffers_unchanged"] = True
    result["original_eval_after_wrapper_train"] = True
    if device.type == "cuda":
        torch.cuda.synchronize()
        result["gpu_name"] = torch.cuda.get_device_name(device)
        result["max_cuda_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    result["status"] = "PASS_IMPLEMENTATION_CHECKS_ONLY"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"status": "RUNNING", "scope": "finite_reference_integration_only"}
    start = time.monotonic()
    try:
        run(args, result)
    except BaseException as error:
        result.update(status="FAILED", error=repr(error))
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - start
        save(args.output, result)
        print(json.dumps({key: result[key] for key in ["status", "elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
