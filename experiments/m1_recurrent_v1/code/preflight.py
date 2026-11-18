"""Actual released DeepPro + official RFR DCN + frozen F supervision preflight.

Only disposable in-memory models are optimized. The paired initialization is
created/read through runtime.ensure_initialization and is never overwritten with
preflight updates. Full-frame feasibility uses training video frames, not dev or
confirmation frames. No spatial tiling or replacement DCN is used here.
"""
from __future__ import annotations

import argparse
import gc
import time
import traceback
from pathlib import Path

import numpy as np
import torch

import runtime as rt


GROUPS = {
    "input_projection": "adapter.in_projection.",
    "pda": "adapter.recurrent_cell.pda.",
    "tsfm": "adapter.recurrent_cell.tsfm.",
    "output_projection": "adapter.out_projection.",
}
HASH_KEYS = ("input_sha256", "label_sha256", "auxiliary_input_sha256")


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def timer_start():
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    return time.perf_counter()


def timing(start):
    torch.cuda.synchronize()
    return {
        "seconds": time.perf_counter() - start,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def parameter_groups(values):
    # Squared norms aggregate parameter contributions without cancellation.
    return {group: sum(value * value for name, value in values.items() if name.startswith(prefix)) ** 0.5
            for group, prefix in GROUPS.items()}


def check_official_cell(model):
    cell = model.adapter.recurrent_cell
    assert cell.__class__.__module__ == "experiments.m1_recurrent_v1.upstream.rfr_cell"
    assert cell.__class__.__name__ == "RFRCell"
    assert set(GROUPS) == set(parameter_groups({name: 0. for name, _ in model.named_parameters()}))
    assert any(name.startswith(GROUPS["pda"]) for name, _ in model.named_parameters())
    assert any(name.startswith(GROUPS["tsfm"]) for name, _ in model.named_parameters())


def verify_identity(model, batch, frozen):
    """Use exactly the frozen F microbatch2 execution for the actual eight windows."""
    model.eval()
    x = torch.from_numpy(batch["x"]).to(next(model.parameters()).device)
    assert tuple(x.shape) == (8, 1, 40, 128, 128)
    with torch.no_grad():
        began = timer_start()
        expected = tuple(value.cpu() for value in frozen.common.old.forward(rt.original(model), x))
        baseline_cost = timing(began)
        began = timer_start()
        actual = tuple(value.cpu() for value in frozen.common.old.forward(model, x))
        wrapped_cost = timing(began)
    exact = [torch.equal(a, b) for a, b in zip(actual, expected)]
    maximum_error = [float((a - b).abs().max()) for a, b in zip(actual, expected)]
    assert all(exact), f"Zero initialization changed released outputs: {maximum_error}"
    return {"status": "PASS", "shape": list(x.shape), "microbatch": 2,
            "feature_and_logit_exact": exact, "maximum_absolute_errors": maximum_error,
            "baseline": baseline_cost, "initialized_wrapper": wrapped_cost}


def train_condition(condition, state, output, report):
    frozen = rt.frozen_f()
    model, initialization = rt.load_initialized(condition)
    check_official_cell(model)
    result = {"status": "RUNNING", "condition": condition, "mode": rt.MODES[condition],
              "initial_model_state_sha256": rt.tensor_hash(model.state_dict()),
              "original_state_sha256": initialization["original_state_sha256"],
              "steps": []}
    report["training"][condition] = result
    rt.save_json(output, report)
    batch = frozen.data.batch_data(0, state)
    result["zero_initialization"] = verify_identity(model, batch, frozen)
    rt.assert_frozen(model, initialization["original_state_sha256"])
    rt.save_json(output, report)
    del batch
    model.train()
    criterion = rt.load_module("_m1_preflight_official_loss", rt.MODEL_SOURCE).SoftLoUloss().cuda()
    parameters = {name: value for name, value in model.named_parameters() if value.requires_grad}
    assert parameters and all(name.startswith("adapter.") for name in parameters)
    optimizer = torch.optim.Adam(parameters.values(), lr=1e-5, weight_decay=1e-4,
                                 betas=(.9, .999), eps=1e-8)
    for index in (0, 1):
        batch = frozen.data.batch_data(index, state)
        before = {name: value.detach().clone() for name, value in parameters.items()}
        optimizer.zero_grad(set_to_none=True)
        began = timer_start()
        terms, stats, features, logits, risk, cache = frozen.common.compute(
            model, batch, index, criterion, rt.SETTINGS)
        loss = terms["base"] + rt.SETTINGS["lambda_readout"] * terms["frame"]
        assert bool(torch.isfinite(loss))
        loss.backward()
        gradient_norms = {}
        for name, parameter in parameters.items():
            assert parameter.grad is not None, f"Missing gradient: {name}"
            assert bool(torch.isfinite(parameter.grad).all()), f"Nonfinite gradient: {name}"
            gradient_norms[name] = float(parameter.grad.norm())
        grouped_gradients = parameter_groups(gradient_norms)
        assert grouped_gradients["output_projection"] > 0
        if index == 0:
            assert all(grouped_gradients[name] == 0 for name in GROUPS if name != "output_projection")
        else:
            assert all(value > 0 for value in grouped_gradients.values()), grouped_gradients
        optimizer.step()
        cost = timing(began)
        update_norms = {name: float((parameter.detach() - before[name]).norm())
                        for name, parameter in parameters.items()}
        grouped_updates = parameter_groups(update_norms)
        assert grouped_updates["output_projection"] > 0
        if index == 1:
            assert all(value > 0 for value in grouped_updates.values()), grouped_updates
        rt.assert_frozen(model, initialization["original_state_sha256"])
        row = {"step": index + 1, "plan_index": index, "input_shape": list(batch["x"].shape),
               **{key: batch[key] for key in HASH_KEYS},
               "complete_input_tensor_sha256": rt.tensor_hash({"x": torch.from_numpy(batch["x"])}),
               "loss": float(loss.detach()),
               "loss_terms": {key: float(value.detach()) for key, value in terms.items() if key != "gap"},
               "parameter_gradient_norms": gradient_norms, "group_gradient_norms": grouped_gradients,
               "parameter_update_norms": update_norms, "group_update_norms": grouped_updates,
               "original_unchanged": True, **cost}
        result["steps"].append(row)
        rt.save_json(output, report)
        print({"condition": condition, "step": index + 1, "loss": row["loss"],
               "group_gradient_norms": grouped_gradients, **cost}, flush=True)
        del batch, terms, stats, features, logits, risk, cache, loss, before
    rt.assert_frozen(model, initialization["original_state_sha256"])
    result.update(status="PASS", original_parameters_and_bn_unchanged=True,
                  initial_state_file_unmodified=(rt.sha(rt.INITIAL) == report["initial_checkpoint_sha256"]),
                  optimizer="fresh Adam lr1e-5 weight_decay1e-4 betas0.9,0.999 eps1e-8",
                  first_step_upstream_update_interpretation="loss gradients are zero; any update is Adam coupled weight decay",
                  second_step_upstream_loss_gradients_nonzero=True,
                  optimized_state_saved=False)
    assert result["initial_state_file_unmodified"]
    rt.save_json(output, report)
    del model, optimizer, parameters, criterion
    cleanup()


def fullframe_condition(condition, state):
    sequences = state[2]
    sequence_name = sorted(sequences)[0]
    sequence = sequences[sequence_name]
    # SequenceData was constructed only from the frozen training selection.
    assert sequence_name not in rt.SEQS
    frames = np.array((sequence.frames[:40].astype(np.float32) - 85.0799) / 47.4845, copy=True)
    assert frames.shape == (40, 512, 640)
    model, initialization = rt.load_initialized(condition)
    model.eval()
    x = torch.from_numpy(frames)[None, None].cuda()
    result = {"status": "RUNNING", "condition": condition, "source_split": "training",
              "sequence": sequence_name, "frames": [0, 40], "input_shape": list(x.shape),
              "raw_file_sha256": rt.sha(sequence.raw_path),
              "spatial_execution": "full-frame; no tiling; official PDA/TSFM/DCN"}
    with torch.no_grad():
        began = timer_start()
        baseline = rt.original(model)(x)
        baseline = tuple(value.cpu() for value in baseline)
        result["baseline"] = timing(began)
        cleanup()
        began = timer_start()
        actual = tuple(value.cpu() for value in model(x))
        result["initialized_wrapper"] = timing(began)
    exact = [torch.equal(a, b) for a, b in zip(baseline, actual)]
    result["feature_and_logit_exact"] = exact
    result["maximum_absolute_errors"] = [float((a - b).abs().max()) for a, b in zip(baseline, actual)]
    assert all(exact), result["maximum_absolute_errors"]
    rt.assert_frozen(model, initialization["original_state_sha256"])
    result.update(status="PASS", original_parameters_and_bn_unchanged=True)
    del baseline, actual, x, model
    cleanup()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=rt.E / "checks/GPU_PREFLIGHT.json")
    parser.add_argument("--skip-fullframe", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Preserve existing preflight evidence; choose --output: {args.output}")
    report = {"status": "RUNNING", "training": {}, "fullframe": {}, "source_sha256": rt.source_hashes(),
              "model_settings": rt.MODEL_SETTINGS, "supervision_settings": rt.SETTINGS,
              "preflight_only": True, "optimized_models_are_disposable": True,
              "torch_version": str(torch.__version__), "cuda_version": str(torch.version.cuda),
              "start_unix": time.time(), "custom_dcn_bitwise_determinism_claimed": False}
    rt.save_json(args.output, report)
    try:
        assert torch.cuda.is_available(), "Actual DCN integration requires CUDA"
        rt.setup()
        report["device"] = torch.cuda.get_device_name()
        report["pytorch_deterministic_algorithms_requested"] = torch.are_deterministic_algorithms_enabled()
        report["pytorch_deterministic_warn_only"] = torch.is_deterministic_algorithms_warn_only_enabled()
        report["cudnn_deterministic"] = torch.backends.cudnn.deterministic
        report["tf32_matmul"] = torch.backends.cuda.matmul.allow_tf32
        report["tf32_cudnn"] = torch.backends.cudnn.allow_tf32
        rt.ensure_initialization()
        report["initial_checkpoint_sha256"] = rt.sha(rt.INITIAL)
        report["release_checkpoint_sha256"] = rt.sha(rt.CP)
        state = rt.frozen_f().data.inputs()
        for condition in rt.CONDITIONS:
            train_condition(condition, state, args.output, report)
        current, history = (report["training"][name] for name in rt.CONDITIONS)
        assert current["initial_model_state_sha256"] == history["initial_model_state_sha256"]
        for left, right in zip(current["steps"], history["steps"]):
            for key in (*HASH_KEYS, "complete_input_tensor_sha256"):
                assert left[key] == right[key], (key, left["step"])
        report["paired_initialization_and_actual_exposure_identical"] = True
        rt.save_json(args.output, report)
        if args.skip_fullframe:
            report["fullframe"] = {"status": "SKIPPED", "reason": "Explicit --skip-fullframe; fullframe feasibility unverified"}
        else:
            for condition in rt.CONDITIONS:
                try:
                    report["fullframe"][condition] = fullframe_condition(condition, state)
                except Exception as error:
                    report["fullframe"][condition] = {
                        "status": "FAIL", "error": repr(error), "traceback": traceback.format_exc(),
                        "cuda_out_of_memory": isinstance(error, torch.cuda.OutOfMemoryError),
                        "operator_or_spatial_substitution_used": False}
                    cleanup()
                rt.save_json(args.output, report)
            assert all(value["status"] == "PASS" for value in report["fullframe"].values()), "Fullframe feasibility failed"
        assert rt.sha(rt.INITIAL) == report["initial_checkpoint_sha256"]
        assert report["source_sha256"] == rt.source_hashes(), "Preflight source changed during execution"
        report["status"] = "TRAINING_PASS_FULLFRAME_SKIPPED" if args.skip_fullframe else "PASS"
        report["end_unix"] = time.time()
        rt.save_json(args.output, report)
        print({"status": report["status"], "output": str(args.output)}, flush=True)
    except BaseException as error:
        report.update(status="FAIL", error=repr(error), traceback=traceback.format_exc(), end_unix=time.time())
        rt.save_json(args.output, report)
        raise


if __name__ == "__main__":
    main()
