"""Verify tiled frozen frontend plus globally evaluated M1, using training data."""
import gc
import time
import traceback

import numpy as np
import torch

import runtime as rt


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def measure(call):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    began = time.perf_counter()
    result = call()
    torch.cuda.synchronize()
    return result, {"seconds": time.perf_counter()-began,
                    "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved()}


def compare(actual, expected):
    error = float((actual-expected).abs().max())
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-5)
    return error


@torch.no_grad()
def small_grid(condition, frames):
    model, initialization = rt.load_initialized(condition)
    model.eval()
    x = torch.from_numpy(frames[:, :160, :192].copy())[None, None].cuda()
    full_f = model.base.layer1(model.base.conv_in(x))
    full_h = model.base.TPro(full_f.permute(0, 3, 4, 1, 2))
    tiled_f, tiled_h = model.frozen_frontend_tiled(x)
    errors = {"F": compare(tiled_f, full_f), "H": compare(tiled_h, full_h)}
    del full_f, full_h, tiled_f, tiled_h
    baseline = model.base(x)[1].cpu()
    # Activate the actual RFR branch; otherwise zero initialization could hide
    # a wrong spatial propagation or frequency-attention execution path.
    torch.manual_seed(4646)
    model.adapter.out_projection.weight.normal_(0, .01)
    model.adapter.out_projection.bias.fill_(.001)
    expected = tuple(y.cpu() for y in model(x))
    actual = tuple(y.cpu() for y in model.forward_full_spatial(x))
    errors.update(features=compare(actual[0], expected[0]), logits=compare(actual[1], expected[1]))
    delta = float((expected[1]-baseline).abs().max())
    assert delta > 1e-4
    rt.assert_frozen(model, initialization["original_state_sha256"])
    result = {"status": "PASS", "input_shape": list(x.shape), "active_Wout": True,
              "maximum_absolute_errors": errors, "logit_change_from_zero_Wout": delta,
              "comparison": "Original full frontend/global M1/full head vs tiled frozen frontend/global M1/time-chunked pointwise head",
              "assert_close_atol": 1e-4, "assert_close_rtol": 1e-5,
              "original_parameters_and_bn_unchanged": True}
    del model, x, baseline, expected, actual
    cleanup()
    return result


@torch.no_grad()
def fullframe(condition, frames):
    model, initialization = rt.load_initialized(condition)
    model.eval()
    before = rt.tensor_hash(model.state_dict())
    x = torch.from_numpy(frames.copy())[None, None].cuda()
    baseline, baseline_cost = measure(lambda: rt.frozen_f().runtime.tiled_logits(model.base, x, core=128, halo=16))
    cleanup()
    adapter_calls = []
    recurrence_shapes = []
    handle = model.adapter.register_forward_pre_hook(lambda _, args: adapter_calls.append([list(t.shape) for t in args]))
    cell_handle = model.adapter.recurrent_cell.register_forward_pre_hook(
        lambda _, args: recurrence_shapes.append([list(t.shape) for t in args]))
    output, cost = measure(lambda: model.forward_full_spatial(x))
    handle.remove(); cell_handle.remove()
    actual = output[1].cpu()
    error = compare(actual, baseline)
    disagreement = int(((actual >= 0) != (baseline >= 0)).sum())
    assert disagreement == 0
    assert adapter_calls == [[[1, 32, 40, 512, 640], [1, 32, 40, 512, 640]]]
    assert len(recurrence_shapes) == 39
    assert all(shapes == [[1, 16, 512, 640], [1, 16, 512, 640]] for shapes in recurrence_shapes)
    assert all(bool(torch.isfinite(y).all()) for y in output)
    rt.assert_frozen(model, initialization["original_state_sha256"])
    assert before == rt.tensor_hash(model.state_dict())
    result = {"status": "PASS", "condition": condition, "input_shape": list(x.shape),
              "baseline": baseline_cost, "initialized_wrapper": cost,
              "baseline_reference": "Unmodified historical tiled_logits(core128,halo16)",
              "maximum_absolute_logit_error": error, "threshold05_disagreements": disagreement,
              "global_m1_single_call": True, "adapter_call_shapes": adapter_calls,
              "recurrent_call_count": len(recurrence_shapes), "recurrent_call_shapes": recurrence_shapes,
              "output_shapes": [list(y.shape) for y in output], "outputs_finite": True,
              "all_model_state_unchanged": True,
              "spatial_execution": "Only original frozen frontend tiled128/halo16; M1 full512x640; pointwise original head temporalchunk1"}
    del model, x, output, actual, baseline
    cleanup()
    return result


def main():
    path = rt.E / "checks/SPATIAL_PREFLIGHT.json"
    if path.exists():
        raise FileExistsError(path)
    report = {"status": "RUNNING", "small_grid": {}, "fullframe": {}, "source_sha256": rt.source_hashes(),
              "preflight_only": True, "source_split": "training", "global_m1_spatial_tiling": False,
              "frontend_core": 128, "frontend_halo": 16, "head_temporal_chunk": 1,
              "start_unix": time.time(), "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda}
    rt.save_json(path, report)
    try:
        state = rt.frozen_f().data.inputs()
        sequence_name = sorted(state[2])[0]
        sequence = state[2][sequence_name]
        assert sequence_name not in rt.SEQS
        frames = np.array((sequence.frames[:40].astype(np.float32)-85.0799)/47.4845, copy=True)
        assert frames.shape == (40, 512, 640)
        report.update(sequence=sequence_name, frame_range=[0, 40], raw_file_sha256=rt.sha(sequence.raw_path))
        for condition in rt.CONDITIONS:
            print("small_grid", condition, flush=True)
            report["small_grid"][condition] = small_grid(condition, frames)
            rt.save_json(path, report)
        for condition in rt.CONDITIONS:
            print("fullframe", condition, flush=True)
            report["fullframe"][condition] = fullframe(condition, frames)
            rt.save_json(path, report)
        report.update(status="PASS", source_sha256_final=rt.source_hashes(), end_unix=time.time())
    except BaseException as error:
        report.update(status="FAIL", error=repr(error), traceback=traceback.format_exc(), end_unix=time.time())
        rt.save_json(path, report)
        raise
    rt.save_json(path, report)
    print({"status": report["status"], "output": str(path)}, flush=True)


if __name__ == "__main__":
    main()
