"""Paired read-only S4 training-runtime replay. Output is separate from every formal run."""
import argparse
from collections import Counter
import gc
import json
from pathlib import Path
from statistics import mean, median
import sys
from time import perf_counter
from unittest.mock import patch

import numpy as np
import torch
from scipy.ndimage import maximum_filter
from scipy.stats import rankdata

EXPERIMENT = Path(__file__).resolve().parents[2]
PROJECT = EXPERIMENT.parents[1]
sys.path.insert(0, str(PROJECT))

from experiments.irstd_independent_v1.checks.s4_binding import verify_execution
from experiments.irstd_independent_v1.data.dataset import MaskClipDataset
from experiments.irstd_independent_v1.engine import s4_checkpoints as checkpoints
from experiments.irstd_independent_v1.engine import s4_train as train
from experiments.irstd_independent_v1.engine.checkpoints import capture_rng, configure_determinism, require_exact, restore_rng
from experiments.irstd_independent_v1.evaluation.s2b_froc import _peaks


BASE = "q64_v8"
ALT = "q256_v81"
CHUNKS = {BASE: (64, 8), ALT: (256, 81)}
OFFICIAL = EXPERIMENT / "reports/s4_runs/s4_24k_seed46_20261118_01"
PEAK_K = 16
RTOL, ATOL = 1e-4, 1e-5


def check(predicate, message):
    if not predicate:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_bytes())


def report(path, record):
    # Each object is immutable, inspectable immediately, and outside formal outputs.
    with path.open("x", encoding="utf8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")


def restore(path, digest, context, device, branch):
    # This scratch replay may use CPU or another GPU. Validate all model/AdamW
    # bytes before changing chunks. A GPU checkpoint's CUDA RNG cannot run on CPU.
    configure_determinism()
    state = checkpoints._bound_state(path, digest, context)
    model = checkpoints._model_from_state(state, device).train()
    optimizer = checkpoints.make_optimizer(model)
    optimizer.load_state_dict(state["optimizer"])
    require_exact(state["optimizer"], optimizer.state_dict(), "S4 AdamW")
    checkpoints._check_optimizer(model, optimizer, state["step"])
    rng = dict(state["rng"])
    if device.type == "cpu":
        rng["torch_cuda"] = []
    restore_rng(rng, device)
    require_exact(rng, capture_rng(device), "paired replay RNG")
    aggregator = model.video_aggregator
    check(aggregator.checkpoint_chunks is True and (aggregator.query_chunk, aggregator.velocity_chunk) == CHUNKS[BASE],
          "Archive is not the expected S4 starting configuration")
    aggregator.query_chunk, aggregator.velocity_chunk = CHUNKS[branch]
    return model, optimizer, state["step"], state["runtime"]


def flat_values(model, field):
    # Copy outside the timer. Concatenation on CPU avoids allocating an observer buffer on the measured GPU.
    values = [(p if field == "weights" else p.grad).detach().cpu().reshape(-1) for p in model.parameters()]
    return torch.cat(values).numpy()


def restrict_to_eligible(target, valid, box):
    x0, y0, x1, y1 = box
    fg = target.target_map.detach().cpu().numpy()[0, 0] == 1
    native = valid.detach().cpu().numpy()[0, 0] & ~target.ignore_map.detach().cpu().numpy()[0, 0]
    covered = maximum_filter((fg & native).astype(np.uint8), size=7, mode="constant", cval=0).astype(bool)
    return (native & ~fg & ~covered)[y0:y1, x0:x1].copy()


def timed_arm(path, digest, context, branch, rows, warmup, data, lookup, device, output):
    model, opt, cursor, runtime = restore(path, digest, context, device, branch)
    groups = train._group_parameters(model)
    prior = flat_values(model, "weights")
    measurements, tensors = [], []
    original_loss = train.s4_detection_loss
    held = {}

    def retain_loss_inputs(prediction, targets, valid, box):
        # Capture only references to the original tensors (no reduction or device transfer).
        held["logits"], held["targets"], held["valid"] = prediction.logits.detach(), targets, valid
        return original_loss(prediction, targets, valid, box)

    for slot, row in enumerate(rows):
        check(row["step"] == cursor + slot + 1, "Paired update cursor diverged")
        with patch.object(train, "s4_detection_loss", retain_loss_inputs):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = perf_counter()
            result = train._optimize(model, opt, groups, data, lookup, row, device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = perf_counter() - start
        weights = flat_values(model, "weights")
        logits = held["logits"].cpu().numpy().copy().reshape(64, 64)
        eligible = restrict_to_eligible(held["targets"], held["valid"], row["core_xyxy"])
        held.clear()
        update = weights.astype(np.float64) - prior.astype(np.float64)
        gradients = flat_values(model, "gradients")
        check(eligible.sum() == result["eligible_hard_pixels"], "Hard-pixel candidate mask disagrees with S4")
        check(all(np.isfinite(a).all() for a in (weights, logits, gradients, update))
              and result["finite_after_update"], "Nonfinite instrumented update")
        tensors.append({"logits": logits, "gradients": gradients, "weights": weights, "update": update,
                        "eligible": eligible, "hard_count": result["selected_hard_pixels"]})
        measured = dict(result, wall_seconds=elapsed, phase="warmup" if slot < warmup else "measured")
        measurements.append(measured)
        report(output / f"{branch}_step{row['step']:06d}.json", measured)
        print(f"{context['mode']} {branch} {measured['phase']} step={row['step']} seconds={elapsed:.3f}", flush=True)
        prior = weights
    return measurements, tensors, runtime


def difference(reference, candidate):
    a, b = reference.astype(np.float64).ravel(), candidate.astype(np.float64).ravel()
    diff = b - a
    ref_l2 = float(np.linalg.norm(a))
    return {"elements": len(a), "reference_l2": ref_l2, "delta_l2": float(np.linalg.norm(diff)),
            "relative_l2": float(np.linalg.norm(diff) / ref_l2) if ref_l2 else None,
            "max_abs": float(np.max(np.abs(diff))), "max_abs_reference": float(np.max(np.abs(a))),
            "outside_tolerance": int(np.count_nonzero(np.abs(diff) > ATOL + RTOL * np.abs(a))),
            "bitwise_equal": bool(np.array_equal(reference.view(np.uint8), candidate.view(np.uint8)))}


def ranked_indices(values, eligible):
    flat = np.flatnonzero(eligible)
    return flat[np.argsort(-values.ravel()[flat], kind="stable")]


def compare_rank(first, second):
    ref, new = first["logits"], second["logits"]
    check(np.array_equal(first["eligible"], second["eligible"]) and first["hard_count"] == second["hard_count"],
          "Labels or hard-negative quota changed between paired runs")
    a = rankdata(ref.ravel())
    b = rankdata(new.ravel())
    rho = float(np.corrcoef(a, b)[0, 1]) if np.ptp(a) and np.ptp(b) else None
    yx_a, _ = _peaks(ref)
    yx_b, _ = _peaks(new)
    k = min(PEAK_K, len(yx_a), len(yx_b))
    ha = ranked_indices(ref, first["eligible"])[:first["hard_count"]]
    hb = ranked_indices(new, second["eligible"])[:second["hard_count"]]
    return {"pixel_spearman_average_ties": rho, "peak_top1_same": bool(np.array_equal(yx_a[0], yx_b[0])),
            "peak_count_reference": len(yx_a), "peak_count_candidate": len(yx_b), "top_peak_k": k,
            "top_peak_overlap": len(set(map(tuple, yx_a[:k])).intersection(map(tuple, yx_b[:k]))),
            "hard_pixel_k": len(ha), "hard_pixel_overlap": len(set(ha).intersection(hb))}


def compare_mode(mode, arms, rows, warmup, output):
    comparisons = []
    for i, row in enumerate(rows):
        rlog, robs, _ = arms[BASE]
        clog, cobs, _ = arms[ALT]
        delta = {k: difference(robs[i][k], cobs[i][k]) for k in ("logits", "gradients", "weights", "update")}
        loss_ref, loss_alt = rlog[i]["total"], clog[i]["total"]
        info = {"step": row["step"], "phase": rlog[i]["phase"], "role": row["sample_role"],
                "identical_pre_update_weights_and_AdamW": i == 0, "finite": True, **delta,
                "loss_reference": loss_ref, "loss_candidate": loss_alt, "loss_abs_diff": abs(loss_ref - loss_alt),
                "loss_relative_diff": abs(loss_ref-loss_alt)/max(abs(loss_ref),1e-30),
                "rank": compare_rank(robs[i], cobs[i])}
        comparisons.append(info)
    rates = {}
    for branch in CHUNKS:
        logs = arms[branch][0][warmup:]
        times = [v["wall_seconds"] for v in logs]
        peaks = [v["peak_allocated_mib"] for v in logs if v["peak_allocated_mib"] is not None]
        rates[branch] = {"seconds_per_step_mean": mean(times), "seconds_per_step_median": median(times),
                         "total_seconds": sum(times), "seconds_by_step": times,
                         "peak_allocated_mib": max(peaks) if peaks else None}
    ratios = [a/b for a,b in zip(rates[BASE]["seconds_by_step"], rates[ALT]["seconds_by_step"])]
    summary = {"mode": mode, "speed": rates, "measured_updates": len(ratios),
               "ratio_of_total_baseline_over_candidate": rates[BASE]["total_seconds"] / rates[ALT]["total_seconds"],
               "paired_speedup_by_step": ratios, "median_paired_speedup": median(ratios),
               "same_start_single_update": comparisons[0], "paired_trajectory": comparisons}
    report(output / "comparison.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--modes", choices=("Msep","Mraw","Mint"), nargs="+", default=["Msep","Mraw","Mint"])
    parser.add_argument("--checkpoint-step", type=int, choices=(8,300), default=8)
    parser.add_argument("--warmup", type=int, choices=(0,1,2), default=2)
    parser.add_argument("--output", "--out", type=Path, required=True, help="New separate results directory")
    args = parser.parse_args()
    output = args.output.resolve()
    runtime_dir = Path(__file__).resolve().parent
    check(not output.is_relative_to(PROJECT) or output.is_relative_to(runtime_dir),
          "Repository outputs must be in the separate reports/s4_runtime directory")
    check(len(set(args.modes)) == len(args.modes), "Duplicate model mode")
    device = torch.device(args.device)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if device.type == "cuda":
        check(torch.cuda.is_available() and torch.cuda.device_count() == 1,
              "Bind exactly one authorized GPU before running the CUDA comparison")
    protocol, _ = verify_execution(OFFICIAL)
    protocol_sha, execution_sha = train._sha(OFFICIAL / "protocol.json"), train._sha(OFFICIAL / "execution_binding.json")
    schedule_path = OFFICIAL / "schedule_46.json"
    rows = read_json(schedule_path)["updates"][args.checkpoint_step:args.checkpoint_step + args.warmup + 8]
    check(train._sha(schedule_path) == protocol["schedule_sha256"]["46"] and len(rows) == args.warmup + 8,
          "Frozen training schedule changed")
    check(Counter(r["sample_role"] for r in rows[args.warmup:]) == {"positive":4,"random_background":3,"empty_frame":1},
          "Measured eight updates do not have the registered 4:3:1 composition")
    manifest = EXPERIMENT / protocol["proposal"]["data"]["manifest"]
    check(train._sha(manifest) == protocol["proposal"]["data"]["manifest_sha256"], "Frozen source manifest changed")
    dataset = MaskClipDataset(read_json(manifest), PROJECT, "train")
    lookup = {key:i for i,key in enumerate(dataset.centers)}
    check(len(lookup) == len(dataset) == 6048, "Unexpected training population")
    bindings = {}
    for mode in args.modes:
        path = OFFICIAL / mode / "seed46/checkpoints" / f"step_{args.checkpoint_step:06d}.pt"
        sidecar = read_json(path.with_suffix(".json"))
        ctx = checkpoints.identity(protocol,mode,46,protocol_sha256=protocol_sha,execution_sha256=execution_sha)
        check(sidecar["step"] == args.checkpoint_step and sidecar["identity"] == ctx
              and sidecar["checkpoint_sha256"] == train._sha(path), f"Incorrect starting checkpoint: {path}")
        bindings[mode] = (path, sidecar["checkpoint_sha256"], ctx)
    output.mkdir(parents=True, exist_ok=False)
    meta = {"purpose":"Independent short scheduling comparison on training data", "device":str(device), "cpu_threads":2,
            "torch_version":str(torch.__version__), "gpu_name":torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "warmup_steps":args.warmup, "measured_steps":8, "seed":46, "source_step":args.checkpoint_step,
            "source_protocol_sha256":protocol_sha, "execution_binding_sha256":execution_sha,
            "script_sha256":train._sha(Path(__file__)), "inputs_sha256":{k:v[1] for k,v in bindings.items()},
            "schedule_sha256":protocol["schedule_sha256"]["46"], "ordered_rows":rows, "modes":args.modes,
            "tolerance":{"rtol":RTOL,"atol":ATOL}, "candidate_chunks":CHUNKS[ALT], "baseline_chunks":CHUNKS[BASE],
            "settings":"Identical seed46 checkpoint and AdamW per arm, FP32 deterministic, recomputation enabled.",
            "timing":"Elapsed synchronized S4 _optimize: PNG decoding and transfers, forward/backward, mining/loss, AdamW, all per-step module diagnostics. One observer stores tensor references. Full checkpointing, periodic diagnostic/evaluation panels, journal fsync, observer snapshots and comparisons excluded.",
            "memory":"CUDA max allocated from official _optimize with one arm resident. Detached references to core logits and original full masks are retained until the update completes. CPU memory is not measured.",
            "trajectory":"First update starts from identical model and optimizer. Later updates include accumulated trajectory differences. Eight paired sequential updates per arm are descriptive, not independent replications.",
            "cache":"Identical data read prior to each arm to reduce filesystem cache order effects; no samples are cached in Python for the timed updates.",
            "rng":"CUDA replay restores the single saved CUDA RNG. CPU replay restores Python/NumPy/Torch CPU RNG and omits the unavailable CUDA stream; CPU performance applies only to this machine."}
    report(output / "inputs.json", meta)
    summaries = {}
    for mode in args.modes:
        output_mode = output / mode
        output_mode.mkdir()
        path, digest, ctx = bindings[mode]
        # Fixed order per model, independent of the selected subset or previous timings.
        order = (ALT, BASE) if mode == "Mraw" else (BASE, ALT)
        arms = {}
        for branch in order:
            for row in rows:
                dataset[lookup[row["source"], row["center"]]]
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            arms[branch] = timed_arm(path, digest, ctx, branch, rows, args.warmup, dataset, lookup, device, output_mode)
        result = compare_mode(mode, arms, rows, args.warmup, output_mode)
        summaries[mode] = {"arm_order":order, "speedup_ratio_of_total":result["ratio_of_total_baseline_over_candidate"],
                           "source_runtime":arms[BASE][2], "per_step_comparisons":f"{mode}/comparison.json"}
    report(output / "summary.json", summaries)


if __name__ == "__main__":
    main()
