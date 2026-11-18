"""Checkpoint-bound, restartable evaluation on the frozen S4 native center panels."""
from dataclasses import replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from ..checks.check_s0 import atomic_text
from ..data.dataset import MaskClipDataset, _frame_path, _load_gray
from .s4_metrics import evaluate_frames, prepare_frame


EXP = Path(__file__).resolve().parents[1]
PROJECT = EXP.parents[1]
MODES = {"M0": None, "Msep": "additive_only", "Mraw": "raw_joint", "Mint": "interaction"}
PANELS = {"development_monitor": ("dev", "diagnostic_dev_ids", (6000,18000,30000)),
          "development": ("dev", "dev_ids", (12000,24000,36000)),
          "historical_test": ("test", "test_ids", (24000,36000))}


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def _read(path):
    return json.loads(path.read_bytes())


def _write(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _checkpoint(run, mode, seed, step):
    return run / mode / f"seed{seed}" / "checkpoints" / f"step_{step:06d}.pt"


def _historical_gate(run, protocol, step, seed, mode, inputs):
    """Verify the common endpoint and every frozen development threshold before any test file access."""
    path = run / "historical_lock.json"
    lock = _read(path)
    decision = run / "budget_decision.json"
    _check(lock["stage"] == "S4" and lock["state"] == "HISTORICAL_LOCKED" and lock["run_id"] == run.name
           and lock["selected_step"] == step == _read(decision)["selected_step"]
           and lock["protocol_sha256"] == inputs["protocol_sha256"] and lock["budget_decision_sha256"] == _sha(decision)
           and lock["execution_binding_sha256"] == inputs["execution_binding_sha256"]
           and set(lock["runs"]) == set(protocol["modes"]) == set(MODES),
           "Historical evaluation lacks its common model/budget/protocol lock")
    for name, seeds in lock["runs"].items():
        _check(set(seeds) == {"46"}, "A historical endpoint is missing or includes an unauthorized seed")
        record = seeds[str(seed)]
        dev = run / name / f"seed{seed}" / f"dev_{step}" / "metrics.json"
        _check((record["mode"], record["seed"], record["step"]) == (name, seed, step)
               and record["checkpoint_sha256"] == _sha(_checkpoint(run, name, seed, step))
               and record["dev_metrics_sha256"] == _sha(dev), f"Historical endpoint changed: {name}")
        points = _read(dev)["descriptive"]["budgets"]
        _check(record["thresholds"] == {b: points[b]["threshold"] for b in ("0.5","1","5")},
               f"Frozen threshold differs from full development: {name}")
    inputs["historical_lock_sha256"] = _sha(path)
    return lock["runs"][mode][str(seed)]["thresholds"]


def _save_logits(path, scores):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        temp = Path(f.name)
        try:
            np.save(f, scores, allow_pickle=False)
            f.flush()
            os.fsync(f.fileno())
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)


def _journal(path, allowed):
    """Recover complete, fsynced frame receipts; discard a partial trailing append after an interruption."""
    rows = {}
    if path.exists():
        with path.open("rb+") as f:
            while True:
                start = f.tell()
                line = f.readline()
                if not line:
                    break
                if not line.endswith(b"\n"):
                    f.truncate(start)
                    break
                row = json.loads(line)
                key = row["frame_id"]
                _check(key in allowed and key not in rows, f"Invalid or duplicate frame receipt: {key}")
                rows[key] = row
    return rows


def _data_sources(manifest, split):
    # The historical split is visited here only after _historical_gate returned successfully.
    sources = {}
    for source in manifest["sequences"]:
        if source["split"] != split:
            continue
        records = {}
        h = {role: hashlib.sha256() for role in ("image", "mask")}
        for record in source["frames"]:
            number = record["number"]
            for role, directory in (("image", "images"), ("mask", "masks")):
                path = _frame_path(PROJECT, record[role], source["id"], number, directory)
                h[role].update((_sha(path) + "\n").encode("ascii"))
                if role == "mask":
                    records[number] = path
        for role, digest in h.items():
            _check(digest.hexdigest() == source[role + "_files_sha256"], f"Changed {source['id']} {role} input")
        sources[source["id"]] = records
    return sources


@torch.inference_mode()
def _predict(model, dataset, idx, center, device):
    clip = dataset.inference_item(idx)
    shape = tuple(clip.scene_meta[0].original_hw)
    _check(clip.output_frame_id == (center,) and bool(clip.pixel_valid.all()), f"Incorrect native input at {center}")
    clip = replace(clip, frames=clip.frames.to(device), time_offsets=clip.time_offsets.to(device),
                   pixel_valid=clip.pixel_valid.to(device))
    output = model.predict_tiled(clip, core_hw=(64, 64))
    _check(output.output_frame_id == (center,) and output.logits.dtype == torch.float32
           and tuple(output.logits.shape) == (1, 1, *shape) and bool(torch.isfinite(output.logits).all()),
           f"Invalid predicted native FP32 map at {center}")
    return output.logits[0, 0].cpu().numpy()


def _run_panel(run, protocol, model, mode, seed, step, population, checkpoint_sha256, folder):
    split, key, steps = PANELS[population]
    ids = protocol[key]
    expected_size = {"development_monitor": 32, "development": 672, "historical_test": 1680}[population]
    _check(step in steps and len(ids) == len(set(ids)) == expected_size, "Wrong panel or monitoring step")
    inputs = {"mode": mode, "seed": seed, "step": step, "population": population, "checkpoint_sha256": checkpoint_sha256,
              "protocol_sha256": _sha(run / "protocol.json"), "execution_binding_sha256": _sha(run / "execution_binding.json"),
              "center_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
              "query_chunk": 64 if mode != "M0" else None, "velocity_chunk": 8 if mode != "M0" else None,
              "core_hw": [64,64], "score_domain": "native_fp32_logits"}
    locked = _read(run / "execution_binding.json")
    _check(locked["can_launch_training"] is True and locked["run_id"] == run.name
           and locked["common_protocol_sha256"] == inputs["protocol_sha256"], "No formal execution binding")
    thresholds = _historical_gate(run, protocol, step, seed, mode, inputs) if split == "test" else None
    marker = folder / "evaluation_inputs.json"
    if marker.exists():
        _check(_read(marker) == inputs, f"Cannot mix logits from different inputs at {folder}")
    else:
        _check(not any(folder.iterdir()), f"Unbound artifacts at {folder}")
        _write(marker, inputs)
    map_dir = folder / "logits"
    map_dir.mkdir(exist_ok=True)
    journal = folder / "frames.jsonl"
    rows = _journal(journal, set(ids))
    def map_file(name):
        return map_dir / (name.replace("/", "_") + ".npy")
    for name, row in rows.items():
        _check(_sha(map_file(name)) == row["sha256"], f"Cached raw native prediction has changed: {name}")
    binding_path = folder / "binding.json"
    if binding_path.exists():
        binding = _read(binding_path)
        metrics = folder / "metrics.json"
        _check(all(binding[k] == v for k,v in inputs.items()) and binding["metrics_sha256"] == _sha(metrics)
               and binding["frame_journal_sha256"] == _sha(journal)
               and set(rows) == set(ids) and binding["frame_sha256"] == {k:v["sha256"] for k,v in rows.items()},
               f"Endpoint binding mismatch: {binding_path}")
        desc = _read(metrics)["descriptive"]
        _check(desc["curve_sha256"] == _sha(folder / desc["curve_file"]), "FROC archive changed")
        return binding
    dataset_spec = protocol["proposal"]["data"]
    manifest_path = EXP / dataset_spec["manifest"]
    _check(_sha(manifest_path) == dataset_spec["manifest_sha256"], "Dataset manifest changed")
    manifest = _read(manifest_path)
    dataset = MaskClipDataset(manifest, PROJECT, split)
    sources = _data_sources(manifest, split)
    lookup = {f"{s}/{n:05d}": i for i,(s,n) in enumerate(dataset.centers)}
    _check(set(ids) <= set(lookup), "Requested center not found in the S0 split")
    device = next(model.parameters()).device
    prepared = []
    resumed = len(rows)
    start_time = time.monotonic()
    with journal.open("ab", buffering=0) as receipts:
        for position, name in enumerate(ids, start=1):
            s, n = name.split("/")
            mask_path = sources[s][int(n)]
            centroid_path = mask_path.parent.parent / "masks_centroid" / mask_path.name
            centroid_hash = _sha(centroid_path) if centroid_path.is_file() else None
            if name in rows:
                _check(rows[name]["centroid_sha256"] == centroid_hash, f"Author labels changed for {name}")
                logits = np.load(map_file(name), allow_pickle=False)
            else:
                then = time.monotonic()
                logits = _predict(model, dataset, lookup[name], name, device)
                seconds = time.monotonic() - then
                _save_logits(map_file(name), logits)
                row = {"frame_id": name, "sha256": _sha(map_file(name)), "centroid_sha256": centroid_hash,
                       "inference_seconds": seconds}
                receipts.write((json.dumps(row, allow_nan=False) + "\n").encode())
                os.fsync(receipts.fileno())
                rows[name] = row
            centroid = _load_gray(centroid_path) if centroid_hash is not None else None
            prepared.append(prepare_frame(name, logits, _load_gray(mask_path), centroid_mask=centroid))
            if position % 8 == 0 or position == len(ids):
                print(f"{mode}/seed{seed} {population} step={step} {position}/{len(ids)} native centers", flush=True)
    result = evaluate_frames(prepared, folder, population=population, leave_one_source_out=(population == "development"),
                             frozen_thresholds=thresholds)
    _check(result["descriptive"]["frames"] == len(ids), "Evaluation did not cover every requested native center")
    binding = {**inputs, "frames": len(ids), "frame_sha256": {k:rows[k]["sha256"] for k in ids},
               "frame_journal_sha256": _sha(journal), "metrics_sha256": _sha(folder / "metrics.json"),
               "centers_restored_on_final_invocation": resumed, "final_invocation_seconds": time.monotonic()-start_time,
               "inference_seconds": sum(rows[k]["inference_seconds"] for k in ids)}
    _write(binding_path,binding)
    return binding


def evaluate_checkpoint(run_dir, protocol, model, mode, seed, step, *, population, checkpoint_sha256):
    """Write dev_<step> or test_<step> and return its binding. The caller supplies the loaded model and checkpoint SHA."""
    run = Path(run_dir)
    _check(mode in MODES and seed == 46 and protocol["seeds"] == [46] and population in PANELS,
           "Unauthorized model, seed or evaluation population")
    _check(protocol == _read(run / "protocol.json") and _sha(_checkpoint(run, mode, seed, step)) == checkpoint_sha256,
           "Wrong protocol or checkpoint bytes for this evaluation")
    expected_class = "VideoBaseline" if mode == "M0" else "HypothesisEvidenceDetector"
    _check(type(model).__name__ == expected_class, "The supplied model class does not match its declared configuration")
    if mode != "M0":
        aggregator = model.video_aggregator
        _check(aggregator.encoder.mode == MODES[mode] and (aggregator.query_chunk,aggregator.velocity_chunk) == (64,8),
               "Evaluation requires the frozen evidence mode and q64/v8")
    split = PANELS[population][0]
    folder = run / mode / f"seed{seed}" / f"{split}_{step}"
    folder.mkdir(parents=True, exist_ok=True)
    fd = os.open(folder, os.O_RDONLY)
    prior = model.training
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        model.eval()
        return _run_panel(run, protocol, model, mode, seed, step, population, checkpoint_sha256, folder)
    finally:
        model.train(prior)
        os.close(fd)
