"""Exploratory equal-input comparison of small nonlinear traffic detection readouts."""
import hashlib
import json

import numpy as np
import torch
from torch import nn

from readout import RUNS, froc, label_candidates, standardize
from training_settings import TRAIN_SEED, reset_training_rng


VARIANTS = {"group_span_mean": (2, 3), "group_mean_per_span": (2,), "separate_groups_spans": ()}
STEPS = 400
BATCH = 256
HIDDEN = 16


def scores_for_seed(features, labels, train):
    rng = reset_training_rng()
    x = torch.from_numpy(features)
    y = torch.from_numpy(labels.astype(np.float32))
    positive = np.flatnonzero(train & (labels == 1))
    negative = np.flatnonzero(train & (labels == 0))
    if not (len(positive) and len(negative)):
        raise ValueError("Need positives and negatives in the training set.")
    network = nn.Sequential(nn.Linear(x.shape[1], HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 1))
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.001, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    for _ in range(STEPS):
        ids = np.concatenate([rng.choice(positive, BATCH // 2), rng.choice(negative, BATCH // 2)])
        logits = network(x[ids])[:, 0]
        loss = criterion(logits, y[ids])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return network(x)[:, 0].numpy().copy()


def digest(name):
    h = hashlib.sha256()
    with (RUNS / name).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    frames = json.loads((RUNS / "labels.json").read_text())["frames"]
    with np.load(RUNS / "records.npz", allow_pickle=False) as records:
        maps = records["maps"]
        common = np.column_stack([records["common"], records["discovery"]])
        xy, frame_index = records["xy"], records["frame"]
    if maps.shape != (len(xy), 2, 2, 3, 5, 5) or not np.isfinite(maps).all():
        raise ValueError("Invalid response cache.")
    labels, neighbors = label_candidates(xy, frame_index, frames)
    train = np.array([frames[i]["split"] == "train" for i in frame_index]) & (labels >= 0)
    dev = [i for i,f in enumerate(frames) if f["split"] == "calibration"]
    subsets = {"all_dev": dev}
    for seq in sorted({frames[i]["sequence"] for i in dev}):
        subsets[seq] = [i for i in dev if frames[i]["sequence"] == seq]
    common = standardize(common, train).astype(np.float32)
    result = {"status": "exploratory follow-up, same development videos as the linear pilot",
              "fixed_settings": {"seed": TRAIN_SEED, "steps": STEPS, "hidden_relu": HIDDEN, "balanced_batch": BATCH,
                                 "optimizer": "AdamW", "learning_rate": 0.001, "weight_decay": 0.01,
                                 "loss": "BCEWithLogitsLoss", "backend": "CPU single-thread deterministic",
                                 "input": "4 common values and 300 response entries with train-only standardization; same initial weights/sample ids by seed"},
              "variants": {"group_span_mean": "per proposal and perturbation, average the two groups and all three span responses with equal weights; retain same position/velocity projection and broadcast",
                           "group_mean_per_span": "per proposal, span and perturbation, average A/B, and broadcast over group",
                           "separate_groups_spans": "complete input: two proposals, two groups, three spans and 5x5 perturbations"},
              "interpretation": "All arms derive from the same cached observations; pooling destroys ordered axes but retains the contribution of every span and their potentially unequal effective temporal weights. Common discovery scores always retain some A information.",
              "metrics": "empirical FROC evaluated on each development subset exactly as in readout.py; separately swept thresholds for descriptive, not deployment, use",
              "sha256": {n: digest(n) for n in ("records.npz","labels.json")},
              "train_positive_candidates": int((train & (labels == 1)).sum()), "train_negative_candidates": int((train & (labels == 0)).sum()), "methods": {}}
    for variant, axes in VARIANTS.items():
        selected = maps.mean(axis=axes, keepdims=True) if axes else maps
        response = np.broadcast_to(selected, maps.shape).reshape(len(maps), -1)
        scaled = standardize(response, train).astype(np.float32)
        x = np.ascontiguousarray(np.column_stack([common, scaled]))
        del scaled, selected, response
        scores = scores_for_seed(x, labels, train)
        result["methods"][variant] = {str(TRAIN_SEED): {
            name: froc(scores, ids, frames, frame_index, labels, neighbors)
            for name, ids in subsets.items()}}
        del x
    output_dir = RUNS / f"seed{TRAIN_SEED}"
    output_dir.mkdir(exist_ok=True)
    out = output_dir / "nonlinear_scores.json"
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    tmp.replace(out)
    for method, seeds in result["methods"].items():
        print(method, {s: {g: [v["froc"][str(b)]["tp"] for b in (0.5, 1, 5)] for g, v in info.items()}
                       for s,info in seeds.items()}, flush=True)


if __name__ == "__main__":
    main()
