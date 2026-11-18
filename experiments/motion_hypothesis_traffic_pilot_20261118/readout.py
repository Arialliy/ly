"""Paired, CPU-only readouts for thermal traffic candidates. Inputs live in runs/."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


RUNS = Path(__file__).resolve().parent / "runs"
BUDGETS = (0.5, 1, 5)
METHODS = (
    "A_long_peak", "A_long_full", "B_long_full", "A_multi_full", "B_multi_full",
    "AB_multi_stats", "AB_multi_full",
)


def response_input(maps, method):
    # Restore the same [K, group, span, y, x] layout to every representation.
    if method == "A_long_peak":
        selected = maps[:, :, :1, -1:].max(axis=(-1, -2), keepdims=True)
    elif method in ("A_long_full", "B_long_full"):
        group = 0 if method.startswith("A_") else 1
        selected = maps[:, :, group:group+1, -1:]
    elif method in ("A_multi_full", "B_multi_full"):
        group = 0 if method.startswith("A_") else 1
        selected = maps[:, :, group:group+1]
    elif method == "AB_multi_stats":
        peak = maps.max(axis=(-1, -2))
        center = maps[..., 2, 2]
        mean = maps.mean(axis=(-1, -2))
        dx = center - 0.5 * (maps[..., 2, 0] + maps[..., 2, 4])
        dy = center - 0.5 * (maps[..., 0, 2] + maps[..., 4, 2])
        stats = np.stack([peak, center, mean, dx, dy], axis=-1)
        selected = stats[..., None, :]
    elif method == "AB_multi_full":
        selected = maps
    else:
        raise ValueError(method)
    return np.broadcast_to(selected, maps.shape).reshape(len(maps), -1)


def standardize(array, train):
    center = array[train].mean(axis=0, dtype=np.float64)
    scale = np.maximum(array[train].std(axis=0, dtype=np.float64), 1e-6)
    return np.clip((array - center) / scale, -8, 8) / np.sqrt(array.shape[1])


def fit_and_score(common, response, train, labels):
    x = np.column_stack([standardize(common, train), standardize(response, train)])
    xt = np.ascontiguousarray(x[train])
    yt = labels[train].astype(float)
    weight = np.where(yt == 1, 0.5/max(1, yt.sum()), 0.5/max(1, (1 - yt).sum()))
    if len(np.unique(yt)) != 2:
        raise RuntimeError("Training requires positive and negative candidates.")
    penalty = 0.01

    def objective(params):
        linear = xt @ params[:-1] + params[-1]
        value = weight @ (np.logaddexp(0, linear) - yt * linear) + 0.5 * penalty * (params[:-1] @ params[:-1])
        error = weight * (expit(linear) - yt)
        grad = np.r_[xt.T @ error + penalty * params[:-1], error.sum()]
        return value, grad

    fit = minimize(objective, np.zeros(x.shape[1]+1), jac=True, method="L-BFGS-B",
                   options={"maxiter": 250, "ftol": 1e-12, "gtol": 1e-7})
    if not fit.success:
        raise RuntimeError(fit.message)
    return x @ fit.x[:-1] + fit.x[-1], int(fit.nit)


def contacts(points, boxes):
    """Any point in a closed bbox expanded by 1 native pixel matches."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    p = points[:, None, :]
    return np.all((p >= boxes[None, :, :2] - 1) &
                  (p <= boxes[None, :, :2] + boxes[None, :, 2:] + 1), axis=2)


def label_candidates(xy, frame_index, frames):
    labels = np.zeros(len(xy), dtype=np.int8)
    neighbors = [()] * len(xy)
    for fi, frame in enumerate(frames):
        ids = np.flatnonzero(frame_index == fi)
        good = contacts(xy[ids], frame["targets"])
        irrelevant = contacts(xy[ids], frame["ignore"]).any(axis=1)
        labels[ids[irrelevant]] = -1
        labels[ids[good.any(axis=1)]] = 1
        for i, row in zip(ids, good):
            neighbors[i] = tuple(np.flatnonzero(row).tolist())
    return labels, neighbors


def froc(scores, group_frames, frames, frame_index, labels, neighbors):
    total = sum(len(frames[j]["targets"]) for j in group_frames)
    frame_count = len(group_frames)
    ids = np.flatnonzero(np.isin(frame_index, group_frames))
    valid = ids[labels[ids] >= 0]
    order = valid[np.argsort(-scores[valid], kind="stable")]
    owners = {j: {} for j in group_frames}

    # One extra candidate can improve the maximum cardinality by at most one.
    # Reassign prior matches through an augmenting path in crowded frames.
    def augment(i, seen, matched):
        for box_id in neighbors[i]:
            if box_id in seen:
                continue
            seen.add(box_id)
            previous = matched.get(box_id)
            if previous is None or augment(previous, seen, matched):
                matched[box_id] = i
                return True
        return False

    fp = tp = 0
    best = {str(b): {"tp": 0, "fp": 0, "fn": total, "recall": 0.0 if total else None, "fp_per_frame": 0.0, "logit_threshold": None} for b in BUDGETS}
    for k, idx in enumerate(order):
        if augment(idx, set(), owners[frame_index[idx]]):
            tp += 1
        else:
            fp += 1
        if k+1 < len(order) and scores[order[k+1]] == scores[idx]:
            continue
        for budget in BUDGETS:
            if fp <= budget * frame_count and tp > best[str(budget)]["tp"]:
                best[str(budget)] = {"tp": tp, "fp": fp, "fn": total - tp,
                                     "recall": tp / total if total else None,
                                     "fp_per_frame": fp / frame_count,
                                     "logit_threshold": float(scores[idx])}
    return {"frames": frame_count, "targets": total, "candidates": len(ids),
            "ignored_candidates": int(len(ids) - len(valid)),
            "candidate_oracle_tp": tp, "candidate_oracle_recall": tp / total if total else None,
            "froc": best}


def main():
    records = np.load(RUNS / "records.npz", allow_pickle=False)
    frames = json.loads((RUNS / "labels.json").read_text())["frames"]
    maps, common, discovery, xy, frame_index = (records[k] for k in ("maps", "common", "discovery", "xy", "frame"))
    n = len(maps)
    if maps.shape != (n, 2, 2, 3, 5, 5) or common.shape != (n, 2) or discovery.shape != (n, 2) or xy.shape != (n, 2) or frame_index.shape != (n,):
        raise ValueError("Unexpected input shapes.")
    if not all(np.isfinite(v).all() for v in (maps, common, discovery, xy)) or not np.issubdtype(frame_index.dtype, np.integer) or frame_index.min() < 0 or frame_index.max() >= len(frames):
        raise ValueError("Nonfinite or invalid candidate input.")
    for split in ("train", "calibration"):
        if not any(f["split"] == split for f in frames):
            raise ValueError("Missing split: " + split)
    if {f["sequence"] for f in frames if f["split"] == "train"} & {f["sequence"] for f in frames if f["split"] != "train"}:
        raise ValueError("Split shares a source sequence.")
    key = [(f["sequence"], f["index"]) for f in frames]
    if len(key) != len(set(key)) or {f["split"] for f in frames} != {"train", "calibration"}:
        raise ValueError("Duplicate frames or unexpected split.")
    labels, neighbors = label_candidates(xy, frame_index, frames)
    train = np.array([frames[i]["split"] == "train" for i in frame_index]) & (labels >= 0)
    devframes = [i for i, f in enumerate(frames) if f["split"] == "calibration"]
    subsets = {"all_dev": devframes}
    for source in sorted({frames[i]["sequence"] for i in devframes}):
        subsets[source] = [i for i in devframes if frames[i]["sequence"] == source]
    common = np.column_stack([common, discovery])
    result = {"readout": "class-balanced linear logistic, L2=0.01; common 4-vector and 300 map entries standardized on eligible training candidates, each block scaled by sqrt(width); fixed capacity, effective rank varies",
              "matching": "max-cardinality one-to-one at each tied score threshold; 1px expanded boxes; duplicates count as FP, ignore-only candidates excluded",
              "metric": "descriptive empirical FROC on specified development set, thresholds swept per reported subset; no deployment operating point",
              "train_positive_candidates": int(np.sum(train & (labels == 1))),
              "train_negative_candidates": int(np.sum(train & (labels == 0))),
              "methods": {}}
    for method in METHODS:
        scores, nit = fit_and_score(common, response_input(maps, method), train, labels)
        result["methods"][method] = {"optimizer_iterations": nit}
        for name, frame_list in subsets.items():
            result["methods"][method][name] = froc(scores, frame_list, frames, frame_index, labels, neighbors)
    out = RUNS / "scores.json"
    temporary = out.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    temporary.replace(out)
    print(out)


if __name__ == "__main__":
    main()
