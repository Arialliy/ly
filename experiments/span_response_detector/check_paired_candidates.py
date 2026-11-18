"""CPU-only paired span diagnostic on previously exposed civil traffic clips."""
import csv
import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PILOT = HERE.parent / "motion_hypothesis_traffic_pilot_20261118"
sys.path.insert(0, str(HERE.parent / "motion_joint_factorial"))
sys.path.insert(0, str(PILOT))
from motion_evidence import prepare_window, extract_evidence
from readout import contacts
from span_relations import span_relations

OUT = HERE / "runs/matched_premise_20261119_v1"
DIRECTIONS = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
              (0, 1), (1, -1), (1, 0), (1, 1))


def memberships(xy, boxes, margin):
    if margin == 1:
        return contacts(xy, boxes)
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    points = xy[:, None, :]
    return ((points >= boxes[None, :, :2]) &
            (points <= boxes[None, :, :2] + boxes[None, :, 2:])).all(axis=2)


def measure(window, xy):
    grids, strengths = [], []
    for chunk in torch.as_tensor(xy).split(32):
        evidence = extract_evidence(window, chunk, "split")
        relations, valid = span_relations(evidence)
        assert valid.all(), "interior points must have full support"
        grids.append(relations.cumsum(dim=3).numpy())
        short_strength = evidence["patches"][:, :, 1, 1, 2, 2, 1, 1].abs()
        strengths.append(short_strength.mean(dim=1).numpy())
    return np.concatenate(grids), np.concatenate(strengths)


def scores(grids):
    # [point, exchange, anchor, span, direction]; each ray uses 1- and 2-cell offsets.
    rays = np.stack([(grids[..., 2+dy, 2+dx] + grids[..., 2+2*dy, 2+2*dx])/2
                     for dy, dx in DIRECTIONS], axis=-1)
    short, long = rays[:, :, :, 0], rays[:, :, :, 2]
    return {"short": short.min(axis=-1).mean(axis=1),
            "long": long.min(axis=-1).mean(axis=1),
            "change": (long-short).min(axis=-1).mean(axis=1)}


def pair(positive, negative, traces, rank, edge, strength):
    if not len(positive) or not len(negative):
        return []
    pos, neg = np.asarray(positive)[:, None], np.asarray(negative)[None, :]
    rms = np.sqrt(np.mean((traces[pos]-traces[neg])**2, axis=-1))
    distances = np.stack((rms/.5, np.abs(traces[pos, 8]-traces[neg, 8])/.5,
                          np.abs(rank[pos]-rank[neg])/.1, np.abs(edge[pos]-edge[neg])/32,
                          np.abs(strength[pos]-strength[neg])/.5))
    eligible = distances.max(axis=0) <= 1
    # A single extra valid match outweighs the total possible valid assignment cost.
    penalty = len(positive)+1
    cost = np.where(eligible, distances.mean(axis=0), 2*penalty)
    cost = np.column_stack((cost, np.full((len(positive), len(positive)), penalty)))
    pi, ni = linear_sum_assignment(cost)
    return [(positive[i], negative[j], float(rms[i, j])) for i, j in zip(pi, ni)
            if j < len(negative) and eligible[i, j]]


def key_wins(rows, key):
    values = np.array([r[key] for r in rows])
    return {"wins": int((values > 0).sum()), "ties": int((values == 0).sum()),
            "losses": int((values < 0).sum()),
            "median_target_minus_background": float(np.median(values)) if len(values) else None}


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    frames = json.loads((PILOT/"runs/release_labels.json").read_bytes())["frames"]
    split = json.loads((ROOT/"results/references/DATA_SELECTION.json").read_bytes())["splits"]
    assert all(f["sequence"] in split[f["split"]] and f["split"] in ("train", "calibration")
               for f in frames)
    sources = sorted({f["sequence"] for f in frames})
    with np.load(PILOT/"runs/release_sample.npz", allow_pickle=False) as data:
        coords, logits, frame_ids = (data[k] for k in ("xy", "base", "frame"))
    rows, frame_counts, paired_grids = [], [], []
    for fi, f in enumerate(frames):
        seq, idx = f["sequence"], f["index"]
        raw = np.load(ROOT/"data/stage1"/seq/"raw_frames.npy", mmap_mode="r")
        window = prepare_window(raw[idx-8:idx+9].copy())
        xy, base = coords[frame_ids == fi], logits[frame_ids == fi]
        assert len(xy) == 512 and np.isfinite(base).all() and np.all(np.diff(base) <= 0)
        rank = np.arange(len(base), dtype=float)/(len(base)-1)
        x, y = xy.T
        height, width = window.shape[-2:]
        edge = np.minimum.reduce((x, y, width-1-x, height-1-y))
        inside = edge >= 25
        xy, base, edge, rank = (a[inside] for a in (xy, base, edge, rank))
        in_box = memberships(xy, f["targets"], 0)
        near_boxes = memberships(xy, f["targets"]+f["ignore"], 1)
        unique = in_box & (near_boxes.sum(axis=1)[:, None] == 1)
        box_to_point = {box: int(np.flatnonzero(unique[:, box])[0])
                        for box in range(unique.shape[1]) if unique[:, box].any()}
        box_for_point = {point: box for box, point in box_to_point.items()}
        positive = list(box_for_point)
        negative = np.flatnonzero(~near_boxes.any(axis=1))
        grids, strength = measure(window, xy)
        s = scores(grids)
        traces = window[:, xy[:, 1], xy[:, 0]].T.numpy()
        pairs = pair(positive, negative, traces, rank, edge, strength)
        frame_counts.append(dict(sequence=seq, center=idx, boxes=len(f["targets"]),
                                 eligible_target_peaks=len(positive), eligible_background_peaks=len(negative),
                                 pairs=len(pairs)))
        for pi, ni, rms in pairs:
            box = box_for_point[pi]
            area = np.prod(f["targets"][box][2:])
            r = dict(sequence=seq, split=f["split"], center=idx, box=box, area=int(area),
                     target_xy=xy[pi].tolist(), background_xy=xy[ni].tolist(), profile_rms=rms,
                     score_rank_gap=float(abs(rank[pi]-rank[ni])),
                     edge_gap=int(abs(edge[pi]-edge[ni])), strength_gap=float(abs(strength[pi]-strength[ni])),
                     center_contrast_gap=float(abs(traces[pi, 8]-traces[ni, 8])))
            for anchor, a in (("moving", 1), ("static", 0)):
                for span, values in s.items():
                    r[anchor+"_"+span] = float(values[pi, a]-values[ni, a])
            rows.append(r)
            paired_grids.append(grids[[pi, ni]])
    labels = [a+"_"+s for a in ("moving", "static") for s in ("short","long","change")]
    def summary(subset):
        return {"pairs": len(subset), **{k:key_wins(subset,k) for k in labels}}
    result = {"frames": frame_counts, "sources":{q:summary([r for r in rows if r["sequence"]==q])
              for q in sources}, "pooled":summary(rows),
              "target_bbox_area":{label: summary([r for r in rows if low < r["area"] <= high])
              for label,low,high in (("1-16",0,16),("17-64",16,64),("65-144",64,144))}}
    assert all(r["profile_rms"]<=.5 and r["score_rank_gap"]<=.1 and r["edge_gap"]<=32
               and r["strength_gap"]<=.5 and r["center_contrast_gap"]<=.5 for r in rows)
    outputs = [OUT/n for n in ("summary.json", "pairs.csv", "paired_Q.npz")]
    if any(p.exists() for p in outputs):
        raise FileExistsError("Refusing to replace a completed diagnostic")
    if rows:
        with outputs[1].open("x") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    np.savez_compressed(outputs[2], Q=np.asarray(paired_grids))
    outputs[0].write_text(json.dumps(result, indent=2)+"\n")
    print("frames",len(frames),"pairs",len(rows),"sources",Counter(r["sequence"] for r in rows))


if __name__ == "__main__":
    main()
