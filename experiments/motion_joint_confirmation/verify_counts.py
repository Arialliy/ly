"""Independent read-only count check using SciPy peaks and graph matching.

Reads predictions already committed to runs/ and recomputes the transferred
operating point, probability 0.5, and the empirical confirmation operating points.
"""
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching


HERE = Path(__file__).resolve().parent
METHODS = ("original", "A", "B", "C", "D")
COHORT = ("DJI_0075_3", "DJI_0139_2", "DJI_0343_3")
CENTERS = (19, 59, 99, 139)


def active_peaks(im):
    max_im = maximum_filter(im, size=5, mode="constant", cval=-np.inf)
    row, col = np.nonzero(im == max_im)
    order = np.lexsort((col, row, -im[row, col]))
    mask = np.zeros(im.shape, dtype=bool)
    selected = []
    for i in order:
        y, x = row[i], col[i]
        if not mask[y, x]:
            mask[max(y-2, 0):y+3, max(x-2, 0):x+3] = True
            selected.append(i)
    return np.column_stack((col[selected], row[selected])), im[row[selected], col[selected]]


def annotated_boxes(meta, frame):
    ids = {im["id"] for im in meta["images"] if im["frame_id"] == frame}
    assert len(ids) == 1
    boxes = [[], []]
    for item in meta["annotations"]:
        if item["image_id"] not in ids:
            continue
        x, y, w, h = item["bbox"]
        relevant = (item["category_id"] in range(5) and not (item.get("ignore") or item.get("iscrowd"))
                    and 0 < w*h <= 144 and max(w,h) <= 16)
        boxes[int(not relevant)].append((x-1,y-1,x+w+1,y+h+1))
    return [np.asarray(b).reshape(-1,4) for b in boxes]


def incidence(xy, limits):
    p = xy[:, None, :]
    return ((p >= limits[None, :, :2]) & (p <= limits[None, :, 2:])).all(axis=2)


def matches(edges):
    if not edges.shape[0] or not edges.shape[1]:
        return 0
    matched = maximum_bipartite_matching(csr_matrix(edges), perm_type="column")
    return int(np.count_nonzero(matched != -1))


def counts(observations, threshold):
    tp = fp = 0
    for scores, eligible in observations:
        n = 0 if threshold is None else int(np.searchsorted(-scores, -threshold, side="right"))
        num_matched = matches(eligible[:n])
        tp += num_matched
        fp += n - num_matched
    return (tp, fp)


def empirical_operating_points(observations):
    """Bound a threshold scan using the monotone false-positive count.

    A frame cannot match more than its number of boxes. Its (boxes + global
    FP budget + 1)th nonignored score is consequently beyond every requested
    budget unless an identical threshold still has to be processed as one tie.
    """
    budgets = (0.5, 1, 5)
    bound = len(observations) * 5
    scores = np.unique(np.concatenate([s[:m.shape[1]+bound+1] for s,m in observations]))[::-1]
    best = {str(b): (0,0) for b in budgets}
    for t in scores:
        tp, fp = counts(observations, t)
        if fp > bound:
            break
        for b in budgets:
            if fp <= b * len(observations) and tp > best[str(b)][0]:
                best[str(b)] = tp, fp
    return best


def verify():
    destination = HERE / "runs"
    previous = json.loads((HERE.parent/"motion_joint_factorial/runs/scores.json").read_text())
    reported = json.loads((destination/"scores.json").read_text())
    for_model = {arm: {seq: [] for seq in COHORT} for arm in METHODS}
    for seq in COHORT:
        meta = json.loads((HERE.parents[1]/"data/stage1"/seq/"annotations.json").read_text())
        for frame in CENTERS:
            target, ignore = annotated_boxes(meta, frame)
            with np.load(destination/"predictions"/f"{seq}_{frame:03d}.npz", allow_pickle=False) as maps:
                for arm in METHODS:
                    xy, scores = active_peaks(maps[arm])
                    valid = incidence(xy,target)
                    keep = valid.any(axis=1) | ~incidence(xy,ignore).any(axis=1)
                    for_model[arm][seq].append((scores[keep], valid[keep]))
    result = {}
    for arm, groups in for_model.items():
        groups["all_confirmation"] = sum([groups[s] for s in COHORT], [])
        threshold = previous["methods"][arm]["all_dev"]["froc"]["1"]["logit_threshold"]
        result[arm] = {}
        for group, observations in groups.items():
            actual = reported["methods"][arm][group]
            table = {
                "fixed_development_primary": counts(observations, threshold),
                "fixed_probability_0.5": counts(observations, 0.0),
            }
            for label, pair in table.items():
                assert pair == (actual[label]["tp"], actual[label]["fp"]), (arm, group, label, pair)
            curve = empirical_operating_points(observations)
            for label,pair in curve.items():
                row = actual["froc"][label]
                assert pair == (row["tp"],row["fp"]), (arm,group,label,pair,row)
            result[arm][group] = {"transferred":table["fixed_development_primary"],"curve_1":curve["1"]}
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    verify()
