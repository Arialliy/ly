"""Independently recount S2b's endpoint operating points from saved native logits and original masks."""
import argparse
from collections import defaultdict, deque
import hashlib
from itertools import groupby
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import label, maximum_filter
from scipy.special import expit


BUDGETS = (0.5, 1, 5)


def regional_peaks(scores):
    """Flood equal-score plateaus from local candidate pixels, checking their entire perimeter."""
    if scores.dtype != np.float32 or scores.ndim != 2 or not scores.size or not np.isfinite(scores).all():
        raise ValueError("Expected finite native FP32 logits")
    h, w = scores.shape
    candidates = scores == maximum_filter(scores, size=3, mode="constant", cval=-np.inf)
    seen = np.zeros((h, w), dtype=bool)
    representatives = []
    for index in np.flatnonzero(candidates):
        by, bx = divmod(int(index), w)
        if seen[by, bx]:
            continue
        score = float(scores[by, bx])
        queue = deque([(by, bx)])
        seen[by, bx] = True
        is_maximum = True
        representative = (by, bx)
        while queue:
            y, x = queue.popleft()
            representative = min(representative, (y, x))
            for yy in range(max(0, y-1), min(h, y+2)):
                for xx in range(max(0, x-1), min(w, x+2)):
                    neighbor = float(scores[yy, xx])
                    if neighbor > score:
                        is_maximum = False
                    elif neighbor == score and not seen[yy, xx]:
                        seen[yy, xx] = True
                        queue.append((yy, xx))
        if is_maximum:
            representatives.append((score, *representative))

    # Check conflicts against already retained coordinates in nearby spatial buckets.
    kept = []
    cells = defaultdict(list)
    for score, y, x in sorted(representatives, key=lambda p: (-p[0], p[1], p[2])):
        cy, cx = y // 3, x // 3
        conflicts = any(abs(y - yy) <= 2 and abs(x - xx) <= 2
                        for i in range(cy - 1, cy + 2) for j in range(cx - 1, cx + 2)
                        for yy, xx in cells.get((i, j), ()))
        if not conflicts:
            kept.append((score, y, x))
            cells[cy, cx].append((y, x))
    return kept


def budget_points(events, truth):
    """Enumerate the grouped score sweep up to the largest requested false-positive count."""
    n, gt = len(truth), sum(truth.values())
    counts = {frame: 0 for frame in truth}
    matched = set()
    results = {str(b): {"threshold": None, "tp": 0, "fp": 0, "empty_frame_fp": 0, "pd": 0.,
                        "per_frame": {k: (0, 0) for k in truth}} for b in BUDGETS}
    total = 0
    for score, group in groupby(sorted(events, reverse=True), key=lambda p: p[0]):
        for _, frame, eligible in group:
            counts[frame] += 1
            total += 1
            if eligible:
                matched.add(frame)
        tp = len(matched)
        fp = total - tp
        if fp > n * max(BUDGETS):
            break   # FP never decreases in the one-GT-per-frame diagnostic population.
        for b in BUDGETS:
            old = results[str(b)]
            if fp > b * n or (tp, -fp) <= (old["tp"], -old["fp"]):
                continue
            per_frame = {k: (int(k in matched), count - int(k in matched)) for k, count in counts.items()}
            results[str(b)] = {"threshold": score, "tp": tp, "fp": fp, "pd": tp / gt,
                               "empty_frame_fp": sum(counts[k] for k in truth if not truth[k]),
                               "per_frame": per_frame}
    return results


def native_loss(scores, mask):
    z = scores.astype(np.float64)
    bg = float(np.logaddexp(0, z[~mask]).mean())
    if not mask.any():
        return {"total": bg, "foreground_bce": None, "background_bce": bg,
                "dice_loss_positive": None, "empty_frame_total": bg}
    fg = float(np.logaddexp(0, -z[mask]).mean())
    p = expit(z)
    dice = 1 - (2 * p[mask].sum() + 1) / (p.sum() + mask.sum() + 1)
    return {"total": (fg + bg + dice) / 2, "foreground_bce": fg, "background_bce": bg,
            "dice_loss_positive": float(dice), "empty_frame_total": None}


def verify_step(run_dir, samples, root, step):
    archive = run_dir / f"logits_{step:03d}.npz"
    snapshot = json.loads((run_dir / f"snapshot_{step:03d}.json").read_bytes())
    if snapshot["step"] != step:
        raise ValueError("Snapshot step mismatch")
    truth, events, losses = {}, [], []
    with np.load(archive, allow_pickle=False) as data:
        expected = {s["sample_id"] for s in samples}
        if len(data.files) != len(expected) or set(data.files) != expected:
            raise ValueError("Stored logits do not exhaust the frozen sample IDs")
        for row in samples:
            name = row["sample_id"]
            mask_path = root / row["center_mask"]["path"]
            if hashlib.sha256(mask_path.read_bytes()).hexdigest() != row["center_mask"]["sha256"]:
                raise ValueError(f"Center mask digest changed: {name}")
            with Image.open(mask_path) as image:
                mask = np.asarray(image) == 255
            _, n_gt = label(mask, structure=np.ones((3, 3)))
            if n_gt != int(row["positive"]) or np.count_nonzero(mask) != row["foreground_pixels"]:
                raise ValueError(f"Expected the frozen 0-or-1 truth per frame: {name}")
            scores = data[name]
            if scores.shape != mask.shape or list(scores.shape) != row["native_hw"]:
                raise ValueError(f"Geometry mismatch at {name}")
            truth[name] = int(n_gt)
            center = np.argwhere(mask).mean(axis=0) if n_gt else None
            for score, y, x in regional_peaks(scores):
                eligible = center is not None and (y - center[0])**2 + (x - center[1])**2 <= 9
                events.append((score, name, bool(eligible)))
            losses.append(native_loss(scores, mask))
    expected_froc = snapshot["froc"]
    if (len(truth), sum(truth.values())) != (expected_froc["frame_count"], expected_froc["gt"]):
        raise ValueError("FROC population differs from frozen masks")
    points = budget_points(events, truth)
    for budget, result in points.items():
        recorded = expected_froc["budgets"][budget]
        frame_counts = result.pop("per_frame")
        if any((recorded["per_frame"][k]["tp"], recorded["per_frame"][k]["fp"]) != v
               for k, v in frame_counts.items()) or any(recorded[k] != value for k, value in result.items()):
            raise ValueError(f"Independent endpoint counts or threshold differ: step {step}, {budget} FP/frame")
    average = {key: float(np.mean([loss[key] for loss in losses if loss[key] is not None])) for key in losses[0]}
    for key, value in average.items():
        if not np.isclose(value, snapshot["loss"][key], atol=1e-6, rtol=1e-5):
            raise ValueError(f"Independent float64 loss differs: step {step}, {key}")
    return {"budgets": points, "native_nms_peak_count": len(events), "loss_float64": average,
            "logits_sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}


def verify_endpoints(run_dir: Path, subset: dict, project_root: Path) -> dict:
    steps = {str(step): verify_step(Path(run_dir), subset["samples"], Path(project_root), step) for step in (0, 400)}
    return {"status": "PASS", "steps": steps, "source": "Native FP32 saved logits and original training masks",
            "method": "Independent equal-plateau flood, Chebyshev NMS, tied-score sweep, native centroids; "
                      "all operating points and per-frame counts, plus float64 loss components."}


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    subset = json.loads((root / "experiments/irstd_independent_v1/manifests/s2b_train12.json").read_bytes())
    print(json.dumps(verify_endpoints(args.run_dir, subset, root), indent=2, allow_nan=False))
