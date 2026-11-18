"""Frozen S0 peak rules and descriptive FROC on native S2b train-center logits.

The fixed subset has at most one annotated component per frame. At any threshold,
the eligible peak closest to that centroid gives the maximum-cardinality,
minimum-distance match. An additional eligible peak contributes one false alarm.
"""
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import tempfile

import numpy as np
from scipy.ndimage import label
from skimage.morphology import local_maxima


@dataclass(frozen=True)
class FramePeaks:
    frame_id: str
    truth_xy: np.ndarray
    yx: np.ndarray     # Greedy selection order: descending logit, row, column
    scores: np.ndarray
    distance_sq: np.ndarray


def _peaks(logits):
    score = np.asarray(logits)
    if score.dtype != np.float32 or score.ndim != 2 or not score.size or not np.isfinite(score).all():
        raise ValueError("Expected a finite, nonempty native H x W FP32 logit image")

    # The sentinel ring makes a constant image a single border-touching maximum.
    padded = np.pad(score, 1, constant_values=-np.inf)
    maxima = local_maxima(padded, connectivity=2, allow_borders=True)[1:-1, 1:-1]
    regions, n = label(maxima, structure=np.ones((3, 3), dtype=bool))
    flat = np.flatnonzero(maxima)
    representatives = np.full(n + 1, score.size, dtype=np.intp)
    np.minimum.at(representatives, regions.ravel()[flat], flat)
    y, x = np.unravel_index(representatives[1:], score.shape)
    order = np.lexsort((x, y, -score[y, x]))

    # Padding makes the 5x5 Chebyshev suppression neighbourhood uniform at edges.
    height, width = score.shape
    blocked = np.zeros((height + 4, width + 4), dtype=bool)
    keep = []
    for idx in order:
        row, col = y[idx], x[idx]
        if not blocked[row + 2, col + 2]:
            keep.append(idx)
            blocked[row:row + 5, col:col + 5] = True
    keep = np.asarray(keep, dtype=np.intp)
    yx = np.column_stack((y[keep], x[keep])).astype(np.int32)
    return yx, score[y[keep], x[keep]]


def _frame_peaks(frame_id, logits, centroids):
    xy = np.asarray(centroids, dtype=np.float64)
    if xy.size == 0:
        xy = xy.reshape(0, 2)
    if xy.shape not in ((0, 2), (1, 2)):
        raise ValueError("S2b FROC requires at most one (x,y) centroid per frame")
    yx, scores = _peaks(logits)
    if len(xy):
        height, width = logits.shape
        if not np.isfinite(xy).all() or (xy < 0).any() or (xy[0] > (width - 1, height - 1)).any():
            raise ValueError(f"Invalid native centroid at {frame_id}")
        delta = yx[:, ::-1] - xy[0]
        distance_sq = np.square(delta).sum(axis=1)
    else:
        distance_sq = np.full(len(scores), np.inf)
    return FramePeaks(frame_id, xy, yx, scores, distance_sq)


def _curve(frames):
    all_scores = np.concatenate([frame.scores for frame in frames])
    tp_starts = []
    for frame in frames:
        starts = np.zeros(len(frame.scores), dtype=np.int32)
        eligible = np.flatnonzero(frame.distance_sq <= 9.0)
        if eligible.size:
            starts[eligible[0]] = 1  # highest scoring eligible peak; counted once at its threshold
        tp_starts.append(starts)
    order = np.argsort(-all_scores, kind="stable")
    ordered_scores = all_scores[order]
    group_ends = np.flatnonzero(np.r_[ordered_scores[:-1] != ordered_scores[1:], True])
    thresholds = np.r_[np.float32(np.inf), ordered_scores[group_ends]]
    tp = np.r_[0, np.cumsum(np.concatenate(tp_starts)[order])[group_ends]]
    detections = np.r_[0, group_ends + 1]
    return thresholds, tp, detections - tp


def _at_budget(frames, thresholds, tp, fp, budget):
    allowed = fp <= budget * len(frames)
    best_tp = tp[allowed].max()
    best_fp = fp[allowed & (tp == best_tp)].min()
    index = np.flatnonzero((tp == best_tp) & (fp == best_fp))[0]
    threshold = thresholds[index]
    per_frame = {}
    empty_fp = 0
    per_source = {}
    for frame in frames:
        kept = frame.scores >= threshold
        n = int(kept.sum())
        eligible = np.flatnonzero(kept & (frame.distance_sq <= 9.0))
        matched = int(bool(eligible.size))
        record = {"gt": len(frame.truth_xy), "tp": matched, "fp": n - matched}
        if matched:
            closest = eligible[np.argmin(frame.distance_sq[eligible])]
            record["matched_peak_xy"] = frame.yx[closest, ::-1].tolist()
            record["match_distance_px"] = float(np.sqrt(frame.distance_sq[closest]))
        if not len(frame.truth_xy):
            empty_fp += n
        per_frame[frame.frame_id] = record
        source = frame.frame_id.split("/")[0]
        totals = per_source.setdefault(source, {"frames": 0, "gt": 0, "tp": 0, "fp": 0})
        totals["frames"] += 1
        for name in ("gt", "tp", "fp"):
            totals[name] += record[name]
    for name, expected in (("tp", best_tp), ("fp", best_fp)):
        if sum(row[name] for row in per_frame.values()) != expected:
            raise RuntimeError(f"Native one-to-one matching disagrees with FROC event counts for {name}")
    gt = sum(len(frame.truth_xy) for frame in frames)
    return {"threshold": None if index == 0 else float(threshold), "empty_prediction_point": bool(index == 0),
            "pd": float(best_tp / gt) if gt else None, "tp": int(best_tp), "fp": int(best_fp),
            "fp_per_frame": float(best_fp / len(frames)), "empty_frame_fp": empty_fp,
            "per_frame": per_frame, "per_sequence": per_source}


def _save_curve(path, frames, thresholds, tp, fp):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".froc-", delete=False) as stream:
        temp = Path(stream.name)
        try:
            # Keep peaks and centroids so the threshold counts can be replayed independently.
            np.savez_compressed(stream, thresholds=thresholds, tp=tp, fp=fp,
                                frame_ids=np.asarray([f.frame_id for f in frames]),
                                gt_xy=np.asarray([f.truth_xy[0] if len(f.truth_xy) else [np.nan, np.nan]
                                                  for f in frames], dtype=np.float64),
                                peak_count=np.asarray([len(f.scores) for f in frames], dtype=np.int32),
                                peak_yx=np.concatenate([f.yx for f in frames]),
                                peak_scores=np.concatenate([f.scores for f in frames]))
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_froc(frames, curve_path):
    """Evaluate all distinct raw-logit thresholds for (ID, native logits, centroids_xy) records."""
    prepared = [_frame_peaks(frame_id, logits, centroids) for frame_id, logits, centroids in frames]
    if not prepared or len({f.frame_id for f in prepared}) != len(prepared):
        raise ValueError("Need at least one frame, identified by a unique source/frame ID")
    thresholds, tp, fp = _curve(prepared)
    points = {f"{budget:g}": _at_budget(prepared, thresholds, tp, fp, budget) for budget in (0.5, 1, 5)}
    digest = _save_curve(curve_path, prepared, thresholds, tp, fp)
    return {"frame_count": len(prepared), "empty_frames": sum(not len(f.truth_xy) for f in prepared),
            "gt": sum(len(f.truth_xy) for f in prepared), "score_domain": "native_fp32_logits",
            "threshold_count_including_empty": len(thresholds), "budgets": points,
            "curve_file": Path(curve_path).name, "curve_sha256": digest}


def check_froc_fixtures():
    # Flat, border and diagonal plateaus; interior of a non-maximal plateau must never survive.
    flat = np.zeros((5, 6), dtype=np.float32)
    if _peaks(flat)[0].tolist() != [[0, 0]]:
        raise AssertionError("Flat plateau must have one lexicographic representative")
    edge = np.zeros((11, 13), dtype=np.float32)
    edge[:7, :7] = 1
    edge[0, 0] = 2
    edge[0, 12] = edge[1, 11] = 3
    if _peaks(edge)[0].tolist() != [[0, 12], [0, 0]]:
        raise AssertionError("Plateau/border definition changed")
    suppressed = np.zeros((7, 9), dtype=np.float32)
    suppressed[0, 0] = suppressed[2, 2] = suppressed[0, 3] = 1
    if _peaks(suppressed)[0].tolist() != [[0, 0], [0, 3]]:
        raise AssertionError("Tie order or inclusive Chebyshev suppression changed")

    image = np.zeros((9, 11), dtype=np.float32)
    image[1, 1], image[1, 4] = 8, 5
    negative = np.zeros_like(image)
    negative[0, 0] = negative[8, 10] = 8
    frames = [_frame_peaks("A/1", image, [(4.0, 1.0)]), _frame_peaks("B/1", negative, [])]
    thresholds, tp, fp = _curve(frames)
    if not (np.array_equal(thresholds, [np.inf, 8, 5]) and tp.tolist() == [0, 1, 1]
            and fp.tolist() == [0, 2, 3]):
        raise AssertionError("Ties, inclusive radius=3, or one-to-one scoring changed")
    if _at_budget(frames, thresholds, tp, fp, 0.5)["tp"] != 0:
        raise AssertionError("Score ties must enter the FROC curve as a group")
    chosen = _at_budget(frames, thresholds, tp, fp, 1)
    if (chosen["empty_frame_fp"], chosen["threshold"], chosen["per_frame"]["A/1"]["matched_peak_xy"]) != (2, 8, [1, 1]):
        raise AssertionError("Operating point selection, empty-frame counts, or coordinate order changed")
    # A second target first detected at threshold 5 makes that lower threshold optimal.
    extra = np.zeros_like(image)
    extra[8, 10] = 5
    frames.append(_frame_peaks("C/1", extra, [(10, 8)]))
    thresholds, tp, fp = _curve(frames)
    closest = _at_budget(frames, thresholds, tp, fp, 5)
    if closest["threshold"] != 5 or closest["per_frame"]["A/1"]["matched_peak_xy"] != [4, 1]:
        raise AssertionError("Wrong minimum-distance match among multiple eligible peaks")
    try:
        _frame_peaks("A/2", image, [(1, 1), (5, 1)])
    except ValueError:
        pass
    else:
        raise AssertionError("The single-GT metric accepted multiple components")
    return ["regional_plateaus_and_borders", "deterministic_suppression", "tied_scores",
            "inclusive_centroid_matching", "minimum_distance_among_eligible", "empty_frame_budget",
            "reject_multi_component_frames"]
