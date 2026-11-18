"""Native-resolution S4 peak FROC and independent DeepPro-Plus centroid-mask evaluation."""
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import auc

from ..checks.check_s0 import atomic_text
from ..data.dataset import _components
from .s2b_froc import FramePeaks, _at_budget, _curve, _frame_peaks, _save_curve


BUDGETS = (0.5, 1, 5)
PROJECT = Path(__file__).resolve().parents[3]
AUTHOR_ROOT = PROJECT / "sources/official_deeppro"
AUTHOR_SOURCES = {
    "ShootingRules.py": "888d4a26ff2877dc64550bc919fa60bf18e04e0af795aa8e8c304aa0b1e5c4e2",
    "test.py": "99b58962061f101bcaf1cb4e787a46599ab947805444af910d405324cf314b9b",
    "write_results.py": "47984378234a8f6e1ec3e18d83a50af8b93cce52645f1e04042afe52f04654ef",
    "data_utils/TestDataLoader.py": "03210e84ac9acfc3fd85ef5bcd2c787cfc126e2580e622fca9c3297c3e1c022e",
}
AUTHOR_THRESHOLDS = np.array([0, 1e-20, 1e-10, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1,
                              .2, .3, .35, .4, .45, .5, .55, .6, .65, .7, .8, .85, .9, .95, .99, 1])
TEST_LIST = PROJECT / "data/public/NUDT-MIRSDT/test.txt"
TEST_LIST_SHA = "eb039aa35aaabab2eae559b9d627e53d3fa114eb0410f343498f154c81a9d5d0"


@lru_cache(maxsize=1)
def _author_evaluator():
    for name, expected in AUTHOR_SOURCES.items():
        if hashlib.sha256((AUTHOR_ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"DeepPro-Plus reference source changed: {name}")
    path = AUTHOR_ROOT / "ShootingRules.py"
    spec = importlib.util.spec_from_file_location("s4_deeppro_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ShootingRules().eval()


def _author_counts(probability, centroid):
    """Reuse the upstream rule on each distinct thresholded probability map."""
    evaluator = _author_evaluator()
    counts = []
    previous_size = -1
    for threshold in AUTHOR_THRESHOLDS:
        size = np.count_nonzero(probability >= threshold)
        # The sets are nested, so equal sizes yield precisely the same input mask.
        if size != previous_size:
            result = tuple(map(int, evaluator(probability[None], centroid[None], threshold)))
            previous_size = size
        counts.append(result)
    return np.asarray(counts, dtype=np.int64)


@dataclass(frozen=True)
class PreparedFrame:
    peaks: FramePeaks
    native_hw: tuple
    loss: dict
    author_counts: np.ndarray | None
    intersection: int
    union: int


def prepare_frame(frame_id, native_logits, mask, *, centroid_mask=None):
    """Reduce one native FP32 map and its native binary mask, then release both."""
    if not isinstance(frame_id, str) or len(frame_id.split("/")) != 2 or not all(frame_id.split("/")):
        raise ValueError("A unique source/frame identifier is required")
    logits = np.asarray(native_logits)
    mask = np.asarray(mask)
    if mask.shape != logits.shape or logits.ndim != 2 or logits.dtype != np.float32:
        raise ValueError("Expected one full native FP32 score map and a same-size binary mask")
    if not mask.size or not (np.isin(mask, [0, 1]).all() or np.isin(mask, [0, 255]).all()):
        raise ValueError("Unsupported mask encoding")
    binary = np.ascontiguousarray(mask != 0, dtype=np.uint8)
    components = _components(binary)
    # The frozen S0 data audit found at most one instance per eligible native mask.
    peaks = _frame_peaks(frame_id, logits, [o.centroid_xy for o in components])
    foreground = binary.astype(bool)
    probability = torch.from_numpy(np.ascontiguousarray(logits)).sigmoid().numpy()
    z = logits.astype(np.float64)
    fg_loss = float(np.logaddexp(0, -z[foreground]).mean()) if foreground.any() else None
    bg_loss = float(np.logaddexp(0, z[~foreground]).mean()) if not foreground.all() else None
    balanced = np.mean([x for x in (fg_loss, bg_loss) if x is not None])
    dice = None
    if foreground.any():
        dice = float(1 - (2 * probability[foreground].sum(dtype=np.float64) + 1)
                     / (probability.sum(dtype=np.float64) + foreground.sum() + 1))
    loss = {"total": float(balanced + 0.5 * (dice or 0)), "foreground_bce": fg_loss,
            "background_bce": bg_loss, "dice_loss": dice}
    native = None
    if centroid_mask is not None:
        centroid = np.asarray(centroid_mask)
        if centroid.shape != mask.shape or not (np.isin(centroid, [0, 1]).all() or np.isin(centroid, [0, 255]).all()):
            raise ValueError(f"Invalid DeepPro-Plus centroid mask: {frame_id}")
        native = _author_counts(probability, (centroid != 0).astype(np.uint8))
    predicted = probability > .5  # test.py uses strict > for IoU and inclusive >= for ShootingRules.
    return PreparedFrame(peaks, binary.shape, loss, native, int((predicted & foreground).sum()),
                         int((predicted | foreground).sum()))


def _fixed_point(frames, threshold):
    applied = np.inf if threshold is None else float(threshold)
    per_frame = {}
    for f in frames:
        keep = f.scores >= applied
        eligible = np.flatnonzero(keep & (f.distance_sq <= 9))
        gt = len(f.truth_xy)
        tp = int(bool(eligible.size))
        row = {"gt": gt, "tp": tp, "fp": int(keep.sum()) - tp}
        if tp:
            best = eligible[np.argmin(f.distance_sq[eligible])]
            row["matched_peak_xy"] = f.yx[best, ::-1].tolist()
            row["match_distance_px"] = float(np.sqrt(f.distance_sq[best]))
        per_frame[f.frame_id] = row
    gt, tp, fp = (sum(r[key] for r in per_frame.values()) for key in ("gt", "tp", "fp"))
    by_source = {}
    for frame_id, row in per_frame.items():
        source = frame_id.split("/")[0]
        totals = by_source.setdefault(source, {"frames": 0, "gt": 0, "tp": 0, "fp": 0})
        totals["frames"] += 1
        for key in ("gt", "tp", "fp"):
            totals[key] += row[key]
    return {"threshold": threshold, "frames": len(frames), "gt": gt, "tp": tp, "fp": fp, "per_sequence": by_source,
            "empty_frames": sum(r["gt"] == 0 for r in per_frame.values()),
            "empty_frame_fp": sum(r["fp"] for r in per_frame.values() if r["gt"] == 0),
            "pd": tp / gt if gt else None, "fp_per_frame": fp / len(frames), "per_frame": per_frame}


def _descriptive(frames, curve=None):
    thresholds, tp, fp = _curve(frames) if curve is None else curve
    return {str(b): _at_budget(frames, thresholds, tp, fp, b) for b in BUDGETS}


def _native_group(group):
    counts = np.sum([f.author_counts for f in group], axis=0)
    pixels = sum(int(np.prod(f.native_hw)) for f in group)
    fa = counts[:, 0] / pixels
    gt = int(counts[0, 2])
    pd = counts[:, 1] / gt if gt else None
    half = int(np.flatnonzero(AUTHOR_THRESHOLDS == 0.5)[0])
    union = sum(frame.union for frame in group)
    return {"frames": len(group), "pixels": pixels, "gt": gt,
            "foreground_iou_at_probability_0.5": sum(f.intersection for f in group) / union if union else None,
            "pd_at_probability_0.5": float(pd[half]) if gt else None,
            "fa_at_probability_0.5": float(fa[half]),
            "tp_at_probability_0.5": int(counts[half, 1]), "false_pixels_at_probability_0.5": int(counts[half, 0]),
            "auc": float(auc(fa, pd)) if gt else None,
            "false_pixels": counts[:, 0].tolist(), "true_components": counts[:, 1].tolist(),
            "fa": fa.tolist(), "pd": pd.tolist() if gt else None}


def _fit_summary(frames):
    def average(key, rows):
        values = [r.loss[key] for r in rows if r.loss[key] is not None]
        return float(np.mean(values)) if values else None
    empty = [f for f in frames if not len(f.peaks.truth_xy)]
    return {"frames": len(frames), "loss": {key: average(key, frames) for key in frames[0].loss},
            "empty_frame_loss": average("total", empty), "empty_frames": len(empty),
            "per_frame": {f.peaks.frame_id: f.loss for f in frames}}


def _historical_sources(groups):
    release = TEST_LIST.read_bytes()
    if hashlib.sha256(release).hexdigest() != TEST_LIST_SHA:
        raise ValueError("Released sequence-order file changed")
    rows = [line.split("/") for line in release.decode().splitlines() if line]
    sources = list(dict.fromkeys(fields[0] for fields in rows))
    expected = {f"{seq}/{Path(name).stem}" for seq, _, name in rows if 9 <= int(Path(name).stem) <= 92}
    evaluated = {f.peaks.frame_id for group in groups.values() for f in group}
    if len(rows) != 2000 or len(sources) != 20 or len(expected) != 1680 or expected != evaluated:
        raise ValueError("Historical evaluation requires the exact 17-frame valid release centers")
    return sources


def evaluate_frames(frames, output_dir, *, leave_one_source_out=False, frozen_thresholds=None,
                    population="development"):
    """Save the pooled FROC, development sensitivity or actual historical transfer, and native metrics."""
    frames = list(frames)
    if not frames or len({f.peaks.frame_id for f in frames}) != len(frames):
        raise ValueError("Empty frame list or duplicate source/frame IDs")
    groups = defaultdict(list)
    for frame in frames:
        groups[frame.peaks.frame_id.split("/")[0]].append(frame)
    if population not in ("development", "historical_test", "development_monitor"):
        raise ValueError(f"Unrecognized S4 evaluation population: {population}")
    if population == "historical_test":
        if not isinstance(frozen_thresholds, dict) or set(frozen_thresholds) != {"0.5", "1", "5"}:
            raise ValueError("Historical evaluation requires all three frozen development thresholds")
        if any(t is not None and not np.isfinite(t) for t in frozen_thresholds.values()):
            raise ValueError("Non-finite historical threshold; encode the empty prediction point with null")
        source_order = _historical_sources(groups)
    elif frozen_thresholds is not None:
        raise ValueError("Frozen threshold transfer is defined on historical test")
    if leave_one_source_out and (population != "development" or len(groups) < 2):
        raise ValueError("Development sensitivity requires at least two development sources")
    output = Path(output_dir)
    output.mkdir(exist_ok=True, parents=True)
    peaks = [f.peaks for f in frames]
    curve = _curve(peaks)
    curve_path = output / "froc.npz"
    description = {"frames": len(frames), "gt": sum(len(p.truth_xy) for p in peaks),
                   "empty_frames": sum(not len(p.truth_xy) for p in peaks), "curve_file": curve_path.name,
                   "curve_sha256": _save_curve(curve_path, peaks, *curve),
                   "thresholds_including_empty": len(curve[0]), "budgets": _descriptive(peaks, curve)}
    del curve
    missing = [f.peaks.frame_id for f in frames if f.author_counts is None]
    if missing:
        native = {"available": False, "reason": "DeepPro-Plus masks_centroid unavailable for some or all supplied centers",
                  "missing_count": len(missing), "first_missing": missing[0]}
    else:
        native = {"available": True, "scope": ("17-frame valid-center subset (1680 of 2000 released test frames)"
                                               if population == "historical_test" else population),
                  "score_domain": "sigmoid_of_native_fp32_logits", "label_source": "masks_centroid",
                  "author_source_root": "sources/official_deeppro", "source_sha256": AUTHOR_SOURCES,
                  "probability_thresholds": AUTHOR_THRESHOLDS.tolist(), "overall": _native_group(frames),
                  "per_source": {s: _native_group(g) for s, g in groups.items()}}
        if population == "historical_test":
            # write_results.py indices refer to the unmodified release order in TestDataLoader.py.
            group_indices = {"low": (7, 13, 14, 15, 16, 17, 18, 19), "high": (0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12)}
            native["release_test_list_sha256"] = TEST_LIST_SHA
            native["snr_groups"] = {}
            for name, indices in group_indices.items():
                sources = [source_order[i] for i in indices]
                native["snr_groups"][name] = {"sources": sources, **_native_group([f for s in sources for f in groups[s]])}
    sensitivity = {}
    if leave_one_source_out:
        for held in groups:
            retained = [f.peaks for s, other in groups.items() if s != held for f in other]
            points = _descriptive(retained)
            sensitivity[held] = {"sources": sorted(set(groups) - {held}), "frames": len(retained),
                                 "gt": sum(len(f.truth_xy) for f in retained), "budgets": {
                                     b: {k: row[k] for k in ("threshold", "tp", "fp", "pd", "fp_per_frame")}
                                     for b, row in points.items()}}
    transferred = {b: _fixed_point(peaks, t) for b, t in (frozen_thresholds or {}).items()}
    result = {"population": population, "descriptive": description, "leave_one_source_out": sensitivity,
              "transferred": transferred, "native_author": native, "fit": _fit_summary(frames)}
    atomic_text(output / "metrics.json", json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
