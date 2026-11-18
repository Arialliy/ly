"""Fixed training pairs and release-defined cohorts for actual-update checks.

No model is run here. ``--build-plan`` uses annotation/old-cache coverage only.
``prepare_entry`` consumes the root runner's release logits; ``measure`` reads
the same frozen regions/cohorts from any supplied model logits. All outputs are
JSON-compatible and all historical artifacts remain read-only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

E = Path(__file__).resolve().parents[1]
ROOT = E.parents[1]
V1 = ROOT / "experiments/track_contrast_v1"
V2 = ROOT / "experiments/track_contrast_v2"
Q = ROOT / "archive/infrared_problem_screen_20261009/temporal_diagnosis_20261010"
PLAN = E / "PAIR_PLAN.json"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@lru_cache(maxsize=1)
def segment_plan():
    return read(V2 / "SEGMENT_PLAN.json")


def build_plan():
    """Balance each source 4/role, with same-source disjoint-frame pairs.

    Deterministic backtracking prefers old nonempty C pairs, then less used
    sources, then SHA256(seed=46, entry IDs). This is coverage-based design,
    never selection on a new update, teacher score, or development response.
    """
    if PLAN.exists():
        raise FileExistsError("PAIR_PLAN already exists; do not select again")
    segment = segment_plan()
    train_path = Path(next(p for p in segment["source_sha256"] if p.endswith("TRAIN_PLAN.json")))
    train = read(train_path)
    entries = train["entries"][:256]
    receipts = {i: read(V1 / "cache" / f"entry_{i:03d}.json") for i in range(256)}
    frame_sets = {e["step"]: {p["sequence"]: set().union(
        *(set(range(s, s + 40)) for s in p["starts"])) for p in e["pairs"]} for e in entries}
    eligible = {i for i in range(256) if segment["entries"][i]["coverage"]["segments"]
                and sum(w["selection"]["target_pixels"] for w in receipts[i]["keep_stats"])}
    candidates = []
    for a in sorted(eligible):
        for b in sorted(eligible):
            if a >= b or frame_sets[a].keys() != frame_sets[b].keys():
                continue
            if any(frame_sets[a][s] & frame_sets[b][s] for s in frame_sets[a]):
                continue
            rank = hashlib.sha256(f"46/effective-update/{a}/{b}".encode("utf-8")).hexdigest()
            candidates.append((rank, a, b, tuple(sorted(frame_sets[a]))))
    sources = sorted(train["train_sequences"])
    assert len(sources) == 8
    by_source = {s: [c for c in candidates if s in c[3]] for s in sources}
    counts, used, chosen = Counter(), set(), []

    def search():
        if len(chosen) == 16:
            return all(counts[s] == 4 for s in sources)
        options = {s: [c for c in by_source[s] if c[1] not in used and c[2] not in used
                       and all(counts[t] < 4 for t in c[3])] for s in sources if counts[s] < 4}
        if any(len(v) < 4 - counts[s] for s, v in options.items()):
            return False
        s = min(options, key=lambda s: (len(options[s]) / (4 - counts[s]), s))
        order = sorted(options[s], key=lambda c: (
            -int(bool(max(receipts[c[1]]["pairs_with_two_negatives"],
                          receipts[c[2]]["pairs_with_two_negatives"]))),
            sum(counts[t] for t in c[3]), c[0]))
        for c in order:
            chosen.append(c); used.update(c[1:3]); counts.update(c[3])
            if search():
                return True
            chosen.pop(); used.difference_update(c[1:3]); counts.subtract(c[3])
        return False

    assert search(), "Cannot meet the registered source-balance/disjointness contract"
    annotations = {s: read(Q / "stage1_goal/data" / s / "annotations.json") for s in sources}
    source_frames = {s: [int(Path(im["file_name"]).stem) for im in annotations[s]["images"]]
                     for s in sources}
    pairs = []
    for pair_id, (rank, a, b, names) in enumerate(chosen):
        # If only one side has old C pairs, that side performs the update.
        # Otherwise a fixed hash bit assigns roles, independent of scores.
        ca, cb = receipts[a]["pairs_with_two_negatives"], receipts[b]["pairs_with_two_negatives"]
        if bool(ca) != bool(cb):
            update, observer = (a, b) if ca else (b, a)
        else:
            update, observer = (a, b) if int(rank[-1], 16) % 2 == 0 else (b, a)
        source_overlap = {s: sorted({source_frames[s][i] for i in frame_sets[update][s]}
                                   & {source_frames[s][i] for i in frame_sets[observer][s]}) for s in names}
        assert not any(source_overlap.values())
        pairs.append({"pair_id": pair_id, "update_index": update, "observer_index": observer,
            "update_sources": list(names), "observer_sources": list(names),
            "selection_hash": rank, "overlapping_source_frames": source_overlap,
            "update_C_pairs": receipts[update]["pairs_with_two_negatives"],
            "observer_C_pairs": receipts[observer]["pairs_with_two_negatives"],
            "structural_zero_track_update": not bool(receipts[update]["pairs_with_two_negatives"])})
    selected = {}
    for i in sorted(used):
        windows = []
        for kind, entry in [("primary", entries[i]), ("auxiliary", receipts[i]["aux_entry"])]:
            for pair in entry["pairs"]:
                for start in pair["starts"]:
                    s = pair["sequence"]
                    windows.append({"window": len(windows), "kind": kind, "sequence": s,
                        "start": start, "crop_yx": pair["crop_yx"],
                        "source_frames": source_frames[s][start:start + 40]})
        r2 = read(V2 / "cache" / f"entry_{i:03d}.json")
        selected[str(i)] = {"index": i, "windows": windows,
            "v2_segments": r2["effective_coverage"]["segments"],
            "v2_target_observations": r2["target_observations"],
            "v2_background_regions": r2["background_regions"],
            "old_reliable_keep_target_pixels": sum(w["selection"]["target_pixels"]
                                                     for w in receipts[i]["keep_stats"]),
            "C_pairs": receipts[i]["pairs_with_two_negatives"],
            **{k: receipts[i][k] for k in ("input_sha256", "label_sha256", "auxiliary_input_sha256")},
            "v1_cache_sha256": receipts[i]["sha256"], "v2_cache_sha256": r2["sha256"],
            "v1_receipt_sha256": sha(V1 / "cache" / f"entry_{i:03d}.json"),
            "v2_receipt_sha256": sha(V2 / "cache" / f"entry_{i:03d}.json")}
    sources_hash = {str(p): sha(p) for p in [train_path, V2 / "SEGMENT_PLAN.json", Path(__file__)]}
    for s in sources:
        p = Q / "stage1_goal/data" / s / "annotations.json"
        sources_hash[str(p)] = sha(p)
    result = {"schema_version": 1, "status": "FROZEN_BEFORE_NEW_TEACHER_OR_UPDATE_RESULTS",
        "seed": 46, "candidate_entry_range": [0, 255], "eligible_entries": len(eligible),
        "eligible_same_sources_disjoint_frame_pairs": len(candidates),
        "pairs": pairs, "entries": selected, "source_sha256": sources_hash,
        "update_source_pair_counts": dict(Counter(s for p in pairs for s in p["update_sources"])),
        "observer_source_pair_counts": dict(Counter(s for p in pairs for s in p["observer_sources"])),
        "update_entries_with_C_pairs": sum(not p["structural_zero_track_update"] for p in pairs),
        "structural_zero_track_pair_ids": [p["pair_id"] for p in pairs if p["structural_zero_track_update"]],
        "selection": "Same two sources per pair; all original 40-frame input unions disjoint within pair; 32 entry IDs unique; each source four pairs per role; annotation segments and old keep target pixels required; deterministic preference for old C-pair availability, then source balance and seed46 hash. No new teacher score or update outcome used.",
        "cohort_contract": {"target_pool": "unique target indices exposed by old v2 segment_indices; different window contexts retained",
            "weak": "release bbox peak probability < 0.5", "veryweak": "release bbox peak probability < 0.01",
            "reliable": "all annotation objects with surviving original v1 keep group0 pixels; no strict-small restriction; retain original reliable pixels",
            "hard_background": "unique original v2 local/global background ROI indices attached to exposed targets",
            "margin": "original bbox-mean logit minus max(valid local/global ROI-mean logits)",
            "position": "argmax inside each fixed reliable bbox, row-major tie; pixel-center xy; not complete detector localization"},
        "limitations": ["Observation entries were already in original training; this is transfer of one new step, not held-out generalization.",
            "Within each pair no original frame is shared, so no frame-crop observation is shared. Across different pairs frames/tracks can recur; 16 pairs are not independent datasets.",
            "Source identity and object identity can recur between temporal blocks; frame disjointness does not imply identity independence.",
            "Release weak/reliable cohort counts are unknown until the fixed teacher pass; no entry will be substituted after seeing those counts.",
            "Three structural-zero C updates remain in the fixed denominator; they cannot establish track-gradient interference.",
            "Weak peak response is not an actual matching failure definition. Reliable positions are fixed-bbox peaks, not complete component matching results.",
            "Original annotation visibility/completeness limitations and background exclusion limits remain unchanged."]}
    assert len(used) == 32 and all(v == 4 for v in result["update_source_pair_counts"].values())
    assert result["update_entries_with_C_pairs"] == 13
    PLAN.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def _numpy(value):
    import numpy as np
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _sigmoid(value):
    import numpy as np
    a = np.asarray(value, dtype=np.float64)
    return np.exp(-np.logaddexp(0., -a))


def _roi_values(logits, regions):
    import numpy as np
    means, peaks, xy = [], [], []
    for wi, ti, x0, y0, x1, y1 in regions:
        roi = logits[wi, ti, y0:y1, x0:x1]
        assert roi.size and np.isfinite(roi).all()
        y, x = np.unravel_index(int(roi.argmax()), roi.shape)
        means.append(float(roi.mean(dtype=np.float64)))
        peaks.append(float(roi[y, x])); xy.append([float(x0+x+.5), float(y0+y+.5)])
    return np.asarray(means), np.asarray(peaks), xy


@lru_cache(maxsize=4)
def _keep(index):
    import numpy as np
    path = V1 / "cache" / f"entry_{index:03d}.npz"
    with np.load(path, allow_pickle=False) as source:
        mask = source["keep_groups"] == 0
        return source["keep_indices"][mask].copy(), source["keep_probabilities"][mask].copy()


def prepare_entry(index, batch, state, teacher_logits, risk, cache):
    """Return frozen JSON cohort; ``cache`` is the old v2 risk cache.

    Weak strata are fixed from release peaks before intervention. Reliable
    object membership comes from original keep pixels, whose actual matcher
    provenance is already audited; the historical cohort is not reselected.
    """
    import numpy as np
    from data import window_annotations
    index = int(index)
    registered = read(PLAN)["entries"][str(index)]
    for key in ("input_sha256", "label_sha256", "auxiliary_input_sha256"):
        assert batch[key] == registered[key], key
    assert sha(V1 / "cache" / f"entry_{index:03d}.npz") == registered["v1_cache_sha256"]
    assert sha(V2 / "cache" / f"entry_{index:03d}.npz") == registered["v2_cache_sha256"]
    logits = _numpy(teacher_logits)
    assert logits.shape == (8, 40, 128, 128) and np.isfinite(logits).all()
    regions = _numpy(cache["target_regions"]).astype(np.int64)
    segments = _numpy(cache["segment_indices"])
    exposed = np.unique(segments[segments >= 0]).astype(np.int64)
    annotation = segment_plan()["entries"][index]
    assert np.array_equal(regions, np.asarray(annotation["target_regions"]))
    means, peaks, peak_xy = _roi_values(logits, regions[exposed])
    peak_probability = _sigmoid(peaks)
    qp = _numpy(risk["q_positive"])
    assert np.allclose(qp[exposed], means, rtol=2e-5, atol=2e-5)
    weak = exposed[peak_probability < .5]
    veryweak = exposed[peak_probability < .01]
    negatives = _numpy(cache["negative_indices"])[exposed]
    backgrounds = np.unique(negatives[negatives >= 0]).astype(np.int64)
    bg_regions = _numpy(cache["background_regions"])[backgrounds].astype(np.int64)
    assert len(backgrounds) and (negatives >= 0).any(axis=1).all()
    keep_indices, keep_probabilities = _keep(index)
    keep_mask = np.zeros(logits.shape, bool)
    keep_mask[tuple(keep_indices.T)] = True
    frame_annotations = window_annotations(batch, state)
    reliable_regions, reliable_objects, old_ids, unknown_pixels = [], [], set(), 0
    for wi, frames in enumerate(frame_annotations):
        accounted = np.zeros_like(keep_mask[wi])
        window = registered["windows"][wi]
        for ti, anns in enumerate(frames):
            for ann in anns:
                x, y, w, h = map(float, ann["bbox"])
                if not all(math.isfinite(v) for v in [x,y,w,h]) or w <= 0 or h <= 0:
                    continue
                x0,y0,x1,y1 = [math.ceil(v-.5) for v in (x,y,x+w,y+h)]
                if min(x0,y0) < 0 or max(x1,y1) > 128:
                    continue
                pixels = keep_mask[wi, ti, y0:y1, x0:x1]
                if not pixels.any():
                    continue
                assert not ann.get("ignore",0) and not ann.get("iscrowd",0)
                assert x >= 9 and y >= 9 and x+w <= 119 and y+h <= 119
                # Original reliable boxes cannot overlap any other GT bbox.
                for other in anns:
                    if other["id"] == ann["id"]:
                        continue
                    u,v,a,b = map(float, other["bbox"])
                    assert max(x,u) >= min(x+w,u+a) or max(y,v) >= min(y+h,v+b)
                region = [wi,ti,x0,y0,x1,y1]
                reliable_regions.append(region)
                reliable_objects.append({"window":wi,"slot":ti,"sequence":window["sequence"],
                    "frame":window["start"]+ti,"source_frame":window["source_frames"][ti],
                    "annotation_id":int(ann["id"]),"track_id":int(ann["track_id"]),
                    "bbox_local":ann["bbox"],"rect":region[2:],"surviving_keep_pixels":int(pixels.sum())})
                old_ids.add((wi,ti,int(ann["id"])))
                accounted[ti,y0:y1,x0:x1] |= pixels
        unknown_pixels += int((keep_mask[wi] & ~accounted).sum())
    assert unknown_pixels == 0, "Original keep target pixels lack an eligible annotation mapping"
    reliable_mean, reliable_peak, reliable_xy = _roi_values(logits, reliable_regions)
    targets = []
    for j, target_index in enumerate(exposed):
        d = copy.deepcopy(annotation["descriptors"][int(target_index)])
        d.update(target_index=int(target_index), region=regions[target_index].tolist(),
            release_mean_logit=float(means[j]), release_peak_logit=float(peaks[j]),
            release_peak_probability=float(peak_probability[j]), release_peak_xy=peak_xy[j])
        targets.append(d)
    result = {"schema_version":1,"index":index,"pair_plan_sha256":sha(PLAN),
        "entry_input_sha256":batch["input_sha256"],
        "exposed_target_indices":exposed.tolist(),"weak_target_indices":weak.tolist(),
        "veryweak_target_indices":veryweak.tolist(),"target_objects":targets,
        "background_indices":backgrounds.tolist(),"background_regions":bg_regions.tolist(),
        "reliable_regions":reliable_regions,"reliable_objects":reliable_objects,
        "reliable_release_peak_xy":reliable_xy,
        "reliable_release_mean_logits":reliable_mean.tolist(),
        "reliable_release_peak_probabilities":_sigmoid(reliable_peak).tolist(),
        "keep_target_reference":{"path":str(V1/"cache"/f"entry_{index:03d}.npz"),
            "sha256":registered["v1_cache_sha256"],"pixels":len(keep_indices)},
        "counts":{"exposed_targets":len(exposed),"weak":len(weak),"veryweak":len(veryweak),
            "reliable":len(reliable_regions),"hardbg":len(backgrounds),"keep_target_pixels":len(keep_indices)},
        "original_keep_mapping":{"old_keep_objects":len(old_ids),
            "unmapped_original_keep_pixels":unknown_pixels,
            "selection_uses_original_keep_membership":True,
            "actual_matcher_not_rerun":"Reuse audited original keep group0; map its surviving pixels to full-core, nonoverlap annotation boxes."},
        "aggregation":"One vote per unique exposed target index / fixed reliable object / unique hard-background ROI. Overlapping segment repeats deduplicated for diagnostics; different input windows remain separate contextual observations. Training losses and their multiplicity are unchanged.",
        "definitions":{"weak_logit":"mean original bbox-mean logits on release-weak exposed targets",
            "weak_margin":"mean(q_positive - q_negative), q_negative=max valid local/global ROI means",
            "weak_probability_margin":"mean(sigmoid(q_positive)-sigmoid(q_negative))",
            "hardbg_logit":"mean ROI-mean logit over unique original hard-background ROIs attached to all exposed targets",
            "reliable_logit":"mean bbox-mean logit over original keep reliable annotation objects",
            "reliable_score":"mean within-bbox peak probability over those same fixed objects",
            "reliable_peak_xy":"fixed-bbox peak pixel centers; row-major tie; not full detector component localization",
            "keep_target_mse":"mean squared probability error on all original keep group0 pixels (pooled pixels; diagnostic only)",
            "weak_fixed_release_peak_logit":"mean logit at fixed release-selected bbox peak coordinates; extra diagnostic"}}
    json.dumps(result,ensure_ascii=False,allow_nan=False)
    return result


def measure(logits, risk, cohort):
    """Return fixed-cohort metrics, with null for any empty diagnostic group.

    The runner computes reliable_peak_shift relative to F2304 using the
    returned ordered reliable_peak_xy. This function never changes membership.
    """
    import numpy as np
    z = _numpy(logits)
    assert z.shape == (8,40,128,128) and np.isfinite(z).all()
    qp, qn = _numpy(risk["q_positive"]), _numpy(risk["q_negative"])
    weak = np.asarray(cohort["weak_target_indices"],np.int64)
    veryweak = np.asarray(cohort["veryweak_target_indices"],np.int64)
    bg_mean,bg_peak,_ = _roi_values(z,cohort["background_regions"])
    reliable_mean,reliable_peak,reliable_xy = _roi_values(z,cohort["reliable_regions"])
    def mean(a):
        return float(np.mean(a,dtype=np.float64)) if len(a) else None
    result = {"weak_logit":mean(qp[weak]),"weak_margin":mean(qp[weak]-qn[weak]),
        "weak_probability_margin":mean(_sigmoid(qp[weak])-_sigmoid(qn[weak])),
        "weak_hardbg_logit":mean(qn[weak]),"veryweak_logit":mean(qp[veryweak]),
        "veryweak_margin":mean(qp[veryweak]-qn[veryweak]),
        "veryweak_probability_margin":mean(_sigmoid(qp[veryweak])-_sigmoid(qn[veryweak])),
        "hardbg_logit":mean(bg_mean),"hardbg_score":mean(_sigmoid(bg_mean)),
        "hardbg_peak_score":mean(_sigmoid(bg_peak)),
        "reliable_logit":mean(reliable_mean),"reliable_score":mean(_sigmoid(reliable_peak)),
        "reliable_peak_xy":reliable_xy,"counts":dict(cohort["counts"])}
    weak_set=set(weak.tolist()); fixed=[]; current_peaks=[]
    for obj in cohort["target_objects"]:
        if obj["target_index"] not in weak_set:
            continue
        wi,ti,x0,y0,x1,y1=obj["region"]
        x,y=obj["release_peak_xy"]
        fixed.append(float(z[wi,ti,int(y-.5),int(x-.5)]))
        current_peaks.append(float(z[wi,ti,y0:y1,x0:x1].max()))
    result.update(weak_fixed_release_peak_logit=mean(fixed),weak_peak_logit=mean(current_peaks),
                  weak_peak_score=mean(_sigmoid(current_peaks)))
    indices,teacher=_keep(int(cohort["index"]))
    errors=(_sigmoid(z[tuple(indices.T)])-teacher.astype(np.float64))**2
    result["keep_target_mse"]=mean(errors)
    result["keep_target_component_original_reduction"]=sum(
        float(errors[indices[:,0]==wi].mean()) if (indices[:,0]==wi).any() else 0. for wi in range(8))/16.
    json.dumps(result,ensure_ascii=False,allow_nan=False)
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-plan",action="store_true")
    args=parser.parse_args()
    if args.build_plan:
        plan=build_plan()
        print(json.dumps({"status":plan["status"],"pairs":len(plan["pairs"]),
            "unique_entries":len(plan["entries"]),"track_nonempty_updates":plan["update_entries_with_C_pairs"],
            "zero_track_pair_ids":plan["structural_zero_track_pair_ids"],"sha256":sha(PLAN)},ensure_ascii=False))
