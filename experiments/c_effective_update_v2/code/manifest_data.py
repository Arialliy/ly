"""Select qualified existing training cohorts and preserve identity hierarchy.

Selection reads release-cohort membership, old exposure plans and annotations.
It never reads intervention metric values. The seven original common pairs
are retained, and at most two extra pairs can be made without reusing their
14 entries. Saved update endpoints make all new optimizer steps unnecessary.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import csv
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path

E = Path(__file__).resolve().parents[1]
ROOT = E.parents[1]
OLD = ROOT / "experiments/c_effective_update_v1"
C1 = ROOT / "experiments/track_contrast_v1"
C2 = ROOT / "experiments/track_contrast_v2"
Q = ROOT / "archive/infrared_problem_screen_20261009/temporal_diagnosis_20261010"
COMMON_IDS = (6, 8, 9, 10, 12, 13, 15)
PLAN = E / "PAIR_PLAN.json"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


@lru_cache(maxsize=1)
def load_plan():
    return read(PLAN)


@lru_cache(maxsize=32)
def cohort(index):
    record = load_plan()["entries"][str(int(index))]
    path = Path(record["cohort_path"])
    assert sha(path) == record["cohort_sha256"]
    return read(path)


def entry_windows(index):
    return load_plan()["entries"][str(int(index))]["windows"]


def _source_set(entry):
    return {w["sequence"] for w in entry["windows"]}


def _frames(entry):
    return {(w["sequence"], int(f)) for w in entry["windows"] for f in w["source_frames"]}


def _tracks(value):
    weak = set(value["weak_target_indices"])
    return {(x["sequence"], int(x["video_id"]), int(x["track_id"]))
            for x in value["target_objects"] if x["target_index"] in weak}


def _relation(a, b):
    sa, sb = _source_set(a), _source_set(b)
    return "same_source" if sa == sb else "cross_source" if not sa & sb else "mixed_source"


def build_manifest():
    """Enumerate a fixed existing-coverage pool; no new teacher screening."""
    if any((E / n).exists() for n in ("PAIR_PLAN.json", "PAIR_MANIFEST.csv", "ELIGIBILITY_AUDIT.json")):
        raise FileExistsError("A current manifest already exists; do not reselect")
    old = read(OLD / "PAIR_PLAN.json")
    cs = {int(i): read(OLD / "cohorts" / f"entry_{int(i):03d}.json") for i in old["entries"]}
    original = [copy.deepcopy(p) for p in old["pairs"] if p["pair_id"] in COMMON_IDS]
    assert tuple(p["pair_id"] for p in original) == COMMON_IDS
    common_entries = {p[k] for p in original for k in ("update_index", "observer_index")}
    qualified = {i for i, c in cs.items() if c["counts"]["weak"] > 0
                 and c["counts"]["hardbg"] > 0 and c["counts"]["reliable"] > 0}
    assert common_entries <= qualified
    available = sorted(qualified - common_entries)
    assert available == [25, 32, 63, 230, 235]
    old_update = {p["update_index"]: p for p in old["pairs"]}
    assert all(i in old_update for i in available)
    candidate_rows, candidates = [], []
    for a, b in itertools.combinations(available, 2):
        ea, eb = old["entries"][str(a)], old["entries"][str(b)]
        relation = _relation(ea, eb)
        overlap = sorted(_frames(ea) & _frames(eb))
        allowed = not overlap and relation in ("same_source", "cross_source")
        candidate_rows.append({"entries": [a, b], "source_relation": relation,
            "sources": [sorted(_source_set(ea)), sorted(_source_set(eb))],
            "weak_track_counts": [len(_tracks(cs[a])), len(_tracks(cs[b]))],
            "overlapping_source_frame_count": len(overlap), "qualifies_preferred_relation": allowed,
            "reason": "qualified" if allowed else "overlapping original frames" if overlap else
                "mixed sources retained as separate class; not needed when strict same/cross candidates exist"})
        if allowed:
            candidates.append((a, b))
    baseline_tracks = set().union(*(_tracks(cs[i]) for i in common_entries))
    alternatives = []
    max_pairs = min(9, len(available) // 2)
    for n in range(max_pairs, -1, -1):
        for combination in itertools.combinations(candidates, n):
            selected = [i for pair in combination for i in pair]
            if len(set(selected)) != len(selected):
                continue
            tracks = set().union(*(_tracks(cs[i]) for i in selected)) if selected else set()
            frequency = Counter(t for i in selected for t in _tracks(cs[i]))
            rank = hashlib.sha256(("46/effective-update-v2/" + repr(combination)).encode("utf-8")).hexdigest()
            score = (-len(tracks - baseline_tracks), -len({t[0] for t in tracks}),
                     max(frequency.values(), default=0), sum(v*v for v in frequency.values()), rank)
            alternatives.append((score, combination, tracks))
        if alternatives:
            break
    score, chosen, added_tracks = min(alternatives, key=lambda x: x[0])
    pairs = []
    for p in original:
        p.update(source_relation=_relation(old["entries"][str(p["update_index"])],
                                           old["entries"][str(p["observer_index"])]),
                 is_reused=True, old_pair_id=p["pair_id"], reuse_update_from_pair_id=p["pair_id"],
                 reuse_observer_from_pair_id=p["pair_id"], reuse_mode="original_pair",
                 selection_reason="All and only the seven original pairs with both release-weak cohorts nonempty; unchanged pairing.",
                 new_optimizer_steps=0)
        pairs.append(p)
    for offset, (a, b) in enumerate(chosen):
        rank = hashlib.sha256(f"46/effective-update-v2/role/{a}/{b}".encode("utf-8")).hexdigest()
        u, o = (a, b) if int(rank[-1], 16) % 2 == 0 else (b, a)
        eu, eo = old["entries"][str(u)], old["entries"][str(o)]
        pairs.append({"pair_id": 16+offset, "update_index": u, "observer_index": o,
            "update_sources": sorted(_source_set(eu)), "observer_sources": sorted(_source_set(eo)),
            "source_relation": _relation(eu, eo), "is_reused": False, "old_pair_id": None,
            "reuse_update_from_pair_id": old_update[u]["pair_id"], "reuse_observer_from_pair_id": None,
            "observer_cohort_was_v1_update_pair_id": old_update[o]["pair_id"],
            "reuse_mode": "new_pair_saved_update_endpoint", "new_optimizer_steps": 0,
            "update_C_pairs": eu["C_pairs"], "observer_C_pairs": eo["C_pairs"],
            "structural_zero_track_update": not bool(eu["C_pairs"]),
            "overlapping_source_frames": [], "role_selection_sha256": rank,
            "selection_reason": "Global metadata-only matching: maximize additional trusted weak tracks, then weak-source coverage; minimize track repetition; seed46 hash breaks ties and assigns roles. No common-entry or within-new-entry reuse. Saved update endpoint is reused; this observer response has not previously been evaluated."})
    used = {p[k] for p in pairs for k in ("update_index", "observer_index")}
    assert len(pairs) == 9 and len(used) == 18
    assert [(p["update_index"], p["observer_index"]) for p in pairs if not p["is_reused"]] == [(230, 25), (235, 32)]
    entries = {}
    for i in sorted(used):
        e = copy.deepcopy(old["entries"][str(i)])
        cp = OLD / "cohorts" / f"entry_{i:03d}.json"
        e.update(cohort_path=str(cp), cohort_sha256=sha(cp), cohort_counts=cs[i]["counts"],
                 weak_track_keys=[f"{s}|video={v}|track={t}" for s,v,t in sorted(_tracks(cs[i]))])
        entries[str(i)] = e
    identity_path = C1 / "SAMPLER_AUDIT.json"
    identities = {x["sequence"]: {k: x[k] for k in ("video_id", "excluded_uncertain_track_ids",
        "annotation_sha256", "visibility_claim")} for x in read(identity_path)["sequence_audits"]}
    for i in used:
        for s,v,t in _tracks(cs[i]):
            assert identities[s]["video_id"] == v and t not in identities[s]["excluded_uncertain_track_ids"]
    evidence = [OLD/"PAIR_PLAN.json", OLD/"PROTOCOL_FREEZE.json", identity_path,
                C1/"PAIR_PLAN.json", C2/"SEGMENT_PLAN.json", Path(__file__)]
    evidence += [OLD/"cohorts"/f"entry_{i:03d}.json" for i in sorted(cs)]
    prior_result_ids = sorted({p["reuse_update_from_pair_id"] for p in pairs})
    # Hash old results for exact endpoint provenance; no metric values are read.
    evidence += [OLD/"pairs"/f"pair_{p:02d}{suffix}" for p in prior_result_ids for suffix in (".json", ".npz")]
    result = {"schema_version": 1, "status": "SELECTED_BEFORE_READONLY_REPLAY", "seed": 46,
        "pairs": pairs, "entries": entries, "identity_audits": identities,
        "source_sha256": {str(path): sha(path) for path in evidence},
        "common_old_pair_ids": list(COMMON_IDS), "added_pair_ids": [16,17],
        "total_pairs": len(pairs), "new_optimizer_steps": 0,
        "source_relation_counts": dict(Counter(p["source_relation"] for p in pairs)),
        "eligibility_pool": {"old_entry_range": [0,255], "existing_exact_release_cohort_entries":32,
            "qualified_weak_and_background_entries":len(qualified), "existing_weak_empty_entries":32-len(qualified),
            "common_pair_entries":len(common_entries), "remaining_qualified_entries":available,
            "remaining_qualified_never_previously_updated_entries":[],
            "unavailable_exact_release_peak_eligibility_entries":[i for i in range(256) if i not in cs],
            "unavailable_exact_release_peak_eligibility_count":256-len(cs),
            "new_release_screening_performed":False},
        "selection_rule": "Use all old common7 unchanged. Enumerate same_source or cross_source, frame-disjoint pairs from the remaining existing qualified cohorts only, no entry reuse. Select maximum pair count (<=9); maximize new trusted weak tracks, then represented weak sources; minimize maximal and squared per-track entry repetition; SHA256(seed46, sorted pair tuple) breaks ties. Independent SHA256 assigns update/observer roles. Never inspect intervention metrics to select.",
        "metadata_contract": {"identity_key":"sequence|video=<video_id>|track=<track_id>",
            "identity_status_values":["trusted","uncertain_excluded_from_track_evidence"],
            "uncertain_identity_behavior":"Keep reliable observations and raw labels for source/object diagnostics; scoped_track_key=null, do not count as trusted trajectory evidence.",
            "weak_qualification":"Old release bbox peak probability <0.5 on unique exposed v2 target indices, original background reference valid.",
            "source_relation":"same=identical complete source sets; cross=disjoint sets; mixed=partial overlap, separate category",
            "original_C_links":"Exact (window,slot,annotation_id) join to old C pair descriptors; only pairs with both cached negatives valid are active.",
            "old_values_available":"Old scalar group means and per-reliable-object peak xy. No saved per-weak-object qplus/qminus/peak; no saved per-reliable-object scores.",
            "readout_reconstruction":"Saved parameter delta is applied to F2304, model hash checked, then read-only forward saves per-object values. Zero optimizer steps. New observation response cannot be inferred from old update batch means."}}
    save(PLAN, result)
    load_plan.cache_clear()
    candidate_report = {"status":"QUALIFIED_POOL_EXHAUSTED_WITH_TWO_ADDITIONAL_PAIRS",
        "uses_intervention_effects_for_selection":False,"uses_development":False,"uses_confirmation":False,
        "new_teacher_forwards":0,"new_optimizer_steps":0,"eligible_remaining_entries":available,
        "candidate_pairs":candidate_rows,"strict_same_cross_candidates":len(candidates),
        "maximum_disjoint_additional_pairs":max_pairs,"selected_pairs":[p for p in pairs if not p["is_reused"]],
        "unselected_qualified_entries":sorted(set(available)-set(i for ab in chosen for i in ab)),
        "old_common_weak_tracks":len(baseline_tracks),"all_selected_weak_tracks":len(baseline_tracks|added_tracks),
        "new_weak_tracks":[{"source":s,"video_id":v,"track_id":t} for s,v,t in sorted(added_tracks-baseline_tracks)],
        "old_common_weak_sources":sorted({t[0] for t in baseline_tracks}),
        "all_selected_weak_sources":sorted({t[0] for t in baseline_tracks|added_tracks}),
        "winning_metadata_score":{"negative_new_track_count":score[0],"negative_selected_weak_source_count":score[1],
            "maximum_track_entry_repetition":score[2],"sum_squared_track_entry_repetitions":score[3],"tie_hash":score[4]},
        "all_maximum_size_disjoint_matchings":[{"pairs":[list(ab) for ab in comb],
            "metadata_score":list(candidate_score), "new_weak_track_count":len(tracks-baseline_tracks),
            "selected_weak_sources":sorted({t[0] for t in tracks})}
            for candidate_score,comb,tracks in sorted(alternatives,key=lambda x:x[0])],
        "unknown_eligibility_entry_count":224,
        "existing_prediction_search":"Read-only filename inventory under experiments and archive/infrared_problem_screen_20261009 found no additional npz/npy prediction/teacher/runs assets for any of the eight training source names. Old C caches store selected keep pixels only, and v2 risk caches store ROI coordinates, so neither certifies every target bbox peak.",
        "limits":["Nine pairs are the available qualified paired workload, not a statistical sufficiency guarantee.",
            "The two added pairs create new cross-source observer measurements of old update endpoints, not new optimizer interventions.",
            "The 224 entries lacking exact teacher peak data remain qualification-unknown, not weak-negative; no further screening is performed to fill the quota.",
            "All extra selected entries are outside the original common14 and unique within the added set; repeated source frames and tracks across other entries remain possible.",
            "Weak targets have audited video-scoped identity; reliable keep objects can include identity-uncertain labels and are flagged separately.",
            "Cross-source and same-source effects must be summarized separately. Two cross-source pairs do not establish broad transfer.",
            "Old7 per-track numerical effects require read-only endpoint replay; do not manufacture them from aggregate means."],
        "pair_plan_sha256":sha(PLAN)}
    save(E/"ELIGIBILITY_AUDIT.json",candidate_report)
    fields=["pair_id","is_reused","reuse_mode","reuse_update_from_pair_id","update_index","observe_index",
            "update_source","observe_source","update_window","observe_window","weak_object_ids",
            "reliable_object_ids","background_reference_ids","weak_track_keys","source_relation",
            "eligibility_reason","update_weak_count","observe_weak_count","update_C_pairs","new_optimizer_steps"]
    with (E/"PAIR_MANIFEST.csv").open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for p in pairs:
            u,o=p["update_index"],p["observer_index"]
            def ids(index,kind):
                c=cs[index]
                if kind=="weak":return [f"entry={index}|target={t}" for t in c["weak_target_indices"]]
                if kind=="reliable":return [f"entry={index}|reliable={j}|ann={x['annotation_id']}" for j,x in enumerate(c["reliable_objects"])]
                return [f"entry={index}|background={b}" for b in c["background_indices"]]
            def cell(v):return json.dumps(v,ensure_ascii=False,separators=(",",":"))
            writer.writerow({"pair_id":p["pair_id"],"is_reused":p["is_reused"],"reuse_mode":p["reuse_mode"],
                "reuse_update_from_pair_id":p["reuse_update_from_pair_id"],"update_index":u,"observe_index":o,
                "update_source":cell(p["update_sources"]),"observe_source":cell(p["observer_sources"]),
                "update_window":cell(entries[str(u)]["windows"]),"observe_window":cell(entries[str(o)]["windows"]),
                "weak_object_ids":cell({"update":ids(u,"weak"),"observer":ids(o,"weak")}),
                "reliable_object_ids":cell({"update":ids(u,"reliable"),"observer":ids(o,"reliable")}),
                "background_reference_ids":cell({"update":ids(u,"background"),"observer":ids(o,"background")}),
                "weak_track_keys":cell({r:entries[str(i)]["weak_track_keys"] for r,i in [("update",u),("observer",o)]}),
                "source_relation":p["source_relation"],"eligibility_reason":p["selection_reason"],
                "update_weak_count":cs[u]["counts"]["weak"],"observe_weak_count":cs[o]["counts"]["weak"],
                "update_C_pairs":p["update_C_pairs"],"new_optimizer_steps":0})
    return result


@lru_cache(maxsize=1)
def _segment_plan():
    return read(C2/"SEGMENT_PLAN.json")


@lru_cache(maxsize=1)
def _contrast_plan():
    return read(C1/"PAIR_PLAN.json")


@lru_cache(maxsize=32)
def _links(index):
    import numpy as np
    segment=_segment_plan()["entries"][index]
    segment_links=defaultdict(list)
    for si,indices in enumerate(segment["segment_indices"]):
        for target in indices:
            if target>=0:segment_links[int(target)].append(si)
    target_lookup={(d["window"],d["slot"],d["annotation_id"]):i
                   for i,d in enumerate(segment["descriptors"])}
    old=_contrast_plan()["entries"][index]
    contrast_links=defaultdict(lambda:{"descriptor_indices":[],"active_anchor_pairs":[],"active_endpoint_pairs":[]})
    with np.load(C1/"cache"/f"entry_{index:03d}.npz",allow_pickle=False) as cache:
        active=(cache["negative_indices"]>=0).all(axis=1)
        assert np.array_equal(cache["positive_pairs_C"],np.asarray(old["positive_pairs_C"],dtype=np.int64).reshape(-1,2))
    for j,d in enumerate(old["descriptors"]):
        contrast_links[(d["window"],d["slot"],d["annotation_id"])]["descriptor_indices"].append(j)
    for pi,(a,b) in enumerate(old["positive_pairs_C"]):
        if not active[pi]:continue
        for role,j in [("active_anchor_pairs",a),("active_endpoint_pairs",b)]:
            d=old["descriptors"][j]
            contrast_links[(d["window"],d["slot"],d["annotation_id"])][role].append(pi)
    return target_lookup,segment_links,dict(contrast_links)


def _enrich(index, obj, kind, ordinal=None):
    d=copy.deepcopy(obj);s=d["sequence"];wi=int(d["window"]);ti=int(d["slot"])
    window=entry_windows(index)[wi];audit=load_plan()["identity_audits"][s]
    assert window["sequence"]==s and window["source_frames"][ti]==d["source_frame"]
    assert d.get("video_id",audit["video_id"])==audit["video_id"]
    video=int(audit["video_id"]);track=int(d["track_id"])
    trusted=track not in audit["excluded_uncertain_track_ids"]
    key=f"{s}|video={video}|track={track}"
    lookup,segments,contrast=_links(index)
    lookup_key=(wi,ti,d["annotation_id"]);target_index=lookup.get(lookup_key)
    d.update(source=s,video_id=video,raw_scoped_track_key=key,
        scoped_track_key=key if trusted else None,
        identity_audit_status="trusted" if trusted else "uncertain_excluded_from_track_evidence",
        object_key=f"entry={index}|{kind}={ordinal if ordinal is not None else d['target_index']}",
        source_object_frame_key=f"{s}|frame={d['source_frame']}|ann={d['annotation_id']}",
        window_context={"entry_index":index,"window":wi,"start":window["start"],"crop_yx":window["crop_yx"],
            "slot":ti,"source_frame":d["source_frame"]},
        physical_frame_context_key=f"{s}|frame={d['source_frame']}|crop={window['crop_yx'][0]},{window['crop_yx'][1]}|ann={d['annotation_id']}",
        v2_target_index=target_index,v2_segment_indices=segments.get(target_index,[]) if target_index is not None else [],
        original_C_links=contrast.get(lookup_key,{"descriptor_indices":[],"active_anchor_pairs":[],"active_endpoint_pairs":[]}))
    if kind=="target":
        assert trusted and target_index==d["target_index"]
    return d


@lru_cache(maxsize=32)
def weak_metadata(index):
    import numpy as np
    index=int(index);c=cohort(index);weak=set(c["weak_target_indices"])
    with np.load(C2/"cache"/f"entry_{index:03d}.npz",allow_pickle=False) as cache:
        negatives=cache["negative_indices"].copy()
    out={}
    for d in c["target_objects"]:
        ti=d["target_index"]
        if ti not in weak:continue
        value=_enrich(index,d,"target")
        value["negative_reference_ids"]=[f"entry={index}|background={int(b)}" for b in negatives[ti] if b>=0]
        value["negative_reference_slots"]=[{"kind":kind,"background_index":int(b),
            "reference_id":f"entry={index}|background={int(b)}" if b>=0 else None}
            for kind,b in zip(("local","global"),negatives[ti])]
        value["veryweak_release_peak"] = ti in set(c["veryweak_target_indices"])
        assert value["negative_reference_ids"]
        out[ti]=value
    assert len(out)==c["counts"]["weak"]
    return out


@lru_cache(maxsize=32)
def reliable_metadata(index):
    index=int(index)
    return [_enrich(index,d,"reliable",j) for j,d in enumerate(cohort(index)["reliable_objects"])]


def background_metadata(index, region, background_index):
    wi,ti,x0,y0,x1,y1=map(int,region);w=entry_windows(index)[wi];cy,cx=w["crop_yx"]
    absolute_roi=[x0+cx,y0+cy,x1+cx,y1+cy]
    physical_key=f"{w['sequence']}|frame={w['source_frames'][ti]}|roi="+",".join(map(str,absolute_roi))
    return {"reference_id":f"entry={index}|background={int(background_index)}",
        "background_index":int(background_index),"region":[wi,ti,x0,y0,x1,y1],
        "source":w["sequence"],"sequence":w["sequence"],"window":wi,"slot":ti,
        "frame":w["start"]+ti,"source_frame":w["source_frames"][ti],
        "window_context":{"entry_index":int(index),"window":wi,"start":w["start"],"crop_yx":w["crop_yx"]},
        "absolute_roi":absolute_roi,"track_id":None,"scoped_track_key":None,
        "source_background_frame_key":physical_key,"source_object_frame_key":physical_key,
        "background_rule":"original fixed v2 local/global candidate ROI; max candidate may change with logits, ROI set may not"}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build",action="store_true")
    args=parser.parse_args()
    if args.build:
        p=build_manifest()
        print(json.dumps({"pairs":len(p["pairs"]),"new_pair_ids":p["added_pair_ids"],
            "new_optimizer_steps":0,"plan_sha256":sha(PLAN)},ensure_ascii=False))
