"""Read-only verification of metadata qualification and object identity keys.

No intervention-result metrics or model predictions are loaded. This checks the
already selected manifest; it cannot change membership or launch optimization.
"""
from collections import Counter
import csv
import hashlib
import itertools
import json

import manifest_data as md


def main():
    plan = md.load_plan()
    old = md.read(md.OLD / "PAIR_PLAN.json")
    cohorts = {int(i): md.read(md.OLD / "cohorts" / f"entry_{int(i):03d}.json")
               for i in old["entries"]}
    eligible = {i for i, value in cohorts.items()
                if all(value["counts"][k] > 0 for k in ("weak", "reliable", "hardbg"))}
    reused = [p for p in plan["pairs"] if p["reuse_mode"] == "original_pair"]
    assert [p["pair_id"] for p in reused] == list(md.COMMON_IDS)
    original_pairs = {p["pair_id"]: p for p in old["pairs"]}
    common_entries = {p[k] for p in reused for k in ("update_index", "observer_index")}
    assert common_entries <= eligible
    for p in reused:
        original = original_pairs[p["pair_id"]]
        assert all(p[k] == original[k] for k in ("update_index", "observer_index"))
    available = sorted(eligible - common_entries)
    candidates = []
    for a, b in itertools.combinations(available, 2):
        ea, eb = old["entries"][str(a)], old["entries"][str(b)]
        if not (md._frames(ea) & md._frames(eb)) and md._relation(ea, eb) in ("same_source", "cross_source"):
            candidates.append((a, b))
    baseline_tracks = set().union(*(md._tracks(cohorts[i]) for i in common_entries))
    alternatives = []
    for count in range(len(available)//2, -1, -1):
        for matching in itertools.combinations(candidates, count):
            selected = [i for ab in matching for i in ab]
            if len(selected) != len(set(selected)):
                continue
            tracks = set().union(*(md._tracks(cohorts[i]) for i in selected)) if selected else set()
            frequency = Counter(t for i in selected for t in md._tracks(cohorts[i]))
            digest = hashlib.sha256(("46/effective-update-v2/" + repr(matching)).encode()).hexdigest()
            score = (-len(tracks-baseline_tracks), -len({t[0] for t in tracks}),
                     max(frequency.values(), default=0), sum(v*v for v in frequency.values()), digest)
            alternatives.append((score, matching))
        if alternatives:
            break
    winner = min(alternatives)[1]
    additions = [p for p in plan["pairs"] if p["reuse_mode"] != "original_pair"]
    assert tuple(tuple(sorted((p["update_index"], p["observer_index"]))) for p in additions) == winner
    ids = [p[k] for p in plan["pairs"] for k in ("update_index", "observer_index")]
    assert len(ids) == len(set(ids)) == 18
    for p in plan["pairs"]:
        u, o = p["update_index"], p["observer_index"]
        eu, eo = plan["entries"][str(u)], plan["entries"][str(o)]
        assert u in eligible and o in eligible and not (md._frames(eu) & md._frames(eo))
        assert md._relation(eu, eo) == p["source_relation"]
        assert original_pairs[p["reuse_update_from_pair_id"]]["update_index"] == u
        assert p["new_optimizer_steps"] == 0
    with (md.E / "PAIR_MANIFEST.csv").open(newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    assert [int(x["pair_id"]) for x in csv_rows] == [p["pair_id"] for p in plan["pairs"]]
    records = []
    examples = {}
    frame_to_track = {}
    background_identity = {}
    weak_tracks = set()
    for index in ids:
        c = md.cohort(index)
        weak = md.weak_metadata(index)
        reliable = md.reliable_metadata(index)
        backgrounds = [md.background_metadata(index, region, bi)
                       for bi, region in zip(c["background_indices"], c["background_regions"])]
        assert (len(weak), len(reliable), len(backgrounds)) == tuple(c["counts"][k] for k in ("weak", "reliable", "hardbg"))
        valid_backgrounds = {b["reference_id"] for b in backgrounds}
        for obj in weak.values():
            assert obj["identity_audit_status"] == "trusted" and obj["scoped_track_key"]
            assert all(k in valid_backgrounds for k in obj["negative_reference_ids"])
            weak_tracks.add(obj["scoped_track_key"])
        uncertain = 0
        for obj in list(weak.values()) + reliable:
            physical_key = obj["source_object_frame_key"]
            identity = (obj["source"], obj["source_frame"], obj["annotation_id"], obj["raw_scoped_track_key"])
            assert frame_to_track.setdefault(physical_key, identity) == identity
            if obj["identity_audit_status"] != "trusted":
                assert obj["scoped_track_key"] is None
                uncertain += 1
                examples.setdefault("uncertain_reliable", obj)
            else:
                assert obj["scoped_track_key"] == obj["raw_scoped_track_key"]
        for obj in backgrounds:
            key = obj["source_background_frame_key"]
            identity = (obj["source"], obj["source_frame"], tuple(obj["absolute_roi"]))
            assert background_identity.setdefault(key, identity) == identity
            assert obj["scoped_track_key"] is None and obj["track_id"] is None
        examples.setdefault("weak", next(iter(weak.values())))
        examples.setdefault("reliable", reliable[0])
        examples.setdefault("background", backgrounds[0])
        records.append({"entry_index": index, "weak": len(weak), "reliable": len(reliable),
                        "background": len(backgrounds), "uncertain_reliable_observations": uncertain})
    assert len(weak_tracks) == 39
    report = {"status": "PASS", "pair_count": len(plan["pairs"]), "distinct_entry_count": len(ids),
              "original_common_pairs_preserved": True, "metadata_ranking_reproduced": True,
              "selection_reads_intervention_effects": False, "new_teacher_forwards": 0,
              "new_optimizer_steps": 0, "trusted_weak_track_count": len(weak_tracks),
              "background_key_unique_physical_rois": len(background_identity),
              "identity_key_consistency": True, "records": records, "examples": examples,
              "plan_sha256": md.sha(md.PLAN), "manifest_sha256": md.sha(md.E / "PAIR_MANIFEST.csv"),
              "manifest_source_sha256": md.sha(md.E / "code/manifest_data.py"),
              "checker_sha256": md.sha(__file__),
              "review_scope": "Metadata-only execution verification; independent intervention-effect review is separate."}
    md.save(md.E / "evaluation/MANIFEST_CHECK.json", report)
    print(json.dumps({k: report[k] for k in ("status", "pair_count", "trusted_weak_track_count", "new_optimizer_steps")}))


if __name__ == "__main__":
    main()
