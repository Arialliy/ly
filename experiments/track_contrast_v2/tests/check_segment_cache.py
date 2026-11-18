"""CPU-only independent geometry, source, coverage and continuity audit."""
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
E = Path(__file__).resolve().parents[1]
ROOT = E.parents[1]
V1 = ROOT / "experiments/track_contrast_v1"
STAGE = ROOT / "archive/infrared_problem_screen_20261009/temporal_diagnosis_20261010/stage1_goal"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            result.update(block)
    return result.hexdigest()


def independent_segments(descriptors, valid):
    groups = defaultdict(list)
    for row, d in enumerate(descriptors):
        groups[d["window"], d["sequence"], d["video_id"], d["track_id"]].append(row)
    expected, windows = [], []
    for key in sorted(groups):
        runs, current = [], []
        for row in sorted(groups[key], key=lambda i: descriptors[i]["frame"]):
            d = descriptors[row]
            if not valid[row]:
                if current: runs.append(current)
                current = []
                continue
            if current:
                previous = descriptors[current[-1]]
                if any(d[f] != previous[f]+1 for f in ("slot", "frame", "source_frame")):
                    runs.append(current); current = []
            current.append(row)
        if current: runs.append(current)
        for run in runs:
            if len(run) < 8: continue
            starts = list(range(0, len(run)-7, 4))
            if starts[-1] != len(run)-8: starts.append(len(run)-8)
            for start in starts:
                expected.append(run[start:start+8]); windows.append(key[0])
    return np.asarray(expected, np.int64).reshape(-1, 8), np.asarray(windows, np.int64)


def main():
    required = [E/"cache"/f"entry_{i:03d}{suffix}" for i in range(256) for suffix in (".npz", ".json")]
    assert all(p.is_file() for p in required), "All256 complete cache entries required"
    spec = importlib.util.spec_from_file_location("independent_v1_geometry", V1/"tests/check_cache.py")
    geometry = importlib.util.module_from_spec(spec); spec.loader.exec_module(geometry)
    plan = read(E/"SEGMENT_PLAN.json"); oldplan = read(STAGE/"pilot/TRAIN_PLAN.json")
    audits = read(V1/"SAMPLER_AUDIT.json")
    bad = {a["sequence"]: set(a["excluded_uncertain_track_ids"]) for a in audits["sequence_audits"]}
    annotations = geometry.load_annotations(oldplan["train_sequences"])
    original_by_id = {name: {a["id"]: a for frame in frames for a in frame} for name,frames in annotations.items()}
    totals, sources, source_observations = Counter(), Counter(), Counter()
    identities = set(); rows = []; evidence = {}; source_hashes = {}
    for name, value in plan["source_sha256"].items():
        assert sha(name) == value
    assert len(plan["entries"]) == 256 and plan["config"]["length"] == 8 and plan["config"]["stride"] == 4
    for index, entry in enumerate(plan["entries"]):
        assert entry["index"] == index
        path = E/"cache"/f"entry_{index:03d}.npz"; receipt_path = path.with_suffix(".json")
        receipt = read(receipt_path); oldreceipt_path = V1/"cache"/f"entry_{index:03d}.json"
        oldreceipt = read(oldreceipt_path)
        assert receipt["sha256"] == sha(path)
        evidence[str(path.relative_to(E))] = receipt["sha256"]
        evidence[str(receipt_path.relative_to(E))] = sha(receipt_path)
        assert receipt["v1_cache_receipt_sha256"] == entry["v1_cache_receipt_sha256"] == sha(oldreceipt_path)
        for key in ("input_sha256", "label_sha256", "auxiliary_input_sha256"):
            assert receipt[key] == oldreceipt[key]
        for source, expected in receipt["sources"].items():
            if source not in source_hashes: source_hashes[source] = sha(source)
            assert source_hashes[source] == expected
        with np.load(path, allow_pickle=False) as f: cache = {k: f[k] for k in f.files}
        assert json.loads(str(cache["metadata_json"])) == {k:v for k,v in receipt.items() if k != "sha256"}
        regions, backgrounds, neg = cache["target_regions"], cache["background_regions"], cache["negative_indices"]
        segs, windows = cache["segment_indices"], cache["segment_windows"]
        geometry.check_regions(regions); geometry.check_regions(backgrounds)
        np.testing.assert_array_equal(regions, np.asarray(entry["target_regions"], np.int64).reshape(-1,6))
        assert neg.dtype == np.int64 and neg.shape == (len(regions),2)
        assert (neg>=-1).all() and (neg<len(backgrounds)).all()
        _, bg, _, nonstrict = geometry.geometry(oldplan["entries"][index], oldreceipt["aux_entry"], annotations)
        for wi,ti,x0,y0,x1,y1 in backgrounds:
            assert bg[wi,ti,y0:y1,x0:x1].all(), (index,"background ROI not wholly legal")
        for row,(region,d) in enumerate(zip(regions,entry["descriptors"])):
            assert max(d["bbox"][2:])<=16 and d["track_id"] not in bad[d["sequence"]]
            original=original_by_id[d["sequence"]][d["annotation_id"]]
            assert original["bbox"]==d["bbox"] and original["track_id"]==d["track_id"]
            assert region.tolist()==[d["window"],d["slot"],*d["rect"]]
            for branch in range(2):
                if neg[row,branch]<0: continue
                roi=backgrounds[neg[row,branch]]
                expected_window=region[0] if branch==0 or region[0]>=4 else region[0]+4
                assert roi[0]==expected_window and roi[1]==region[1]
                assert np.array_equal(roi[4:6]-roi[2:4],region[4:6]-region[2:4])
                if branch==0: assert int(((roi[2:4]-region[2:4])**2).sum())<=32**2
        valid=(neg>=0).any(1)
        expected,expected_windows=independent_segments(entry["descriptors"],valid)
        np.testing.assert_array_equal(segs,expected);np.testing.assert_array_equal(windows,expected_windows)
        assert not segs.size or valid[segs].all()
        assert segs.dtype==np.int64 and windows.dtype==np.int64
        covered=set(segs.ravel().tolist())
        for segment in segs:
            d=entry["descriptors"][int(segment[0])]
            sources[d["sequence"]]+=1;identities.add((d["sequence"],d["video_id"],d["track_id"]))
        for i in covered: source_observations[entry["descriptors"][i]["sequence"]]+=1
        counts={"entry":index,"targets":len(regions),"background_regions":len(backgrounds),
            "observations_with_background":int(valid.sum()),"observations_without_background":int((~valid).sum()),
            "local_only":int(((neg[:,0]>=0)&(neg[:,1]<0)).sum()),
            "global_only":int(((neg[:,0]<0)&(neg[:,1]>=0)).sum()),
            "both_backgrounds":int((neg>=0).all(1).sum()),"segments":len(segs),
            "covered_observations":len(covered),"segment_frame_exposures":int(segs.size),
            "non_strict_annotation_window_instances_excluded":nonstrict}
        assert counts["targets"]==receipt["target_observations"]
        assert counts["segments"]==receipt["effective_coverage"]["segments"]
        assert counts["covered_observations"]==receipt["effective_coverage"]["covered_observations"]
        totals.update({k:v for k,v in counts.items() if k!="entry"});totals["entries"]+=1
        totals["entries_with_segments"]+=bool(len(segs));rows.append(counts)
        if index%32==0: print(json.dumps(counts),flush=True)
    for source,expected in source_hashes.items(): assert sha(source)==expected
    report={"status":"PASS","scope":"All256 new segment caches; independent annotation geometry and independent segment reconstruction; CPU only.",
        "totals":dict(totals),"segment_sources":dict(sources),"covered_observations_by_source":dict(source_observations),
        "identities":len(identities),"source_count":len(sources),"entry_audit":rows,
        "plan_sha256":sha(E/"SEGMENT_PLAN.json"),"audit_script_sha256":sha(Path(__file__)),
        "cache_sha256":evidence,"source_sha256":source_hashes,
        "checks":{"all_background_ROI_pixels_annotation_legal":True,"includes_nonstrict_target_exclusion":True,
            "no_both_missing_background_in_segments":True,"frame_sourceframe_slot_breaks_respected":True,
            "reference_length_stride_tail_rule_independently_reconstructed":True,"all8_window_scope":True,
            "original_C_exposure_hashes_equal":True,"no_development_or_confirmation_data_access":True},
        "limitations":["Visibility/occlusion truth is absent; this is a continuous legal annotation proxy.",
            "No separate annotation completeness regions are supplied; unannotated targets cannot be excluded with certainty.",
            "Repeated observations and overlapping segments are exposure counts,not independent samples.",
            "Release teacher probabilities are recomputed by preparation; this audit checks mask legality and provenance,not CPU re-ranking of floating-point hard negatives."]}
    out=E/"resources/SEGMENT_CACHE_AUDIT.json";out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    text=["# C-v2 段缓存独立审计","","状态：PASS。","",
        f"检查 {totals['entries']} 批、{totals['targets']} 个目标观察、{totals['background_regions']} 个背景ROI；无背景观察 {totals['observations_without_background']} 个。",
        f"有效连续段 {totals['segments']} 条，覆盖 {totals['covered_observations']} 个观察、{len(identities)} 个视频内身份及 {len(sources)} 个来源；重复段内曝光 {totals['segment_frame_exposures']} 个。","",
        "所有背景ROI逐像素满足独立重建的全标注排除掩码，包括非严格小目标；目标/背景窗口、帧、尺寸与局部距离检查通过。",
        "连续段按窗口、视频、身份及输入/来源帧号独立重建，缺背景/缺帧不桥接，长度8/步幅4/尾对齐与缓存完全一致。原C三项曝光哈希均保持。","",
        "没有可见性真值或额外标注完整性区域；结果只支持连续合法标注代理与现有标注覆盖内的背景排除。","",
        "本次只进行CPU文件和几何检查，未重跑模型，不读取开发/确认数据。",""]
    out.with_suffix(".md").write_text("\n".join(text),encoding="utf-8")
    print(json.dumps({"status":"PASS","totals":dict(totals),"sources":len(sources),"identities":len(identities)}),flush=True)


if __name__=="__main__":main()
