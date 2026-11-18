"""Prepare annotation-only plans, then release-teacher segment backgrounds.

Run --audit-only on CPU first. GPU work is launched only through the root
launcher with --shard 0/1/2 --shards 3. Every write stays in track_contrast_v2.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
E = Path(__file__).resolve().parents[1]
ROOT = E.parents[1]
V1 = ROOT / "experiments/track_contrast_v1"
sys.path.insert(0, str(E / "train_modules"))
from segments import CONFIG, REFERENCE, SegmentSampler, build_segments, old

PLAN = E / "SEGMENT_PLAN.json"
AUDIT = E / "SEGMENT_AUDIT.json"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                                   allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_plan():
    if PLAN.exists() or AUDIT.exists():
        raise FileExistsError("Segment plan/audit already exists; preserve the registered artifact")
    train = read(old.STAGE / "pilot/TRAIN_PLAN.json")
    sampler = SegmentSampler()
    totals, sources, histogram, comparisons = Counter(), Counter(), Counter(), Counter()
    identities = set()
    entries = []
    for entry in train["entries"][:256]:
        receipt = read(V1 / "cache" / f"entry_{entry['step']:03d}.json")
        result = sampler.sample(entry, receipt["aux_entry"])
        result["v1_cache_receipt_sha256"] = old.file_hash(V1 / "cache" / f"entry_{entry['step']:03d}.json")
        entries.append(result)
        totals["entries"] += 1
        totals["entries_with_segments"] += bool(result["segment_indices"])
        totals["eligible_observations"] += len(result["descriptors"])
        for key in ["segments", "covered_observations", "tail_segments", "tail_extra_observations", "segment_frame_exposures"]:
            totals[key] += result["coverage"][key]
        histogram.update(result["coverage"]["run_length_histogram"])
        for segment in result["segment_indices"]:
            descriptor = result["descriptors"][segment[0]]
            sources[descriptor["sequence"]] += 1
            identities.add((descriptor["sequence"], descriptor["video_id"], descriptor["track_id"]))
        for length in [4, 8, 12, 16]:
            value = build_segments(result["descriptors"], length=length, stride=length//2)["coverage"]
            for key in ["segments", "covered_observations", "tail_segments", "tail_extra_observations"]:
                comparisons[f"L{length}_{key}"] += value[key]
    source_paths = [REFERENCE, E / "train_modules/segments.py", Path(__file__),
                    V1 / "train_modules/track_pair_sampler.py", V1 / "SAMPLER_AUDIT.json",
                    old.STAGE / "pilot/TRAIN_PLAN.json"]
    source_paths += [sequence.path for sequence in sampler.sequences.values()]
    provenance = {str(path): old.file_hash(path) for path in source_paths}
    save(PLAN, {"schema_version": 1, "config": CONFIG, "source_sha256": provenance, "entries": entries})
    audit = {"status": "PASS_ANNOTATION_PLAN_BACKGROUND_PENDING", "config": CONFIG,
        "source_sha256": provenance, "plan_sha256": old.file_hash(PLAN),
        "totals": dict(totals), "selected_identities": len(identities),
        "selected_sources": len(sources), "segments_by_source": dict(sources),
        "run_length_histogram": dict(sorted(histogram.items())), "training_only_length_comparison": dict(comparisons),
        "source_audits": [sequence.audit for sequence in sampler.sequences.values()],
        "limitations": ["Source visibility/occlusion state absent: continuous legal annotation proxy only.",
            "Repeated overlapping window/segment observations are exposures, not independent targets.",
            "Background eligibility remains pending until release-teacher cache preparation.",
            "All8 windows contribute; no restriction to the old common B/C pairing subset.",
            "No development or confirmation images, annotations or predictions accessed."]}
    save(AUDIT, audit)
    lines = ["# C-v2 连续标注段覆盖审计", "", "状态：标注计划通过；背景缓存待检查。", "",
        "直接调用用户参考 visible_segments；连续 run 尾端补一个末段对齐窗口。固定长度8、步幅4，只用完整段。", "",
        f"前256批共 {totals['eligible_observations']} 个合法目标观察曝光；{totals['entries_with_segments']} 批有段，共 {totals['segments']} 段，覆盖 {totals['covered_observations']} 观察。",
        f"覆盖 {len(identities)} 个视频内身份、{len(sources)} 个训练来源；尾对齐新增 {totals['tail_segments']} 段，补覆盖 {totals['tail_extra_observations']} 个观察。", "",
        "全部8个原输入窗口均参与候选，沿用最大边≤16、完整核心框、无重叠及34个疑义身份排除；不要求能构成旧B/C共同配对。", "",
        "标注没有可见性/遮挡字段，这里是连续合法标注代理，不能宣称已验证连续可见。缺标、非法框、身份疑义、输入/来源帧号断点均打断段，不插值；无合法背景也将重新断段。", "",
        "背景来自发布teacher，全ROI排除所有已标目标及忽略/不完整区域；local缺失允许合法global一路，不放宽背景掩码。", "",
        "| 来源 | 段曝光 |", "|---|---:|", *[f"| {key} | {value} |" for key,value in sorted(sources.items())], "",
        "这些是重复训练曝光，不是独立样本数；段长对照仅基于训练标注，未查看开发或确认数据。", ""]
    AUDIT.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": audit["status"], "totals": totals,
                      "sources": len(sources), "identities": len(identities)}, ensure_ascii=False), flush=True)


def prepare(shard, shards):
    import numpy as np
    import torch
    from runtime import CP, MODEL_SOURCE, setup, load_model, sha
    from data import inputs, batch_data
    sys.path.insert(0, str(V1 / "train_modules"))
    from region_contrast_loss import mine_background_regions

    assert 0 <= shard < shards and shards >= 1
    plan = read(PLAN)
    assert plan["config"] == CONFIG and len(plan["entries"]) == 256
    for path, expected in plan["source_sha256"].items():
        assert old.file_hash(path) == expected, path
    sources = {str(path): sha(path) for path in [PLAN, CP, MODEL_SOURCE, Path(__file__),
        E / "train_modules/segments.py", E / "code/data.py", E / "code/runtime.py",
        V1 / "train_modules/region_contrast_loss.py"]}
    setup(); state = inputs(); model, _ = load_model(checkpoint=CP)
    cache_dir = E / "cache"; cache_dir.mkdir(parents=True, exist_ok=True)
    begun = time.perf_counter(); indices = []
    for index in range(shard, 256, shards):
        path = cache_dir / f"entry_{index:03d}.npz"; receipt_path = path.with_suffix(".json")
        if path.exists():
            receipt = read(receipt_path)
            assert receipt["sources"] == sources and receipt["sha256"] == sha(path)
            indices.append(index)
            continue
        entry = plan["entries"][index]
        old_receipt_path = V1 / "cache" / f"entry_{index:03d}.json"
        assert sha(old_receipt_path) == entry["v1_cache_receipt_sha256"]
        old_receipt = read(old_receipt_path)
        batch = batch_data(index, state)
        for key in ["input_sha256", "label_sha256", "auxiliary_input_sha256"]:
            assert batch[key] == old_receipt[key]
        with torch.inference_mode():
            probabilities = torch.cat([model(torch.from_numpy(batch["x"][i:i+2]).cuda())[1].sigmoid().cpu()
                                       for i in range(0, 8, 2)]).numpy()
        regions = np.asarray(entry["target_regions"], np.int64).reshape(-1, 6)
        globals_ = np.where(regions[:, 0] < 4, regions[:, 0] + 4, regions[:, 0])
        backgrounds, negatives = mine_background_regions(probabilities, batch["background"], regions,
                                                         globals_, local_radius=32)
        valid = (negatives >= 0).any(1)
        segment_plan = build_segments(entry["descriptors"], valid,
            length=CONFIG["length"], stride=CONFIG["stride"], tail_alignment=CONFIG["tail_alignment"])
        segment_indices = np.asarray(segment_plan["segment_indices"], np.int64).reshape(-1, CONFIG["length"])
        segment_windows = np.asarray(segment_plan["segment_windows"], np.int64)
        for wi, ti, x0, y0, x1, y1 in backgrounds:
            assert batch["background"][wi, ti, y0:y1, x0:x1].all()
        assert not len(segment_indices) or valid[segment_indices].all()
        meta = {"index": index, "sources": sources, "config": CONFIG,
            "v1_cache_receipt_sha256": sha(old_receipt_path), "annotation_plan_sha256": entry["plan_sha256"],
            **{key: batch[key] for key in ["input_sha256", "label_sha256", "auxiliary_input_sha256"]},
            "target_observations": len(regions), "background_regions": len(backgrounds),
            "observations_with_background": int(valid.sum()),
            "observations_with_both_backgrounds": int((negatives >= 0).all(1).sum()),
            "observations_local_only": int(((negatives[:, 0] >= 0) & (negatives[:, 1] < 0)).sum()),
            "observations_global_only": int(((negatives[:, 0] < 0) & (negatives[:, 1] >= 0)).sum()),
            "observations_without_background": int((~valid).sum()),
            "annotation_coverage": entry["coverage"], "effective_coverage": segment_plan["coverage"],
            "background_teacher": "original_release_checkpoint", "visibility_basis": CONFIG["visibility_rule"]}
        temporary = path.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, target_regions=regions, background_regions=backgrounds,
                negative_indices=negatives, segment_indices=segment_indices, segment_windows=segment_windows,
                metadata_json=np.array(json.dumps(meta, ensure_ascii=False)))
        temporary.replace(path)
        meta["sha256"] = sha(path); save(receipt_path, meta); indices.append(index)
        if len(indices) == 1 or len(indices) % 8 == 0:
            print(json.dumps({"shard": shard, "entry": index, "done": len(indices),
                "targets": len(regions), "segments": len(segment_indices),
                "without_background": int((~valid).sum()), "elapsed": time.perf_counter()-begun}), flush=True)
    for path, expected in sources.items():
        assert sha(path) == expected, path
    save(cache_dir / f"SHARD_{shard}.json", {"status": "COMPLETE", "indices": indices,
        "sources": sources, "elapsed_seconds": time.perf_counter()-begun})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--shard", type=int)
    parser.add_argument("--shards", type=int, default=3)
    args = parser.parse_args()
    if args.audit_only:
        build_plan()
    else:
        if args.shard is None:
            parser.error("GPU cache preparation requires an explicit --shard")
        prepare(args.shard, args.shards)


if __name__ == "__main__":
    main()
