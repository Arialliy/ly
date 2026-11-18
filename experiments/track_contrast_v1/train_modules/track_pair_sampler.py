#!/usr/bin/env python3
"""Annotation-only common B/C target descriptor and positive-assignment plan.

All descriptor boxes refer to the four unchanged primary windows, in exactly
the order used by frozen train_pilot.build_batch. A B pair is obtained by
swapping the endpoints of two legal C pairs from different video-scoped IDs.
Thus B/C have identical anchor and endpoint multisets; only correspondence
changes. Sampling never reads predictions, image pixels, or development data.

The dataset does not expose per-frame visibility/occlusion states. We retain
that status as unknown and use only observed, identity-consistent, nonoverlap
boxes across every actual frame in a gap; this is not a visibility guarantee.

Public interface:
    sampler = TrackPairSampler(data_root, train_sequences, seed=46)
    plan = sampler.sample(entry)  # JSON-serializable; original entry unchanged
    plan['target_regions']: [N,6] (window,slot,x0,y0,x1,y1), half-open ints
    plan['positive_pairs_B'/'positive_pairs_C']: [P,2] descriptor indices
    plan['background_requests']: P identical anchor requests for both groups

Global-background window indices 4 + primary_window are a contract with the
cache preparer. This module chooses no global image crop and no hard negatives.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import hashlib
import heapq
import json
import math
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
PROJECT = BASE.parents[1]
STAGE = PROJECT / "archive/infrared_problem_screen_20261009/temporal_diagnosis_20261010/stage1_goal"
CONFIG = {
    "seed": 46, "window_length": 40, "patch": 128, "core_margin": 9,
    "gaps": [1, 4], "max_pairs_per_gap": 64, "max_pairs_per_entry": 128,
    "target_max_side": 16, "matching_size": "bbox_area",
    "size_rule": "anchor_and_C_endpoint_share_floor_log2_area_bin; swap_only_within_same_bin",
    "maximum_area_ratio": 2.0, "require_all_intermediate_frames_legal": True,
    "positive_annotation_rule": "finite_in_bounds_box;ignore=0;iscrowd=0;identity_consistent;no_overlap_with_any_annotation;full_crop_core",
    "visibility_rule": "field_absent_is_unknown_not_visible;do_not_interpolate_missing_observations",
    "B_assignment": "disjoint_two_cycles_of_C_endpoints;different_sequence_video_track_identity;same_category;same_gap_group",
    "selection": "sha256_seed_step_order_within_identity;largest_identity_pools_paired;hash_rank_complete_swap_units;cap64_per_gap",
    "background_rule": "two_shared_requests_per_anchor:local_primary_and_global_aux_window;negative_selection_owned_by_teacher_cache",
}


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def valid_box(annotation):
    box = annotation.get("bbox", [])
    return (len(box) == 4 and all(isinstance(x, (int, float)) and math.isfinite(x) for x in box)
            and box[2] > 0 and box[3] > 0)


def overlaps(a, b):
    x, y, w, h = a["bbox"]; u, v, q, r = b["bbox"]
    return max(x, u) < min(x + w, u + q) and max(y, v) < min(y + h, v + r)


def full_core(annotation, crop):
    cy, cx = crop; x, y, w, h = annotation["bbox"]
    margin, patch = CONFIG["core_margin"], CONFIG["patch"]
    return x >= cx + margin and y >= cy + margin and x + w <= cx + patch - margin and y + h <= cy + patch - margin


def local_rect(annotation, crop):
    cy, cx = crop; x, y, w, h = annotation["bbox"]
    return [math.ceil(x - cx - .5), math.ceil(y - cy - .5),
            math.ceil(x + w - cx - .5), math.ceil(y + h - cy - .5)]


def identity(descriptor):
    return descriptor["sequence"], descriptor["video_id"], descriptor["track_id"]


def descriptor_key(descriptor):
    return (descriptor["window"], descriptor["slot"], descriptor["annotation_id"])


class AnnotationSequence:
    def __init__(self, data_root, sequence):
        self.name = sequence
        self.path = Path(data_root) / sequence / "annotations.json"
        data = json.loads(self.path.read_text(encoding="utf-8"))
        images = data["images"]
        assert len(images) == 160
        self.source_frames = [int(Path(im["file_name"]).stem) for im in images]
        assert self.source_frames == list(range(self.source_frames[0], self.source_frames[0] + 160))
        assert all(im["width"] == 640 and im["height"] == 512 for im in images)
        self.video_ids = [int(im["video_id"]) for im in images]
        assert len(set(self.video_ids)) == 1
        index = {im["id"]: i for i, im in enumerate(images)}
        assert len(index) == 160
        self.frames = [[] for _ in images]
        seen = set()
        for annotation in data["annotations"]:
            assert annotation["id"] not in seen and annotation["image_id"] in index
            seen.add(annotation["id"])
            self.frames[index[annotation["image_id"]]].append(annotation)

        categories, local_ids, reverse_ids = defaultdict(set), defaultdict(set), defaultdict(set)
        bad_ids = set(); reasons = Counter(); state_fields = Counter()
        identity_counts = Counter()
        for annotations in self.frames:
            counts = Counter(a.get("track_id") for a in annotations)
            for track, count in counts.items():
                if count > 1:
                    bad_ids.add(track); identity_counts["duplicate_track_frame_observations"] += count
            for a in annotations:
                track = a.get("track_id"); local = a.get("tracking_id")
                if not isinstance(track, int) or local is None:
                    bad_ids.add(track); identity_counts["missing_or_invalid_identity_observations"] += 1
                categories[track].add(a.get("category_id"))
                # Local numbers recur across categories in these annotations;
                # retained track_id is the identity key. Reuse of local number
                # 0 in two categories alone is not an identity conflict.
                local_ids[track].add(str(local)); reverse_ids[(a.get("category_id"), str(local))].add(track)
                for key in ["visibility", "occlusion", "occluded", "status"]:
                    state_fields[key + ("_present" if key in a else "_absent")] += 1
        for track in categories:
            if len(categories[track]) != 1:
                bad_ids.add(track); identity_counts["category_changing_tracks"] += 1
            if len(local_ids[track]) != 1:
                bad_ids.add(track); identity_counts["local_id_changing_tracks"] += 1
        ambiguous_local_ids = []
        for (category, local), tracks in reverse_ids.items():
            if len(tracks) > 1:
                bad_ids.update(tracks); identity_counts["ambiguous_local_id_groups"] += 1
                ambiguous_local_ids.append({"category_id": category, "tracking_id": local,
                                            "track_ids": sorted(tracks)})

        self.eligible = []
        for annotations in self.frames:
            eligible = {}
            for a in sorted(annotations, key=lambda a: a["id"]):
                reason = None
                if a.get("track_id") in bad_ids: reason = "uncertain_identity"
                elif not valid_box(a): reason = "invalid_box"
                elif a.get("ignore", 0) or a.get("iscrowd", 0): reason = "ignore_or_crowd"
                else:
                    x, y, w, h = a["bbox"]
                    if x < 0 or y < 0 or x + w > 640 or y + h > 512: reason = "out_of_bounds"
                    elif max(w, h) > CONFIG["target_max_side"]: reason = "not_strict_small"
                    elif any(other["id"] != a["id"] and valid_box(other) and overlaps(a, other)
                             for other in annotations): reason = "overlap_any_annotation"
                    elif any(k in a for k in ["visibility", "occlusion", "occluded", "status"]):
                        # Current frozen data has no such fields. A future schema
                        # change must be reviewed, not silently interpreted here.
                        reason = "unreviewed_explicit_state_schema"
                if reason: reasons[reason] += 1
                else: eligible[a["track_id"]] = a
            self.eligible.append(eligible)
        gaps = 0
        for track in categories:
            frames = [i for i, annotations in enumerate(self.frames) if any(a.get("track_id") == track for a in annotations)]
            gaps += sum(b > a + 1 for a, b in zip(frames, frames[1:]))
        self.audit = {
            "sequence": sequence, "video_id": self.video_ids[0], "annotation_count": len(data["annotations"]),
            "track_count": len(categories), "eligible_strict_small_observations": sum(map(len, self.eligible)),
            "uncertain_tracks": len(bad_ids), "identity_findings": dict(identity_counts),
            "excluded_uncertain_track_ids": sorted(bad_ids),
            "ambiguous_category_local_ID_groups": ambiguous_local_ids,
            "track_observation_gaps": gaps, "excluded_observations": dict(reasons),
            "state_fields": dict(state_fields), "annotation_sha256": file_hash(self.path),
            "visibility_claim": "unknown; continuous annotated boxes do not prove absence of occlusion",
        }

    def descriptor(self, annotation, frame, window, start, crop):
        rect = local_rect(annotation, crop)
        assert 0 <= rect[0] < rect[2] <= 128 and 0 <= rect[1] < rect[3] <= 128
        return {
            "sequence": self.name, "video_id": self.video_ids[frame], "track_id": annotation["track_id"],
            "tracking_id": str(annotation["tracking_id"]), "category_id": annotation["category_id"],
            "annotation_id": annotation["id"], "frame": frame, "source_frame": self.source_frames[frame],
            "window": window, "window_start": start, "slot": frame - start,
            "rect": rect, "bbox": annotation["bbox"], "bbox_area": annotation["bbox"][2] * annotation["bbox"][3],
            "visibility_status": "not_available_in_source_annotations",
        }


class TrackPairSampler:
    def __init__(self, data_root=STAGE / "data", train_sequences=None, seed=46):
        selection = json.loads((STAGE / "DATA_SELECTION.json").read_text(encoding="utf-8"))
        allowed = selection["splits"]["train"]
        names = allowed if train_sequences is None else list(train_sequences)
        assert names == allowed, "Only the unchanged eight training sources are allowed"
        assert seed == CONFIG["seed"]
        self.seed = seed
        self.sequences = {name: AnnotationSequence(data_root, name) for name in names}

    def sample(self, entry):
        return build_pair_plan(entry, self.sequences, self.seed)


def build_pair_plan(entry, sequences, seed=46):
    """Return annotation-only common descriptor pool and B/C assignments."""
    assert seed == 46 and len(entry["pairs"]) == 2
    step = int(entry["step"])
    groups = defaultdict(lambda: defaultdict(list))
    coverage = Counter()
    for pair_index, pair in enumerate(entry["pairs"]):
        seq = sequences[pair["sequence"]]; crop = pair["crop_yx"]
        assert len(pair["starts"]) == 2
        for within_pair, start in enumerate(pair["starts"]):
            window = pair_index * 2 + within_pair
            assert 0 <= start <= 120
            for gap in CONFIG["gaps"]:
                for frame in range(start, start + 40 - gap):
                    for track, annotation in seq.eligible[frame].items():
                        if not all(track in seq.eligible[t] and full_core(seq.eligible[t][track], crop)
                                   for t in range(frame, frame + gap + 1)):
                            continue
                        endpoint = seq.eligible[frame + gap][track]
                        coverage[f"C_legal_candidates_gap{gap}"] += 1
                        anchor = seq.descriptor(annotation, frame, window, start, crop)
                        end = seq.descriptor(endpoint, frame + gap, window, start, crop)
                        area_bin = math.floor(math.log2(anchor["bbox_area"]))
                        if math.floor(math.log2(end["bbox_area"])) != area_bin:
                            coverage[f"excluded_size_bin_change_gap{gap}"] += 1
                            continue
                        key = (gap, anchor["category_id"], area_bin)
                        candidate = {"anchor": anchor, "endpoint": end, "gap": gap, "area_bin": area_bin}
                        groups[key][identity(anchor)].append(candidate)

    units = defaultdict(list)
    for group, pools in sorted(groups.items()):
        gap = group[0]
        queues = {}
        for scoped, candidates in pools.items():
            candidates.sort(key=lambda c: content_hash([seed, step, "candidate", descriptor_key(c["anchor"]), descriptor_key(c["endpoint"])]))
            queues[scoped] = deque(candidates)
        heap = [(-len(q), scoped) for scoped, q in queues.items()]
        heapq.heapify(heap)
        while len(heap) >= 2:
            na, ida = heapq.heappop(heap); nb, idb = heapq.heappop(heap)
            first, second = queues[ida].popleft(), queues[idb].popleft()
            assert ida != idb
            unit = [first, second]
            units[gap].append(unit)
            if na < -1: heapq.heappush(heap, (na + 1, ida))
            if nb < -1: heapq.heappush(heap, (nb + 1, idb))
    # Counts are computed once per gap, not once per category bucket.
    selected = []
    for gap in CONFIG["gaps"]:
        coverage[f"B_C_common_candidates_gap{gap}"] = 2 * len(units[gap])
        units[gap].sort(key=lambda u: content_hash([seed, step, "swap_unit", gap,
                            [[descriptor_key(c["anchor"]), descriptor_key(c["endpoint"])] for c in u]]))
        chosen = units[gap][:CONFIG["max_pairs_per_gap"] // 2]
        coverage[f"selected_pairs_gap{gap}"] = len(chosen) * 2
        selected.extend(chosen)

    descriptors, by_key, regions = [], {}, []
    positive_C, positive_B, metadata, requests = [], [], [], []
    def add(descriptor):
        key = descriptor_key(descriptor)
        if key not in by_key:
            by_key[key] = len(descriptors)
            descriptors.append(descriptor)
            regions.append([descriptor["window"], descriptor["slot"], *descriptor["rect"]])
        else: assert descriptors[by_key[key]] == descriptor
        return by_key[key]
    for swap_unit_id, unit in enumerate(selected):
        for i, candidate in enumerate(unit):
            anchor, end_C, end_B = candidate["anchor"], candidate["endpoint"], unit[1 - i]["endpoint"]
            assert identity(anchor) == identity(end_C) and identity(anchor) != identity(end_B)
            assert anchor["category_id"] == end_B["category_id"] == end_C["category_id"]
            assert max(anchor["bbox_area"], end_B["bbox_area"]) / min(anchor["bbox_area"], end_B["bbox_area"]) < 2
            assert max(anchor["bbox_area"], end_C["bbox_area"]) / min(anchor["bbox_area"], end_C["bbox_area"]) < 2
            a, c, b = add(anchor), add(end_C), add(end_B)
            pair_index = len(positive_C)
            positive_C.append([a, c]); positive_B.append([a, b])
            same_video = (anchor["sequence"], anchor["video_id"]) == (end_B["sequence"], end_B["video_id"])
            metadata.append({
                "pair_index": pair_index, "swap_unit_id": swap_unit_id, "gap_group": candidate["gap"], "C_actual_source_gap": end_C["source_frame"] - anchor["source_frame"],
                "B_same_video": same_video, "B_actual_source_gap": end_B["source_frame"] - anchor["source_frame"] if same_video else None,
                "B_same_primary_window": anchor["window"] == end_B["window"], "B_slot_delta": end_B["slot"] - anchor["slot"],
                "category_id": anchor["category_id"], "area_bin": candidate["area_bin"],
                "C_area_ratio": max(anchor["bbox_area"], end_C["bbox_area"]) / min(anchor["bbox_area"], end_C["bbox_area"]),
                "B_area_ratio": max(anchor["bbox_area"], end_B["bbox_area"]) / min(anchor["bbox_area"], end_B["bbox_area"]),
            })
            x0, y0, x1, y1 = anchor["rect"]
            requests.append({
                "pair_index": pair_index, "swap_unit_id": swap_unit_id, "anchor_descriptor_index": a,
                "sequence": anchor["sequence"], "video_id": anchor["video_id"],
                "frame": anchor["frame"], "source_frame": anchor["source_frame"],
                "primary_window": anchor["window"], "global_window": 4 + anchor["window"],
                "slot": anchor["slot"], "target_rect": anchor["rect"],
                "region_width": x1 - x0, "region_height": y1 - y0,
                "negative_kinds": ["local", "global"], "selection_group": "shared_B_C",
            })
    assert Counter(x[0] for x in positive_B) == Counter(x[0] for x in positive_C)
    assert Counter(x[1] for x in positive_B) == Counter(x[1] for x in positive_C)
    assert len(positive_C) <= 128 and all(m["C_actual_source_gap"] == m["gap_group"] for m in metadata)
    result = {"step": step, "target_regions": regions, "descriptors": descriptors,
              "positive_pairs_B": positive_B, "positive_pairs_C": positive_C,
              "pair_metadata": metadata, "background_requests": requests,
              "pair_count": len(positive_C), "coverage": dict(coverage),
              "visibility_status": "unknown_not_imputed",
              "B_time_rule": "gap_group_matches_C_pool;actual_B_gap_reported_or_null_for_different_videos"}
    result["plan_sha256"] = content_hash(result)
    return result


def write_json(path, value):
    path = Path(path)
    assert path.resolve().is_relative_to(BASE.resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-plan", type=Path, default=STAGE / "pilot/TRAIN_PLAN.json")
    parser.add_argument("--output", type=Path, default=BASE / "PAIR_PLAN.json")
    args = parser.parse_args()
    frozen = json.loads(args.train_plan.read_text(encoding="utf-8"))
    assert frozen["seed"] == 46 and frozen["steps"] == 512
    sampler = TrackPairSampler(STAGE / "data", frozen["train_sequences"])
    for name, seq in sampler.sequences.items():
        assert seq.audit["annotation_sha256"] == frozen["data_fingerprints"][name]["annotations_sha256"]
    entries = [sampler.sample(entry) for entry in frozen["entries"]]
    assert [e["step"] for e in entries] == list(range(512))
    source_hashes = {"TRAIN_PLAN.json": file_hash(args.train_plan),
                     "track_pair_sampler.py": file_hash(Path(__file__)),
                     **{str(seq.path.relative_to(PROJECT)): seq.audit["annotation_sha256"] for seq in sampler.sequences.values()}}
    total = Counter(); sources = Counter(); classes = Counter(); tracks = set(); b_cross_video = 0
    for entry in entries:
        total.update(entry["coverage"])
        for (anchor, _), meta in zip(entry["positive_pairs_C"], entry["pair_metadata"]):
            d = entry["descriptors"][anchor]
            sources[d["sequence"]] += 1; classes[d["category_id"]] += 1; tracks.add(identity(d))
            b_cross_video += not meta["B_same_video"]
    plan = {"schema_version": 1, "config": CONFIG, "source_sha256": source_hashes,
            "training_entries": 512, "repeat_epochs": 4, "training_steps": 2048,
            "entries": entries, "negative_selection_status": "requests_only; cache_preparer_must_select_common_legal_negatives_or_drop_entire_two_row_swap_units_in_B_C"}
    write_json(args.output, plan)
    audit = {"status": "PASS_ANNOTATION_AND_POSITIVE_ASSIGNMENT", "config": CONFIG,
             "source_sha256": source_hashes, "pair_plan_path": str(args.output.relative_to(BASE)),
             "pair_plan_sha256": file_hash(args.output), "sequence_audits": [s.audit for s in sampler.sequences.values()],
             "training_entries": 512, "entries_with_pairs": sum(e["pair_count"] > 0 for e in entries),
             "entries_without_pairs": sum(e["pair_count"] == 0 for e in entries), "totals": dict(total),
             "selected_pair_instances": sum(e["pair_count"] for e in entries), "selected_tracks": len(tracks),
             "selected_pair_instances_by_source": {name: sources[name] for name in sampler.sequences}, "selected_pair_instances_by_class": dict(classes),
             "B_cross_video_pairs": b_cross_video,
             "entry_audit": [{"step": e["step"], "pair_count": e["pair_count"], "plan_sha256": e["plan_sha256"], **e["coverage"]} for e in entries],
             "checks": {"only_original_eight_training_sources": True, "video_scoped_ID": True,
                        "all_gap_intermediate_observations_legal": True, "B_C_same_anchors": True,
                        "B_C_same_endpoint_multiset": True, "B_C_same_target_descriptor_multiset": True,
                        "B_C_same_background_requests": True, "B_C_same_pair_count": True,
                        "same_category_and_area_ratio_below_2": True, "model_response_used": False,
                        "development_or_confirmation_access": False, "negative_regions_selected_and_checked": False},
             "limitations": ["No visibility or occlusion state fields exist in supplied training annotations; retained as unknown.",
                             "Same ID plus continuous legal annotation is an auditable label proxy, not independent proof of physical identity or visibility.",
                             "Negative requests are not yet negative samples; cache preparation must exclude all annotated targets/ignore regions and drop the entire two-row swap unit if either anchor lacks legal negatives.",
                             "Only common cross-identity-compatible examples receive B/C auxiliary loss; all groups retain unchanged detection exposure."]}
    write_json(BASE / "SAMPLER_AUDIT.json", audit)
    lines = ["# 训练配对身份与合法范围审计", "",
             "结论：已生成标注侧共同 B/C 配对清单，未读取图像或模型预测。负样本实际区域由共同教师缓存随后选择并审计。", "",
             f"原 512 条四窗曝光中，{audit['entries_with_pairs']} 条具有共同配对，{audit['entries_without_pairs']} 条无共同配对；正式 2048 步原样重复四轮。",
             f"截断后保留 {audit['selected_pair_instances']} 个配对实例，来自 {len(tracks)} 个视频内身份。gap 1/4 各最多 64 对，每批最多 128 对。", "",
             "每个 C 对要求同一原视频/track_id、实际输入帧差 1 或 4，间隔内每帧均有有效无重叠框且完整处于原裁剪核心区；缺标不插值。B 在同 gap 池、同类别、同 log2 框面积桶中交换两条不同身份的 C 终点，anchor 与两个候选终点面积比均小于 2。B 的实际时间差单独记录，跨视频时间差为 null。", "",
             "B/C 使用完全相同的目标描述及 anchor、终点多重集合、配对数量和背景请求；唯一区别是正终点 assignment。按完整二元交换单元确定性取样，避免截断破坏双射。", "",
             "训练标注没有 visibility、occlusion、occluded 或 status 字段。这个状态保留为未知；连续框不能被表述为已经证明无遮挡，标注身份也没有被独立视觉验证。当前清单排除显式 ignore/crowd、身份冲突、框重叠、越界和缺标。", "",
             "发现同一视频、类别和原 tracking_id 对应多个 track_id 的身份歧义；这些 track 全部剔除，不自行合并，也不将它们互当 B 的不同身份。具体编号保存在 JSON 的 ambiguous_category_local_ID_groups 中。", "",
             "| 来源 | 标注数 | 身份数 | 剔除疑义身份 | 严格有效目标帧 | 已选配对实例 |", "|---|---:|---:|---:|---:|---:|"]
    for name, seq in sampler.sequences.items():
        a = seq.audit
        lines.append(f"| {name} | {a['annotation_count']} | {a['track_count']} | {a['uncertain_tracks']} | {a['eligible_strict_small_observations']} | {sources[name]} |")
    lines += ["", "本表的配对实例包括重复训练窗曝光，不能作为独立目标样本数。背景请求每 anchor 包含同窗 local 与 global 辅助窗两类，actual global crop 由缓存实现固定；全图候选覆盖不等于穷举全图最大响应。", "",
              "`PAIR_PLAN.json` 保存 512 条逐批清单和每批内容哈希；`SAMPLER_AUDIT.json` 保存来源、筛除与覆盖统计。负样本不可用时，缓存必须同步删除该 swap_unit_id 的完整两行（B/C 同时），不能只删除单行而破坏终点多重集合。", ""]
    (BASE / "SAMPLER_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: audit[k] for k in ["status", "entries_with_pairs", "entries_without_pairs", "selected_pair_instances", "selected_tracks", "pair_plan_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
