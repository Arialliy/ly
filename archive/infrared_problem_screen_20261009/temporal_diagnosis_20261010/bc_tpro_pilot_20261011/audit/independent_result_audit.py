#!/usr/bin/env python3
"""Independent CPU result audit on predeclared frames; no scorer imports."""
from __future__ import annotations

import ast
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

BASE = Path(__file__).resolve().parents[1]
DATA = BASE.parent / "stage1_goal/data"
FIELDS = ["gt_count", "matched_count", "component_count", "unmatched_components",
          "positive_pixels", "positive_pixels_outside_boxes"]
HASHES = {}
CHECKS = []


def digest(path):
    path = Path(path)
    if str(path) not in HASHES:
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8*1024*1024), b""):
                h.update(block)
        HASHES[str(path)] = h.hexdigest()
    return HASHES[str(path)]


def check(name, passed, **details):
    CHECKS.append(dict(name=name, passed=bool(passed), **details))
    if not passed:
        raise AssertionError(json.dumps(CHECKS[-1]))


def read_json(path):
    digest(path)
    return json.loads(Path(path).read_text())


def prepare_annotations(sequence):
    data = read_json(DATA / sequence / "annotations.json")
    images = data["images"]
    annotated = collections.defaultdict(list)
    for a in data["annotations"]:
        annotated[a["image_id"]].append(a)
    frames, metadata, offsets = [], [], [0]
    for t, im in enumerate(images):
        valid = []
        for a in sorted(annotated[im["id"]], key=lambda a: (a["track_id"], a["id"])):
            x, y, w, h = map(float, a["bbox"])
            if a.get("ignore", 0) or w <= 0 or h <= 0:
                continue
            if min(im["width"], x+w) <= max(0, x) or min(im["height"], y+h) <= max(0, y):
                continue
            valid.append(dict(a, bbox=[x, y, w, h]))
        frames.append(valid)
        for a in valid:
            x, y, w, h = a["bbox"]
            overlap = False
            for b in valid:
                if a["id"] == b["id"]:
                    continue
                xx, yy, ww, hh = b["bbox"]
                if min(x+w, xx+ww) > max(x, xx) and min(y+h, yy+hh) > max(y, yy):
                    overlap = True
            inside = x >= 0 and y >= 0 and x+w <= im["width"] and y+h <= im["height"]
            metadata.append(dict(frame=t, source_frame=int(Path(im["file_name"]).stem),
                                 image_id=im["id"], track_id=a["track_id"], annotation_id=a["id"],
                                 category_id=a["category_id"], bbox=a["bbox"],
                                 strict_small=int(max(w, h) <= 16 and inside and not overlap)))
        offsets.append(len(metadata))
    return frames, metadata, np.array(offsets)


def independent_components(probability, threshold):
    """Explicit 8-neighbor flood fill, independent of scipy's label routine."""
    height, width = probability.shape
    positive = probability >= threshold
    visited = np.zeros_like(positive)
    components = []
    for flat in np.flatnonzero(positive):
        sy, sx = divmod(int(flat), width)
        if visited[sy, sx]:
            continue
        visited[sy, sx] = True
        pending = [(sy, sx)]
        area, peak_index, peak_value = 0, int(flat), float(probability[sy, sx])
        while pending:
            y, x = pending.pop()
            area += 1
            p, index = float(probability[y, x]), y*width+x
            if p > peak_value or (p == peak_value and index < peak_index):
                peak_index, peak_value = index, p
            for yy in range(max(0, y-1), min(height, y+2)):
                for xx in range(max(0, x-1), min(width, x+2)):
                    if positive[yy, xx] and not visited[yy, xx]:
                        visited[yy, xx] = True
                        pending.append((yy, xx))
        py, px = divmod(peak_index, width)
        components.append(dict(x=px+.5, y=py+.5, score=peak_value, area=area))
    return components, positive


def independent_match(probability, gt, threshold):
    comps, positive = independent_components(probability, threshold)
    adjacency = []
    for ann in gt:
        x, y, w, h = ann["bbox"]
        selected = [i for i, c in enumerate(comps) if x <= c["x"] < x+w and y <= c["y"] < y+h]
        selected.sort(key=lambda i: (((comps[i]["x"]-x-w/2)/w)**2 + ((comps[i]["y"]-y-h/2)/h)**2,
                                    -comps[i]["score"], i))
        adjacency.append(selected)
    # Iterative DFS reproduces the frozen ordered augmenting-path protocol,
    # while implementing its state transitions independently of the scorer.
    owner = {}
    for start in range(len(gt)):
        seen = set()
        stack = [(start, iter(adjacency[start]), None)]
        while stack:
            target, choices, entered_via = stack[-1]
            candidate = next(choices, None)
            if candidate is None:
                stack.pop()
                continue
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate in owner:
                stack.append((owner[candidate], iter(adjacency[owner[candidate]]), candidate))
                continue
            owner[candidate] = target
            while len(stack) > 1:
                _, _, incoming = stack.pop()
                owner[incoming] = stack[-1][0]
            break
    assignments = {target: c for c, target in owner.items()}
    box_pixels = np.zeros(probability.shape, bool)
    yy, xx = np.arange(probability.shape[0])+.5, np.arange(probability.shape[1])+.5
    matched, maxima, peak_x, peak_y, target_area = [], [], [], [], []
    for i, ann in enumerate(gt):
        x, y, w, h = ann["bbox"]
        ys, xs = np.flatnonzero((yy >= y) & (yy < y+h)), np.flatnonzero((xx >= x) & (xx < x+w))
        region = np.ix_(ys, xs)
        box_pixels[region] = True
        maxima.append(float(probability[region].max()) if len(xs) and len(ys) else np.nan)
        match = assignments.get(i)
        matched.append(int(match is not None))
        peak_x.append(comps[match]["x"] if match is not None else np.nan)
        peak_y.append(comps[match]["y"] if match is not None else np.nan)
        target_area.append(comps[match]["area"] if match is not None else 0)
    counts = [len(gt), len(assignments), len(comps), len(comps)-len(assignments),
              int(positive.sum()), int((positive & ~box_pixels).sum())]
    areas = np.array([c["area"] for c in comps], dtype=np.int32)
    unmatched = np.array([int(i not in owner) for i in range(len(comps))], dtype=np.uint8)
    return dict(matched=np.array(matched), box_max=np.array(maxima, np.float32),
                peak_x=np.array(peak_x, np.float32), peak_y=np.array(peak_y, np.float32),
                component_area=np.array(target_area), frame_counts=np.array(counts),
                areas=areas, unmatched=unmatched)


def main():
    began = time.perf_counter()
    torch.set_num_threads(1)
    scope = read_json(BASE / "audit/AUDIT_SCOPE.json")
    source = BASE / "code/bc_tpro.py"
    tree = ast.parse(source.read_text())
    correction = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "correction")
    branch = [node for node in ast.walk(correction) if isinstance(node, ast.If) and "self.variant" in ast.unparse(node.test)]
    check("shared_source_correction_variant_branch_is_only_background_subtraction", len(branch) == 1
          and ast.unparse(branch[0].test) == "self.variant == 'bc_tpro'"
          and len(branch[0].body) == 1 and not branch[0].orelse
          and ast.unparse(branch[0].body[0]) == "correction = correction - self.association_apply(background, value)",
          correction_source=ast.get_source_segment(source.read_text(), correction))
    runs = {variant: read_json(BASE / "training" / variant / "run/RUN.json") for variant in scope["conditions"]}
    first, second = (runs[x] for x in scope["conditions"])
    for key in ["model_config", "bc_tpro_source_sha256", "new_module_initial_state_sha256", "plan_sha256",
                "checkpoint_sha256", "model_source_sha256", "seed", "new_module_seed", "steps", "reference_lambda"]:
        check("same_between_variants_" + key, first[key] == second[key])
    metadata_by_seq = {seq: prepare_annotations(seq) for seq in scope["sequences"]}
    check("independent_frozen_small_target_total", sum(sum(r["strict_small"] for r in vals[1])
          for vals in metadata_by_seq.values()) == 5200)
    audited_rows, tiling = [], []
    for variant in scope["conditions"]:
        run = runs[variant]
        checkpoint = BASE / "training" / variant / "run/final_model.pth"
        cp = torch.load(checkpoint, map_location="cpu", weights_only=True)
        gamma = float(cp["model_state_dict"]["TPro.gamma"])
        check(variant + "_trained_weight_identity", run["status"] == "complete"
              and run["steps"] == run["final_step"] == cp["step"] == 512
              and run["variant"] == cp["variant"] == variant
              and digest(checkpoint) == run["final_checkpoint_sha256"]
              and digest(source) == run["bc_tpro_source_sha256"]
              and cp["model_config"] == run["model_config"] and gamma != 0,
              gamma=gamma, checkpoint_sha256=digest(checkpoint))
        for seq in scope["sequences"]:
            identity = variant + "/" + seq
            pred = BASE / "predictions" / variant / seq
            ev = BASE / "evaluation" / variant / seq
            receipt = read_json(pred / "COMPLETE.json")
            scores_receipt = read_json(ev / "COMPLETE.json")
            starts_union = sorted(set(scope["plans"]["O"] + scope["plans"]["S"]))
            check(identity + "_inference_protocol", receipt["status"] == "COMPLETE"
                  and receipt["variant"] == variant and receipt["sequence"] == seq
                  and receipt["checkpoint_sha256"] == digest(checkpoint)
                  and receipt["bc_source_sha256"] == digest(source)
                  and receipt["model_source_sha256"] == run["model_source_sha256"]
                  and receipt["reference_checkpoint_sha256"] == run["checkpoint_sha256"]
                  and receipt["model_config"] == run["model_config"]
                  and receipt["raw_sha256"] == digest(DATA / seq / "raw_frames.npy")
                  and receipt["normalization"] == [85.0799, 47.4845]
                  and receipt["precision"] == "FP32" and receipt["tile_core"] == 128
                  and receipt["tile_halo"] == 64 and receipt["model_labels_access"] is False
                  and receipt["official_starts"] == scope["plans"]["O"]
                  and receipt["diagnostic_starts"] == starts_union
                  and receipt["script_sha256"] == digest(BASE / "code/infer_bc.py"))
            witnesses = receipt["tiling_checks"]
            check(identity + "_trained_tiling_witnesses", len(witnesses) == 3
                  and [w["yx"] for w in witnesses] == [[0, 0], [176, 224], [352, 448]]
                  and all(w["gamma"] == gamma and w["threshold05_disagreements"] == 0
                          and math.isfinite(w["max_logit_error"]) and math.isfinite(w["max_probability_error"])
                          for w in witnesses))
            tiling.extend([dict(variant=variant, sequence=seq, **w) for w in witnesses])
            gt, meta, offsets = metadata_by_seq[seq]
            check(identity + "_scoring_metadata", scores_receipt["status"] == "COMPLETE"
                  and scores_receipt["condition"] == variant and scores_receipt["sequence"] == seq
                  and scores_receipt["threshold"] == scope["threshold"] == .5
                  and scores_receipt["objects"] == meta and scores_receipt["checkpoint_sha256"] == digest(checkpoint))
            with np.load(ev / "scores.npz", allow_pickle=False) as ar:
                scores = {k: ar[k] for k in ar.files}
            with np.load(ev / "component_areas.npz", allow_pickle=False) as ar:
                areas = {k: ar[k] for k in ar.files}
            digest(ev / "scores.npz"); digest(ev / "component_areas.npz")
            digest(ev / "FRAME_COUNTS.csv")
            with (ev / "FRAME_COUNTS.csv").open() as stream:
                frame_csv = {(r["arrangement"], int(r["frame"])): r for r in csv.DictReader(stream)}
            check(identity + "_score_array_protocol", float(scores["threshold"]) == .5
                  and scores["plans"].tolist() == list(scope["plans"])
                  and np.array_equal(scores["offsets"], offsets)
                  and scores["strict_small"].tolist() == [bool(r["strict_small"]) for r in meta])
            sample_probabilities = {}
            for start in starts_union:
                p = pred / "windows" / f"window_{start:03d}.npz"
                check(identity + f"_window_{start}_hash", digest(p) == receipt["window_sha256"][p.name])
                with np.load(p, allow_pickle=False) as ar:
                    prob = ar["probabilities"]
                    check(identity + f"_window_{start}_valid", int(ar["start"]) == start and int(ar["end"]) == start+40
                          and prob.shape == (40, 512, 640) and prob.dtype == np.float32
                          and np.isfinite(prob).all() and np.all((prob >= 0) & (prob <= 1)))
                    for frame in scope["frames_fixed_before_reading_scores"]:
                        if start <= frame < start+40:
                            sample_probabilities[start, frame] = prob[frame-start].copy()
            del prob
            fused_path = pred / "official_fused.npz"
            check(identity + "_official_fused_hash", digest(fused_path) == receipt["fused_sha256"])
            with np.load(fused_path, allow_pickle=False) as ar:
                check(identity + "_official_fused_starts", ar["starts"].tolist() == scope["plans"]["O"])
                full_fused = ar["probabilities"]
                original_samples = {frame: full_fused[frame].copy() for frame in scope["frames_fixed_before_reading_scores"]}
            del full_fused
            for pi, (arrangement, starts) in enumerate(scope["plans"].items()):
                coverage = np.zeros(160, np.int16)
                for start in starts:
                    coverage[start:start+40] += 1
                check(identity + "/" + arrangement + "_five_window_coverage", coverage.sum() == 200
                      and np.all(coverage > 0) and np.array_equal(scores["coverage"][pi], coverage))
                for frame in scope["frames_fixed_before_reading_scores"]:
                    title = identity + f"/{arrangement}/{frame}"
                    selected_starts = [s for s in starts if s <= frame < s+40]
                    fused = np.maximum.reduce([sample_probabilities[s, frame] for s in selected_starts])
                    if arrangement == "O":
                        check(title + "_fused_probability_exact", np.array_equal(fused, original_samples[frame]))
                    independent = independent_match(fused, gt[frame], .5)
                    lo, hi = offsets[frame:frame+2]
                    for key in ["matched", "box_max", "peak_x", "peak_y", "component_area"]:
                        check(title + "_" + key, np.array_equal(independent[key], scores[key][pi, lo:hi], equal_nan=True))
                    check(title + "_frame_counts", np.array_equal(independent["frame_counts"], scores["frame_counts"][pi, frame]))
                    csv_row = frame_csv[arrangement, frame]
                    check(title + "_frame_csv", all(int(csv_row[k]) == int(v) for k, v in zip(FIELDS, independent["frame_counts"]))
                          and float(csv_row["threshold"]) == .5 and int(csv_row["coverage_windows"]) == coverage[frame])
                    al, ah = areas[arrangement + "_offsets"][frame:frame+2]
                    check(title + "_component_areas_and_unmatched", np.array_equal(independent["areas"], areas[arrangement + "_areas"][al:ah])
                          and np.array_equal(independent["unmatched"], areas[arrangement + "_unmatched"][al:ah]))
                    unmatched_areas = independent["areas"][independent["unmatched"].astype(bool)]
                    check(title + "_background_area_csv", int(unmatched_areas.sum()) == int(csv_row["unmatched_component_area"])
                          and int(unmatched_areas.max(initial=0)) == int(csv_row["unmatched_component_max_area"])
                          and int(independent["areas"].max(initial=0)) == int(csv_row["component_max_area"]))
                    audited_rows.append(dict(variant=variant, sequence=seq, arrangement=arrangement, frame=frame,
                                             covering_starts=selected_starts, **dict(zip(FIELDS, map(int, independent["frame_counts"])))))
    check("predeclared_total_recomputed_frames", len(audited_rows) == 40)
    digest(__file__)
    result = dict(status="PASS", scope=scope, checks=CHECKS, check_count=len(CHECKS),
                  audited_frame_rows=audited_rows, tiling_witnesses=tiling,
                  input_sha256=HASHES, elapsed_seconds=time.perf_counter()-began,
                  limitations=["Only 40 of 1280 new fused frame-arrangement cases independently rescored.",
                               "Existing GPU trained tiling witnesses were inspected and source-bound; GPU inference was not independently rerun.",
                               "Auditor authored model code but did not author or call the scoring implementation; this is not an independent architecture review.",
                               "Diagnostic peak-in-box matching and unmatched components are not official Pd/Fa."])
    (BASE / "INDEPENDENT_RESULT_AUDIT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    text = ("# 独立结果抽样审计\n\n"
            f"状态：PASS。{len(CHECKS)} 项检查通过；CPU耗时 {result['elapsed_seconds']:.2f} 秒。\n\n"
            "预先固定每段帧 [0,39,80,120,159]，对两模型×两片段×O/S共40个融合帧独立复算。"
            "从原标注独立构建目标列表，并用独立8邻域遍历、行优先同值峰、半开框匹配及有序增广匹配核对逐目标得分/峰/面积和全部帧级背景计数。"
            "没有导入或调用原评分器。\n\n"
            "所有抽帧输出与 scores.npz、FRAME_COUNTS.csv、组件面积记录一致。四份推理的权重/源码/八窗池与五窗融合身份吻合；"
            "12个已保存的训练后direct/tiled见证均在0.5阈值无判决差异。共享代码、初始状态、训练计划和配置绑定一致，"
            "候选关联函数中的变体分支仅在完整模型减去背景关联项。\n\n"
            "范围限制：这是40/1280个融合帧的抽样输出审计，不是全部预测的独立复算；平铺证据是核验已有GPU记录，未另跑GPU。"
            "审计员参与模型实现、独立于评分实现，因此不能代替新的模型数学审查。未匹配组件不是官方Fa。\n\n"
            "完整检查、输入SHA和抽帧计数见 [INDEPENDENT_RESULT_AUDIT.json](INDEPENDENT_RESULT_AUDIT.json)。\n")
    (BASE / "INDEPENDENT_RESULT_AUDIT.md").write_text(text)
    print(json.dumps(dict(status="PASS", checks=len(CHECKS), audited_frames=len(audited_rows), seconds=result["elapsed_seconds"])))


if __name__ == "__main__":
    main()
