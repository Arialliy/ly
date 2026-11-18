#!/usr/bin/env python3
"""Independent CPU audit of every cache and eight fixed teacher replays.

Cache and production modules are read-only. All geometry masks below are
rebuilt from source annotations rather than imported from cache preparation.
Eight annotation-selected entries (0..7, covering all sources) additionally
replay the original release model on CPU and call the frozen actual matcher.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

BASE = Path(__file__).resolve().parents[1]
PROJECT = BASE.parents[1]
STAGE = PROJECT / "archive/infrared_problem_screen_20261009/temporal_diagnosis_20261010/stage1_goal"
REPORT = BASE / "resources/CACHE_AUDIT.json"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def jread(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_report(report):
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    temporary = REPORT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(REPORT)
    lines = ["# 共同教师缓存独立审计", "", f"状态：{report['status']}。", "",
             "全量检查独立由原始标注重建背景排除域；额外 8 条固定训练曝光使用原发布模型 CPU 重算并调用原实际组件匹配，未访问开发/确认来源。", ""]
    if "full_cache" in report:
        value = report["full_cache"]
        lines += [f"全量检查 {value['entries']} 批、{value['requested_pairs']} 个预定配对；背景筛选后有效 {value['valid_pairs']} 对，{value['invalid_swap_units']} 个完整交换单元失效。",
                  "B/C 的有效 anchor 和 endpoint 多重集合保持完全一致；失效以两行 swap_unit 为单位。所有负 ROI 都完全位于独立重建的背景掩码内，包含非严格小目标、忽略/拥挤区域及裁块外圈的排除。", "",
                  f"检查 keep 坐标 {value['keep_coordinates']} 个，类别为可靠目标/背景，概率有限且在 [0,1]；背景概率不超过 0.1，每窗不超过 1024 点。", ""]
    if "cpu_replay" in report:
        lines += ["| 固定批次 | CPU 重算窗 | keep 概率最大绝对误差 | 可靠目标像素与缓存一致 |", "|---|---:|---:|---|"]
        for row in report["cpu_replay"]:
            lines.append(f"| {row['entry']} | 8 | {row['cached_probability_max_abs_error']:.8g} | {row['target_mask_exact']} |")
        lines += ["", "CPU 重算仅为实现检查；该审计不把 CPU 数值作为训练输出，也不新增正式检测对照。训练侧标注没有可见性/遮挡或未标目标完整性真值；合法性仍限定为已有标注与固定排除规则。", ""]
    if "error" in report: lines += [f"失败：{report['error']}", ""]
    REPORT.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def valid_box(a):
    b = a.get("bbox", [])
    return len(b) == 4 and np.isfinite(b).all() and b[2] > 0 and b[3] > 0


def rect(box, margin=0):
    x, y, w, h = box
    return tuple(max(0, min(128, math.ceil(v - .5))) for v in (x-margin, y-margin, x+w+margin, y+h+margin))


def paint(mask, frame, box, margin=0):
    x0, y0, x1, y1 = rect(box, margin)
    mask[frame, y0:y1, x0:x1] = True


def expected_aux(entry):
    value = json.loads(json.dumps(entry))
    grid = [(y, x) for y in range(0, 385, 64) for x in range(0, 513, 64)]
    for pi, pair in enumerate(value["pairs"]):
        idx = (entry["step"] * 17 + pi * 29 + 46) % 63
        if list(grid[idx]) == pair["crop_yx"]: idx = (idx + 1) % 63
        pair["crop_yx"] = list(grid[idx])
    return value


def load_annotations(names):
    result = {}
    for name in names:
        path = STAGE / "data" / name / "annotations.json"
        value = jread(path)
        frames = [[] for _ in range(160)]
        index = {im["id"]: i for i, im in enumerate(value["images"])}
        for a in value["annotations"]: frames[index[a["image_id"]]].append(a)
        result[name] = frames
    return result


def geometry(entry, auxiliary, annotations):
    windows, masks, target_allowed, nonstrict = [], [], [], 0
    for exposure in (entry, auxiliary):
        for pair in exposure["pairs"]:
            cy, cx = pair["crop_yx"]
            for start in pair["starts"]:
                bg = np.zeros((40, 128, 128), bool); bg[:, 9:119, 9:119] = True
                targets = np.zeros_like(bg); frames = []
                for ti in range(40):
                    local = []
                    for a in annotations[pair["sequence"]][start + ti]:
                        if not valid_box(a):
                            bg[ti] = False; local.append(dict(a)); continue
                        x, y, w, h = a["bbox"]
                        value = dict(a); value["bbox"] = [x-cx, y-cy, w, h]
                        exclusion = np.zeros((1, 128, 128), bool)
                        paint(exclusion, 0, value["bbox"], 2)
                        bg[ti] &= ~exclusion[0]
                        if max(w, h) > 16 and exclusion.any(): nonstrict += 1
                        if x+w <= cx or y+h <= cy or x >= cx+128 or y >= cy+128: continue
                        local.append(value)
                        if (not a.get("ignore", 0) and not a.get("iscrowd", 0)
                            and x-cx >= 9 and y-cy >= 9 and x-cx+w <= 119 and y-cy+h <= 119):
                            paint(targets, ti, value["bbox"])
                    frames.append(local)
                windows.append(frames); masks.append(bg); target_allowed.append(targets)
    assert len(windows) == 8
    return windows, np.stack(masks), np.stack(target_allowed), nonstrict


def check_regions(regions, max_window=8):
    assert regions.ndim == 2 and regions.shape[1] == 6 and regions.dtype == np.int64
    if len(regions):
        assert ((regions[:, 0] >= 0) & (regions[:, 0] < max_window)).all()
        assert ((regions[:, 1] >= 0) & (regions[:, 1] < 40)).all()
        assert (regions[:, 2:4] >= 0).all() and (regions[:, 4:6] <= 128).all()
        assert (regions[:, 4:6] > regions[:, 2:4]).all()


def full_check(train, pair_plan, annotations):
    totals = Counter(); entry_rows = []; evidence = {}; hashes = {}
    for entry in train["entries"]:
        index = entry["step"]; plan = pair_plan["entries"][index]
        path = BASE / "cache" / f"entry_{index:03d}.npz"
        receipt_path = path.with_suffix(".json")
        receipt = jread(receipt_path)
        assert receipt["sha256"] == sha(path), index
        evidence[str(path.relative_to(BASE))] = receipt["sha256"]
        evidence[str(receipt_path.relative_to(BASE))] = sha(receipt_path)
        for name, expected in receipt["sources"].items():
            if name not in hashes: hashes[name] = sha(name)
            assert hashes[name] == expected, (index, name)
        with np.load(path, allow_pickle=False) as archive:
            cache = {name: archive[name] for name in archive.files}
        meta = json.loads(str(cache["metadata_json"]))
        assert meta == {k: v for k, v in receipt.items() if k != "sha256"}
        assert meta["index"] == index
        auxiliary = expected_aux(entry)
        assert [p["crop_yx"] for p in meta["aux_entry"]["pairs"]] == [p["crop_yx"] for p in auxiliary["pairs"]]
        frames, bg_mask, allowed_target, nonstrict = geometry(entry, auxiliary, annotations)
        targets, backgrounds = cache["target_regions"], cache["background_regions"]
        pc, pb, neg = cache["positive_pairs_C"], cache["positive_pairs_B"], cache["negative_indices"]
        check_regions(targets, 4); check_regions(backgrounds)
        np.testing.assert_array_equal(targets, np.asarray(plan["target_regions"], np.int64).reshape(-1, 6))
        np.testing.assert_array_equal(pc, np.asarray(plan["positive_pairs_C"], np.int64).reshape(-1, 2))
        np.testing.assert_array_equal(pb, np.asarray(plan["positive_pairs_B"], np.int64).reshape(-1, 2))
        assert neg.shape == (len(pc), 2) and neg.dtype == np.int64
        assert (neg >= -1).all() and (neg < len(backgrounds)).all()
        valid = (neg >= 0).all(1)
        assert (~valid | (neg >= 0).all(1)).all() and (neg[~valid] == -1).all()
        units = defaultdict(list)
        for i, item in enumerate(plan["pair_metadata"]): units[item["swap_unit_id"]].append(i)
        for indices in units.values():
            assert len(indices) == 2 and len(set(valid[indices].tolist())) == 1
        assert np.array_equal(pc[valid, 0], pb[valid, 0])
        assert Counter(pc[valid, 1]) == Counter(pb[valid, 1])
        assert int(valid.sum()) == receipt["pairs_with_two_negatives"]
        assert len(pc) == receipt["pairs"] == plan["pair_count"]
        for wi, ti, x0, y0, x1, y1 in backgrounds:
            assert bg_mask[wi, ti, y0:y1, x0:x1].all(), (index, "background ROI intersects excluded annotation")
        for i in np.flatnonzero(valid):
            anchor = targets[pc[i, 0]]
            for branch in range(2):
                roi = backgrounds[neg[i, branch]]
                assert roi[0] == anchor[0] + 4 * branch and roi[1] == anchor[1]
                assert np.array_equal(roi[4:6]-roi[2:4], anchor[4:6]-anchor[2:4])
                if branch == 0: assert int(((roi[2:4]-anchor[2:4])**2).sum()) <= 32**2
        coords, probs, groups = cache["keep_indices"], cache["keep_probabilities"], cache["keep_groups"]
        assert coords.dtype == np.int64 and coords.shape == (len(probs), 4)
        assert probs.dtype == np.float32 and groups.dtype == np.int64 and groups.shape == probs.shape
        assert np.isfinite(probs).all() and ((probs >= 0) & (probs <= 1)).all()
        assert (coords >= 0).all() and (coords < np.array([8, 40, 128, 128])).all()
        assert np.isin(groups, [0, 1]).all()
        flat = np.ravel_multi_index(tuple(coords.T), (8, 40, 128, 128))
        assert len(np.unique(flat)) == len(flat)
        background = groups == 1; target = groups == 0
        assert (probs[background] <= .1).all()
        assert bg_mask[tuple(coords[background].T)].all()
        assert allowed_target[tuple(coords[target].T)].all()
        for wi in range(8):
            wm = coords[:, 0] == wi
            assert int((wm & background).sum()) <= 1024
            stats = receipt["keep_stats"][wi]["selection"]
            assert int((wm & background).sum()) == stats["background_pixels"]
            assert int((wm & target).sum()) == stats["target_pixels"]
        counts = {"entry": index, "requested_pairs": len(pc), "valid_pairs": int(valid.sum()),
                  "invalid_swap_units": sum(not valid[indices[0]] for indices in units.values()),
                  "background_regions": len(backgrounds), "keep_coordinates": len(coords),
                  "nonstrict_annotation_window_instances_excluded": nonstrict}
        totals.update({k: v for k, v in counts.items() if k != "entry"}); totals["entries"] += 1
        totals["entries_with_valid_pairs"] += bool(valid.any()); entry_rows.append(counts)
        if index % 64 == 0: print(json.dumps({"cache_audit_entry": index, **counts}), flush=True)
    return dict(totals), entry_rows, evidence


def cpu_replay(train, annotations):
    sys.path.insert(0, str(BASE / "code"))
    import torch
    import runtime
    runtime.setup(threads=2)
    assert not torch.cuda.is_available(), "Audit must run in CPU-only container"
    model, _ = runtime.load_model(device="cpu")
    matcher = runtime.load_module("independent_keep_matcher", runtime.Q / "analysis/analyze.py")
    selected, seen = [], set()
    for entry in train["entries"]:
        names = {p["sequence"] for p in entry["pairs"]}
        if names - seen: selected.append(entry["step"]); seen.update(names)
        if len(seen) == 8: break
    for entry in train["entries"]:
        if len(selected) == 8: break
        if entry["step"] not in selected: selected.append(entry["step"])
    selected.sort(); assert selected == list(range(8))
    arrays = {name: np.load(STAGE / "data" / name / "raw_frames.npy", mmap_mode="r") for name in train["train_sequences"]}
    results = []
    for index in selected:
        began = time.perf_counter(); entry = train["entries"][index]; auxiliary = expected_aux(entry)
        frames, bg_mask, _, _ = geometry(entry, auxiliary, annotations)
        crops = []
        for exposure in (entry, auxiliary):
            for pair in exposure["pairs"]:
                cy, cx = pair["crop_yx"]
                for start in pair["starts"]:
                    crops.append((arrays[pair["sequence"]][start:start+40, cy:cy+128, cx:cx+128].astype(np.float32)-85.0799)/47.4845)
        x = np.stack(crops)[:, None]
        with torch.inference_mode():
            probabilities = torch.cat([model(torch.from_numpy(x[i:i+2]))[1].sigmoid() for i in range(0, 8, 2)]).numpy()
        with np.load(BASE / "cache" / f"entry_{index:03d}.npz", allow_pickle=False) as cache:
            coords, cached, groups = cache["keep_indices"], cache["keep_probabilities"], cache["keep_groups"]
        observed = probabilities[tuple(coords.T)]
        error = float(np.abs(observed-cached).max(initial=0))
        np.testing.assert_allclose(observed, cached, atol=2e-5, rtol=1e-4)
        reliable = np.zeros_like(probabilities, bool); forbidden = np.zeros_like(probabilities, bool)
        matched, missed, ignored, overlap, bad_core = 0, 0, 0, 0, 0
        for wi in range(8):
            for ti, annotations_frame in enumerate(frames[wi]):
                if any(not valid_box(a) or "id" not in a or "track_id" not in a for a in annotations_frame):
                    forbidden[wi, ti] = True; continue
                gt, _ = matcher.prepare_gt(annotations_frame, 128, 128)
                rows, _, _ = matcher.match_frame(probabilities[wi, ti], gt, .5)
                by_id = {r["annotation_id"]: r for r in rows}
                for a in annotations_frame:
                    row = by_id.get(a["id"]); x0, y0, w, h = a["bbox"]
                    ok = bool(row and row["matched"] and not row["association_ambiguous"]
                              and not row["label_overlap_count"] and not row["partly_outside"]
                              and not a.get("ignore", 0) and not a.get("iscrowd", 0)
                              and x0 >= 9 and y0 >= 9 and x0+w <= 119 and y0+h <= 119)
                    matched += int(bool(row and row["matched"]))
                    missed += int(bool(row and not row["matched"]))
                    ignored += int(bool(a.get("ignore", 0) or a.get("iscrowd", 0)))
                    overlap += int(bool(row and row["label_overlap_count"]))
                    bad_core += int(not (x0 >= 9 and y0 >= 9 and x0+w <= 119 and y0+h <= 119))
                    if ok: paint(reliable[wi], ti, a["bbox"])
                    else: paint(forbidden[wi], ti, a["bbox"], 2)
        reliable &= ~forbidden
        assert not forbidden[tuple(coords.T)].any(), (index, "cached keep intersects independently matched forbidden region")
        cached_target = np.zeros_like(reliable)
        cached_target[tuple(coords[groups == 0].T)] = True
        assert np.array_equal(cached_target, reliable), (index, "replayed reliable target mask differs")
        assert bg_mask[tuple(coords[groups == 1].T)].all()
        result = {"entry": index, "sources": [p["sequence"] for p in entry["pairs"]],
                  "cached_probability_max_abs_error": error, "target_mask_exact": True,
                  "matched_annotation_window_instances": matched, "missed_annotation_window_instances": missed,
                  "ignore_crowd_annotation_window_instances": ignored, "overlap_annotation_window_instances": overlap,
                  "outside_core_annotation_window_instances": bad_core, "keep_forbidden_intersections": 0,
                  "elapsed_seconds": time.perf_counter()-began}
        results.append(result); print(json.dumps({"cpu_teacher_replay": result}), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-only", action="store_true", help="Do not report final PASS without CPU teacher replay")
    args = parser.parse_args()
    required = [BASE / "cache" / f"entry_{i:03d}{suffix}" for i in range(512) for suffix in [".json", ".npz"]]
    missing = [str(p.relative_to(BASE)) for p in required if not p.is_file()]
    if missing: raise FileNotFoundError(f"Cache not complete: {len(missing)} required files missing; first {missing[:4]}")
    train = jread(STAGE / "pilot/TRAIN_PLAN.json"); pair_plan = jread(BASE / "PAIR_PLAN.json")
    assert len(train["entries"]) == len(pair_plan["entries"]) == 512
    annotations = load_annotations(train["train_sequences"])
    report = {"status": "RUNNING", "scope": "512 training caches; CPU replay of eight fixed entries; no GPU/no evaluation-source access",
              "audit_script_sha256": sha(Path(__file__)), "pair_plan_sha256": sha(BASE / "PAIR_PLAN.json")}
    try:
        totals, entries, evidence = full_check(train, pair_plan, annotations)
        report.update(full_cache=totals, entry_audit=entries, cache_sha256=evidence)
        report["status"] = "PASS_GEOMETRY_ONLY_PENDING_CPU_REPLAY"
        write_report(report)
        print(json.dumps({"geometry_audit_status": report["status"], "totals": totals}), flush=True)
        if args.geometry_only:
            report["status"] = "PASS_GEOMETRY_ONLY_PENDING_CPU_REPLAY"
        else:
            report["cpu_replay"] = cpu_replay(train, annotations)
            report["status"] = "PASS"
    except Exception as exc:
        report.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
        write_report(report)
        raise
    write_report(report)
    print(json.dumps({"status": report["status"], "path": str(REPORT), "totals": report["full_cache"]}), flush=True)


if __name__ == "__main__":
    main()
