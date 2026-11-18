"""Evaluate paired full-resolution thermal traffic predictions on the development videos."""
import json

import cv2
import numpy as np

from protocol import CENTERS, DEV, RUN, frame_record
from readout import BUDGETS as PILOT_BUDGETS, froc, label_candidates


METHODS = ("original", "A", "B", "C", "D")
BUDGETS = tuple(map(str, PILOT_BUDGETS))


def peaks(logits):
    """All 5x5 local maxima, resolving plateaus in descending score, row, column order."""
    if logits.shape != (512, 640) or logits.dtype != np.float32 or not np.isfinite(logits).all():
        raise ValueError("Expected finite float32 native logits with shape (512,640).")
    rows, cols = np.nonzero(logits == cv2.dilate(logits, np.ones((5, 5), np.uint8)))
    order = np.lexsort((cols, rows, -logits[rows, cols]))
    blocked = np.zeros((logits.shape[0] + 4, logits.shape[1] + 4), dtype=bool)
    keep = []
    for k in order:
        y, x = rows[k], cols[k]
        if not blocked[y + 2, x + 2]:
            keep.append(k)
            blocked[y:y + 5, x:x + 5] = True
    y, x = rows[keep], cols[keep]
    return np.column_stack((x, y)), logits[y, x]


def at_threshold(threshold, scores, frame_index, labels, neighbors, frames, subsets):
    keep = np.flatnonzero(scores >= threshold) if threshold is not None else np.empty(0, dtype=int)
    fixed_labels = labels[keep]
    fixed_frames = frame_index[keep]
    fixed_neighbors = [neighbors[i] for i in keep]
    counts = {}
    for name, ids in subsets.items():
        fixed = froc(scores[keep], ids, frames, fixed_frames, fixed_labels, fixed_neighbors)
        tp = fixed["candidate_oracle_tp"]
        fp = fixed["candidates"] - fixed["ignored_candidates"] - tp
        counts[name] = {
            "tp": tp, "fp": fp, "fn": fixed["targets"] - tp,
            "recall": fixed["candidate_oracle_recall"], "fp_per_frame": fp / fixed["frames"],
            "ignored_candidates": fixed["ignored_candidates"], "logit_threshold": threshold,
        }
    return counts


def evaluate_method(frames, xy_scores, subsets):
    xy = np.concatenate([xy for xy, _ in xy_scores])
    scores = np.concatenate([scores for _, scores in xy_scores])
    frame_index = np.repeat(np.arange(len(frames)), [len(s) for _, s in xy_scores])
    labels, neighbors = label_candidates(xy, frame_index, frames)
    out = {name: froc(scores, ids, frames, frame_index, labels, neighbors) for name, ids in subsets.items()}
    primary = out["all_dev"]["froc"]["1"]
    for label, threshold in (("fixed_probability_0.5", 0.0), ("at_pooled_primary_threshold", primary["logit_threshold"])):
        counts = at_threshold(threshold, scores, frame_index, labels, neighbors, frames, subsets)
        for name in out:
            out[name][label] = counts[name]
    for key in ("tp", "fp"):
        observed = out["all_dev"]["at_pooled_primary_threshold"][key]
        by_source = [out[name]["at_pooled_primary_threshold"][key] for name in subsets if name != "all_dev"]
        if observed != primary[key] or (by_source and sum(by_source) != observed):
            raise AssertionError(f"Pooled threshold counts do not reconcile for {key}")
    return out


CONTRASTS = {
    "B-A": {"B": 1, "A": -1},
    "D-C": {"D": 1, "C": -1},
    "C-A": {"C": 1, "A": -1},
    "D-B": {"D": 1, "B": -1},
    "(D-C)-(B-A)": {"D": 1, "C": -1, "B": -1, "A": 1},
    "D-original": {"D": 1, "original": -1},
}


def contrasts(methods, groups):
    out = {}
    for group in groups:
        out[group] = {}
        for name, weights in CONTRASTS.items():
            rows = [(methods[method][group], weight) for method, weight in weights.items()]
            delta = {
                budget: sum(row["froc"][budget]["tp"] * weight for row, weight in rows)
                for budget in BUDGETS
            }
            fixed = {
                key: sum(row["fixed_probability_0.5"][key] * weight for row, weight in rows)
                for key in ("tp", "fp")
            }
            out[group][name] = {"delta_tp_at_fp_budgets": delta, "fixed_probability_0.5_delta": fixed}
    return out


def markdown(report):
    lines = [
        "# 红外交通视频 2×2 开发集结果",
        "",
        "使用相同视频时刻和框标注，在每个模型自己的整幅分数图上提取全部局部峰。"
        "A、B 使用全部候选选择帧估计速度，C、D 只用发现组估计速度，再由另一组检验。"
        "A、C 使用两步表示，B、D 保留时间子集和邻近速度的联合响应。",
        "",
        "表中前三列分别为每帧最多 0.5、1、5 个虚警时实际取得的检出数。"
        "每列、每个模型、每个来源都单独扫阈值。后两列统一使用概率 ≥ 0.5（logit ≥ 0）。",
        "",
    ]
    for group, effects in report["contrasts"].items():
        reference = report["methods"]["original"][group]
        name = "两个开发来源合并" if group == "all_dev" else group
        lines += [
            f"## {name}",
            "",
            f'{reference["frames"]} 个中心帧，{reference["targets"]} 个逐帧目标框。',
            "",
            "| 模型 | 0.5 虚警/帧 | 1 虚警/帧 | 5 虚警/帧 | 固定阈值检出 | 固定阈值虚警 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for model in METHODS:
            row = report["methods"][model][group]
            values = [row["froc"][b]["tp"] for b in BUDGETS]
            fixed = row["fixed_probability_0.5"]
            lines.append("| " + " | ".join(map(str, [model, *values, fixed["tp"], fixed["fp"]])) + " |")
        lines += [
            "",
            "| 配对差值 | Δ检出 @ 0.5 | Δ检出 @ 1 | Δ检出 @ 5 | 固定阈值 Δ检出 | 固定阈值 Δ虚警 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for contrast, delta in effects.items():
            values = [delta["delta_tp_at_fp_budgets"][b] for b in BUDGETS]
            fixed = delta["fixed_probability_0.5_delta"]
            entries = [f"{v:+d}" for v in [*values, fixed["tp"], fixed["fp"]]]
            lines.append("| " + " | ".join([contrast, *entries]) + " |")
        lines.append("")
    sources = [group for group in report["contrasts"] if group != "all_dev"]
    lines += [
        "## 合并主工作点的来源分解",
        "",
        "每个模型沿用在两个开发来源合并后、每帧最多 1 个虚警时选定的同一阈值。"
        "各来源单元格为检出数 / 虚警数，两来源相加得到合并列。阈值从这些开发数据上选取。",
        "",
        "| 模型 | 原始 logit 阈值 | " + " | ".join([*sources, "合并"]) + " |",
        "|---|---:|" + "---:|" * (len(sources) + 1),
    ]
    for model in METHODS:
        groups = report["methods"][model]
        fixed = [groups[name]["at_pooled_primary_threshold"] for name in [*sources, "all_dev"]]
        threshold = fixed[-1]["logit_threshold"]
        display = "无检出" if threshold is None else f"{threshold:.6g}"
        cells = [f'{row["tp"]} / {row["fp"]}' for row in fixed]
        lines.append("| " + " | ".join([model, display, *cells]) + " |")
    lines.append("")
    lines += [
        "B−A 与 D−C 衡量两种速度选择方式下联合表示各自带来的变化。"
        "交互项 (D−C)−(B−A) 衡量这两个增量的差。C−A 与 D−B 分别衡量分组选择在两种表示下的增量。",
        "",
        "各模型在 5×5 邻域中提取局部峰，同分按行列顺序处理，2 像素范围内留一个。"
        "峰落入外扩 1 像素的合格框即可参与最大匹配。一个峰与一个框最多各计一次，重复检出算虚警，"
        "仅命中忽略框的峰从评价中排除。逐帧框是召回率分母，同一实例出现在多个时刻会多次计入。",
        "",
        "这两个交通来源已有开发使用历史，各自属于不同场景。按来源分别扫阈值后，检出数不可直接相加得到合并结果。"
        "经验曲线、固定种子训练及上表的交互差值仅描述本次开发实验，无法单独证明泛化、统计显著性或论文级创新。",
        "",
    ]
    return "\n".join(lines)


def main():
    frames = [frame_record(seq, idx) for seq in DEV for idx in CENTERS]
    subsets = {"all_dev": list(range(len(frames)))}
    subsets.update({seq: [i for i, f in enumerate(frames) if f["sequence"] == seq] for seq in DEV})
    observations = {name: [] for name in METHODS}
    for record in frames:
        path = RUN / "predictions" / f'{record["sequence"]}_{record["index"]:03d}.npz'
        with np.load(path, allow_pickle=False) as maps:
            for name in METHODS:
                observations[name].append(peaks(maps[name]))
    result = {
        "methods": {name: evaluate_method(frames, observations[name], subsets) for name in METHODS},
        "frames": [{"sequence": f["sequence"], "index": f["index"], "targets": len(f["targets"])} for f in frames],
        "threshold_protocol": "FROC swept on each reported subset with tied scores grouped; separate fixed logit >= 0.0",
        "matching": "5x5 local maxima; 2px Chebyshev NMS; maximum-cardinality 1-to-1 to 1px-expanded boxes; ignore-only predictions excluded",
    }
    result["contrasts"] = contrasts(result["methods"], subsets)
    for filename, contents in (("scores.json", json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"),
                               ("RESULTS.md", markdown(result))):
        path = RUN / filename
        temporary = path.with_suffix(".tmp")
        temporary.write_text(contents, encoding="utf-8")
        temporary.replace(path)
    print(RUN / "RESULTS.md", flush=True)


if __name__ == "__main__":
    main()
