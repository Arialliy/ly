"""Compare frozen motion adapters on held-out thermal traffic videos."""
import json
from pathlib import Path
import sys

import numpy as np

from locked_protocol import CENTERS, COHORT, OLD_RUN, RUN, frame_record

# Reuse the exact peak, target matching, tie grouping and curve conventions.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "motion_joint_factorial"))
from evaluate import BUDGETS, CONTRASTS, METHODS, at_threshold, froc, label_candidates, peaks

SCENES = {"DJI_0075_3": "道路", "DJI_0139_2": "道路", "DJI_0343_3": "水面"}
POOLED = "all_confirmation"
LOCKED = "fixed_development_primary"
P05 = "fixed_probability_0.5"
POOLED_OP = "confirmation_pooled_primary"


def score_model(frames, observations, groups, locked_threshold):
    xy = np.concatenate([pair[0] for pair in observations])
    scores = np.concatenate([pair[1] for pair in observations])
    frame_index = np.repeat(np.arange(len(frames)), [len(pair[1]) for pair in observations])
    labels, neighbors = label_candidates(xy, frame_index, frames)
    out = {name: froc(scores, ids, frames, frame_index, labels, neighbors) for name, ids in groups.items()}
    primary = out[POOLED]["froc"]["1"]
    for key, t in ((LOCKED, locked_threshold), (P05, 0.0), (POOLED_OP, primary["logit_threshold"])):
        counts = at_threshold(t, scores, frame_index, labels, neighbors, frames, groups)
        for name, row in counts.items():
            out[name][key] = row
        for field in ("tp", "fp", "fn"):
            if sum(counts[name][field] for name in groups if name != POOLED) != counts[POOLED][field]:
                raise AssertionError(f"Source totals disagree at {key}: {field}")
        if key == POOLED_OP and any(counts[POOLED][field] != primary[field] for field in ("tp", "fp")):
            raise AssertionError("Curve endpoint failed reconstruction from pooled threshold")
    return out


def paired_effects(models, groups):
    out = {group: {} for group in groups}
    for group in groups:
        for name, weights in CONTRASTS.items():
            rows = [(models[m][group], mult) for m, mult in weights.items()]
            curve = {b: sum(row["froc"][b]["tp"]*mult for row, mult in rows) for b in BUDGETS}
            fixed = {k: {field: sum(row[k][field]*mult for row, mult in rows) for field in ("tp","fp")}
                     for k in (LOCKED, P05, POOLED_OP)}
            out[group][name] = {"delta_tp_at_fp_budgets": curve, **fixed}
    return out


def operating_cell(row):
    return f'{row["tp"]} / {row["fp"]}'


def render(report):
    lines = [
        "# 锁定模型的红外交通视频复核",
        "",
        "A、B 用整段选速度，C、D 分组选择和检验速度。B、D 保留各时段对邻近速度的联合响应。"
        "B−A、D−C 衡量两种选速方式下联合表示的增量。",
        "",
        "锁定阈值取自之前两个开发视频合并时每帧最多 1 个虚警的工作点，每个模型沿用自己开发时确定的 logit 阈值。"
        "移到本批视频后，实际虚警可能超过原预算。固定概率 0.5 对应 logit ≥ 0。",
        "",
        "| 模型 | 从开发集带入的 logit 阈值 |",
        "|---|---:|",
    ]
    for m,t in report["development_thresholds"].items():
        threshold = "无预测" if t is None else f"{t:.6g}"
        lines.append(f"| {m} | {threshold} |")
    lines += ["", "各表中的数对均为检出数 / 虚警数。前三列在所列样本内分别扫阈值，给出各预算下检出数最多的工作点。"
              "单个来源各自挑出的阈值可能不同，其工作点一般不能直接相加。", ""]
    methods = report["methods"]
    for group, deltas in report["contrasts"].items():
        ref = methods["original"][group]
        heading = "所有确认视频合并" if group == POOLED else f"{group}（{SCENES[group]}）"
        lines += [f"## {heading}", "", f'{ref["frames"]} 个中心帧，{ref["targets"]} 个合格逐帧小目标框。', ""]
        if not ref["targets"]:
            lines += ["所评帧没有合格小目标框，召回率无定义（N/A）。扫阈值时程序沿用空检出作为同分基准，"
                      "这类来源的背景虚警请看两个固定阈值的实际计数。", ""]
        lines += ["| 模型 | ≤0.5 虚警/帧 | ≤1 虚警/帧 | ≤5 虚警/帧 | 原开发阈值 | 固定概率 0.5 |",
                  "|---|---:|---:|---:|---:|---:|"]
        for m in METHODS:
            row = methods[m][group]
            cells = [operating_cell(row["froc"][b]) for b in BUDGETS]
            lines.append("| " + " | ".join([m,*cells,operating_cell(row[LOCKED]),operating_cell(row[P05])]) + " |")
        lines += ["", "| 配对 | Δ检出 @ 0.5 | Δ检出 @ 1 | Δ检出 @ 5 | 原开发阈值 Δ检出 / Δ虚警 |",
                  "|---|---:|---:|---:|---:|"]
        for name,delta in deltas.items():
            c = delta["delta_tp_at_fp_budgets"]
            d = delta[LOCKED]
            lines.append("| " + " | ".join([name,*(f"{c[b]:+d}" for b in BUDGETS),f'{d["tp"]:+d} / {d["fp"]:+d}']) + " |")
        lines.append("")
    sources = [group for group in report["contrasts"] if group != POOLED]
    lines += ["## 同一确认阈值下的来源分解", "",
              "下表对每个模型沿用确认视频合并后每帧最多 1 个虚警所对应的同一阈值，各来源的两项计数均可相加。",
              "", "| 模型 | " + " | ".join([*sources,"合并"]) + " |", "|---|"+"---:|"*(len(sources)+1)]
    for m in METHODS:
        cells = [operating_cell(methods[m][group][POOLED_OP]) for group in [*sources,POOLED]]
        lines.append("| " + " | ".join([m,*cells]) + " |")
    lines += ["", "沿用前一阶段的 5×5 局部峰、同分决胜和一对一框匹配规则，忽略只落入忽略框的预测，"
              "重复命中算虚警。不同帧上的同一实例可能多次计入框数。结果描述本批视频，"
              "这些帧和框之间存在相关性，不能作为独立重复来推断显著性。", ""]
    return "\n".join(lines)


def main():
    old_path = OLD_RUN / "scores.json"
    old = json.loads(old_path.read_text())
    thresholds = {m: old["methods"][m]["all_dev"]["froc"]["1"]["logit_threshold"] for m in METHODS}
    lock = json.loads((RUN/"LOCK.json").read_text())
    if thresholds != lock["dev_logit_threshold"]:
        raise ValueError("Development thresholds differ from the pre-inference lock")
    frames = [frame_record(seq,idx) for seq in COHORT for idx in CENTERS]
    groups = {POOLED: list(range(len(frames)))}
    groups.update({seq:[i for i,f in enumerate(frames) if f["sequence"]==seq] for seq in COHORT})
    by_model = {m: [] for m in METHODS}
    for f in frames:
        path = RUN/"predictions"/f'{f["sequence"]}_{f["index"]:03d}.npz'
        with np.load(path,allow_pickle=False) as raw:
            for m in METHODS:
                by_model[m].append(peaks(raw[m]))
    models = {m: score_model(frames,by_model[m],groups,thresholds[m]) for m in METHODS}
    report = {"models_locked": True, "development_thresholds":thresholds,
              "frames":[{k:f[k] for k in ("sequence","index")} | {"targets":len(f["targets"]),"ignore":len(f["ignore"])} for f in frames],
              "methods":models, "contrasts":paired_effects(models,groups),
              "operating_points":"Confirmation per-group FROC budgets 0.5/1/5, old pooled development <=1.0 FP/frame threshold, fixed logit >= 0 and per-model pooled confirmation <=1.0 FP/frame thresholds."}
    for name,contents in (("scores.json",json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n"),
                          ("RESULTS.md",render(report))):
        path = RUN/name
        temp = path.with_suffix(".tmp")
        temp.write_text(contents,encoding="utf-8")
        temp.replace(path)
    print(RUN/"RESULTS.md",flush=True)


if __name__ == "__main__":
    main()
