#!/usr/bin/env python3
"""Plot the fixed 2048-step H/HL result after complete, verified scoring.

Usage from any working directory: python3 /path/to/this/plot_results.py
Only existing package-local CSV/JSON records are read. Missing/incomplete final
results fail before any plot or caption is emitted. No inference is performed.
"""
from pathlib import Path
import csv
import hashlib
import json

EXPERIMENT = Path(__file__).resolve().parents[1]
PACKAGE = EXPERIMENT.parents[1]
OUTPUT = EXPERIMENT / "figures"
FINAL = EXPERIMENT / "evaluation" / "step_2048"
CONDITIONS = ("release", "H", "HL")
PLANS = ("O", "S")
LABELS = ("Release", "H: head", "HL: head + LFP")
FIELDS = ("M", "unmatched_components", "low_all_05_matched", "L_vs_release_same")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path):
    return {"path": str(path.relative_to(PACKAGE)), "sha256": sha(path)}


def load_inputs():
    summary_path = FINAL / "SUMMARY.csv"
    complete_path = FINAL / "SUMMARY_COMPLETE.json"
    required = [summary_path, complete_path]
    required += [EXPERIMENT / f"training/{c}/run/RUN.json" for c in ("H", "HL")]
    missing = [str(p.relative_to(PACKAGE)) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Final result inputs are not complete; no figure generated:\n" + "\n".join(missing))
    complete = json.loads(complete_path.read_text())
    assert complete["status"] == "COMPLETE" and complete["is_main"]
    assert complete["step"] == complete["main_step"] == 2048
    assert not complete["preflight"]
    assert set(complete["conditions"]) == set(CONDITIONS)
    assert set(complete["plans"]) == set(PLANS)
    assert complete["output_sha256"]["SUMMARY.csv"] == sha(summary_path)
    with summary_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    table = {(r["condition"], r["arrangement"]): r for r in rows}
    assert len(rows) == len(table) == len(CONDITIONS) * len(PLANS)
    assert set(table) == {(c, p) for c in CONDITIONS for p in PLANS}
    for r in rows:
        assert r["group"] == "pooled" and int(r["is_main"]) == 1 and int(r["step"]) == complete["step"]
        assert float(r["threshold"]) == complete["threshold"]
        assert int(r["small_gt"]) == complete["strict_small_target_frames"]
        assert int(r["small_tracks"]) == complete["strict_small_tracks"]
        assert int(r["clips"]) == complete["development_clips"]
        assert int(r["low_all_05_gt"]) == complete["cohort_counts"]["low_all_05"]
        for field in FIELDS:
            assert int(r[field]) >= 0
        assert int(r["M"]) == int(r["release_M_same"]) + int(r["G_vs_release_same"]) - int(r["L_vs_release_same"])
    assert len({r["U_budget"] for r in rows}) == 1
    runs = {c: json.loads((EXPERIMENT / f"training/{c}/run/RUN.json").read_text()) for c in ("H", "HL")}
    for run in runs.values():
        assert run["status"] == "complete" and run["final_step"] == complete["step"]
    assert runs["H"]["seed"] == runs["HL"]["seed"]
    return table, complete, runs, required


def main():
    table, complete, runs, required = load_inputs()
    # Import plotting libraries only after confirming final result availability.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"],
        "font.size": 10, "axes.labelsize": 11, "xtick.labelsize": 9,
        "ytick.labelsize": 10, "legend.fontsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 110, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": .12,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    colors = {"O": "#0072B2", "S": "#D55E00"}
    hatches = {"O": "", "S": "//"}
    low_n = complete["cohort_counts"]["low_all_05"]
    xlabel = ["Matched strict-small target-frames, M (count)",
              "Unmatched components, U (count)",
              f"Recovered low-response target-frames (of {low_n:,})",
              "Lost strict-small matches vs release (count)"]
    budget = int(table[("release", "O")]["U_budget"])
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 6.8))
    y = np.arange(len(CONDITIONS)); height = .32
    for i, (ax, field, label) in enumerate(zip(axes.flat, FIELDS, xlabel)):
        largest = max(int(r[field]) for r in table.values())
        reference_extent = max(largest, budget if field == "unmatched_components" else 0, 1)
        ax.set_xlim(0, reference_extent * 1.19)
        for p, shift in (("O", -height/2), ("S", height/2)):
            values = [int(table[(c,p)][field]) for c in CONDITIONS]
            ax.barh(y+shift, values, height, color=colors[p], edgecolor="white",
                    hatch=hatches[p], linewidth=.5, zorder=2)
            for yy, value in zip(y+shift, values):
                inside = value > reference_extent * .28
                opts = dict(color="white", weight="bold", bbox=dict(facecolor=colors[p], edgecolor="none", pad=.3)) if inside else dict(color="#222222")
                ax.annotate(f"{value:,}", (value,yy), xytext=(-5 if inside else 5,0),
                            textcoords="offset points", ha="right" if inside else "left",
                            va="center", fontsize=9, zorder=6, **opts)
        ax.set_yticks(y, LABELS)
        ax.invert_yaxis()
        ax.set_xlabel(label)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        ax.grid(axis="x", color="#dddddd", linewidth=.45)
        ax.set_axisbelow(True)
        ax.text(-.015, 1.045, f"({'abcd'[i]})", transform=ax.transAxes, weight="bold", fontsize=11)
        if field == "unmatched_components":
            ax.axvline(budget, color="#333333", linestyle="--", linewidth=1, zorder=4)
    legend = [Patch(facecolor=colors["O"], label="O: original five windows"),
              Patch(facecolor=colors["S"], hatch="//", edgecolor="white", label="S: symmetric five windows"),
              Line2D([0],[0],color="#333333",linestyle="--",label=f"U budget = {budget:,}")]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.56,1.025), ncol=3, frameon=False)
    fig.subplots_adjust(left=.15, right=.99, top=.94, bottom=.10, wspace=.50, hspace=.45)
    for ext in ("png", "pdf"):
        fig.savefig(OUTPUT / f"final_results.{ext}")
    plt.close(fig)
    seed = runs["H"]["seed"]
    caption = f'''# H / HL 固定终点结果图

![固定终点结果](final_results.png)

[矢量 PDF](final_results.pdf) · [数据来源和哈希](final_results.sources.json)

本图只读取最终 `evaluation/step_2048/SUMMARY.csv`，并核对同目录完成记录中的 SHA256。
H 为只训练原预测头；HL 为相同原预测头加空间 LFP。二者均为 seed {seed}，主终点固定 {complete["step"]} 步。
H/HL 分别有 {runs["H"]["trainable_parameters"]}/{runs["HL"]["trainable_parameters"]} 个可训练参数；原网络非预测头参数和全部 BN 仿射参数、运行统计保持冻结。
数据为 {complete["development_clips"]} 个开发片段、{complete["strict_small_tracks"]} 条轨迹、{complete["strict_small_target_frames"]:,} 个冻结严格小目标帧。
O/S 分别表示原五窗与对称五窗安排，阈值为 `>= {complete["threshold"]}`，全部坐标从零开始。
仅有单个 seed，因此不绘制误差条，不主张统计显著性或独立来源泛化。

四个面板依次表示：

- (a) 严格小目标实际匹配数 M。预测二值组件的最大响应点与有效标注框完成一对一匹配后，统计冻结严格小目标集合中的成功对象。
- (b) 全部有效标注参与匹配后，剩余的未匹配组件 U。虚线为冻结诊断预算 {budget:,}，不是官方 Fa。
- (c) 冻结的 {low_n:,} 个“原模型在全部已测窗口中框响应低于 0.5”的目标帧中，当前配置实际匹配成功的数量。
- (d) 相对同一安排发布权重的严格小目标新增丢失，来自对象集合差，不是匹配总数差。该图不涵盖其他尺度对象的保持性。

| 配置 | 安排 | M | U | 低响应恢复 | 相对发布新增丢失 |
| --- | --- | ---: | ---: | ---: | ---: |
'''
    for c in CONDITIONS:
        for p in PLANS:
            r=table[(c,p)]
            caption += f'| {c} | {p} | {r["M"]} | {r["unmatched_components"]} | {r["low_all_05_matched"]} | {r["L_vs_release_same"]} |\n'
    caption += '''
检测数上升、背景预算通过和原成功保持是不同要求，不能仅凭一个面板判定方法成功。
本图不替代框外前景像素、未匹配组件面积、保护集合、轨迹连续漏检及全部尺度对象的完整评价。
LFP 的独立贡献应比较 HL 与 H；相对发布权重的变化同时包含预测头适配的影响。
M/U 都是开发诊断计数，不是论文 Pd/Fa。

复现：在包含 Matplotlib 和 NumPy 的环境运行 `python3 experiments/head_lfp_joint_20261012/figures/plot_results.py`。
脚本通过自身位置读取包内结果，移动整个整理目录后仍可定位；不访问原项目宿主机路径。
未完成或哈希不一致的主终点结果会报错，不会输出占位图。
'''
    (OUTPUT / "CAPTION.md").write_text(caption)
    inputs = required + [EXPERIMENT / "code/summarize_joint.py", EXPERIMENT / "PROTOCOL.md"]
    provenance = {
        "figure": "final_results", "source_files": [record(p) for p in inputs],
        "script": record(Path(__file__).resolve()), "plotted_fields": list(FIELDS),
        "conditions": list(CONDITIONS), "arrangements": list(PLANS),
        "main_step": complete["step"], "seed": seed,
        "raster_dpi": 300, "pdf_contains_vector_text_and_geometry": True,
        "all_inputs_package_relative": True, "axes_start_at_zero": True,
        "no_new_training_inference_or_scoring": True,
        "rows": [table[(c,p)] for c in CONDITIONS for p in PLANS],
        "outputs": [record(OUTPUT / f"final_results.{ext}") for ext in ("png", "pdf")],
    }
    (OUTPUT / "final_results.sources.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"status":"COMPLETE", "figure":str(OUTPUT/"final_results.png"), "main_step":complete["step"]}))


if __name__ == "__main__":
    main()
