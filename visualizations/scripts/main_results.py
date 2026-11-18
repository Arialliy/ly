"""Absolute diagnostic counts; zero-based axes, no error bars for one seed."""
from plot_common import *
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def main():
    table, src = read_summary()
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8), sharey=True)
    y = np.arange(len(CONDITIONS)); height = .32
    fields = ["M", "unmatched_components"]
    labels = ["Matched strict-small target-frames, M (count)",
              "Unmatched components, U (count)"]
    budget = int(table[("release", "O")]["U_budget"])
    for j, (ax, field, xlabel) in enumerate(zip(axes, fields, labels)):
        for a, shift, color, hatch in [("O", -height/2, COLORS[0], ""),
                                       ("S", height/2, COLORS[1], "//")]:
            values = [int(table[(c, a)][field]) for c in CONDITIONS]
            ax.barh(y+shift, values, height, color=color, hatch=hatch,
                    edgecolor="white", linewidth=.5, zorder=2)
            for yy, value in zip(y+shift, values):
                ax.annotate(f"{value:,}", (value, yy), xytext=(-6, 0),
                            textcoords="offset points", ha="right", va="center",
                            fontsize=9, color="white", weight="bold",
                            bbox=dict(facecolor=color,edgecolor="none",pad=.2))
        maxval = max(int(r[field]) for r in table.values())
        ax.set_xlim(0, maxval * 1.10)
        ax.set_xlabel(xlabel)
        ax.set_yticks(y, LABELS)
        panel(ax, "ab"[j])
    axes[0].invert_yaxis()
    axes[1].axvline(budget, linestyle="--", color="#333333", linewidth=1, zorder=4)
    legend = [Patch(facecolor=COLORS[0], label="O: original arrangement"),
              Patch(facecolor=COLORS[1], hatch="//", edgecolor="white", label="S: symmetric arrangement"),
              Line2D([0], [0], color="#333333", linestyle="--", label=f"U budget = {budget:,}")]
    fig.legend(handles=legend, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(.55, 1.065))
    fig.subplots_adjust(left=.155, right=.98, bottom=.15, top=.94, wspace=.10)
    finish(fig, "main_results", [src], {
        "plotted_fields": fields,
        "scope": "pooled over two clips; fixed arrangements; threshold >= 0.5",
        "counts_are_not_paper_Pd_or_Fa": True,
        "one_seed_no_confidence_intervals": True,
        "rows": [table[(c,a)] for c in CONDITIONS for a in ["O","S"]],
    })


if __name__ == "__main__":
    main()
