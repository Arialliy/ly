"""Absolute background-cost changes vs release with the same arrangement."""
from plot_common import *
import numpy as np
from matplotlib.patches import Patch


def main():
    table, src = read_summary()
    configs = [
        ("unmatched_components", "delta_U_vs_release_same", "$\\Delta$ unmatched components (count)"),
        ("positive_pixels_outside_boxes", "delta_A_vs_release_same", "$\\Delta$ foreground outside boxes (pixels)"),
        ("unmatched_component_area_sum", "delta_unmatched_component_area_vs_release", "$\\Delta$ unmatched-component area (pixels)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), sharey=True)
    y = np.arange(len(CONDITIONS)); height=.32
    for j, (ax, (field, deltafield, label)) in enumerate(zip(axes, configs)):
        vals=[]
        for a, shift, color, hatch in [("O", -height/2, COLORS[0], ""),
                                       ("S", height/2, COLORS[1], "//")]:
            d=[int(table[(c,a)][field])-int(table[("release",a)][field]) for c in CONDITIONS]
            assert d == [int(table[(c,a)][deltafield]) for c in CONDITIONS]
            vals.extend(d)
            ax.barh(y+shift, d, height, color=color, hatch=hatch, edgecolor="white", linewidth=.5, zorder=2)
            for yy, v in zip(y+shift,d):
                ax.annotate(f"{v:+,}" if v else "0", (v,yy), xytext=(-5 if v<0 else 5,0),
                            textcoords="offset points", va="center", ha="right" if v<0 else "left", fontsize=9)
        span=max(max(abs(v) for v in vals),1)
        ax.set_xlim(min(0,min(vals))-.32*span, max(0,max(vals))+.20*span)
        ax.axvline(0,color="#333333",lw=1,zorder=3)
        ax.set_yticks(y,LABELS)
        ax.set_xlabel(label)
        panel(ax,"abc"[j])
    axes[0].invert_yaxis()
    fig.legend(handles=[Patch(facecolor=COLORS[0],label="O: original arrangement"),
                        Patch(facecolor=COLORS[1],hatch="//",edgecolor="white",label="S: symmetric arrangement")],
               loc="upper center",bbox_to_anchor=(.56,1.065),ncol=2,frameon=False)
    fig.subplots_adjust(left=.14,right=.995,bottom=.15,top=.94,wspace=.16)
    finish(fig,"background_deltas",[src],{
        "plotted_fields":[x[0] for x in configs],
        "operation":"condition minus release in same arrangement; absolute changes, not percentages",
        "negative_means_lower_recorded_background_cost":True,
        "panels_have_separate_pixel_or_count_scales":True,
        "not_evidence_of_target_recovery":True,
        "release_reference_values":{a:{x[0]:int(table[("release",a)][x[0]]) for x in configs} for a in ["O","S"]},
    })


if __name__ == "__main__":
    main()
