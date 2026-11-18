"""Shared plotting style and provenance; all inputs are package-relative."""
from pathlib import Path
import csv
import hashlib
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PACKAGE = Path(__file__).resolve().parents[2]
OUT = PACKAGE / "visualizations"
RESULTS = PACKAGE / "results" / "tlfp_pilot_20261012"
CONDITIONS = ["release", "S", "T", "T_current", "T_bypass"]
LABELS = ["Release", "S-LFP", "T-LFP", "T: current only", "T: bypass"]
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#555555"]
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 10, "axes.labelsize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 10,
    "legend.fontsize": 9, "figure.dpi": 110, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.10,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "stix",
})


def read_summary():
    p = RESULTS / "evaluation" / "SUMMARY.csv"
    with p.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(CONDITIONS) * 2
    table = {(r["condition"], r["arrangement"]): r for r in rows}
    assert len(table) == len(rows)
    assert all(r["group"] == "pooled" for r in rows)
    assert len({r["small_gt"] for r in rows}) == 1
    assert len({r["threshold"] for r in rows}) == 1
    return table, p


def source_record(path):
    return {
        "path": str(path.relative_to(PACKAGE)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def finish(fig, name, sources, detail):
    OUT.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    record = {
        "figure": name,
        "source_files": [source_record(p) for p in sources],
        "script": source_record(Path(__file__).with_name(f"{name}.py")),
        "shared_script": source_record(Path(__file__)),
        "outputs": [f"{name}.png", f"{name}.pdf"],
        "raster_dpi": 300, "pdf_vector_text_and_geometry": True,
        **detail,
    }
    (OUT / f"{name}.sources.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Generated {name}.png and {name}.pdf")


def panel(ax, letter):
    ax.text(-0.015, 1.03, f"({letter})", transform=ax.transAxes,
            ha="left", va="bottom", weight="bold", fontsize=11)
    ax.grid(axis="x", color="#dddddd", linewidth=.45, zorder=0)
    ax.set_axisbelow(True)
