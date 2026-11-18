"""Read existing completed results; no model execution or checkpoint selection."""
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
Q = HERE.parent
EVAL = Q / "supervision_alignment_control_20261012/evaluation"


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def save_csv(name, rows):
    with (HERE / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


source_rows = read_csv(EVAL / "SUMMARY.csv")
reference = {r["arrangement"]: r for r in source_rows if r["condition"] == "release"}
rows = []
for r in source_rows:
    baseline = reference[r["arrangement"]]
    row = {"condition": r["condition"], "step": 0 if r["condition"] == "release" else 512,
           "arrangement": r["arrangement"], "threshold": float(r["threshold"])}
    for name in ("M", "G_vs_release_same", "L_vs_release_same", "unmatched_components",
                 "positive_pixels_outside_boxes", "unmatched_component_area_sum",
                 "low_all_05_matched", "extreme_all_001_matched", "both_fixed_success_matched"):
        row[name] = int(r[name])
    row["U_ratio_to_same_arrangement_release"] = int(r["unmatched_components"]) / int(baseline["unmatched_components"])
    row["U_ratio_to_frozen_4015_budget"] = int(r["unmatched_components"]) / 4015
    row["A_ratio_to_same_arrangement_release"] = int(r["positive_pixels_outside_boxes"]) / int(baseline["positive_pixels_outside_boxes"])
    row["passes_U_budget"] = int(r["unmatched_components"]) <= 4015
    assert row["M"] - int(baseline["M"]) == row["G_vs_release_same"] - row["L_vs_release_same"]
    rows.append(row)
save_csv("ENDPOINT_COMPARISON.csv", rows)

joint = [r for r in read_csv(EVAL / "WINDOW_DEPENDENCE.csv")
         if r["group"] == "pooled" and r["cohort"] == "both_fixed_success"]
retention = []
for r in joint:
    row = {"condition": r["condition"]}
    for name in ("target_frames", "both_matched", "O_only", "S_only", "neither_matched"):
        row[name] = int(r[name])
    row["lost_both_success_status"] = row["target_frames"] - row["both_matched"]
    row["lost_both_success_status_fraction"] = row["lost_both_success_status"] / row["target_frames"]
    assert row["target_frames"] == sum(row[k] for k in ("both_matched", "O_only", "S_only", "neither_matched"))
    retention.append(row)
save_csv("RETENTION_646.csv", retention)

inputs = [EVAL / "SUMMARY.csv", EVAL / "WINDOW_DEPENDENCE.csv"]
checks = {"no_new_training": True, "no_model_forward": True,
          "endpoint_rows": len(rows), "retention_rows": len(retention),
          "gain_loss_identities_passed": True, "retention_partition_passed": True,
          "two_U_ratio_denominators_explicit": True,
          "scope": "Recomputed tables from existing completed results, not an independent rescore of predictions.",
          "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}}
(HERE / "TABLE_CHECK.json").write_text(json.dumps(checks, indent=2) + "\n")
print(json.dumps({"endpoint_rows": len(rows), "retention_rows": len(retention), "checks": "PASS"}))
