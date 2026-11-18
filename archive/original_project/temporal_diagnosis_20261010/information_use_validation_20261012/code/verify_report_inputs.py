"""Independent stdlib checks of frozen bindings and reported set/ranking counts."""
from pathlib import Path
import collections
import csv
import hashlib
import json
import math

N = Path(__file__).resolve().parents[1]
checks = []


def check(name, condition):
    checks.append({"name": name, "passed": bool(condition)})
    if not condition:
        raise AssertionError(name)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(name):
    with (N / name).open() as f:
        return list(csv.DictReader(f))


for name in ["A_FREEZE.json", "A_CPU_FREEZE.json", "B_FREEZE.json"]:
    freeze = json.loads((N / name).read_text())
    for relative, digest in freeze.get("code_sha256", {}).items():
        check(name + ":" + relative, sha(N / relative) == digest)
    if name == "A_FREEZE.json":
        check("A protocol", sha(N / "A_PROTOCOL.json") == freeze["protocol_sha256"])
        check("A regions", sha(N / "A_REGIONS.csv") == freeze["regions_sha256"])
    if name == "B_FREEZE.json":
        check("B protocol", sha(N / "probe_protocol.md") == freeze["protocol_sha256"])
        check("B samples", sha(N / "geometry/SAMPLE_PROPOSAL.json") == freeze["samples_sha256"])

for sequence in ["DJI_0026_2", "DJI_0067_2"]:
    folder = N / "onoff" / sequence
    receipt = json.loads((folder / "COMPLETE.json").read_text())
    check(sequence + ": unchanged state", receipt["model_state_sha256"] == receipt["unchanged_final_model_state_sha256"])
    check(sequence + ": 8 exact on maps", len(receipt["on_cache_checks"]) == 8 and all(x["old_on_probabilities_bitwise_identical"] for x in receipt["on_cache_checks"]))
    for filename, digest in receipt["window_sha256"].items():
        check(sequence + ":" + filename, sha(folder / "windows" / filename) == digest)

objects = read("A_DETECTION_OBJECTS.csv")
check("all 10400 target arrangement rows", len(objects) == 10400)
for row in read("A_DETECTION_SUMMARY.csv"):
    selected = [r for r in objects if r["arrangement"] == row["arrangement"] and (row["group"] == "pooled" or r["sequence"] == row["group"])]
    on = {(r["sequence"], r["annotation_id"]) for r in selected if int(r["on_matched"])}
    off = {(r["sequence"], r["annotation_id"]) for r in selected if int(r["off_matched"])}
    for key, count in [("M_on", len(on)), ("M_off", len(off)), ("G_on_vs_off", len(on - off)), ("L_on_vs_off", len(off - on))]:
        check(row["group"] + row["arrangement"] + key, int(row[key]) == count)
    check(row["group"] + row["arrangement"] + "net", int(row["net_on_vs_off"]) == len(on) - len(off))
    if row["group"] == "pooled":
        check(row["arrangement"] + "frozen low denominator", sum(int(r["low_all_05"]) for r in selected) == 4342)
        check(row["arrangement"] + "frozen protected denominator", sum(int(r["both_fixed_success"]) for r in selected) == 646)

manifest = json.loads((N / "FEATURES_COMPLETE.json").read_text())
check("10 feature-source records", len(manifest["files"]) == 10)
for entry in manifest["files"]:
    check(entry["sequence"] + ": features", sha(entry["path"]) == entry["sha256"])
    check(entry["sequence"] + ": metadata", sha(entry["metadata_path"]) == entry["metadata_sha256"])
    info = json.loads(Path(entry["metadata_path"]).read_text())
    check(entry["sequence"] + ": unchanged feature model", info.get("state_bitwise_unchanged", info.get("no_model_inference", False)))

predictions = read("probe_readout/PREDICTIONS.csv")
check("128 class observations", len(predictions) == 128)
train_sources = {r["sequence"] for r in predictions if r["split"] == "train"}
dev_sources = {r["sequence"] for r in predictions if r["split"] == "dev"}
check("source separation", not train_sources & dev_sources and len(train_sources) == 3 and len(dev_sources) == 2)
for row in read("probe_readout/SUMMARY.csv"):
    selected = [r for r in predictions if r["split"] == row["split"] and (row["group"] == "pooled" or r["sequence"] == row["group"])]
    key = row["condition"] + "_logit"
    positives = [float(r[key]) for r in selected if r["label"] == "1"]
    negatives = [float(r[key]) for r in selected if r["label"] == "0"]
    auc = sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives) / (len(positives) * len(negatives))
    pairs = collections.defaultdict(dict)
    for r in selected:
        pairs[r["sample_index"]][r["label"]] = float(r[key])
    check(str((row["condition"], row["split"], row["group"], "paired classes")), all(set(pair) == {"0", "1"} for pair in pairs.values()))
    ranking = sum((v["1"] > v["0"]) + 0.5 * (v["1"] == v["0"]) for v in pairs.values()) / len(pairs)
    check(str((row["condition"], row["split"], row["group"], "AUROC")), abs(auc - float(row["AUROC"])) < 1e-14)
    check(str((row["condition"], row["split"], row["group"], "paired ranking")), abs(ranking - float(row["paired_positive_above_negative_fraction"])) < 1e-14)
models = json.loads((N / "probe_readout/MODELS.json").read_text())
check("normalizer train-only", models["normalizer_uses_dev"] is False)
check("P0 second block exactly zero", all(w == 0 for w in models["models"]["P0"]["weight"][32:]))
for condition, model in models["models"].items():
    check(condition + ": fixed solver bound", model["optimizer_iterations"] <= 200 and len(model["weight"]) == 64 and model["lambda_l2"] == 0.01)

for relative, expected in [("mutual_temporal_pilot_20261012/REPORT.md", "f5f7c081822b7d6457ebeb9a5182ad4e2169c4410a5a843eea2192cf5d8fa9db"), ("route_closeout_20261012/CLOSEOUT.md", "51fbeb57367bc153ec0df14f469ec7790756ec446dc7c1565a82ae400adcebd6")]:
    check("old negative result unchanged: " + relative, sha(N.parent / relative) == expected)

result = {"status": "PASS", "checks": checks, "check_count": len(checks), "source_sha256": sha(__file__), "scope": "Frozen bindings, same-checkpoint state, raw cache hashes, actual match set differences, feature identities, independently recomputed probe rankings. No new model inference or fitting."}
(N / "REPORT_INPUT_AUDIT.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"status": "PASS", "checks": len(checks)}))
