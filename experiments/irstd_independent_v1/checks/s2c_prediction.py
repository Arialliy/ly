"""Check all fixed S2c engineering centers at the archived step-400 working point."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ..data.dataset import MaskClipDataset
from ..engine.checkpoints import configure_determinism, read_checkpoint
from ..engine.predict import predict_native
from ..evaluation.s2b_froc import _frame_peaks
from ..models.video_baseline import VideoBaseline
from .check_s0 import atomic_text


CENTERS = range(9, 93)
NATIVE_SIZES = {"Sequence1": (157, 209), "Sequence17": (209, 302)}
ATOL, RTOL = 1e-5, 1e-4
THRESHOLD = 59.03249740600586


def load_json(path):
    return json.loads(path.read_bytes())


def write_json(path, obj):
    atomic_text(path, json.dumps(obj, indent=2, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selection(data, manifest):
    indices = {}
    selected = {s["id"]: s for s in manifest["sequences"] if s["id"] in NATIVE_SIZES}
    lookup = {pair: i for i, pair in enumerate(data.centers)}
    for name, native_hw in NATIVE_SIZES.items():
        entry = selected[name]
        if (entry["split"] != "train" or set(entry["centers"]) != set(CENTERS)
                or tuple(entry["native_shape_hw"]) != native_hw):
            raise ValueError(f"Unexpected split, centers or shape for {name}")
        for center in CENTERS:
            indices[f"{name}/{center:05}"] = lookup[name, center]
    return indices, selected


def verify_source_files(project, sequences, role):
    key = "image" if role == "images" else "mask"
    verified = {}
    for name, source in sequences.items():
        aggregate = hashlib.sha256()
        for row in source["frames"]:
            aggregate.update((digest(project / row[key]) + "\n").encode("ascii"))
        expected = source[key + "_files_sha256"]
        if aggregate.hexdigest() != expected:
            raise ValueError(f"Native {role} do not match their frozen hash for {name}")
        verified[name] = expected
    return verified


def fixed_threshold_sample(sample_id, scores, centroids):
    peaks = _frame_peaks(sample_id, scores, centroids)
    kept = np.flatnonzero(peaks.scores >= THRESHOLD)
    eligible = kept[peaks.distance_sq[kept] <= 9]
    best = int(eligible[peaks.distance_sq[eligible].argmin()]) if eligible.size else None
    xy = peaks.yx[kept, ::-1]
    return {"gt": len(centroids), "tp": int(best is not None), "fp": len(kept) - int(best is not None),
            "matched_peak_xy": peaks.yx[best, ::-1].tolist() if best is not None else None,
            "peak_xy": xy.tolist()}


def totals(rows, mode):
    def counts(group):
        out = Counter()
        for row in group:
            sample = row["fixed_threshold"][mode]
            out.update({"frames": 1, "gt": sample["gt"], "tp": sample["tp"], "fp": sample["fp"],
                        "empty_frames": int(sample["gt"] == 0),
                        "empty_frame_fp": sample["fp"] if sample["gt"] == 0 else 0})
        return dict(out, pd=out["tp"] / out["gt"] if out["gt"] else None,
                    fp_per_frame=out["fp"] / out["frames"])
    return {**counts(rows), "by_sequence": {source: counts([r for r in rows if r["id"].split("/")[0] == source])
                                           for source in NATIVE_SIZES}}


def evaluate_saved(rows, run_dir, dataset, indices):
    for row in rows:
        sample_id = row["id"]
        # Supervised access is confined to this pass after all prediction files have closed.
        _, targets = dataset[indices[sample_id]]
        if (tuple(targets.target_map.shape) != (1, 1, *row["native_hw"])
                or bool(targets.ignore_map.any())):
            raise ValueError(f"{sample_id}: labels do not cover the native score image")
        centroids = [obj.centroid_xy for obj in targets.objects[0]]
        if len(centroids) != int(sample_id.split("/")[0] == "Sequence1"):
            raise ValueError(f"Unexpected native component count on {sample_id}")
        result = {}
        for mode, artifact in row["logits"].items():
            path = run_dir / artifact["file"]
            if artifact["sha256"] != digest(path):
                raise ValueError(f"Stored native scores changed: {path}")
            result[mode] = fixed_threshold_sample(sample_id, np.load(path, allow_pickle=False), centroids)
        result["same_peak_positions"] = (sorted(result["full"]["peak_xy"]) == sorted(result["tiled"]["peak_xy"]))
        result["same_matched_position"] = result["full"]["matched_peak_xy"] == result["tiled"]["matched_peak_xy"]
        row["fixed_threshold"] = result


def run_prediction(project_root, run_dir, device):
    experiment = Path(project_root) / "experiments/irstd_independent_v1"
    report = load_json(experiment / "reports/S2b.json")
    archive = experiment / report["run_dir"]
    checkpoint = archive / report["checkpoint"]["path"]
    snap = load_json(archive / "snapshot_400.json")
    if (report["status"] != "PASS" or snap != report["snapshots"][-1] or snap["step"] != 400
            or snap["froc"]["score_domain"] != "native_fp32_logits"
            or any(b["threshold"] != THRESHOLD for b in snap["froc"]["budgets"].values())):
        raise ValueError("S2b endpoint or frozen raw threshold does not match the execution plan")
    checkpoint_hash = report["checkpoint"]["sha256"]
    state = read_checkpoint(checkpoint, checkpoint_hash)
    if (state["stage"], state["global_step"], state["model_config"]["trainable_parameters"]) != ("S2b", 400, 22921):
        raise ValueError("Incorrect frozen checkpoint")
    manifest_path = experiment / "manifests/nudt_mirsdt_mask17.json"
    if digest(manifest_path) != state["dependencies"]["manifests/nudt_mirsdt_mask17.json"]:
        raise ValueError("Data manifest changed from the fitted checkpoint")
    manifest = load_json(manifest_path)
    data = MaskClipDataset(manifest, project_root, "train")
    indices, selected = selection(data, manifest)
    image_hashes = verify_source_files(project_root, selected, "images")
    configure_determinism()
    model = VideoBaseline().to(device=device, dtype=torch.float32)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    rows = predict_native(model, data, indices, run_dir, device, (64, 64), ATOL, RTOL)
    observed_ids = [row["id"] for row in rows]
    if (len(observed_ids) != 168 or set(observed_ids) != set(indices)
            or any(tuple(row["native_hw"]) != NATIVE_SIZES[row["id"].split("/")[0]] for row in rows)):
        raise ValueError("Prediction population or its native geometry changed")
    mask_hashes = verify_source_files(project_root, selected, "masks")
    evaluate_saved(rows, run_dir, data, indices)
    manifest_output = run_dir / "prediction_manifest.json"
    write_json(manifest_output, {"checkpoint_sha256": checkpoint_hash, "score_domain": "native_fp32_logits",
                                 "frozen_threshold": THRESHOLD, "atol": ATOL, "rtol": RTOL, "frames": rows})
    missed = sum(row["coverage"]["unwritten_pixels"] for row in rows)
    repeated = sum(row["coverage"]["repeated_pixels"] for row in rows)
    pixels = sum(row["native_pixels"] for row in rows)
    bad = [r for r in rows if r["agreement"]["outside_tolerance_pixels"]]
    def worst(key, metric):
        row = max(rows, key=lambda r: r["agreement"][key][metric])
        return {"id": row["id"], **row["agreement"][key]}
    same_full_size = pixels == 8_058_204
    success = same_full_size and missed == 0 and repeated == 0 and not bad
    result = {"status": "PASS" if success else "FAIL", "checkpoint_step": 400, "checkpoint_sha256": checkpoint_hash,
              "manifest": manifest_output.name, "manifest_sha256": digest(manifest_output),
              "train_sources": list(NATIVE_SIZES), "centers_inclusive": [9, 92], "native_frames": len(rows),
              "native_pixels": pixels, "unwritten_pixels": missed, "repeated_pixels": repeated,
              "actual_core_writes": sum(row["coverage"]["core_count"] for row in rows),
              "raw_source_sha256": {"images": image_hashes, "masks": mask_hashes},
              "full_vs_tiled": {"reference": "full", "atol": ATOL, "rtol": RTOL,
                  "outside_tolerance_pixels": sum(r["agreement"]["outside_tolerance_pixels"] for r in bad),
                  "outside_tolerance_frames": len(bad), "outside_tolerance_ids": [r["id"] for r in bad],
                  "worst_abs_error": worst("max_abs_error_location", "abs_error"),
                  "worst_tolerance_ratio": worst("max_tolerance_ratio_location", "tolerance_ratio")},
              "frozen_threshold": {"raw_logit": THRESHOLD, "source": "S2b step 400",
                  "full": totals(rows, "full"), "tiled": totals(rows, "tiled"),
                  "changed_peak_position_ids": [r["id"] for r in rows if not r["fixed_threshold"]["same_peak_positions"]],
                  "changed_match_ids": [r["id"] for r in rows if not r["fixed_threshold"]["same_matched_position"]]}}
    write_json(run_dir / "prediction.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    report = run_prediction(Path(__file__).resolve().parents[3], args.run_dir.resolve(), torch.device(args.device))
    print(f"S2c prediction: {report['status']}, {report['native_frames']} centers, "
          f"{report['full_vs_tiled']['outside_tolerance_pixels']} pixels outside tolerance.", flush=True)
    raise SystemExit(0 if report["status"] == "PASS" else 1)
