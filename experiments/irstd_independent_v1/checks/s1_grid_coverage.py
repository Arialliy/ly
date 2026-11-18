"""Select an S1 grid from frozen synthetic training mask centroids, without track-ID labels."""
import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile

import cv2
import numpy as np
import yaml


EXPERIMENT_PATH = Path("experiments/irstd_independent_v1")
OFFSETS = np.arange(-8, 9, dtype=np.float64)
CONTEXT = OFFSETS != 0
COUNT_KEYS = ("all_centers", "fitted", "center_empty", "center_multiple", "context_empty", "context_multiple")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class Component:
    count: int
    xy: np.ndarray | None
    wh: np.ndarray | None


def component(mask):
    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n != 2:
        return Component(n - 1, None, None)
    return Component(1, centroids[1], stats[1, [cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT]])


def read_components(root, source):
    """Authenticate all training mask bytes with the sequence digest frozen and verified by S0."""
    by_id = {}
    digest = hashlib.sha256()
    sequence = source["id"]
    for frame in sorted(source["frames"], key=lambda item: item["number"]):
        number = frame["number"]
        relative = Path("data/public/NUDT-MIRSDT", sequence, "masks", f"{number:05d}.png")
        require(frame["mask"] == relative.as_posix() and number not in by_id, f"Invalid mask entry: {relative}")
        raw = (root / relative).read_bytes()
        digest.update((sha(raw) + "\n").encode("ascii"))
        mask = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
        require(mask is not None and mask.dtype == np.uint8 and mask.shape == tuple(source["native_shape_hw"]),
                f"Native mask shape/type mismatch at {relative}")
        require(not np.any((mask != 0) & (mask != 255)), f"Non-binary mask at {relative}")
        by_id[number] = component(mask)
    require(digest.hexdigest() == source["mask_files_sha256"], f"Mask content differs from S0: {sequence}")
    return by_id


def exclusion(window):
    center = window[8]
    if center.count != 1:
        return "center_empty" if center.count == 0 else "center_multiple"
    if any(item.count == 0 for item in window):
        return "context_empty"
    if any(item.count != 1 for item in window):
        return "context_multiple"
    return None


def fit_velocity(displacements):
    """For each window minimize sum ||delta_xy(t) - t v||^2 at a fixed measured center."""
    return np.einsum("t,ntc->nc", OFFSETS, displacements) / np.dot(OFFSETS, OFFSETS)


def summarize(data):
    array = np.asarray(data, dtype=np.float64)
    if not array.size:
        return {"n": 0}
    q = np.percentile(array, [0, 50, 90, 95, 100])
    return {"n": int(array.size), **dict(zip(("min", "median", "p90", "p95", "max"), map(float, q)))}


def measure(velocity, size, radius, step, spec):
    grid_velocity = np.clip(step * np.rint(velocity / step), -radius, radius)
    in_range = (np.abs(velocity) <= radius).all(axis=1)
    endpoint_error = spec["endpoint_offset"] * np.abs(grid_velocity - velocity)
    quantized = (endpoint_error <= spec["max_box_fraction_per_axis"] * size).all(axis=1)
    return grid_velocity, in_range, quantized


def coverage_counts(in_range, quantized):
    n = in_range.size
    return {
        "range_count": int(in_range.sum()), "quantization_count": int(quantized.sum()),
        "range_fraction": float(in_range.mean()) if n else None,
        "quantization_fraction": float(quantized.mean()) if n else None,
    }


def model_errors(displacements, approximation, size):
    delta = displacements[:, CONTEXT] - OFFSETS[None, CONTEXT, None] * approximation[:, None]
    distances = np.linalg.norm(delta, axis=-1)
    normalized = np.abs(delta) / size[:, None, :]
    per_context = normalized.max(axis=-1)
    return {
        "all_context_euclidean_px": summarize(distances),
        "window_p90_euclidean_px": summarize(np.percentile(distances, 90, axis=1)),
        "window_p90_max_axis_error_in_box_units": summarize(np.percentile(per_context, 90, axis=1)),
        "all_contexts_within_half_box_both_axes_fraction": float((per_context <= 0.5).all(axis=1).mean()),
    }


def self_check():
    start = np.array([7.25, 16.5])
    expected = np.array([[1.25, -0.5], [-3.0, 0.0]])
    coordinates = start + OFFSETS[None, :, None] * expected[:, None, :]
    observed = fit_velocity(coordinates - coordinates[:, 8:9, :])
    np.testing.assert_allclose(observed, expected, atol=1e-12, rtol=0)
    spec = {"endpoint_offset": 8, "max_box_fraction_per_axis": 0.5}
    nearest, within, ok = measure(observed, np.array([[4, 8], [8, 8]]), 2, 0.5, spec)
    np.testing.assert_array_equal(nearest, [[1.0, -0.5], [-2.0, 0.0]])
    np.testing.assert_array_equal(within, [True, False])
    np.testing.assert_array_equal(ok, [True, False])
    mask = np.zeros((5, 7), np.uint8)
    empty = component(mask)
    mask[1, 2] = mask[2, 3] = 255
    positive = component(mask)
    require(positive.count == 1 and empty.count == 0, "Expected diagonal 8-connectivity")
    np.testing.assert_array_equal(positive.xy, (2.5, 1.5))
    np.testing.assert_array_equal(positive.wh, (2, 2))
    window = [positive] * 17
    window[7] = empty
    require(exclusion(window) == "context_empty", "Missing context not counted")
    window[8] = empty
    require(exclusion(window) == "center_empty", "Empty center must take precedence")
    mask[4, 6] = 255
    window[8] = component(mask)
    require(exclusion(window) == "center_multiple", "Ambiguous center was usable")
    window[8], window[7] = positive, component(mask)
    require(exclusion(window) == "context_multiple", "Ambiguous context was usable")
    return ["anchored_signed_fit", "nearest_grid_boundary_and_clipping", "diagonal_component_geometry",
            "empty_and_multiple_exclusions"]


def load_protocol(experiment):
    raw = (experiment / "configs/s1_grid_protocol.yaml").read_bytes()
    policy = yaml.safe_load(raw)
    candidates, selection = policy["candidates"], policy["selection"]
    require(policy["population"]["split"] == "train" and candidates["radii"] == [2, 3, 4, 6]
            and candidates["steps"] == [1.0, 0.5], "Unexpected population or grid family")
    required = {"min_range_fraction": 0.95, "min_quantization_fraction": 0.90,
                "endpoint_offset": 8, "max_box_fraction_per_axis": 0.5}
    require(all(selection.get(k) == v for k, v in required.items()), "Unrecognized coverage policy")
    return policy, sha(raw)


def analyze_grid_coverage(project_root):
    root = Path(project_root).resolve()
    experiment = root / EXPERIMENT_PATH
    policy, protocol_hash = load_protocol(experiment)
    tests = self_check()
    report_raw = (experiment / "reports/S0.json").read_bytes()
    s0 = json.loads(report_raw)
    raw = (experiment / "manifests/nudt_mirsdt_mask17.json").read_bytes()
    manifest = json.loads(raw)
    require(s0["stage"] == "S0" and s0["status"] == "PASS" and s0["data_manifest_hash"] == sha(raw),
            "No matching S0 PASS for this exact manifest")
    require(manifest["dataset"] == "NUDT-MIRSDT" and manifest["clip_length"] == 17 and
            manifest["center_index"] == 8 and manifest["seed"] == 46, "Unsupported S0 data contract")
    sources, index, displacements, sizes = {}, [], [], []
    mask_total = 0
    for source in manifest["sequences"]:
        if source["split"] != "train":
            continue
        name = source["id"]
        require(source["official_split"] == "train" and name not in sources, f"Invalid training source {name}")
        by_id = read_components(root, source)
        mask_total += len(by_id)
        counts = dict.fromkeys(COUNT_KEYS, 0)
        centers = source["centers"]
        require(len(set(centers)) == len(centers), f"Duplicate centers in {name}")
        for number in centers:
            window = [by_id[n] for n in range(number - 8, number + 9)]
            counts["all_centers"] += 1
            reason = exclusion(window)
            if reason is not None:
                counts[reason] += 1
                continue
            counts["fitted"] += 1
            coordinates = np.stack([item.xy for item in window])
            displacements.append(coordinates - coordinates[8])
            sizes.append(window[8].wh)
            index.append(name)
        require(sum(counts[key] for key in COUNT_KEYS[1:]) == counts["all_centers"], "Unaccounted windows")
        sources[name] = {"counts": counts, "mask_files_sha256": source["mask_files_sha256"]}
    totals = {key: sum(row["counts"][key] for row in sources.values()) for key in COUNT_KEYS}
    expected = manifest["counts"]["train"]
    require((len(sources), mask_total, totals["all_centers"]) ==
            (expected["sources"], expected["listed_frames"], expected["eligible_centers"]), "Incomplete training set")
    positions = np.array(displacements, dtype=np.float64).reshape(-1, 17, 2)
    wh = np.array(sizes, dtype=np.float64).reshape(-1, 2)
    index = np.array(index, dtype=str)
    velocities = fit_velocity(positions)
    candidates = []
    selection = policy["selection"]
    for radius in policy["candidates"]["radii"]:
        for step in policy["candidates"]["steps"]:
            _, in_range, quantized = measure(velocities, wh, radius, step, selection)
            counts = coverage_counts(in_range, quantized)
            n = totals["fitted"]
            passes = bool(n and counts["range_count"] >= selection["min_range_fraction"] * n and
                          counts["quantization_count"] >= selection["min_quantization_fraction"] * n)
            candidates.append({"radius": radius, "step": step, "hypotheses": int(1 + 2*radius/step)**2,
                               **counts, "passes": passes})
    selected = min((row for row in candidates if row["passes"]),
                   key=lambda r: (r["hypotheses"], r["radius"], r["step"]), default=None)
    diagnostics = {}
    if totals["fitted"]:
        diagnostics.update(
            width_px=summarize(wh[:, 0]), height_px=summarize(wh[:, 1]), abs_vx=summarize(abs(velocities[:, 0])),
            abs_vy=summarize(abs(velocities[:, 1])), continuous_fit=model_errors(positions, velocities, wh))
    macro = {}
    if selected is not None:
        radius, step = selected["radius"], selected["step"]
        axis = np.arange(-radius, radius + step / 2, step, dtype=np.float64).tolist()
        selected = {**selected, "vx": axis, "vy": axis}
        nearest, in_range, quantized = measure(velocities, wh, radius, step, selection)
        diagnostics["selected_grid_vs_observed_centroids"] = model_errors(positions, nearest, wh)
        for name, row in sources.items():
            subset = index == name
            row["selected"] = coverage_counts(in_range[subset], quantized[subset])
        for metric in ("range_fraction", "quantization_fraction"):
            values = [row["selected"][metric] for row in sources.values() if row["counts"]["fitted"]]
            macro[metric] = {"mean": float(np.mean(values)), **summarize(values)}
    return {
        "stage": "S1", "split": "train", "status": "PASS" if selected else "BLOCKED",
        "dataset": "NUDT-MIRSDT", "training_steps": 0, "data_manifest_sha256": sha(raw),
        "diagnostic_code_sha256": sha(Path(__file__).read_bytes()), "protocol_sha256": protocol_hash,
        "policy": policy, "policy_sha256": sha(json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()),
        "s0_report_sha256": sha(report_raw), "mask_frames_authenticated": mask_total,
        "counts": totals, "sources": sources, "candidates": candidates, "selected": selected,
        "source_macro_selected": macro, "fit_diagnostics": diagnostics, "tests": tests,
        "interpretation": [
            "Range and endpoint quantization fractions use all fitted windows, including outside-grid motion.",
            "Coverage of proxy velocities and quantization around fitted lines is conditional on 17 single-component masks.",
            "Counts retain all eligible training centers. Exclusions prioritize center status, then empty context, then multiple context.",
            "Fixed-center least squares uses equally weighted physical frame times and native mask centroids. Identity and velocity ground truth are absent.",
            "Residuals use the 16 noncenter times. Box units divide x and y errors by center-mask width and height respectively.",
            "Nearest-grid ties use round-to-even. The box criterion includes equality. A missing or fragmented mask precludes fitting.",
            "Neighboring windows overlap. Source-macro means weight each fitted source equally and omit sources with zero fitted windows.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze_grid_coverage(args.project_root)
    output = args.output or (args.project_root / EXPERIMENT_PATH / "reports/S1_grid_coverage.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(dir=output.parent, mode="w", encoding="utf-8", delete=False) as stream:
        tmp = Path(stream.name)
    try:
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, output)
    finally:
        tmp.unlink(missing_ok=True)
    print(json.dumps({"status": result["status"], "counts": result["counts"], "selected": result["selected"]}))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
