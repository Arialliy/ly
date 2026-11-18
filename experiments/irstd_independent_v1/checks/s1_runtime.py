"""S1 runtime helpers: archived S0 integrity, fixed TRAIN clips, sampler costs.

These helpers do not change S0 records, select a device, set thread counts,
train a model, or read development/test images. Assertion failures are failures;
unavailable input files/devices are explicitly reported as BLOCKED.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import resource
import sys
from time import perf_counter


_MANIFEST = Path("manifests/nudt_mirsdt_mask17.json")
_REAL_SAMPLES = (("Sequence1", 9), ("Sequence17", 9), ("Sequence2", 9))


def _check(condition, message):
    if not condition:
        raise AssertionError(message)


def _relative_file(root: Path, relative) -> Path:
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Expected a safe relative path: {relative}")
    return root / relative


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_s0(experiment: Path) -> dict:
    """Verify exactly the code/config files and manifest recorded by S0.

    The archived commit is provenance, not an equality gate: new S1 files need
    not have existed at that commit. Existing recorded files must still match.
    This does not reopen/re-audit the released images or change the S0 report.
    Missing or mismatching integrity evidence raises an exception.
    """
    experiment = Path(experiment)
    report = json.loads((experiment / "reports/S0.json").read_text())
    _check(report.get("stage") == "S0", "Expected an S0 stage record")
    _check(report.get("status") == "PASS", "S0 has not passed")
    _check(report.get("next_allowed_stage") == "S1", "S0 does not authorize S1")
    recorded = report.get("code_and_config_sha256")
    _check(isinstance(recorded, dict) and recorded and "contracts.py" in recorded,
           "S0 must record its code/config hashes, including contracts.py")
    checked = []
    for relative, expected in recorded.items():
        path = _relative_file(experiment, relative)
        _check(path.is_file(), f"S0-recorded file is missing: {relative}")
        actual = _sha256(path)
        _check(actual == expected, f"S0-recorded file changed: {relative}")
        checked.append({"path": relative, "sha256": actual})
    manifest_path = experiment / _MANIFEST
    _check(manifest_path.is_file(), f"S0 manifest is missing: {_MANIFEST}")
    manifest_hash = _sha256(manifest_path)
    _check(manifest_hash == report.get("data_manifest_hash"), "S0 manifest hash changed")
    return {
        "status": "PASS", "gate_kind": "archived_integrity",
        "recorded_code_commit": report.get("code_commit"),
        "commit_equality_required": False,
        "verified_code_and_config": checked,
        "verified_file_count": len(checked),
        "data_manifest": str(_MANIFEST), "data_manifest_sha256": manifest_hash,
        "new_files_enumerated": False, "raw_data_reaudited": False,
        "remaining_S0_limits": report.get("remaining_limits", []),
    }


def _validate_selection(coverage, protocol, config, manifest):
    """Reconstruct the selection from integer counts, not claimed PASS fields or rounded rates."""
    counts = coverage["counts"]
    all_sources = {s["id"]: len(s["centers"]) for s in manifest["sequences"] if s["split"] == "train"}
    source_stats = coverage["sources"]
    _check(coverage["split"] == protocol["population"]["split"] == "train"
           and coverage["stage"] == "S1" and coverage["status"] == "PASS",
           "Formal coverage must have passed on the training split")
    _check(set(all_sources) == set(source_stats), "Coverage does not use precisely the frozen training sources")
    categories = ("fitted", "center_empty", "center_multiple", "context_empty", "context_multiple")
    for source, eligible in all_sources.items():
        local = source_stats[source]["counts"]
        _check(all(type(local[k]) is int and local[k] >= 0 for k in categories)
               and sum(local[k] for k in categories) == local["all_centers"] == eligible,
               f"Unaccounted center or invalid fit count for {source}")
    for key in ("all_centers", *categories):
        _check(counts[key] == sum(row["counts"][key] for row in source_stats.values()),
               f"Source-derived training denominator mismatch for {key}")
    fitted = counts["fitted"]
    _check(fitted > 0, "No fitted training windows")
    grid_policy = protocol["candidates"]
    remaining = {(r, s) for r in grid_policy["radii"] for s in grid_policy["steps"]}
    passed = []
    for row in coverage["candidates"]:
        radius, step = row["radius"], row["step"]
        _check((radius, step) in remaining, "Duplicate or unregistered velocity candidate")
        remaining.remove((radius, step))
        axis_points = round(2 * radius / step) + 1
        _check(row["hypotheses"] == axis_points ** 2, "Incorrect candidate size")
        qualifies = True
        for metric in ("range", "quantization"):
            n = row[metric + "_count"]
            _check(type(n) is int and 0 <= n <= fitted
                   and math.isclose(row[metric + "_fraction"], n / fitted, rel_tol=1e-12, abs_tol=1e-12),
                   "Candidate count and fraction disagree")
            qualifies = qualifies and n / fitted >= protocol["selection"]["min_" + metric + "_fraction"]
        _check(row["passes"] is qualifies, "Candidate result differs from prespecified thresholds")
        if qualifies:
            passed.append(row)
    _check(not remaining and passed, "Missing candidate or no grid passes the frozen criteria")
    best = min(passed, key=lambda row: (row["hypotheses"], row["radius"], row["step"]))
    chosen = coverage["selected"]
    _check(isinstance(chosen, dict) and all(chosen[key] == best[key] for key in
           ("radius", "step", "hypotheses", "range_count", "quantization_count")),
           "Reported selection is not the deterministic minimum passing grid")
    points = math.isqrt(best["hypotheses"])
    axis = [-best["radius"] + i * best["step"] for i in range(points)]
    grid = config["velocity_grid"]
    _check(chosen["vx"] == chosen["vy"] == grid["vx"] == grid["vy"] == axis,
           "Formal grid axes differ from the independently reconstructed winner")
    for metric in ("range_count", "quantization_count"):
        _check(best[metric] == sum(row["selected"][metric] for row in source_stats.values()),
               "Selected candidate does not match its per-source counts")
    for key in ("query_hw", "feature_channels", "max_peak_cuda_allocated_mib"):
        _check(config["resource"][key] == protocol["resource"][key], f"Resource protocol mismatch: {key}")
    return dict(training_sources=len(all_sources), all_centers=counts["all_centers"],
                fitted=fitted, selected=best, actual_axes=axis)


def verify_formal_coverage(experiment, config):
    import yaml
    grid = config["velocity_grid"]
    _check(grid["purpose"] == "formal" and grid.get("formal_coverage_approved") is True,
           "Formal execution requires an approved, evidence-backed grid")
    evidence = grid["coverage_evidence"]
    _check(isinstance(evidence, dict), "Formal coverage evidence must contain named paths and digests")
    cov_path = _relative_file(experiment, evidence["path"])
    protocol_path = _relative_file(experiment, evidence["protocol_path"])
    cov_hash, protocol_hash = _sha256(cov_path), _sha256(protocol_path)
    _check((cov_hash, protocol_hash) == (evidence["sha256"], evidence["protocol_sha256"]),
           "Frozen coverage evidence or preregistration hash mismatch")
    coverage = json.loads(cov_path.read_bytes())
    protocol = yaml.safe_load(protocol_path.read_bytes())
    manifest_path = experiment / _MANIFEST
    _check(coverage["policy"] == protocol, "Coverage policy differs from registered YAML contents")
    _check(coverage["s0_report_sha256"] == _sha256(experiment / "reports/S0.json"),
           "Coverage is for a different S0 receipt")
    _check(coverage["data_manifest_sha256"] == _sha256(manifest_path),
           "Coverage receipt is for another manifest")
    _check(coverage["diagnostic_code_sha256"] == _sha256(experiment / "checks/s1_grid_coverage.py"),
           "Coverage diagnostic changed since selection")
    _check(coverage["protocol_sha256"] == protocol_hash, "Evidence is for a different protocol")
    summary = _validate_selection(coverage, protocol, config, json.loads(manifest_path.read_bytes()))
    return dict(status="PASS", evidence=evidence, **summary)


def _manual_bilinear(frames, pixel_valid, shapes, queries, velocities, offsets):
    """Independent scalar interpolation; no sampler/grid_sample/grid normalization.

    Inputs are CPU NumPy arrays. All stored images start at full-image (0,0).
    Positive-weight support must exist; zero-weight neighbours at image edges
    do not invalidate an exact integer sample.
    """
    import numpy as np

    batch, _, channels, _, _ = frames.shape
    count = queries.shape[1]
    my, mx = velocities.shape[:2]
    values = np.zeros((batch, count, channels, 16, my, mx), dtype=np.float64)
    valid = np.zeros((batch, count, 16, my, mx), dtype=bool)
    context_indices = [i for i in range(17) if i != 8]
    for b, (height, width) in enumerate(shapes):
        for q, (px, py) in enumerate(queries[b]):
            if not (0 <= px <= width - 1 and 0 <= py <= height - 1):
                continue
            for t, frame_index in enumerate(context_indices):
                for vy_index in range(my):
                    for vx_index in range(mx):
                        vx, vy = velocities[vy_index, vx_index]
                        x = float(px + offsets[b, t] * vx)
                        y = float(py + offsets[b, t] * vy)
                        if not (0 <= x <= width - 1 and 0 <= y <= height - 1):
                            continue
                        x0, y0 = math.floor(x), math.floor(y)
                        dx, dy = x - x0, y - y0
                        result = np.zeros(channels, dtype=np.float64)
                        usable = True
                        for xx, yy, weight in (
                            (x0, y0, (1-dx)*(1-dy)), (x0+1, y0, dx*(1-dy)),
                            (x0, y0+1, (1-dx)*dy), (x0+1, y0+1, dx*dy),
                        ):
                            if weight == 0:
                                continue
                            if not (0 <= xx < width and 0 <= yy < height
                                    and pixel_valid[b, frame_index, 0, yy, xx]):
                                usable = False
                                break
                            result += weight * frames[b, frame_index, :, yy, xx]
                        if usable:
                            values[b, q, :, t, vy_index, vx_index] = result
                            valid[b, q, t, vy_index, vx_index] = True
    return values, valid


def check_real_batch(experiment: Path, project_root: Path, device: str = "cpu", *,
                     configured_axes=None, query_chunk=256, velocity_chunk=8) -> dict:
    """Check 3 fixed, S0-selected TRAIN clips; never substitute synthetic data.

    Missing manifest, raw images or center masks returns BLOCKED before loading
    any images. Invalid metadata, interpolation or masks raises an exception.
    The loader opens 17 images and one center mask for each selected sample.
    """
    experiment, project_root = Path(experiment), Path(project_root)
    sample_ids = [f"{name}/{center:05d}" for name, center in _REAL_SAMPLES]
    base = {"scope": "fixed_train_clips_only", "sample_ids": sample_ids,
            "device": str(device), "training_steps": 0}
    manifest_path = experiment / _MANIFEST
    if not manifest_path.is_file():
        return dict(base, status="BLOCKED", reason="missing_manifest",
                    missing_files=[str(manifest_path)])
    manifest = json.loads(manifest_path.read_text())
    _check(manifest.get("clip_length") == 17 and manifest.get("center_index") == 8,
           "Expected S0's 17-frame, center-index-8 contract")
    sources = {}
    for source in manifest["sequences"]:
        _check(source["id"] not in sources, "Duplicate source in manifest")
        sources[source["id"]] = source
    required_paths = []
    for name, center in _REAL_SAMPLES:
        _check(name in sources and sources[name]["split"] == "train",
               f"Required real check sample is not TRAIN: {name}")
        source = sources[name]
        _check(center in source["centers"], f"Missing eligible center: {name}/{center}")
        records = {row["number"]: row for row in source["frames"]}
        _check(len(records) == len(source["frames"]), f"Duplicate frame in {name}")
        for number in range(center-8, center+9):
            _check(number in records, f"Missing physical frame record: {name}/{number}")
            required_paths.append(_relative_file(project_root, records[number]["image"]))
        required_paths.append(_relative_file(project_root, records[center]["mask"]))
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        return dict(base, status="BLOCKED", reason="missing_training_images_or_center_masks",
                    required_file_count=len(required_paths), missing_file_count=len(missing),
                    missing_files=missing)

    import torch
    from ..data.dataset import MaskClipDataset, collate_clips
    requested_device = torch.device(device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        return dict(base, status="BLOCKED", reason="requested_cuda_unavailable")
    dataset = MaskClipDataset(manifest, project_root, "train")
    lookup = {key: index for index, key in enumerate(dataset.centers)}
    clip, labels = collate_clips([dataset[lookup[key]] for key in _REAL_SAMPLES])
    _check(list(clip.output_frame_id) == sample_ids, "Unexpected real sample IDs/order")
    _check(clip.frames.shape[:3] == (3, 17, 1) and clip.center_index == 8,
           "Real batch does not satisfy the S0 temporal shape")
    expected_offsets = torch.arange(-8, 9, dtype=torch.float32).repeat(3, 1)
    torch.testing.assert_close(clip.time_offsets, expected_offsets, atol=0, rtol=0)
    shapes = [meta.original_hw for meta in clip.scene_meta]
    _check(len(set(shapes)) > 1, "Real check must exercise heterogeneous padding")
    _check([bool(objects) for objects in labels.objects] == [True, False, True],
           "S0 positive/negative/different-size sample polarity changed")
    queries = []
    for b, (height, width) in enumerate(shapes):
        expected_valid = torch.zeros_like(clip.pixel_valid[b])
        expected_valid[..., :height, :width] = True
        _check(torch.equal(clip.pixel_valid[b], expected_valid), "Padding validity changed")
        _check(torch.equal(labels.ignore_map[b, 0], ~expected_valid[0, 0]),
               "Padding loss-ignore map changed")
        _check(bool((clip.frames[b][~expected_valid] == 0).all()), "Nonzero batch padding")
        queries.append([(width//2, height//2), (0, 0), (width-1, height-1),
                        (width-1, height//2), (width, height//2)])
    queries_cpu = torch.tensor(queries, dtype=torch.float32)
    fixed_axis = [-1.0, -0.5, 0.0, 0.5, 1.0]
    fractional = _compare_real_grid(clip, queries_cpu, fixed_axis, fixed_axis, requested_device, 3, 7)
    report = dict(base, required_file_count=len(required_paths),
                  batch_shape=list(clip.frames.shape), native_shapes_hw=[list(s) for s in shapes],
                  query_count_per_image=5, context_slots=16, **fractional)
    if configured_axes is not None:
        vx, vy = configured_axes
        report["configured_grid"] = _compare_real_grid(
            clip, queries_cpu, vx, vy, requested_device, query_chunk, velocity_chunk)
    return report


def _compare_real_grid(clip, queries_cpu, vx, vy, device, query_chunk, velocity_chunk):
    """Run the actual grid against a reference constructed directly from its supplied axes."""
    import numpy as np
    import torch
    from ..geometry import make_velocity_grid, sample_clip_features

    grid = make_velocity_grid(vx, vy, device=device)
    y_ref, x_ref = np.meshgrid(np.array(vy, dtype=np.float64), np.array(vx, dtype=np.float64),
                              indexing="ij")
    reference_grid = np.stack((x_ref, y_ref), axis=-1)
    features = clip.frames.to(device)
    kwargs = dict(query_chunk=query_chunk, velocity_chunk=velocity_chunk)
    with torch.no_grad():
        queries = queries_cpu.to(device)
        sampled = sample_clip_features(features, clip, queries, grid, **kwargs)
        contaminated = features.masked_fill(~clip.pixel_valid.to(device), 1e4)
        repeated = sample_clip_features(contaminated, clip, queries, grid, **kwargs)
    expected_offsets = torch.tensor([list(range(-8, 0)) + list(range(1, 9))],
                                    dtype=torch.float32).repeat(len(clip.scene_meta), 1)
    torch.testing.assert_close(sampled.time_offsets.cpu(), expected_offsets, atol=0, rtol=0)
    shapes = [scene.original_hw for scene in clip.scene_meta]
    expected, valid = _manual_bilinear(clip.frames.numpy(), clip.pixel_valid.numpy(), shapes,
                                       queries_cpu.numpy(), reference_grid, expected_offsets.numpy())
    got = sampled.values.cpu().numpy()
    np.testing.assert_allclose(got, expected, atol=1e-5, rtol=1e-4)
    np.testing.assert_array_equal(sampled.sample_valid.cpu().numpy(), valid)
    np.testing.assert_array_equal(sampled.velocity_valid.cpu().numpy(), valid.all(axis=2))
    _check(not bool(sampled.sample_valid[:, 4].any()), "Outside-image query became valid")
    _check(bool((sampled.values[:, 4] == 0).all()), "Nonzero values outside image")
    zero_y, zero_x = list(vy).index(0), list(vx).index(0)
    _check(bool(sampled.sample_valid[:, 2:4, :, zero_y, zero_x].all()),
           "Zero-velocity observations on the exact image edge must remain valid")
    torch.testing.assert_close(repeated.values, sampled.values, atol=0, rtol=0)
    _check(torch.equal(repeated.sample_valid, sampled.sample_valid) and
           torch.equal(repeated.velocity_valid, sampled.velocity_valid), "Padding changed validity")
    return dict(status="PASS", vx=list(vx), vy=list(vy),
                velocity_grid_shape=list(grid.shape), **kwargs,
                max_abs_reference_error=float(np.max(np.abs(got - expected))),
                tolerance={"atol": 1e-5, "rtol": 1e-4}, valid_samples=int(valid.sum()),
                invalid_samples=int(valid.size - valid.sum()),
                checks=["independent_scalar_bilinear", "both_validity_masks", "exact_image_edges",
                        "outside_image_is_invalid_zero", "padding_poison_invariance"])


def _rss_high_water_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def profile_sampler(device, velocities_xy, query_hw=(32, 32), channels=9,
                    query_chunk=256, velocity_chunk=8) -> dict:
    """Measure one deterministic sampler forward/backward, never a training step.

    One no-grad warmup precedes one timed forward and one timed backward of
    mean-squared sampled evidence. CUDA timings synchronize the requested
    device. CPU RSS is process-lifetime high-water, not sampler peak or CUDA
    memory. No speed/memory threshold or automatic device selection is applied.
    """
    import torch
    from ..contracts import ClipBatch, SceneMeta
    from ..geometry.sampler import sample_clip_features
    from ..geometry.tiles import dense_queries

    requested_device = torch.device(device)
    if requested_device.type not in ("cpu", "cuda"):
        raise ValueError("S1 profiling supports an explicitly requested cpu or cuda device")
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        return {"status": "BLOCKED", "device": str(requested_device),
                "reason": "requested_cuda_unavailable", "cuda_resource_gate": "UNAVAILABLE"}
    if (len(query_hw) != 2 or any(type(v) is not int or v <= 0 for v in query_hw)
            or type(channels) is not int or channels <= 0):
        raise ValueError("query_hw and channels must be positive integers")
    if any(type(v) is not int or v <= 0 for v in (query_chunk, velocity_chunk)):
        raise ValueError("query_chunk and velocity_chunk must be positive integers")
    velocity = torch.as_tensor(velocities_xy, dtype=torch.float32, device=requested_device)
    if velocity.ndim == 2 and velocity.shape[-1] == 2:
        velocity = velocity[None]
    if (velocity.ndim != 3 or velocity.shape[-1] != 2
            or min(velocity.shape[:2]) <= 0 or not bool(torch.isfinite(velocity).all())):
        raise ValueError("velocities_xy must have finite shape [My,Mx,2] or [V,2]")
    if velocity.requires_grad:
        velocity = velocity.detach()
    max_xy = velocity.abs().reshape(-1, 2).amax(dim=0).cpu().tolist()
    pad_x, pad_y = (math.ceil(8 * value) + 1 for value in max_xy)
    query_h, query_w = query_hw
    height, width = query_h + 2*pad_y, query_w + 2*pad_x
    rss_before = _rss_high_water_bytes()
    # Deterministic, bounded synthetic intensities; global RNG state is untouched.
    count = 17 * channels * height * width
    features = torch.arange(count, dtype=torch.float32, device=requested_device)
    features = features.remainder(251).div(250).reshape(1, 17, channels, height, width)
    features.requires_grad_(True)
    offsets = torch.arange(-8, 9, dtype=torch.float32, device=requested_device)[None]
    pixel_valid = torch.ones((1, 17, 1, height, width), dtype=torch.bool, device=requested_device)
    meta = SceneMeta("S1Synthetic", tuple(range(1, 18)), (height, width),
                     (0, 0), (0, 0, width, height))
    clip = ClipBatch(features[:, :, :1].detach(), offsets, pixel_valid, 8,
                     ("S1Synthetic/00009",), (meta,))
    queries = dense_queries((pad_x, pad_y, pad_x+query_w, pad_y+query_h),
                            device=requested_device, dtype=torch.float32)

    def synchronize():
        if requested_device.type == "cuda":
            torch.cuda.synchronize(requested_device)

    def run():
        return sample_clip_features(features, clip, queries, velocity,
                                    query_chunk=query_chunk, velocity_chunk=velocity_chunk)

    with torch.no_grad():
        warmup = run()
    del warmup
    synchronize()
    allocated_before = reserved_before = None
    if requested_device.type == "cuda":
        allocated_before = torch.cuda.memory_allocated(requested_device)
        reserved_before = torch.cuda.memory_reserved(requested_device)
        torch.cuda.reset_peak_memory_stats(requested_device)
    start = perf_counter()
    sampled = run()
    synchronize()
    forward_seconds = perf_counter() - start
    start = perf_counter()
    sampled.values.square().mean().backward()
    synchronize()
    backward_seconds = perf_counter() - start
    _check(sampled.values.shape[3] == 16, "Profile must use all 16 noncenter contexts")
    _check(bool(sampled.sample_valid.all()), "Profile halo did not support all queries")
    _check(features.grad is not None and bool(torch.isfinite(features.grad).all())
           and bool((features.grad != 0).any()), "Profile feature gradient is missing/invalid")
    _check(bool((features.grad[:, 8] == 0).all()), "Profile center must have zero context gradient")
    cuda = None
    if requested_device.type == "cuda":
        cuda = {
            "allocated_before_measurement_bytes": allocated_before,
            "reserved_before_measurement_bytes": reserved_before,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(requested_device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(requested_device),
            "scope": "allocator peak after warmup, includes retained inputs and timed autograd",
        }

    def tensor_bytes(tensor):
        return int(tensor.numel() * tensor.element_size())

    return {
        "status": "PASS", "scope": "synthetic_sampler_only_not_full_model",
        "device": str(requested_device), "torch_version": torch.__version__,
        "input_shape": list(features.shape), "query_hw": list(query_hw),
        "velocity_grid_shape": list(velocity.shape), "context_slots": 16,
        "query_chunk": query_chunk, "velocity_chunk": velocity_chunk,
        "warmup_forward_iterations": 1, "measured_iterations": 1,
        "forward_seconds": forward_seconds, "backward_seconds": backward_seconds,
        "backward_objective": "mean_squared_sampled_evidence; no optimizer or model training",
        "tensor_bytes": {
            "feature_input": tensor_bytes(features), "feature_gradient": tensor_bytes(features.grad),
            "clip_frames_extra_allocation": 0, "pixel_valid": tensor_bytes(pixel_valid),
            "time_offsets": tensor_bytes(offsets), "queries": tensor_bytes(queries),
            "velocities": tensor_bytes(velocity), "sampled_values": tensor_bytes(sampled.values),
            "sample_valid": tensor_bytes(sampled.sample_valid),
            "velocity_valid": tensor_bytes(sampled.velocity_valid),
        },
        "tensor_bytes_note": "Named tensor sizes, not a complete autograd/workspace peak; clip.frames is a feature view.",
        "cuda_memory": cuda,
        "cuda_resource_gate": "MEASURED_NO_THRESHOLD" if cuda is not None else "UNAVAILABLE_ON_CPU",
        "cpu_rss": {
            "before_profile_high_water_bytes": rss_before,
            "process_high_water_bytes": _rss_high_water_bytes(),
            "scope": "process_lifetime_high_water_including_prior_checks; not sampler-only or CUDA memory",
        },
        "performance_threshold_applied": False, "training_steps": 0,
    }
