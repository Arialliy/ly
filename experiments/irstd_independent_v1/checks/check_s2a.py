"""S2a numerical and end-to-end checks for the independently trained video baseline."""
from dataclasses import replace
from pathlib import Path
import inspect
import json
import math
import sys
from unittest.mock import patch

import torch

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.irstd_independent_v1.contracts import CENTER_INDEX, ClipBatch, Prediction, SceneMeta, Targets
from experiments.irstd_independent_v1.data.dataset import MaskClipDataset, collate_clips
from experiments.irstd_independent_v1.losses.detection import detection_loss
from experiments.irstd_independent_v1.models.video_baseline import VideoBaseline


ATOL = 1e-5
RTOL = 1e-4
REAL_CENTERS = (("Sequence1", 9), ("Sequence17", 9), ("Sequence2", 9))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    return float((actual.detach() - expected.detach()).abs().max().item())


def rejected(call, message):
    try:
        call()
    except (ValueError, TypeError):
        return
    raise AssertionError(message)


def fixture(device):
    h, w = 9, 13
    # Fixed, nonconstant, unit-interval signals, independent of random model initialization.
    frames = torch.arange(17 * h * w, device=device).remainder(31).float().reshape(1, 17, 1, h, w) / 31
    offsets = torch.arange(-8, 9, device=device, dtype=torch.float32)[None]
    meta = SceneMeta("Fixture", tuple(range(1, 18)), (h, w), (0, 0), (0, 0, w, h))
    return ClipBatch(frames, offsets, torch.ones_like(frames, dtype=torch.bool), 8,
                     ("Fixture/00009",), (meta,))


def check_inputs(model, device):
    clip = fixture(device)
    require(list(inspect.signature(model.forward).parameters) == ["clip"], "forward must take only ClipBatch")
    wrong_times = clip.time_offsets.clone()
    wrong_times[0, -1] = 7
    wrong_frames = list(clip.scene_meta[0].frame_numbers)
    wrong_frames[12] += 1
    bad = {
        "last_offset_repeats_previous": replace(clip, time_offsets=wrong_times),
        "center_id_mismatch": replace(clip, output_frame_id=("Fixture/00010",)),
        "discontinuous_frames": replace(clip, scene_meta=(replace(clip.scene_meta[0],
                                               frame_numbers=tuple(wrong_frames)),)),
        "center_index_mismatch": replace(clip, center_index=7),
    }
    for name, invalid in bad.items():
        rejected(lambda: model(invalid), f"Accepted invalid clip: {name}")
    return list(bad)


def check_one_level_contrast(model, device):
    """Closed-form variance for one changed uint8 pixel in a clipped window, computed in float64."""
    h, w = 7, 9
    levels = torch.arange(256, dtype=torch.float32) / 255
    fractional_levels = torch.tensor([0.0123456, 0.374216, 216.123 / 255, 0.923456, 0.99456])
    all_levels = torch.cat((levels, fractional_levels))
    constants = all_levels[:, None, None, None, None].expand(-1, 1, 1, h, w).contiguous().to(device)
    with torch.no_grad():
        flat = model.contrast_channel(constants, torch.ones_like(constants, dtype=torch.bool))
    require(torch.count_nonzero(flat).item() == 0, "Flat intensity background has nonzero contrast")

    rows, columns = torch.arange(h)[:, None], torch.arange(w)[None, :]
    height = (rows + 3).clamp_max(h) - (rows - 2).clamp_min(0)
    width = (columns + 3).clamp_max(w) - (columns - 2).clamp_min(0)
    n = (height * width).double()
    native_cases = 2 * (levels.numel() - 1)
    backgrounds = torch.cat((levels[:-1], levels[1:], fractional_levels, fractional_levels))
    impulses = torch.cat((levels[1:], levels[:-1], fractional_levels + 1 / 255, fractional_levels - 1 / 255))
    delta = (impulses.double() - backgrounds.double())[:, None, None]
    max_native_error, max_fractional_error = 0.0, 0.0
    for y, x in ((h // 2, w // 2), (0, w // 2), (h - 1, w - 1)):
        image = backgrounds[:, None, None].expand(-1, h, w).clone()
        image[:, y, x] = impulses
        frames = image[:, None, None].to(device)
        with torch.no_grad():
            z = model.contrast_channel(frames, torch.ones_like(frames, dtype=torch.bool))[:, 0, 0]
        contains_spike = ((rows - y).abs() <= 2) & ((columns - x).abs() <= 2)
        spike_site = (rows == y) & (columns == x)
        mean_delta = contains_spike * delta / n
        variance = contains_spike * delta.square() * (n - 1) / n.square()
        reference = (spike_site * delta - mean_delta) / (variance + (1 / 255) ** 2).sqrt()
        expected = reference.to(device=device, dtype=torch.float32)
        max_native_error = max(max_native_error, close(z[:native_cases], expected[:native_cases]))
        max_fractional_error = max(max_fractional_error, close(z[native_cases:], expected[native_cases:]))
    return {"flat_uint8_levels": len(levels), "flat_nonzero_outputs": 0,
            "signed_one_level_impulse_cases": native_cases * 3, "max_abs_error": max_native_error,
            "fractional_backgrounds": fractional_levels.tolist(),
            "fractional_signed_impulse_cases": 6 * len(fractional_levels),
            "fractional_max_abs_error": max_fractional_error}


def check_contrast(model, device):
    # Compare the actual production transform to per-pixel, float64 neighborhood enumeration.
    clip = fixture(device)
    frames = (clip.frames * 4).floor() / 4
    valid = clip.pixel_valid.clone()
    valid[:, 0] = False
    valid[:, 1:, :, 2:4, 4] = False
    valid[:, 1:, :, 5, 3:6] = False
    signal = frames.masked_fill(~valid, float("nan"))
    frames_cpu, valid_cpu = frames[0, :, 0].cpu(), valid[0, :, 0].cpu()
    reference = torch.zeros_like(frames_cpu)
    for t, y, x in valid_cpu.nonzero().tolist():
        region = (t, slice(max(0, y - 2), y + 3), slice(max(0, x - 2), x + 3))
        v = frames_cpu[region][valid_cpu[region]].double()
        reference[t, y, x] = ((frames_cpu[t, y, x] - v.mean()) /
                               (v.var(unbiased=False) + (1 / 255) ** 2).sqrt())
    with torch.no_grad():
        actual = model.contrast_channel(signal, valid)
    err = close(actual[0, :, 0], reference.to(device))
    return {"reference_max_abs_error": err, "pixels": reference.numel(),
            "invalid_pixels": int((~valid).sum().item()),
            "quantized_backgrounds": check_one_level_contrast(model, device)}


def check_loss(device):
    # Frame 0 is mixed, frame 1 is empty, frame 2 is all foreground.
    raw_logits = [[-0.8, 0.3, 1.7, -2.0, 0.9], [0.0] * 5, [0.0] * 5]
    raw_target = [[1, 0, 0, 0, 0], [0] * 5, [1] * 5]
    values = torch.tensor(raw_logits, dtype=torch.float32, device=device).reshape(3, 1, 1, 5)
    labels = torch.tensor(raw_target, dtype=torch.float32, device=device).reshape_as(values)
    valid = torch.ones_like(values, dtype=torch.bool)
    valid[:, :, :, 4] = False
    ignore = torch.zeros_like(valid)
    ignore[:, :, :, 3] = True
    mask = valid & ~ignore
    logits = values.masked_fill(~mask, float("nan")).requires_grad_()
    ids = ("a/00009", "b/00009", "c/00009")
    pred = Prediction(logits, ids)
    target = Targets("mask", labels.masked_fill(~mask, float("nan")), ignore, ((), (), ()))
    loss = detection_loss(pred, target, valid)
    reference_grad = torch.zeros_like(values)
    reference_losses = []
    for b, row in enumerate(raw_logits):
        z, y = row[:3], raw_target[b][:3]
        p = [1 / (1 + math.exp(-v)) for v in z]
        classes = set(y)
        bce = 0.0
        numerator = 1 + 2 * sum(pp * yy for pp, yy in zip(p, y))
        denominator = 1 + sum(p) + sum(y)
        for x, (v, pp, yy) in enumerate(zip(z, p, y)):
            weight = 1 / (len(classes) * y.count(yy))
            bce += weight * (math.log1p(math.exp(v)) - yy * v)
            grad = weight * (pp - yy)
            if any(y):
                grad += 0.5 * (numerator - 2 * yy * denominator) / denominator ** 2 * pp * (1 - pp)
            reference_grad[b, 0, 0, x] = grad / len(raw_logits)
        reference_losses.append(bce + (0.5 * (1 - numerator / denominator) if any(y) else 0))
    loss_error = close(loss, loss.new_tensor(sum(reference_losses) / len(reference_losses)))
    actual_grad, = torch.autograd.grad(loss, logits)
    gradient_error = close(actual_grad, reference_grad)
    require(bool((actual_grad[~mask] == 0).all()), "Ignored logits received a gradient")
    empty_pred = Prediction(logits[1:2], ids[1:2])
    empty_target = Targets("mask", target.target_map[1:2], ignore[1:2], ((),))
    empty_loss = detection_loss(empty_pred, empty_target, valid[1:2])
    close(empty_loss, empty_loss.new_tensor(math.log(2)))
    no_supervision = valid.clone()
    no_supervision[0] = False
    rejected(lambda: detection_loss(pred, target, no_supervision), "Accepted an unsupervised sample")
    require(bool((actual_grad[1][mask[1]] > 0).all()), "Empty frame must suppress positive logits")
    return {"max_abs_loss_error": loss_error, "max_abs_gradient_error": gradient_error,
            "empty_zero_logit_bce": float(empty_loss.detach()), "ignored_gradient_nonzeros": 0}


def check_temporal_path(model, device):
    clip = fixture(device)
    encoder = model.frame_encoder
    contrast = model.contrast_channel
    readout = model.video_aggregator.temporal_features
    with torch.no_grad():
        changed = clip.frames.clone()
        changed[:, 8] = 1 - changed[:, 8]
        changed_clip = replace(clip, frames=changed)
        frame_features = encoder(clip.frames, clip.pixel_valid)
        modified = encoder(changed, clip.pixel_valid)
        temporal, cover = readout(frame_features, contrast(clip.frames, clip.pixel_valid), clip)
        altered, cover2 = readout(modified, contrast(changed, clip.pixel_valid), changed_clip)
        require(torch.equal(temporal, altered) and torch.equal(cover, cover2),
                "Center image leaked into the temporal readout")
        require(bool((cover == 1).all()), "Context coverage must use 16 noncenter slots as denominator")
        center_change = float((frame_features[:, 8] - modified[:, 8]).abs().max())
        center_logit_change = float((model(clip).logits - model(changed_clip).logits).abs().max())
        require(min(center_change, center_logit_change) > 0, "Center image does not reach the spatial path")

        changed_context = clip.frames.clone()
        changed_context[:, 0] = 1 - changed_context[:, 0]
        lateral = replace(clip, frames=changed_context)
        context_logit_change = float((model(clip).logits - model(lateral).logits).abs().max())
        require(context_logit_change > 0, "Changing only the first context frame did not change any logit")

    # Test dependence using the trained path's actual initialized weights and differentiable input.
    feature_probe = frame_features.detach().requires_grad_()
    signal = contrast(clip.frames, clip.pixel_valid).detach()
    temporal, _ = readout(feature_probe, signal, clip)
    grad, = torch.autograd.grad(temporal.square().sum(), feature_probe)
    norms = grad.abs().sum(dim=(0, 2, 3, 4)).detach().cpu()
    require(bool(torch.isfinite(norms).all()) and float(norms[CENTER_INDEX]) == 0
            and bool((torch.cat((norms[:8], norms[9:])) > 0).all()),
            "Temporal feature dependence does not match the 16 noncenter slots")
    holes = clip.pixel_valid.clone()
    holes[:, 0, :, 3, 3] = False
    missing = replace(clip, pixel_valid=holes)
    with torch.no_grad():
        _, coverage = readout(frame_features, signal, missing)
    require(float(coverage[0, 0, 3, 3]) == 15 / 16, "One missing context frame must reduce coverage by 1/16")
    return {"temporal_l1_feature_gradient_by_slot": norms.tolist(), "center_temporal_max_change": 0.0,
            "center_spatial_feature_max_change": center_change, "center_input_logit_max_change": center_logit_change,
            "first_context_input_logit_max_change": context_logit_change, "coverage_with_one_missing": 15 / 16}


def on_device(sample, device):
    clip, labels = sample
    clip = replace(clip, frames=clip.frames.to(device), pixel_valid=clip.pixel_valid.to(device),
                   time_offsets=clip.time_offsets.to(device))
    labels = replace(labels, target_map=labels.target_map.to(device), ignore_map=labels.ignore_map.to(device))
    return clip, labels


def get_real_samples(project_root, device):
    manifest_path = project_root / "experiments/irstd_independent_v1/manifests/nudt_mirsdt_mask17.json"
    ds = MaskClipDataset(json.loads(manifest_path.read_bytes()), project_root, "train")
    index = {key: i for i, key in enumerate(ds.centers)}
    samples = [on_device(ds[index[key]], device) for key in REAL_CENTERS]
    actual_ids = tuple(clip.output_frame_id[0] for clip, _ in samples)
    require(actual_ids == tuple(f"{name}/{center:05d}" for name, center in REAL_CENTERS),
            "Training sample identity or order changed")
    fg = [int(labels.target_map.sum()) for _, labels in samples]
    require(fg[0] > 0 and fg[1] == 0 and fg[2] > 0, "Expected two positive samples and one empty sample")
    return samples


def check_actual_tiles(model, clip, expected_logits, core_hw):
    """Trace executed source/region calls and independently check all native writeback partitions."""
    covered = {frame_id: torch.zeros(meta.original_hw, dtype=torch.int32)
               for frame_id, meta in zip(clip.output_frame_id, clip.scene_meta)}
    with patch.object(model, "_region_logits", wraps=model._region_logits) as trace:
        tiled = model.predict_tiled(clip, core_hw=core_hw)
    require(tiled.output_frame_id == clip.output_frame_id, "Tiled prediction changed output IDs")
    for args, _kwargs in trace.call_args_list:
        source, tile = args
        frame_id, = source.output_frame_id
        height, width = covered[frame_id].shape
        x0, y0, x1, y1 = tile.core_xyxy
        q = (max(0, x0 - 2), max(0, y0 - 2), min(width, x1 + 2), min(height, y1 + 2))
        src = (max(0, x0 - 4), max(0, y0 - 4), min(width, x1 + 4), min(height, y1 + 4))
        require(tile.query_xyxy == q and tile.input_xyxy == src,
                "Tile failed independent decoder/encoder halo calculation")
        require(tuple(source.frames.shape[-2:]) == (src[3] - src[1], src[2] - src[0])
                and source.scene_meta[0].tile_origin_xy == src[:2], "Wrong loaded extent or global origin")
        covered[frame_id][y0:y1, x0:x1] += 1
    for frame_id, coverage in covered.items():
        require(bool((coverage == 1).all()), f"Output has skipped or duplicated core pixels for {frame_id}")
    err = close(tiled.logits, expected_logits)
    return {"core_hw": list(core_hw), "tile_calls": trace.call_count, "max_abs_logit_error": err,
            "unique_native_pixels": sum(mask.numel() for mask in covered.values())}


def check_spatial_equivalence(model, samples):
    batch, targets = collate_clips(samples)
    singles = []
    with torch.no_grad():
        for clip, _ in samples:
            prediction = model(clip)
            require(prediction.output_frame_id == clip.output_frame_id
                    and prediction.logits.shape == clip.frames[:, 8].shape,
                    "Native output shape or center ID mismatch")
            require(bool(torch.isfinite(prediction.logits).all()), "Nonfinite native prediction")
            singles.append(prediction.logits)
        batched = model(batch)
        require(batched.output_frame_id == batch.output_frame_id and batched.logits.shape == targets.target_map.shape,
                "Padded output lost IDs or dimensions")
        errors = []
        for i, single in enumerate(singles):
            h, w = single.shape[-2:]
            errors.append(close(batched.logits[i:i+1, :, :h, :w], single))
        require(bool((batched.logits[~batch.pixel_valid[:, 8]] == 0).all()), "Padded logits are nonzero")
        poison = replace(batch, frames=batch.frames.masked_fill(~batch.pixel_valid, float("nan")))
        poisoned = model(poison)
        poison_err = close(poisoned.logits, batched.logits)
        nll = detection_loss(batched, targets, batch.pixel_valid[:, 8])
        native_losses = [detection_loss(Prediction(z, c.output_frame_id), target, c.pixel_valid[:, 8])
                         for (c, target), z in zip(samples, singles)]
        padding_loss_error = close(nll, torch.stack(native_losses).mean())
        tiled_full = check_actual_tiles(model, poison, batched.logits, (64, 64))
        tiled_rect = check_actual_tiles(model, samples[0][0], singles[0], (59, 43))
    return {"real_batch_shape": list(batch.frames.shape), "single_to_batch_max_abs_errors": errors,
            "nan_padding_logit_error": poison_err, "padded_loss_error": padding_loss_error,
            "tile_checks": [tiled_full, tiled_rect]}


def module_update(model, samples):
    groups = {"frame_encoder": model.frame_encoder, "time_convolutions": model.video_aggregator.temporal,
              "fusion": model.video_aggregator.fusion, "spatial_decoder": model.spatial_decoder,
              "detection_head": model.detection_head}
    params = dict(model.named_parameters())
    names_by_id = {id(p): name for name, p in params.items()}
    membership = {name: [names_by_id[id(p)] for p in module.parameters()] for name, module in groups.items()}
    listed = [name for names in membership.values() for name in names]
    require(len(listed) == len(set(listed)) == len(params), "Group audit omitted or duplicated parameters")
    before = {name: p.detach().clone() for name, p in params.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    optimizer.zero_grad(set_to_none=True)
    model.train()
    records = []
    for clip, labels in samples:
        pred = model(clip)
        require(len(pred.diagnostics) == 0, "Default Prediction should not retain diagnostic tensors")
        loss = detection_loss(pred, labels, clip.pixel_valid[:, 8])
        require(bool(torch.isfinite(loss)), "Nonfinite real supervised loss")
        nfg = int(labels.target_map.sum())
        row = {"center": clip.output_frame_id[0], "native_hw": list(clip.scene_meta[0].original_hw),
               "foreground_pixels": nfg, "loss": float(loss.detach())}
        if nfg == 0:
            gradient, = torch.autograd.grad(loss, pred.logits, retain_graph=True)
            bg = gradient[clip.pixel_valid[:, 8] & ~labels.ignore_map]
            require(bool(torch.isfinite(bg).all() and (bg > 0).all()), "No background suppression on real empty frame")
            row["min_background_logit_gradient"] = float(bg.min())
        (loss / len(samples)).backward()
        records.append(row)
    for name, p in params.items():
        require(p.grad is not None and bool(torch.isfinite(p.grad).all()), f"Missing or nonfinite gradient: {name}")
    optimizer.step()
    stats = {}
    for group_name, names in membership.items():
        grad_sq = sum(params[n].grad.detach().double().square().sum() for n in names)
        update_sq = sum((params[n].detach() - before[n]).double().square().sum() for n in names)
        norms = (float(grad_sq.sqrt()), float(update_sq.sqrt()))
        require(all(math.isfinite(x) and x > 0 for x in norms), f"Disconnected or unchanged group: {group_name}")
        stats[group_name] = {"parameters": sum(params[n].numel() for n in names),
                             "gradient_l2": norms[0], "update_l2": norms[1]}
    return {"engineering_optimizer_updates": 1, "aggregation": "equal average over three separate B=1 forwards",
            "optimizer": "AdamW", "lr": 1e-3, "weight_decay": 1e-4, "samples": records, "groups": stats}


def verify_s2a(project_root: Path, device: torch.device) -> dict:
    """Return JSON-serializable evidence after each design assertion succeeds."""
    torch.manual_seed(46)
    model = VideoBaseline().to(device).eval()
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    require(total == 22921 and all(p.requires_grad for p in model.parameters()),
            "Model size or trainable-parameter contract changed")
    rejected_inputs = check_inputs(model, device)
    contrast = check_contrast(model, device)
    losses = check_loss(device)
    branches = check_temporal_path(model, device)
    real = get_real_samples(project_root, device)
    boundaries = check_spatial_equivalence(model, real)
    update = module_update(model, real)
    return {"status": "PASS", "seed": 46, "trainable_parameters": total,
            "tolerance": {"atol": ATOL, "rtol": RTOL}, "rejected_inputs": rejected_inputs,
            "local_contrast": contrast, "analytical_loss": losses, "branch_dependence": branches,
            "spatial_equivalence": boundaries, "engineering_optimizer_updates": 1, "update": update}


if __name__ == "__main__":
    from experiments.irstd_independent_v1.checks.s2a_gate import main
    raise SystemExit(main())
