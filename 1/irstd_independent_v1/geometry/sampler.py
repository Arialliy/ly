"""Differentiable, per-time motion-hypothesis evidence with explicit geometry.

Coordinates denote native pixel centers: x=0 is the first pixel center. Their
normalized-grid equivalent uses align_corners=False and the normalization
``2 * (local_xy + 0.5) / (storage_width, storage_height) - 1``. Actual interpolation
uses four native-coordinate gathers: avoiding a normalized-coordinate round trip
prevents numerical weights on zero-weight padded neighbors. Physical validity is
independent of bottom/right tensor padding.
"""
from dataclasses import dataclass

import torch

from ..contracts import ClipBatch


ALIGN_CORNERS = False


class MissingHaloError(ValueError):
    """A real-image observation is unavailable in the loaded source tile."""


@dataclass(frozen=True)
class SampledEvidence:
    values: torch.Tensor          # [B,N,C,T,My,Mx], invalid time samples are zero
    sample_valid: torch.Tensor    # bool [B,N,T,My,Mx]
    velocity_valid: torch.Tensor  # bool [B,N,My,Mx], sample_valid.all(time)
    time_offsets: torch.Tensor    # [B,T], context times in original frame units
    queries_xy: torch.Tensor      # [B,N,2], original-image coordinates
    velocities_xy: torch.Tensor   # [My,Mx,2], original pixels / frame interval


def _real_tensor(value, name, *, device, dtype):
    tensor = torch.as_tensor(value, device=device)
    if tensor.requires_grad:
        raise TypeError(f"{name} is fixed geometry in S1; only feature gradients are supported")
    if tensor.is_complex() or tensor.dtype == torch.bool:
        raise TypeError(f"{name} must contain real numeric coordinates")
    # Re-read the original input in the requested dtype. Converting the
    # inferred tensor would already have rounded Python float lists to FP32.
    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must contain only finite values")
    return tensor


def _batch_geometry(value, name, batch, *, device, dtype, positive=False):
    tensor = _real_tensor(value, name, device=device, dtype=dtype)
    if tensor.shape == (2,):
        tensor = tensor[None].expand(batch, -1)
    if tensor.shape != (batch, 2):
        raise ValueError(f"{name} must have shape [B,2] (or a common length-2 pair)")
    if not bool((tensor == tensor.round()).all()):
        raise ValueError(f"{name} must contain integer pixel geometry")
    if positive and not bool((tensor > 0).all()):
        raise ValueError(f"{name} sizes must be positive")
    return tensor


def _positive_chunk(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def sample_context(
    features: torch.Tensor,
    time_offsets: torch.Tensor,
    queries_xy: torch.Tensor,
    velocities_xy: torch.Tensor,
    pixel_valid: torch.Tensor | None = None,
    image_hw: torch.Tensor | None = None,
    origin_xy: torch.Tensor | None = None,
    source_hw: torch.Tensor | None = None,
    *,
    query_chunk: int = 256,
    velocity_chunk: int = 8,
) -> SampledEvidence:
    """Sample ``features[t]`` at ``query + time_offsets[t] * velocity``.

    ``features`` is float32/float64 ``[B,T,C,H,W]``; T must be positive.
    ``queries_xy`` is ``[N,2]`` or ``[B,N,2]`` and may be empty. Time and both
    velocity axes are preserved. The supplied time axis is not sorted, averaged,
    or filtered by this low-level function. Geometry is fixed in S1: time,
    query, velocity, and image/source geometry tensors requiring gradients are
    rejected. Gradients with respect to features remain supported.

    ``image_hw`` is the true full-image size, ``origin_xy`` is the loaded tile's
    origin in that image, and ``source_hw`` is the tile's actual loaded extent
    before bottom/right batch padding. Full-image defaults are allowed. Every
    cropped source, including one with origin (0,0), must explicitly supply
    image_hw and source_hw. Omitting source_hw declares that the complete real
    image is loaded; a mask cannot distinguish missing crop extent from holes
    within an otherwise loaded image, so crop extent is never inferred from it.

    A sample is valid only if every *positive-weight* bilinear support pixel is
    inside the real image, loaded source, and pixel_valid. Real-image samples
    missing from the source raise MissingHaloError instead of silently becoming
    background. Queries outside the real image yield only invalid zero samples.
    Invalid samples have zero derivative with respect to features. Chunking only
    changes working-memory use, never the mathematical operator.
    """
    if not isinstance(features, torch.Tensor) or features.ndim != 5:
        raise ValueError("features must be a tensor with shape [B,T,C,H,W]")
    if features.dtype not in (torch.float32, torch.float64):
        raise TypeError("features must use float32 or float64")
    batch, times, channels, height, width = features.shape
    if min(batch, times, channels, height, width) <= 0:
        raise ValueError("features requires positive B, T, C, H, and W")
    query_chunk = _positive_chunk(query_chunk, "query_chunk")
    velocity_chunk = _positive_chunk(velocity_chunk, "velocity_chunk")
    device, dtype = features.device, features.dtype

    times_tensor = _real_tensor(time_offsets, "time_offsets", device=device, dtype=dtype)
    if times_tensor.shape != (batch, times):
        raise ValueError("time_offsets must have shape [B,T] matching features")
    queries = _real_tensor(queries_xy, "queries_xy", device=device, dtype=dtype)
    if queries.ndim == 2 and queries.shape[-1] == 2:
        queries = queries[None].expand(batch, -1, -1)
    if queries.ndim != 3 or queries.shape[0] != batch or queries.shape[-1] != 2:
        raise ValueError("queries_xy must have shape [N,2] or [B,N,2]")
    velocities = _real_tensor(velocities_xy, "velocities_xy", device=device, dtype=dtype)
    if velocities.ndim != 3 or velocities.shape[-1] != 2 or min(velocities.shape[:2]) <= 0:
        raise ValueError("velocities_xy must have nonempty shape [My,Mx,2]")
    my, mx = velocities.shape[:2]
    velocity_count = my * mx
    velocities_flat = velocities.reshape(velocity_count, 2)

    origins = _batch_geometry(
        (0, 0) if origin_xy is None else origin_xy, "origin_xy", batch,
        device=device, dtype=dtype,
    )
    if not bool((origins >= 0).all()):
        raise ValueError("origin_xy must be nonnegative; source tiles must be clipped to the real image")
    cropped_origin = bool((origins != 0).any())
    if cropped_origin and (image_hw is None or source_hw is None):
        raise ValueError("Nonzero origin_xy requires explicit image_hw and source_hw; do not infer tile halo")
    images = _batch_geometry(
        (height, width) if image_hw is None else image_hw, "image_hw", batch,
        device=device, dtype=dtype, positive=True,
    )
    sources = _batch_geometry(
        images if source_hw is None else source_hw, "source_hw", batch,
        device=device, dtype=dtype, positive=True,
    )
    storage_hw = features.new_tensor((height, width))
    if not bool((sources <= storage_hw).all()):
        raise ValueError("source_hw exceeds stored features; a crop must provide its actual loaded extent")
    if not bool((origins + sources.flip(-1) <= images.flip(-1)).all()):
        raise ValueError("origin_xy plus source_hw exceeds the real image extent")

    if pixel_valid is None:
        valid_pixels = torch.ones((batch, times, 1, height, width), dtype=torch.bool, device=device)
    else:
        if not isinstance(pixel_valid, torch.Tensor) or pixel_valid.dtype != torch.bool:
            raise TypeError("pixel_valid must be a boolean tensor")
        if pixel_valid.shape != (batch, times, 1, height, width):
            raise ValueError("pixel_valid must have shape [B,T,1,H,W] matching features")
        valid_pixels = pixel_valid.to(device=device)

    count = queries.shape[1]
    if count == 0:
        # This empty view retains the feature autograd connection.
        values = features[:, :0].reshape(batch, 0, channels, times, my, mx)
        sample_valid = torch.empty((batch, 0, times, my, mx), dtype=torch.bool, device=device)
        return SampledEvidence(values, sample_valid, sample_valid.all(dim=2),
                               times_tensor, queries, velocities)

    image_max_xy = images.flip(-1) - 1
    source_max_xy = sources.flip(-1) - 1
    query_valid = ((queries >= 0) & (queries <= image_max_xy[:, None])).all(dim=-1)
    flat_features = features.reshape(batch * times, channels, height * width)
    flat_valid = valid_pixels.reshape(batch * times, height * width)
    result_chunks = []
    validity_chunks = []

    for q_start in range(0, count, query_chunk):
        q_stop = min(count, q_start + query_chunk)
        local_count = q_stop - q_start
        values_for_velocities = []
        masks_for_velocities = []
        for v_start in range(0, velocity_count, velocity_chunk):
            v_stop = min(velocity_count, v_start + velocity_chunk)
            local_velocities = v_stop - v_start
            global_xy = (
                queries[:, None, q_start:q_stop, None, :]
                + times_tensor[:, :, None, None, None]
                * velocities_flat[None, None, None, v_start:v_stop, :]
            )
            if not bool(torch.isfinite(global_xy).all()):
                raise ValueError("query + time * velocity overflowed the coordinate dtype")
            real_sample = (
                (global_xy >= 0) & (global_xy <= image_max_xy[:, None, None, None])
            ).all(dim=-1) & query_valid[:, None, q_start:q_stop, None]
            local_xy = global_xy - origins[:, None, None, None, :]
            in_source = (
                (local_xy >= 0) & (local_xy <= source_max_xy[:, None, None, None])
            ).all(dim=-1)
            missing = real_sample & ~in_source
            if bool(missing.any()):
                b, t, q, v = torch.nonzero(missing, as_tuple=False)[0].tolist()
                coordinate = global_xy[b, t, q, v].detach().cpu().tolist()
                raise MissingHaloError(
                    "Missing source halo: a valid real-image sample at "
                    f"{coordinate} (batch={b}, time={t}, query={q_start + q}, "
                    f"velocity={v_start + v}) is outside the loaded source extent. "
                    "Load sufficient halo or correct source_hw/origin_xy."
                )

            # Do not convert huge/invalid coordinates to integer gather indices.
            safe_xy = torch.where(real_sample[..., None], local_xy, torch.zeros_like(local_xy))
            floor_xy = safe_xy.floor()
            fraction = safe_xy - floor_xy
            lower = floor_xy.to(torch.long)
            dx, dy = fraction.unbind(dim=-1)
            support_valid = torch.ones_like(real_sample)
            # Test positive factors, not their product: tiny nonzero weights
            # must not be mistaken for zero if a low-precision product underflows.
            corners = ((0, 0, (1 - dx) * (1 - dy), torch.ones_like(real_sample)),
                       (1, 0, dx * (1 - dy), dx > 0),
                       (0, 1, (1 - dx) * dy, dy > 0),
                       (1, 1, dx * dy, (dx > 0) & (dy > 0)))
            corner_indices = []
            for offset_x, offset_y, weight, positive_weight in corners:
                ix = (lower[..., 0] + offset_x).clamp(0, width - 1)
                iy = (lower[..., 1] + offset_y).clamp(0, height - 1)
                indices = (iy * width + ix).reshape(batch * times, -1)
                corner_indices.append(indices)
                support = flat_valid.gather(1, indices).reshape(
                    batch, times, local_count, local_velocities
                )
                support_valid = support_valid & (~positive_weight | support)
            sample_ok = real_sample & support_valid

            sampled = None
            for indices, (_, _, weight, positive_weight) in zip(corner_indices, corners):
                gathered = flat_features.gather(
                    2, indices[:, None].expand(-1, channels, -1)
                ).reshape(batch, times, channels, local_count, local_velocities)
                use_corner = (sample_ok & positive_weight)[:, :, None]
                # Mask before multiplication: even NaN/Inf poison in unused
                # padding cannot turn an exact-zero contribution into NaN.
                gathered = torch.where(use_corner, gathered, torch.zeros_like(gathered))
                contribution = gathered * weight[:, :, None]
                sampled = contribution if sampled is None else sampled + contribution
            values_for_velocities.append(sampled.permute(0, 3, 2, 1, 4))
            masks_for_velocities.append(sample_ok.permute(0, 2, 1, 3))
        result_chunks.append(torch.cat(values_for_velocities, dim=-1))
        validity_chunks.append(torch.cat(masks_for_velocities, dim=-1))

    values = torch.cat(result_chunks, dim=1).reshape(batch, count, channels, times, my, mx)
    sample_valid = torch.cat(validity_chunks, dim=1).reshape(batch, count, times, my, mx)
    return SampledEvidence(values, sample_valid, sample_valid.all(dim=2),
                           times_tensor, queries, velocities)


def sample_clip_features(
    features: torch.Tensor,
    clip: ClipBatch,
    queries_xy: torch.Tensor,
    velocities_xy: torch.Tensor,
    **kwargs,
) -> SampledEvidence:
    """Use S0 geometry and remove the center from features, masks, and times.

    A 17-frame S0 clip produces 16 context slots in original order. Full images
    with heterogeneous batch padding use each SceneMeta.original_hw as source_hw.
    Every cropped clip, including a crop at origin (0,0), requires explicit
    source_hw. The default declares complete original images, not a crop inferred
    from pixel_valid. Nonzero origins are additionally checked at runtime; padded
    storage size never substitutes for the actual loaded tile extent.
    """
    if not isinstance(clip, ClipBatch):
        raise TypeError("clip must be an S0 ClipBatch")
    if not isinstance(features, torch.Tensor) or features.ndim != 5:
        raise ValueError("features must have shape [B,T,C,H,W]")
    batch, times, _, height, width = features.shape
    if clip.frames.ndim != 5 or clip.frames.shape != (batch, times, 1, height, width):
        raise ValueError("features must match clip batch, time, and native spatial dimensions")
    if len(clip.scene_meta) != batch:
        raise ValueError("clip.scene_meta must have one record per batch element")
    if times < 2 or isinstance(clip.center_index, bool) or not isinstance(clip.center_index, int):
        raise ValueError("clip requires at least two frames and an integer center_index")
    if not 0 <= clip.center_index < times:
        raise ValueError("clip.center_index is outside the time axis")
    if clip.time_offsets.shape != (batch, times):
        raise ValueError("clip.time_offsets must match [B,T]")
    if not bool((clip.time_offsets[:, clip.center_index] == 0).all()):
        raise ValueError("clip center_index must identify time offset zero")
    if clip.pixel_valid.shape != (batch, times, 1, height, width):
        raise ValueError("clip.pixel_valid must match feature geometry")
    for name in ("time_offsets", "pixel_valid", "image_hw", "origin_xy"):
        if name in kwargs:
            raise ValueError(f"{name} is derived from ClipBatch and cannot be overridden")

    images = [meta.original_hw for meta in clip.scene_meta]
    origins = [meta.tile_origin_xy for meta in clip.scene_meta]
    source_hw = kwargs.pop("source_hw", None)
    if source_hw is None:
        if any(tuple(origin) != (0, 0) for origin in origins):
            raise ValueError("Cropped ClipBatch requires explicit source_hw, separate from batch padding")
        source_hw = images
    indices = torch.tensor(
        [index for index in range(times) if index != clip.center_index],
        dtype=torch.long, device=features.device,
    )
    context_times = clip.time_offsets.to(device=features.device).index_select(1, indices)
    if bool((context_times == 0).any()):
        raise ValueError("Noncenter context must not contain another zero time offset")
    return sample_context(
        features.index_select(1, indices), context_times,
        queries_xy, velocities_xy,
        pixel_valid=clip.pixel_valid.to(device=features.device).index_select(1, indices),
        image_hw=images, origin_xy=origins, source_hw=source_hw,
        **kwargs,
    )
