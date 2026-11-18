"""Pixel-aligned motion observations for the civilian thermal traffic experiment.

Coordinates and velocities are (x, y). Spatial and velocity grids in the output
are ordered (y, x). The input is an unscaled 17-frame intensity window.
"""

from functools import lru_cache

import torch
import torch.nn.functional as F


DISCOVERY_OFFSETS = (-1, 2, 3, -4, 5, -6, -7, 8)
SPAN_OFFSETS = ((-1, 2, 3, -4), (-1, 2, 5, -6), (3, -4, -7, 8))


def _blur(frames, sigma):
    # Independent images in the batch, including when computing local variance.
    radius = round(3 * sigma)
    axis = torch.arange(-radius, radius + 1, device=frames.device, dtype=frames.dtype)
    weights = torch.softmax(-0.5 * (axis / sigma).square(), dim=0)
    padded = F.pad(frames, (radius, radius, 0, 0), mode="replicate")
    blurred = F.conv2d(padded, weights.view(1, 1, 1, -1))
    padded = F.pad(blurred, (0, 0, radius, radius), mode="replicate")
    return F.conv2d(padded, weights.view(1, 1, -1, 1))


@torch.no_grad()
def prepare_window(raw):
    """Normalize [17,H,W] intensity frames (8-bit scale). Output float32."""
    raw = torch.as_tensor(raw)
    if raw.ndim != 3 or raw.shape[0] != 17 or min(raw.shape[-2:]) < 3:
        raise ValueError("expected [17,H,W] frames with H,W >= 3")
    frames = raw.float().unsqueeze(1)
    # Centering improves the stability of the second-moment calculation.
    frames = frames - frames.mean(dim=(-2, -1), keepdim=True)
    mean, square_mean = _blur(torch.cat((frames, frames.square())), 3.0).chunk(2)
    variance = (square_mean - mean.square()).clamp_min(0)
    return ((_blur(frames, 0.8) - mean) * (variance + 4).rsqrt()).squeeze(1).contiguous()


def _xy_grid(axis):
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((x.flatten(), y.flatten()), dim=-1)


@lru_cache(maxsize=8)
def _grids(device):
    times = torch.arange(-8, 9, device=device)
    speeds = _xy_grid(torch.arange(-2, 3, device=device))
    speeds = speeds[speeds.abs().sum(dim=1) > 0]
    perturb = _xy_grid(torch.arange(-2, 3, device=device) * 0.5)
    patch = _xy_grid(torch.arange(-1, 2, device=device))
    discovery = torch.zeros((2, 17), dtype=torch.float32, device=device)
    subsets = torch.zeros((2, 4, 17), dtype=torch.float32, device=device)
    subsets[:, 0] = 1
    for direction, sign in enumerate((1, -1)):
        discovery[direction, [8 + sign * t for t in DISCOVERY_OFFSETS]] = 1
        for span_index, span in enumerate(SPAN_OFFSETS, start=1):
            # The validation span has the opposite sign to its discovery group.
            subsets[direction, span_index, [8 - sign * t for t in span]] = 1
    return times, speeds, perturb, patch, discovery, subsets


def _pick_velocity(frames, xy, selection):
    times, speeds, _, _, discovery, _ = _grids(frames.device)
    nframes, height, width = frames.shape
    path = xy[None, :, None, :] + times[:, None, None, None] * speeds[None, None, :, :]
    x, y = path.unbind(-1)
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    index = y.clamp(0, height - 1) * width + x.clamp(0, width - 1)
    values = frames.view(nframes, -1).gather(1, index.flatten(1)).view_as(x)
    weights = discovery if selection == "split" else torch.ones_like(discovery)
    # The same denominator applies to every candidate in one discovery group.
    # Missing samples contribute zero, preventing a lone edge sample from
    # receiving the same support as a path visible through the whole group.
    counts = torch.einsum("dt,tnk->ndk", weights, valid.float())
    scores = torch.einsum("dt,tnk->ndk", weights, values * valid).abs()
    enough = counts * 2 >= weights.sum(dim=1)[None, :, None]
    best = scores.masked_fill(~enough, -torch.inf).argmax(dim=-1)
    moving = speeds[best].float()
    candidates = torch.stack((torch.zeros_like(moving), moving), dim=2)
    supported = torch.stack((torch.ones_like(best, dtype=torch.bool), enough.any(-1)), dim=2)
    return candidates, supported


@torch.no_grad()
def extract_evidence(window, xy, selection):
    """Measure a chunk of pixel locations under the 'full' or 'split' rule.

    Returns patches [N,2 directions,2 anchors,4 subsets,5,5,3,3], valid
    [N,2,2,4,5,5] as fractions, velocity [N,2,2,2] xy, and usable [N].
    Direction 0 discovers in DISCOVERY_OFFSETS and tests the mirrored spans;
    direction 1 swaps roles. Subset 0 always measures each of the 17 original
    frames once. Subsets 1..3 contain four frames apiece. Both the two-step
    and joint encoders use this evidence; the model masks hidden cells.
    Use the output of prepare_window, and pass coordinates in chunks.
    """
    if selection not in ("full", "split"):
        raise ValueError("selection must be 'full' or 'split'")
    if window.ndim != 3 or window.shape[0] != 17 or window.dtype != torch.float32:
        raise ValueError("use the [17,H,W] float32 output of prepare_window")
    xy = torch.as_tensor(xy, device=window.device)
    if xy.ndim != 2 or xy.shape[1] != 2 or xy.is_floating_point():
        raise ValueError("expected [N,2] integer pixel coordinates in xy order")
    xy = xy.long()
    n, (height, width) = len(xy), window.shape[-2:]
    if not n:
        return dict(patches=window.new_empty((0, 2, 2, 4, 5, 5, 3, 3)),
                    valid=window.new_empty((0, 2, 2, 4, 5, 5)),
                    velocity=window.new_empty((0, 2, 2, 2)),
                    usable=torch.empty(0, dtype=torch.bool, device=window.device))
    times, _, perturb, patch, _, subsets = _grids(window.device)
    velocity, supported = _pick_velocity(window, xy, selection)
    speed_grid = velocity.unsqueeze(-2) + perturb
    centers = (xy[None, :, None, None, None, :] +
               times[:, None, None, None, None, None] * speed_grid[None])
    x, y = centers.unbind(-1)
    # Every admitted 3x3 observation has identical, complete spatial support.
    present = ((x >= 1) & (x <= width - 2) & (y >= 1) & (y <= height - 2)
               & supported[None, :, :, :, None])
    grid = centers.unsqueeze(-2) + patch
    scale = window.new_tensor((2 / (width - 1), 2 / (height - 1)))
    grid.mul_(scale).sub_(1)
    values = F.grid_sample(window.unsqueeze(1), grid.view(17, n * 2 * 2 * 25, 9, 2),
                           mode="bilinear", padding_mode="zeros", align_corners=True)
    values = values.view(17, n, 2, 2, 25, 9) * present.unsqueeze(-1)
    sums = torch.einsum("dst,tndakq->ndaskq", subsets, values)
    counts = torch.einsum("dst,tndak->ndask", subsets, present.float())
    patches = (sums / counts.clamp_min(1).unsqueeze(-1)).reshape(n, 2, 2, 4, 5, 5, 3, 3)
    valid = (counts / subsets.sum(-1)[None, :, None, :, None]).reshape(n, 2, 2, 4, 5, 5)
    usable = valid[:, 0, 0, 0, 2, 2] > 0
    return dict(patches=patches, valid=valid, velocity=velocity, usable=usable)
