"""Dense tile geometry in native image coordinates; no model or label access.

All boxes are ``(x0, y0, x1, y1)`` with exclusive right/bottom bounds.
Sizes are ``(height, width)``. Cores partition the image exactly once; queries
also cover the decoder's spatial neighbourhood, and input crops supply the
motion, frame-encoder/contrast, and interpolation support around those queries.

For axis ``a`` the conservative input halo outside the query is
``ceil(max_{t,v} |time_offsets[t] * velocities_xy[v,a]|)
  + max(encoder_radius, contrast_radius) + interp_radius``.
The decoder radius is already included when forming the query box. Adding
these supports is necessary when encoding a cropped raw clip: supplying only
the displaced sampling points would make encoded features depend on artificial
crop-edge padding. This guarantee assumes local, stride-one operators with the
declared radii; crop-dependent global normalization is not covered. True image
edges remain clipped, and the sampler must still exclude out-of-image paths.

Planning uses only the standard library. ``dense_queries`` imports PyTorch
lazily and creates coordinates, not learned features or model outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite
from numbers import Real
from operator import index
from typing import Any, Iterable


Box = tuple[int, int, int, int]
SizeHW = tuple[int, int]


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not bool")
    try:
        result = index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _items(value: Any, name: str) -> list[Any]:
    # Tensor/array inputs are geometry metadata; no torch/numpy import needed.
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence")
    try:
        return list(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence") from exc


def _size(value: Any, name: str) -> SizeHW:
    values = _items(value, name)
    if len(values) != 2:
        raise ValueError(f"{name} must contain (height, width)")
    return (
        _integer(values[0], f"{name}[0]", minimum=1),
        _integer(values[1], f"{name}[1]", minimum=1),
    )


def _box(value: Any, name: str) -> Box:
    values = _items(value, name)
    if len(values) != 4:
        raise ValueError(f"{name} must contain (x0, y0, x1, y1)")
    box = tuple(_integer(v, f"{name}[{i}]") for i, v in enumerate(values))
    if box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError(f"{name} must be nonempty with exclusive upper bounds")
    return box


def _real(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _contains(outer: Box, inner: Box) -> bool:
    return (outer[0] <= inner[0] and outer[1] <= inner[1]
            and inner[2] <= outer[2] and inner[3] <= outer[3])


def _expand(box: Box, pad_x: int, pad_y: int, image_hw: SizeHW) -> Box:
    height, width = image_hw
    return (max(0, box[0] - pad_x), max(0, box[1] - pad_y),
            min(width, box[2] + pad_x), min(height, box[3] + pad_y))


@dataclass(frozen=True)
class Tile:
    """One immutable tile, with all boxes in full-image coordinates.

    ``core_slice_in_query`` indexes the last two dimensions of a query-grid
    result. ``source_origin_xy`` and ``input_hw`` describe the actual crop,
    before any batch padding, for conversion between source and image space.
    """

    core_xyxy: Box
    query_xyxy: Box
    input_xyxy: Box
    original_hw: SizeHW

    def __post_init__(self) -> None:
        image_hw = _size(self.original_hw, "original_hw")
        object.__setattr__(self, "original_hw", image_hw)
        full_image = (0, 0, image_hw[1], image_hw[0])
        for name in ("core_xyxy", "query_xyxy", "input_xyxy"):
            box = _box(getattr(self, name), name)
            if not _contains(full_image, box):
                raise ValueError(f"{name} must be entirely inside the original image")
            object.__setattr__(self, name, box)
        if not _contains(self.query_xyxy, self.core_xyxy):
            raise ValueError("query_xyxy must contain core_xyxy")
        if not _contains(self.input_xyxy, self.query_xyxy):
            raise ValueError("input_xyxy must contain query_xyxy")

    @property
    def source_origin_xy(self) -> tuple[int, int]:
        return self.input_xyxy[:2]

    @property
    def input_hw(self) -> SizeHW:
        x0, y0, x1, y1 = self.input_xyxy
        return y1 - y0, x1 - x0

    @property
    def query_hw(self) -> SizeHW:
        x0, y0, x1, y1 = self.query_xyxy
        return y1 - y0, x1 - x0

    @property
    def core_slice_in_query(self) -> tuple[slice, slice]:
        x0, y0, x1, y1 = self.core_xyxy
        qx0, qy0, _, _ = self.query_xyxy
        return slice(y0 - qy0, y1 - qy0), slice(x0 - qx0, x1 - qx0)


def plan_tiles(
    image_hw: Iterable[int],
    core_hw: Iterable[int],
    time_offsets: Iterable[float],
    velocities_xy: Iterable[Iterable[float]],
    encoder_radius: int = 0,
    contrast_radius: int = 0,
    decoder_radius: int = 0,
    interp_radius: int = 1,
) -> list[Tile]:
    """Partition a nonempty image, with conservative, axis-specific halos.

    ``time_offsets`` is a nonempty 1-D sequence. ``velocities_xy`` has shape
    ``[V,2]`` and is nonempty. Time and velocity must use compatible units:
    e.g. original frame intervals and original pixels per frame interval.
    All radii are nonnegative integers in original pixels. Cores may exceed
    the image size; the resulting final core is clipped to the image.

    Returned tiles are ordered by increasing core y, then x. Core pixels
    occur exactly once; query/input overlap supplies context only. No query
    is emitted outside the real image, including at right/bottom edges.
    """
    image_hw = _size(image_hw, "image_hw")
    core_hw = _size(core_hw, "core_hw")
    encoder_radius = _integer(encoder_radius, "encoder_radius")
    contrast_radius = _integer(contrast_radius, "contrast_radius")
    decoder_radius = _integer(decoder_radius, "decoder_radius")
    interp_radius = _integer(interp_radius, "interp_radius")

    times = [_real(t, "time_offsets entry") for t in _items(time_offsets, "time_offsets")]
    velocities = []
    for row in _items(velocities_xy, "velocities_xy"):
        pair = _items(row, "velocities_xy row")
        if len(pair) != 2:
            raise ValueError("velocities_xy must have shape [V,2]")
        velocities.append((_real(pair[0], "vx"), _real(pair[1], "vy")))
    if not times or not velocities:
        raise ValueError("time_offsets and velocities_xy must both be nonempty")

    # Each time is paired with every velocity, so max |t*v| factors exactly.
    max_time = max(abs(t) for t in times)
    displacement = tuple(max_time * max(abs(v[axis]) for v in velocities)
                         for axis in (0, 1))
    if not all(isfinite(value) for value in displacement):
        raise ValueError("time/velocity products must be finite")
    local_support = max(encoder_radius, contrast_radius) + interp_radius
    pad_x, pad_y = (ceil(value) + local_support for value in displacement)

    height, width = image_hw
    core_h, core_w = core_hw
    tiles = []
    for y0 in range(0, height, core_h):
        for x0 in range(0, width, core_w):
            core = (x0, y0, min(width, x0 + core_w), min(height, y0 + core_h))
            query = _expand(core, decoder_radius, decoder_radius, image_hw)
            source = _expand(query, pad_x, pad_y, image_hw)
            tiles.append(Tile(core, query, source, image_hw))
    return tiles


def dense_queries(box: Iterable[int], device: Any = None, dtype: Any = None):
    """Return row-major ``[N,2]`` global ``(x,y)`` pixel-center coordinates.

    ``box`` must be a nonempty, nonnegative integer box. Pass the already
    clipped ``tile.query_xyxy`` for image-safe queries. The output has one row
    per pixel (y outermost, x innermost), not just selected high-score points.
    A floating dtype is required; the default is ``torch.float32``. Dtypes
    unable to represent every integer in the box exactly are rejected.
    """
    x0, y0, x1, y1 = _box(box, "box")
    import torch

    if dtype is None:
        dtype = torch.float32
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise ValueError("dense query dtype must be a floating torch dtype")
    largest_coordinate = max(x1 - 1, y1 - 1)
    if largest_coordinate > 2.0 / torch.finfo(dtype).eps:
        raise ValueError("dtype cannot represent every query pixel coordinate exactly")
    ys = torch.arange(y0, y1, device=device, dtype=dtype)
    xs = torch.arange(x0, x1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(-1, 2)
