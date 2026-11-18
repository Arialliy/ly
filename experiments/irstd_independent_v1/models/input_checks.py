"""Validate the S0 clip contract at the independent detector boundary."""
import torch

from ..contracts import CENTER_INDEX, CLIP_LENGTH, ClipBatch, SceneMeta
from ..geometry.tiles import Tile


def _integer_tuple(value, size):
    return isinstance(value, tuple) and len(value) == size and all(type(x) is int for x in value)


def validate_clip(clip: ClipBatch, *, tile: Tile | None = None) -> None:
    """Check the physical timeline, native geometry and raw observations before encoding.

    Without a tile, frames contain whole images padded only on the right and bottom.
    For a standalone crop, the caller supplies its single explicit Tile. Interior
    invalid observations are allowed in either case and their stored values are ignored.
    """
    if not isinstance(clip, ClipBatch):
        raise TypeError("Expected a ClipBatch from the S0 video interface")
    frames = clip.frames
    if not isinstance(frames, torch.Tensor) or frames.dtype != torch.float32:
        raise TypeError("Clip frames must be float32 raw grayscale intensities")
    if frames.ndim != 5 or tuple(frames.shape[1:3]) != (CLIP_LENGTH, 1):
        raise ValueError(f"Expected frame shape [B,{CLIP_LENGTH},1,H,W]")
    batch, _, _, height, width = frames.shape
    if min(batch, height, width) == 0:
        raise ValueError("Frame batch and spatial dimensions must be nonempty")
    if type(clip.center_index) is not int or clip.center_index != CENTER_INDEX:
        raise ValueError(f"Expected center_index={CENTER_INDEX}")
    if len(clip.output_frame_id) != batch or len(clip.scene_meta) != batch:
        raise ValueError("Each sample requires a center frame ID and scene metadata")
    if tile is not None and (not isinstance(tile, Tile) or batch != 1 or tile.input_hw != (height, width)):
        raise ValueError("A crop must contain exactly the input extent of one explicit Tile")

    valid = clip.pixel_valid
    if not isinstance(valid, torch.Tensor) or valid.dtype != torch.bool:
        raise TypeError("pixel_valid must be a boolean tensor")
    if valid.shape != frames.shape or valid.device != frames.device:
        raise ValueError("pixel_valid must match the frame shape and device")
    offsets = clip.time_offsets
    if not isinstance(offsets, torch.Tensor) or offsets.dtype != torch.float32 or offsets.requires_grad:
        raise TypeError("time_offsets must be fixed float32 geometry")
    if offsets.shape != (batch, CLIP_LENGTH) or offsets.device != frames.device:
        raise ValueError("time_offsets must match the batch, time dimension and device")
    expected_offsets = torch.arange(CLIP_LENGTH, dtype=offsets.dtype, device=offsets.device) - CENTER_INDEX
    if not torch.equal(offsets, expected_offsets.expand(batch, -1)):
        raise ValueError("time_offsets must equal the consecutive physical offsets -8 through +8")

    for index, (meta, frame_id) in enumerate(zip(clip.scene_meta, clip.output_frame_id)):
        if not isinstance(meta, SceneMeta):
            raise TypeError(f"Missing SceneMeta at batch position {index}")
        numbers = meta.frame_numbers
        if (not _integer_tuple(numbers, CLIP_LENGTH) or numbers[0] < 0
                or numbers != tuple(range(numbers[0], numbers[0] + CLIP_LENGTH))):
            raise ValueError(f"Expected 17 consecutive physical frame numbers for {meta.source_id!r}")
        if (not isinstance(meta.source_id, str) or not meta.source_id
                or frame_id != f"{meta.source_id}/{numbers[CENTER_INDEX]:05d}"):
            raise ValueError(f"Output frame ID {frame_id!r} does not match the physical center")
        if (not _integer_tuple(meta.original_hw, 2) or min(meta.original_hw) <= 0
                or not _integer_tuple(meta.tile_origin_xy, 2) or not _integer_tuple(meta.core_xyxy, 4)):
            raise ValueError(f"Invalid native pixel geometry for {frame_id}")
        native_h, native_w = meta.original_hw
        if tile is not None:
            if (meta.original_hw != tile.original_hw or meta.tile_origin_xy != tile.source_origin_xy
                    or meta.core_xyxy != tile.core_xyxy):
                raise ValueError(f"Scene metadata disagrees with the supplied Tile for {frame_id}")
        else:
            if meta.tile_origin_xy != (0, 0) or meta.core_xyxy != (0, 0, native_w, native_h):
                raise ValueError(f"Whole-image metadata required for {frame_id}. Crops need an explicit Tile")
            if height < native_h or width < native_w:
                raise ValueError(f"Frame storage is smaller than the native extent of {frame_id}")
            if valid[index, ..., native_h:, :].any() or valid[index, ..., :, native_w:].any():
                raise ValueError(f"Batch padding marked as a valid observation for {frame_id}")

    if not ((~valid) | ((frames >= 0) & (frames <= 1))).all():
        raise ValueError("Valid raw intensities must be finite and lie in [0,1]")
