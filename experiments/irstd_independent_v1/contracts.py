"""S0 contracts. Coordinates are native pixels, with (x,y) positions and (H,W) sizes."""
from dataclasses import dataclass, field
from typing import Literal

import torch


CLIP_LENGTH = 17
CENTER_INDEX = CLIP_LENGTH // 2


@dataclass(frozen=True)
class SceneMeta:
    source_id: str
    frame_numbers: tuple[int, ...]
    original_hw: tuple[int, int]
    tile_origin_xy: tuple[int, int]
    core_xyxy: tuple[int, int, int, int]  # Exclusive right and bottom in original coordinates.


@dataclass(frozen=True)
class ClipBatch:
    frames: torch.Tensor        # float32 [B,T,1,H,W], original uint8 intensity divided by 255
    time_offsets: torch.Tensor  # float32 [B,T], original frame intervals
    pixel_valid: torch.Tensor   # bool [B,T,1,H,W], false in batch padding
    center_index: int
    output_frame_id: tuple[str, ...]
    scene_meta: tuple[SceneMeta, ...]


@dataclass(frozen=True)
class MaskObject:
    """8-connected component of a released center-frame mask; no track ID is inferred."""
    area: int
    bbox_xyxy: tuple[int, int, int, int]
    centroid_xy: tuple[float, float]


@dataclass(frozen=True)
class Targets:
    label_type: Literal["mask", "bbox_center"]
    target_map: torch.Tensor  # float32 [B,1,H,W]
    ignore_map: torch.Tensor  # bool [B,1,H,W]
    objects: tuple[tuple[MaskObject, ...], ...]


@dataclass(frozen=True)
class Prediction:
    logits: torch.Tensor  # float [B,1,H,W]
    output_frame_id: tuple[str, ...]
    diagnostics: dict[str, torch.Tensor] = field(default_factory=dict)
