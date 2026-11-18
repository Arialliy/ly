"""M0's spatial detector with its video path replaced by time–velocity evidence."""
from dataclasses import replace
from math import ceil
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import yaml

from ..contracts import CENTER_INDEX, CLIP_LENGTH, ClipBatch, Prediction
from ..geometry import make_velocity_grid, sample_context
from ..geometry.tiles import Tile, dense_queries, plan_tiles
from .evidence_encoder import EvidenceEncoder
from .input_checks import validate_clip
from .video_baseline import VideoBaseline, _query_slices


class HypothesisAggregator(nn.Module):
    """Sample fixed physical paths, then encode all axes before spatial decoding."""

    def __init__(self, mode, query_chunk, velocity_chunk, checkpoint_chunks):
        super().__init__()
        if any(type(n) is not int or n <= 0 for n in (query_chunk, velocity_chunk)):
            raise ValueError("Query and velocity chunk sizes must be positive integers")
        if type(checkpoint_chunks) is not bool:
            raise TypeError("checkpoint_chunks must be a boolean")
        config = Path(__file__).resolve().parents[1] / "configs/s1_formal.yaml"
        grid = yaml.safe_load(config.read_bytes())["velocity_grid"]
        self.register_buffer("velocities_xy", make_velocity_grid(grid["vx"], grid["vy"]))
        context_indices = [i for i in range(CLIP_LENGTH) if i != CENTER_INDEX]
        self.register_buffer("context_indices", torch.tensor(context_indices), persistent=False)
        self.encoder = EvidenceEncoder(mode=mode)
        self.query_chunk = query_chunk
        self.velocity_chunk = velocity_chunk
        self.checkpoint_chunks = checkpoint_chunks

    def forward(self, frame_features, contrast, clip, query_region=None):
        # prepare the context once per source crop; source/observation masks remain distinct
        features = torch.cat((frame_features, contrast), dim=2).index_select(1, self.context_indices)
        observed = clip.pixel_valid.index_select(1, self.context_indices)
        offsets = clip.time_offsets.index_select(1, self.context_indices)
        image_hw = [m.original_hw for m in clip.scene_meta]
        origins = [m.tile_origin_xy for m in clip.scene_meta]
        rows, cols = _query_slices(query_region)
        center = frame_features[:, CENTER_INDEX, :, rows, cols]
        batch, _, height, width = center.shape
        box = query_region.query_xyxy if query_region is not None else (0, 0, width, height)
        source_hw = [query_region.input_hw] if query_region is not None else image_hw
        queries = dense_queries(box, device=features.device, dtype=features.dtype)
        center = center.flatten(2).transpose(1, 2)

        def encode_chunk(context, positions, appearance):
            evidence = sample_context(context, offsets, positions, self.velocities_xy, observed,
                                      image_hw, origins, source_hw, query_chunk=self.query_chunk,
                                      velocity_chunk=self.velocity_chunk)
            return self.encoder(evidence, appearance)

        rows_out = []
        for start in range(0, len(queries), self.query_chunk):
            stop = start + self.query_chunk
            args = (features, queries[start:stop], center[:, start:stop])
            if self.checkpoint_chunks and self.training and torch.is_grad_enabled():
                # Recompute sampling AND encoding. Retaining every sampled volume until
                # backward would defeat the point of streaming over query positions.
                block = checkpoint(encode_chunk, *args, use_reentrant=False, preserve_rng_state=False)
            else:
                block = encode_chunk(*args)
            rows_out.append(block)
        fused = torch.cat(rows_out, dim=1).transpose(1, 2).reshape(batch, 32, height, width)
        return fused.masked_fill(~clip.pixel_valid[:, CENTER_INDEX, :, rows, cols], 0)


class HypothesisEvidenceDetector(VideoBaseline):
    """Use the same trainable frame encoder, decoder, head and full-frame entry as M0."""

    def __init__(self, mode="interaction", query_chunk=64, velocity_chunk=8, checkpoint_chunks=True):
        super().__init__()
        self.video_aggregator = HypothesisAggregator(mode, query_chunk, velocity_chunk, checkpoint_chunks)

    def tiles(self, image_hw, core_hw=(64, 64)):
        grid = self.video_aggregator.velocities_xy.detach().reshape(-1, 2)
        return plan_tiles(image_hw, core_hw, range(-CENTER_INDEX, CENTER_INDEX + 1), grid,
                          encoder_radius=self.frame_encoder.radius, contrast_radius=self.contrast_channel.radius,
                          decoder_radius=self.spatial_decoder.radius, interp_radius=1)

    def _check_halo(self, tile):
        # Explicitly supplied tiles need enough support for the encoder and decoder as
        # well as for interpolation. The S1 sampler alone can only detect the latter.
        h, w = tile.original_hw
        def expand(box, pad_x, pad_y):
            x0, y0, x1, y1 = box
            return max(x0-pad_x, 0), max(y0-pad_y, 0), min(x1+pad_x, w), min(y1+pad_y, h)

        def covers(outer, inner):
            return all(outer[i] <= inner[i] for i in (0, 1)) and all(outer[i] >= inner[i] for i in (2, 3))

        speed = self.video_aggregator.velocities_xy.abs().amax(dim=(0, 1)).tolist()
        local = max(self.frame_encoder.radius, self.contrast_channel.radius) + 1
        pad_x, pad_y = (ceil(v * CENTER_INDEX) + local for v in speed)
        radius = self.spatial_decoder.radius
        if (not covers(tile.query_xyxy, expand(tile.core_xyxy, radius, radius)) or
                not covers(tile.input_xyxy, expand(tile.query_xyxy, pad_x, pad_y))):
            raise ValueError("Tile lacks the decoder, motion, interpolation or encoder halo")

    @staticmethod
    def _crop(clip, tile, index):
        x0, y0, x1, y1 = tile.input_xyxy
        batch = slice(index, index + 1)
        meta = replace(clip.scene_meta[index], tile_origin_xy=tile.source_origin_xy, core_xyxy=tile.core_xyxy)
        return ClipBatch(frames=clip.frames[batch, :, :, y0:y1, x0:x1], time_offsets=clip.time_offsets[batch],
                         pixel_valid=clip.pixel_valid[batch, :, :, y0:y1, x0:x1], center_index=CENTER_INDEX,
                         output_frame_id=(clip.output_frame_id[index],), scene_meta=(meta,))

    def _core_prediction(self, source, tile):
        rows, cols = tile.core_slice_in_query
        logits = self._region_logits(source, tile)[..., rows, cols]
        return Prediction(logits, source.output_frame_id)

    def forward_tile(self, source: ClipBatch, tile: Tile) -> Prediction:
        """Train or predict directly from an explicitly cropped, halo-bearing source."""
        validate_clip(source, tile=tile)
        self._check_halo(tile)
        return self._core_prediction(source, tile)

    def forward_core(self, clip: ClipBatch, tile: Tile, sample_index: int = 0) -> Prediction:
        """Extract one halo-bearing tile from a whole-image batch and return only its core."""
        validate_clip(clip)
        if type(sample_index) is not int or not 0 <= sample_index < len(clip.scene_meta):
            raise ValueError("Invalid batch sample_index")
        if tile.original_hw != clip.scene_meta[sample_index].original_hw:
            raise ValueError("Tile belongs to a different native image geometry")
        self._check_halo(tile)
        return self._core_prediction(self._crop(clip, tile, sample_index), tile)

    @torch.no_grad()
    def predict_tiled(self, clip: ClipBatch, core_hw=(64, 64)) -> Prediction:
        validate_clip(clip)
        logits = torch.zeros_like(clip.frames[:, CENTER_INDEX])
        for index, meta in enumerate(clip.scene_meta):
            for tile in self.tiles(meta.original_hw, core_hw):
                core = self._core_prediction(self._crop(clip, tile, index), tile)
                x0, y0, x1, y1 = tile.core_xyxy
                logits[index:index+1, :, y0:y1, x0:x1] = core.logits
        return Prediction(logits, clip.output_frame_id)
