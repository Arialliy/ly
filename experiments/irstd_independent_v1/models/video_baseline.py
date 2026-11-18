"""M0: native-resolution fixed-position video aggregation with a separate centre-image feature."""
from dataclasses import replace

import torch
from torch import nn
from torch.nn import functional as F

from ..contracts import CENTER_INDEX, CLIP_LENGTH, ClipBatch, Prediction
from ..geometry.tiles import Tile, plan_tiles
from .frame_encoder import ContrastChannel, FrameEncoder
from .input_checks import validate_clip
from .spatial_decoder import SpatialDecoder


def _query_slices(tile: Tile | None) -> tuple[slice, slice]:
    if tile is None:
        return slice(None), slice(None)
    x0, y0, x1, y1 = tile.query_xyxy
    ox, oy = tile.source_origin_xy
    return slice(y0 - oy, y1 - oy), slice(x0 - ox, x1 - ox)


class TemporalAggregator(nn.Module):
    def __init__(self):
        super().__init__()
        self.temporal = nn.ModuleList()
        in_channels = 11
        for dilation in (1, 2, 4, 1):
            self.temporal.append(nn.Conv3d(in_channels, 16, (3, 1, 1),
                                           dilation=(dilation, 1, 1), padding=(dilation, 0, 0)))
            in_channels = 16
        self.fusion = nn.Conv2d(25, 32, 1)

    def temporal_features(self, frame_features: torch.Tensor, contrast: torch.Tensor,
                          clip: ClipBatch, query_region: Tile | None = None):
        """Return the 16-channel context readout and observed fraction on the requested grid."""
        rows, cols = _query_slices(query_region)
        valid = clip.pixel_valid[..., rows, cols]
        # The centre observation is withheld, but this time slot's hidden state remains usable.
        observable = valid.clone()
        observable[:, CENTER_INDEX] = False
        image_evidence = torch.cat((frame_features[..., rows, cols], contrast[..., rows, cols]), dim=2)
        image_evidence = image_evidence.masked_fill(~observable, 0)
        time = (clip.time_offsets / CENTER_INDEX)[:, :, None, None, None]
        time_channel = time.expand_as(valid).masked_fill(~valid, 0)
        hidden = torch.cat((image_evidence, time_channel, observable.to(time.dtype)), dim=2)
        hidden = hidden.transpose(1, 2).contiguous()
        spatial_valid = valid.transpose(1, 2)
        for layer in self.temporal:
            hidden = F.silu(layer(hidden)).masked_fill(~spatial_valid, 0)
        coverage = observable.sum(dim=1).to(frame_features.dtype) / (CLIP_LENGTH - 1)
        return hidden[:, :, CENTER_INDEX], coverage

    def forward(self, frame_features: torch.Tensor, contrast: torch.Tensor,
                clip: ClipBatch, query_region: Tile | None = None) -> torch.Tensor:
        rows, cols = _query_slices(query_region)
        context, coverage = self.temporal_features(frame_features, contrast, clip, query_region)
        center = frame_features[:, CENTER_INDEX, :, rows, cols]
        fused = self.fusion(torch.cat((context, center, coverage), dim=1))
        return F.silu(fused).masked_fill(~clip.pixel_valid[:, CENTER_INDEX, :, rows, cols], 0)


class VideoBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.frame_encoder = FrameEncoder()
        self.contrast_channel = ContrastChannel()
        self.video_aggregator = TemporalAggregator()
        self.spatial_decoder = SpatialDecoder()
        self.detection_head = nn.Conv2d(32, 1, 1)

    def _region_logits(self, clip: ClipBatch, tile: Tile | None = None) -> torch.Tensor:
        # The aggregator sees source-sized features and global Tile metadata. Its output uses the
        # query grid (including the decoder's halo), the same interface needed by S3.
        features = self.frame_encoder(clip.frames, clip.pixel_valid)
        contrast = self.contrast_channel(clip.frames, clip.pixel_valid)
        query_features = self.video_aggregator(features, contrast, clip, tile)
        rows, cols = _query_slices(tile)
        center_valid = clip.pixel_valid[:, CENTER_INDEX, :, rows, cols]
        decoded = self.spatial_decoder(query_features, center_valid)
        return self.detection_head(decoded).masked_fill(~center_valid, 0)

    def forward(self, clip: ClipBatch) -> Prediction:
        """Produce full native-resolution logits with bottom/right padding on mixed-size batches."""
        validate_clip(clip)
        return Prediction(self._region_logits(clip), clip.output_frame_id)

    @torch.no_grad()
    def predict_tiled(self, clip: ClipBatch, core_hw: tuple[int, int] = (64, 64)) -> Prediction:
        """Tile each native scene separately and write each final core once at global coordinates."""
        validate_clip(clip)
        logits = torch.zeros_like(clip.frames[:, CENTER_INDEX])
        for b, meta in enumerate(clip.scene_meta):
            tiles = plan_tiles(meta.original_hw, core_hw, (0,), ((0, 0),),
                               encoder_radius=self.frame_encoder.radius,
                               contrast_radius=self.contrast_channel.radius,
                               decoder_radius=self.spatial_decoder.radius, interp_radius=0)
            for tile in tiles:
                x0, y0, x1, y1 = tile.input_xyxy
                sample = slice(b, b + 1)
                source = ClipBatch(frames=clip.frames[sample, :, :, y0:y1, x0:x1],
                                   time_offsets=clip.time_offsets[sample],
                                   pixel_valid=clip.pixel_valid[sample, :, :, y0:y1, x0:x1],
                                   center_index=CENTER_INDEX, output_frame_id=(clip.output_frame_id[b],),
                                   scene_meta=(replace(meta, tile_origin_xy=tile.source_origin_xy,
                                                       core_xyxy=tile.core_xyxy),))
                query_logits = self._region_logits(source, tile)
                rows, cols = tile.core_slice_in_query
                x0, y0, x1, y1 = tile.core_xyxy
                logits[sample, :, y0:y1, x0:x1] = query_logits[..., rows, cols]
        return Prediction(logits, clip.output_frame_id)
