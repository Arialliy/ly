"""Frozen release backbone and a trainable motion branch for the A–D experiment.

Training can use cached TPro pixels and motion evidence through forward_points.
The regular forward reads raw 40-frame thermal clips and returns center logits.
"""

import importlib.util
from functools import lru_cache
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "archive/original_project/official_deeppro"
RELEASE = SOURCE / "log/sem_seg/RGB-T__2025-01-14_15-27__SoftLoUloss_DeepPro-Plus_DataL40/checkpoints/best_model.pth"


@lru_cache(maxsize=1)
def _release_api():
    path = ROOT / "sources/temporal_diagnosis_20261010/top1_background_control_20261011/code/train_pilot.py"
    spec = importlib.util.spec_from_file_location("motion_factorial_release", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_official_detector(device="cpu"):
    detector, _ = _release_api().load_model(SOURCE, RELEASE)
    return detector.to(device).eval().requires_grad_(False)


def normalize_frames(raw_frames):
    config = _release_api().CONFIG
    return (raw_frames.float() - config["mean"]) / config["std"]


class MotionEncoder(nn.Module):
    """Read an aligned patch per velocity, with whole-clip and span channels.

    Each exchange direction uses its own velocity origin. Merge only after both
    have been encoded. The stationary and proposed moving anchors stay distinct.
    """

    def __init__(self, joint):
        super().__init__()
        visible = torch.ones(4, 5, 5, dtype=torch.bool)
        if not joint:
            visible[1:] = False
            visible[1:, 2, 2] = True
        self.register_buffer("visible", visible, persistent=False)
        self.patch = nn.Linear(9, 4)
        # For each of 4 spans: four patch features plus valid sample fraction.
        self.grid = nn.Sequential(nn.Conv2d(4 * 5, 6, 3, padding=1), nn.ReLU())
        self.anchor = nn.Sequential(nn.Linear(6 * 25 + 2, 12), nn.ReLU())
        self.direction = nn.Sequential(nn.Linear(2 * 12, 8), nn.ReLU())

    def forward(self, evidence):
        patches = evidence["patches"].to(self.patch.weight)
        fraction = evidence["valid"].to(self.patch.weight)
        velocity = evidence["velocity"].to(self.patch.weight)
        n = patches.shape[0]
        if patches.shape != (n, 2, 2, 4, 5, 5, 3, 3):
            raise ValueError("Expected [N,2 exchanges,2 anchors,4 rows,5,5,3,3] patches")
        if fraction.shape != patches.shape[:-2] or velocity.shape != (n, 2, 2, 2):
            raise ValueError("Unexpected fraction or velocity shape")

        present = (fraction > 0) & self.visible
        masked_patches = torch.where(present[..., None, None], patches, 0.0)
        localized = self.patch(masked_patches.flatten(-2)).relu()
        localized = torch.where(present[..., None], localized, 0.0)
        fraction = torch.where(present, fraction, 0.0)
        features = torch.cat((localized, fraction[..., None]), dim=-1)
        # [point, exchange, anchor, span, feature, vy, vx].
        features = features.movedim(-1, -3).reshape(n * 4, 20, 5, 5)
        grid = self.grid(features).reshape(n, 2, 2, 6 * 25)
        anchors = self.anchor(torch.cat((grid, velocity), dim=-1))
        return self.direction(anchors.flatten(-2)).mean(dim=1)


class FactorialDetector(nn.Module):
    def __init__(self, detector, variant):
        super().__init__()
        if variant not in ("A", "B", "C", "D"):
            raise ValueError("variant must be A, B, C, or D")
        self.variant = variant
        self.selection = "full" if variant in ("A", "B") else "split"
        self.detector = detector.eval().requires_grad_(False)
        self.encoder = MotionEncoder(joint=variant in ("B", "D"))
        self.projection = nn.Linear(8, 32)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        self.to(device=next(detector.parameters()).device)

    def train(self, mode=True):
        super().train(mode)
        self.detector.eval()
        return self

    @torch.no_grad()
    def temporal_features(self, normalized_frames, center=19):
        """Input [B,1,40,H,W] in release units. Return [B,32,H,W] pre-head."""
        if normalized_frames.ndim != 5 or normalized_frames.shape[1:3] != (1, 40):
            raise ValueError("Expected normalized [B,1,40,H,W] frames")
        if not 0 <= center < 40:
            raise ValueError("Center outside 40-frame clip")
        seq = self.detector.layer1(self.detector.conv_in(normalized_frames))
        temporal = self.detector.TPro(seq.permute(0, 3, 4, 1, 2))
        return temporal[:, :, center].contiguous()

    def base_points(self, features):
        """Score [N,32] through the frozen, pointwise release head."""
        if features.ndim != 2 or features.shape[1] != 32:
            raise ValueError("Expected [N,32] cached TPro features")
        if len(features) == 0:
            return features.to(self.projection.weight).new_empty(0)
        features = features.to(self.projection.weight).T[None, :, None, None, :]
        return self.detector.conv_out2(self.detector.conv_out1(features)).reshape(-1)

    def forward_points(self, features, evidence):
        """Score [N,32] cached features, one row per evidence record."""
        if len(features) != len(evidence["patches"]):
            raise ValueError("Features and evidence must have matching point counts")
        if len(features) == 0:
            return self.base_points(features)
        correction = self.projection(self.encoder(evidence))
        usable = evidence["usable"].to(device=correction.device, dtype=torch.bool)
        if usable.shape != (len(features),):
            raise ValueError("Expected one usability flag per point")
        fused = features.to(correction) + torch.where(usable[:, None], correction, 0.0)
        return self.base_points(fused)

    def center_logits(self, normalized_frames, evidence_provider, center=19, chunk_size=128):
        """Score every pixel; provider(batch_idx, xy, selection) returns evidence.

        xy are integer x,y pixel coordinates relative to the provided clip.
        """
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        features = self.temporal_features(normalized_frames, center)
        b, _, h, w = features.shape
        frame_logits = []
        for batch_idx in range(b):
            chunks = []
            for start in range(0, h * w, chunk_size):
                flat = torch.arange(start, min(start + chunk_size, h * w), device=features.device)
                xy = torch.stack((flat % w, flat // w), dim=-1)
                points = features[batch_idx, :, xy[:, 1], xy[:, 0]].T
                evidence = evidence_provider(batch_idx, xy, self.selection)
                chunks.append(self.forward_points(points, evidence))
            frame_logits.append(torch.cat(chunks).reshape(h, w))
        return torch.stack(frame_logits)

    def forward(self, raw_frames, center=19, chunk_size=128):
        """Infer [B,H,W] logits on raw 8-bit intensity clips [B,1,40,H,W]."""
        if __package__:
            from .motion_evidence import extract_evidence, prepare_window
        else:
            from motion_evidence import extract_evidence, prepare_window
        if raw_frames.ndim != 5 or raw_frames.shape[1:3] != (1, 40):
            raise ValueError("Expected raw [B,1,40,H,W] frames")
        if not 8 <= center < 32:
            raise ValueError("The 17-frame motion window must fit in the clip")
        raw_frames = raw_frames.to(self.projection.weight.device)
        windows = [prepare_window(raw[0, center-8:center+9]) for raw in raw_frames]

        def evidence_provider(batch_idx, xy, selection):
            return extract_evidence(windows[batch_idx], xy, selection=selection)

        return self.center_logits(normalize_frames(raw_frames), evidence_provider, center, chunk_size)
