"""Limited original-head adaptation with the unchanged, optional spatial LFP.

The four named prediction-convolution tensors are the only trainable original
state. Eval mode freezes every original BN; it does NOT disable autograd through
TPro. No labels, outputs, loss, or inference postprocessing are changed here.
"""
from __future__ import annotations
import copy
import torch
from torch import nn

HEAD_NAMES = frozenset({"conv_out1.0.weight", "conv_out1.0.bias",
                        "conv_out2.weight", "conv_out2.bias"})


class HeadLFPAdaptation(nn.Module):
    def __init__(self, original, feature_refiner=None, variant=None):
        super().__init__()
        for name in ("conv_in", "layer1", "TPro", "conv_out1", "conv_out2"):
            if not hasattr(original, name):
                raise ValueError(f"DeepPro-Plus lacks {name}")
        if any(hasattr(original, name) for name in ("TPro1", "TPro2", "TPro3")):
            raise ValueError("Only the frozen single-TPro Plus architecture is allowed")
        expected = "H" if feature_refiner is None else "HL"
        if variant not in (None, expected):
            raise ValueError("Variant disagrees with adapter presence")
        if feature_refiner is not None and getattr(feature_refiner, "mode", None) != "spatial":
            raise ValueError("Only the already verified spatial LFP is allowed")
        self.variant = expected
        self.original = original
        self.feature_refiner = feature_refiner
        parameters = dict(original.named_parameters())
        if not HEAD_NAMES.issubset(parameters):
            raise ValueError("Prediction layer names differ from frozen Plus")
        if not isinstance(original.conv_out1[0], nn.Conv3d) or not isinstance(original.conv_out2, nn.Conv3d):
            raise ValueError("Both permitted prediction layers must be Conv3d")
        if sum(p.numel() for p in parameters.values()) != 70913:
            raise ValueError("Original parameter count must be 70913")
        if sum(parameters[name].numel() for name in HEAD_NAMES) != 273:
            raise ValueError("Exact four prediction tensors must contain 273 parameters")
        original.requires_grad_(False)
        for name in HEAD_NAMES:
            parameters[name].requires_grad_(True)
        if feature_refiner is not None:
            feature_refiner.requires_grad_(True)
            if sum(p.numel() for p in feature_refiner.parameters()) != 100:
                raise ValueError("Spatial LFP must contain exactly 100 parameters")
        self.train(True)

    def train(self, mode=True):
        super().train(mode)
        self.original.eval()
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
                for parameter in module.parameters(recurse=False):
                    parameter.requires_grad_(False)
        return self

    def parameter_groups(self, head_lr=1e-5, adapter_lr=1e-4):
        head = [p for name, p in self.original.named_parameters() if name in HEAD_NAMES]
        groups = [{"params": head, "lr": head_lr, "name": "original_prediction_convolutions"}]
        if self.feature_refiner is not None:
            groups.append({"params": list(self.feature_refiner.parameters()), "lr": adapter_lr, "name": "spatial_lfp"})
        ids = [id(p) for group in groups for p in group["params"]]
        if len(ids) != len(set(ids)) or set(ids) != {id(p) for p in self.parameters() if p.requires_grad}:
            raise AssertionError("Optimizer parameters duplicated or incomplete")
        return groups

    def set_inference_mode(self, mode="normal"):
        if mode not in ("normal", "bypass"):
            raise ValueError("Only normal/bypass are allowed in the spatial experiment")
        if self.feature_refiner is not None:
            self.feature_refiner.set_inference_mode(mode)
        return self

    def forward(self, seq_imgs):
        features = self.original.layer1(self.original.conv_in(seq_imgs))
        if self.feature_refiner is not None:
            features = self.feature_refiner(features)
        # Do not wrap this in no_grad: LFP must learn through the frozen TPro.
        features = self.original.TPro(features.permute(0, 3, 4, 1, 2))
        features = self.original.conv_out1(features)
        logits = self.original.conv_out2(features).squeeze(dim=1)
        return features, logits


def build_pair(release_model, spatial_init):
    """Independent models with equal release tensors, not shared parameters."""
    return (HeadLFPAdaptation(copy.deepcopy(release_model)),
            HeadLFPAdaptation(copy.deepcopy(release_model), copy.deepcopy(spatial_init)))
