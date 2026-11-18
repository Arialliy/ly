"""Sparse motion-residual adapter for the official RGB-T DeepPro-Plus 40-frame release.

Run the CPU identity/gradient check in the project's fixed container:
    PYTHONPATH=/project/environment/site python /project/experiments/motion_hypothesis_traffic_pilot_20261118/deeppro_motion_adapter.py

Mount the project at /project and allow writes only in this experiment's runs directory.
The default check reads one training crop and saves runs/adapter_check.json.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "archive/original_project/official_deeppro"
RELEASE = SOURCE / "log/sem_seg/RGB-T__2025-01-14_15-27__SoftLoUloss_DeepPro-Plus_DataL40/checkpoints/best_model.pth"
MOTION_SHAPE = (2, 2, 3, 5, 5)


def release_loader():
    path = ROOT / "sources/temporal_diagnosis_20261010/top1_background_control_20261011/code/train_pilot.py"
    spec = importlib.util.spec_from_file_location("project_release_load", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DeepProMotionAdapter(nn.Module):
    """One 300-value motion map per (t,y,x) on the original logit grid.

    Supply unique integral positions [B,N,3] and maps [B,N,2,2,3,5,5]
    or [B,N,300]. With no positions and maps, or enabled=False, use the
    original two-output forward. Frozen batch normalization always uses release statistics.
    """
    def __init__(self, detector, hidden=16):
        super().__init__()
        self.detector = detector.eval().requires_grad_(False)
        self.residual = nn.Sequential(nn.Linear(300, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def train(self, mode=True):
        super().train(mode)
        self.detector.eval()
        return self

    def forward(self, frames, positions=None, motion_maps=None, *, enabled=True):
        features, logits = self.detector(frames)
        if not enabled or (positions is None and motion_maps is None):
            return features, logits
        if positions is None or motion_maps is None:
            raise ValueError("Both motion_maps and positions must be supplied.")
        b, t, h, w = logits.shape
        if positions.ndim != 3 or positions.shape[0] != b or positions.shape[2] != 3:
            raise ValueError("Expected positions of shape [B,N,3] (t,y,x).")
        n = positions.shape[1]
        if motion_maps.shape not in ((b, n, 300), (b, n, *MOTION_SHAPE)):
            raise ValueError("Unexpected motion map shape.")
        if positions.dtype not in (torch.int32, torch.int64):
            raise TypeError("Positions must be integers.")
        if n == 0:
            return features, logits
        pos = positions.to(device=logits.device, dtype=torch.long)
        if torch.any(pos < 0) or torch.any(pos >= pos.new_tensor((t, h, w))):
            raise ValueError("Motion position outside output grid.")
        index = (pos[..., 0] * h + pos[..., 1]) * w + pos[..., 2]
        ordered = index.sort(dim=1).values
        if torch.any(ordered[:, 1:] == ordered[:, :-1]):
            raise ValueError("A logit cannot have multiple residuals.")
        dense = logits.reshape(b, -1)
        motion_maps = motion_maps.reshape(b, n, 300).to(self.residual[0].weight)
        delta = self.residual(motion_maps).squeeze(-1).to(logits)
        corrected = dense.scatter(1, index, dense.gather(1, index) + delta)
        return features, corrected.reshape_as(logits)


def main():
    from extract import features as motion_features

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(46)
    api = release_loader()
    detector, model_source = api.load_model(SOURCE, RELEASE)
    model = DeepProMotionAdapter(detector).train()
    sequence = "DJI_0043_2"
    train = json.loads((ROOT / "results/references/DATA_SELECTION.json").read_text())["splits"]["train"]
    assert sequence in train and all(not m.training for m in detector.modules())
    raw_file = ROOT / "data/stage1" / sequence / "raw_frames.npy"
    raw = np.load(raw_file, mmap_mode="r")[0:40, 216:296, 280:360].copy()
    center = 20
    motion = motion_features(raw[center-8:center+9])
    n = min(4, len(motion["maps"]))
    if n < 1:
        raise RuntimeError("No raw-image candidate in the fixed training crop.")
    xy = motion["xy"][:n]
    positions = torch.from_numpy(np.column_stack([np.full(n, center), xy[:, 1], xy[:, 0]]))[None]
    maps = torch.from_numpy(motion["maps"][:n])[None]
    frames = ((torch.from_numpy(raw).float() - api.CONFIG["mean"]) / api.CONFIG["std"])[None, None]

    with torch.no_grad():
        expected_feats, expected = detector(frames)
        base_features, initialized = model(frames, positions, maps)
        _, bypassed = model(frames, positions, maps, enabled=False)
        _, no_inputs = model(frames)
        _, empty = model(frames, positions[:, :0], maps[:, :0])
        assert torch.equal(expected_feats, base_features)
        assert all(torch.equal(actual, expected) for actual in (initialized, bypassed, no_inputs, empty))
        model.residual[-1].bias.fill_(0.25)
        _, altered = model(frames, positions, maps)
        changed = torch.zeros_like(expected, dtype=torch.bool)
        changed[0, positions[0, :, 0], positions[0, :, 1], positions[0, :, 2]] = True
        delta = altered - expected
        assert torch.all(delta[~changed] == 0) and torch.allclose(delta[changed], torch.full((n,), 0.25), atol=2e-6, rtol=0)
        model.residual[-1].bias.zero_()

    _, train_logits = model(frames, positions, maps)
    train_logits[changed].sum().backward()
    grad = float(model.residual[-1].weight.grad.abs().sum())
    assert grad > 0 and model.residual[-1].bias.grad.item() == n
    assert all(p.grad is None and not p.requires_grad for p in model.detector.parameters())
    result = {"kind": "training-crop integration check, untrained adapter; does not evaluate detection",
              "software": {"torch": str(torch.__version__), "device": "cpu", "seed": 46},
              "release": str(RELEASE.relative_to(ROOT)), "sha256_release": api.sha256(RELEASE),
              "dated_detector_source": str(model_source.relative_to(ROOT)), "sha256_dated_detector_source": api.sha256(model_source),
              "strict_model_state_load": True, "loader": "project top1_background_control train_pilot.load_model",
              "sha256_adapter": api.sha256(__file__), "sha256_motion_extractor": api.sha256(HERE / "extract.py"),
              "raw_file": str(raw_file.relative_to(ROOT)), "sha256_raw_file": api.sha256(raw_file),
              "source_sequence": sequence, "split": "train", "crop_tyx_exclusive": [[0,40],[216,296],[280,360]],
              "clip_shape": list(frames.shape), "logit_shape": list(expected.shape), "feature_shape": list(expected_feats.shape),
              "normalization_mean_std": [api.CONFIG["mean"],api.CONFIG["std"]], "rescored_tyx": positions[0].tolist(),
              "initial_max_abs_logit_diff": float((initialized-expected).abs().max()),
              "bypass_none_empty_exact": True, "features_exact": True, "nonzero_bias": 0.25,
              "changed_voxels": int(torch.count_nonzero(delta)), "max_abs_unselected_change": float(delta[~changed].abs().max()),
              "last_layer_gradient_l1": grad, "frozen_release_has_no_grads": True, "optimizer_updates": 0}
    out = HERE / "runs/adapter_check.json"
    temp = out.with_suffix(".tmp")
    temp.write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n")
    temp.replace(out)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
