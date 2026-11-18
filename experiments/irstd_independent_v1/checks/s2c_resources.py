"""Measure a complete, warmed M0 training update in a fresh CUDA process."""
import argparse
import json
import math
import os
from pathlib import Path
from time import perf_counter

import torch

from .check_s2b import write
from ..contracts import CENTER_INDEX
from ..data.dataset import MaskClipDataset
from ..engine.checkpoints import restore_training
from ..engine.s2b_train import sample_on_device
from ..losses.detection import detection_loss


EXPERIMENT = Path(__file__).resolve().parents[1]
PROJECT = EXPERIMENT.parents[1]
MIB = 1024 ** 2


def check_update(model, optimizer, expected_step):
    for name, param in model.named_parameters():
        buffers = optimizer.state[param]
        if int(buffers["step"].item()) != expected_step or param.grad is None:
            raise ValueError(f"Incomplete AdamW update for {name} at step {expected_step}")
        tensors = (param.detach(), param.grad, buffers["exp_avg"], buffers["exp_avg_sq"])
        if not all(torch.isfinite(value).all().item() for value in tensors):
            raise FloatingPointError(f"Nonfinite training state for {name} at step {expected_step}")


def run(run_dir, device, report):
    protocol = json.loads((run_dir / "protocol.json").read_bytes())
    if protocol["resource_sample_id"] != "Sequence32/00009" or protocol["allocated_budget_mib"] != 8192:
        raise ValueError("Resource protocol differs from the registered sample or budget")
    s2b = json.loads((EXPERIMENT / "reports/S2b.json").read_bytes())
    if (s2b["checkpoint"]["sha256"] != protocol["checkpoint_sha256"] or
            os.environ.get("S2_IMAGE_ID") != s2b["environment"]["image"]):
        raise ValueError("Resource run requires the bound checkpoint and original Docker image")
    if device.type != "cuda":
        raise ValueError("Training memory gate requires CUDA")
    path = EXPERIMENT / s2b["run_dir"] / s2b["checkpoint"]["path"]
    model, optimizer, restored = restore_training(path, protocol["checkpoint_sha256"], EXPERIMENT, device)
    if restored["stage"] != "S2b" or restored["global_step"] != 400:
        raise ValueError("Expected the original S2b step-400 training state")
    del restored
    manifest = json.loads((EXPERIMENT / "manifests/nudt_mirsdt_mask17.json").read_bytes())
    training = [row for row in manifest["sequences"] if row["split"] == "train"]
    largest = max(math.prod(row["native_shape_hw"]) for row in training)
    largest_ids = [row["id"] for row in training if math.prod(row["native_shape_hw"]) == largest]
    if largest_ids != ["Sequence32"] or largest != 272 * 407:
        raise ValueError("Expected Sequence32 to have the largest native training area")
    dataset = MaskClipDataset(manifest, PROJECT, "train")
    clip, targets = sample_on_device(dataset[dataset.centers.index(("Sequence32", 9))], device)
    if (clip.output_frame_id != (protocol["resource_sample_id"],)
            or tuple(clip.frames.shape) != (1, 17, 1, 272, 407) or clip.frames.dtype != torch.float32
            or not clip.pixel_valid.all().item() or targets.ignore_map.any().item()):
        raise ValueError("Native, complete Sequence32 training clip was expected")
    props = torch.cuda.get_device_properties(device)
    report.update(sample_id=protocol["resource_sample_id"], input_shape=list(clip.frames.shape),
                  native_hw=[272, 407], train_sources=len(training), largest_area_pixels=largest,
                  precision="FP32", batch_size=1, checkpoint_sha256=protocol["checkpoint_sha256"],
                  restored_optimizer_step=400, allocated_budget_mib=protocol["allocated_budget_mib"],
                  optimizer="AdamW", environment={"gpu": props.name, "total_device_bytes": props.total_memory,
                  "host_gpu_index": os.environ.get("S2_GPU_INDEX"), "gpu_uuid": os.environ.get("S2_GPU_UUID"),
                  "image": os.environ["S2_IMAGE_ID"], "torch": str(torch.__version__), "cuda": torch.version.cuda})
    model.train()
    for phase, step in (("warmup", 401), ("measured", 402)):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        if phase == "measured":
            torch.cuda.reset_peak_memory_stats(device)
            before_allocated = torch.cuda.memory_allocated(device)
            before_reserved = torch.cuda.memory_reserved(device)
        start = perf_counter()
        prediction = model(clip)
        torch.cuda.synchronize(device)
        after_forward = perf_counter()
        loss = detection_loss(prediction, targets, clip.pixel_valid[:, CENTER_INDEX])
        torch.cuda.synchronize(device)
        after_loss = perf_counter()
        loss.backward()
        torch.cuda.synchronize(device)
        after_backward = perf_counter()
        optimizer.step()
        torch.cuda.synchronize(device)
        end = perf_counter()
        report["diagnostic_optimizer_updates"] += 1
        if phase == "measured":
            report["measured_updates"] += 1
            allocated = torch.cuda.max_memory_allocated(device)
            reserved = torch.cuda.max_memory_reserved(device)
            report["memory"] = {"allocated_before_bytes": before_allocated, "reserved_before_bytes": before_reserved,
                                "peak_allocated_bytes": allocated, "peak_allocated_mib": allocated / MIB,
                                "peak_reserved_bytes": reserved, "peak_reserved_mib": reserved / MIB}
            report["timing_ms"] = {"forward": 1000 * (after_forward - start),
                                   "loss": 1000 * (after_loss - after_forward),
                                   "backward": 1000 * (after_backward - after_loss),
                                   "optimizer_step": 1000 * (end - after_backward),
                                   "total": 1000 * (end - start)}
        scalar = loss.detach().item()
        if prediction.output_frame_id != clip.output_frame_id or not math.isfinite(scalar):
            raise ValueError(f"Invalid output or loss during {phase}")
        report[phase + "_loss"] = scalar
        check_update(model, optimizer, step)
        del prediction, loss
    report["allocated_within_budget"] = report["memory"]["peak_allocated_bytes"] <= report["allocated_budget_mib"] * MIB
    if not report["allocated_within_budget"]:
        raise RuntimeError("Peak CUDA allocation exceeds 8192 MiB")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    path = args.run_dir / "resources.json"
    if path.exists():
        raise FileExistsError(path)
    report = {"stage": "S2c-3", "diagnostic_optimizer_updates": 0, "measured_updates": 0,
              "measurement_scope": "Single synchronized FP32 training update with resident input, model and AdamW."}
    try:
        run(args.run_dir, torch.device(args.device), report)
        report["status"] = "PASS"
    except Exception as exc:
        report.update(status="FAIL", error=str(exc))
        raise
    finally:
        write(path, report)


if __name__ == "__main__":
    main()
