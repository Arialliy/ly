"""One paired S4 training or evaluation job, resumable at optimizer boundaries."""
import argparse
from dataclasses import replace
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
from tempfile import NamedTemporaryFile

import numpy as np
import torch

from ..checks.check_s2b import write
from ..contracts import CENTER_INDEX, Prediction
from ..data.dataset import MaskClipDataset, _components, _load_gray
from ..engine.checkpoints import capture_rng, configure_determinism, restore_rng
from ..engine.s2b_train import sample_on_device
from ..evaluation.s2b_diagnostics import _frame_loss
from ..evaluation.s2b_froc import evaluate_froc
from ..geometry.tiles import Tile
from ..losses.detection import detection_loss
from ..models.hypothesis_detector import HypothesisEvidenceDetector
from ..models.video_baseline import VideoBaseline
from .s4_inference import stream_predictions
from .s4_sampling import select_hard


EXP = Path(__file__).resolve().parents[1]
PROJECT = EXP.parents[1]
MODES = ("M0", "additive_only", "raw_joint", "interaction")
MILESTONES = (0, 384, 768, 1536)


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_execution(run_dir):
    binding = read(run_dir / "execution_binding.json")
    if binding["protocol_sha256"] != digest(run_dir / "protocol.json"):
        raise ValueError("The registered S4 protocol changed")
    for path, checksum in binding["code_sha256"].items():
        if digest(PROJECT / path) != checksum:
            raise ValueError(f"Formal S4 code changed: {path}")


def atomic_save(path, writer):
    with NamedTemporaryFile(dir=path.parent, prefix=".s4-", delete=False) as out:
        tmp = Path(out.name)
        try:
            writer(out)
            out.flush()
            os.fsync(out.fileno())
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)


def make_model(mode, seed, device):
    configure_determinism()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if mode == "M0":
        model = VideoBaseline()
    else:
        model = HypothesisEvidenceDetector(mode=mode, query_chunk=1024, velocity_chunk=81,
                                           checkpoint_chunks=False)
    return model.to(device)


def make_tile(box, hw):
    # Both networks read the same native input. S1 uses the 19-pixel outer
    # halo; the M0 temporal operator only needs local spatial neighbors.
    h, w = hw
    def expand(rect, radius):
        x, y, end_x, end_y = rect
        return max(x-radius, 0), max(y-radius, 0), min(end_x+radius, w), min(end_y+radius, h)
    core = tuple(box)
    query = expand(core, 2)
    return Tile(core, query, expand(query, 19), hw)


def core_prediction(model, clip, box):
    tile = make_tile(box, clip.scene_meta[0].original_hw)
    if isinstance(model, HypothesisEvidenceDetector):
        pred = model.forward_core(clip, tile)
    else:
        source = HypothesisEvidenceDetector._crop(clip, tile, 0)
        rows, cols = tile.core_slice_in_query
        pred = Prediction(model._region_logits(source, tile)[..., rows, cols], clip.output_frame_id)
    if pred.output_frame_id != clip.output_frame_id or pred.logits.shape != (1, 1, 64, 64):
        raise ValueError("Unexpected native training core")
    return pred


def core_labels(target, clip, box):
    x, y, ex, ey = box
    mask = target.target_map[..., y:ey, x:ex]
    local_objects = []
    for obj in target.objects[0]:
        x0, y0, x1, y1 = obj.bbox_xyxy
        if x1 <= x or y1 <= y or x0 >= ex or y0 >= ey:
            continue
        if x0 < x or y0 < y or x1 > ex or y1 > ey:
            raise ValueError("Registered S4 core contains a partial target")
        cx, cy = obj.centroid_xy
        local_objects.append(replace(obj, bbox_xyxy=(x0-x, y0-y, x1-x, y1-y),
                                     centroid_xy=(cx-x, cy-y)))
    labels = replace(target, target_map=mask, ignore_map=target.ignore_map[..., y:ey, x:ex],
                     objects=(tuple(local_objects),))
    valid = clip.pixel_valid[:, CENTER_INDEX, :, y:ey, x:ex]
    return labels, valid


class Samples:
    def __init__(self, manifest, split, device):
        self.dataset = MaskClipDataset(manifest, PROJECT, split)
        self.lookup = {key: i for i, key in enumerate(self.dataset.centers)}
        self.device = device
        self.masks = {(s["id"], f["number"]): PROJECT / f["mask"] for s in manifest["sequences"]
                      if s["split"] == split for f in s["frames"]}

    def load(self, key):
        return sample_on_device(self.dataset[self.lookup[key]], self.device)

    def mask(self, frame_id):
        source, center = frame_id.split("/")
        return _load_gray(self.masks[source, int(center)])


@torch.inference_mode()
def diagnostics(model, train, dev, protocol, folder, step):
    model.eval()
    rows = []
    for row in protocol["fit_centers"]:
        key = row["source"], row["center"]
        clip, target = train.load(key)
        box = row["core_xyxy"]
        labels, valid = core_labels(target, clip, box)
        pred = core_prediction(model, clip, box)
        values = _frame_loss(pred, labels, valid)
        gt = _components(labels.target_map[0, 0].cpu().numpy().astype(np.uint8))
        if bool(gt) != row["positive"] or len(gt) > 1:
            raise ValueError("Fit panel no longer matches its original labels")
        if gt:
            y, x = np.unravel_index(pred.logits[0, 0].argmax().item(), (64, 64))
            values["hit"] = bool(math.dist((x, y), gt[0].centroid_xy) <= 3)
        rows.append({"frame_id": clip.output_frame_id[0], **values})
    pos = [r for r in rows if r["foreground_bce"] is not None]
    if len(rows) != 12 or len(pos) != 8:
        raise ValueError("Wrong fixed fit panel size")
    result = {"step": step, "fit": {"loss": float(np.mean([r["total"] for r in rows])),
              "foreground_bce": float(np.mean([r["foreground_bce"] for r in pos])),
              "top1_hits": sum(r["hit"] for r in pos), "positive_frames": len(pos), "by_frame": rows}}
    if step:
        metric_frames = []
        for name, scores in stream_predictions(model, dev.dataset, protocol["diagnostic_dev_ids"], train.device):
            gt = _components((dev.mask(name) > 0).astype(np.uint8))
            metric_frames.append((name, scores, [o.centroid_xy for o in gt]))
        curve = evaluate_froc(metric_frames, folder / f"diagnostic_froc_{step}.npz")
        result["dev"] = {"pd_at_1": curve["budgets"]["1"]["pd"], "froc": curve}
    write(folder / f"diagnostics_{step}.json", result)
    return result


def train_job(run_dir, folder, mode, seed, until, device):
    schedule_path = run_dir / f"schedule_{seed}.json"
    protocol = read(run_dir / "protocol.json")
    schedule_sha = digest(schedule_path)
    if schedule_sha != protocol["schedule_sha256"][str(seed)]:
        raise ValueError("Schedule changed")
    schedule = read(schedule_path)
    refreshes = {r["before_step"]: r for r in schedule["refreshes"]}
    manifest = read(EXP / "manifests/nudt_mirsdt_mask17.json")
    train, dev = (Samples(manifest, split, device) for split in ("train", "dev"))
    model = make_model(mode, seed, device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001, foreach=False)
    protocol_sha = digest(run_dir / "protocol.json")
    history, mined, start = [], {}, 0
    latest = folder / "latest.pt"
    def save_checkpoint(step):
        state = {"step": step, "mode": mode, "seed": seed, "model": model.state_dict(),
                 "optimizer": opt.state_dict(), "rng": capture_rng(device), "history": history,
                 "mined": mined, "schedule_sha256": schedule_sha, "protocol_sha256": protocol_sha}
        atomic_save(latest, lambda out: torch.save(state, out))
        if step in MILESTONES:
            path = folder / f"step_{step}.pt"
            if not path.exists():
                os.link(latest, path)
        write(folder / "progress.json", {"completed": step, "until": until, "mode": mode, "seed": seed,
                                         "last": history[-1] if history else None})
    if latest.exists():
        state = torch.load(latest, map_location="cpu", weights_only=True)
        if (state["mode"], state["seed"], state["schedule_sha256"], state["protocol_sha256"]) != (
                mode, seed, schedule_sha, protocol_sha):
            raise ValueError("Resume job or protocol mismatch")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        restore_rng(state["rng"], device)
        history, mined, start = state["history"], state["mined"], state["step"]
        if len(history) != start or start > until:
            raise ValueError("Unexpected resume cursor")
        if start in MILESTONES and not (folder / f"step_{start}.pt").exists():
            os.link(latest, folder / f"step_{start}.pt")
        del state
    else:
        save_checkpoint(0)
    for step in range(start, until + 1):
        if step > start:
            row = schedule["updates"][step-1]
            if row["step"] != step:
                raise ValueError("Schedule cursor mismatch")
            clip, target = train.load((row["source"], row["center"]))
            if step in refreshes:
                score_list = []
                model.eval()
                with torch.no_grad():
                    for cand in refreshes[step]["candidates"]:
                        if (cand["source"], cand["center"]) != (row["source"], row["center"]):
                            raise ValueError("Hard mining changed frame identity")
                        labels, valid = core_labels(target, clip, cand["core_xyxy"])
                        if labels.target_map.any() or labels.ignore_map.any() or not valid.all():
                            raise ValueError("Hard mining can only score valid GT-free cores")
                        score_list.append(core_prediction(model, clip, cand["core_xyxy"]).logits.max().item())
                mined[step] = score_list
            if row["kind"] == "hard":
                refresh = row["refresh_step"]
                box = select_hard(refreshes[refresh]["candidates"], mined[refresh], row["hard_rank"])["core_xyxy"]
            else:
                box = row["core_xyxy"]
            model.train()
            opt.zero_grad(set_to_none=True)
            labels, valid = core_labels(target, clip, box)
            fg = int(labels.target_map.sum())
            if bool(fg) != (row["kind"] == "positive") or (fg and fg != int(target.target_map.sum())):
                raise ValueError("Core labels disagree with scheduled kind")
            loss = detection_loss(core_prediction(model, clip, box), labels, valid)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at {step}")
            loss.backward()
            params = list(model.parameters())
            if any(p.grad is None for p in params):
                raise ValueError("Disconnected parameter")
            norm = torch.linalg.vector_norm(torch.cat([p.grad.detach().reshape(-1) for p in params]).double()).item()
            if not math.isfinite(norm):
                raise FloatingPointError(f"Nonfinite gradient at {step}")
            opt.step()
            for p in params:
                adam = opt.state[p]
                if (int(adam["step"].item()) != step or not all(torch.isfinite(t).all().item()
                        for t in (p, adam["exp_avg"], adam["exp_avg_sq"]))):
                    raise FloatingPointError(f"Invalid optimizer state at {step}")
            history.append({"step": step, "frame_id": clip.output_frame_id[0], "kind": row["kind"],
                            "core_xyxy": box, "foreground": fg, "loss": loss.item(), "gradient_l2": norm})
            del clip, target, loss, labels, valid
            if step % 64 == 0 or step == until:
                save_checkpoint(step)
                print(f"{mode}/{seed} {step}/{until}", flush=True)
        if step in MILESTONES and not (folder / f"diagnostics_{step}.json").exists():
            diagnostics(model, train, dev, protocol, folder, step)
    write(folder / f"training_{until}.json", {"mode": mode, "seed": seed, "step": until,
          "checkpoint_sha256": digest(folder / f"step_{until}.pt"), "protocol_sha256": protocol_sha,
          "schedule_sha256": schedule_sha, "peak_allocated_mib": torch.cuda.max_memory_allocated(device)/2**20,
          "history": history, "hard_refresh_scores": mined})


def evaluate_job(run_dir, folder, mode, seed, step, device):
    from ..evaluation.s4_metrics import evaluate_frames, prepare_frame
    protocol = read(run_dir / "protocol.json")
    dev = Samples(read(EXP / "manifests/nudt_mirsdt_mask17.json"), "dev", device)
    weights = folder / f"step_{step}.pt"
    bound = read(folder / f"training_{step}.json")
    if digest(weights) != bound["checkpoint_sha256"] or digest(run_dir / "protocol.json") != bound["protocol_sha256"]:
        raise ValueError("Evaluation weights or protocol mismatch")
    model = make_model(mode, seed, device)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True)["model"])
    model.eval()
    output = folder / f"eval_{step}"
    output.mkdir(exist_ok=True)
    evaluation_inputs = {"step": step, "checkpoint_sha256": bound["checkpoint_sha256"],
                         "protocol_sha256": bound["protocol_sha256"],
                         "execution_sha256": digest(run_dir / "execution_binding.json")}
    receipt = output / "inputs.json"
    if receipt.exists():
        if read(receipt) != evaluation_inputs:
            raise ValueError("Partial predictions belong to a different evaluation")
    elif any(output.iterdir()):
        raise ValueError("Unbound partial predictions in evaluation directory")
    else:
        write(receipt, evaluation_inputs)
    ids = protocol["dev_ids"]
    if len(ids) != 672 or set(ids) != {f"{s}/{n:05d}" for s,n in dev.dataset.centers}:
        raise ValueError("Unexpected formal development population")
    maps = output / "logits"
    maps.mkdir(exist_ok=True)
    paths = {name: maps/(name.replace("/", "_")+".npy") for name in ids}
    remaining = [k for k in ids if not paths[k].is_file()]
    if remaining:
        for name, logits in stream_predictions(model, dev.dataset, remaining, device):
            atomic_save(paths[name], lambda out: np.save(out, logits, allow_pickle=False))
    frames = []
    for name in ids:
        logits = np.load(paths[name], allow_pickle=False)
        frames.append(prepare_frame(name, logits, dev.mask(name)))
    evaluate_frames(frames, output, leave_one_source_out=True)
    write(output / "binding.json", {"checkpoint_sha256": bound["checkpoint_sha256"], "mode": mode, "seed": seed,
          "step": step, "protocol_sha256": bound["protocol_sha256"], "frame_sha256": {k:digest(p) for k,p in paths.items()},
          "metrics_sha256": digest(output/"metrics.json")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("action", choices=("train", "evaluate"))
    parser.add_argument("mode", choices=MODES)
    parser.add_argument("seed", type=int, choices=(46, 47, 48))
    parser.add_argument("step", type=int, choices=(768, 1536))
    args = parser.parse_args()
    verify_execution(args.run_dir)
    folder = args.run_dir / args.mode / f"seed{args.seed}"
    folder.mkdir(parents=True, exist_ok=True)
    lock = os.open(folder, os.O_RDONLY)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fn = train_job if args.action == "train" else evaluate_job
        fn(args.run_dir, folder, args.mode, args.seed, args.step, torch.device("cuda:0"))
    finally:
        os.close(lock)
