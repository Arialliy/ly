"""Paired path-readout intervention on archived, preselected development cases."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from evaluation import ReleaseEvaluation
from features import CHECKPOINT_SHA256, MODEL_SHA256, extract_features, load_release, project_logits


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SAMPLES = ROOT / "experiments/correspondence_premise_20261118_retry1/SAMPLES.jsonl"
SEQUENCE = "DJI_0067_2"
DATA = ROOT / "data/stage1" / SEQUENCE
ARMS = ("identity", "dp", "compatible", "reflected")
RADIUS = 16
TOLERANCE = 0.0002


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def cases():
    output = []
    for line in SAMPLES.read_text().splitlines():
        row = json.loads(line)
        if (row["split"] == "dev" and row["competition"]["0.05"]
                and row["explicit_candidate_competition"] and row["best_path_hits_tol1"] < 6):
            output.append(row)
    assert len(output) == 15 and sum(not row["release_matched"] for row in output) == 14
    assert all(row["sequence"] == SEQUENCE for row in output)
    return output


def path_offsets(row):
    candidates = [p for p in row["candidate_paths"]
                  if p["hits_tol1"] == 7 and p["gap_per_frame"] <= 0.05]
    chosen = max(candidates, key=lambda p: p["score"])
    correct = np.asarray(chosen["positions"]) - row["anchor"]
    best = np.asarray(row["best_path"]) - row["anchor"]
    assert correct.shape == best.shape == (8, 2)
    assert np.all(correct[-1] == 0) and np.all(best[-1] == 0)
    return np.stack((np.zeros_like(correct), best, correct, -correct)), chosen


def verify_probability(recreated, archived):
    error = float(np.max(np.abs(recreated - archived)))
    if (not np.isfinite(error) or error > TOLERANCE
            or not np.array_equal(recreated >= 0.5, archived >= 0.5)):
        raise ValueError(f"Original frozen inference mismatch, max absolute error {error}")
    return error


def matched_ids(rows):
    return {int(ann_id) for ann_id, v in rows.items() if v["matched"]}


@torch.inference_mode()
def trial(model, raw, ev, row):
    offsets, chosen = path_offsets(row)
    x, y = row["anchor"]
    x0, y0, x1, y1 = x-RADIUS, y-RADIUS, x+RADIUS+1, y+RADIUS+1
    mn, mx = offsets.min(axis=(0, 1)), offsets.max(axis=(0, 1))
    roi = (x0+int(mn[0]), y0+int(mn[1]), x1+int(mx[0]), y1+int(mx[1]))
    lx, ly = x0-roi[0], y0-roi[1]
    size = 2*RADIUS+1
    frame, aid = row["frame"], row["annotation_id"]
    assert row["slots"] == list(range(frame-7, frame+1))
    hmaps, windows = [], []
    for start in ev.covering_starts(frame):
        source = extract_features(model, raw, start, roi)
        native = source[:, :, :, ly:ly+size, lx:lx+size]
        inputs = native.repeat(len(ARMS), 1, 1, 1, 1)
        history = [(t-start, j) for j, t in enumerate(row["slots"][:-1]) if t >= start]
        for arm in range(1, len(ARMS)):
            for t, j in history:
                dx, dy = map(int, offsets[arm, j])
                inputs[arm, :, t] = source[0, :, t, ly+dy:ly+dy+size, lx+dx:lx+dx+size]
        assert torch.equal(inputs[:, :, frame-start], native[:, :, frame-start].expand_as(inputs[:, :, frame-start]))
        delta = (inputs - native).abs().amax(dim=(1, 2, 3, 4)).tolist()
        p = project_logits(model, inputs, frame-start).sigmoid().cpu().numpy()
        assert np.isfinite(p).all()
        err = verify_probability(p[0], ev.windows[start, frame][y0:y1, x0:x1])
        windows.append(dict(start=int(start), historical_slots=len(history), replay_error=err,
                            max_feature_change=dict(zip(ARMS, delta))))
        hmaps.append(p)
    fused_roi = np.maximum.reduce(hmaps)
    original = ev.fused[frame]
    fused_error = verify_probability(fused_roi[0], original[y0:y1, x0:x1])
    baseline, base_counts = ev.baseline[frame]
    base_ids = matched_ids(baseline)
    assert bool(baseline[aid]["matched"]) == bool(row["release_matched"])
    results = {}
    for arm, patch in zip(ARMS, fused_roi):
        current_map = original.copy()
        current_map[y0:y1, x0:x1] = patch
        rows, counts = ev.score(frame, current_map)
        selected = rows[aid]
        ids = matched_ids(rows)
        lost, gained = sorted(base_ids - ids), sorted(ids - base_ids)
        unmatched_change = int(counts["unmatched_components"] - base_counts["unmatched_components"])
        if arm == "identity":
            assert not lost and not gained and unmatched_change == 0
        results[arm] = dict(matched=bool(selected["matched"]), box_max=float(selected["box_max"]),
                            gained_ids=gained, lost_ids=lost, unmatched_component_delta=unmatched_change)
    return dict(sample=row["sample_id"], frame=frame, track=row["track_id"], annotation=aid,
                category=row["category_id"], original_matched=bool(row["release_matched"]),
                region=[x0,y0,x1,y1], offsets={k:v.tolist() for k,v in zip(ARMS, offsets)},
                dp_hits=row["best_path_hits_tol1"], chosen_gap=chosen["gap_per_frame"],
                original_box_max=float(baseline[aid]["box_max"]), windows=windows,
                fused_replay_error=fused_error, results=results)


def main():
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    selected = cases()
    ev = ReleaseEvaluation(sorted({r["frame"] for r in selected}))
    print("Archive and matcher verified:", ev.verification, flush=True)
    model = load_release()
    rawfile = DATA / "raw_frames.npy"
    raw = np.load(rawfile, mmap_mode="r")
    outcomes = []
    for row in selected:
        case = trial(model, raw, ev, row)
        outcomes.append(case)
        print("Verified", case["sample"], flush=True)
    results = dict(arms=ARMS, sha256=dict(samples=digest(SAMPLES), frames=digest(rawfile),
                   annotations=digest(DATA / "annotations.json"), checkpoint=CHECKPOINT_SHA256,
                   model_source=MODEL_SHA256), gpu=torch.cuda.get_device_name(),
                   archive_replay=ev.verification, cases=outcomes)
    (HERE / "RESULTS.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
