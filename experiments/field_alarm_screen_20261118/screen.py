"""Measure the fraction of annotation-free score-map regions with an exceedance."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CLIPS = ("DJI_0026_2", "DJI_0067_2")


def tiles(image, side):
    height, width = image.shape
    return image.reshape(height // side, side, width // side, side).transpose(0, 2, 1, 3)


def count_blocks(probabilities, frames, labels, side):
    counts = {s: [0, 0] for s in (1, 16, 32, side)}
    for index, image in frames:
        mask = np.ones(probabilities[index].shape, dtype=bool)
        height, width = mask.shape
        for x, y, w, h in labels[image["id"]]:
            mask[max(0, math.floor(y) - 4):min(height, math.ceil(y + h) + 4),
                 max(0, math.floor(x) - 4):min(width, math.ceil(x + w) + 4)] = False
        clean = tiles(mask, side).all(axis=(-2, -1))
        alarms = tiles(probabilities[index] >= 0.5, side)[clean]
        if not alarms.size:
            continue
        for small, pair in counts.items():
            blocks = alarms.reshape(len(alarms), side // small, small, side // small, small)
            active = blocks.any(axis=(2, 4))
            pair[0] += int(active.sum())
            pair[1] += active.size
    return counts


def forecast(counts, side):
    predictions = {}
    for size, (events, total) in counts.items():
        if not total:
            predictions[size] = None
            continue
        rate = events / total
        predictions[size] = 1.0 if rate == 1 else -math.expm1((side // size)**2 * math.log1p(-rate))
    return predictions


def measure(clip):
    source = ROOT / "results/tlfp_pilot_20261012/predictions/S" / clip
    meta = json.loads((source / "COMPLETE.json").read_text())
    filename = source / "official_fused.npz"
    with filename.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    assert digest == meta["fused_sha256"] and meta["diagnostic_mode"] == "spatial"
    with np.load(filename) as data:
        probabilities = data["probabilities"]
        assert (data["coverage"] > 0).all()
    assert probabilities.shape == (160, 512, 640) and np.isfinite(probabilities).all()
    annfile = ROOT / "data/stage1" / clip / "annotations.json"
    annotated = json.loads(annfile.read_text())
    images = sorted(annotated["images"], key=lambda image: image["frame_id"])
    assert [image["frame_id"] for image in images] == list(range(160))
    labels = defaultdict(list)
    for obj in annotated["annotations"]:
        labels[obj["image_id"]].append(obj["bbox"])
    frames = list(enumerate(images))
    result = {"input_sha256": digest, "annotation_sha256": hashlib.sha256(annfile.read_bytes()).hexdigest()}
    for side in (64, 128):
        train = count_blocks(probabilities, frames[:80], labels, side)
        check = count_blocks(probabilities, frames[80:], labels, side)
        result[side] = {"calibration_counts": train, "check_counts": check,
                        "prediction_from_calibration": forecast(train, side),
                        "posthoc_from_check_marginal": forecast(check, side)}
    return result


if __name__ == "__main__":
    results = {clip: measure(clip) for clip in CLIPS}
    (OUT / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
