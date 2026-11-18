"""Descriptive checks for repeated images and visually similar source backgrounds."""

import argparse
from collections import Counter, defaultdict
import hashlib
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANIFEST = Path("experiments/irstd_independent_v1/manifests/nudt_mirsdt_mask17.json")


def thumbnail(image, side):
    return np.asarray(image.resize((side, side), Image.Resampling.BOX), dtype=np.float64)


def pearson(left, right):
    a = left.ravel() - left.mean()
    b = right.ravel() - right.mean()
    scale = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / scale) if scale > 0 else None


def temporal_checks(root, left, right):
    lf, rf = left["frames"], right["frames"]
    checks = []
    for quarter in range(5):
        a, b = (frames[quarter * (len(frames) - 1) // 4] for frames in (lf, rf))
        with Image.open(root / a["image"]) as ia, Image.open(root / b["image"]) as ib:
            x, y = thumbnail(ia, 96), thumbnail(ib, 96)
            checks.append({
                "left_frame": a["number"],
                "right_frame": b["number"],
                "correlation_24": pearson(thumbnail(ia, 24), thumbnail(ib, 24)),
                "correlation_96": pearson(x, y),
                "quadrant_correlations_96": [
                    pearson(x[i:i + 48, j:j + 48], y[i:i + 48, j:j + 48])
                    for i in (0, 48) for j in (0, 48)
                ],
            })
    return checks


def audit_background_overlap(project_root, manifest):
    root = Path(project_root)
    sequences = manifest["sequences"]
    thumbs = {}
    decoded = defaultdict(set)
    checked = 0
    for source in sequences:
        for i, frame in enumerate(source["frames"]):
            with Image.open(root / frame["image"]) as image:
                key = (image.size, image.mode, hashlib.sha256(image.tobytes()).digest())
                decoded[key].add(source["split"])
                if i == 0:
                    thumbs[source["id"]] = thumbnail(image, 24)
            checked += 1

    scores = []
    groups = Counter()
    for a, b in combinations(sequences, 2):
        if a["split"] == b["split"]:
            continue
        group = "-".join(sorted([a["split"], b["split"]]))
        groups[group] += 1
        score = pearson(thumbs[a["id"]], thumbs[b["id"]])
        if score is not None:
            scores.append({
                "left": a["id"], "left_split": a["split"],
                "right": b["id"], "right_split": b["split"], "correlation": score,
            })
    scores.sort(key=lambda r: (-r["correlation"], r["left"], r["right"]))
    indexed = {s["id"]: s for s in sequences}
    followup = []
    if scores:
        pair = scores[0]
        followup = temporal_checks(root, indexed[pair["left"]], indexed[pair["right"]])
    return {
        "method": (
            "Exact duplicate keys use native dimensions, mode and SHA256 of decoded pixels. "
            "Rank cross-split source pairs by Pearson correlation of the two first listed "
            "frames resized to 24x24 with Pillow BOX. Exclude flat thumbnails from ranking. "
            "For the highest-scoring pair also compare the first, quartile and last listed "
            "frames, and four fixed quadrants (TL,TR,BL,BR) at 96x96. These descriptive "
            "scores measure image similarity; physical source identities are unknown."
        ),
        "decoded_frames": checked,
        "exact_cross_split_duplicate_frames": sum(len(splits) > 1 for splits in decoded.values()),
        "cross_split_pairs": sum(groups.values()),
        "cross_split_pairs_by_group": dict(sorted(groups.items())),
        "comparable_pairs": len(scores),
        "top_pairs": scores[:5],
        "top_pair_temporal_checks": followup,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    path = args.manifest or args.project_root / MANIFEST
    report = audit_background_overlap(args.project_root, json.loads(path.read_text(encoding="utf-8")))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
