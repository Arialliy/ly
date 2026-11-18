"""Optimistic union of the two teachers, without removing duplicate boxes."""
from collections import defaultdict
from pathlib import Path
import json

from track_ceiling import max_covered


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def main():
    coco = json.loads((OUT / "predictions.json").read_text())
    domain_file = ROOT / "experiments/rgb_domain_control_20261118/results.json"
    domain = {(r["sequence"],r["frame"]): r for r in json.loads(domain_file.read_text())["frame_counts"]}
    results = {}
    for seq in sorted({row["sequence"] for row in coco}):
        rows = [r for r in coco if r["sequence"] == seq]
        ann = json.loads((ROOT / "data/stage1" / seq / "annotations.json").read_text())
        frames = {i["id"]:i["frame_id"] for i in ann["images"]}
        selected = {r["frame"] for r in rows}
        seen = defaultdict(set)
        for obj in ann["annotations"]:
            frame = frames[obj["image_id"]]
            if obj["category_id"] == 3 and frame in selected:
                seen[obj["track_id"]].add(frame)
        caps = {k: {} for k in ("coco", "visdrone", "sum_without_dedup")}
        truth = {}
        for r in rows:
            f = r["frame"]
            other = domain[seq,f]
            assert r["G"] == other["G"]
            truth[f] = r["G"]
            caps["coco"][f] = sum(s >= 0.2 for s in r["tiled"]["scores"])
            caps["visdrone"][f] = other["tiled"]["pedestrian_and_people"]["0.2"]
            caps["sum_without_dedup"][f] = caps["coco"][f] + caps["visdrone"][f]
        result = {"tracks": len(seen), "object_frames": sum(truth.values())}
        for name, counts in caps.items():
            result[name] = {"max_object_frames": sum(min(truth[f],d) for f,d in counts.items()),
                            "min_uncovered_sampled_tracks": len(seen)-max_covered(seen,counts)}
        results[seq] = result
    (OUT / "combined_ceiling.json").write_text(json.dumps(results, indent=2)+"\n")


if __name__ == "__main__":
    main()
