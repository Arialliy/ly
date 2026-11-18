"""Post-hoc upper bound on identities receiving any box at the sampled times."""
from collections import defaultdict
from pathlib import Path
import json


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def max_covered(track_times, counts):
    # A box can optimistically be spent on any one annotated person in its frame.
    # Assign as many different people as possible at least one of these boxes.
    slots = {frame: [(frame, j) for j in range(count)] for frame, count in counts.items()}
    owner = {}

    def assign(track, tried):
        for frame in sorted(track_times[track]):
            for slot in slots[frame]:
                if slot in tried:
                    continue
                tried.add(slot)
                if slot not in owner or assign(owner[slot], tried):
                    owner[slot] = track
                    return True
        return False

    return sum(assign(track, set()) for track in sorted(track_times))


def main():
    # The first assignment needs to move so that the second person can use frame 0.
    assert max_covered({1: {0, 1}, 2: {0}}, {0: 1, 1: 1}) == 2
    assert max_covered({1: {0, 1}, 2: {1}, 3: {1}}, {0: 2, 1: 1}) == 2
    rows = json.loads((OUT / "predictions.json").read_text())
    summary = {}
    for seq in sorted({row["sequence"] for row in rows}):
        selected = {r["frame"]: r for r in rows if r["sequence"] == seq}
        ann = json.loads((ROOT / "data/stage1" / seq / "annotations.json").read_text())
        frames = {img["id"]: img["frame_id"] for img in ann["images"]}
        seen = defaultdict(set)
        for a in ann["annotations"]:
            frame = frames[a["image_id"]]
            if a["category_id"] == 3 and frame in selected:
                seen[a["track_id"]].add(frame)
        result = {"represented_tracks": len(seen)}
        for mode in ("full", "tiled"):
            for threshold in (0.5, 0.2):
                counts = {f: sum(s >= threshold for s in r[mode]["scores"]) for f,r in selected.items()}
                best = max_covered(seen, counts)
                result[f"{mode}_{threshold}"] = {"max_tracks_covered": best, "min_tracks_without_box": len(seen)-best}
        summary[seq] = result
    (OUT / "track_ceiling.json").write_text(json.dumps(summary, indent=2)+"\n")


if __name__ == "__main__":
    main()
