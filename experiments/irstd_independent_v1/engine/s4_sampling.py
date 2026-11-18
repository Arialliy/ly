"""Build training-only native-core schedules and rank current-model hard negatives."""
from collections import Counter, defaultdict
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import random

from PIL import Image

from ..data.s2b_subset import archive_member


CATEGORIES = ("positive", "random_background", "positive", "empty",
              "positive", "random_background", "positive", "hard")


def _digest(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def _rng(*key):
    return random.Random(int(_digest(key), 16))


def build_training_index(project_root, manifest):
    """Read only native training-center masks, verifying them against S0's release inventory."""
    project_root = Path(project_root)
    exp = project_root / "experiments/irstd_independent_v1"
    s0 = json.loads((exp / "reports/S0.json").read_bytes())
    raw = (exp / "manifests/nudt_mirsdt_mask17.json").read_bytes()
    if (s0["status"] != "PASS" or hashlib.sha256(raw).hexdigest() != s0["data_manifest_hash"]
            or json.loads(raw) != manifest):
        raise ValueError("S4 training requires the frozen S0 manifest")
    provenance = s0["pixel_audit"]["provenance"]
    raw = (project_root / provenance["inventory"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != provenance["inventory_sha256"]:
        raise ValueError("Frozen release inventory mismatch")
    inventory = {entry["member"]: entry["sha256"] for entry in json.loads(raw)["files"]}
    sources = {}
    positive = 0
    empty = 0
    for sequence in manifest["sequences"]:
        if sequence["split"] != "train":
            continue
        name = sequence["id"]
        if sequence["official_split"] != "train" or name in sources:
            raise ValueError(f"Invalid training source {name}")
        height, width = sequence["native_shape_hw"]
        frames = {frame["number"]: frame for frame in sequence["frames"]}
        centers = []
        for number in sequence["centers"]:
            raw, info = archive_member(project_root, frames[number]["mask"], inventory)
            with Image.open(BytesIO(raw)) as mask:
                if mask.format != "PNG" or mask.mode != "L" or mask.size != (width, height):
                    raise ValueError(f"Invalid native mask at {info['path']}")
                counts = mask.histogram()
                if counts[0] + counts[255] != width * height:
                    raise ValueError(f"Non-binary/ignore mask at {info['path']}")
                box = mask.getbbox()
            centers.append({"center": number, "foreground_box": list(box) if box else None,
                            "foreground_pixels": counts[255]})
            positive += int(box is not None)
            empty += int(box is None)
        sources[name] = {"native_hw": [height, width], "centers": centers}
    expected = s0["pixel_audit"]["splits"]["train"]
    if (positive, empty, len(sources)) != (expected["positive_centers"], expected["empty_centers"],
                                            expected["sources"]):
        raise ValueError("Training-center population no longer agrees with S0")
    return {"split": "train", "manifest_sha256": s0["data_manifest_hash"], "sources": sources,
            "positive_centers": positive, "empty_centers": empty}


def _draw_centers(pools, seed, name):
    """Each complete round visits every eligible source; deplete its centers before reusing."""
    if not pools:
        raise ValueError(f"No training centers available for {name}")
    visits = Counter()
    center_orders = {}
    round_id = 0
    while True:
        sources = sorted(pools)
        _rng(seed, name, "sources", round_id).shuffle(sources)
        for source in sources:
            eligible = pools[source]
            epoch, slot = divmod(visits[source], len(eligible))
            if slot == 0:
                order = list(eligible)
                _rng(seed, name, "centers", source, epoch).shuffle(order)
                center_orders[source] = order
            visits[source] += 1
            yield source, center_orders[source][slot]
        round_id += 1


def _positive_box(rng, size, native_hw, gt):
    h, w = native_hw
    x0, y0, x1, y1 = gt
    if x1 - x0 > size or y1 - y0 > size:
        raise ValueError(f"The entire positive mask {gt} does not fit the training core")
    x = rng.randint(max(0, x1 - size), min(x0, w - size))
    y = rng.randint(max(0, y1 - size), min(y0, h - size))
    return [x, y, x + size, y + size]


def _negative_boxes(rng, size, native_hw, gt, count=1):
    """Sample distinct native top-left pixels outside all GT-intersecting core placements."""
    h, w = native_hw
    nx, ny = w - size + 1, h - size + 1
    if gt is None:
        regions = [(0, 0, nx, ny)]
    else:
        x0, y0, x1, y1 = gt
        lo_x, hi_x = max(0, x0 - size + 1), min(nx, x1)
        lo_y, hi_y = max(0, y0 - size + 1), min(ny, y1)
        regions = [(0, 0, nx, lo_y), (0, hi_y, nx, ny), (0, lo_y, lo_x, hi_y), (hi_x, lo_y, nx, hi_y)]
    areas = [(x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in regions]
    boxes = []
    for offset in rng.sample(range(sum(areas)), min(count, sum(areas))):
        for (x0, y0, x1, _), area in zip(regions, areas):
            if offset < area:
                dy, dx = divmod(offset, x1 - x0)
                x, y = x0 + dx, y0 + dy
                boxes.append([x, y, x + size, y + size])
                break
            offset -= area
    return boxes


def _hard_candidates(rng, size, native_hw, gt, count):
    """Prefer disjoint equal-area cores; use distinct overlapping ones for small frames."""
    h, w = native_hw
    ox, oy = rng.choice((0, w % size)), rng.choice((0, h % size))
    pool = []
    for y in range(oy, h - size + 1, size):
        for x in range(ox, w - size + 1, size):
            if gt is None or (x + size <= gt[0] or gt[2] <= x or y + size <= gt[1] or gt[3] <= y):
                pool.append([x, y, x + size, y + size])
    if len(pool) < count:
        extra = _negative_boxes(rng, size, native_hw, gt, count + len(pool))
        pool.extend(box for box in extra if box not in pool)
        if len(pool) < count:
            raise ValueError(f"Only {len(pool)} distinct full-size, GT-free hard candidates; need {count}")
        pool = pool[:count]
    return rng.sample(pool, count)


def build_schedule(index, seed, steps, core_size, refresh_every, candidates_per_refresh=8):
    """Return a JSON-ready, prefix-stable schedule; hard slots name a refresh and score rank."""
    if type(seed) is not int or any(type(n) is not int or n <= 0 for n in
                                    (steps, core_size, refresh_every, candidates_per_refresh)):
        raise ValueError("Expected integer seed and positive integer scheduling parameters")
    if refresh_every % len(CATEGORIES) or candidates_per_refresh < 2 or index["split"] != "train":
        raise ValueError("Require train sources and refreshes every 8*k updates with at least two candidates")
    positive, empty = defaultdict(list), defaultdict(list)
    sources = index["sources"]
    for name, record in sources.items():
        if core_size > min(record["native_hw"]):
            raise ValueError(f"{name} is smaller than the registered training core")
        for row in record["centers"]:
            bucket = empty if row["foreground_box"] is None else positive
            bucket[name].append(row)
    streams = {key: _draw_centers(empty if key == "empty" else positive, seed, key)
               for key in set(CATEGORIES)}
    updates, refreshes = [], []
    for step in range(1, steps + 1):
        kind = CATEGORIES[(step - 1) % len(CATEGORIES)]
        if kind == "hard":
            if (step - len(CATEGORIES)) % refresh_every == 0:
                source, frame = next(streams[kind])
                boxes = _hard_candidates(_rng(seed, kind, step), core_size, sources[source]["native_hw"],
                                         frame["foreground_box"], candidates_per_refresh)
                candidates = [{"source": source, "center": frame["center"], "core_xyxy": box} for box in boxes]
                refreshes.append({"before_step": step, "source": source, "center": frame["center"],
                                  "candidates": candidates})
            refresh = refreshes[-1]
            rank = ((step - refresh["before_step"]) // len(CATEGORIES)) % 2
            updates.append({"step": step, "kind": kind, "source": refresh["source"], "center": refresh["center"],
                            "refresh_step": refresh["before_step"], "hard_rank": rank})
        else:
            source, frame = next(streams[kind])
            rng = _rng(seed, kind, step, source, frame["center"])
            if kind == "positive":
                box = _positive_box(rng, core_size, sources[source]["native_hw"], frame["foreground_box"])
            else:
                boxes = _negative_boxes(rng, core_size, sources[source]["native_hw"], frame["foreground_box"])
                if not boxes:
                    raise ValueError(f"No GT-free core for {source}/{frame['center']:05d}")
                box = boxes[0]
            updates.append({"step": step, "kind": kind, "source": source, "center": frame["center"],
                            "core_xyxy": box})
    exposures = {}
    for kind in dict.fromkeys(CATEGORIES):
        rows = [r for r in updates if r["kind"] == kind]
        exposures[kind] = {"updates": len(rows), "unique_centers": len({(r["source"], r["center"]) for r in rows}),
                           "updates_per_source": dict(sorted(Counter(r["source"] for r in rows).items()))}
    return {"seed": seed, "steps": steps, "core_size": core_size, "refresh_every": refresh_every,
            "candidates_per_refresh": candidates_per_refresh, "index_sha256": _digest(index),
            "updates": updates, "refreshes": refreshes, "exposure": exposures}


def select_hard(candidates, scores, rank=0):
    """Select a current-model peak-score rank, resolving equal scores by the prescheduled order."""
    scores = [float(score) for score in scores]
    if (len(scores) != len(candidates) or type(rank) is not int or not 0 <= rank < len(scores)
            or not all(math.isfinite(score) for score in scores)):
        raise ValueError("Expected one finite raw peak score per candidate and a valid selection rank")
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    return candidates[order[rank]]
