"""Freeze label-derived S4 native cores and role-specific without-replacement traversals."""
from collections import Counter
import hashlib
from io import BytesIO
import json
from pathlib import Path
import random

from PIL import Image


ROLE_CYCLE = ("positive", "random_background", "positive", "random_background",
              "positive", "empty_frame", "positive", "random_background")


def _rng(*key):
    value = json.dumps(key, ensure_ascii=True, separators=(",", ":")).encode()
    return random.Random(int.from_bytes(hashlib.sha256(value).digest(), "big"))


def training_index(project, manifest, s0):
    """Verify all training/dev files against S0, decoding only the training-center masks."""
    records = {}
    sources = {}
    verified = {"train": [], "dev": []}
    for sequence in manifest["sequences"]:
        split = sequence["split"]
        if split not in verified:
            continue
        source = sequence["id"]
        height, width = sequence["native_shape_hw"]
        centers = set(sequence["centers"])
        if sequence["official_split"] != "train" or len(centers) != 84:
            raise ValueError(f"Unexpected S0 sequence provenance: {source}")
        sums = {"image": hashlib.sha256(), "mask": hashlib.sha256()}
        for item in sequence["frames"]:
            n = item["number"]
            for role in ("image", "mask"):
                raw = (Path(project) / item[role]).read_bytes()
                sums[role].update((hashlib.sha256(raw).hexdigest() + "\n").encode("ascii"))
                if split != "train" or role != "mask" or n not in centers:
                    continue
                with Image.open(BytesIO(raw)) as im:
                    if im.mode != "L" or im.size != (width, height):
                        raise ValueError(f"Training mask format differs: {source}/{n:05d}")
                    histogram = im.histogram()
                    if histogram[0] + histogram[255] != width * height:
                        raise ValueError(f"Non-binary native mask: {source}/{n:05d}")
                    box = im.getbbox()
                records[f"{source}/{n:05d}"] = (width, height, box)
        for role, h in sums.items():
            if h.hexdigest() != sequence[role + "_files_sha256"]:
                raise ValueError(f"Source {source} {role} files no longer match S0")
        verified[split].append({"source": source, "image_files": len(sequence["frames"]),
                                "mask_files": len(sequence["frames"]), "eligible_centers": len(centers),
                                "image_files_sha256": sums["image"].hexdigest(),
                                "mask_files_sha256": sums["mask"].hexdigest()})
        if split == "train":
            sources[source] = [f"{source}/{n:05d}" for n in sequence["centers"]]
    expected = s0["pixel_audit"]["splits"]["train"]
    if (len(records), len(sources), sum(r[2] is None for r in records.values())) != (
            expected["eligible_centers"], expected["sources"], expected["empty_centers"]):
        raise ValueError("S4 training pools differ from S0")
    return records, sources, verified


def _placements(width, height, box, size, edge_only):
    """Partition legal native top-left positions into disjoint half-open rectangles."""
    nx, ny = width - size + 1, height - size + 1
    if nx <= 0 or ny <= 0:
        raise ValueError(f"Full {size}-pixel core does not fit image {(width, height)}")
    if box is None:
        regions = [(0, 0, nx, ny)]
    else:
        x0, y0, x1, y1 = box
        lo_x, hi_x = max(0, x0-size+1), min(nx, x1)
        lo_y, hi_y = max(0, y0-size+1), min(ny, y1)
        regions = [(0, 0, nx, lo_y), (0, hi_y, nx, ny),
                   (0, lo_y, lo_x, hi_y), (hi_x, lo_y, nx, hi_y)]
    if edge_only:
        # Top & bottom rows include the corners; the left/right edges exclude them.
        edges = [(0, 0, nx, 1)]
        if ny > 1:
            edges.append((0, ny-1, nx, ny))
        if ny > 2:
            edges.append((0, 1, 1, ny-1))
            if nx > 1:
                edges.append((nx-1, 1, nx, ny-1))
        regions = [(max(a, i), max(b, j), min(c, k), min(d, l))
                   for a,b,c,d in regions for i,j,k,l in edges]
    return [(a,b,c,d) for a,b,c,d in regions if a < c and b < d]


def _core(record, role, rng, edge, size=64):
    width, height, box = record
    if role == "positive":
        if box is None:
            raise ValueError("Positive pool contained an empty center")
        x0, y0, x1, y1 = box
        x = rng.randint(max(0, x1-size), min(x0, width-size))
        y = rng.randint(max(0, y1-size), min(y0, height-size))
    else:
        if role == "empty_frame" and box is not None:
            raise ValueError("Empty-frame pool contained a positive center")
        legal = _placements(width, height, box, size, edge)
        area = sum((c-a)*(d-b) for a,b,c,d in legal)
        if area == 0:
            raise ValueError(f"No legal background core: image={(width,height)}, GT={box}, edge={edge}")
        rank = rng.randrange(area)
        for a,b,c,d in legal:
            local = (c-a)*(d-b)
            if rank < local:
                dy, dx = divmod(rank, c-a)
                x, y = a + dx, b + dy
                break
            rank -= local
    return [x, y, x+size, y+size]


def _row(name, role, core):
    source, number = name.split("/")
    return {"source": source, "center": int(number), "kind": role, "sample_role": role, "core_xyxy": core}


def diagnostic_cores(records, sources):
    """Use two deterministic training cores per released source, independent of all job seeds."""
    rows = []
    for source, centers in sources.items():
        rng = _rng("s4-diagnostics-v1", 46, source)
        positive = [n for n in centers if records[n][2] is not None]
        empty = [n for n in centers if records[n][2] is None]
        first = rng.choice(positive or empty)
        background_pool = [n for n in (empty or centers) if n != first]
        second = rng.choice(background_pool)
        for name, role in ((first, "positive" if positive else "empty_frame"),
                           (second, "empty_frame" if empty else "random_background")):
            row = _row(name, role, _core(records[name], role, rng, False))
            rows.append(dict(row, positive=role == "positive"))
    return rows


def make_schedule(records, seed, steps=36000, size=64):
    names = sorted(records)
    pools = {"positive": [n for n in names if records[n][2] is not None],
             "random_background": names, "empty_frame": [n for n in names if records[n][2] is None]}
    epochs = {role: [] for role in pools}
    visits = Counter()
    exposures = {role: Counter() for role in pools}
    snapshots = {}
    updates = []
    background_slot = 0
    for step in range(1, steps+1):
        role = ROLE_CYCLE[(step-1) % len(ROLE_CYCLE)]
        epoch, offset = divmod(visits[role], len(pools[role]))
        if offset == 0:
            order = pools[role].copy()
            _rng("s4-pool-v1", seed, role, epoch).shuffle(order)
            epochs[role].append(order)
        name = epochs[role][epoch][offset]
        visits[role] += 1
        exposures[role][name] += 1
        if role != "positive":
            background_slot += 1
        edge = role != "positive" and background_slot % 5 == 0
        core = _core(records[name], role, _rng("s4-core-v1", seed, step, name, role), edge, size)
        updates.append(dict(_row(name, role, core), step=step, edge_priority=edge))
        if step in (24000, steps):
            snapshots[str(step)] = {"role_counts": dict(visits),
                                    "center_exposures": {r: dict(c) for r,c in exposures.items()},
                                    "edge_priority_slots": background_slot // 5}
    return {"seed": seed, "steps": steps, "core_size": size, "role_cycle": list(ROLE_CYCLE),
            "pool_sizes": {k: len(v) for k,v in pools.items()}, "pool_epoch_orders": epochs,
            "prefix_exposure": snapshots, "updates": updates}
