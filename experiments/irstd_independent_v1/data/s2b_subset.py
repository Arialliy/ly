"""Freeze the training-only S2b small-fit manifest and its complete update order."""
import argparse
from collections import Counter, defaultdict
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = Path("experiments/irstd_independent_v1")
FROZEN = EXPERIMENT / "manifests/s2b_train12.json"
SEED = 46
UPDATES = 400


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def hash_order(*parts):
    return digest("|".join(str(p) for p in (SEED, *parts)).encode("utf-8"))


def archive_member(root, relative, inventory):
    path = Path(relative)
    raw = (root / path).read_bytes()
    sha = digest(raw)
    name = path.relative_to("data/public").as_posix()
    if inventory.get(name) != sha:
        raise ValueError(f"Image or mask differs from authenticated release: {relative}")
    return raw, {"path": relative, "sha256": sha}


def read_pool(root, manifest, inventory):
    pools = {True: defaultdict(list), False: defaultdict(list)}
    records = {}
    for sequence in manifest["sequences"]:
        if sequence["split"] != "train":
            continue
        source = sequence["id"]
        if sequence["official_split"] != "train" or source in records:
            raise ValueError(f"Duplicate or unofficial training source: {source}")
        frames = {frame["number"]: frame for frame in sequence["frames"]}
        records[source] = frames
        height, width = sequence["native_shape_hw"]
        for center in sequence["centers"]:
            raw, info = archive_member(root, frames[center]["mask"], inventory)
            with Image.open(BytesIO(raw)) as mask:
                if mask.mode != "L" or mask.size != (width, height):
                    raise ValueError(f"Unexpected mask encoding or dimensions: {info['path']}")
                histogram = mask.histogram()
            if histogram[0] + histogram[255] != width * height:
                raise ValueError(f"Mask has non-binary values: {info['path']}")
            positive = histogram[255] > 0
            pools[positive][source].append({
                "sample_id": f"{source}/{center:05d}", "source": source, "center": center,
                "positive": positive, "foreground_pixels": histogram[255],
                "native_hw": [height, width], "center_mask": info,
            })
    return pools, records


def select_centers(pools):
    """Use scarce empty sources first; spread nonempty samples over remaining sources."""
    by_source = defaultdict(list)
    selected = []
    for polarity, required in ((False, 4), (True, 8)):
        label = "nonempty" if polarity else "empty"
        available = {src: list(rows) for src, rows in pools[polarity].items()}
        order = sorted(available, key=lambda src: (bool(by_source[src]), hash_order("source", label, src), src))
        remaining = required
        while remaining:
            previous = remaining
            for src in order:
                if not available[src]:
                    continue
                def rank(row):
                    spacing = min((abs(row["center"] - c) for c in by_source[src]), default=0)
                    return -spacing, hash_order("center", label, src, row["center"]), row["center"]
                chosen = min(available[src], key=rank)
                available[src].remove(chosen)
                selected.append(chosen)
                by_source[src].append(chosen["center"])
                remaining -= 1
                if not remaining:
                    break
            if remaining == previous:
                raise ValueError(f"Insufficient training candidates for {label}: need {required}")
    return sorted(selected, key=lambda row: (not row["positive"], row["sample_id"]))


def build_subset(project_root):
    root = Path(project_root).resolve()
    exp = root / EXPERIMENT
    manifest_raw = (exp / "manifests/nudt_mirsdt_mask17.json").read_bytes()
    manifest = json.loads(manifest_raw)
    s0_raw = (exp / "reports/S0.json").read_bytes()
    s2a_raw = (exp / "reports/S2a.json").read_bytes()
    s0, s2a = json.loads(s0_raw), json.loads(s2a_raw)
    if ((s2a["status"], s2a["next_allowed_stage"], s2a["seed"]) != ("PASS", "S2b", SEED)
            or s0["status"] != "PASS" or manifest["seed"] != SEED
            or s2a["prerequisites"]["S0_sha256"] != digest(s0_raw)
            or s2a["prerequisites"]["data_manifest_sha256"] != digest(manifest_raw)
            or s0["data_manifest_hash"] != digest(manifest_raw)):
        raise ValueError("S2b subset prerequisites do not match approved run records")
    provenance = s0["pixel_audit"]["provenance"]
    inventory_raw = (root / provenance["inventory"]).read_bytes()
    if digest(inventory_raw) != provenance["inventory_sha256"]:
        raise ValueError("Authenticated release member inventory has changed")
    inventory = {row["member"]: row["sha256"] for row in json.loads(inventory_raw)["files"]}
    pools, frames = read_pool(root, manifest, inventory)
    available = {key: sum(map(len, pools[flag].values())) for flag, key in ((True,"nonempty"), (False,"empty"))}
    counts = s0["pixel_audit"]["splits"]["train"]
    if (available["nonempty"], available["empty"], len(frames)) != (
            counts["positive_centers"], counts["empty_centers"], counts["sources"]):
        raise ValueError("Candidate counts differ from the complete frozen S0 training split")

    samples = select_centers(pools)
    for row in samples:
        row["images"] = []
        height, width = row["native_hw"]
        for number in range(row["center"] - 8, row["center"] + 9):
            raw, info = archive_member(root, frames[row["source"]][number]["image"], inventory)
            with Image.open(BytesIO(raw)) as image:
                if image.size != (width, height) or image.mode != "L":
                    raise ValueError(f"Unexpected training frame geometry: {info['path']}")
            row["images"].append({"number": number, **info})
    ids = [row["sample_id"] for row in samples]
    schedule = []
    for epoch in range((UPDATES + len(ids) - 1) // len(ids)):
        schedule.extend(sorted(ids, key=lambda frame_id: (hash_order("epoch", epoch, frame_id), frame_id)))
    schedule = schedule[:UPDATES]
    lookups = {row["sample_id"]: row for row in samples}
    return {
        "stage": "S2b", "seed": SEED, "dataset": manifest["dataset"], "split": "train",
        "dependencies": {"data_manifest_sha256": digest(manifest_raw), "s0_sha256": digest(s0_raw),
                         "s2a_sha256": digest(s2a_raw), "inventory_sha256": digest(inventory_raw),
                         "selector_sha256": digest(Path(__file__).read_bytes())},
        "selection_rule": "SHA256 over UTF-8 pipe-separated fields, starting with seed 46. Select empty then "
                          "nonempty independently by round-robin of sources, unused sources first then hash "
                          "of seed|source|class|source_id. Within each source maximize minimum distance to "
                          "already selected physical centers from that source; break ties by hash of "
                          "seed|center|class|source_id|physical_frame. Classes are named empty/nonempty.",
        "schedule_rule": "Order all 12 IDs by SHA256 of seed|epoch|zero_based_epoch|sample_id per epoch. "
                         "Concatenate epochs then truncate after 400 B=1 updates.",
        "candidate_centers": available,
        "sources_with_empty_centers": {src: len(rows) for src, rows in sorted(pools[False].items())},
        "samples": samples, "schedule": schedule,
        "exposure": {"optimizer_updates": len(schedule), "per_sample": dict(sorted(Counter(schedule).items())),
                     "per_source": dict(sorted(Counter(lookups[s]["source"] for s in schedule).items())),
                     "nonempty_updates": sum(lookups[s]["positive"] for s in schedule),
                     "empty_updates": sum(not lookups[s]["positive"] for s in schedule)},
    }


def freeze(path, raw, verify):
    if verify or path.exists():
        if path.read_bytes() != raw:
            raise ValueError(f"Existing frozen subset differs from deterministic rebuild: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as temp:
        try:
            temp.write(raw)
            temp.flush()
            os.fsync(temp.fileno())
            try:
                os.link(temp.name, path)
            except FileExistsError:
                if path.read_bytes() != raw:
                    raise ValueError(f"Different subset was frozen concurrently: {path}")
        finally:
            os.unlink(temp.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--verify", action="store_true", help="Rebuild and compare exact serialized bytes")
    args = parser.parse_args()
    subset = build_subset(args.project_root)
    raw = (json.dumps(subset, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    path = args.project_root / FROZEN
    freeze(path, raw, args.verify)
    print(json.dumps({"path": str(path), "sha256": digest(raw),
                      "samples": [r["sample_id"] for r in subset["samples"]],
                      "exposure": subset["exposure"]}, indent=2))


if __name__ == "__main__":
    main()
