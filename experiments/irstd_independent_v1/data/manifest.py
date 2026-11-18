"""Freeze native NUDT-MIRSDT mask clips from the published train/test lists."""

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import tempfile
import zlib


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET = Path("data/public/NUDT-MIRSDT")
OUTPUT_PATH = Path("experiments/irstd_independent_v1/manifests/nudt_mirsdt_mask17.json")
SOURCE = re.compile(r"Sequence([1-9][0-9]*)")
MEMBER = re.compile(r"(Sequence[1-9][0-9]*)/Mix/([0-9]{5})\.mat")
SEED = 46
RADIUS = 8
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def source_order(source):
    return int(source.removeprefix("Sequence"))


def read_official_list(root, split):
    path = root / f"{split}.txt"
    raw = path.read_bytes()
    members = defaultdict(dict)
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        match = MEMBER.fullmatch(line)
        if not match:
            raise ValueError(f"{path}:{line_number}: unexpected frame path {line!r}")
        source, stem = match.groups()
        number = int(stem)
        if number in members[source]:
            raise ValueError(f"{path}:{line_number}: duplicate member {line}")
        members[source][number] = line
    if not members:
        raise ValueError(f"{path}: empty published split")
    return members, hashlib.sha256(raw).hexdigest()


def eligible_centers(numbers):
    """Physical window is [center-8, center+8]; every frame must be listed."""
    present = set(numbers)
    return [
        center for center in sorted(present)
        if all(number in present for number in range(center - RADIUS, center + RADIUS + 1))
    ]


def checked_path(root, relative):
    path = root / relative
    if not path.resolve(strict=True).is_relative_to(root):
        raise ValueError(f"File escapes dataset root: {path}")
    if not path.is_file() or not path.stat().st_size:
        raise ValueError(f"Expected a nonempty regular file: {path}")
    return path


def read_png_header_and_hash(path):
    raw = path.read_bytes()
    if (len(raw) < 33 or raw[:8] != PNG_MAGIC or
            raw[8:16] != b"\x00\x00\x00\rIHDR" or
            struct.unpack(">I", raw[29:33])[0] != zlib.crc32(raw[12:29])):
        raise ValueError(f"Invalid PNG header: {path}")
    width, height, depth, color, compression, filtering, interlace = struct.unpack(
        ">IIBBBBB", raw[16:29]
    )
    if (width < 1 or height < 1 or
            (depth, color, compression, filtering, interlace) != (8, 0, 0, 0, 0)):
        raise ValueError(f"Expected native noninterlaced grayscale uint8 PNG: {path}")
    return (height, width), hashlib.sha256(raw).hexdigest()


def sequence_record(root, source, split, official_split, members):
    frames = []
    native_shape = None
    list_hash = hashlib.sha256()
    image_hash = hashlib.sha256()
    mask_hash = hashlib.sha256()
    for number, member in sorted(members.items()):
        checked_path(root, member)
        stem = Path(member).stem
        image = Path(source, "images", f"{stem}.png")
        mask = Path(source, "masks", f"{stem}.png")
        image_shape, image_digest = read_png_header_and_hash(checked_path(root, image))
        mask_shape, mask_digest = read_png_header_and_hash(checked_path(root, mask))
        if image_shape != mask_shape:
            raise ValueError(f"{source}/{stem}: image {image_shape} and mask {mask_shape} differ")
        if native_shape is not None and image_shape != native_shape:
            raise ValueError(f"{source}/{stem}: frame size changed from {native_shape} to {image_shape}")
        native_shape = image_shape
        list_hash.update(f"{member}\n".encode("ascii"))
        image_hash.update(f"{image_digest}\n".encode("ascii"))
        mask_hash.update(f"{mask_digest}\n".encode("ascii"))
        frames.append({
            "number": number,
            "image": (DATASET / image).as_posix(),
            "mask": (DATASET / mask).as_posix(),
        })
    return {
        "id": source,
        "split": split,
        "official_split": official_split,
        "native_shape_hw": list(native_shape),
        "official_members_sha256": list_hash.hexdigest(),
        "image_files_sha256": image_hash.hexdigest(),
        "mask_files_sha256": mask_hash.hexdigest(),
        "frames": frames,
        "centers": eligible_centers(members),
    }


def build_manifest(project_root, *, dev_count=8):
    root = (Path(project_root).resolve() / DATASET).resolve(strict=True)
    train, train_hash = read_official_list(root, "train")
    test, test_hash = read_official_list(root, "test")
    conflict = train.keys() & test.keys()
    if conflict:
        raise ValueError(f"Official train/test share sources: {sorted(conflict, key=source_order)}")
    if not 0 < dev_count < len(train):
        raise ValueError(f"Development requires 1 to {len(train) - 1} train sources; got {dev_count}")
    ranked_train = sorted(
        train, key=lambda source: (hashlib.sha256(f"{SEED}:{source}".encode("ascii")).digest(), source)
    )
    dev = set(ranked_train[:dev_count])
    official = {**train, **test}
    local_sources = {p.name for p in root.iterdir() if p.is_dir() and SOURCE.fullmatch(p.name)}
    missing = official.keys() - local_sources
    if missing:
        raise ValueError(f"Official source directories missing: {sorted(missing, key=source_order)}")

    records = []
    counts = {}
    for source in sorted(official, key=source_order):
        official_split = "train" if source in train else "test"
        split = "dev" if source in dev else official_split
        record = sequence_record(root, source, split, official_split, official[source])
        records.append(record)
        count = counts.setdefault(split, {"sources": 0, "listed_frames": 0, "eligible_centers": 0})
        count["sources"] += 1
        count["listed_frames"] += len(record["frames"])
        count["eligible_centers"] += len(record["centers"])

    return {
        "schema_version": 1,
        "dataset": "NUDT-MIRSDT",
        "dataset_root": DATASET.as_posix(),
        "clip_length": 2 * RADIUS + 1,
        "center_index": RADIUS,
        "seed": SEED,
        "official_lists": {
            "train": {"path": (DATASET / "train.txt").as_posix(), "sha256": train_hash},
            "test": {"path": (DATASET / "test.txt").as_posix(), "sha256": test_hash},
        },
        "policy": {
            "split_unit": "Published SequenceN directory. Physical scene identities are unavailable.",
            "dev_selection": "First dev_count official train IDs ranked by SHA256 of ASCII '46:SequenceN', then ID.",
            "context": "Use only physical frame numbers listed in the source's official split; require all 17 consecutive numbers.",
            "labels": "Use images/<stem>.png and the native masks/<stem>.png paired by the published Mix/<stem>.mat entry.",
            "source_hashes": "Official members hash the ASCII paths with LF in physical frame order. Image/mask hashes hash the ASCII hex SHA256 of each encoded PNG with LF in that order.",
        },
        "dev_count": dev_count,
        "dev_source_ids": sorted(dev, key=source_order),
        "unassigned_source_ids": sorted(local_sources - official.keys(), key=source_order),
        "counts": counts,
        "sequences": records,
    }


def write_manifest(manifest, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    stream = tempfile.NamedTemporaryFile(
        dir=output.parent, prefix=f".{output.name}.", delete=False,
        mode="w", encoding="utf-8", newline="\n",
    )
    tmp = Path(stream.name)
    try:
        with stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, output)
    finally:
        tmp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path, help="Default: <project-root>/" + OUTPUT_PATH.as_posix())
    args = parser.parse_args()
    output = args.output or args.project_root / OUTPUT_PATH
    manifest = build_manifest(args.project_root)
    write_manifest(manifest, output)
    print(json.dumps({"manifest": str(output), "counts": manifest["counts"],
                      "dev_source_ids": manifest["dev_source_ids"]}, indent=2))


if __name__ == "__main__":
    main()
