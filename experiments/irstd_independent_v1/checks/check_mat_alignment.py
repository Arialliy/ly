"""Verify the physical frame index of each NUDT-MIRSDT PNG against the published MAT."""

import argparse
from collections import Counter, deque
import hashlib
from io import BytesIO
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.io import loadmat


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET = Path("data/public/NUDT-MIRSDT")
MANIFEST = Path("experiments/irstd_independent_v1/manifests/nudt_mirsdt_mask17.json")
INVENTORY = Path("provenance/PUBLIC_MEMBERS_NUDT-MIRSDT.json")


class MatAlignmentError(ValueError):
    """Preserve the measured result for the stage report when any comparison fails."""

    def __init__(self, result):
        self.result = result
        super().__init__(json.dumps(result, ensure_ascii=False))


def verify_mat_alignment(project_root, manifest):
    root = Path(project_root) / DATASET
    official = {
        split: (root / f"{split}.txt").read_text(encoding="utf-8").splitlines()
        for split in ("train", "test")
    }
    raw_inventory = json.loads((Path(project_root) / INVENTORY).read_text(encoding="utf-8"))
    hashes = {
        row["member"]: row["sha256"]
        for row in raw_inventory["files"] if row["member"].endswith(".mat")
    }
    seen = {"train": [], "test": []}
    by_split = {s: {"pairs_checked": 0, "matching_pairs": 0} for s in ("train", "dev", "test")}
    result = {
        "pairs_checked": 0,
        "matching_pairs": 0,
        "mismatching_pairs": 0,
        "mat_archive_hash_matches": 0,
        "history_planes_checked": 0,
        "matching_history_planes": 0,
        "by_split": by_split,
        "mismatch_examples": [],
    }

    def note(member, detail):
        if len(result["mismatch_examples"]) < 12:
            result["mismatch_examples"].append({"member": member, "detail": detail})

    for sequence in manifest["sequences"]:
        split = sequence["split"]
        source = sequence["id"]
        official_split = "test" if split == "test" else "train"
        history = deque(maxlen=4)
        previous_number = -1
        for frame in sequence["frames"]:
            number = frame["number"]
            if number <= previous_number:
                raise ValueError(f"{source}: frame order is not strictly increasing at {number}")
            previous_number = number
            member = f"{source}/Mix/{number:05d}.mat"
            member_image = DATASET / source / "images" / f"{number:05d}.png"
            if Path(frame["image"]) != member_image:
                raise ValueError(f"{member}: image path does not identify the same source/frame")
            raw = (root / member).read_bytes()
            observed = hashlib.sha256(raw).hexdigest()
            if observed == hashes.get(f"NUDT-MIRSDT/{member}"):
                result["mat_archive_hash_matches"] += 1
            else:
                note(member, "MAT SHA256 differs from archive inventory or member is absent")
            array = loadmat(BytesIO(raw), variable_names=["Mix"]).get("Mix")
            image = cv2.imread(str(Path(project_root) / member_image), cv2.IMREAD_UNCHANGED)
            if array is None or array.dtype != np.uint8 or array.ndim != 3 or array.shape[0] != 5:
                raise ValueError(f"{member}: expected native uint8 Mix[5,H,W]")
            if image is None or image.dtype != np.uint8 or image.ndim != 2:
                raise ValueError(f"{member_image}: expected native uint8 image[H,W]")
            seen[official_split].append(member)
            result["pairs_checked"] += 1
            by_split[split]["pairs_checked"] += 1
            if np.array_equal(image, array[-1]):
                result["matching_pairs"] += 1
                by_split[split]["matching_pairs"] += 1
            else:
                result["mismatching_pairs"] += 1
                note(member, "image differs from the final plane of Mix")
            for earlier, earlier_image in history:
                lag = number - earlier
                if lag <= 4:
                    result["history_planes_checked"] += 1
                    if np.array_equal(earlier_image, array[-1 - lag]):
                        result["matching_history_planes"] += 1
                    else:
                        note(member, f"Mix plane at lag {lag} differs from listed frame {earlier}")
            history.append((number, image))
    for split in official:
        if Counter(seen[split]) != Counter(official[split]):
            note(f"{split}.txt", "compared members differ from published entries")
    if result["mismatch_examples"]:
        raise MatAlignmentError(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    manifest_path = args.manifest or args.project_root / MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    try:
        result = verify_mat_alignment(args.project_root, manifest)
    except MatAlignmentError as error:
        print(json.dumps(error.result, indent=2))
        raise SystemExit(1) from None
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
