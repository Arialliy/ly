"""Independently decode and authenticate the S0 manifest's official frames."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def _sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def _distribution(values):
    if not values:
        return {"n": 0}
    quantiles = np.percentile(values, [0, 50, 90, 95, 100])
    summary = dict(zip(("min", "median", "p90", "p95", "max"), map(float, quantiles)))
    return {"n": len(values), **summary}


def _verified_bytes(root, release_root, digests, path):
    member = Path(release_root.name) / path.relative_to(release_root)
    raw = path.read_bytes()
    digest = _sha256(raw)
    if digests.get(member.as_posix()) != digest:
        raise ValueError(f"Release member hash mismatch or missing member: {path.relative_to(root)}")
    return raw, digest


def _decode_checked(raw, path, shape):
    frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if frame is None or frame.dtype != np.uint8 or frame.shape != shape:
        raise ValueError(f"OpenCV expected uint8 grayscale of native shape {shape} at {path}")
    return frame


def audit_pixels(project_root: Path, manifest: dict) -> dict:
    root = Path(project_root).resolve()
    release_root = root / "data/public/NUDT-MIRSDT"
    receipt_path = root / "provenance/PUBLIC_ARCHIVE_NUDT-MIRSDT.json"
    receipt = json.loads(receipt_path.read_bytes())
    if (receipt["dataset"] != manifest["dataset"] or receipt["status"] != "COMPLETE" or
            receipt["full_zip_crc"] != "PASS" or root / receipt["extracted_directory"] != release_root):
        raise ValueError("Inconsistent NUDT-MIRSDT archive receipt")
    inventory_path = root / receipt["member_hash_manifest"]
    inventory_raw = inventory_path.read_bytes()
    inventory_digest = _sha256(inventory_raw)
    if inventory_digest != receipt["member_hash_manifest_sha256"]:
        raise ValueError("Member inventory differs from its pinned archive receipt")
    inventory = json.loads(inventory_raw)
    digests = {row["member"]: row["sha256"] for row in inventory["files"]}
    if (inventory["dataset"] != manifest["dataset"] or len(digests) != len(inventory["files"]) or
            len(digests) != receipt["files"]):
        raise ValueError("Incomplete, duplicate, or inconsistent member inventory")

    official = {"train": set(), "test": set()}
    for split, entries in official.items():
        path = release_root / f"{split}.txt"
        raw, digest = _verified_bytes(root, release_root, digests, path)
        if digest != manifest["official_lists"][split]["sha256"]:
            raise ValueError(f"Official split digest differs from manifest: {split}")
        for line in raw.decode("utf-8").splitlines():
            source, role, filename = line.split("/")
            frame_id = (source, int(Path(filename).stem))
            if role != "Mix" or frame_id in entries:
                raise ValueError(f"Duplicate or invalid official frame: {line}")
            entries.add(frame_id)

    per_split = {split: dict(sources=0, frames=0, eligible_centers=0, positive_centers=0,
                             empty_centers=0, components_at_centers=0) for split in ("train", "dev", "test")}
    shapes = {split: set() for split in per_split}
    seen_frames = set()
    seen_images = {}
    widths, heights, areas = [], [], []
    distances, overlap_distances = [], []
    consecutive_pairs = 0
    examples = dict(positive=None, negative=None, different_size=None)

    for sequence in manifest["sequences"]:
        source, split = sequence["id"], sequence["split"]
        counts = per_split[split]
        counts["sources"] += 1
        shape = tuple(sequence["native_shape_hw"])
        shapes[split].add(shape)
        centers = set(sequence["centers"])
        official_split = "train" if split in ("train", "dev") else "test"
        previous_number, previous_single, previous_mask = None, None, None

        for frame in sorted(sequence["frames"], key=lambda record: record["number"]):
            number = frame["number"]
            frame_id = (source, number)
            if frame_id in seen_frames or frame_id not in official[official_split]:
                raise ValueError(f"Repeated or misplaced frame {frame_id} in {split}")
            seen_frames.add(frame_id)
            counts["frames"] += 1
            image_path = release_root / source / "images" / f"{number:05d}.png"
            mask_path = release_root / source / "masks" / f"{number:05d}.png"
            if root / frame["image"] != image_path or root / frame["mask"] != mask_path:
                raise ValueError(f"Wrong release image or mask for {frame_id}")
            image_raw, image_digest = _verified_bytes(root, release_root, digests, image_path)
            mask_raw, _ = _verified_bytes(root, release_root, digests, mask_path)
            prior_split = seen_images.setdefault(image_digest, split)
            if prior_split != split:
                raise ValueError(f"Identical image PNG across {prior_split} and {split}: {image_path}")
            _decode_checked(image_raw, image_path, shape)
            mask = _decode_checked(mask_raw, mask_path, shape)
            if np.any((mask != 0) & (mask != 255)):
                raise ValueError(f"Invalid binary mask values at {mask_path}")
            if number not in centers:
                continue

            labels, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            components = labels - 1
            counts["eligible_centers"] += 1
            counts["components_at_centers"] += components
            counts["positive_centers" if components else "empty_centers"] += 1
            if split != "train":
                continue
            widths.extend(stats[1:, cv2.CC_STAT_WIDTH].tolist())
            heights.extend(stats[1:, cv2.CC_STAT_HEIGHT].tolist())
            areas.extend(stats[1:, cv2.CC_STAT_AREA].tolist())
            sample = dict(source_id=source, number=number, height=shape[0], width=shape[1], components=components)
            polarity = "positive" if components else "negative"
            if examples[polarity] is None:
                examples[polarity] = sample
            first = examples["positive"]
            if first and components and examples["different_size"] is None and (first["height"], first["width"]) != shape:
                examples["different_size"] = sample

            single = centroids[1] if components == 1 else None
            if previous_number == number - 1:
                consecutive_pairs += 1
                if single is not None and previous_single is not None:
                    displacement = float(np.linalg.norm(single - previous_single))
                    distances.append(displacement)
                    if np.any(mask & previous_mask):
                        overlap_distances.append(displacement)
            previous_number, previous_single, previous_mask = number, single, mask

    if seen_frames != official["train"] | official["test"] or official["train"] & official["test"]:
        raise ValueError("Manifest does not exhaust disjoint official train and test frames")
    for split, counts in per_split.items():
        counts["distinct_native_shapes_hw"] = len(shapes[split])
    return {
        "provenance": {
            "receipt": str(receipt_path.relative_to(root)),
            "inventory": str(inventory_path.relative_to(root)), "inventory_sha256": inventory_digest,
            "archive_sha256_recorded": receipt["archive_sha256"], "frames_checked": len(seen_frames),
            "images_and_masks_checked": 2 * len(seen_frames), "unique_image_png_hashes": len(seen_images),
            "decoder": f"OpenCV {cv2.__version__} IMREAD_UNCHANGED",
        },
        "splits": per_split,
        "train_statistics": {
            "population": "Count one object proxy per 8-connected component in each eligible native training center mask.",
            "component_width_px": _distribution(widths), "component_height_px": _distribution(heights),
            "component_area_px": _distribution(areas), "adjacent_center_pairs": consecutive_pairs,
            "single_component_pair_centroid_shift_px_per_frame": _distribution(distances),
            "overlapping_single_component_pair_centroid_shift_px_per_frame": _distribution(overlap_distances),
            "motion_interpretation": (
                "Measure image-plane centroid displacement between consecutive eligible centers with one "
                "component in each. The overlap subset also shares mask pixels and favors smaller shifts. "
                "Object identity, camera compensation and speed ground truth are unavailable. A disappearing "
                "object could be replaced by a different visible object. The range describes the sampled pairs."
            ),
        },
        "train_samples": examples,
    }
