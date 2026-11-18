"""Audit frame pairs and stage pedestrian labels in the five specified source clips."""

from collections import Counter
import json
from pathlib import Path

from PIL import Image


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PUBLIC = ROOT / "data/public/VT-Tiny-MOT"
SOURCES = {
    "DJI_0043_2": "train",
    "DJI_0057_1": "train",
    "DJI_0115_2": "train",
    "DJI_0217_2": "train",
    "DJI_0067_2": "development",
}
FRAME_COUNT = 160


def inspect_source(source, role):
    annotation_file = ROOT / "data/stage1" / source / "annotations.json"
    labels = json.loads(annotation_file.read_text())
    categories = {row["id"]: row["name"] for row in labels["categories"]}
    assert categories[3] == "pedestrian"
    frames = sorted(labels["images"], key=lambda row: row["frame_id"])
    assert len(frames) == FRAME_COUNT
    frame_ids = {row["id"] for row in frames}
    assert len(frame_ids) == FRAME_COUNT

    image_specs = {"00": Counter(), "01": Counter()}
    for index, frame in enumerate(frames):
        filename = f"{index:05d}.jpg"
        assert frame["file_name"] == f"{source}/01/{filename}"
        assert frame["frame_id"] == frame["mot_frame_id"] == index
        for modality, specs in image_specs.items():
            path = PUBLIC / "train2017" / source / modality / filename
            with Image.open(path) as image:
                image.load()
                specs[f"{image.width}x{image.height} {image.mode}"] += 1
                if modality == "01":
                    assert image.size == (frame["width"], frame["height"])

    people = [row for row in labels["annotations"] if row["category_id"] == 3]
    assert all(row["image_id"] in frame_ids for row in people)
    usable = [row for row in people if not (row["iscrowd"] or row["ignore"])]
    track_frames = {(row["track_id"], row["image_id"]) for row in usable}
    assert len(track_frames) == len(usable)
    return {
        "source": source,
        "role": role,
        "stage_split": labels["subset_provenance"]["split"],
        "paired_numeric_indices": len(frames),
        "decoded_specs": image_specs,
        "ir_pedestrian_boxes_excluding_ignore_crowd": len(usable),
        "ir_pedestrian_ignored_or_crowd": len(people) - len(usable),
        "ir_pedestrian_tracks": len({row["track_id"] for row in usable}),
        "ir_frames_with_pedestrian": len({row["image_id"] for row in usable}),
        "ir_pedestrian_boxes_area_lt_256_px2": sum(
            row["bbox"][2] * row["bbox"][3] < 256 for row in usable
        ),
    }


def main():
    sources = [inspect_source(source, role) for source, role in SOURCES.items()]
    sums = {}
    for role in ("train", "development"):
        subset = [row for row in sources if row["role"] == role]
        sums[role] = {
            field: sum(row[field] for row in subset)
            for field in (
                "paired_numeric_indices", "ir_pedestrian_boxes_excluding_ignore_crowd",
                "ir_pedestrian_tracks", "ir_pedestrian_boxes_area_lt_256_px2",
            )
        }
    result = {
        "frames": "0 through 159, matched by numeric filename and thermal annotation frame_id/mot_frame_id",
        "time_evidence": "Index pairing only. Camera timestamps and synchronization error were not measured.",
        "track_unit": "Distinct thermal track_id within source. Each usable row has a distinct (track_id, image_id).",
        "sources": sources,
        "role_totals": sums,
        "split_wide_train_label_files_listed_only": sorted(
            p.name for p in (PUBLIC / "annotations").glob("*train2017.json")
        ),
        "asset_inventory_observation": (
            "The source-local stage JSONs contain only modality 01. A filename search of these "
            "source IDs in the project found no separately identifiable RGB pedestrian predictions "
            "or modality-00 labels by clip. Split-wide annotation contents were not inspected."
        ),
        "rgb_box_counts_and_teacher_coverage": None,
        "registration": (
            "Ying et al. describe homography alignment and cropping, residual disparity across "
            "depth, and unequal label coverage under poor visibility. No local per-person spatial "
            "agreement or RGB omission rate was measured."
        ),
        "primary_source": "https://arxiv.org/html/2406.14482v2#S3.SS1",
    }
    (HERE / "summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
