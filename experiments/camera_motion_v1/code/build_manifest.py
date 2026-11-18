#!/usr/bin/env python3
"""Re-use frozen input provenance; enumerate only authorized motion inputs.

This writes the requested per-frame MOTION_MANIFEST.csv. It never imports the
detector, reads predictions, or opens held-out confirmation images/annotations.
The upstream VT paper describes cross-modal alignment, not temporal
stabilization. Therefore local temporal registration is false while upstream
temporal registration remains unknown. Official per-sequence CM is unknown.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
PROJECT = BASE.parents[1]
Q = PROJECT / "archive/infrared_problem_screen_20261009/temporal_diagnosis_20261010"
STAGE = Q / "stage1_goal"
PLANS = {"O": [0, 36, 72, 108, 120], "S": [0, 24, 60, 96, 120]}
VT_PAPER = "https://arxiv.org/html/2412.10861v1"
VT_REPO = "https://github.com/xuqingyu26/HGTMT"
RGBT_REPO = "https://github.com/XinyiYing/RGBT-Tiny#mot-label"
NUDT_REPO = "https://github.com/TinaLRJ/Multi-frame-infrared-small-target-detection-DTUM#datasets"
FIELDS = [
    "dataset", "source_family", "source_nature", "sequence", "source_video_id",
    "local_clip", "split", "historical_split_name", "official_source_split",
    "purpose", "modality", "channel", "frame", "source_frame",
    "source_frame_id", "source_mot_frame_id", "source_image_id",
    "input_sampling_interval_frames", "published_dataset_fps", "height", "width",
    "raw_dtype", "raw_path", "raw_index", "source_image_path", "source_image_sha256",
    "source_array_sha256", "source_manifest", "source_manifest_sha256",
    "temporal_registration_added_locally", "upstream_temporal_registration",
    "cross_modal_alignment_description", "official_cm_attribute", "cm_status_reason",
    "actual_input_preprocessing", "O_covering_starts", "S_covering_starts",
    "window_length", "anchor_slot_zero_based", "covering_anchor_local_frames",
    "covering_anchor_source_frames", "detector_inference_scope",
    "source_paper_url", "source_repository_url", "dataset_relationship_url",
]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def relative(path: Path) -> str:
    return str(path.relative_to(PROJECT))


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def encode(values) -> str:
    return ";".join(str(x) for x in values)


def vt_rows():
    selection_path = STAGE / "DATA_SELECTION.json"
    selection = read_json(selection_path)
    integrity = read_json(STAGE / "data/DATA_INTEGRITY.json")
    assert integrity["status"] == "PASS"
    assert digest(selection_path) == integrity["selection_sha256"]
    frozen_rows = {row["sequence"]: row for row in integrity["rows"]}
    train = selection["splits"]["train"]
    dev = selection["splits"]["calibration"]
    heldout = set(selection["splits"]["confirmation"])
    assert len(train) == 8 and dev == ["DJI_0026_2", "DJI_0067_2"]
    assert not set(train + dev) & heldout
    rows = []
    for seq in train + dev:
        path = STAGE / "data" / seq / "FRAME_MANIFEST.json"
        manifest = read_json(path)
        manifest_hash = digest(path)
        assert manifest_hash == frozen_rows[seq]["frame_manifest_sha256"]
        assert manifest["frame_count"] == 160 and not manifest["raw_inputs_resized"]
        assert manifest["shape"] == [160, 512, 640] and manifest["dtype"] == "uint8"
        assert [r["index"] for r in manifest["frames"]] == list(range(160))
        source_frames = [r["raw_filename_number"] for r in manifest["frames"]]
        assert source_frames == list(range(source_frames[0], source_frames[0] + 160))
        # Prior full pixel/CRC/annotation audit is reused, not rerun. Hashing the
        # actual array binds this run's input to that checked materialization.
        raw_path = path.parent / manifest["raw_frames_array"]
        assert digest(raw_path) == manifest["array_sha256"]
        is_dev = seq in dev
        for frame in manifest["frames"]:
            index = frame["index"]
            cover = {p: [s for s in starts if s <= index < s + 40] if is_dev else []
                     for p, starts in PLANS.items()}
            anchors = sorted({s + 19 for starts in cover.values() for s in starts})
            source_image = path.parent / frame["relative_path"]
            assert source_image.is_file()
            assert frame["source_file_name"].split("/")[1] == "01"
            rows.append(dict(
                dataset="VT-Tiny-MOT", source_family="RGBT-Tiny/VT-Tiny-MOT",
                source_nature="real_thermal_video", sequence=seq,
                source_video_id=frame["source_video_id"], local_clip=f"{seq}:{source_frames[0]}-{source_frames[-1]}",
                split="development" if is_dev else "train",
                historical_split_name=manifest["split"], official_source_split="train2017",
                purpose="paired_detection_and_motion_diagnosis" if is_dev else "image_only_motion_measurement",
                modality="thermal", channel="01", frame=index,
                source_frame=frame["raw_filename_number"], source_frame_id=frame["source_frame_id"],
                source_mot_frame_id=frame["source_mot_frame_id"], source_image_id=frame["source_image_id"],
                input_sampling_interval_frames=1, published_dataset_fps=15,
                height=512, width=640, raw_dtype="uint8", raw_path=relative(raw_path), raw_index=index,
                source_image_path=relative(source_image), source_image_sha256=frame["sha256"],
                source_array_sha256=manifest["array_sha256"], source_manifest=relative(path),
                source_manifest_sha256=manifest_hash, temporal_registration_added_locally="false",
                upstream_temporal_registration="unknown_not_documented_as_temporally_stabilized",
                cross_modal_alignment_description="paper_describes_visible_thermal_homography_not_temporal_stabilization",
                official_cm_attribute="unknown", cm_status_reason="no_per_sequence_CM_in_reused_official_manifests",
                actual_input_preprocessing="native_L_JPEG_decode_to_uint8_npy; detector_FP32_(x-85.0799)/47.4845; no_resize_no_temporal_warp",
                O_covering_starts=encode(cover["O"]), S_covering_starts=encode(cover["S"]),
                window_length=40 if is_dev else "", anchor_slot_zero_based=19 if is_dev else "",
                covering_anchor_local_frames=encode(anchors),
                covering_anchor_source_frames=encode(source_frames[a] for a in anchors),
                detector_inference_scope="C1_C2_fixed_O_S" if is_dev else "none",
                source_paper_url=VT_PAPER, source_repository_url=VT_REPO, dataset_relationship_url=RGBT_REPO,
            ))
    assert len(rows) == 1600
    return rows


def nudt_rows():
    # Fixed first sequence / first 40 frames of the official training manifest.
    # This is an implementation check source, not an additional benchmark run.
    from PIL import Image
    root = PROJECT / "data/public/NUDT-MIRSDT"
    split_path = root / "train.txt"
    train = split_path.read_text(encoding="utf-8").splitlines()
    expected = [f"Sequence1/Mix/{i:05d}.mat" for i in range(1, 41)]
    assert train[:40] == expected
    assert all(not name.startswith("Sequence1/") for name in (root / "test.txt").read_text(encoding="utf-8").splitlines())
    split_hash = digest(split_path)
    rows = []
    for index, member in enumerate(expected):
        source_frame = int(Path(member).stem)
        path = root / "Sequence1/images" / f"{source_frame:05d}.png"
        with Image.open(path) as im:
            assert im.mode == "L" and im.format == "PNG"
            width, height = im.size
        rows.append(dict(
            dataset="NUDT-MIRSDT", source_family="NUDT-MIRSDT", source_nature="synthetic_sequence",
            sequence="Sequence1", source_video_id="", local_clip="Sequence1:1-40",
            split="train", historical_split_name="official_train.txt", official_source_split="train",
            purpose="coordinate_implementation_check_only", modality="synthetic_infrared", channel="L",
            frame=index, source_frame=source_frame, source_frame_id="", source_mot_frame_id="", source_image_id="",
            input_sampling_interval_frames=1, published_dataset_fps="unknown", height=height, width=width,
            raw_dtype="uint8", raw_path=relative(path), raw_index="", source_image_path=relative(path),
            source_image_sha256=digest(path), source_array_sha256="", source_manifest=relative(split_path),
            source_manifest_sha256=split_hash, temporal_registration_added_locally="false",
            upstream_temporal_registration="not_applicable_as_real_camera_motion_evidence",
            cross_modal_alignment_description="not_applicable_single_modality",
            official_cm_attribute="not_applicable_synthetic", cm_status_reason="not_real_camera_motion_evidence",
            actual_input_preprocessing="native_L_PNG_decode; coordinate_checks_only; no_resize_no_detector",
            O_covering_starts="", S_covering_starts="", window_length=40, anchor_slot_zero_based=19,
            covering_anchor_local_frames="19", covering_anchor_source_frames="20", detector_inference_scope="none",
            source_paper_url="",
            source_repository_url=NUDT_REPO, dataset_relationship_url="",
        ))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE / "MOTION_MANIFEST.csv")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(BASE):
        raise ValueError("New artifacts must stay under camera_motion_v1")
    rows = vt_rows() + nudt_rows()
    assert len(rows) == 1640
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(".csv.tmp")
    try:
        with temp.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        temp.replace(output)
    finally:
        temp.unlink(missing_ok=True)
    print(json.dumps({"status": "complete", "path": str(output), "rows": len(rows),
                      "VT_train_frames": 1280, "VT_development_frames": 320,
                      "NUDT_train_coordinate_check_frames": 40,
                      "confirmation_frames_accessed": 0, "sha256": digest(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
