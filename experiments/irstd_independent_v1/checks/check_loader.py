"""Exercise the S0 loader in a real mixed-size batch and on deliberately broken inputs."""
from copy import deepcopy
from dataclasses import fields
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.utils.data import DataLoader, Subset

from ..data.dataset import MaskClipDataset, collate_clips


def _check(condition, message):
    if not condition:
        raise AssertionError(message)


def _read_native(path):
    pixels = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    _check(pixels is not None and pixels.ndim == 2 and pixels.dtype == np.uint8,
           f"OpenCV failed to read a native grayscale frame at {path}")
    return pixels


def _fixture(root):
    frames = []
    for number in range(1, 18):
        record = {"number": number}
        for key, folder in (("image", "images"), ("mask", "masks")):
            path = Path("SequenceFixture", folder, f"{number:05d}.png")
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            pixels = np.full((5, 9), number, np.uint8) if key == "image" else np.zeros((5, 9), np.uint8)
            if key == "mask" and number == 9:
                # One diagonal 8-connected pair and a separated singleton.
                pixels[[1, 2, 4], [3, 4, 8]] = 255
            Image.fromarray(pixels).save(root / path)
            record[key] = str(path)
        frames.append(record)
    return {"sequences": [{"id": "SequenceFixture", "split": "train", "frames": frames, "centers": [9]}]}


def _verify_rejections():
    raises = TestCase().assertRaisesRegex
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        good = _fixture(root)

        def dataset(manifest=good):
            return MaskClipDataset(manifest, root, "train")

        _, labels = dataset()[0]
        got = {(o.area, o.bbox_xyxy, o.centroid_xy) for o in labels.objects[0]}
        _check(got == {(2, (3, 1, 5, 3), (3.5, 1.5)), (1, (8, 4, 9, 5), (8.0, 4.0))},
               "8-connected area, native pixel centers or exclusive bounding boxes changed")

        missing = deepcopy(good)
        del missing["sequences"][0]["frames"][6]
        with raises(ValueError, "consecutive"):
            dataset(missing)
        for edge in (1, 17):
            boundary = deepcopy(good)
            boundary["sequences"][0]["centers"] = [edge]
            with raises(ValueError, "consecutive"):
                dataset(boundary)

        duplicate = deepcopy(good)
        duplicate["sequences"].append(dict(good["sequences"][0], split="test"))
        with raises(ValueError, "Repeated sequence"):
            dataset(duplicate)
        mismatched = deepcopy(good)
        center = mismatched["sequences"][0]["frames"][8]
        center["mask"] = mismatched["sequences"][0]["frames"][9]["mask"]
        with raises(ValueError, "Invalid masks path"):
            dataset(mismatched)[0]
        center["mask"] = center["image"]
        with raises(ValueError, "Invalid masks path"):
            dataset(mismatched)[0]
        center["image"] = good["sequences"][0]["frames"][8]["mask"]
        with raises(ValueError, "Invalid images path"):
            dataset(mismatched).inference_item(0)
        center["image"] = "OtherSource/images/00009.png"
        with raises(ValueError, "Invalid images path"):
            dataset(mismatched).inference_item(0)

        paths = good["sequences"][0]["frames"]
        mask_path = root / paths[8]["mask"]
        Image.fromarray(np.full((5, 9), 127, np.uint8)).save(mask_path)
        with raises(ValueError, "0/255"):
            dataset()[0]
        Image.fromarray(np.zeros((9, 5), np.uint8)).save(mask_path)
        with raises(ValueError, "Center mask geometry"):
            dataset()[0]
        Image.fromarray(np.zeros((9, 5), np.uint8)).save(root / paths[0]["image"])
        with raises(ValueError, "Frame geometry"):
            dataset().inference_item(0)
    return ["fixture_8_connected_objects_xyxy_centroids", "reject_missing_context_frame",
            "reject_temporal_wrap_at_both_ends", "reject_repeated_source_across_splits",
            "reject_shifted_center_mask", "reject_image_mask_role_swaps",
            "reject_cross_source_context", "reject_nonbinary_mask",
            "reject_center_mask_shape_change", "reject_context_frame_shape_change"]


def _font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _save_figure(rows, figure_path):
    """Keep all three scene images native; only the explicitly marked crop uses nearest zoom."""
    margin, gap, zoom_side = 16, 14, 112
    widest = max(row[1].shape[1] for row in rows)
    column_x = [margin + i * (widest + gap) for i in range(4)]
    height = 82 + sum(max(gray.shape[0], zoom_side) + 44 for _, gray, _ in rows)
    canvas = Image.new("RGB", (column_x[-1] + zoom_side + margin, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 10), "Training center frames and released pixel masks", fill="black", font=_font(20))
    for x, label in zip(column_x, ("Native image", "Binary mask", "Mask overlay", "4x crop")):
        draw.text((x, 48), label, fill="black", font=_font(14))
    y = 76
    for info, gray, binary in rows:
        h, w = gray.shape
        subtitle = (f"{info['frame_id']}   {w} x {h} px   |   "
                    f"{info['components']} components, {info['foreground_pixels']} foreground px")
        draw.text((margin, y), subtitle, fill="black", font=_font(14))
        y += 22
        image = Image.fromarray(gray).convert("RGB")
        mask = Image.fromarray(binary.astype(np.uint8) * 255)
        overlay = np.array(image)
        if binary.any():
            yy, xx = np.nonzero(binary)
            cy, cx = int(yy[0]), int(xx[0])
            overlay[binary] = ((overlay[binary].astype(float) + [255, 92, 22]) / 2).astype(np.uint8)
        else:
            cy, cx = h // 2, w // 2
        marked = Image.fromarray(overlay)
        side = min(zoom_side // 4, w, h)
        x0, y0 = min(max(0, cx-side//2), w-side), min(max(0, cy-side//2), h-side)
        crop = marked.crop((x0, y0, x0+side, y0+side)).resize((4*side, 4*side), Image.NEAREST)
        ImageDraw.Draw(marked).rectangle((x0, y0, x0+side-1, y0+side-1), outline=(44, 198, 234))
        for x, tile in zip(column_x, (image, mask, marked, crop)):
            canvas.paste(tile, (x, y))
        y += max(h, zoom_side) + 22
    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(figure_path)


def verify_loader(project_root, manifest, train_samples, figure_path):
    root = Path(project_root)
    dataset = MaskClipDataset(manifest, root, "train")
    sample_keys = {key: i for i, key in enumerate(dataset.centers)}
    selected = [(role, row) for role, row in train_samples.items() if row is not None]
    _check({role for role, _ in selected} >= {"positive", "different_size"},
           "Need two real training samples with different native sizes")
    indices = [sample_keys[row["source_id"], row["number"]] for _, row in selected]
    # A stock DataLoader must work with the real Dataset and its documented collator.
    batch, labels = next(iter(DataLoader(Subset(dataset, indices), batch_size=len(indices),
                                         collate_fn=collate_clips, shuffle=False)))
    n = len(indices)
    shapes = {(row["height"], row["width"]) for _, row in selected}
    _check(len(shapes) >= 2, "Batch did not exercise mixed native sizes")
    hmax, wmax = max(h for h, _ in shapes), max(w for _, w in shapes)
    _check(batch.frames.shape == batch.pixel_valid.shape == (n, 17, 1, hmax, wmax) and
           labels.target_map.shape == labels.ignore_map.shape == (n, 1, hmax, wmax) and
           batch.frames.dtype == labels.target_map.dtype == batch.time_offsets.dtype == torch.float32 and
           batch.pixel_valid.dtype == labels.ignore_map.dtype == torch.bool and
           labels.label_type == "mask" and batch.center_index == 8, "Tensor shape/dtype contract changed")
    torch.testing.assert_close(batch.time_offsets, torch.arange(-8, 9, dtype=torch.float32).repeat(n, 1))
    _check({f.name for f in fields(batch)} ==
           {"frames", "time_offsets", "pixel_valid", "center_index", "output_frame_id", "scene_meta"},
           "Unexpected inference input field")

    rows = []
    for i, (role, row) in enumerate(selected):
        name, center = row["source_id"], row["number"]
        h, w = row["height"], row["width"]
        meta = batch.scene_meta[i]
        numbers = tuple(range(center-8, center+9))
        source = root / manifest["dataset_root"] / name
        paths = [source / "images" / f"{number:05d}.png" for number in numbers]
        source_frames = np.stack([_read_native(path) for path in paths])
        expected = torch.from_numpy(source_frames).to(torch.float32) / 255.0
        torch.testing.assert_close(batch.frames[i, :, 0, :h, :w], expected, rtol=0, atol=0)
        _check((meta.source_id, meta.frame_numbers, meta.original_hw, meta.tile_origin_xy, meta.core_xyxy) ==
               (name, numbers, (h, w), (0, 0), (0, 0, w, h)), f"Scene metadata mismatch for {name}/{center}")
        frame_id = f"{name}/{center:05d}"
        _check(batch.output_frame_id[i] == frame_id, "Wrong output frame ID")
        usable = torch.zeros((hmax, wmax), dtype=torch.bool)
        usable[:h, :w] = True
        torch.testing.assert_close(batch.pixel_valid[i, :, 0], usable.expand(17, -1, -1))
        torch.testing.assert_close(labels.ignore_map[i, 0], ~usable)
        _check(not torch.any(batch.frames[i, :, 0, ~usable]) and not torch.any(labels.target_map[i, 0, ~usable]),
               "Right/bottom padding must contain zero signal")
        raw_mask = _read_native(source / "masks" / f"{center:05d}.png")
        _check(np.isin(raw_mask, [0, 255]).all(), "Unexpected published mask encoding")
        binary = raw_mask == 255
        np.testing.assert_array_equal(labels.target_map[i, 0, :h, :w].numpy(), binary)
        components = len(labels.objects[i])
        _check(components == row["components"] and sum(o.area for o in labels.objects[i]) == binary.sum(),
               "Mask objects do not agree with the independent native-pixel inventory")
        if role in ("positive", "negative"):
            _check(bool(components) == (role == "positive"), f"Requested sample polarity failed: {role}")
        with patch.object(Image, "open", wraps=Image.open) as opened:
            inference = dataset.inference_item(indices[i])
        _check([Path(call.args[0]) for call in opened.call_args_list] == paths,
               "Inference must open only the expected 17 image PNGs, in order")
        torch.testing.assert_close(inference.frames[0], batch.frames[i, :, :, :h, :w], rtol=0, atol=0)
        _check(inference.scene_meta == (meta,) and inference.output_frame_id == (frame_id,),
               "Inference metadata differs from supervised access")
        info = dict(role=role, frame_id=frame_id, original_hw=[h, w], components=components,
                    foreground_pixels=int(binary.sum()))
        rows.append((info, source_frames[8], binary))
    checks = ["heterogeneous_train_dataloader", "inference_fields_and_dtypes", "opencv_all_17_native_pixels",
              "center_index_offsets_ids_and_geometry", "native_mask_and_components",
              "right_bottom_padding_validity_and_ignore", "inference_opens_only_source_images",
              "inference_equals_supervised_inputs"]
    checks.extend(_verify_rejections())
    _save_figure(rows, figure_path)
    return dict(passed_tests=checks, sample_ids=list(batch.output_frame_id),
                batch_shapes=dict(frames=list(batch.frames.shape), time_offsets=list(batch.time_offsets.shape),
                                  pixel_valid=list(batch.pixel_valid.shape), target_map=list(labels.target_map.shape),
                                  ignore_map=list(labels.ignore_map.shape)),
                samples=[row[0] for row in rows], figure=Path(figure_path).name)
