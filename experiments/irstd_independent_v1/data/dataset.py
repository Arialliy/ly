"""Load released NUDT-MIRSDT grayscale images and center-frame masks from the S0 manifest."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from ..contracts import CENTER_INDEX, CLIP_LENGTH, ClipBatch, MaskObject, SceneMeta, Targets


def _frame_path(root, relative, source_id, number, role):
    """Enforce source, frame number and image/mask role before opening the native file."""
    path = Path(relative)
    if (path.is_absolute() or ".." in path.parts or len(path.parts) < 3 or
            path.parent.name != role or path.parent.parent.name != source_id or
            path.suffix.lower() != ".png" or not path.stem.isdecimal() or int(path.stem) != number):
        raise ValueError(f"Invalid {role} path for {source_id} frame {number}: {relative}")
    return root / path


def _load_gray(path):
    with Image.open(path) as image:
        if image.format != "PNG" or image.mode != "L":
            raise ValueError(f"Expected native 8-bit grayscale PNG at {path}, got {image.format}/{image.mode}")
        return np.array(image, dtype=np.uint8, copy=True)


def _components(mask):
    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    objects = []
    for k in range(1, n):
        x, y, w, h, area = map(int, stats[k])
        objects.append(MaskObject(area, (x, y, x + w, y + h), tuple(map(float, centroids[k]))))
    return tuple(objects)


class MaskClipDataset(Dataset):
    """A sample is (ClipBatch, Targets), both with B=1. Use collate_clips for batches.

    inference_item reads only the 17 images. Supervised access separately opens one mask.
    Window lookup uses actual frame numbers, without temporal padding or resizing.
    """
    def __init__(self, manifest, project_root, split):
        if split not in ("train", "dev", "test"):
            raise ValueError(f"Unknown split {split}")
        self.project_root = Path(project_root)
        self._frames = {}
        self.centers = []
        source_ids = set()
        for source in manifest["sequences"]:
            name = source["id"]
            if name in source_ids:
                raise ValueError(f"Repeated sequence id {name}")
            source_ids.add(name)
            if source["split"] != split:
                continue
            by_number = {}
            for record in source["frames"]:
                number = record["number"]
                if type(number) is not int or number in by_number:
                    raise ValueError(f"Noninteger or duplicate frame in {name}: {number}")
                by_number[number] = record
            centers = source["centers"]
            if len(centers) != len(set(centers)):
                raise ValueError(f"Duplicate output center in {name}")
            for center in centers:
                if type(center) is not int or any(
                        n not in by_number for n in range(center - CENTER_INDEX, center + CENTER_INDEX + 1)):
                    raise ValueError(f"17 consecutive physical frames required for {name}, center {center}")
                self.centers.append((name, center))
            self._frames[name] = by_number

    def __len__(self):
        return len(self.centers)

    def inference_item(self, index):
        source_id, center = self.centers[index]
        numbers = tuple(range(center - CENTER_INDEX, center + CENTER_INDEX + 1))
        records = self._frames[source_id]
        arrays = []
        for n in numbers:
            path = _frame_path(self.project_root, records[n]["image"], source_id, n, "images")
            arrays.append(_load_gray(path))
        shape = arrays[CENTER_INDEX].shape
        if any(frame.shape != shape for frame in arrays):
            raise ValueError(f"Frame geometry changes within {source_id} window at {center}")
        height, width = shape
        meta = SceneMeta(source_id, numbers, (height, width), (0, 0), (0, 0, width, height))
        frames = torch.from_numpy(np.stack(arrays)).to(torch.float32).div_(255.0)[None, :, None]
        offsets = torch.arange(CLIP_LENGTH, dtype=torch.float32)[None, :] - CENTER_INDEX
        frame_id = f"{source_id}/{Path(records[center]['image']).stem}"
        return ClipBatch(frames, offsets, torch.ones_like(frames, dtype=torch.bool),
                         CENTER_INDEX, (frame_id,), (meta,))

    def __getitem__(self, index):
        inputs = self.inference_item(index)
        source_id, center = self.centers[index]
        mask_path = self._frames[source_id][center]["mask"]
        path = _frame_path(self.project_root, mask_path, source_id, center, "masks")
        native = _load_gray(path)
        if native.shape != inputs.scene_meta[0].original_hw:
            raise ValueError(f"Center mask geometry mismatch at {path}")
        if np.any((native != 0) & (native != 255)):
            raise ValueError(f"Center mask at {path} has values outside the release convention 0/255")
        binary = (native > 0).astype(np.uint8)
        target = torch.from_numpy(binary).to(torch.float32)[None, None]
        return inputs, Targets("mask", target, torch.zeros_like(target, dtype=torch.bool), (_components(binary),))


def collate_clips(samples):
    """Pad differently sized native samples on bottom/right. Coordinates stay in the original scene."""
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    clips, targets = zip(*samples)
    max_h = max(c.frames.shape[-2] for c in clips)
    max_w = max(c.frames.shape[-1] for c in clips)
    pads = [(0, max_w-c.frames.shape[-1], 0, max_h-c.frames.shape[-2]) for c in clips]

    def pad_and_join(tensors, fill):
        return torch.cat([F.pad(t, p, value=fill) for t, p in zip(tensors, pads)], dim=0)

    batch = ClipBatch(pad_and_join([c.frames for c in clips], 0),
                      torch.cat([c.time_offsets for c in clips]),
                      pad_and_join([c.pixel_valid for c in clips], False),
                      CENTER_INDEX, tuple(c.output_frame_id[0] for c in clips),
                      tuple(c.scene_meta[0] for c in clips))
    labels = Targets("mask", pad_and_join([t.target_map for t in targets], 0),
                     pad_and_join([t.ignore_map for t in targets], True),
                     tuple(t.objects[0] for t in targets))
    return batch, labels
