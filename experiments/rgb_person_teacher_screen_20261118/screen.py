"""Unconstrained count ceilings on one-to-one RGB-to-thermal person coverage."""
from collections import Counter
from pathlib import Path
import json
import time

from PIL import Image
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.ops import nms
from torchvision.transforms.functional import to_tensor


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SEQUENCES = ("DJI_0043_2", "DJI_0057_1", "DJI_0115_2", "DJI_0217_2", "DJI_0067_2")


def predict(model, image, regions):
    boxes, scores = [], []
    for left, top, right, bottom in regions:
        crop = image.crop((left, top, right, bottom))
        result = model([to_tensor(crop).cuda()])[0]
        mask = (result["labels"] == 1) & (result["scores"] >= 0.2)
        shift = result["boxes"].new_tensor([left, top, left, top])
        boxes.append(result["boxes"][mask] + shift)
        scores.append(result["scores"][mask])
    boxes, scores = torch.cat(boxes), torch.cat(scores)
    if len(regions) > 1:
        keep = nms(boxes, scores, 0.5)
        boxes, scores = boxes[keep], scores[keep]
    return {"boxes": boxes.cpu().tolist(), "scores": scores.cpu().tolist()}


@torch.inference_mode()
def main():
    start = time.monotonic()
    model = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
    checkpoint = OUT / "fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth"
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.cuda().eval()
    records = []
    for seq in SEQUENCES:
        annotation_file = ROOT / "data/stage1" / seq / "annotations.json"
        ann = json.loads(annotation_file.read_text())
        assert [c["name"] for c in ann["categories"] if c["id"] == 3] == ["pedestrian"]
        counts = Counter(a["image_id"] for a in ann["annotations"] if a["category_id"] == 3)
        images = {i["frame_id"]: i for i in ann["images"]}
        for frame in range(9, 160, 20):
            basename = Path(images[frame]["file_name"]).name
            path = ROOT / "data/public/VT-Tiny-MOT/train2017" / seq / "00" / basename
            with Image.open(path) as img:
                assert img.mode == "RGB" and img.size == (640, 512)
                w, h = img.size
                cw, ch = round(0.6 * w), round(0.6 * h)
                crops = [(x, y, x+cw, y+ch) for x in (0, w-cw) for y in (0, h-ch)]
                records.append({"sequence": seq, "frame": frame, "G": counts[images[frame]["id"]],
                                "full": predict(model, img, [(0, 0, w, h)]),
                                "tiled": predict(model, img, crops)})
    summary = {}
    for seq in (*SEQUENCES, "pooled"):
        group = [r for r in records if seq == "pooled" or r["sequence"] == seq]
        annotated = sum(r["G"] for r in group)
        metrics = {"frames": len(group), "G": annotated}
        for mode in ("full","tiled"):
            for threshold in (0.5, 0.2):
                available = [sum(s >= threshold for s in r[mode]["scores"]) for r in group]
                ceiling = sum(min(n, r["G"]) for n,r in zip(available, group))
                metrics[f"{mode}_{threshold}"] = {"predictions": sum(available),
                    "max_match_count": ceiling, "min_shortfall": annotated-ceiling,
                    "max_match_rate": ceiling/annotated if annotated else None}
        summary[seq] = metrics
    (OUT / "predictions.json").write_text(json.dumps(records, allow_nan=False))
    result = {"runtime_seconds": time.monotonic()-start, "device": torch.cuda.get_device_name(),
              "torch": str(torch.__version__), "summary":summary}
    (OUT / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
