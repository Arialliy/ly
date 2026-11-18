"""Paired RGB traffic-frame count control with a frozen VisDrone model."""
from collections import Counter
from hashlib import sha256
from pathlib import Path
import json
import time

import cv2
import numpy as np
import onnxruntime as ort

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SEQUENCES = ["DJI_0043_2", "DJI_0057_1", "DJI_0115_2", "DJI_0217_2", "DJI_0067_2"]
CLASSES = {"pedestrian_and_people": (0, 1), "pedestrian": (0,)}
SHA = "5abd18e5b630c330e68fe56b2bdc3875b10fbea07fe6912bde3cb9e68b58ca23"


def detect(session, image, region):
    left, top, right, bottom = region
    crop = image[top:bottom, left:right]
    h, w = crop.shape[:2]
    scale = min(640/w, 640/h)
    rw, rh = int(w*scale), int(h*scale)
    px, py = (640-rw)//2, (640-rh)//2
    canvas = np.full((640, 640, 3), 114, np.uint8)
    canvas[py:py+rh, px:px+rw] = cv2.resize(crop, (rw, rh))
    tensor = np.ascontiguousarray((canvas[..., ::-1].astype(np.float32)/255.).transpose(2,0,1)[None])
    pred = session.run(["output0"], {"images": tensor})[0]
    assert pred.shape == (1, 14, 8400) and np.isfinite(pred).all()
    row = pred[0].T
    label = row[:,4:].argmax(1)
    score = row[:,4:].max(1)
    keep = (label <= 1) & (score >= .2)
    row, score, label = row[keep], score[keep], label[keep]
    xy = row[:,:2] - row[:,2:4]/2
    # Undo the exact integer letterbox resize before moving into full-frame coordinates.
    xy = (xy - [px, py]) * [w/rw, h/rh]
    end = xy + row[:,2:4] * [w/rw, h/rh]
    xy = np.clip(xy, [0,0], [w,h]) + [left,top]
    end = np.clip(end, [0,0], [w,h]) + [left,top]
    boxes = np.column_stack((xy, end-xy))
    valid = (boxes[:,2] > 0) & (boxes[:,3] > 0)
    return np.column_stack((boxes, score, label))[valid]


def suppress(rows):
    if len(rows) == 0:
        return rows
    indices = cv2.dnn.NMSBoxes(rows[:,:4].tolist(), rows[:,4].tolist(), 0.0, 0.5)
    return rows[np.asarray(indices, dtype=int).reshape(-1)]


def count_views(views):
    counts = {}
    for group, classes in CLASSES.items():
        kept = [suppress(v[np.isin(v[:,5], classes)]) for v in views]
        merged = suppress(np.concatenate(kept)) if len(kept) > 1 else kept[0]
        counts[group] = {str(t): int(np.count_nonzero(merged[:,4] >= t)) for t in (.5, .2)}
    return counts


def main():
    start = time.monotonic()
    cv2.setNumThreads(1)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    model_file = OUT/"best.onnx"
    assert sha256(model_file.read_bytes()).hexdigest() == SHA
    session = ort.InferenceSession(str(model_file), options, providers=["CPUExecutionProvider"])
    records, provenance = [], {}
    for seq in SEQUENCES:
        annotations = (ROOT/"data/stage1"/seq/"annotations.json").read_bytes()
        index = json.loads(annotations)
        assert [c["name"] for c in index["categories"] if c["id"] == 3] == ["pedestrian"]
        targets = Counter(a["image_id"] for a in index["annotations"] if a["category_id"] == 3)
        images = {i["frame_id"]: i for i in index["images"]}
        aggregate_digest = sha256()
        for frame in range(9, 160, 20):
            item = images[frame]
            path = ROOT/"data/public/VT-Tiny-MOT/train2017"/seq/"00"/Path(item["file_name"]).name
            data = path.read_bytes()
            aggregate_digest.update(sha256(data).digest())
            image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            assert image.shape == (512, 640, 3)
            h, w = image.shape[:2]
            cw, ch = round(w*.6), round(h*.6)
            regions = [(x, y, x+cw, y+ch) for x in (0, w-cw) for y in (0, h-ch)]
            records.append({"sequence":seq, "frame":frame, "G":targets[item["id"]],
                            "full":count_views([detect(session, image, (0,0,w,h))]),
                            "tiled":count_views([detect(session,image,r) for r in regions])})
        provenance[seq] = {"annotation_sha256":sha256(annotations).hexdigest(),
                           "rgb_ordered_sha256_digest_of_digests":aggregate_digest.hexdigest()}
    summary = {}
    for seq in SEQUENCES + ["pooled"]:
        rows = [r for r in records if seq == "pooled" or r["sequence"] == seq]
        g = sum(r["G"] for r in rows)
        metrics = {"frames":len(rows), "G":g}
        for group in CLASSES:
            metrics[group] = {}
            for mode in ("full","tiled"):
                for threshold in ("0.5","0.2"):
                    detections = [r[mode][group][threshold] for r in rows]
                    matched = sum(min(d,r["G"]) for d,r in zip(detections, rows))
                    metrics[group][mode+"_"+threshold] = {
                        "predictions":sum(detections), "max_match_count":matched, "min_shortfall":g-matched,
                        "max_match_rate":matched/g if g else None}
        summary[seq] = metrics
    results = {"model":"RISEF/yolov11s-visdrone", "commit":"16d8d1709ae6fd7a9920398fbed5612e6208e0f0",
               "model_sha256":SHA, "input_digests":provenance, "protocol_sha256":sha256((OUT/"PROTOCOL.md").read_bytes()).hexdigest(),
               "runtime":ort.__version__, "opencv":cv2.__version__,"providers":session.get_providers(),
               "runtime_seconds":time.monotonic()-start,"summary":summary, "frame_counts":records}
    (OUT/"results.json").write_text(json.dumps(results, indent=2, allow_nan=False)+"\n")
    print(json.dumps(summary["pooled"], indent=2), flush=True)


if __name__ == "__main__":
    main()
