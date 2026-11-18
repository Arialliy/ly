"""Decode a prespecified real validation clip, preserving native resolution."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--video', type=Path, required=True)
p.add_argument('--annotation', type=Path, required=True)
p.add_argument('--output-dir', type=Path, required=True)
p.add_argument('--start', type=int, default=0)
p.add_argument('--frames', type=int, default=160)
a = p.parse_args()
annotation = json.loads(a.annotation.read_text())
cap = cv2.VideoCapture(str(a.video))
if not cap.isOpened():
    raise RuntimeError('Cannot open video')
fps = cap.get(cv2.CAP_PROP_FPS)
count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frames = []
max_channel_difference = 0
for i in range(a.start + a.frames):
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f'Video ended at frame {i}, before requested clip end')
    if i < a.start:
        continue
    max_channel_difference = max(max_channel_difference, int(np.ptp(frame.astype(np.int16), axis=2).max()))
    # Original DeepPro PIL loader keeps channel zero of an RGB image.
    frames.append(frame[:, :, 2].copy())
cap.release()
frames = np.stack(frames)
assert len(annotation['exist']) == count == len(annotation['gt_rect'])
assert frames.shape[0] >= 40
a.output_dir.mkdir(parents=True, exist_ok=True)
np.save(a.output_dir / 'raw_frames.npy', frames)
clip_annotation = {
    'exist': annotation['exist'][a.start:a.start+a.frames],
    'gt_rect': annotation['gt_rect'][a.start:a.start+a.frames],
    'coordinate_format': 'xywh, native image coordinates',
    'frame_indices': list(range(a.start, a.start+a.frames)),
}
(a.output_dir / 'annotations.json').write_text(json.dumps(clip_annotation, indent=2)+'\n')
metadata = {
    'video': str(a.video.resolve()), 'annotation': str(a.annotation.resolve()),
    'video_sha256': hashlib.sha256(a.video.read_bytes()).hexdigest(),
    'annotation_sha256': hashlib.sha256(a.annotation.read_bytes()).hexdigest(),
    'decoded_frames_sha256': hashlib.sha256(frames.tobytes()).hexdigest(),
    'frame_interval': [a.start, a.start+a.frames], 'interval_convention': 'zero-based, end-exclusive',
    'original_video_frames': count, 'fps': fps, 'shape': list(frames.shape),
    'dtype': str(frames.dtype), 'min': int(frames.min()), 'max': int(frames.max()),
    'max_channel_difference': max_channel_difference,
    'channel_policy': 'red channel after decoding BGR, equivalent to original PIL channel zero',
    'resize': None, 'interpolation': None, 'source_split': 'val',
    'purpose': 'cross-dataset diagnostic; not paper benchmark reproduction',
}
(a.output_dir / 'provenance.json').write_text(json.dumps(metadata, indent=2)+'\n')
print(json.dumps(metadata, indent=2))
