"""Describe real-clip outputs without inventing mask ground truth or Pd/Fa."""
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

p = Path(__file__).resolve().parent
clip = p / 'first_val_clip'
images = np.load(clip / 'raw_frames.npy', mmap_mode='r')
ann = json.loads((clip / 'annotations.json').read_text())
with np.load(p / 'first_val_inference.npz', allow_pickle=False) as z:
    probs = z['probabilities']
    windows = z['windows']
    coverage = z['coverage']
    model_metadata = json.loads(str(z['metadata_json']))
assert probs.shape == images.shape and np.isfinite(probs).all()
rows = []
centers = []
for i, (score, rect, exist) in enumerate(zip(probs, ann['gt_rect'], ann['exist'])):
    x, y, w, h = map(float, rect)
    centers.append([x+w/2, y+h/2])
    left, top = max(0, int(np.floor(x))), max(0, int(np.floor(y)))
    right, bottom = min(score.shape[1], int(np.ceil(x+w))), min(score.shape[0], int(np.ceil(y+h)))
    box = score[top:bottom, left:right]
    inside_count = int((box > .5).sum())
    total_count = int((score > .5).sum())
    rows.append({'frame': ann['frame_indices'][i], 'visible': int(exist),
                 'width': w, 'height': h, 'gt_box_max_probability': float(box.max()),
                 'gt_box_mean_probability': float(box.mean()),
                 'above_0_5_pixels_inside_box': inside_count,
                 'above_0_5_pixels_outside_box': total_count-inside_count})
with (p / 'first_val_frame_responses.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
centers = np.asarray(centers)
steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
box_max = np.array([r['gt_box_max_probability'] for r in rows])
summary = {'status': 'official_checkpoint_forward_completed', 'frames': len(rows),
           'source_split': 'val', 'cross_dataset': True, 'diagnostic_seed': model_metadata['seed'],
           'native_size': list(images.shape[1:]), 'input_window': 40,
           'windows': windows.tolist(), 'coverage_min_max': [int(coverage.min()), int(coverage.max())],
           'checkpoint_sha256': model_metadata['checkpoint_sha256'],
           'target_width_range': [min(r['width'] for r in rows), max(r['width'] for r in rows)],
           'target_height_range': [min(r['height'] for r in rows), max(r['height'] for r in rows)],
           'annotation_center_step_pixels': {'median': float(np.median(steps)), 'p90': float(np.quantile(steps,.9)), 'max': float(steps.max())},
           'box_max_probability': {'min': float(box_max.min()), 'median': float(np.median(box_max)), 'max': float(box_max.max())},
           'frames_with_any_above_0_5_pixel_in_gt_box': int((box_max>.5).sum()),
           'metric_limits': ['Box response is not an object match or Pd.',
                             'Outside-box responses are not necessarily false alarms: dataset annotates designated target.',
                             'No exact masks or target stop/start event labels; not a test of point-target stationary failures.',
                             'Acausal model; wall time on shared GPU is not a deployment latency measurement.'],
           'local_new_method_tested': False}
(p / 'FIRST_INFERENCE_RESULT.json').write_text(json.dumps(summary, indent=2)+'\n')
Image.fromarray(images[0]).save(p / 'first_val_original_frame0.png')
print(json.dumps(summary,indent=2))
