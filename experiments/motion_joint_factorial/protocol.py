"""Fixed data, sampling and runtime settings for the civil traffic 2x2 comparison."""
import functools
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = HERE / 'runs'
OLD = HERE.parent / 'motion_hypothesis_traffic_pilot_20261118'
sys.path.insert(0, str(OLD))
from extract import DEV, TRAIN, labels as split_boxes
from readout import contacts

CENTERS = (19, 59, 99, 139)
SEED = 46
ARMS = 'ABCD'
STEPS = 300
BATCH = 128


def setup():
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    splits = json.loads((ROOT / 'results/references/DATA_SELECTION.json').read_text())['splits']
    assert set(TRAIN) <= set(splits['train']) and set(DEV) == set(splits['calibration'])
    assert not set(TRAIN + DEV) & set(splits['confirmation'])


@functools.lru_cache(None)
def metadata(seq):
    return json.loads((ROOT / 'data/stage1' / seq / 'annotations.json').read_text())


def frame_record(seq, idx):
    meta = metadata(seq)
    images = [im['id'] for im in meta['images'] if im['frame_id'] == idx]
    if len(images) != 1:
        raise ValueError((seq, idx, images))
    good, ignore = split_boxes(a for a in meta['annotations'] if a['image_id'] == images[0])
    return dict(sequence=seq, index=idx, targets=good, ignore=ignore,
                split='train' if seq in TRAIN else 'calibration')


def frame_path(group, seq, idx, suffix='.npz'):
    return RUN / group / f'{seq}_{idx:03d}{suffix}'


def workers_frames(seqs, shard, workers):
    return [(s, i) for s in seqs for i in CENTERS][shard::workers]


def raw_frames(seq):
    data = np.load(ROOT / 'data/stage1' / seq / 'raw_frames.npy', mmap_mode='r')
    assert data.shape == (160, 512, 640) and data.dtype == np.uint8
    return data


def save_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.partial.npz')
    np.savez(temp, **arrays)
    temp.replace(path)


def training_points(seq, idx, base):
    """Center-supervised positive pixels, plus score-hard and uniform negatives."""
    h, w = base.shape
    record = frame_record(seq, idx)
    positives = set()
    for x, y, bw, bh in record['targets']:
        cx, cy = int(np.floor(x+bw/2)), int(np.floor(y+bh/2))
        for dx, dy in ((0,0),(-1,0),(1,0),(0,-1),(0,1)):
            px, py = cx+dx, cy+dy
            if 0 <= px < w and 0 <= py < h and x <= px <= x+bw and y <= py <= y+bh:
                positives.add((px,py))
    positive = np.array(sorted(positives), dtype=np.int32).reshape(-1,2)
    avoid = record['targets'] + record['ignore']
    peaks = base == cv2.dilate(base, np.ones((5,5), np.uint8))
    ys,xs = np.where(peaks)
    order = np.lexsort((xs, ys, -base[ys,xs]))
    hard = np.stack((xs[order],ys[order]),axis=1)
    hard = hard[~contacts(hard, avoid).any(1)][:128]
    rng = np.random.default_rng(np.random.SeedSequence([SEED, TRAIN.index(seq), idx]))
    # Sampling without replacement keeps point supervision and arm exposure identical.
    flat = rng.choice(h*w, 2048, replace=False)
    uniform = np.column_stack((flat%w,flat//w))
    uniform = uniform[~contacts(uniform,avoid).any(1)]
    occupied = set(map(tuple,hard))
    uniform = np.asarray([p for p in uniform if tuple(p) not in occupied][:128])
    if len(hard) != 128 or len(uniform) != 128:
        raise RuntimeError("Insufficient valid background pixels")
    xy = np.vstack((positive, hard, uniform)).astype('int32')
    target = np.arange(len(xy)) < len(positive)
    return xy, target
