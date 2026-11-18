"""Create paired, label-blind motion maps for the civilian-traffic development pilot."""
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
RUN = OUT / 'runs'
TRAIN = ['DJI_0030_9', 'DJI_0043_2', 'DJI_0057_1', 'DJI_0115_2', 'DJI_0323_2']
DEV = ['DJI_0026_2', 'DJI_0067_2']
DISCOVER = (-1, 2, 3, -4, 5, -6, -7, 8)
SPANS = ((-1, 2, 3, -4), (-1, 2, 5, -6), (3, -4, -7, 8))
PERTURB = np.stack(np.meshgrid(np.arange(-1, 1.01, .5), np.arange(-1, 1.01, .5)), -1).astype('float32')
SPEEDS = np.array([(x, y) for y in range(-2, 3) for x in range(-2, 3) if (x, y) != (0, 0)], dtype='float32')


def signal(im):
    raw = im.astype('float32')
    local = cv2.GaussianBlur(raw, (0, 0), 3)
    smoothed = cv2.GaussianBlur(raw, (0, 0), .8)
    var = cv2.GaussianBlur(raw * raw, (0, 0), 3) - local * local
    z = (smoothed - local) / np.sqrt(np.maximum(var, 0) + 4)
    return z, np.abs(raw - local)


def positions(center):
    strength = np.abs(center)
    local_peaks = strength == cv2.dilate(strength, np.ones((5, 5), np.uint8))
    ys, xs = np.where(local_peaks & (strength > 0))
    h, w = center.shape
    usable = (xs >= 25) & (ys >= 25) & (xs < w-25) & (ys < h-25)
    xs, ys = xs[usable], ys[usable]
    tiles = (ys // 24) * ((w+23) // 24) + xs // 24
    order = np.lexsort((xs, ys, -strength[ys, xs], tiles))
    chosen = []
    last = -1
    kept = []
    for i in order:
        if tiles[i] != last:
            last = tiles[i]
            kept = []
        if len(kept) < 2 and all(max(abs(xs[i]-x), abs(ys[i]-y)) > 2 for x, y in kept):
            kept.append((xs[i], ys[i]))
            chosen.append(i)
    return np.stack((xs[chosen], ys[chosen]), -1).astype('int32')


def sample(im, coords):
    x, y = coords[..., 0], coords[..., 1]
    col, row = x.astype('int32'), y.astype('int32')
    dx = x-col
    dy = y-row
    return ((1-dy)*((1-dx)*im[row,col] + dx*im[row,col+1])
            + dy*((1-dx)*im[row+1,col] + dx*im[row+1,col+1])).astype('float32')


def features(window):
    views = [signal(im) for im in window]
    center, contrast = views[8]
    xy = positions(center)
    x, y = xy.T
    sign = np.sign(center[y, x])
    common = np.stack((np.abs(center[y,x]), contrast[y,x]), -1)
    scores = np.mean([sample(views[8+t][0], xy[:,None,:]+t*SPEEDS[None,:,:])
                      for t in DISCOVER], axis=0) * sign[:,None]
    best = scores.argmax(axis=1)
    chosen = np.stack((np.zeros((len(xy),2), 'float32'), SPEEDS[best]), axis=1)
    still = np.mean([views[8+t][0][y,x] for t in DISCOVER], axis=0)*sign
    discovery = np.stack((still, scores[np.arange(len(xy)),best]), axis=-1)
    maps = np.empty((len(xy), 2, 2, 3, 5, 5), 'float32')
    vel = chosen[:,:,None,None,:]+PERTURB[None,None,...]
    for group, flip in enumerate((1,-1)):
        # Reuse shared time measurements across all three spans.
        samples = {}
        for shift in DISCOVER:
            t = shift*flip
            coords = xy[:,None,None,None,:]+t*vel
            samples[shift] = sample(views[8+t][0],coords) * sign[:,None,None,None]
        for j, span in enumerate(SPANS):
            maps[:,:,group,j] = np.mean([samples[t] for t in span],axis=0)
    return dict(maps=maps, common=common, discovery=discovery, xy=xy)


def labels(annotations):
    targets, ignore = [], []
    for a in annotations:
        x,y,w,h = a['bbox']
        valid = (a['category_id'] in (0,1,2,3,4) and not a.get('ignore',0)
                 and not a.get('iscrowd',0) and 0 < w*h <= 144 and max(w,h) <= 16)
        (targets if valid else ignore).append([x,y,w,h])
    return targets, ignore


def main():
    cv2.setNumThreads(1)
    RUN.mkdir(exist_ok=True)
    frozen = json.loads((ROOT/'results/references/DATA_SELECTION.json').read_text())['splits']
    assert set(TRAIN) <= set(frozen['train']) and set(DEV) == set(frozen['calibration'])
    assert not set(TRAIN+DEV) & set(frozen['confirmation'])
    frames = []
    arrays = defaultdict(list)
    for seq in TRAIN+DEV:
        base = ROOT/'data/stage1'/seq
        meta = json.loads((base/'annotations.json').read_text())
        images = sorted(meta['images'],key=lambda im:im['frame_id'])
        assert [im['frame_id'] for im in images] == list(range(160))
        raw = np.load(base/'raw_frames.npy',mmap_mode='r')
        assert raw.shape == (160, 512, 640)
        by_img = defaultdict(list)
        for a in meta['annotations']:
            by_img[a['image_id']].append(a)
        for idx in range(8,len(raw)-8,17):
            values = features(raw[idx-8:idx+9])
            target,ignore = labels(by_img[images[idx]['id']])
            slot=len(frames)
            frames.append(dict(sequence=seq, split='train' if seq in TRAIN else 'calibration',
                               index=idx, targets=target, ignore=ignore))
            for k,value in values.items():
                arrays[k].append(value)
            arrays['frame'].append(np.full(len(values['xy']),slot,'int32'))
        print(seq, 'finished', flush=True)
    (RUN/'labels.json').write_text(json.dumps({'frames':frames},separators=(',',':'))+'\n')
    data={k:np.concatenate(a) for k,a in arrays.items()}
    assert np.isfinite(data['maps']).all() and data['maps'].shape[1:] == (2,2,3,5,5)
    np.savez(RUN/'records.npz',**data)
    print('centers',len(frames),'candidates',len(data['xy']),flush=True)


if __name__=='__main__':
    main()
