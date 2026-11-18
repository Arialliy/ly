"""Frozen approximate Gaussian-derived binary support; no image/model scoring.

The public loader binarizes nonzero mask values. This generator approximates
the unavailable preceding Gaussian rasterization, not the author's exact masks.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import numpy as np

LABEL_CONFIG = {
    'version': 1,
    'condition': 'gaussian_softiou_approx',
    'center': 'original_float_bbox_geometric_center',
    'pixel_coordinates': 'x=j+0.5,y=i+0.5; half-open original bbox',
    'sigma_x': 'original_bbox_width/6',
    'sigma_y': 'original_bbox_height/6',
    'support': 'truncate outside original bbox; no minimum sigma or peak normalization',
    'quantization': 'floor(255*Gaussian) to uint8',
    'target': 'quantized>0 cast float32; max union over positive boxes',
    'positive_objects': 'all valid non-ignore non-crowd boxes fully inside crop core9; no size/class filter',
    'valid_domain': 'old_background_mask union full_positive_box_rects, then excluded-object bbox+2px overrides',
    'excluded_objects': 'all valid ignore/crowd or non-full-core boxes; original float box+2px',
    'crop_core_margin': 9,
    'outside_positive_box_guard_margin': 2,
    'loss': 'unchanged official SoftLoUloss on gathered valid pixels per window; four-window mean',
    'smooth': 0.0,
    'empty_target_window': 'retain official loss1 and zero gradient; log occurrence',
    'alignment_limit': 'approximate mask generator and masked domain; not exact author training reproduction',
}

Q = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('gaussian_frozen_batch_helpers', Q/'top1_background_control_20261011/code/train_pilot.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def gaussian_region(box, crop_yx=(0, 0), height=128, width=128):
    """Return clipped raster rectangle and uint8 samples without moving center."""
    x, y, w, h = map(float, box)
    if not np.isfinite([x,y,w,h]).all() or w <= 0 or h <= 0:
        raise ValueError('Invalid Gaussian bbox')
    cy, cx = crop_yx
    rect = base.pixel_rect((x-cx, y-cy, w, h), height, width)
    x0,y0,x1,y1 = rect
    if x0 == x1 or y0 == y1:
        return rect, np.zeros((y1-y0,x1-x0),dtype=np.uint8)
    # Float64 evaluates the declared geometry before explicit uint8 quantization.
    xs = np.arange(x0,x1,dtype=np.float64)+cx+0.5
    ys = np.arange(y0,y1,dtype=np.float64)+cy+0.5
    exponent = -0.5*(((ys[:,None]-(y+h/2))/(h/6))**2 + ((xs[None,:]-(x+w/2))/(w/6))**2)
    quantized = np.floor(255*np.exp(exponent)).astype(np.uint8)
    return rect, quantized


def build_targets(entry, sequences, background_masks=None):
    """Construct one label/valid tensor for exactly the old four-window batch."""
    if background_masks is None:
        _,_,background_masks,_ = base.build_batch(entry,sequences)
    if background_masks.shape != (4,40,128,128) or background_masks.dtype != np.bool_:
        raise ValueError('Expected unchanged old background bool mask [4,40,128,128]')
    valid = background_masks.copy()
    target = np.zeros(valid.shape,dtype=np.float32)
    per_window = []
    wi = 0
    for pair in entry['pairs']:
        seq = sequences[pair['sequence']]
        crop = pair['crop_yx']
        cy,cx = crop
        for start in pair['starts']:
            count_positive=0; count_excluded=0; raw_support=0; zero_support=0
            for t,anns in enumerate(seq.annotations[start:start+40]):
                excluded=[]
                for ann in anns:
                    if not base.valid_box(ann):
                        continue
                    if base.positive_annotation(ann) and base.full_core(ann,crop):
                        rect,q = gaussian_region(ann['bbox'],crop)
                        x0,y0,x1,y1=rect
                        valid[wi,t,y0:y1,x0:x1]=True
                        binary=(q>0).astype(np.float32)
                        target[wi,t,y0:y1,x0:x1]=np.maximum(target[wi,t,y0:y1,x0:x1],binary)
                        raw_support+=int(np.count_nonzero(binary))
                        zero_support+=int(not np.any(binary))
                        count_positive+=1
                    else:
                        excluded.append(ann)
                        count_excluded+=1
                # Exclusion wins over union, including overlapping positive boxes.
                for ann in excluded:
                    x,y,w,h=ann['bbox']
                    x0,y0,x1,y1=base.pixel_rect((x-cx-2,y-cy-2,w+4,h+4))
                    valid[wi,t,y0:y1,x0:x1]=False
            target[wi,~valid[wi]]=0
            n=int(valid[wi].sum());mass=int(target[wi].sum())
            if not n:
                raise ValueError('No valid supervised pixels in a frozen window')
            per_window.append({'positive_box_frames':count_positive,'excluded_box_frames':count_excluded,
                               'valid_pixels':n,'target_positive_pixels':mass,
                               'sum_object_support_before_union_and_exclusion':raw_support,
                               'positive_box_frames_without_quantized_support':zero_support,
                               'pure_negative_window':mass==0})
            wi+=1
    if wi != 4 or not np.all(np.isin(target,[0,1])) or np.any(target[~valid]):
        raise AssertionError('Invalid binary supervision batch')
    stats={'per_window':per_window,'valid_pixels_per_window':[r['valid_pixels'] for r in per_window],
           'target_mass_per_window':[r['target_positive_pixels'] for r in per_window],
           'positive_boxes_per_window':[r['positive_box_frames'] for r in per_window],
           'pure_negative_windows':sum(r['pure_negative_window'] for r in per_window),
           'target_mass':int(target.sum()),'valid_fraction':float(valid.mean())}
    return target,valid,stats
