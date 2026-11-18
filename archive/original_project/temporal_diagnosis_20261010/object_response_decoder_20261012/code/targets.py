"""Frozen, annotation-only object response supervision and compact target cache.

All geometry reuses the already registered Gaussian/valid-domain helpers.
No developer/evaluation images or model predictions enter this module.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

Q = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location('ord_legacy_gaussian_labels', Q/'supervision_alignment_control_20261012/code/gaussian_labels.py')
legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(legacy)
base = legacy.base

LABEL_CONFIG = {
    'version': 1,
    'identity': '(sequence, category_id, str(tracking_id)); retain numeric track_id without editing source',
    'support': 'unchanged legacy gaussian_region floor(255*g)>0, sigma=w/6,h/6, bbox truncated',
    'positive': 'all valid non-ignore/non-crowd boxes fully inside crop core9, nonempty support after exclusions',
    'valid': 'exact legacy build_targets valid domain',
    'background': 'exact legacy build_batch full-GT bbox+2 exclusion and core9',
    'negative_eligibility': 'all window-valid tracks observed now AND no excluded-object bbox+2 intersects pre-exclusion core9 AND no empty-support positive object now',
    'missing': 'unknown; no inferred absence or interpolation',
    'overlap': 'separate object supports, allowed to overlap; excluded bbox+2 overrides all supports',
    'empty_support': 'omit positive and assignment; disable negatives on that frame; no forced point',
    'coordinates': 'original full-frame float bbox; crop origin y,x retained; flattened local raster row-major indices',
    'length': 40, 'patch': 128, 'core_margin': 9,
}


def _canonical_key(sequence, ann):
    if 'tracking_id' not in ann or 'category_id' not in ann or 'track_id' not in ann:
        raise ValueError('Object annotation lacks author identity fields')
    return (str(sequence), int(ann['category_id']), str(ann['tracking_id']))


def _guard_rect(ann, crop_yx):
    cy, cx = crop_yx
    x, y, w, h = map(float, ann['bbox'])
    return base.pixel_rect((x-cx-2, y-cy-2, w+4, h+4))


def _intersects_core(rect):
    x0,y0,x1,y1=rect
    return max(x0,9)<min(x1,119) and max(y0,9)<min(y1,119)


def background_mask(sequence_data, start, crop_yx):
    """Equivalent annotation part of frozen build_batch, without reading pixels."""
    result=np.zeros((40,128,128),dtype=np.bool_)
    result[:,9:119,9:119]=True
    for t,anns in enumerate(sequence_data.annotations[start:start+40]):
        for ann in anns:
            if base.valid_box(ann):
                x0,y0,x1,y1=_guard_rect(ann,crop_yx)
                result[t,y0:y1,x0:x1]=False
    return result


def target_digest(target):
    """Hash complete semantic target, independent of NPZ compression and stats."""
    h=hashlib.sha256()
    h.update(json.dumps({k:target[k] for k in ('sequence','start','crop_yx','track_keys')},sort_keys=True,separators=(',',':')).encode())
    for k in ('valid','background','negative_eligible'):
        a=np.asarray(target[k],dtype=np.bool_)
        h.update(k.encode());h.update(str(a.shape).encode());h.update(a.tobytes(order='C'))
    for obs in target['observations']:
        h.update(json.dumps({k:v for k,v in obs.items() if k!='support_indices'},sort_keys=True,separators=(',',':')).encode())
        h.update(np.asarray(obs['support_indices'],dtype='<i8').tobytes(order='C'))
    return h.hexdigest()


def build_window_targets(sequence_data, window_spec, background_masks=None):
    """Return compact dict targets for one frozen 40×128×128 crop window.

    window_spec = {sequence: str, start: int, crop_yx: (y,x)}.
    background_masks optionally accepts the exact old build_batch single-window
    bool [40,128,128] slice; it is checked against annotation reconstruction.
    """
    sequence=str(window_spec.get('sequence',sequence_data.name))
    if sequence!=sequence_data.name:
        raise ValueError('Sequence key mismatch')
    start=int(window_spec['start']);crop=tuple(int(x) for x in window_spec['crop_yx'])
    if not (0<=start<=len(sequence_data.annotations)-40) or len(crop)!=2 or not (0<=crop[0]<=384 and 0<=crop[1]<=512):
        raise ValueError('Illegal 40-frame window or original-image crop origin')
    expected_bg=background_mask(sequence_data,start,crop)
    if background_masks is not None:
        supplied=np.asarray(background_masks)
        if supplied.dtype!=np.bool_ or supplied.shape!=(40,128,128) or not np.array_equal(supplied,expected_bg):
            raise ValueError('Background differs from frozen all-GT build_batch rule')
    background=expected_bg
    valid=background.copy();observations=[]
    excluded_intersects=np.zeros(40,dtype=np.bool_)
    empty_support_frames=np.zeros(40,dtype=np.bool_)
    stats=Counter();seen=set();frame_keys=[]
    for t,anns in enumerate(sequence_data.annotations[start:start+40]):
        positive=[];excluded=[]
        for ann in sorted(anns,key=lambda a:a['id']):
            if not base.valid_box(ann):
                stats['invalid_box_frames']+=1
                continue
            if base.positive_annotation(ann) and base.full_core(ann,crop):
                key=_canonical_key(sequence,ann)
                if (t,key) in seen:
                    raise ValueError('Duplicate author track key within one frame')
                seen.add((t,key))
                rect,q=legacy.gaussian_region(ann['bbox'],crop)
                x0,y0,x1,y1=rect
                valid[t,y0:y1,x0:x1]=True
                positive.append((ann,key,rect,q))
                stats['full_core_positive_box_frames']+=1
            else:
                excluded.append(ann);stats['excluded_box_frames_all_image']+=1
        for ann in excluded:
            x0,y0,x1,y1=_guard_rect(ann,crop)
            valid[t,y0:y1,x0:x1]=False
            if _intersects_core((x0,y0,x1,y1)):
                excluded_intersects[t]=True
                stats['excluded_guard_intersecting_core_box_frames']+=1
        now=set()
        for ann,key,rect,q in positive:
            x0,y0,x1,y1=rect
            support=(q>0)&valid[t,y0:y1,x0:x1]
            yy,xx=np.nonzero(support)
            if len(yy)==0:
                stats['empty_support_positive_box_frames']+=1
                stats['empty_support_raw_quantization' if not np.any(q>0) else 'empty_support_after_exclusion']+=1
                empty_support_frames[t]=True
                continue
            indices=((yy+y0)*128+(xx+x0)).astype(np.int64)
            observations.append(dict(frame=t,track_key=key,annotation_id=int(ann['id']),numeric_track_id=int(ann['track_id']),bbox_original=tuple(float(v) for v in ann['bbox']),support_indices=indices))
            now.add(key)
            stats['valid_object_frames']+=1
            stats['object_support_pixels_with_overlap_multiplicity']+=len(indices)
        frame_keys.append(now)
    track_keys=tuple(sorted({o['track_key'] for o in observations}))
    track_index={k:i for i,k in enumerate(track_keys)}
    observations.sort(key=lambda o:(o['frame'],o['track_key'],o['annotation_id']))
    for o in observations:o['track_index']=track_index[o['track_key']]
    negative=np.asarray([len(keys)==len(track_keys) and not excluded_intersects[t] and not empty_support_frames[t] for t,keys in enumerate(frame_keys)],dtype=np.bool_)
    stats.update({'tracks':len(track_keys),'negative_eligible_frames':int(negative.sum()),'negative_ineligible_frames':int((~negative).sum()),'empty_support_frames':int(empty_support_frames.sum()),'excluded_guard_frames':int(excluded_intersects.sum()),'missing_window_track_frames':sum(len(keys)!=len(track_keys) for keys in frame_keys),'valid_pixels':int(valid.sum()),'background_pixels':int(background.sum()),'empty_background_frames':int((background.reshape(40,-1).sum(1)==0).sum()),'empty_valid_frames':int((valid.reshape(40,-1).sum(1)==0).sum())})
    result=dict(sequence=sequence,start=start,crop_yx=crop,height=128,width=128,valid=valid,background=background,negative_eligible=negative,observations=observations,track_keys=track_keys,stats=dict(stats))
    if np.any(background&~valid):raise AssertionError('Background must lie in the shared valid domain')
    result['digest']=target_digest(result)
    return result


def union_response(target):
    """For legacy label equality audit only; not per-object persistent storage."""
    result=np.zeros((40,128,128),dtype=np.float32)
    flat=result.reshape(40,-1)
    for obs in target['observations']:flat[obs['frame'],obs['support_indices']]=1
    return result


def save_target(path, target):
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    metadata={k:target[k] for k in ('sequence','start','crop_yx','height','width','track_keys','stats','digest')}
    metadata['observations']=[{k:v for k,v in o.items() if k!='support_indices'} for o in target['observations']]
    indices=[o['support_indices'] for o in target['observations']]
    offsets=np.cumsum([0]+[len(a) for a in indices],dtype=np.int64)
    joined=np.concatenate(indices) if indices else np.empty(0,dtype=np.int64)
    np.savez_compressed(path,metadata=np.asarray(json.dumps(metadata,sort_keys=True,separators=(',',':'))),valid=np.packbits(target['valid'].reshape(-1)),background=np.packbits(target['background'].reshape(-1)),negative_eligible=target['negative_eligible'],support_offsets=offsets,support_indices=joined)


def load_target(path):
    with np.load(path,allow_pickle=False) as z:
        result=json.loads(str(z['metadata'].item()))
        result['crop_yx']=tuple(result['crop_yx']);result['track_keys']=tuple(tuple(k) for k in result['track_keys'])
        for k in ('valid','background'):result[k]=np.unpackbits(z[k]).reshape(40,128,128).astype(np.bool_)
        result['negative_eligible']=z['negative_eligible'].astype(np.bool_)
        offsets=z['support_offsets'];indices=z['support_indices']
        for i,obs in enumerate(result['observations']):
            obs['track_key']=tuple(obs['track_key']);obs['bbox_original']=tuple(obs['bbox_original']);obs['support_indices']=indices[offsets[i]:offsets[i+1]].copy()
    if target_digest(result)!=result['digest']:raise ValueError('Cached target semantic digest mismatch')
    return result


def window_specs(entry):
    return [dict(sequence=p['sequence'],start=int(start),crop_yx=tuple(p['crop_yx'])) for p in entry['pairs'] for start in p['starts']]


def build_entry_targets(entry, sequences, background_masks=None):
    """Return four dicts in exactly the frozen build_batch input order."""
    if background_masks is not None and (background_masks.shape!=(4,40,128,128) or background_masks.dtype!=np.bool_):
        raise ValueError('Expected four-window legacy boolean background masks')
    return [build_window_targets(sequences[spec['sequence']],spec,None if background_masks is None else background_masks[wi]) for wi,spec in enumerate(window_specs(entry))]


def targets_digest(targets):
    return hashlib.sha256(''.join(target['digest'] for target in targets).encode()).hexdigest()


def precompute_plan_targets(sequences, plan, cache_directory, verify_legacy=True, save_cache=True):
    """Compute all 2,048 old exposures once, with exact old B/V/union checks.

    Data duplication across exposure occurrences is retained intentionally to
    bind the cache to the fixed exposure plan. Four training cycles reuse it.
    """
    cache_directory=Path(cache_directory)
    if cache_directory.exists() and any(cache_directory.iterdir()):raise FileExistsError(cache_directory)
    cache_directory.mkdir(parents=True,exist_ok=True)
    totals=Counter();receipts=[];aggregate=hashlib.sha256()
    for entry in plan['entries']:
        if verify_legacy:
            _,_,bg,_=base.build_batch(entry,sequences)
            union,oldvalid,_=legacy.build_targets(entry,sequences,bg)
        for wi,spec in enumerate(window_specs(entry)):
            target=build_window_targets(sequences[spec['sequence']],spec,bg[wi] if verify_legacy else None)
            if verify_legacy and (not np.array_equal(target['valid'],oldvalid[wi]) or not np.array_equal(union_response(target),union[wi])):
                raise AssertionError(f"Object response labels differ from legacy support/valid at step {entry['step']} window {wi}")
            relative=f"step_{int(entry['step']):04d}_window_{wi}.npz"
            if save_cache:save_target(cache_directory/relative,target)
            receipt=dict(step=int(entry['step']),window=wi,path=relative,semantic_sha256=target['digest'],sequence=spec['sequence'],start=spec['start'],crop_yx=spec['crop_yx'],stats=target['stats'])
            receipts.append(receipt);totals.update(target['stats']);aggregate.update(target['digest'].encode())
        if int(entry['step'])%64==0:print(json.dumps({'targets_step':int(entry['step']),'windows':len(receipts)}),flush=True)
    manifest=dict(version=1,label_config=LABEL_CONFIG,cache_npz_written=save_cache,exposures=len(receipts),old_steps=len(plan['entries']),repeat_cycles=4,new_steps=4*len(plan['entries']),semantic_stream_sha256=aggregate.hexdigest(),legacy_background_valid_and_union_exact=bool(verify_legacy),totals_once=dict(totals),totals_four_cycles={k:int(v)*4 for k,v in totals.items()},windows=receipts)
    (cache_directory/'MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    return manifest
