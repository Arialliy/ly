"""Frozen four-window detection exposure plus identical global auxiliary crops."""
import copy
import hashlib
import json
import sys
from types import SimpleNamespace
import numpy as np
from runtime import ROOT, E, Q, CP, SOURCE, DATA, sha

sys.path.insert(0, str(ROOT/'sources/temporal_diagnosis_20261010/tlfp_pilot_20261012/code'))
import train_tlfp as historical
sys.path.insert(0, str(E/'train_modules'))


def inputs():
    args = SimpleNamespace(plan=Q/'stage1_goal/pilot/TRAIN_PLAN.json',
        selection=Q/'stage1_goal/DATA_SELECTION.json', checkpoint=CP,
        source=SOURCE, dataset_root=DATA)
    base, plan, names, sequences, source = historical.previous.validate_inputs(args)
    old = [json.loads(x) for x in historical.OLD_LOG.read_text().splitlines()]
    assert len(old) == 512
    return base, plan, sequences, old


def auxiliary_entry(entry, base, sequences):
    """Cycle all 63 full-frame 128px crop positions, independent of responses.

    Overlapping 64px grid gives global spatial coverage. These are globally
    sampled candidates, NOT an exhaustive whole-frame teacher maximum.
    """
    out = copy.deepcopy(entry)
    grid = [(y,x) for y in range(0,385,64) for x in range(0,513,64)]
    for pi, pair in enumerate(out['pairs']):
        # 17 is coprime to 63; both source slots cycle every global location.
        idx = (entry['step']*17 + pi*29 + 46) % len(grid)
        if list(grid[idx]) == pair['crop_yx']:
            idx = (idx+1) % len(grid)
        pair['crop_yx'] = list(grid[idx])
    out['extra_specs'] = base.extra_specs_for_pairs(out['pairs'], sequences)
    return out


def batch_data(index, state):
    base, plan, sequences, old = state
    entry = plan['entries'][index]
    x, positives, bg, specs = base.build_batch(entry, sequences)
    y, valid, label_stats = historical.build_targets(entry,sequences,background_masks=bg)
    input_sha = historical.data_hash(x,positives,bg,specs)
    label_sha = historical.label_hash(y,valid)
    assert input_sha == old[index]['input_sha256']
    assert label_sha == old[index]['label_sha256']
    aux = auxiliary_entry(entry,base,sequences)
    ax, apos, abg, aspecs = base.build_batch(aux,sequences)
    aux_sha = historical.data_hash(ax,apos,abg,aspecs)
    return dict(x=np.concatenate([x,ax]),targets=y,valid=valid,
        det_background=bg,background=np.concatenate([bg,abg]),
        entry=entry,aux_entry=aux,input_sha256=input_sha,label_sha256=label_sha,
        auxiliary_input_sha256=aux_sha,label_stats=label_stats)


def window_annotations(batch, state):
    base, _, sequences, _ = state
    windows = []
    for entry in (batch['entry'],batch['aux_entry']):
        for pair in entry['pairs']:
            seq=sequences[pair['sequence']]; cy,cx=pair['crop_yx']
            for start in pair['starts']:
                frames=[]
                for anns in seq.annotations[start:start+40]:
                    local=[]
                    for ann in anns:
                        value=copy.deepcopy(ann)
                        if base.valid_box(ann):
                            x,y,w,h=ann['bbox']
                            value['bbox']=[x-cx,y-cy,w,h]
                            if x+w<=cx or y+h<=cy or x>=cx+128 or y>=cy+128:
                                continue
                        local.append(value)
                    frames.append(local)
                windows.append(frames)
    return windows


def source_fingerprints():
    paths=[E/'code/data.py',E/'code/train.py',E/'code/prepare.py',E/'code/calibrate.py',
        E/'code/runtime.py',E/'train_modules/track_pair_sampler.py',
        E/'train_modules/region_contrast_loss.py',E/'train_modules/reliable_keep_loss.py',
        historical.LABEL_PATH,historical.TOP1_PATH,
        Q/'mutual_temporal_pilot_20261012/code/train_mutual.py',
        Q/'top1_background_control_20261011/code/train_pilot.py']
    return {str(p):sha(p) for p in paths}


def cache_entry(index):
    path=E/'cache'/f'entry_{index:03d}.npz'
    with np.load(path,allow_pickle=False) as source:
        return {k:source[k] for k in source.files}


def cache_tensors(cache, device):
    import torch
    return {k:torch.from_numpy(v).to(device) for k,v in cache.items()
        if k not in ('metadata_json',)}
