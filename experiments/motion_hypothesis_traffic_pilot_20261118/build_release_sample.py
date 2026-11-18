"""Cache the official forty-frame DeepPro-Plus peak scores and local motion evidence."""
import hashlib
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from extract import DEV, DISCOVER, PERTURB, ROOT, RUN, SPANS, SPEEDS, TRAIN, sample, signal

ARCHIVE = ROOT / 'archive/infrared_problem_screen_20261009'
DIAG = ARCHIVE / 'temporal_diagnosis_20261010'
NATIVE = DIAG / 'low_response_information_20261011/logits'
SOURCE = ARCHIVE / 'official_deeppro'
MODEL = SOURCE / 'log/sem_seg/RGB-T__2025-01-14_15-27__SoftLoUloss_DeepPro-Plus_DataL40'
CHECKPOINT = MODEL / 'checkpoints/best_model.pth'
PEAK_LIMIT = 512


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as file:
        for part in iter(lambda: file.read(2**22), b''):
            result.update(part)
    return result.hexdigest()


def native_starts(center, starts):
    feasible = [s for s in starts if s <= center-8 and center+8 < s+40]
    return min(feasible, key=lambda s: (abs(s+19.5-center), s))


def peaks(logits):
    height, width = logits.shape
    is_peak = logits == cv2.dilate(logits, np.ones((5,5), np.uint8))
    ys, xs = np.nonzero(is_peak)
    ranking = np.lexsort((xs, ys, -logits[ys, xs]))
    used = np.zeros((height+4, width+4), dtype=np.uint8)
    keep = []
    for i in ranking:
        x, y = xs[i], ys[i]
        if used[y+2, x+2]:
            continue
        used[y:y+5, x:x+5] = 1
        keep.append((x,y))
        if len(keep) == PEAK_LIMIT:
            break
    return np.asarray(keep, dtype=np.int32)


def motion_maps(window, xy):
    # 25 pixels covers all perturbations and the extra bilinear neighbor.
    # Outside the sensor use zero standardized contrast, without removing candidates.
    pad = 25
    z = [np.pad(signal(im)[0], pad) for im in window]
    xy = xy + pad
    x, y = xy.T
    polarity = np.sign(z[8][y,x])
    trials = [sample(z[8+t], xy[:,None,:]+t*SPEEDS) for t in DISCOVER]
    path_score = np.mean(trials, axis=0) * polarity[:,None]
    best = SPEEDS[np.argmax(path_score, axis=1)]
    chosen = np.stack([np.zeros_like(best),best],axis=1)
    perturbed = chosen[:,:,None,None,:] + PERTURB
    maps = np.empty((len(xy),2,2,3,5,5), dtype=np.float32)
    for group, flip in enumerate([1,-1]):
        samples = {t:sample(z[8+flip*t], xy[:,None,None,None,:]+flip*t*perturbed)
                   for t in DISCOVER}
        for j, group_times in enumerate(SPANS):
            maps[:,:,group,j] = np.mean([samples[t] for t in group_times],axis=0) * polarity[:,None,None,None]
    return maps


def main():
    out = RUN/'release_sample.npz'
    out_labels = RUN/'release_labels.json'
    if out.exists() or out_labels.exists():
        raise FileExistsError('Archived outputs already exist')
    freeze = json.loads((ROOT/'results/references/DATA_SELECTION.json').read_text())['splits']
    assert set(TRAIN)<=set(freeze['train']) and set(DEV)==set(freeze['calibration'])
    assert not set(TRAIN+DEV) & set(freeze['confirmation'])
    archived = json.loads((NATIVE/'COMPLETE.json').read_text())
    assert archived['status']=='COMPLETE'
    key = lambda p: str(p.relative_to(ROOT))
    input_sha = {key(CHECKPOINT):digest(CHECKPOINT), key(MODEL/'DeepPro-Plus.py'):digest(MODEL/'DeepPro-Plus.py')}
    assert input_sha[key(CHECKPOINT)]=='4fd4aa2736d9a33f2bf1459099068d8b4a7e202ff80aecdb4a3277fd13c7abcc'
    for dep in [SOURCE/'networks/layers/TPro.py',SOURCE/'networks/layers/basic.py',NATIVE/'COMPLETE.json',RUN/'labels.json',Path(__file__),Path(__file__).with_name('extract.py')]:
        input_sha[key(dep)] = digest(dep)
    torch.manual_seed(46)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    loader=module('release_loader',ROOT/'sources/temporal_diagnosis_20261010/top1_background_control_20261011/code/train_pilot.py')
    inference=module('release_tiler',DIAG/'infer_with_windows.py')
    model, loaded_source=loader.load_model(SOURCE, CHECKPOINT)
    assert loaded_source==MODEL/'DeepPro-Plus.py'
    model=model.cuda().eval()
    orig_labels=json.loads((RUN/'labels.json').read_text())['frames']
    frames=[]
    arrays=defaultdict(list)
    check={}
    for seq in TRAIN+DEV:
        src=ROOT/'data/stage1'/seq/'raw_frames.npy'
        input_sha[key(src)]=digest(src)
        imgs=np.load(src,mmap_mode='r')
        assert imgs.shape==(160,512,640) and imgs.dtype==np.uint8
        labels=[f for f in orig_labels if f['sequence']==seq and (seq in DEV or f['index'] in (8,25))]
        assert len(labels)==(9 if seq in DEV else 2)
        if seq in TRAIN:
            with torch.inference_mode():
                x=torch.from_numpy((imgs[:40].astype(np.float32)-85.0799)/47.4845)[None,None].cuda()
                predicted=inference.tiled_logits(model,x,128,16)[0].numpy()
            del x
        if seq==DEV[0]:
            # Top-left native tile from window 0, same halo, normalization, and padding.
            with torch.inference_mode():
                x=torch.from_numpy((imgs[:40,:144,:144].astype(np.float32)-85.0799)/47.4845)[None,None].cuda()
                actual=model(x)[1][0,8,:128,:128].cpu().numpy()
            cached=np.load(NATIVE/seq/'window_000_logits.npy',mmap_mode='r')[8,:128,:128]
            check=dict(sequence=seq, native_start=0, slot=8, corner_tile_shape=list(actual.shape),
                       max_abs_error=float(np.max(np.abs(actual-cached))),exact=np.array_equal(actual,cached))
            check['exact']=bool(check['exact'])
            if check['max_abs_error']>1e-4:
                raise AssertionError(check)
            del x
        for f in labels:
            idx=f['index']
            start=0 if seq in TRAIN else native_starts(idx,archived['starts'])
            slot=idx-start
            if seq in DEV:
                path=NATIVE/seq/f'window_{start:03d}_logits.npy'
                if key(path) not in input_sha:
                    input_sha[key(path)]=digest(path)
                    cached_hashes=[v for k,v in archived['output_sha256'].items() if k.endswith('/'+seq+'/'+path.name)]
                    assert cached_hashes==[input_sha[key(path)]]
                logits=np.load(path,mmap_mode='r')[slot]
            else:
                logits=predicted[slot]
            assert logits.shape==(512,640) and np.isfinite(logits).all()
            xy=peaks(logits)
            assert len(xy)==PEAK_LIMIT
            arrays['xy'].append(xy)
            arrays['base'].append(logits[xy[:,1],xy[:,0]])
            arrays['frame'].append(np.full(len(xy),len(frames),dtype=np.int32))
            arrays['maps'].append(motion_maps(imgs[idx-8:idx+9],xy))
            frames.append(dict(f,native_start=start,slot=slot))
        print(seq,len(labels),'centers done',flush=True)
    data={k:np.concatenate(v) for k,v in arrays.items()}
    assert len(frames)==28 and data['maps'].shape==(28*PEAK_LIMIT,2,2,3,5,5)
    assert np.isfinite(data['maps']).all()
    meta=dict(frames=frames,sha256=input_sha,native_model_parameters=sum(p.numel() for p in model.parameters()),
              native_backend=torch.cuda.get_device_name(0),verification=check,
              spatial_tile_core=128,spatial_tile_halo=16,existing_motion_offsets=list(DISCOVER),
              speed_units='pixels/frame',outside_image_response='zero after local standardization',
              peak_definition='native unfused 40-frame logits; 5x5 local maxima; descending score then ascending y,x; 2px Chebyshev NMS including ties; first 512',
              note='Both discovery and validation motion observations are read from raw frames; release peak location selection uses the full native 40-frame clip.')
    tmp=out.with_name('release_sample.partial.npz')
    np.savez(tmp,**data)
    tlabel=out_labels.with_suffix('.partial.json')
    tlabel.write_text(json.dumps(meta,ensure_ascii=False,separators=(',',':'))+'\n')
    tmp.replace(out)
    tlabel.replace(out_labels)
    print('done',data['maps'].shape,check,flush=True)

if __name__=='__main__':
    main()
