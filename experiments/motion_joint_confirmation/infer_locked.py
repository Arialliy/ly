"""Frozen all-pixel prediction, with a complete development-image parity witness."""
import argparse
import json
import time

from locked_protocol import CENTERS, COHORT, OLD_RUN, RUN, verify_lock
import numpy as np
import torch
from cache_backbone import native_center
from experiment import model_trainables
from model import FactorialDetector, load_official_detector
from motion_evidence import extract_evidence, prepare_window
from protocol import raw_frames, save_npz, setup


@torch.inference_mode()
def load_models():
    release=load_official_detector('cuda')
    models={}
    for arm in 'ABCD':
        model=FactorialDetector(release,arm).eval()
        weights=torch.load(OLD_RUN/'models'/f'{arm}.pt',map_location='cpu',weights_only=True)
        params=model_trainables(model)
        if weights.keys()!=params.keys():
            raise ValueError('Checkpoint keys differ from frozen model')
        for key,param in params.items():
            param.copy_(weights[key])
        models[arm]=model
    return release,models


@torch.inference_mode()
def score_frame(release,models,seq,idx):
    raw=raw_frames(seq)
    features,base=native_center(release,raw[idx-19:idx+21])
    window=prepare_window(torch.from_numpy(raw[idx-8:idx+9].copy()).cuda())
    h,w=base.shape
    packed=torch.from_numpy(features).cuda().view(-1,32)
    outputs={arm:torch.empty(h*w,device='cuda') for arm in models}
    for start in range(0,h*w,1024):
        stop=min(start+1024,h*w)
        ids=torch.arange(start,stop,device='cuda')
        xy=torch.stack((ids%w,ids//w),dim=-1)
        for selection,letters in (('full','AB'),('split','CD')):
            ev=extract_evidence(window,xy,selection)
            for arm in letters:
                outputs[arm][start:stop]=models[arm].forward_points(packed[start:stop],ev)
    maps={'original':base,**{a:m.reshape(h,w).cpu().numpy() for a,m in outputs.items()}}
    if not all(np.isfinite(a).all() for a in maps.values()):
        raise RuntimeError('Nonfinite scores')
    return maps


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--parity',action='store_true')
    p.add_argument('--shard',type=int,default=0)
    p.add_argument('--workers',type=int,default=1)
    a=p.parse_args()
    if not 0<=a.shard<a.workers:
        p.error('invalid shard')
    setup()
    verify_lock()
    release,models=load_models()
    if a.parity:
        seq,idx='DJI_0026_2',19
        scores=score_frame(release,models,seq,idx)
        with np.load(OLD_RUN/'predictions'/f'{seq}_{idx:03d}.npz') as archive:
            error={key:float(np.abs(arr-archive[key]).max()) for key,arr in scores.items()}
        if max(error.values())>1e-5:
            raise RuntimeError(f'Full-frame parity error: {error}')
        receipt={'sequence':seq,'center':idx,'max_abs_error_per_map':error,'pixels_per_map':512*640}
        (RUN/'PARITY.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(receipt,flush=True)
        return
    if not (RUN/'PARITY.json').is_file():
        raise RuntimeError('Run the development parity witness first')
    jobs=[(seq,idx) for seq in COHORT for idx in CENTERS][a.shard::a.workers]
    for seq,idx in jobs:
        path=RUN/'predictions'/f'{seq}_{idx:03d}.npz'
        if path.exists():
            continue
        began=time.perf_counter()
        save_npz(path,**score_frame(release,models,seq,idx))
        print(seq,idx,'all five dense maps',round(time.perf_counter()-began,2),'seconds',flush=True)


if __name__=='__main__':
    main()
