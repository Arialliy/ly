"""Sensitivity of frozen C and D to the assignment of off-center rows, on development only."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PREVIOUS=HERE.parent/'motion_joint_factorial'
sys.path.insert(0,str(PREVIOUS))
from experiment import model_trainables, motion_window
from model import FactorialDetector, load_official_detector
from motion_evidence import _grids, extract_evidence
from protocol import CENTERS, DEV, frame_path, setup
from evaluate import peaks, evaluate_method


def example():
    times, _, _, _, _, masks = _grids(torch.device('cpu'))
    stable=torch.tensor([4.,8.,4.]).repeat(17,1)
    changing=stable.clone()
    changing[times<0]=torch.tensor([7.,8.,1.])
    changing[times>0]=torch.tensor([1.,8.,7.])
    weights=masks / masks.sum(-1,keepdim=True)
    pooled=torch.einsum('dst,tk->dsk',weights,changing)
    control=torch.einsum('dst,tk->dsk',weights,stable)
    assert torch.allclose(pooled,control,atol=1e-6,rtol=0)
    return {'before':changing[times<0].mean(0).tolist(),'after':changing[times>0].mean(0).tolist(),
            'subsets_each_direction':pooled.tolist(),'against_stable_max_error':(pooled-control).abs().max().item()}


@torch.inference_mode()
def run():
    from protocol import frame_record
    setup()
    release=load_official_detector('cuda')
    models={}
    for name in ('C','D'):
        model=FactorialDetector(release,name).eval()
        state=torch.load(PREVIOUS/'runs/models'/f'{name}.pt',weights_only=True,map_location='cpu')
        params=model_trainables(model)
        assert set(state)==set(params)
        for key,p in params.items():
            p.copy_(state[key])
        models[name]=model
    observed={m:[] for m in ('reference_D','swapped_D')}
    frames=[]
    differences=[]
    rows=[]
    for seq in DEV:
        for t in CENTERS:
            with np.load(frame_path('backbone',seq,t)) as data, np.load(frame_path('predictions',seq,t)) as archive:
                x=torch.from_numpy(data['features']).cuda().reshape(-1,32)
                original={m:archive[m] for m in models}
            window=motion_window(seq,t)
            h,w=original['D'].shape
            # Outside this margin the previous run's outputs are left intact to keep border conditions fixed.
            margin=25
            yy,xx=torch.meshgrid(torch.arange(margin,h-margin,device='cuda'),torch.arange(margin,w-margin,device='cuda'),indexing='ij')
            pts=torch.stack((xx.flatten(),yy.flatten()),dim=1)
            swapped_D=original['D'].copy()
            delta=[]
            max_c=0.
            max_parity=0.
            for xy in pts.split(1024):
                ix=xy[:,1]*w+xy[:,0]
                ev=extract_evidence(window,xy,'split')
                assert ev['valid'].min().item()==1.
                pert=dict(ev)
                patches=ev['patches'].clone()
                # Exchange the short and long span at neighboring velocities only.
                # Whole-clip maps, fixed-velocity per-span patches, anchors and sampling masks stay fixed.
                for to_row,from_row in [(1,3),(3,1)]:
                    patches[:,:,:,to_row] = ev['patches'][:,:,:,from_row]
                    patches[:,:,:,to_row,2,2]=ev['patches'][:,:,:,to_row,2,2]
                pert['patches']=patches
                points=x[ix]
                coords=xy.cpu().numpy()
                for name,model in models.items():
                    base=model.forward_points(points,ev)
                    modified=model.forward_points(points,pert)
                    ref=original[name][coords[:,1],coords[:,0]]
                    max_parity=max(max_parity,np.abs(ref-base.cpu().numpy()).max().item())
                    if name=='C':
                        max_c=max(max_c,(modified-base).abs().max().item())
                    else:
                        v=modified.cpu().numpy()
                        delta.append((modified-base).abs().cpu().numpy())
                        swapped_D[coords[:,1],coords[:,0]]=v
            if max_parity>1e-4 or max_c!=0.:
                raise ValueError((seq,t,max_c,max_parity))
            d=np.concatenate(delta)
            rows.append(dict(seq=seq,t=t,n=len(d),max_c_change=max_c,parity_error=max_parity,
                             median_d_change=float(np.median(d)),frac_d_over_1e_minus_4=float(np.mean(d>1e-4))))
            differences.append(d)
            observed['reference_D'].append(peaks(original['D']))
            observed['swapped_D'].append(peaks(swapped_D))
            frames.append(frame_record(seq,t))
            print(seq,t,rows[-1],flush=True)
    groups={'all_dev':list(range(len(frames)))}
    counts={m:evaluate_method(frames,peaks_per_frame,groups)['all_dev']['froc'] for m,peaks_per_frame in observed.items()}
    d=np.concatenate(differences)
    return dict(scope='two earlier civil traffic development sources only; frozen trained C/D, no tuning, no training',
                intervention='exchange short and long span off-center joint patches at all full-support interior locations, fixed boundaries',
                example=example(),rows=rows,n_pixels=len(d),fraction_d_abs_logit_change_gt_1e_minus_4=float((d>1e-4).mean()),
                median_d_abs_logit_change=float(np.median(d)),p90_d_abs_logit_change=float(np.quantile(d,.9)),
                counts=counts)


if __name__=='__main__':
    result=run()
    path=HERE/'runs/probe.json'
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)
