"""CPU-only checks of actual motion geometry, routing and learned-parameter loading.

The geometric oracle uses scalar NumPy bilinear samples of the normalized images
with separately specified frame lists. The optimizer witness uses a disposable
copy of each adapter; the only real data read are previously cached training points.
"""
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
STAGE = HERE.parent / "motion_joint_factorial"
sys.path.insert(0, str(STAGE))
from experiment import FIELDS, model_trainables
from model import FactorialDetector, load_official_detector
from motion_evidence import prepare_window, extract_evidence
from protocol import SEED, setup

TIMES = tuple(range(-8, 9))
DISCOVERY = (-7, -6, -4, -1, 2, 3, 5, 8)
CHECKS = ((-3, -2, 1, 4), (-5, -2, 1, 6), (-8, -3, 4, 7))
SPEEDS = [(x, y) for y in range(-2, 3) for x in range(-2, 3) if x or y]
DGRID = (-1., -.5, 0., .5, 1.)
CENTER = (35, 31)
H, W = 63, 71


def scene(speed, contrast):
    y, x = np.mgrid[:H, :W]
    frames = []
    for t in TIMES:
        cx = CENTER[0] + speed[0]*t
        cy = CENTER[1] + speed[1]*t
        squared_distance = (x-cx)**2 + (y-cy)**2
        # Mild, deterministic modulation keeps different temporal spans measurable.
        strength = contrast * (0.85 + 0.15*math.sin(0.7*t))
        frames.append(125. + strength*np.exp(-squared_distance/(2*1.25**2)))
    return torch.from_numpy(np.stack(frames).astype(np.float32))


def scalar_bilinear(im, x, y):
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = min(im.shape[1]-1,x0+1), min(im.shape[0]-1,y0+1)
    u, v = x-x0, y-y0
    return ((1-u)*(1-v)*float(im[y0,x0]) + u*(1-v)*float(im[y0,x1]) +
            (1-u)*v*float(im[y1,x0]) + u*v*float(im[y1,x1]))


def direct_proposal(window, point, times):
    x, y = point
    scores = []
    for vx,vy in SPEEDS:
        values = [window[t+8, y+t*vy, x+t*vx] for t in times
                  if 0 <= x+t*vx < W and 0 <= y+t*vy < H]
        scores.append(abs(np.sum(values,dtype=np.float64)) if 2*len(values)>=len(times) else -np.inf)
    return SPEEDS[np.argmax(scores)], any(np.isfinite(scores))


def geometry_oracle(window, points, selection, actual):
    shape = (len(points),2,2,4,5,5,3,3)
    assert tuple(actual['patches'].shape)==shape
    patches = np.zeros(shape,dtype=float)
    valid = np.zeros(shape[:-2],dtype=float)
    guesses = np.zeros((len(points),2,2,2),dtype=np.float32)
    frames = window.numpy()
    for i,(px,py) in enumerate(points):
        for direction,sign in enumerate((1,-1)):
            discovery = TIMES if selection=="full" else tuple(sign*t for t in DISCOVERY)
            proposal,possible = direct_proposal(frames,(px,py),discovery)
            guesses[i,direction,1] = proposal
            subsets = (TIMES, *(tuple(sign*t for t in ts) for ts in CHECKS))
            for anchor,(vx,vy) in enumerate(((0,0),proposal)):
                if anchor and not possible:
                    continue
                for iy,ix in itertools.product(range(5),repeat=2):
                    # Independent scalar one-pixel bilinear sampling of every 3x3 patch.
                    samples = {}
                    for t in TIMES:
                        cx,cy = px+t*(vx+DGRID[ix]), py+t*(vy+DGRID[iy])
                        if 1<=cx<=W-2 and 1<=cy<=H-2:
                            samples[t] = np.array([[scalar_bilinear(frames[t+8], cx+dx, cy+dy)
                                                    for dx in (-1,0,1)] for dy in (-1,0,1)])
                    for row,ts in enumerate(subsets):
                        kept = [samples[t] for t in ts if t in samples]
                        valid[i,direction,anchor,row,iy,ix] = len(kept)/len(ts)
                        if kept:
                            patches[i,direction,anchor,row,iy,ix] = np.mean(kept,axis=0)
    assert np.array_equal(guesses,actual['velocity'].numpy())
    err = float(np.abs(patches-actual['patches'].numpy()).max())
    frac_err = float(np.abs(valid-actual['valid'].numpy()).max())
    assert err<2e-5 and frac_err<1e-7, (err,frac_err)
    use = valid[:,0,0,0,2,2]>0
    assert np.array_equal(use,actual['usable'].numpy())
    return dict(max_abs_bilinear_patch_error=err,max_abs_valid_fraction_error=frac_err,
                compared_patch_values=patches.size, partial_support_cells=int(((valid>0)&(valid<1)).sum()))


def geometric_checks():
    rows=[]
    test_speeds=((2,-1),(-2,1),(-1,2),(1,-2))
    for speed,contrast in itertools.product(test_speeds, (90,-90)):
        window = prepare_window(scene(speed,contrast))
        for selection in ('full','split'):
            ev=extract_evidence(window,torch.tensor([CENTER]),selection)
            assert torch.all(ev['velocity'][0,:,1]==torch.tensor(speed)), (speed,contrast,selection)
            align=ev['patches'][0,0,1,0,:,:,1,1].numpy()*np.sign(contrast)
            center=float(align[2,2])
            off=[float(align[2,1]),float(align[3,2])]
            assert tuple(np.unravel_index(np.argmax(align),align.shape))==(2,2)
            assert center > max(off)
            assert np.min(ev['patches'][0,:,1,:,2,2,1,1].numpy()*np.sign(contrast))>0
            # One wrong velocity is enough to expose span-specific measurements.
            span_values=ev['patches'][0,0,1,1:,2,3,1,1].numpy()
            assert np.ptp(span_values)>.001
            rows.append(dict(speed=speed,contrast=contrast,selection=selection,signed_peak=center,
                             wrong_vx_vy_responses=off,span_values_at_vx_plus_half=span_values.tolist()))
    # Break near-border ties in the proposal itself using a reproducible background texture.
    gen=torch.Generator().manual_seed(SEED)
    image=scene((2,-1),90)+torch.randn((17,H,W),generator=gen)
    window=prepare_window(image)
    points=[CENTER,(1,2),(W-3,H-2),(0,0)]
    oracle={}
    for mode in ('full','split'):
        oracle[mode]=geometry_oracle(window,points,mode,extract_evidence(window,torch.tensor(points),mode))
        assert oracle[mode]['partial_support_cells']>0

    corpus = scene((2,-1),90)
    probes=torch.tensor([CENTER])
    initial=extract_evidence(prepare_window(corpus),probes,'split')
    prior=scene((-1,2),-80)
    split={}
    for k,sign in enumerate((1,-1)):
        keep=set(sign*t for t in DISCOVERY)
        changed=corpus.clone()
        for t in TIMES:
            if t not in keep:
                changed[t+8]=prior[t+8]
        transformed=prepare_window(changed)
        basewindow=prepare_window(corpus)
        assert torch.equal(transformed[[8+t for t in sorted(keep)]],basewindow[[8+t for t in sorted(keep)]])
        ev=extract_evidence(transformed,probes,'split')
        assert torch.equal(initial['velocity'][:,k],ev['velocity'][:,k])
        assert torch.all(ev['velocity'][0,1-k,1]==torch.tensor([-1,2]))
        val_delta=(initial['patches'][:,k,:,1:]-ev['patches'][:,k,:,1:]).abs().max().item()
        assert val_delta>0.1
        split[str(k)]=dict(unchanged_discovery_velocity=ev['velocity'][0,k,1].tolist(),
                           opposite_group_new_velocity=ev['velocity'][0,1-k,1].tolist(),validation_patch_delta=val_delta)
    changed=corpus.clone()
    changed[8]=prior[8]
    ev=extract_evidence(prepare_window(changed),probes,'split')
    assert torch.equal(initial['velocity'],ev['velocity'])
    assert torch.equal(initial['patches'][:,:,:,1:],ev['patches'][:,:,:,1:])
    whole_delta=(initial['patches'][:,:,:,0]-ev['patches'][:,:,:,0]).abs().max().item()
    assert whole_delta>0.01
    return dict(geometry_cases=rows,direct_numpy_oracle=oracle,
                intervention_by_discovery_direction=split,center_only_whole_patch_delta=whole_delta,
                center_only_validation_patch_delta=0., scope='Split velocity selection uses one independent raw group. Whole-clip rows and frozen 40-frame release features can use both groups.')


def param_group(name):
    return name.rsplit('.',1)[0]


def group_norm(params):
    sums={}
    for name,value in params.items():
        key=param_group(name)
        sums[key]=sums.get(key,0.)+float(value.detach().double().square().sum())
    return {key:np.sqrt(v).item() for key,v in sums.items()}


def take_snapshot(trainables):
    return {k:p.detach().clone() for k,p in trainables.items()}


def delta_norm(old,current):
    return group_norm({k:v-old[k] for k,v in current.items()})


def model_checks():
    release=load_official_detector('cpu')
    training_clip='DJI_0043_2_019.npz'
    native=np.load(STAGE/'runs/backbone'/training_clip)
    evidence=np.load(STAGE/'runs/train_motion'/training_clip)
    assert np.array_equal(native['xy'],evidence['xy'])
    target=native['target']
    n=min(48,int(target.sum()),int((~target).sum()))
    assert n>=8
    ids=np.r_[np.flatnonzero(target)[:n],np.flatnonzero(~target)[:n]]
    x=torch.from_numpy(native['features'][ids])
    y=torch.from_numpy(target[ids].astype('float32'))
    report={}
    for arm in 'ABCD':
        torch.manual_seed(SEED)
        net=FactorialDetector(release,arm).train()
        params=model_trainables(net)
        init=take_snapshot(params)
        opts=torch.optim.AdamW(params.values(),lr=1e-3,weight_decay=.01)
        opt_ids={id(p) for group in opts.param_groups for p in group['params']}
        assert opt_ids=={id(p) for p in params.values()}
        assert len(params)==10 and sum(p.numel() for p in params.values())==3450
        assert all(not p.requires_grad and id(p) not in opt_ids for p in release.parameters())
        assert len(net.encoder.visible.nonzero())==(100 if arm in 'BD' else 28)
        inputs={f:torch.from_numpy(evidence[f'{net.selection}_{f}'][ids]) for f in FIELDS}
        initial_hash=hashlib.sha256(np.concatenate([p.detach().flatten().numpy()
                                                   for p in params.values()]).tobytes()).hexdigest()
        stats=json.loads((STAGE/'runs/models'/f'{arm}.json').read_text())
        assert initial_hash==stats['initial_params_sha256']
        native_scores=net.base_points(x).detach()
        assert torch.equal(net.forward_points(x,inputs).detach(),native_scores)
        archived_error=float((native_scores-torch.from_numpy(native['base'][ids])).abs().max())
        assert archived_error<1e-4
        grads=[]
        for step in (1,2):
            opts.zero_grad(set_to_none=True)
            loss=F.binary_cross_entropy_with_logits(net.forward_points(x,inputs),y)
            loss.backward()
            assert all(p.grad is not None for p in params.values())
            assert all(p.grad is None for p in release.parameters())
            grad=group_norm({key:p.grad for key,p in params.items()})
            assert grad['projection']>0
            if step==1:
                assert all(v==0 for key,v in grad.items() if key!='projection')
            else:
                assert all(v>0 for v in grad.values()), grad
            before=take_snapshot(params)
            opts.step()
            grads.append(dict(step=step,group_gradient_l2=grad,optimizer_step_l2=delta_norm(before,params)))
        filename=STAGE/'runs/models'/f'{arm}.pt'
        saved=torch.load(filename,map_location='cpu',weights_only=True)
        assert saved.keys()==params.keys()
        diffs=delta_norm(init,saved)
        assert all(v>0 for v in diffs.values())
        with torch.no_grad():
            for key,p in params.items():
                p.copy_(saved[key])
            assert all(torch.equal(saved[key],p) for key,p in params.items())
            net.eval()
            original=net.forward_points(x,inputs)
            altered=dict(inputs)
            p=inputs['patches'].clone()
            v=inputs['valid'].clone()
            keep_center=p[:,:,:,1:,2,2].clone()
            keep_center_valid=v[:,:,:,1:,2,2].clone()
            p[:,:,:,1:]=0
            v[:,:,:,1:]=0
            p[:,:,:,1:,2,2]=keep_center
            v[:,:,:,1:,2,2]=keep_center_valid
            altered.update(patches=p,valid=v)
            masked=net.forward_points(x,altered)
            delta=float((original-masked).abs().max())
            assert (delta==0) if arm in 'AC' else (delta>0.001)
            restore_diff=float((original-native_scores).abs().max())
            assert restore_diff>0.001
        report[arm] = dict(checkpoint_sha256=hashlib.sha256(filename.read_bytes()).hexdigest(),
            initial_trainable_sha256=initial_hash,initial_native_logit_max_error=archived_error,
            num_trainable=sum(p.numel() for p in params.values()),visible_cells=int(net.encoder.visible.sum()),
            restored_delta_by_group_l2=diffs,restored_vs_release_max_abs_logit_delta=restore_diff,
            max_abs_logit_delta_clear_joint_cells=delta,disposable_steps=grads)
    return dict(training_frame=training_clip,balanced_points=2*n,device='cpu',
                optimizer_witness='Two cold-start steps on a fixed training-only batch in memory. First-step optimizer deltas can include AdamW decoupled weight decay even for zero task gradients.',
                arms=report)


if __name__=='__main__':
    setup()
    summary=dict(geometry=geometric_checks(),parameters=model_checks(),status='pass')
    output=HERE/'runs/geometry_flow.json'
    output.parent.mkdir(exist_ok=True)
    temp=output.with_suffix('.tmp')
    temp.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    temp.replace(output)
    print(output,flush=True)
