"""Fit all four arms with paired batches and score every pixel of development frames."""
import argparse
import hashlib
import json
import time

import numpy as np
import torch
from torch.nn import functional as F

from model import FactorialDetector, load_official_detector
from motion_evidence import extract_evidence, prepare_window
from protocol import (ARMS, BATCH, CENTERS, DEV, RUN, SEED, STEPS, TRAIN, frame_path,
                      raw_frames, save_npz, setup, workers_frames)

FIELDS = ('patches', 'valid', 'velocity', 'usable')


def motion_window(seq, idx):
    raw = raw_frames(seq)[idx-8:idx+9].copy()
    return prepare_window(torch.from_numpy(raw).cuda())


@torch.no_grad()
def cache_training(shard, workers, chunk):
    for seq,idx in workers_frames(TRAIN,shard,workers):
        dest = frame_path('train_motion',seq,idx)
        if dest.exists():
            continue
        started=time.perf_counter()
        with np.load(frame_path('backbone',seq,idx)) as r:
            xy = torch.as_tensor(r['xy'],device='cuda')
        window = motion_window(seq,idx)
        arrays = {'xy':xy.cpu().numpy()}
        for selection in ('full','split'):
            chunks = [extract_evidence(window,points,selection) for points in xy.split(chunk)]
            for key in FIELDS:
                arrays[f'{selection}_{key}'] = torch.cat([part[key] for part in chunks]).cpu().numpy()
        save_npz(dest,**arrays)
        print(seq,idx,'training motion seconds',round(time.perf_counter()-started,2),flush=True)


def train_data(selection):
    buffers = {k:[] for k in ('features','target','base',*FIELDS)}
    for seq in TRAIN:
        for idx in CENTERS:
            with np.load(frame_path('backbone',seq,idx)) as b, np.load(frame_path('train_motion',seq,idx)) as e:
                if not np.array_equal(b['xy'],e['xy']):
                    raise ValueError('Motion and release feature cache pixel order differ')
                for name in ('features','target','base'):
                    buffers[name].append(b[name])
                for name in FIELDS:
                    buffers[name].append(e[f'{selection}_{name}'])
    return {name:torch.from_numpy(np.concatenate(values)).cuda() for name,values in buffers.items()}


def digest(arr):
    return hashlib.sha256(arr.tobytes()).hexdigest()


def model_trainables(model):
    return {k:p for k,p in model.named_parameters() if p.requires_grad}


def fit(arms):
    release = load_official_detector('cuda')
    for selection, letters in (('full','AB'),('split','CD')):
        if not set(arms).intersection(letters):
            continue
        buf = train_data(selection)
        target = buf['target'].cpu().numpy()
        rng = np.random.default_rng(SEED)
        pos,neg = np.flatnonzero(target),np.flatnonzero(~target)
        batches = np.stack([np.r_[rng.choice(pos,BATCH//2),rng.choice(neg,BATCH//2)]
                            for _ in range(STEPS)])
        batch_ids = torch.from_numpy(batches).cuda()
        for arm in letters:
            if arm not in arms:
                continue
            torch.manual_seed(SEED)
            m = FactorialDetector(release,arm).train()
            params=model_trainables(m)
            init_sha=digest(np.concatenate([p.detach().flatten().cpu().numpy() for p in params.values()]))
            opt = torch.optim.AdamW(params.values(),lr=1e-3,weight_decay=.01)
            with torch.no_grad():
                ids = batch_ids[0]
                evidence = {k:buf[k][ids] for k in FIELDS}
                start_delta = (m.forward_points(buf['features'][ids],evidence)-buf['base'][ids]).abs().max().item()
                if start_delta > 1e-4:
                    raise RuntimeError(f'Native and initialized fused scores differ by {start_delta}')
            losses = []
            torch.cuda.synchronize()
            started = time.perf_counter()
            for step,ids in enumerate(batch_ids):
                evidence = {k:buf[k][ids] for k in FIELDS}
                logits = m.forward_points(buf['features'][ids],evidence)
                loss = F.binary_cross_entropy_with_logits(logits,buf['target'][ids].float())
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                if step in (0,99,199,STEPS-1):
                    losses.append(float(loss.detach()))
            torch.cuda.synchronize()
            if not np.isfinite(losses).all() or not all(torch.isfinite(v).all().item() for v in params.values()):
                raise FloatingPointError('Nonfinite updated weights or loss')
            dest=RUN/'models'
            dest.mkdir(exist_ok=True,parents=True)
            temp=dest/f'{arm}.partial.pt'
            torch.save({k:p.detach().cpu() for k,p in params.items()},temp)
            temp.replace(dest/f'{arm}.pt')
            stats=dict(arm=arm,selection=selection,seed=SEED,updates=STEPS,balanced_batch=BATCH,
                       train_pos=len(pos),train_neg=len(neg),trainable_params=sum(p.numel() for p in params.values()),
                       batch_sha256=digest(batches),initial_params_sha256=init_sha,
                       initial_release_max_error=start_delta,bce_steps_1_100_200_300=losses,
                       train_seconds=time.perf_counter()-started, device=torch.cuda.get_device_name())
            (dest/f'{arm}.json').write_text(json.dumps(stats,indent=2)+'\n')
            print(json.dumps(stats),flush=True)


@torch.inference_mode()
def predict(shard,workers,chunk):
    release=load_official_detector('cuda')
    models={}
    for arm in ARMS:
        m=FactorialDetector(release,arm).eval()
        weights=torch.load(RUN/'models'/f'{arm}.pt',weights_only=True,map_location='cpu')
        if weights.keys() != model_trainables(m).keys():
            raise ValueError('Unexpected adapter checkpoint keys')
        with torch.no_grad():
            for key,p in model_trainables(m).items():
                p.copy_(weights[key])
        models[arm]=m
    for seq,idx in workers_frames(DEV,shard,workers):
        dest=frame_path('predictions',seq,idx)
        if dest.exists():
            continue
        window=motion_window(seq,idx)
        with np.load(frame_path('backbone',seq,idx)) as r:
            base=r['base']
            h,w=base.shape
            features=torch.from_numpy(r['features']).cuda().view(-1,32)
        outs={a:torch.empty(h*w,device='cuda') for a in ARMS}
        id_error=0.
        torch.cuda.synchronize()
        started=time.perf_counter()
        for first in range(0,h*w,chunk):
            last=min(first+chunk,h*w)
            ids=torch.arange(first,last,device='cuda')
            xy=torch.stack((ids%w,ids//w),dim=-1)
            feats=features[first:last]
            for selection,letters in (('full','AB'),('split','CD')):
                ev=extract_evidence(window,xy,selection)
                for arm in letters:
                    outs[arm][first:last]=models[arm].forward_points(feats,ev)
            if first == 0:
                expected=torch.as_tensor(base.reshape(-1)[:last],device='cuda')
                id_error=(models['A'].base_points(feats)-expected).abs().max().item()
                if id_error>1e-4:
                    raise RuntimeError('Cached feature and official output are misaligned')
        arrays={arm:out.reshape(h,w).cpu().numpy() for arm,out in outs.items()}
        if not all(np.isfinite(x).all() for x in arrays.values()):
            raise FloatingPointError('Nonfinite dense logits')
        save_npz(dest,original=base,**arrays)
        print(seq,idx,'dense pixels',h*w,'joint four-arm seconds',round(time.perf_counter()-started,2),
              'base max error',id_error,'peak GPU MiB',round(torch.cuda.max_memory_allocated()/2**20),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('task',choices=('cache','train','predict'))
    parser.add_argument('--arms',choices=('AB','CD','ABCD','A','B','C','D'),default='ABCD')
    parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--chunk',type=int,default=512)
    args=parser.parse_args()
    if not 0 <= args.shard < args.workers or args.chunk<1:
        parser.error('invalid worker index or chunk size')
    setup()
    if args.task == 'train':
        fit(args.arms)
    elif args.task == 'cache':
        cache_training(args.shard,args.workers,args.chunk)
    else:
        predict(args.shard,args.workers,args.chunk)


if __name__ == '__main__':
    main()
