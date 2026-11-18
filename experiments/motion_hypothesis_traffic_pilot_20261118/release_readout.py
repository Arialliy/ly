"""Paired traffic-candidate rescoring on frozen author DeepPro-Plus release predictions."""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from extract import ROOT, RUN, signal
from readout import froc, label_candidates

from training_settings import TRAIN_SEED, reset_training_rng

STEPS = 300


def standardized(arr, train):
    arr = arr.astype('float32')
    center = arr[train].mean(0, dtype='float64')
    scale = np.maximum(arr[train].std(0, dtype='float64'), 1e-5)
    return (np.clip((arr-center)/scale, -8,8)/np.sqrt(arr.shape[1])).astype('float32')


def center_features(frames, frame, xy, base):
    static = np.empty((len(xy),3),dtype='float32')
    for f,record in enumerate(frames):
        native=ROOT/'data/stage1'/record['sequence']/'raw_frames.npy'
        image=np.load(native,mmap_mode='r')[record['index']]
        z,contrast=signal(image)
        idx=np.where(frame==f)[0]
        x,y=xy[idx].T
        static[idx]=np.column_stack((base[idx],z[y,x],contrast[y,x]))
    return static


def fit_residual(features, base, inside, train, labels):
    rng = reset_training_rng()
    module = nn.Sequential(nn.Linear(features.shape[1],16),nn.ReLU(),nn.Linear(16,1))
    nn.init.zeros_(module[-1].weight)
    nn.init.zeros_(module[-1].bias)
    x=torch.from_numpy(features)
    frozen=torch.as_tensor(base,dtype=torch.float32)
    valid=torch.as_tensor(inside,dtype=torch.float32)
    y=torch.as_tensor(labels,dtype=torch.float32)
    pos=np.flatnonzero(train & inside & (labels==1))
    neg=np.flatnonzero(train & inside & (labels==0))
    assert len(pos)>0 and len(neg)>0
    with torch.no_grad():
        assert torch.equal(frozen, frozen+valid*module(x)[:,0])
    opt=torch.optim.AdamW(module.parameters(),lr=.001,weight_decay=.01)
    loss=nn.BCEWithLogitsLoss()
    for _ in range(STEPS):
        ids=np.r_[rng.choice(pos,128),rng.choice(neg,128)]
        logits=frozen[ids]+module(x[ids])[:,0]
        opt.zero_grad(set_to_none=True)
        loss(logits,y[ids]).backward()
        opt.step()
    with torch.no_grad():
        return (frozen+valid*module(x)[:,0]).numpy()


def main():
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    frames=json.loads((RUN/'release_labels.json').read_text())['frames']
    with np.load(RUN/'release_sample.npz',allow_pickle=False) as rec:
        xy,frame,base=(rec[k] for k in ('xy','frame','base'))
        maps=rec['maps'] if 'maps' in rec.files else np.load(RUN/'release_motion_maps.npy')
    base=base.reshape(-1).astype('float32')
    xy=xy.astype(int)
    assert maps.shape==(len(xy),2,2,3,5,5)
    assert np.isfinite(base).all() and np.isfinite(maps).all()
    for f in frames:
        assert f['native_start'] <= f['index']-8 <= f['index']+8 < f['native_start']+40
    labels,neighbors=label_candidates(xy,frame,frames)
    train=np.array([frames[i]['split']=='train' for i in frame]) & (labels>=0)
    inside=(xy[:,0]>=25)&(xy[:,0]<640-25)&(xy[:,1]>=25)&(xy[:,1]<512-25)
    dev=[i for i,f in enumerate(frames) if f['split']=='calibration']
    groups={'all_dev':dev}
    for seq in sorted({frames[i]['sequence'] for i in dev}):
        groups[seq]=[i for i in dev if frames[i]['sequence']==seq]
    def evaluate(scores):
        return {k:froc(scores,ids,frames,frame,labels,neighbors) for k,ids in groups.items()}
    out={'budget_per_frame':512,'seed':TRAIN_SEED,'train_steps':STEPS,'readout':'frozen original logit plus zero-initialized final projection, 16 hidden ReLU, 256 balanced batch, AdamW .001 wd .01, fixed CPU settings. Every learned arm sees identical labeled candidates and keeps off-margin logits exactly.',
         'train_positive_interior_candidates':int(np.sum(train&inside&(labels==1))),
         'train_negative_interior_candidates':int(np.sum(train&inside&(labels==0))),
         'train_centers':sum(f['split']=='train' for f in frames),
         'dev_centers':len(dev),'original':evaluate(base),'methods':{}}
    static=standardized(center_features(frames,frame,xy,base),train&inside)
    for name in ('static','pooled','separate_groups_spans'):
        if name=='static':
            extra=np.zeros((len(maps),300),dtype='float32')
        else:
            values=maps if name=='separate_groups_spans' else np.broadcast_to(maps.mean(axis=(2,3),keepdims=True),maps.shape)
            extra=standardized(values.reshape(len(maps),-1),train&inside)
        feats=np.column_stack((static,extra)).astype('float32')
        scores=fit_residual(feats,base,inside,train,labels)
        assert np.array_equal(scores[~inside],base[~inside])
        out['methods'][name]={str(TRAIN_SEED):evaluate(scores)}
    output_dir=RUN/f'seed{TRAIN_SEED}'
    output_dir.mkdir(exist_ok=True)
    path=output_dir/'release_scores.json'
    path.write_text(json.dumps(out,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    for name,row in [('original',{'release':out['original']}),*out['methods'].items()]:
        print(name,{seed:[(v['tp'],v['fp']) for v in groups['all_dev']['froc'].values()] for seed,groups in row.items()})


if __name__=='__main__':
    main()
