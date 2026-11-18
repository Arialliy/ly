"""Paired low-capacity readouts on identical frozen civil-traffic candidate features.

Run in the same immutable Docker image/overlay as motion_joint_factorial, with
only the present experiment's runs directory writable. Uses five train and two
development sources; performs no selection or training on confirmation videos.
"""
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.stats import rankdata
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent / "motion_joint_factorial"
sys.path.insert(0, str(ORIGINAL))
from cache_backbone import native_center
from evaluate import peaks
from experiment import motion_window
from model import load_official_detector
from motion_evidence import extract_evidence
from protocol import CENTERS, DEV, TRAIN, frame_path, frame_record, raw_frames, setup
from readout import froc, label_candidates

RUN = HERE / "runs"
TOP = 1024
SEED = 46
STEPS = 300
N_HALF = 64
LR = 0.003
L2 = 0.0003
BLOCK_SIZES = (42, 1120, 2880)
DESIGN = {
    "sources": {"train": TRAIN, "development": DEV, "centers": CENTERS},
    "top_peaks": TOP, "seed": SEED, "steps": STEPS, "class_count_per_batch": N_HALF,
    "optimizer": "Adam, learning rate 0.003, explicit L2 0.0003 * sum(square(weights)) / 2; biases exempt",
    "candidate_rule": "Source-blind and label-blind native release 5x5 local maxima with 2-pixel plateau tie suppression, first 1024 by descending logit in each frame, on train and dev; none reselected per probe.",
    "candidate_metric": "Ignore-only peaks removed. Positive candidate iff touches any eligible one-pixel expanded target box. Candidate AUC/AP permit multiple positives per box; FROC is independent maximum one-to-one box matching.",
    "features": "42 common fields (32 original TPro, release score, two directions x two anchors x two velocity components, one usability flag); 1120 common motion fields (2x2x28 cells x (all nine patch pixels + valid fraction)); 2880 extra fields (2x2x72x10). No reduction of local patches. Full split-rule fields computed once and shared. Standardize columns on nonignored training candidates with std floor .05 and divide each block by sqrt(width). C zeros the extra block after scaling.",
    "readouts": "Native training-z-standardized logit skip plus a regularized linear logit, or an 8-ReLU-hidden-unit residual MLP. Both use init=identical baseline logit, common initialized parameters and batches across the two information conditions. D-only input weights initially zero. Equal nominal input shape; effective number of accessible coefficients differs.",
    "decision": "Compare full vs marginal within each readout at FROC <=1 FP/frame, with <=.5 and <=5, per-source FROC, candidate AUC and AP for context. Inspect both readouts, no winner selection or tuning. This is a development probe restricted to fixed proposals, not a new test-set detector."
}


def json_write(name, obj):
    path = RUN / name
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def hash_array(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def feature_blocks(ev, features, base):
    n = len(base)
    values = torch.cat((ev["patches"].flatten(-2), ev["valid"].unsqueeze(-1)), dim=-1)
    # Flatten the subset/2d-velocity indices, preserving every spatial pixel.
    values = values.reshape(n, 2, 2, 100, 10)
    mask = torch.zeros((4,5,5), dtype=torch.bool, device=values.device)
    mask[0] = True
    mask[1:,2,2] = True
    mask = mask.reshape(100)
    common = torch.cat((features, base[:,None], ev["velocity"].flatten(1), ev["usable"].float()[:,None]), dim=1)
    blocks = (common, values[:,:,:,mask].reshape(n,-1), values[:,:,:,~mask].reshape(n,-1))
    assert tuple(b.shape[1] for b in blocks) == BLOCK_SIZES
    return torch.cat(blocks,dim=1)


@torch.inference_mode()
def prepare():
    detector = load_official_detector("cuda")
    all_x, points, reference, frames = [], [], [], []
    for seq in [*TRAIN,*DEV]:
        for idx in CENTERS:
            if seq in TRAIN:
                volume = raw_frames(seq)
                features, base = native_center(detector,volume[idx-19:idx+21])
            else:
                with np.load(frame_path("backbone",seq,idx)) as saved:
                    features,base = saved["features"],saved["base"]
            xy,values=peaks(base)
            xy,values=xy[:TOP],values[:TOP]
            assert len(xy)==TOP and np.all(values[:-1]>=values[1:])
            y,x=xy[:,1],xy[:,0]
            window=motion_window(seq,idx)
            ev=extract_evidence(window,torch.from_numpy(xy).cuda(),"split")
            matrix=feature_blocks(ev,torch.from_numpy(features[y,x]).cuda(),torch.from_numpy(values).cuda())
            all_x.append(matrix.cpu().numpy())
            if seq in DEV:
                with np.load(frame_path("predictions",seq,idx)) as saved:
                    assert np.array_equal(values, saved["original"][y,x])
                    reference.append({arm:saved[arm][y,x] for arm in ("original","C","D")})
            frames.append(frame_record(seq,idx))
            points.append(xy)
        print("processed",seq,flush=True)
    return np.concatenate(all_x),np.concatenate(points),frames,reference


class Readout(nn.Module):
    def __init__(self, kind):
        super().__init__()
        width=sum(BLOCK_SIZES)
        common=sum(BLOCK_SIZES[:2])
        if kind=="linear":
            self.net=nn.Linear(width,1)
            nn.init.zeros_(self.net.weight)
            nn.init.zeros_(self.net.bias)
        else:
            self.net=nn.Sequential(nn.Linear(width,8),nn.ReLU(),nn.Linear(8,1))
            with torch.no_grad():
                self.net[0].weight[:,:common].uniform_(-0.3,0.3)
                self.net[0].weight[:,common:].zero_()
                self.net[0].bias.zero_()
                self.net[2].weight.zero_()
                self.net[2].bias.zero_()

    def forward(self, x, skip):
        return self.net(x).flatten()+skip


def scores_for_arm(kind,x,skip,y,batches,allow_joint):
    torch.manual_seed(SEED)
    model=Readout(kind).cuda()
    opt=torch.optim.Adam(model.parameters(),lr=LR)
    if not allow_joint:
        x=x.clone()
        x[:,sum(BLOCK_SIZES[:2]):]=0
    with torch.no_grad():
        assert torch.equal(model(x[:512],skip[:512]),skip[:512])
    for ids in batches:
        logits=model(x[ids],skip[ids])
        loss=F.binary_cross_entropy_with_logits(logits,y[ids])
        reg=sum(w.square().sum() for w in model.parameters() if w.ndim>1)
        opt.zero_grad(set_to_none=True)
        (loss+L2/2*reg).backward()
        opt.step()
    with torch.no_grad():
        result=model(x,skip).cpu().numpy()
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite readout scores")
    return result, sum(w.numel() for w in model.parameters())


def ranking(scores,y):
    y=y.astype(bool)
    p=int(y.sum())
    n=len(y)-p
    if not n or not p:
        return dict(pos=p,neg=n,auc=None,ap=None)
    auc=(rankdata(scores)[y].sum()-p*(p+1)/2)/(p*n)
    order=np.argsort(-scores,kind="stable")
    ranked=y[order]
    ends=np.flatnonzero(np.r_[scores[order][1:]!=scores[order][:-1],True])
    hit=np.cumsum(ranked)[ends]
    ap=np.sum(np.diff(np.r_[0,hit])*(hit/(ends+1)))/p
    return dict(pos=p,neg=n,auc=float(auc),ap=float(ap))


def main():
    setup()
    RUN.mkdir(exist_ok=True)
    plan=RUN/"probe_readout_plan.json"
    if plan.exists():
        assert json.loads(plan.read_text())==json.loads(json.dumps(DESIGN))
    else:
        json_write(plan.name,DESIGN)
    raw,xy,frames,reference=prepare()
    frame_ids=np.repeat(np.arange(len(frames)),TOP)
    labels,neighbors=label_candidates(xy,frame_ids,frames)
    training=np.array([frames[i]["sequence"] in TRAIN for i in frame_ids])
    dev=~training
    fit=training & (labels>=0)
    eligible=torch.from_numpy(fit).cuda()
    tensor=torch.from_numpy(raw).cuda()
    mean=tensor[eligible].mean(0)
    std=tensor[eligible].std(0,unbiased=False).clamp_min(.05)
    xs=(tensor-mean)/std
    skip=xs[:,32].clone()
    first=0
    for width in BLOCK_SIZES:
        xs[:,first:first+width]/=math.sqrt(width)
        first+=width
    rng=np.random.default_rng(SEED)
    pos,neg=[np.flatnonzero(training & (labels==k)) for k in (1,0)]
    batches=np.stack([np.r_[rng.choice(pos,N_HALF),rng.choice(neg,N_HALF)] for _ in range(STEPS)])
    batch_gpu=torch.from_numpy(batches).cuda()
    y=torch.from_numpy(labels.astype(np.float32)).cuda()
    dev_frames=[r for r in frames if r["sequence"] in DEV]
    dev_frame=np.repeat(np.arange(len(dev_frames)),TOP)
    xy_dev=xy[dev]
    yd,gd=label_candidates(xy_dev,dev_frame,dev_frames)
    assert np.array_equal(yd,labels[dev])
    subsets={"all_dev":list(range(len(dev_frames)))}
    for seq in DEV:
        subsets[seq]=[i for i,r in enumerate(dev_frames) if r["sequence"]==seq]
    def evaluate(coefs):
        result={}
        for src,ids in subsets.items():
            good=np.isin(dev_frame,ids) & (yd>=0)
            a=froc(coefs,ids,dev_frames,dev_frame,yd,gd)
            a["candidate_ranking"]=ranking(coefs[good],yd[good])
            result[src]=a
        return result
    methods={f"frozen_{m}":evaluate(np.concatenate([r[m] for r in reference])) for m in ("original","C","D")}
    capacities={}
    output_scores={k:np.concatenate([r[k] for r in reference]) for k in ("original","C","D")}
    for kind in ("linear","mlp8"):
        for rep in ("C","D"):
            s,n=scores_for_arm(kind,xs,skip,y,batch_gpu,rep=="D")
            key=f"{kind}_{rep}"
            methods[key]=evaluate(s[dev])
            capacities[key]=n
            output_scores[key]=s[dev]
    coverage={s:methods["frozen_original"][s]["candidate_oracle_tp"] for s in subsets}
    for rows in methods.values():
        assert all(rows[s]["candidate_oracle_tp"]==coverage[s] for s in subsets)
    differences={}
    for k in ("linear","mlp8","frozen"):
        c,d=methods[k+"_C"],methods[k+"_D"]
        differences[k]={src:dict(delta_tp={b:d[src]["froc"][b]["tp"]-c[src]["froc"][b]["tp"] for b in ("0.5","1","5")},
                                 delta_auc=d[src]["candidate_ranking"]["auc"]-c[src]["candidate_ranking"]["auc"],
                                 delta_ap=d[src]["candidate_ranking"]["ap"]-c[src]["candidate_ranking"]["ap"]) for src in subsets}
    output=dict(design=DESIGN,eligible_train={"positive":len(pos),"negative":len(neg),"ignored":int((training & (labels<0)).sum())},
                locations_sha256=hash_array(xy),paired_batches_sha256=hash_array(batches),
                train_only_standardizer_sha256=hash_array(np.stack((mean.cpu().numpy(),std.cpu().numpy()))),
                train_center_count=int(np.sum([r["sequence"] in TRAIN for r in frames])),
                dev_centers=[{"sequence":r["sequence"],"index":r["index"]} for r in dev_frames],
                nominal_readout_parameters=capacities,methods=methods,contrasts=differences)
    np.savez_compressed(RUN/"probe_readout_dev_scores.npz",xy=xy_dev,frame=dev_frame,target=yd,**output_scores)
    json_write("probe_readout.json",output)
    for key,rows in methods.items():
        print(key,[(s,[(b,v["tp"],v["fp"]) for b,v in r["froc"].items()],r["candidate_ranking"]) for s,r in rows.items()],flush=True)

if __name__=="__main__":
    main()
