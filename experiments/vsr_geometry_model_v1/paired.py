"""Fixed-budget paired study of the two correction heads, with a frozen DUF baseline."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import numpy as np
import torch

from model import PairedHead, build_evidence


ROOT=Path(__file__).resolve().parent
CACHE=ROOT/"cache"
SEEDS=(46,137,829)
BATCH=6
EPOCHS=15
SIZE=slice(32,160)
MODES=("B","C")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path,data):
    tmp=path.with_suffix(".pending")
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")
    tmp.replace(path)


def state_digest(head):
    h=hashlib.sha256()
    for name,p in head.state_dict().items():
        h.update(name.encode())
        h.update(p.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def load_split(manifest,split,device):
    records=sorted((r for r in manifest["files"] if r["split"]==split),key=lambda r:r["path"])
    expected=96 if split=="train" else 32
    if len(records)!=expected:
        raise ValueError(f"Unexpected number of {split} records")
    collected={}
    for start in range(0,len(records),4):
        arrays={}
        for record in records[start:start+4]:
            path=CACHE/record["path"]
            if sha(path)!=record["sha256"]:
                raise ValueError(f"Modified input: {path}")
            with np.load(path,allow_pickle=False) as inp:
                for key in ("lr","hr","base","forward","backward"):
                    arrays.setdefault(key,[]).append(inp[key])
        t={k:torch.from_numpy(np.stack(v)).to(device=device,dtype=torch.float32) for k,v in arrays.items()}
        ev=build_evidence(t["lr"],t["base"],t["forward"],t["backward"])
        for key,val in {**ev,"base":t["base"],"hr":t["hr"]}.items():
            if not val.isfinite().all():
                raise ValueError(f"Invalid input tensor {key} in {split} at {start}")
            collected.setdefault(key,[]).append(val.detach())
    # Drop each list as it is concatenated to reduce peak memory.
    bank={}
    for key in list(collected):
        bank[key]=torch.cat(collected.pop(key),dim=0)
    return records,bank


def take(bank,indices):
    subset={k:t.index_select(0,indices) for k,t in bank.items()}
    base=subset.pop("base")
    gt=subset.pop("hr")
    return subset,base,gt[...,SIZE,SIZE]


def train_seed(seed,bank,out,device):
    torch.manual_seed(seed)
    b,c=(PairedHead(mode).to(device) for mode in MODES)
    c.load_state_dict(b.state_dict(),strict=True)
    models=dict(B=b,C=c)
    optimizers={k:torch.optim.Adam(m.parameters(),lr=.001,eps=1e-8,foreach=False) for k,m in models.items()}
    initial={k:state_digest(m) for k,m in models.items()}
    assert initial["B"]==initial["C"]
    count=sum(p.numel() for p in b.parameters())
    assert count==sum(p.numel() for p in c.parameters())
    rng=np.random.default_rng(seed)
    history=[]
    batches=[]
    trace=[]
    for epoch in range(EPOCHS):
        permutation=rng.permutation(96).reshape(-1,BATCH)
        for batch in permutation:
            idx=torch.as_tensor(batch,dtype=torch.long,device=device)
            e,base,gt=take(bank,idx)
            pair=[]
            for mode in MODES:
                net,opt=models[mode],optimizers[mode]
                opt.zero_grad(set_to_none=True)
                prediction=net(e,base)
                if not batches and not torch.equal(prediction,base):
                    raise AssertionError("Initial residual head must exactly reproduce frozen A")
                error=prediction[...,SIZE,SIZE]-gt
                loss=(error.square()+.001**2).sqrt().mean()
                loss.backward()
                opt.step()
                pair.append(loss.detach())
            trace.append(torch.stack(pair))
            batches.append(batch.tolist())
        trace_cpu=torch.stack(trace).cpu().numpy()
        if not np.isfinite(trace_cpu).all():
            raise ValueError(f"Nonfinite loss at seed {seed} epoch {epoch+1}")
        history.extend(trace_cpu.tolist())
        trace=[]
        print(f"seed={seed} epoch={epoch+1:02} B={trace_cpu[:,0].mean():.6f} C={trace_cpu[:,1].mean():.6f}",flush=True)
    assert len(batches)==len(history)==240
    final={k:state_digest(m) for k,m in models.items()}
    ckpt={k:{name:t.detach().cpu() for name,t in m.state_dict().items()} for k,m in models.items()}
    checkpoint=out/f"seed_{seed}.pt"
    pending=checkpoint.with_suffix(".pending")
    torch.save(ckpt,pending)
    pending.replace(checkpoint)
    save_json(out/f"seed_{seed}.json",dict(seed=seed,updates=240,parameters_per_head=count,
              initialization_sha256=initial,final_sha256=final,weights_sha256=sha(checkpoint),
              batches=batches,batches_sha256=hashlib.sha256(np.array(batches,dtype="<i4").tobytes()).hexdigest(),
              loss_names=MODES,loss_by_update=history))


@torch.no_grad()
def scores(seed,records,bank,device,out,split):
    states=torch.load(out/f"seed_{seed}.pt",map_location="cpu",weights_only=True)
    heads={mode:PairedHead(mode).to(device).eval() for mode in MODES}
    for mode in MODES:
        heads[mode].load_state_dict(states[mode],strict=True)
    results=[]
    for start in range(0,len(records),4):
        idx=torch.arange(start,min(start+4,len(records)),device=device)
        evidence,base,gt=take(bank,idx)
        predictions={"A":base,**{k:m(evidence,base) for k,m in heads.items()}}
        errors={k:(y[...,SIZE,SIZE].double()-gt.double()).square().mean(dim=(1,2,3)).cpu().tolist()
                for k,y in predictions.items()}
        for j,r in enumerate(records[start:start+4]):
            row=dict(seed=seed,split=split,sequence=r["sequence"],center=r["center"],patch=r["patch"])
            row.update({f"mse_{k}":v[j] for k,v in errors.items()})
            if not all(math.isfinite(row[f"mse_{k}"]) and row[f"mse_{k}"]>0 for k in predictions):
                raise ValueError("Invalid MSE")
            results.append(row)
    return results


def main(out):
    if not torch.cuda.is_available():
        raise RuntimeError("This cache-optimized paired experiment expects one GPU")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    device=torch.device("cuda")
    out.mkdir(parents=True,exist_ok=True)
    config=dict(seeds=SEEDS,epochs=EPOCHS,batch=BATCH,optimizer="Adam(lr=.001,eps=1e-8)",
                loss="Charbonnier eps=.001; uncropped predictions supervised at HR[32:160,32:160]",
                backend=str(torch.__version__),gpu=torch.cuda.get_device_name(),
                source={f:sha(ROOT/f) for f in ("paired.py","model.py","geometry.py","transport.py","cache/MANIFEST.json")})
    cfg=out/"CONFIG.json"
    if cfg.exists() and json.loads(cfg.read_text())!=json.loads(json.dumps(config)):
        raise ValueError("Existing run uses another configuration")
    save_json(cfg,config)
    if (out/"COMPLETE.json").exists():
        raise FileExistsError("Run already completed")
    manifest=json.loads((CACHE/"MANIFEST.json").read_text())
    train_records,train=load_split(manifest,"train",device)
    for seed in SEEDS:
        done=out/f"seed_{seed}.json"
        if done.exists():
            summary=json.loads(done.read_text())
            if summary["weights_sha256"]!=sha(out/f"seed_{seed}.pt") or summary["updates"]!=240:
                raise ValueError(f"Incomplete seed {seed}")
        else:
            train_seed(seed,train,out,device)
    rows=[]
    for seed in SEEDS:
        rows.extend(scores(seed,train_records,train,device,out,"train"))
    del train
    torch.cuda.empty_cache()
    records,bank=load_split(manifest,"evaluation",device)
    for seed in SEEDS:
        rows.extend(scores(seed,records,bank,device,out,"evaluation"))
    patch_csv=out/"patch_metrics.csv"
    with patch_csv.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    groups={}
    for row in rows:
        groups.setdefault((row["split"],row["seed"],row["sequence"]),[]).append(row)
    seq_rows=[]
    for (split,seed,seq),group in sorted(groups.items()):
        seq_rows.append(dict(split=split,seed=seed,sequence=seq,
                     **{f"psnr_{k}":-10*math.log10(sum(r[f"mse_{k}"] for r in group)/len(group)) for k in ("A","B","C")}))
    save_json(out/"sequence_metrics.json",seq_rows)
    assert Counter(r["split"] for r in rows)==dict(train=288,evaluation=96)
    save_json(out/"COMPLETE.json",dict(status="COMPLETE",end_unix=time.time(),seeds=SEEDS,
              updates_per_head_per_seed=240,patches_per_seed=dict(train=96,evaluation=32),
              peak_cuda_bytes=torch.cuda.max_memory_allocated(),patch_metrics_sha256=sha(patch_csv),
              sequence_metrics_sha256=sha(out/"sequence_metrics.json")))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    main(parser.parse_args().out)
