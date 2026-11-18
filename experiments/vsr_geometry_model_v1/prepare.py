"""Build fixed RGB, bidirectional motion and frozen x2 backbone inputs for A/B/C."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from backbone import load_backbone


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CACHE = ROOT / "cache"
SIGMA = .8
RADIUS = 3
HR_SIZE = 192
SHAPES = {"lr": (5,3,96,96), "hr": (3,192,192), "base": (3,192,192),
          "forward": (5,2,96,96), "backward": (5,2,96,96)}
FLOW_OPTIONS = dict(pyr_scale=.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)


def digest(path):
    with path.open("rb") as stream:
        return sha256(stream.read()).hexdigest()


def patches(records, center):
    ordered = sorted(records, key=lambda r:r["frame"])
    assert [r["frame"] for r in ordered] == list(range(center-2,center+3))
    frames = []
    for r in ordered:
        raw = (DATA / r["path"]).read_bytes()
        if sha256(raw).hexdigest()!=r["sha256"]:
            raise ValueError(f"Source SHA mismatch: {r['path']}")
        decoded = cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
        if decoded is None or decoded.shape != (720,1280,3):
            raise ValueError(f"Unexpected image geometry: {r['path']}")
        frames.append(cv2.cvtColor(decoded,cv2.COLOR_BGR2RGB))
    high, low, boxes = [], [], []
    for cy in (240,480):
        for cx in (320,960):
            y, x = cy-HR_SIZE//2, cx-HR_SIZE//2
            high.append(frames[2][y:y+HR_SIZE,x:x+HR_SIZE].astype(np.float32).transpose(2,0,1)/255)
            group=[]
            for frame in frames:
                halo = frame[y-RADIUS:y+HR_SIZE+RADIUS,x-RADIUS:x+HR_SIZE+RADIUS].astype(np.float32)/255
                filtered=cv2.GaussianBlur(halo, (2*RADIUS+1,2*RADIUS+1),SIGMA,sigmaY=SIGMA,
                                          borderType=cv2.BORDER_REFLECT_101)
                # x,y are even on the full frame. A 3px halo gives the full-frame PSF on every retained sample.
                group.append(filtered[RADIUS:-RADIUS:2,RADIUS:-RADIUS:2].transpose(2,0,1))
            low.append(np.stack(group))
            boxes.append([x,y,x+HR_SIZE,y+HR_SIZE])
    return np.stack(low), np.stack(high), boxes


def bidirectional_flow(lr):
    # Farneback's numeric scale is designed for gray values in approximately [0,255].
    gray=[np.ascontiguousarray(cv2.cvtColor(p.transpose(1,2,0),cv2.COLOR_RGB2GRAY)*255) for p in lr]
    h,w=gray[2].shape
    forward=np.zeros((5,2,h,w),dtype=np.float32)
    backward=np.zeros_like(forward)
    for j in (0,1,3,4):
        forward[j]=cv2.calcOpticalFlowFarneback(gray[2],gray[j],None,**FLOW_OPTIONS).transpose(2,0,1)
        backward[j]=cv2.calcOpticalFlowFarneback(gray[j],gray[2],None,**FLOW_OPTIONS).transpose(2,0,1)
    return forward,backward


def validate(path):
    with np.load(path,allow_pickle=False) as archive:
        if set(archive.files)!=set(SHAPES):
            raise ValueError(f"Missing or extra cache keys: {path}")
        for name,shape in SHAPES.items():
            arr=archive[name]
            if arr.shape!=shape or arr.dtype!=np.float32 or not np.isfinite(arr).all():
                raise ValueError(f"Invalid cache field {name}: {path}")
            if name in ("forward","backward") and arr[2].any():
                raise ValueError(f"Center flow must vanish: {path}")
            if name in ("lr","hr") and (arr.min() < -1e-6 or arr.max() > 1+1e-6):
                raise ValueError(f"Out-of-range input: {path}")


@torch.inference_mode()
def main(device):
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    data_manifest=DATA/"MANIFEST.json"
    manifest=json.loads(data_manifest.read_text())
    groups=defaultdict(list)
    for r in manifest["files"]:
        groups[(r["split"],r["sequence"],r["center"])].append(r)
    if Counter(k[0] for k in groups)!=dict(train=24,evaluation=8):
        raise ValueError("Expected 24 training and 8 evaluation windows")
    net=load_backbone(device=device).eval().requires_grad_(False)
    recipe = {
        "data_manifest_sha256":digest(data_manifest),"prepare_sha256":digest(Path(__file__)),
        "backbone_sha256":digest(ROOT/"backbone.py"),
        "backbone_sources":{p.name:digest(p) for p in sorted((ROOT/"third_party").glob("*")) if p.is_file()},
        "blur_sigma_hr_pixels":SIGMA,"gaussian_radius":RADIUS,"flow":FLOW_OPTIONS,
        "grid":"full-frame even HR lattice, acquired after 7x7 Gaussian blur; five RGB [0,1] frames",
        "crop":"192 HR context; center HR region [32:160,32:160] for scoring",
        "torch":str(torch.__version__),"opencv":cv2.__version__,"numpy":np.__version__,
        "device":torch.cuda.get_device_name(0) if str(device).startswith("cuda") else str(device)}
    CACHE.mkdir(exist_ok=True)
    recipe_path=CACHE/"RECIPE.json"
    if recipe_path.exists() and json.loads(recipe_path.read_text())!=recipe:
        raise ValueError("Existing cached inputs use a different recipe")
    recipe_path.write_text(json.dumps(recipe,indent=2,sort_keys=True)+"\n")
    files=[]
    for (split,seq,center),records in sorted(groups.items(),key=lambda kv:(kv[0][0]!="train",kv[0][1:])):
        split_root=CACHE/split
        split_root.mkdir(exist_ok=True)
        low,high,boxes=patches(records,center)
        outputs=[split_root/f"{seq:03}_{center:02}_{j}.npz" for j in range(4)]
        needed=[j for j,p in enumerate(outputs) if not p.exists()]
        if needed:
            predictions=net(torch.from_numpy(low[needed]).to(device)).cpu().float().numpy()
            if predictions.shape!=(len(needed),3,192,192):
                raise ValueError(f"Unexpected x2 backbone output {predictions.shape}")
            for i,j in enumerate(needed):
                forward,backward=bidirectional_flow(low[j])
                path=outputs[j]
                pending=path.with_suffix(".pending")
                with pending.open("wb") as out:
                    np.savez(out,lr=low[j],hr=high[j],base=predictions[i],forward=forward,backward=backward)
                validate(pending)
                pending.replace(path)
        for j,path in enumerate(outputs):
            if j not in needed:
                validate(path)
            files.append(dict(path=str(path.relative_to(CACHE)),split=split,sequence=seq,center=center,
                              patch=j,hr_box_xyxy=boxes[j],sha256=digest(path)))
        print(f"Verified {split}/{seq:03}/{center:02} ({len(files)}/128)",flush=True)
    counts=Counter(f["split"] for f in files)
    assert counts==dict(train=96,evaluation=32)
    result=dict(recipe_sha256=digest(recipe_path),shapes=SHAPES,counts=counts,files=files)
    tmp=CACHE/"MANIFEST.pending"
    tmp.write_text(json.dumps(result,indent=2)+"\n")
    tmp.replace(CACHE/"MANIFEST.json")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--device",default="cuda")
    main(parser.parse_args().device)
