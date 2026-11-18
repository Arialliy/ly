"""Frozen GIM-DKM correspondences, heldout homography validation, label-free inputs."""
import argparse, csv, hashlib, json, math, os, sys, time
from pathlib import Path
import cv2
import numpy as np

E = Path(__file__).resolve().parents[1]
CFG = json.loads((E / "motion/CONFIG.json").read_text())
TRAIN = ["DJI_0030_9", "DJI_0043_2", "DJI_0057_1", "DJI_0113_3", "DJI_0115_2", "DJI_0217_2", "DJI_0309_1", "DJI_0323_2"]
DEV = ["DJI_0026_2", "DJI_0067_2"]
DATA = E.parents[1] / "archive/infrared_problem_screen_20261009/temporal_diagnosis_20261010/stage1_goal/data"

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()

def project(H, p):
    x = np.c_[p, np.ones(len(p))] @ H.T
    return x[:, :2] / x[:, 2:3]

def symmetric_errors(H, src, dst):
    a = np.linalg.norm(project(H, src) - dst, axis=1)
    b = np.linalg.norm(project(np.linalg.inv(H), dst) - src, axis=1)
    return np.sqrt((a*a + b*b) / 2)

def coverage(points, width, height):
    if len(points) < 3: return 0., 0
    hull = cv2.contourArea(cv2.convexHull(points.astype(np.float32))) / (width*height)
    cell = np.clip((points / [width, height] * CFG["grid_size"]).astype(int), 0, CFG["grid_size"]-1)
    return float(hull), len(np.unique(cell, axis=0))

def geometry_stats(H, width, height):
    gx, gy = np.meshgrid(np.linspace(0, width-1, 5), np.linspace(0, height-1, 5))
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
    q = project(H, grid)
    J = np.stack([project(H, grid+[.01, 0])-q, project(H, grid+[0,.01])-q], axis=-1) / .01
    scales = np.linalg.svd(J, compute_uv=False)
    determinants = np.linalg.det(J)
    denominator = np.c_[grid, np.ones(len(grid))] @ H[2]
    corners = project(H, np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]],float))
    ones = np.ones((height,width),np.float32)
    forward = cv2.warpPerspective(ones,H,(width,height),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    valid = cv2.warpPerspective((forward >= 1.-1e-6).astype(np.float32),np.linalg.inv(H),(width,height),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT) >= 1.-1e-6
    center = np.array([[(width-1)/2.,(height-1)/2.]])
    delta = project(H,center)[0]-center[0]
    j = J[len(J)//2]
    s = math.sqrt(abs(np.linalg.det(j)))
    angle = math.degrees(math.atan2(j[1,0]-j[0,1],j[0,0]+j[1,1]))
    return {"displacement_px":float(np.linalg.norm(delta)),"dx_px":float(delta[0]),"dy_px":float(delta[1]),"rotation_deg":angle,"scale":s,
       "min_local_scale":float(scales.min()),"max_local_scale":float(scales.max()),"no_fold":bool(np.all(determinants>0) and (np.all(denominator>0) or np.all(denominator<0))),
       "corner_bounded":bool(np.max(np.abs(corners))<=CFG["corner_abs_limit_image_diagonals"]*math.hypot(width,height)),"roundtrip_valid_fraction":float(valid.mean())}

def fit_validate(src, dst, width, height, seed):
    n=len(src); rng=np.random.default_rng(seed); perm=rng.permutation(n)
    n_val=int(math.ceil(n*CFG["holdout_fraction"])); val=perm[:n_val]; fit=perm[n_val:]
    row={"matches":n,"fit_count":len(fit),"holdout_count":len(val),"trusted":False,"reason":"insufficient_matches"}
    H=np.full((3,3),np.nan); inlier=np.zeros(n,bool); residual=np.full(n,np.nan)
    if n < CFG["min_matches"]:return H,row,fit,val,inlier,residual
    cv2.setRNGSeed(seed & 0x7fffffff)
    H,mask=cv2.findHomography(src[fit],dst[fit],cv2.USAC_MAGSAC,CFG["ransac_threshold_px"],maxIters=CFG["ransac_max_iters"],confidence=CFG["ransac_confidence"])
    if H is None or not np.isfinite(H).all() or abs(np.linalg.det(H))<1e-12:
        row["reason"]="invalid_homography";return np.full((3,3),np.nan),row,fit,val,inlier,residual
    H=H/H[2,2]; inlier[fit]=mask.ravel().astype(bool)
    try:
        residual=symmetric_errors(H,src,dst); stats=geometry_stats(H,width,height)
    except (np.linalg.LinAlgError,cv2.error,ValueError):
        row["reason"]="invalid_geometry";return H,row,fit,val,inlier,residual
    hs,gs=coverage(src[inlier],width,height); hd,gd=coverage(dst[inlier],width,height)
    before=np.linalg.norm(src[val]-dst[val],axis=1); after=residual[val]
    row.update(stats)
    row.update(fit_inliers=int(inlier.sum()),fit_inlier_ratio=float(inlier.sum()/len(fit)),source_hull_fraction=hs,destination_hull_fraction=hd,source_grid_cells=gs,destination_grid_cells=gd,
        heldout_before_median_px=float(np.median(before)),heldout_before_p90_px=float(np.quantile(before,.9)),heldout_after_median_px=float(np.median(after)),heldout_after_p90_px=float(np.quantile(after,.9)),heldout_within_3px_fraction=float(np.mean(after<=3)))
    checks={"fit_inliers":inlier.sum()>=CFG["min_fit_inliers"],"fit_ratio":row["fit_inlier_ratio"]>=CFG["min_fit_inlier_ratio"],"coverage_hull":min(hs,hd)>=CFG["min_inlier_hull_fraction_both"],"coverage_grid":min(gs,gd)>=CFG["min_inlier_grid_cells_both"],"holdout_count":len(val)>=CFG["min_holdout"],"holdout_median":row["heldout_after_median_px"]<=CFG["max_holdout_symmetric_median_px"],"holdout_p90":row["heldout_after_p90_px"]<=CFG["max_holdout_symmetric_p90_px"],"holdout_fraction":row["heldout_within_3px_fraction"]>=CFG["min_holdout_fraction_within_3px"],"local_scale":stats["min_local_scale"]>=CFG["local_scale_range"][0] and stats["max_local_scale"]<=CFG["local_scale_range"][1],"no_fold":stats["no_fold"],"corner_bound":stats["corner_bounded"],"valid_area":stats["roundtrip_valid_fraction"]>=CFG["min_roundtrip_valid_fraction"]}
    row["trusted"]=bool(all(checks.values()));row["reason"]="ok" if row["trusted"] else ";".join(k for k,v in checks.items() if not v)
    return H,row,fit,val,inlier,residual

class GIMMatcher:
    def __init__(self):
        import torch
        sys.path.insert(0,str(E/"resources/gim_minimal"))
        from networks.dkm.models.model_zoo.DKMv3 import DKMv3
        self.torch=torch
        self.model=DKMv3(weights=None,h=CFG["model_h"],w=CFG["model_w"])
        state=torch.load(E/"resources/gim_dkm_100h.ckpt",map_location="cpu")
        if "state_dict" in state:state=state["state_dict"]
        state={k.removeprefix("model."):v for k,v in state.items() if "encoder.net.fc" not in k}
        self.model.load_state_dict(state,strict=True);self.model=self.model.eval().cuda()
        for p in self.model.parameters():p.requires_grad_(False)
    def tensor(self,image):
        torch=self.torch
        if image.ndim==2:image=np.repeat(image[...,None],3,axis=2)
        return torch.from_numpy(image.copy()).permute(2,0,1).float()[None].cuda()/255.
    def __call__(self,source,destination,seed):
        torch=self.torch
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        a=self.tensor(source);b=self.tensor(destination)
        def pad(t):
            h,w=t.shape[-2:]; aspect=CFG["model_w"]/CFG["model_h"]
            nw=max(w,int(h*aspect));nh=max(h,int(w/aspect));l=(nw-w)//2;top=(nh-h)//2
            return torch.nn.functional.pad(t,(l,nw-w-l,top,nh-h-top)),(l,top)
        a,pa=pad(a);b,pb=pad(b)
        with torch.inference_mode():
            dense,certainty=self.model.match(a,b)
            matches,confidence=self.model.sample(dense,certainty,CFG["sample_matches"])
        # DKM uses align_corners=False normalized pixel centers: x=(u+1)*W/2 - .5.
        p=matches[:,:2].cpu().numpy();q=matches[:,2:].cpu().numpy();c=confidence.cpu().numpy()
        h,w=source.shape[:2]; h2,w2=destination.shape[:2]
        hp,wp=a.shape[-2:];hp2,wp2=b.shape[-2:]
        p=(p+1)*[wp,hp]/2-.5-pa;q=(q+1)*[wp2,hp2]/2-.5-pb
        keep=np.isfinite(p).all(1)&np.isfinite(q).all(1)&(p>=0).all(1)&(q>=0).all(1)&(p<=[w-1,h-1]).all(1)&(q<=[w2-1,h2-1]).all(1)&np.isfinite(c)&(c>CFG["correspondence_confidence_filter"])
        return p[keep],q[keep],c[keep]

def seed_for(name):return (int(hashlib.sha256(name.encode()).hexdigest()[:8],16)^CFG["seed"])&0x7fffffff

def pair(matcher,frames,sequence,s,d,kind):
    base=E/"motion"/sequence/"pairs"/f"{s:03d}_to_{d:03d}"
    if base.with_suffix(".json").exists():
        row=json.loads(base.with_suffix(".json").read_text()); arr=np.load(base.with_suffix(".npz"));return arr["H"],row
    start=time.perf_counter();seed=seed_for(f"{sequence}:{s}:{d}")
    src,dst,conf=matcher(frames[s],frames[d],seed)
    H,row,fit,val,inlier,resid=fit_validate(src,dst,frames.shape[2],frames.shape[1],seed)
    row.update(sequence=sequence,source_frame=s,destination_frame=d,kind=kind,seed=seed,seconds=time.perf_counter()-start)
    base.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(base.with_suffix(".npz"),H=H,source_points=src,destination_points=dst,confidence=conf,fit_indices=fit,heldout_indices=val,fit_inlier_mask=inlier,symmetric_residual_px=resid)
    base.with_suffix(".json").write_text(json.dumps(row,indent=2)+"\n")
    return H,row

def save_csv(path,rows):
    keys=sorted(set().union(*(x.keys() for x in rows)))
    with path.open("w",newline="") as f:
        wr=csv.DictWriter(f,fieldnames=keys);wr.writeheader();wr.writerows(rows)

def main():
    p=argparse.ArgumentParser();p.add_argument("--sequences",nargs="+",default=DEV+TRAIN);p.add_argument("--mode",choices=["adjacent","windows","both"],default="both");p.add_argument("--limit",type=int,default=0);a=p.parse_args()
    assert set(a.sequences)<=set(TRAIN+DEV),"Confirmation forbidden"
    import torch
    torch.set_num_threads(2);cv2.setNumThreads(1)
    matcher=GIMMatcher(); count=0
    for seq in a.sequences:
        frames=np.load(DATA/seq/"raw_frames.npy",mmap_mode="r")
        assert len(frames)==160 and frames.dtype==np.uint8,(seq,frames.shape,frames.dtype)
        out=E/"motion"/seq;out.mkdir(exist_ok=True)
        if a.mode in ["windows","both"] and seq in DEV:
            rows=[]
            for start in CFG["window_starts_union"]:
                matrices=[];trusted=[];window_rows=[];anchor=start+CFG["anchor_offset"]
                for s in range(start,start+40):
                    if s==anchor:
                        H=np.eye(3);row={"sequence":seq,"source_frame":s,"destination_frame":anchor,"trusted":True,"reason":"anchor_identity","seconds":0.}
                    else:
                        H,row=pair(matcher,frames,seq,s,anchor,"window");count+=1
                    matrices.append(H);trusted.append(row["trusted"]);wr=dict(row,window_start=start);rows.append(wr);window_rows.append(wr)
                    if a.limit and count>=a.limit:return
                (out/"windows").mkdir(exist_ok=True)
                np.savez_compressed(out/"windows"/f"window_{start:03d}.npz",H=np.array(matrices),trusted=np.array(trusted,bool),window_trusted=np.array(all(trusted)),start=start,anchor=anchor,config_sha256=digest(E/"motion/CONFIG.json"))
                print(json.dumps({"sequence":seq,"window":start,"trusted_frames":sum(trusted),"window_trusted":all(trusted)}),flush=True)
            save_csv(out/"windows.csv",rows)
        if a.mode in ["adjacent","both"]:
            rows=[]
            for s in range(159):
                H,row=pair(matcher,frames,seq,s,s+1,"adjacent");rows.append(row);count+=1
                if s%20==0:print(json.dumps({"sequence":seq,"adjacent":s,"trusted":row["trusted"],"seconds":row["seconds"]}),flush=True)
                if a.limit and count>=a.limit:return
            save_csv(out/"adjacent.csv",rows)
            Hs=np.full((160,3,3),np.nan);trusted=np.zeros(160,bool)
            for t in range(1,160):
                Hs[t]=np.load(out/"pairs"/f"{t-1:03d}_to_{t:03d}.npz")["H"];trusted[t]=rows[t-1]["trusted"]
            np.savez_compressed(out/"adjacent.npz",H=Hs,trusted=trusted)
    print("MOTION_COMPLETE",flush=True)

if __name__=="__main__":main()
