import sys,json,hashlib,csv,time
from pathlib import Path
import numpy as np
E=Path(__file__).resolve().parents[1];sys.path.insert(0,str(E/"code"))
from estimate_motion import TRAIN,DEV,CFG
out={"status":"PASS","config_sha256":hashlib.sha256((E/"motion/CONFIG.json").read_bytes()).hexdigest(),"sequences":{},"pairs_checked":0,"confirmation_access":False}
assert {p.name for p in (E/"motion").glob("DJI_*")}==set(TRAIN+DEV)
for seq in TRAIN+DEV:
 d=E/"motion"/seq;rows=list(csv.DictReader((d/"adjacent.csv").open()));assert len(rows)==159
 z=np.load(d/"adjacent.npz",allow_pickle=False);assert z["H"].shape==(160,3,3) and z["trusted"].shape==(160,) and not z["trusted"][0]
 for t,r in enumerate(rows,1):
  assert int(r["source_frame"])==t-1 and int(r["destination_frame"])==t
  assert (r["trusted"]=="True")==z["trusted"][t]
  H=np.load(d/"pairs"/f"{t-1:03d}_to_{t:03d}.npz")["H"];assert np.allclose(z["H"][t],H,equal_nan=True)
 fs=list((d/"pairs").glob("*.json"));assert len(fs)==(463 if seq in DEV else 159)
 for p in fs:
  r=json.loads(p.read_text());n=r["matches"];a=np.load(p.with_suffix(".npz"),allow_pickle=False)
  assert a["H"].shape==(3,3) and a["source_points"].shape==a["destination_points"].shape==(n,2)
  assert a["fit_inlier_mask"].shape==(n,);fi=a["fit_indices"];vi=a["heldout_indices"]
  assert len(set(fi)&set(vi))==0 and len(fi)+len(vi)==n and len(set(fi)|set(vi))==n
  assert not a["fit_inlier_mask"][vi].any()
  if r["trusted"]:assert np.isfinite(a["H"]).all()
  out["pairs_checked"]+=1
 if seq in DEV:
  wr=list(csv.DictReader((d/"windows.csv").open()));assert len(wr)==320
  for start in CFG["window_starts_union"]:
   a=np.load(d/"windows"/f"window_{start:03d}.npz",allow_pickle=False)
   assert a["H"].shape==(40,3,3) and a["trusted"].shape==(40,)
   assert bool(a["window_trusted"])==bool(a["trusted"].all())
   assert str(a["config_sha256"])==out["config_sha256"]
   assert np.array_equal(a["H"][19],np.eye(3))
   for c in ["C1","C2"]:
    cp=E/"predictions"/c/seq/"COMPLETE.json"
    if cp.exists():
     m=json.loads(cp.read_text());r=next(x for x in m["window_timing"] if x["start"]==start)
     assert r["motion_sha256"]==hashlib.sha256((d/"windows"/f"window_{start:03d}.npz").read_bytes()).hexdigest()
 out["sequences"][seq]={"adjacent":len(rows),"unique_pairs":len(fs),"window_count":8 if seq in DEV else 0}
assert out["pairs_checked"]==2198
(E/"motion/FINAL_CHECK.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out))
