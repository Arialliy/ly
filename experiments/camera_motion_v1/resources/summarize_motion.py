import json,csv,collections,time
from pathlib import Path
import numpy as np
E=Path(__file__).resolve().parents[1]
summary={"sequences":{},"window_schedules":{"O":[0,36,72,108,120],"S":[0,24,60,96,120]}}
def stats(rows):
    trusted=[r for r in rows if str(r.get("trusted")).lower()=="true"]
    d={"pairs":len(rows),"trusted":len(trusted),"reasons":dict(collections.Counter(r.get("reason") for r in rows if str(r.get("trusted")).lower()!="true"))}
    for k in ["displacement_px","rotation_deg","scale","heldout_before_median_px","heldout_after_median_px","heldout_before_p90_px","heldout_after_p90_px","roundtrip_valid_fraction"]:
        v=np.array([float(r[k]) for r in trusted if r.get(k) not in (None,"")])
        if len(v):d[k]={"mean":float(v.mean()),"p10":float(np.quantile(v,.1)),"median":float(np.median(v)),"p90":float(np.quantile(v,.9)),"p95":float(np.quantile(v,.95)),"max":float(v.max())}
    return d
for seq in sorted((E/"motion").glob("DJI_*")):
    out={}
    for name in ["adjacent","windows"]:
        p=seq/(name+".csv")
        if p.exists():out[name]=stats(list(csv.DictReader(p.open())))
    windows={int(p.stem.split("_")[-1]):bool(np.load(p)["window_trusted"]) for p in (seq/"windows").glob("*.npz")}
    if windows:
        out["windows_trust"]=windows
        out["schedule_windows_trusted"]={s:sum(windows.get(k,False) for k in ks) for s,ks in summary["window_schedules"].items()}
    rows=[json.loads(p.read_text()) for p in (seq/"pairs").glob("*.json")]
    out["unique_matched_pairs"]=len(rows);out["unique_pair_seconds"]=sum(r["seconds"] for r in rows)
    summary["sequences"][seq.name]=out
summary["total_unique_pairs"]=sum(r["unique_matched_pairs"] for r in summary["sequences"].values())
summary["total_pair_seconds"]=sum(r["unique_pair_seconds"] for r in summary["sequences"].values())
(E/"motion/MOTION_SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))
