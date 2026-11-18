import json,datetime,shutil,hashlib
from pathlib import Path
import numpy as np
E=Path(__file__).resolve().parents[1];seq=E/"motion/DJI_0067_2";folder=seq/"pairs";valid=[];invalid=[]
for stem in sorted({p.stem for p in folder.iterdir()}):
    jp=folder/(stem+".json");npz=folder/(stem+".npz")
    try:
        r=json.loads(jp.read_text());z=np.load(npz,allow_pickle=False)
        assert z["H"].shape==(3,3);assert len(z["source_points"])==r["matches"]
        assert len(z["destination_points"])==r["matches"];assert len(z["fit_inlier_mask"])==r["matches"]
        assert len(z["fit_indices"])+len(z["heldout_indices"])==r["matches"]
        assert not set(z["fit_indices"]).intersection(z["heldout_indices"])
        for k in z.files:_=z[k]
        valid.append(stem)
    except Exception as exc:
        invalid.append({"pair":stem,"error":str(exc)})
        backup=E/"motion/interrupted_pair_files";backup.mkdir(exist_ok=True)
        for p in [jp,npz]:
            if p.exists():shutil.move(str(p),backup/p.name)
record={"utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"reason":"GPU1 confirmed sw_thermal_slowdown Active,84C,375MHz; no GPUsettings changed","old_receipt":"runtime/gim_gpu1.json","old_container":"camera_motion_gim_gpu1","new_dev_receipt":"runtime/gim_gpu2_resume0067.json","new_train_receipt":"runtime/gim_gpu0_remaining_train.json","config_sha256":hashlib.sha256((E/"motion/CONFIG.json").read_bytes()).hexdigest(),"code_sha256":hashlib.sha256((E/"code/estimate_motion.py").read_bytes()).hexdigest(),"completed_pairs_retained":len(valid),"retained_pairs":valid,"invalid_inflight_pairs_preserved_for_audit_and_recomputed":invalid,"numerical_protocol_changes":False,"seed_changes":False}
(E/"motion/MIGRATION_RECEIPT.json").write_text(json.dumps(record,indent=2)+"\n");print(json.dumps({k:v for k,v in record.items() if k!="retained_pairs"}))
