"""Cross-environment release-cache witness; no new scientific C0 condition."""
import json
import time
import cv2
import numpy as np
import torch
from runtime import *

setup();model,mp=load_model();records=[]
with torch.inference_mode():
    for seq in SEQS:
        raw=np.load(B/'data/stage1'/seq/'raw_frames.npy',mmap_mode='r')
        x=torch.from_numpy((raw[:40].astype(np.float32)-85.0799)/47.4845)[None,None].cuda()
        tick=time.perf_counter();p=tiled_logits(model,x).sigmoid()[0].numpy();seconds=time.perf_counter()-tick
        old=np.load(C0/seq/'windows/window_000.npz')['probabilities']
        diff=np.abs(p-old)
        rec={'sequence':seq,'window':0,'max_probability_error':float(diff.max()),
             'mean_probability_error':float(diff.mean()),'threshold05_disagreements':int(((p>=.5)!=(old>=.5)).sum()),'seconds':seconds}
        records.append(rec);print(json.dumps(rec),flush=True)
        del x,p,old,diff
passed=all(r['max_probability_error']<=2e-5 and r['threshold05_disagreements']==0 for r in records)
save(E/'checks/RUNTIME_EQUIVALENCE.json',{'status':'PASS' if passed else 'FAIL','checkpoint_sha256':sha(CP),
     'model_source_sha256':sha(mp),'torch':torch.__version__,'numpy':np.__version__,'opencv':cv2.__version__,
     'records':records,'scope':'Two complete 40-frame release windows; no training; original C0 cache retained.'})
assert passed, records
