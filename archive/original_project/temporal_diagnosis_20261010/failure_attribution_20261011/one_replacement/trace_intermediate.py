#!/usr/bin/env python3
"""Directly trace the sole intermediate step loss with a retained matched window."""
import csv,json,importlib.util,hashlib
from pathlib import Path
import numpy as np
from scipy.ndimage import label
O=Path(__file__).resolve().parent;Q=O.parent.parent;P=Q/'stage1_goal'
spec=importlib.util.spec_from_file_location('score',P/'pilot/score_pilot.py');sc=importlib.util.module_from_spec(spec);spec.loader.exec_module(sc);helper,_=sc.helper()
rr=[r for r in csv.DictReader((O/'STEP_CHANGES.csv').open()) if r['change']=='loss' and r['after_any_window_match']=='1'];assert len(rr)==1
r=rr[0];seq=r['sequence'];frame=int(r['frame']);aid=int(r['annotation_id']);s=json.loads(r['after_success_starts'])[0]
gt,meta,_,_=sc.gt_data(P/'data'/seq/'annotations.json',helper)
probs={}
for a in [72,96]:
    with np.load(P/'predictions/official'/seq/'windows'/f'window_{a:03d}.npz') as z:probs[a]=z['probabilities'][frame-a]
source=probs[s];fused=np.maximum(probs[72],probs[96])
sr,_,sc=helper.match_frame(source,gt[frame],.5);fr,_,fc=helper.match_frame(fused,gt[frame],.5)
sr=next(x for x in sr if x['annotation_id']==aid);fr=next(x for x in fr if x['annotation_id']==aid)
assert sr['matched']==1 and fr['matched']==0
sl,_=label(source>=.5,np.ones((3,3)));fl,_=label(fused>=.5,np.ones((3,3)))
smask=sl==sr['component_id'];dest=np.unique(fl[smask]);assert len(dest)==1 and dest[0]>0
fid=int(dest[0]);fm=fl==fid;fc=next(x for x in fc if x['component_id']==fid)
allids=[int(a) for a in np.unique(sl[fm]) if a]
report=dict(status='COMPLETE',sequence=seq,frame=frame,source_frame=int(r['source_frame']),annotation_id=aid,track_id=int(r['track_id']),
    step='O->I1',retained_successful_window=s,new_window=96,source_match=sr,fused_match=fr,fused_component=fc,
    source_foreground_preserved=True,source_component_ids_mapped_to_fused_component=allids,direct_multicomponent_merge=len(allids)>1,
    new_pixels_connected=int((fm&(sl==0)).sum()),
    interpretation='This is an intermediate-arrangement loss. The final S removes window72, so endpoint stage differs from this path step.',
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(O/'INTERMEDIATE_FUSION_TRACE.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
