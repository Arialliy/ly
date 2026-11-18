#!/usr/bin/env python3
"""Constructive limits of the support upper bound using the frozen matcher."""
import hashlib,importlib.util,json
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent;Q=OUT.parent.parent;path=Q/'analysis/analyze.py'
s=importlib.util.spec_from_file_location('matcher_bound_audit',path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def evaluate(probabilities,annotations):
    gt,_=m.prepare_gt(annotations,1,len(probabilities[0]))
    rows,stats,components=m.match_frame(np.array(probabilities,np.float32),gt,.5)
    return dict(matched_by_track={str(r['track_id']):r['matched'] for r in rows},peaks=[[c['peak_x'],c['peak_y']] for c in components],stats=stats)
annotations=[dict(id=1,track_id=1,category_id=0,bbox=[0,0,5,1]),dict(id=2,track_id=2,category_id=0,bbox=[0,0,1,1])]
a=[[.9,0,0,0,0]];b=[[0,0,0,0,.8]];fused=np.maximum(a,b).tolist()
overlap=dict(window_a=evaluate(a,annotations),window_b=evaluate(b,annotations),fusion=evaluate(fused,annotations))
assert overlap['window_a']['matched_by_track']['2']==overlap['window_b']['matched_by_track']['2']==0
assert overlap['fusion']['matched_by_track']['2']==1
annotations=[dict(id=1,track_id=1,category_id=0,bbox=[0,0,1,1])]
a=[[.8,0,0,0,0]];b=[[0,.7,.9,0,0]];fused=np.maximum(a,b).tolist()
nonmonotone=dict(window_a=evaluate(a,annotations),window_b=evaluate(b,annotations),fusion=evaluate(fused,annotations))
assert nonmonotone['window_a']['matched_by_track']['1']==1
assert nonmonotone['fusion']['matched_by_track']['1']==0
result=dict(status='PASS',synthetic_diagnostics_only=True,
    overlapping_boxes_can_gain_after_fusion_without_single_window_actual_match=overlap,
    nonoverlapping_box_can_lose_after_adding_window=nonmonotone,
    implication='Nonoverlap is necessary for the actual-match union upper bound; the bound does not imply attainability or monotonicity.',
    helper_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
(OUT/'PEAK_BOUND_CHECK.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
