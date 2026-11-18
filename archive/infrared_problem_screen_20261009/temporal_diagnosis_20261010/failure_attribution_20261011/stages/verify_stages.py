#!/usr/bin/env python3
"""Second calculation: saved joins and independent scipy component tracing."""
import csv,json,hashlib
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from scipy.ndimage import label
S=Path(__file__).resolve().parent;Q=S.parent.parent;V=Q/'problem_validity_20261011';P=Q/'stage1_goal'
read=lambda p:list(csv.DictReader(p.open()))
rows=read(S/'ERROR_STAGE_TABLE.csv');assert len(rows)==5200
k=lambda r:(r['sequence'],int(r['frame']),int(r['annotation_id']))
lookup={k(r):r for r in rows};assert len(lookup)==5200
ctrl=[r for r in read(V/'control_summary/OBJECTS.csv') if r['mode']=='symmetric5_probmax' and r['small_geometry']=='1']
for r in ctrl:
    a=lookup[k(r)];assert a['official_matched']==r['official_matched'] and a['symmetric_matched']==r['control_matched']
windows=defaultdict(list)
for r in read(V/'WINDOWS.csv'):
    if k(r) in lookup:windows[k(r)].append(r)
for a in rows:
    for plan,starts in [('official',[0,36,72,108,120]),('symmetric',[0,24,60,96,120])]:
        rr=[r for r in windows[k(a)] if int(r['window_start']) in starts]
        success=[int(r['window_start']) for r in rr if r['matched']=='1']
        assert sorted(success)==json.loads(a[plan+'_success_starts'])
        assert len(rr)==int(a[plan+'_coverage'])
    if a['cohort']=='loss':
        assert not json.loads(a['symmetric_success_starts'])
        assert set(json.loads(a['official_success_starts']))<=set([36,72,108])
assert Counter(r['cohort'] for r in rows)==Counter(gain=74,loss=54,both_success=646,both_failure=4426)
target=[m for m in read(S/'COMPONENT_MAPPING.csv') if m['original_fusion_loss']=='1' and m['plan']=='official'];assert len(target)==5
audit=[]
for m in target:
    a=lookup[k(m)];seq=m['sequence'];frame=int(m['frame']);s=int(m['window_start'])
    with np.load(P/'predictions/official'/seq/'windows'/f'window_{s:03d}.npz') as z:src=z['probabilities'][frame-s]
    with np.load(P/'predictions/official'/seq/'official_fused.npz') as z:dst=z['probabilities'][frame]
    sl,_=label(src>=.5,np.ones((3,3)));dl,n=label(dst>=.5,np.ones((3,3)))
    sy=int(float(m['source_peak_y'])-.5);sx=int(float(m['source_peak_x'])-.5);cid=sl[sy,sx]
    srcmask=sl==cid;dstids=np.unique(dl[srcmask]);assert len(dstids)==1 and dstids[0]>0
    mask=dl==dstids[0];flat=np.flatnonzero(mask);ix=flat[np.argmax(dst.ravel()[flat])];fy,fx=divmod(int(ix),dst.shape[1])
    assert [fx+.5,fy+.5]==[float(m['fused_peak_x']),float(m['fused_peak_y'])]
    assert int(mask.sum())==int(m['fused_area']) and int(srcmask.sum())==int(m['source_area'])
    connected=set(np.unique(sl[mask]))-{0};assert len(connected)==int(m['source_components_in_fused_component'])
    bx,by,bw,bh=[float(a['bbox_'+z]) for z in ['x','y','w','h']]
    candidate_peaks=0
    for c in range(1,n+1):
        loc=np.flatnonzero(dl.ravel()==c);q=int(loc[np.argmax(dst.ravel()[loc])]);py,px=divmod(q,dst.shape[1])
        candidate_peaks+=int(bx<=px+.5<bx+bw and by<=py+.5<by+bh)
    assert candidate_peaks==0
    audit.append(dict(sequence=seq,source_frame=int(m['source_frame']),annotation_id=int(m['annotation_id']),source_foreground_preserved=True,
        fused_area=int(mask.sum()),source_components_merged=len(connected),fused_peak=[fx+.5,fy+.5],candidate_peaks_in_bbox=candidate_peaks))
report=dict(status='PASS',all5200_final_matches_equal_frozen_tables=True,all_selected_window_memberships_verified=True,
    all54_loss_original_success_windows_removed=True,all5_fusion_losses_independently_traced=audit,
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(S/'VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
