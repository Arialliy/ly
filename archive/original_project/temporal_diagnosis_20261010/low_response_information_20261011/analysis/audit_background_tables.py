#!/usr/bin/env python3
"""Independent literal block audit for frozen low-response output; no resampling design."""
import csv,hashlib,json,math,time
from collections import defaultdict
from pathlib import Path
import numpy as np

BASE=Path(__file__).resolve().parent
RUN=BASE.parent
Q=RUN.parent
P=Q/'stage1_goal'
rows=list(csv.DictReader((BASE/'LOW_RESPONSE_OBJECTS.csv').open()))
assert len(rows)==10400 and len({(r['sequence'],r['frame'],r['annotation_id'],r['arrangement']) for r in rows})==10400

def close(a,b):
    assert abs(float(a)-float(b))<1e-8,(a,b)

def mask_for(aa):
    mask=np.zeros((512,640),bool)
    for r in aa:
        x,y,w,h=map(float,r['bbox'])
        if w<=0 or h<=0:continue
        a,b=max(0,min(640,math.ceil(x-.5))),max(0,min(512,math.ceil(y-.5)))
        c,d=max(0,min(640,math.ceil(x+w-.5))),max(0,min(512,math.ceil(y+h-.5)))
        if c>a and d>b:mask[max(0,b-2):min(512,d+2),max(0,a-2):min(640,c+2)]=True
    return mask

partitions=defaultdict(list)
for r in rows:partitions[(r['sequence'],r['arrangement'])].append(r)
selected=[]
for key,rr in sorted(partitions.items()):
    rr=sorted(rr,key=lambda r:hashlib.sha256(f"audit:{r['sequence']}:{r['frame']}:{r['annotation_id']}:{r['arrangement']}".encode()).digest())
    selected+=rr[:12]
validations=[];started=time.time()
for (seq,plan),rr in partitions.items():
    path=P/'data'/seq/'annotations.json'
    if not path.exists():path=P/'data_diagnostics'/seq/'annotations.json'
    ann=json.loads(path.read_text());frames={im['id']:i for i,im in enumerate(ann['images'])};aas=defaultdict(list)
    for a in ann['annotations']:aas[frames[a['image_id']]].append(a)
    raw=np.load(RUN/'logits'/seq/f'fused_{plan}_logits.npy',mmap_mode='r')
    for r in [s for s in selected if (s['sequence'],s['arrangement'])==(seq,plan)]:
        f=int(r['frame']);z=raw[f];mask=mask_for(aas[f]);x=int(r['crop_x0']);y=int(r['crop_y0']);w=int(r['crop_w']);h=int(r['crop_h'])
        target=float(z[y:y+h,x:x+w].max());close(target,r['target_box_max_logit'])
        local=[];glob=[]
        # Literal independent loops, unlike the main vectorized global reshape.
        for by in range(0,512-h+1,h):
            for bx in range(0,640-w+1,w):
                if not mask[by:by+h,bx:bx+w].any():glob.append(float(z[by:by+h,bx:bx+w].max()))
        mx=2*max(w,h);left=max(0,x-mx);right=min(640,x+w+mx);top=max(0,y-mx);bottom=min(512,y+h+mx)
        for j in range(-100,101):
            by=y+j*h
            if by<top or by+h>bottom:continue
            for i in range(-100,101):
                bx=x+i*w
                if bx<left or bx+w>right or mask[by:by+h,bx:bx+w].any():continue
                local.append(float(z[by:by+h,bx:bx+w].max()))
        for pref,vals in [('local',local),('global',glob)]:
            assert len(vals)==int(r[pref+'_valid_blocks'])
            if vals:
                close(np.quantile(vals,.95),r[pref+'_bg_q95_logit']);close(np.quantile(vals,.99),r[pref+'_bg_q99_logit'])
                assert sum(v>target for v in vals)==int(r[pref+'_blocks_stronger_than_target'])
                assert sum(v>=target for v in vals)==int(r[pref+'_blocks_not_lower_than_target'])
                close((sum(v<target for v in vals)+.5*sum(v==target for v in vals))/len(vals),r[pref+'_target_midrank_percentile'])
        if int(r['reference_valid']):
            rx=int(r['reference_x0']);ry=int(r['reference_y0'])
            assert not mask[ry:ry+h,rx:rx+w].any()
            close(z[ry:ry+h,rx:rx+w].max(),r['reference_max_logit'])
        validations.append({'sequence':seq,'arrangement':plan,'frame':f,'annotation_id':int(r['annotation_id']),'literal_global_and_local_blocks_match':True})
# All rows: immutable geometry/reference selection plus exact identities.
refs=defaultdict(set)
for r in rows:
    refs[(r['sequence'],r['track_id'])].add((r['reference_offset_dx'],r['reference_offset_dy'],r['reference_offset_D'],r['reference_candidate_index']))
    assert int(r['global_background_peaks_not_lower_than_target'])==int(r['global_background_peaks_stronger_than_target'])+int(r['global_background_peaks_tied_with_target'])
    for pref in ['local','global']:
        if int(r[pref+'_valid_blocks']):
            assert int(r[pref+'_blocks_not_lower_than_target'])==int(r[pref+'_blocks_stronger_than_target'])+int(r[pref+'_blocks_tied_with_target'])
            assert int(r[pref+'_advantage_q95'])==int(float(r[pref+'_margin_q95'])>0)
assert len(refs)==130 and all(len(v)==1 for v in refs.values())
result={'status':'PASS','detailed_rows':len(rows),'literal_block_audit_rows':len(validations),'selection':'12 identity-hash rows per sequence x arrangement, no score selection',
        'track_fixed_reference_offsets':len(refs),'count_identities_checked_all_rows':True,'literal_audits':validations,'elapsed_seconds':time.time()-started}
(BASE/'INDEPENDENT_BLOCK_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='literal_audits'}))
