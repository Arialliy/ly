#!/usr/bin/env python3
"""Independent CSV/raw checks using OpenCV components, no scoring-module import."""
from pathlib import Path
import csv,hashlib,json,math,time
import cv2
import numpy as np

OUT=Path(__file__).resolve().parent;Q=OUT.parents[1];P=Q/'stage1_goal';V=Q/'problem_validity_20261011'
def read(p):
    with p.open() as f:return list(csv.DictReader(f))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    started=time.time();checked=0;object_checked=0
    frame_rows={(r['sequence'],r['arrangement'],int(r['frame'])):r for r in read(OUT/'FRAME_METRICS.csv')}
    objects=read(OUT/'OBJECT_METRICS.csv');cs=read(OUT/'COMPONENTS.csv');vobjs=read(V/'OBJECTS.csv')
    for seq in ['DJI_0026_2','DJI_0067_2']:
        for mode,path in [('official',P/'predictions/official'/seq/'official_fused.npz'),('symmetric',V/'simple_controls/symmetric5_probmax'/seq/'fused.npz')]:
            with np.load(path,allow_pickle=False) as z:prob=z['probabilities']
            for t in range(160):
                pred=(prob[t]>=.5).astype(np.uint8);h,w=pred.shape
                n,lab,stats,cent=cv2.connectedComponentsWithStats(pred,connectivity=8)
                ff=frame_rows[(seq,mode,t)]
                assert n-1==int(ff['component_count'])
                assert int(pred.sum())==int(ff['positive_pixels'])
                expected_areas=sorted(int(r['area']) for r in cs if r['sequence']==seq and r['arrangement']==mode and int(r['frame'])==t)
                assert sorted(stats[1:,cv2.CC_STAT_AREA].tolist())==expected_areas
                bbox_union=np.zeros_like(pred,bool);center_union=np.zeros_like(pred,bool)
                gt=[r for r in vobjs if r['sequence']==seq and int(r['frame'])==t]
                ours={(int(r['annotation_id'])):r for r in objects if r['sequence']==seq and r['arrangement']==mode and int(r['frame'])==t}
                for g in gt:
                    x,y,bw,bh=[float(g[k]) for k in ['bbox_x','bbox_y','bbox_w','bbox_h']]
                    # Independent pixel-center inequality rasterization, avoiding crop helper.
                    xx=np.flatnonzero((np.arange(w)+.5>=x)&(np.arange(w)+.5<x+bw))
                    yy=np.flatnonzero((np.arange(h)+.5>=y)&(np.arange(h)+.5<y+bh))
                    bbox_union[np.ix_(yy,xx)]=True
                    boxhit=int(pred[np.ix_(yy,xx)].any())
                    cx=int(np.clip(math.floor(x+bw/2),0,w-1));cy=int(np.clip(math.floor(y+bh/2),0,h-1))
                    hx=np.flatnonzero(np.abs(np.arange(w)-cx)<=1);hy=np.flatnonzero(np.abs(np.arange(h)-cy)<=1)
                    centerhit=int(pred[np.ix_(hy,hx)].any())
                    bx=np.flatnonzero(np.abs(np.arange(w)-cx)<=4);by=np.flatnonzero(np.abs(np.arange(h)-cy)<=4)
                    center_union[np.ix_(by,bx)]=True
                    r=ours[int(g['annotation_id'])]
                    assert boxhit==int(r['box_pixel_hit']) and centerhit==int(r['center_proxy_hit'])
                    object_checked+=1
                assert int((pred.astype(bool)&~bbox_union).sum())==int(ff['positive_pixels_outside_boxes'])
                assert int((pred.astype(bool)&~center_union).sum())==int(ff['center_proxy_background_pixels'])
                checked+=1
    result=dict(status='PASS',frames=checked,objects=object_checked,independent_component_backend='cv2.connectedComponentsWithStats, 8 connectivity',independent_box_rasterization='explicit half-open inequality at pixel centers',independent_neighborhood='coordinate-distance boolean selection',checked_fields=['component_count','all_sorted_component_areas','positive_pixels','positive_pixels_outside_boxes','center_proxy_background_pixels','each_object_center_proxy_hit','each_object_box_pixel_hit'],elapsed_seconds=time.time()-started,source_script_sha256=digest(OUT/'compute_metrics.py'),verifier_sha256=digest(Path(__file__)),summary_sha256=digest(OUT/'SUMMARY.json'))
    (OUT/'RAW_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
