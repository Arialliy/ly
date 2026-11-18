#!/usr/bin/env python3
"""Deterministic plots of every original-five fusion loss, with raw data crops."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

S=Path(__file__).resolve().parent;Q=S.parent.parent;P=Q/'stage1_goal'
O=S/'fusion_trace_figures';O.mkdir(exist_ok=True)
rows=[r for r in csv.DictReader((S/'COMPONENT_MAPPING.csv').open()) if r['original_fusion_loss']=='1' and r['plan']=='official']
objects={(r['sequence'],r['annotation_id']):r for r in csv.DictReader((S/'ERROR_STAGE_TABLE.csv').open())}
components={(r['sequence'],r['annotation_id'],r['component_id']):r for r in csv.DictReader((S/'FOCUS_COMPONENTS.csv').open()) if r['plan']=='official'}
assert len(rows)==5
manifest=[]
for r in rows:
    seq=r['sequence'];frame=int(r['frame']);s=int(r['window_start']);other=json.loads(r['fused_peak_contributing_windows'])[0]
    a=objects[(seq,r['annotation_id'])];c=components[(seq,r['annotation_id'],r['fused_component'])]
    x,y,w,h=[float(a['bbox_'+z]) for z in ['x','y','w','h']]
    l,t,cw,ch=[int(c['bbox_'+z]) for z in ['left','top','width','height']]
    x0=max(0,int(min(x,l))-10);x1=min(640,int(np.ceil(max(x+w,l+cw)))+10)
    y0=max(0,int(min(y,t))-10);y1=min(512,int(np.ceil(max(y+h,t+ch)))+10)
    rawpath=P/'data'/seq/'raw_frames.npy';raw=np.load(rawpath,mmap_mode='r')[frame]
    display_lo,display_hi=np.percentile(raw,[.5,99.5])
    crop=raw[y0:y1,x0:x1]
    arrays=[]
    for start in [s,other]:
        with np.load(P/'predictions/official'/seq/'windows'/f'window_{start:03d}.npz') as z:arrays.append(z['probabilities'][frame-start,y0:y1,x0:x1])
    with np.load(P/'predictions/official'/seq/'official_fused.npz') as z:arrays.append(z['probabilities'][frame,y0:y1,x0:x1])
    fig,axes=plt.subplots(1,4,figsize=(13,4),constrained_layout=True)
    extent=(x0,x1,y1,y0)
    axes[0].imshow(crop,cmap='gray',vmin=display_lo,vmax=display_hi,extent=extent)
    titles=['Current infrared frame',f'Matched window {s}:{s+40}',f'Peak-contributing window {other}:{other+40}','Official five-window maximum']
    for ax,arr in zip(axes[1:],arrays):
        im=ax.imshow(arr,cmap='magma',vmin=0,vmax=1,extent=extent,interpolation='nearest')
        if np.any(arr>=.5) and not np.all(arr>=.5):ax.contour(np.arange(x0,x1)+.5,np.arange(y0,y1)+.5,arr,levels=[.5],colors=['cyan'],linewidths=.8)
    for ax,title in zip(axes,titles):
        ax.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor='orange',linewidth=1.4))
        ax.plot(float(r['source_peak_x']),float(r['source_peak_y']),'x',color='lime',ms=8,mew=1.5)
        ax.plot(float(r['fused_peak_x']),float(r['fused_peak_y']),'+',color='magenta',ms=9,mew=1.5)
        ax.set_title(title,fontsize=9);ax.set_xlim(x0,x1);ax.set_ylim(y1,y0);ax.tick_params(labelsize=7)
    fig.colorbar(im,ax=axes[1:],shrink=.55,label='Probability')
    fig.suptitle(f"{seq} frame {r['source_frame']} | track {r['track_id']} | area {r['source_area']} -> {r['fused_area']}\n"
                 f"Green x: successful-window peak; magenta +: fused peak; orange: annotation; cyan: threshold 0.5",fontsize=10)
    name=f"{seq}_frame_{r['source_frame']}_track_{r['track_id']}"
    fig.savefig(O/(name+'.png'),dpi=180);fig.savefig(O/(name+'.pdf'));plt.close(fig)
    np.savez_compressed(O/(name+'_crop.npz'),raw=crop,successful=arrays[0],other=arrays[1],fused=arrays[2],extent=np.array(extent),bbox=np.array([x,y,w,h]))
    manifest.append(dict(sequence=seq,frame=frame,source_frame=int(r['source_frame']),track_id=int(r['track_id']),annotation_id=int(r['annotation_id']),
        matched_window=s,other_peak_contributor=other,png=name+'.png',pdf=name+'.pdf',crop_npz=name+'_crop.npz',raw_path=str(rawpath),
        display_percentiles=[.5,99.5],display_range=[float(display_lo),float(display_hi)],spatial_crop=[x0,y0,x1,y1],
        inference_or_scoring_modified=False))
(O/'MANIFEST.json').write_text(json.dumps(dict(status='COMPLETE',all_original_five_fusion_losses_included=5,figures=manifest,
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
print(json.dumps(manifest))
