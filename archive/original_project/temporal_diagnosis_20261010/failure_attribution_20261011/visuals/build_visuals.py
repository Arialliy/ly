#!/usr/bin/env python3
"""Deterministic scientific inspection panels, from existing image/probability arrays."""
import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
Q = HERE.parents[1]
P = Q / 'stage1_goal'
V = Q / 'problem_validity_20261011'
OLD = [0,36,72,108,120]
NEW = [0,24,60,96,120]


def readcsv(p):
    return list(csv.DictReader(p.open()))


def key(r):
    return (r['sequence'],int(r['frame']),int(r['annotation_id']))


def box(r):
    return [float(r['bbox_'+s]) for s in ['x','y','w','h']]


def crop_region(r, radius=24):
    x,y,w,h=box(r)
    cx,cy=int(x+w/2),int(y+h/2)
    return max(0,cx-radius),max(0,cy-radius),min(640,cx+radius),min(512,cy+radius)


def mark(ax,r):
    x,y,w,h=box(r)
    ax.add_patch(Rectangle((x,y),w,h,fill=False,lw=1,color='lime'))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--extra-csv',type=Path);ap.add_argument('--only-extra',action='store_true');args=ap.parse_args()
    selection=json.loads((HERE/'SELECTION.json').read_text())
    lookup={key(r):r for r in readcsv(V/'OBJECTS.csv')}
    ws=defaultdict(list)
    for r in readcsv(V/'WINDOWS.csv'):
        if r['sequence'] in ['DJI_0026_2','DJI_0067_2']: ws[key(r)].append(r)
    windows={k:sorted(a,key=lambda r:int(r['window_start'])) for k,a in ws.items()}
    extra=[]
    if args.extra_csv:
        for r in readcsv(args.extra_csv):
            d=dict(lookup[key(r)]);d['transition']=r.get('group',r.get('cohort','all_failure_sample'));extra.append(d)
    figures=[]
    font=ImageFont.load_default()
    seqs=['DJI_0026_2','DJI_0067_2']
    for seq in seqs:
        raw=np.load(P/'data'/seq/'raw_frames.npy',mmap_mode='r')
        with np.load(P/'predictions/official'/seq/'official_fused.npz',allow_pickle=False) as z: old=z['probabilities']
        with np.load(V/'simple_controls/symmetric5_probmax'/seq/'fused.npz',allow_pickle=False) as z: new=z['probabilities']
        probs={}
        for st in range(0,121,12):
            with np.load(P/'predictions/official'/seq/'windows'/f'window_{st:03d}.npz',allow_pickle=False) as z: probs[st]=z['probabilities']
        allrows=[r for r in lookup.values() if r['sequence']==seq]
        tracks=defaultdict(list)
        for r in allrows:tracks[int(r['track_id'])].append(r)
        for a in tracks.values():a.sort(key=lambda r:int(r['frame']))

        if not args.only_extra:
            rr=[r for r in selection['all_transitions'] if r['sequence']==seq]
            for page in range(math.ceil(len(rr)/12)):
                batch=rr[page*12:(page+1)*12]
                canvas=Image.new('RGB',(880,54+len(batch)*132),'white');draw=ImageDraw.Draw(canvas)
                draw.text((8,5),f'{seq} all changed target-frames, sorted track/frame; green=GT; maps [0,1]',fill='black',font=font)
                for j,title in enumerate(['raw t-12','raw current','raw t+12','official max','symmetric max']):draw.text((200+j*132,27),title,fill='black',font=font)
                for ri,r in enumerate(batch):
                    t=int(r['frame']);x0,y0,x1,y1=crop_region(r);bx,by,bw,bh=box(r);yoff=54+ri*132
                    draw.text((5,yoff+6),f"{r['transition']} tr={r['track_id']}\nsrc={r['source_frame']} ann={r['annotation_id']}\nbox={bw:g}x{bh:g}",fill='black',font=font)
                    for j,(arr,isprob) in enumerate([(raw[max(0,t-12)],False),(raw[t],False),(raw[min(159,t+12)],False),(old[t],True),(new[t],True)]):
                        a=arr[y0:y1,x0:x1]
                        rgb=(plt.get_cmap('magma')(a)[...,:3]*255).astype(np.uint8) if isprob else np.repeat(a[...,None],3,axis=-1)
                        im=Image.fromarray(rgb.astype(np.uint8)).resize((128,128),resample=Image.Resampling.NEAREST)
                        dd=ImageDraw.Draw(im);sx=128/(x1-x0);sy=128/(y1-y0)
                        # Neighbor-time panels show current-frame coordinates deliberately, not tracked GT.
                        dd.rectangle(((bx-x0)*sx,(by-y0)*sy,(bx+bw-x0)*sx,(by+bh-y0)*sy),outline=(0,255,0),width=1)
                        canvas.paste(im,(200+j*132,yoff))
                name=f'{seq}_all_transitions_{page+1:02d}.png';canvas.save(HERE/name);figures.append(name)

        reps=[] if args.only_extra else selection['original_fusion_losses']+selection['track_state_representatives']
        reps += extra
        unique={key(r):r for r in reps if r['sequence']==seq}
        for r in unique.values():
            t=int(r['frame']);aid=int(r['annotation_id']);tr=int(r['track_id']);x0,y0,x1,y1=crop_region(r,32)
            bx,by,bw,bh=box(r);win=windows[key(r)]
            successful=[w for w in win if int(w['matched'])]
            failure=[w for w in win if not int(w['matched'])]
            a=successful[0] if successful else win[0]
            b=failure[0] if failure else win[-1]
            sa,sb=int(a['window_start']),int(b['window_start'])
            fig,axs=plt.subplots(2,4,figsize=(14,7.5))
            tiles=[(raw[t],'raw current',False),(old[t],'official fused',True),(new[t],'symmetric fused',True),
                   (probs[sa][t-sa],f"window {sa}, slot {t-sa}, match={a['matched']}",True),
                   (probs[sb][t-sb],f"window {sb}, slot {t-sb}, match={b['matched']}",True)]
            for ax,(arr,title,isprob) in zip(axs.ravel()[:5],tiles):
                ax.imshow(arr,cmap='magma' if isprob else 'gray',vmin=0,vmax=1 if isprob else 255,interpolation='nearest',extent=(0,640,512,0))
                if isprob and np.any(arr[y0:y1,x0:x1]>=.5) and np.any(arr[y0:y1,x0:x1]<.5):
                    ax.contour(np.arange(640)+.5,np.arange(512)+.5,arr,levels=[.5],colors=['cyan'],linewidths=.4)
                mark(ax,r);ax.set_xlim(x0,x1);ax.set_ylim(y1,y0);ax.set_title(title,fontsize=9);ax.tick_params(labelsize=7)
            ax=axs[1,1]
            cx=min(639,max(0,int(bx+bw/2)));cy=min(511,max(0,int(by+bh/2)))
            ax.plot(range(160),raw[:,cy,cx],color='black',lw=.8,label='fixed center pixel (raw 0..255)')
            tx=[];ty=[]
            for o in tracks[tr]:
                tt=int(o['frame']);xx,yy,ww,hh=box(o);px=max(0,min(639,int(xx+ww/2)));py=max(0,min(511,int(yy+hh/2)))
                tx.append(tt);ty.append(int(raw[tt,py,px]))
            ax.plot(tx,ty,color='tab:orange',lw=.8,label='annotated moving center')
            ax.axvline(t,color='red',lw=.7);ax.set_xlim(0,159);ax.set_ylim(0,255);ax.set_title('Raw temporal traces; no visibility inference',fontsize=9);ax.legend(fontsize=6);ax.tick_params(labelsize=7)
            ax=axs[1,2];vals=np.array([[float(o['bbox_x'])+float(o['bbox_w'])/2,float(o['bbox_y'])+float(o['bbox_h'])/2] for o in tracks[tr]])
            ax.scatter(vals[:,0],vals[:,1],c=tx,s=7,cmap='viridis');ax.scatter([bx+bw/2],[by+bh/2],color='red',s=25)
            ax.invert_yaxis();ax.set_aspect('equal',adjustable='datalim');ax.set_title('Annotated image-coordinate trajectory',fontsize=9);ax.tick_params(labelsize=7)
            ax=axs[1,3]
            for y,starts,color,label in [(1,OLD,'tab:blue','official'),(2,NEW,'tab:orange','symmetric')]:
                covered=[s for s in starts if s<=t<s+40]
                for s in covered:ax.plot([s,s+39],[y,y],lw=6,alpha=.4,color=color)
                ax.text(0,y+.16,','.join(map(str,covered)),fontsize=7)
            ax.axvline(t,color='red',lw=.7);ax.set_xlim(0,159);ax.set_ylim(.5,2.7);ax.set_yticks([1,2]);ax.set_yticklabels(['official','symmetric']);ax.set_title('Reference access for this current frame',fontsize=9);ax.tick_params(labelsize=7)
            fig.suptitle(f"{seq} {r.get('transition','sample')} track={tr} source={r['source_frame']} ann={aid} bbox={bw:g}x{bh:g}\nGreen: current bbox; cyan: probability >=0.5 boundary; first successful/first failed diagnostic windows, no best-score selection",fontsize=10)
            fig.tight_layout(rect=(0,0,1,.93))
            name=f'{seq}_track{tr}_frame{t:03d}_ann{aid}.png';fig.savefig(HERE/name,dpi=130);plt.close(fig);figures.append(name)
        print(json.dumps({'sequence':seq,'detail_objects':len(unique),'saved_total':len(figures)}),flush=True)
        del probs,old,new
    target='EXTRA_FIGURES.json' if args.only_extra else 'FIGURES.json'
    (HERE/target).write_text(json.dumps({'figures':figures,'extra_csv':str(args.extra_csv) if args.extra_csv else None,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'limits':'Green boxes at neighboring times in atlas use fixed current coordinates. No human visibility label implied. Raw traces are 8-bit image values, not radiometry.'},indent=2)+'\n')
    html=['<!doctype html><meta charset="utf-8"><title>Failure diagnostic gallery</title><h1>Failure diagnostic gallery</h1><p>Scientific inspection panels. All changed target-frames plus deterministic track-state representatives. See selection and review records for scope.</p>']
    for name in sorted(p.name for p in HERE.glob('*.png')):html.append(f'<h3>{name}</h3><a href="{name}"><img src="{name}" style="max-width:100%"></a>')
    (HERE/'index.html').write_text('\n'.join(html))


if __name__=='__main__':main()
