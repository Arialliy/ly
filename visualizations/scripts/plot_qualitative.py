"""Render fixed real-frame overlays and middle-frame crops from saved arrays."""
import json
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

V=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'savefig.dpi':300,
                     'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
GREEN='#009E73';ORANGE='#D55E00';CYAN='#56B4E9'

def save(fig,name):
    for ext in ['png','pdf']:fig.savefig(V/f'{name}.{ext}',bbox_inches='tight',pad_inches=.08)
    plt.close(fig)

def main():
    plan=json.loads((V/'QUALITATIVE_SELECTION.json').read_text())
    objects=json.loads((V/'data/QUALITATIVE_OBJECTS.json').read_text())
    records=[]
    for seq in plan['sequences']:
        with np.load(V/'data'/f'{seq}_selected_frames.npz',allow_pickle=False) as z:sample={k:z[k] for k in z.files}
        for index,frame in enumerate(sample['frames']):
            obs=[r for r in objects if r['sequence']==seq and int(r['frame'])==int(frame)]
            fig,axes=plt.subplots(1,3,figsize=(12,3.7),layout='constrained')
            for ax,label in zip(axes,['raw','release','T']):
                ax.imshow(sample['raw'][index],cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                if label!='raw':
                    prob=sample[label][index]
                    if prob.min()<.5<prob.max():ax.contour(prob,levels=[.5],colors=[CYAN],linewidths=.55,alpha=.8)
                    for row in obs:
                        x,y,w,h=row['bbox'];matched=int(row[label+'_matched'])
                        ax.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor=GREEN if matched else ORANGE,linewidth=1.05))
                        if matched:ax.plot(float(row[label+'_peak_x']),float(row[label+'_peak_y']),marker='+',color=GREEN,markersize=3,markeredgewidth=.7)
                    count=sum(int(r[label+'_matched']) for r in obs)
                    ax.set_xlabel(('Released DeepPro-Plus' if label=='release' else 'T-LFP')+f'  |  matched {count}/{len(obs)}')
                else:ax.set_xlabel('Raw infrared frame; native 0–255 display')
                ax.set_xticks([]);ax.set_yticks([])
                for spine in ax.spines.values():spine.set_visible(False)
            legend=[Line2D([0],[0],color=GREEN,lw=1.5,label='Matched strict-small GT box'),
                    Line2D([0],[0],color=ORANGE,lw=1.5,label='Missed strict-small GT box'),
                    Line2D([0],[0],color=CYAN,lw=1.5,label='All foreground contours at 0.5')]
            fig.legend(handles=legend,loc='outside lower center',ncol=3,frameon=False,fontsize=8)
            name=f'frame_{seq}_{int(frame):03d}'
            save(fig,name)
            records.append(dict(name=name,sequence=seq,frame=int(frame),arrangement='O',strict_small_gt=len(obs),
                release_matched=sum(int(r['release_matched']) for r in obs),T_matched=sum(int(r['T_matched']) for r in obs)))
        chosen=next(r for r in plan['zoom_objects'] if r['sequence']==seq)
        row=next(r for r in objects if r['sequence']==seq and r['annotation_id']==chosen['annotation_id'])
        index=list(sample['frames']).index(80)
        x,y,w,h=map(float,row['bbox']);height,width=sample['raw'][index].shape
        left=max(0,min(width-96,int(math.floor(x+w/2-48))))
        top=max(0,min(height-96,int(math.floor(y+h/2-48))))
        crop=(slice(top,top+96),slice(left,left+96))
        fig,axes=plt.subplots(1,3,figsize=(8,2.8),layout='constrained')
        axes[0].imshow(sample['raw'][index][crop],cmap='gray',vmin=0,vmax=255,interpolation='nearest')
        axes[0].set_xlabel('Raw crop (96 × 96 pixels)')
        for ax,label in zip(axes[1:],['release','T']):
            im=ax.imshow(sample[label][index][crop],cmap='magma',vmin=0,vmax=1,interpolation='nearest')
            probability=float(row[label+'_box_max'])
            ax.set_xlabel(('Release' if label=='release' else 'T-LFP')+f'  |  box max {probability:.2e}')
        for ax in axes:
            ax.add_patch(Rectangle((x-left,y-top),w,h,fill=False,edgecolor=ORANGE,linewidth=1.3))
            # A visible plotting margin preserves coarse boxes at image boundaries.
            # The white margin is outside the sampled data, never synthetic pixels.
            ax.set_xlim(-2,98);ax.set_ylim(98,-2)
            ax.set_xticks([]);ax.set_yticks([])
            for spine in ax.spines.values():spine.set_visible(False)
        fig.colorbar(im,ax=axes[1:],shrink=.78,label='Predicted probability (fixed 0–1)')
        name=f'zoom_{seq}_080_track_{chosen["track_id"]}'
        save(fig,name)
        records.append(dict(name=name,sequence=seq,frame=80,track_id=int(chosen['track_id']),annotation_id=int(chosen['annotation_id']),
            crop_xywh=[left,top,96,96],bbox=row['bbox'],release_box_max=float(row['release_box_max']),T_box_max=float(row['T_box_max']),
            touches_original_image_boundary=bool(left==0 or top==0 or left+96==width or top+96==height),
            low_all_05=True,extreme_all_001=bool(int(row['extreme_all_001']))))
    (V/'QUALITATIVE_RENDER.json').write_text(json.dumps(dict(status='COMPLETE',figures=records,model_runs=0,
        caution='Ground-truth boxes are coarse annotations. Cyan contours include foreground belonging to all object sizes and are not themselves false-alarm labels. No new matching or physical-visibility judgment.',
        display='Raw uint8 range0..255; all probability crops share0..1; interpolation nearest. White crop margins are outside sampled image data; boundary boxes are preserved.'),indent=2)+'\n')
    print('Rendered six full-frame comparisons and two fixed low-response crops')

if __name__=='__main__':main()
