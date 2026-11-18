#!/usr/bin/env python3
"""Render saved diagnostics only: no evaluation or model calls."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullLocator


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out=Path(__file__).resolve().parent;Q=out.parent
    response_path=Q/'window_offset_experiment/analysis/DJI_0047_3/same_gt_frame_context_summary.csv'
    contrast_path=Q/'motion/DJI_0047_3/TRACK_FRAME_DIAGNOSTICS.csv'
    with response_path.open() as f:
        rows=sorted((r for r in csv.DictReader(f) if int(r['track_id'])==4621500 and float(r['threshold'])==.5),key=lambda r:int(r['source_frame']))
    with contrast_path.open() as f:
        contrast={int(r['raw_frame']):r for r in csv.DictReader(f) if int(r['track_id'])==4621500}
    frames=np.array([int(r['source_frame']) for r in rows])
    assert np.array_equal(frames,np.arange(160))
    original=np.array([float(r['original_fused_boxmax']) for r in rows])
    envelope=np.array([float(r['boxmax_max']) for r in rows])
    contexts=np.array([int(r['context_count']) for r in rows])
    contrast_valid=np.array([contrast[f]['contrast_valid'].lower()=='true' for f in frames])
    difference=np.array([float(contrast[f]['mean_difference']) if contrast_valid[i] else np.nan for i,f in enumerate(frames)])
    rescue=np.array([int(r['oracle_recovery_requires_new_context']) and 18<=int(r['source_frame'])<=159 for r in rows],dtype=bool)
    assert list(frames[rescue])==[53,54,55,56]
    assert np.all(envelope>=original)
    assert np.all(contexts[148:]==1)

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
        'axes.labelsize':10,'legend.fontsize':9,'axes.spines.top':False,
        'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(3,1,figsize=(11.8,8.4),sharex=True,
        gridspec_kw={'height_ratios':[2.2,1.15,.95],'hspace':.31})
    ax=axes[0]
    ax.axvspan(17.5,159.5,color='#eef0f4',zorder=0)
    ax.axvspan(52.5,56.5,color='#cdebd8',alpha=.7,zorder=1)
    ax.plot(frames,envelope,color='#d77820',lw=1.8,label='Any of 11 real windows: oracle response envelope',zorder=3)
    ax.plot(frames,original,color='#225ea8',lw=1.55,label='Original 5-window max fusion',zorder=4)
    ax.scatter(frames[rescue],envelope[rescue],color='#208453',s=27,zorder=5,label='New-window point matches: frames 53–56')
    ax.axhline(.5,color='#666666',ls='--',lw=.9)
    ax.text(158,.5,' 0.5',ha='right',va='bottom',fontsize=9,color='#555555')
    ax.set_yscale('log');ax.set_ylim(1e-9,1.6)
    ax.set_yticks([1e-8,1e-6,1e-4,1e-2,1]);ax.yaxis.set_minor_locator(NullLocator())
    ax.set_ylabel('Maximum probability\ninside annotated bbox')
    ax.set_title('A  Saved target response — same image, different real temporal contexts',loc='left',pad=9)
    ax.legend(loc='lower left',frameon=True,framealpha=.96,edgecolor='#dddddd')
    ax.annotate('Original continuous low-response interval: frames 18–159',
        xy=(104,2e-4),xytext=(78,4e-3),fontsize=9,color='#555555',ha='center',
        arrowprops={'arrowstyle':'-','color':'#999999','lw':.8})
    ax.grid(axis='y',color='#dddddd',lw=.6,alpha=.6)

    ax=axes[1]
    ax.axvspan(17.5,159.5,color='#eef0f4',zorder=0)
    ax.plot(frames,difference,color='#705b96',lw=1.45)
    ax.axhline(0,color='#666666',ls='--',lw=.85)
    ax.set_title('B  Local contrast proxy from bbox and surrounding ring',loc='left',pad=8)
    ax.set_ylabel('BBox mean − ring mean\n(raw grayscale units)')
    ax.grid(axis='y',color='#dddddd',lw=.6,alpha=.6)
    ax.text(.99,.92,'BBox proxy; not target SNR or verified visibility',transform=ax.transAxes,
        fontsize=8.5,color='#555555',ha='right',va='top')

    ax=axes[2]
    ax.step(frames,contexts,where='mid',color='#34495e',lw=1.8)
    ax.fill_between(frames,0,contexts,step='mid',alpha=.12,color='#34495e')
    ax.axvspan(147.5,159.5,color='#f7d9d2',zorder=0)
    ax.set_yticks([1,2,3,4]);ax.set_ylim(.5,4.5)
    ax.set_ylabel('Available real\n40-frame contexts')
    ax.set_title('C  Number of evaluated contexts for each original frame',loc='left',pad=8)
    ax.annotate('Frames 148–159:\nonly one context',xy=(154,1.05),xytext=(132,3.1),
        arrowprops={'arrowstyle':'->','lw':.8,'color':'#8b4336'},color='#8b4336',fontsize=9,ha='center')
    ax.set_xlim(-.5,159.5);ax.set_xticks(np.arange(0,161,20));ax.set_xlabel('Original frame index')
    fig.suptitle('DeepPro-Plus on DJI_0047_3 · annotated track 4621500',fontsize=14,fontweight='bold',y=.98)
    fig.text(.5,.946,'Frozen checkpoint and seed 46 · existing CSV diagnostics only',ha='center',fontsize=10,color='#555555')
    fig.subplots_adjust(top=.888,bottom=.105,left=.105,right=.98)
    fig.text(.105,.042,'Oracle diagnostic, not a method improvement: extra windows increase computation and use a GT-defined response envelope.',fontsize=9,color='#555555')
    fig.text(.105,.024,'The window intervention changes surrounding content and time slot together. Bbox response is not official Pd / Fa.',fontsize=9,color='#555555')
    png=out/'selected_0047_diagnostic.png';pdf=out/'selected_0047_diagnostic.pdf'
    fig.savefig(png,dpi=190,facecolor='white');fig.savefig(pdf,facecolor='white');plt.close(fig)
    details={'track_id':4621500,'sequence':'DJI_0047_3','frame_range':[0,159],
        'input_sha256':{str(response_path):sha(response_path),str(contrast_path):sha(contrast_path)},
        'script_sha256':sha(Path(__file__)),'output_sha256':{png.name:sha(png),pdf.name:sha(pdf)},
        'valid_contrast_frames':int(contrast_valid.sum()),'new_recovered_frames_at_0_5':frames[rescue].tolist(),
        'log_y_limits':[1e-9,1.6],'data_transform':'none; log axis only; invalid contrast shown as NaN gap',
        'scope':'Plot of saved CSVs only; no new metrics, model, inference, thresholds or sample selection.'}
    (out/'selected_0047_diagnostic.json').write_text(json.dumps(details,indent=2)+'\n')
    print(json.dumps(details,indent=2))


if __name__=='__main__':main()
