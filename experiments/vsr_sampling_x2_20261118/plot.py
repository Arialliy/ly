"""Plot sequence-level effects. Uses SUMMARY.json from summarize.py."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


here=Path(__file__).resolve().parent
s=json.loads((here/'SUMMARY.json').read_text())
seqs=list(s['native']['per_sequence'])
pos=np.arange(len(seqs))
fig, ax=plt.subplots(1,2,figsize=(10,4.2),layout='constrained')
for cond,marker,color,label in [('clean','o','#007e87','Point samples'),
                                 ('blur_noise','s','#b35622','Blur and noise')]:
    series=s['controlled'][cond]['estimated']['per_sequence']
    ax[0].plot(pos,[series[k]['classic']['direction_db'] for k in seqs],marker=marker,
               color=color,linewidth=1.5,label=label)
ax[0].set(title='Controlled translations of one frame',ylabel='Directional error reduction (dB)')
ax[0].legend(frameon=False,fontsize=9)
ax[0].set_ylim(bottom=0)
base=s['native']['per_sequence']
delta=[base[k]['classic']['psnr']-base[k]['repeat']['psnr'] for k in seqs]
ax[1].bar(pos,delta,color=['#007e87' if d>0 else '#ae5133' for d in delta])
ax[1].axhline(0,color='#38424b',linewidth=.8)
ax[1].set(title='Five consecutive video frames',ylabel='Five-frame minus single-frame PSNR (dB)')
for panel in ax:
    panel.set_xticks(pos, [k.zfill(3) for k in seqs])
    panel.set_xlabel('REDS sequence')
    panel.grid(axis='y',color='#cccccc',alpha=.5,linewidth=.6)
    panel.set_axisbelow(True)
    panel.spines[['top','right']].set_visible(False)
fig.savefig(here/'sequence_results.png',dpi=170)
