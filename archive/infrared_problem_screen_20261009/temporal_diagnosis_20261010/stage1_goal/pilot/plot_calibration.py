#!/usr/bin/env python3
"""Descriptive plot of all frozen-grid calibration points; no model selection."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P=Path(__file__).resolve().parents[1]
rows=list(csv.DictReader((P/'summary/CALIBRATION_CURVES.csv').open()))
variants=['official','paired_mean','paired_max','ohem','random_pair_max']
labels=['Official','Paired mean','Paired max (candidate)','OHEM','Random pair max']
colors=['#30343b','#3572b0','#c84e44','#359a70','#8a67ad']
fig,ax=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
budget=4015/320
summary=[]
for v,label,color in zip(variants,labels,colors):
    rr=[r for r in rows if r['variant']==v]
    xx=[int(r['unmatched_components'])/int(r['frames']) for r in rr]
    yy=[int(r['strict_small_matched'])/int(r['strict_small_gt']) for r in rr]
    ax[0].plot(xx,yy,'o-',markersize=3,lw=1,color=color,label=label)
    fixed=next(r for r in rr if float(r['threshold'])==.5)
    best=min(rr,key=lambda r:int(r['unmatched_components']))
    summary.append(dict(variant=v,at_half=fixed,min_unmatched_on_frozen_grid=best,
                        grid_has_feasible_point=any(int(r['unmatched_components'])<=4015 for r in rr)))
ax[0].axvline(budget,color='black',ls='--',lw=1)
ax[0].annotate('Official .5 budget',xy=(budget,.88),xytext=(budget*3,.90),fontsize=8,
               arrowprops=dict(arrowstyle='-',lw=.8,color='black'))
ax[0].set(xscale='log',xlabel='Unmatched components per frame (log scale)',
          ylabel='Strict-small diagnostic point match fraction',ylim=(0,1),
          title='All 23 frozen thresholds on calibration only')
ax[0].legend(fontsize=8,loc='upper left')
mins=[int(s['min_unmatched_on_frozen_grid']['unmatched_components'])/320 for s in summary[1:]]
bars=ax[1].bar(range(4),mins,color=colors[1:])
ax[1].axhline(budget,color='black',ls='--',lw=1,label=f'Official .5: {budget:.2f}')
ax[1].bar_label(bars,fmt='%.2f',padding=3,fontsize=9)
ax[1].set_xticks(range(4),['Paired\nmean','Paired\nmax','OHEM','Random\npair max'])
ax[1].set(ylabel='Unmatched components per frame',ylim=(0,max(mins)*1.18),
          title='Minimum reached by each trained variant\n(all occur at threshold 1.0)')
ax[1].legend(fontsize=8,loc='upper left')
for a in ax:
    a.spines[['right','top']].set_visible(False)
    a.grid(axis='y',alpha=.15)
fig.suptitle('Calibration failure: more target matches do not meet the background constraint',fontsize=12)
dest=P/'summary/figures';dest.mkdir(parents=True,exist_ok=True)
for ext in ['png','svg','pdf']:fig.savefig(dest/f'calibration_tradeoff.{ext}',dpi=180)
(P/'summary/CALIBRATION_DESCRIPTIVE.json').write_text(json.dumps(summary,indent=2)+'\n')
print(str(dest/'calibration_tradeoff.png'))
