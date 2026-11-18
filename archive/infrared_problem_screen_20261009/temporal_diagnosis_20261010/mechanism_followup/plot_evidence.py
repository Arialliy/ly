from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
Q=Path(__file__).resolve().parent
with (Q/'SUPPORT_PAIRED.csv').open() as f:rows=list(csv.DictReader(f))
strong=[r for r in rows if int(r['strong_response_flip'])]
suppress=[r for r in strong if int(r['a_suppressed_common_strong'])]
need=[r for r in strong if int(r['b_requires_exclusive_strong'])]
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'font.family':'DejaVu Sans'})
fig,axs=plt.subplots(1,3,figsize=(13.4,4.4))
ax=axs[0];names=['Slot coefficients','Shared features','Exclusive context'];vals=[14,25,58]
bars=ax.barh(names,vals,color=['#718096','#DAA249','#397D91']);ax.invert_yaxis();ax.set_xlim(0,70);ax.set_xlabel('Strong-flip target frames (n=97)');ax.set_title('a  Largest contribution to response gap',loc='left',weight='bold',fontsize=11)
for b,v in zip(bars,vals):ax.text(v+1,b.get_y()+b.get_height()/2,str(v),va='center')
ax.text(.02,-.26,'Frozen internal decomposition;\nnot a unique natural causal attribution.',transform=ax.transAxes,fontsize=9,color='#4a5568')
ax=axs[1];x=np.array([0,.25,.5,.75,1])
for rr,tag,color,label in [(suppress,'a','#B7544D','Suppression cases (n=24)'),(need,'b','#397D91','Support-dependent cases (n=42)')]:
 y=np.array([[float(r[f'{tag}_lambda_{s}_box_max_probability']) for s in ['000','025','050','075','100']] for r in rr]);ax.plot(x,y.mean(0),marker='o',color=color,label=label)
ax.axhline(.5,color='#A0AEC0',ls=':',lw=1);ax.set_ylim(-.04,1.04);ax.set_xticks(x);ax.set_xlabel('Weight on exclusive projected contributions');ax.set_ylabel('Mean box-maximum response');ax.legend(frameon=False,fontsize=8,loc='center left');ax.set_title('b  Opposite roles of additional context',loc='left',weight='bold',fontsize=11)
ax.text(.02,-.26,'Groups selected by endpoint responses.\nInternal ablation; no deployment gain claimed.',transform=ax.transAxes,fontsize=9,color='#4a5568')
ax=axs[2];v=[37,11,49];labels=['Both high','Both low','Still split'];c=['#397D91','#B7544D','#718096'];bars=ax.bar(labels,v,color=c);ax.set_ylim(0,60);ax.set_ylabel('Target frames');ax.set_title('c  Restoring real temporal neighbors',loc='left',weight='bold',fontsize=11)
for b,n in zip(bars,v):ax.text(b.get_x()+b.get_width()/2,n+1,str(n),ha='center')
ax.text(.02,-.26,'97 original strong-flip window pairs.\nAgreement alone does not mean recovery.',transform=ax.transAxes,fontsize=9,color='#4a5568')
fig.subplots_adjust(wspace=.48,left=.115,right=.985,top=.85,bottom=.27)
fig.suptitle('DeepPro-Plus: result-driven diagnosis of temporal-context instability',x=.06,ha='left',fontsize=14,weight='bold')
fig.savefig(Q/'EVIDENCE.png',dpi=200,bbox_inches='tight');fig.savefig(Q/'EVIDENCE.pdf',bbox_inches='tight');plt.close(fig)
print(Q/'EVIDENCE.png')
