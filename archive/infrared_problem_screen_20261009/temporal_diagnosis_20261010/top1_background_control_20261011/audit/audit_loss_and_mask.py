#!/usr/bin/env python3
"""Independent closed-form loss/gradient checks and mask regression audit."""
import ast, datetime, hashlib, importlib.util, json, math, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

OUT=Path(__file__).resolve().parent; N=OUT.parent; Q=N.parent
OLD=Q/'low_response_information_20261011/lambda0_control'
checks={}
def check(name,x):
    assert x,name
    checks[name]=True
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def module(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
old=module('old_loss',OLD/'code/losses.py');new=module('new_loss',N/'code/losses.py')
trainer=module('new_trainer',N/'code/train_pilot.py')
sys.path.insert(0,str(N/'code'))
diag=module('new_diagnostics',N/'code/training_diagnostics.py')
oldtext=(OLD/'code/losses.py').read_text();newtext=(N/'code/losses.py').read_text()
expected=oldtext.replace('from __future__ import annotations','from __future__ import annotations\n\nimport math').replace('F.softplus(selected).mean() if selected.numel()','background_top1pct_mean(F.softplus(selected)) if selected.numel()')
new_ast=ast.parse(newtext);new_ast.body=[x for x in new_ast.body if getattr(x,'name',None)!='background_top1pct_mean']
check('only_optimization_change_is_per_window_unreduced_negative_topk',ast.dump(new_ast,include_attributes=False)==ast.dump(ast.parse(expected),include_attributes=False))
def astnode(text,name):
    return ast.dump(next(x for x in ast.parse(text).body if getattr(x,'name',None)==name),include_attributes=False)
ot=(OLD/'code/train_pilot.py').read_text();nt=(N/'code/train_pilot.py').read_text()
for name in ['pixel_rect','valid_box','positive_annotation','SequenceData','build_batch','load_model','main']:
    check(name+'_source_unchanged',astnode(ot,name)==astnode(nt,name))

# N=[0,1,101,299] distinguishes empty windows, ceil, per-window and global top-k.
# Every negative logit is below zero: selection must not discard probability<0.5.
for with_pos in [False,True]:
    raw=np.stack([np.linspace(-9+i*.4,-2+i*.2,300) for i in range(4)]).reshape(4,1,1,300)
    x=torch.tensor(raw,dtype=torch.float64,requires_grad=True)
    masks=torch.zeros_like(x,dtype=torch.bool);Ns=[0,1,101,299];ks=[0,1,2,3]
    for i,n in enumerate(Ns):masks[i,0,0,:n]=True
    boxes=[[(0,(299,0,300,1))] if with_pos else [] for _ in range(4)]
    specs=[{'pair_index':0,'slots':[0,0],'rect':(297,0,298,1)}, {'pair_index':1,'slots':[0,0],'rect':(295,0,296,1)}]
    y,parts=new.pilot_loss(x,boxes,masks,specs,'paired_max',[])
    negative=sum(sum(math.log1p(math.exp(float(raw[i,0,0,j]))) for j in range(n-k,n))/k if n else 0 for i,(n,k) in enumerate(zip(Ns,ks)))/4
    positive=sum(math.log1p(math.exp(-float(raw[i,0,0,299]))) for i in range(4))/4 if with_pos else 0
    check(f'closed_form_value_pos_{with_pos}',math.isclose(y.item(),positive+negative,rel_tol=1e-12))
    y.backward();want=np.zeros_like(raw)
    for i,(n,k) in enumerate(zip(Ns,ks)):
        if n:
            for j in range(n-k,n):want[i,0,0,j]=1/(1+math.exp(-float(raw[i,0,0,j])))/(4*k)
        if with_pos:want[i,0,0,299]=-1/(1+math.exp(float(raw[i,0,0,299])))/4
    check(f'closed_form_gradient_support_and_outer_equal_weights_pos_{with_pos}',np.allclose(x.grad.numpy(),want,atol=1e-14,rtol=1e-12))
    _,oldparts=old.pilot_loss(x.detach(),boxes,masks,specs,'paired_max',[])
    check(f'positive_and_unused_extra_unchanged_pos_{with_pos}',torch.equal(parts['positive'],oldparts['positive']) and torch.equal(parts['extra'],oldparts['extra']))
    check(f'extra_not_optimized_pos_{with_pos}',parts['extra'].item()>0 and math.isclose(y.item(),parts['positive'].item()+parts['negative'].item(),rel_tol=1e-12))
    grad_before=x.grad.clone();d=diag.background_diagnostics(x,masks)
    check(f'diagnostics_passive_pos_{with_pos}',torch.equal(grad_before,x.grad) and d['background_N_per_window']==Ns and d['background_k_per_window']==ks and math.isclose(d['background_top1pct_mean'],negative,rel_tol=1e-12))
    check(f'fullmean_diagnostics_reproduce_original_pos_{with_pos}',math.isclose(d['background_full_mean'],oldparts['negative'].item(),rel_tol=1e-12))

# Boundary ties are resolved in ORIGINAL flatten order, including sparse masks.
for pattern in ['all_equal','one_above_cutoff']:
    x=torch.full((4,1,1,500),-3.,dtype=torch.float64,requires_grad=True)
    masks=torch.zeros_like(x,dtype=torch.bool);masks[:,:, :,1:203:2]=True
    if pattern=='one_above_cutoff':
        with torch.no_grad():x[0,0,0,201]=-1.
    y,_=new.pilot_loss(x,[[],[],[],[]],masks,[],'paired_mean',[])
    y.backward();want=torch.zeros_like(x)
    for i in range(4):
        indices=[1,201] if i==0 and pattern=='one_above_cutoff' else [1,3]
        for j in indices:want[i,0,0,j]=1/(1+math.exp(-x.detach()[i,0,0,j].item()))/8
    check('sparse_mask_tie_gradient_'+pattern,torch.allclose(x.grad,want,rtol=1e-12,atol=1e-14))

# Synthetic labels check that large, ignored, crowd and other-category boxes are
# masked before positive filtering. Pixel-center geometry computed independently.
anns=[{'id':1,'bbox':[25.2,30.1,7.5,6.2],'category_id':1},
      {'id':2,'bbox':[55,55,22,25],'category_id':999},
      {'id':3,'bbox':[82.4,75.2,9,11],'ignore':1},
      {'id':4,'bbox':[20,90,15,8],'iscrowd':1},
      {'id':5,'bbox':[-10,8,25,15],'category_id':555}]
seq=SimpleNamespace(frames=np.zeros((160,128,128),np.uint8),annotations=[anns for _ in range(160)],strict=[[] for _ in range(160)])
entry={'pairs':[{'sequence':'synthetic','crop_yx':[0,0],'starts':[0,10],'anchor_frame':20}, {'sequence':'synthetic','crop_yx':[0,0],'starts':[30,40],'anchor_frame':45}], 'extra_specs':[]}
_,_,actual,_=trainer.build_batch(entry,{'synthetic':seq})
yy,xx=np.mgrid[:128,:128];xp=xx+.5;yp=yy+.5
oracle=(xx>=9)&(xx<119)&(yy>=9)&(yy<119)
for a in anns:
    x,y,w,h=a['bbox'];oracle&=~((xp>=x-2)&(xp<x+w+2)&(yp>=y-2)&(yp<y+h+2))
check('synthetic_all_annotation_classes_masked_with_2px_margin',np.array_equal(actual,np.broadcast_to(oracle,actual.shape)))

# Real sampling plan and raw annotation provenance is frozen, not response-filtered.
plan=json.loads((Q/'stage1_goal/pilot/TRAIN_PLAN.json').read_text())
check('frozen_plan_512_seed46',len(plan['entries'])==512 and plan['seed']==46)
protocol=json.loads((N/'training/TRAINING_PROTOCOL.json').read_text())
check('declared_sources_unchanged',all(sha(p)==v for p,v in protocol['sources_sha256'].items()))
check('audited_code_matches_prelaunch_frozen_hashes',all(sha(p)==v for p,v in protocol['code_sha256'].items()))
annotations_checked=0
for seqname,fp in plan['data_fingerprints'].items():
    ap=Q/'stage1_goal/data'/seqname/'annotations.json'
    check(seqname+'_annotation_sha_frozen',sha(ap)==fp['annotations_sha256'])
    annotations_checked+=len(json.loads(ap.read_text())['annotations'])
result={'status':'PASS','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checks':checks,'mixed_background_counts':[0,1,101,299], 'expected_selected_counts':[0,1,2,3], 'all_test_negative_logits_below_zero':True,'train_annotations_fingerprint_checked':annotations_checked,'scope':'No additional model training. Exact loss source delta, closed-form values and gradients, passive telemetry, unchanged all-label mask source plus independent pixel-center synthetic mask.'}
(OUT/'LOSS_MASK_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
