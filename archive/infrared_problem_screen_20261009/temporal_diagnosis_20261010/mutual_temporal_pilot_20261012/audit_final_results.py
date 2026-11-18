#!/usr/bin/env python3
"""Independent CPU audit of final results; no imports from experiment scorers.

Frozen sample: frames0,39,80,120,159, both clips, O/S, all5 final conditions.
Components use handwritten flood fill; matching uses iterative DFS augmenting
paths. This checks the prescribed deterministic rule by a different implementation.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np

BASE=Path(__file__).resolve().parent
Q=BASE.parent
CONDITIONS=('E0','E1','E2','E2_mass','E2_current')
SEQUENCES=('DJI_0026_2','DJI_0067_2')
FRAMES=(0,39,80,120,159)
PLANS={'O':(0,36,72,108,120),'S':(0,24,60,96,120)}
FIELDS=('gt_count','matched_count','component_count','unmatched_components',
        'positive_pixels','positive_pixels_outside_boxes')
CHECKS=[]
SOURCE_HASHES={}


def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):result.update(chunk)
    return result.hexdigest()


def check(name,condition,**details):
    CHECKS.append(dict(name=name,passed=bool(condition),**details))
    if not condition:raise AssertionError(name+': '+str(details))


def tracked_hash(path):
    path=Path(path)
    value=digest(path)
    SOURCE_HASHES[str(path)]=value
    return value


def read_json(path):
    tracked_hash(path)
    return json.loads(Path(path).read_text())


def crop_indices(box,h,w):
    x,y,bw,bh=box
    return (min(w,max(0,math.ceil(x-.5))), min(h,max(0,math.ceil(y-.5))),
            min(w,max(0,math.ceil(x+bw-.5))), min(h,max(0,math.ceil(y+bh-.5))))


def annotations_from_json(path):
    data=read_json(path)
    images=data['images']
    by_image=defaultdict(list)
    for a in data['annotations']:by_image[a['image_id']].append(a)
    all_gt=[];flat=[];offsets=[0]
    frame_numbers=[int(Path(im['file_name']).stem) for im in images]
    check(str(path)+' frameorder',frame_numbers==sorted(frame_numbers) and len(images)==160)
    for t,im in enumerate(images):
        h,w=im['height'],im['width']
        good=[]
        for a in sorted(by_image[im['id']],key=lambda v:(v['track_id'],v['id'])):
            box=list(map(float,a['bbox']));x,y,bw,bh=box
            if a.get('ignore',0) or bw<=0 or bh<=0:continue
            if min(float(w),x+bw)<=max(0.,x) or min(float(h),y+bh)<=max(0.,y):continue
            good.append(dict(a,bbox=box,crop=crop_indices(box,h,w)))
        check(f'{path.name} frame{t} uniquetracks',len({a['track_id'] for a in good})==len(good))
        for a in good:
            x,y,bw,bh=a['bbox']
            overlap=False
            for b in good:
                if b['id']==a['id']:continue
                xx,yy,ww,hh=b['bbox']
                if min(x+bw,xx+ww)>max(x,xx) and min(y+bh,yy+hh)>max(y,yy):overlap=True
            outside=x<0 or y<0 or x+bw>w or y+bh>h
            flat.append(dict(frame=t,source_frame=frame_numbers[t],image_id=im['id'],
                track_id=a['track_id'],annotation_id=a['id'],category_id=a['category_id'],
                bbox=a['bbox'],strict_small=int(max(bw,bh)<=16 and not outside and not overlap)))
        all_gt.append(good);offsets.append(len(flat))
    return all_gt,flat,np.asarray(offsets)


def flood_components(probability):
    h,w=probability.shape
    mask=probability>=np.float32(.5)
    flat_mask=mask.ravel()
    labels=np.zeros(h*w,np.int32)
    comps=[]
    for seed in np.flatnonzero(flat_mask):
        seed=int(seed)
        if labels[seed]:continue
        label=len(comps)+1
        stack=[seed];labels[seed]=label
        size=0;peak=seed;best=float(probability.ravel()[seed])
        while stack:
            pixel=stack.pop();size+=1
            score=float(probability.ravel()[pixel])
            if score>best or (score==best and pixel<peak):best,peak=score,pixel
            y,x=divmod(pixel,w)
            for yy in range(max(0,y-1),min(h,y+2)):
                for xx in range(max(0,x-1),min(w,x+2)):
                    candidate=yy*w+xx
                    if flat_mask[candidate] and labels[candidate]==0:
                        labels[candidate]=label;stack.append(candidate)
        py,px=divmod(peak,w)
        comps.append(dict(component_id=label,area=size,peak_x=px+.5,peak_y=py+.5,peak_probability=best))
    return comps,mask,labels.reshape(h,w)


def iterative_matching(edges):
    """Explicit DFS stack preserving the fixed edge order; no recursive scorer."""
    owner={}
    for root in range(len(edges)):
        seen=set();stack=[[root,0,None]]
        while stack:
            target,next_edge,incoming=stack[-1]
            if next_edge>=len(edges[target]):stack.pop();continue
            component=edges[target][next_edge]
            stack[-1][1]+=1
            if component in seen:continue
            seen.add(component)
            if component not in owner:
                owner[component]=target
                for depth in range(len(stack)-1,0,-1):
                    owner[stack[depth][2]]=stack[depth-1][0]
                break
            stack.append([owner[component],0,component])
    return owner


def independent_frame(probability,gt):
    comps,mask,labels=flood_components(probability)
    edges=[]
    for ann in gt:
        x,y,w,h=ann['bbox']
        candidates=[j for j,c in enumerate(comps) if x<=c['peak_x']<x+w and y<=c['peak_y']<y+h]
        candidates.sort(key=lambda j:((((comps[j]['peak_x']-x-w/2)/w)**2+
                                      ((comps[j]['peak_y']-y-h/2)/h)**2),
                                     -comps[j]['peak_probability'],comps[j]['component_id']))
        edges.append(candidates)
    owner=iterative_matching(edges)
    assigned={target:component for component,target in owner.items()}
    inside=np.zeros_like(mask)
    rows=[]
    for j,ann in enumerate(gt):
        x0,y0,x1,y1=ann['crop'];inside[y0:y1,x0:x1]=True
        patch=probability[y0:y1,x0:x1]
        component=comps[assigned[j]] if j in assigned else None
        rows.append(dict(matched=int(component is not None),
            box_max=float(patch.max()) if patch.size else float('nan'),
            peak_x=component['peak_x'] if component else float('nan'),
            peak_y=component['peak_y'] if component else float('nan'),
            component_area=component['area'] if component else 0))
    counts=np.asarray([len(gt),len(assigned),len(comps),len(comps)-len(assigned),
                      int(mask.sum()),int((mask&~inside).sum())],np.int64)
    areas=np.asarray([c['area'] for c in comps],np.int32)
    unmatched=np.asarray([j not in owner for j in range(len(comps))],np.uint8)
    return rows,counts,areas,unmatched


def self_test():
    prob=np.zeros((6,8),np.float32)
    prob[0,0]=.8;prob[1,1]=.8;prob[1,2]=.7;prob[4,6]=.9
    comp,mask,labels=flood_components(prob)
    check('self 8connected diagonal',len(comp)==2 and [c['area'] for c in comp]==[3,1])
    check('self rowmajor maximum tie',comp[0]['peak_x']==.5 and comp[0]['peak_y']==.5)
    check('self stable component order',labels[0,0]==1 and labels[4,6]==2)
    check('self augment reroute',iterative_matching([[0,1],[0]])=={0:1,1:0})
    check('self augment longer path',iterative_matching([[0,1],[1,2],[0]])=={0:2,1:0,2:1})
    prob.fill(0);prob[1,1]=.9;prob[1,4]=.8
    gt=[dict(bbox=[0.,0.,6.,4.],crop=(0,0,6,4)),dict(bbox=[0.,0.,3.,4.],crop=(0,0,3,4))]
    rows,c,a,u=independent_frame(prob,gt)
    check('self full matching reroute',c.tolist()==[2,2,2,0,2,0] and rows[0]['peak_x']==4.5 and rows[1]['peak_x']==1.5)
    prob[1,4]=.49999997
    rows,c,a,u=independent_frame(prob,gt)
    check('self threshold fixed >=0.5',c[2]==1 and c[1]==1)


def metadata_audit(condition,sequence):
    import torch
    infroot=BASE/'predictions'/condition/sequence
    eroot=BASE/'evaluation'/condition/sequence
    prediction=read_json(infroot/'COMPLETE.json')
    evaluation=read_json(eroot/'COMPLETE.json')
    variant=condition[:2]
    diag=condition.split('_',1)[1] if '_' in condition else 'none'
    run=read_json(BASE/'training'/variant/'run/RUN.json')
    checkpoint_path=Path(prediction['checkpoint'])
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    snap=next(x for x in run['snapshots'] if x['step']==512)
    prefix=condition+'/'+sequence
    check(prefix+' final512 identity',run['status']=='complete' and run['steps']==run['final_step']==512 and
          checkpoint['step']==512 and prediction['checkpoint_step']==512 and prediction['checkpoint_is_main'] and
          prediction['training_variant']==variant and checkpoint['variant']==variant and run['variant']==variant)
    check(prefix+' seed46 only',run['seed']==prediction['seed']==46)
    check(prefix+' diagnostic identity',prediction['diagnostic']==diag and prediction['condition']==condition)
    check(prefix+' checkpoint bytes',tracked_hash(checkpoint_path)==snap['sha256']==run['final_checkpoint_sha256']==prediction['checkpoint_sha256'])
    check(prefix+' original release identity',checkpoint['initial_checkpoint_sha256']==run['checkpoint_sha256']==prediction['reference_checkpoint_sha256']=='4fd4aa2736d9a33f2bf1459099068d8b4a7e202ff80aecdb4a3277fd13c7abcc')
    check(prefix+' common supervision',checkpoint['effective_supervision_config']==run['effective_supervision_config'] and
          'top1pct' in checkpoint['effective_supervision_config']['empty_window'] and
          checkpoint['label_config']==run['label_config']==prediction['label_config'])
    check(prefix+' BN no drift',run['bn_initial_sha256']==run['bn_final_sha256'])
    check(prefix+' fixed normalization',prediction['normalization']==[85.0799,47.4845])
    check(prefix+' original spatial treatment',prediction['tile_core']==128 and prediction['tile_halo']==20)
    check(prefix+' raw uint8 input fingerprint',prediction['raw_sha256']==tracked_hash(Q/'stage1_goal/data'/sequence/'raw_frames.npy'))
    check(prefix+' starts and fusion',prediction['official_starts']==list(PLANS['O']) and
          prediction['diagnostic_starts']==sorted(set(PLANS['O']+PLANS['S'])) and
          prediction['fusion']=='pixelwise probability maximum over official starts only')
    check(prefix+' source threshold',evaluation['threshold']==.5 and evaluation['checkpoint_step']==512 and evaluation['main_endpoint'])
    check(prefix+' labels not model inputs',prediction['model_labels_access'] is False)
    check(prefix+' expected capacity',prediction['parameters']==run['parameter_count']==(70913 if variant=='E0' else 72194))
    check(prefix+' tiled threshold equivalence',len(prediction['tiling_checks'])==3 and all(c['threshold05_disagreements']==0 for c in prediction['tiling_checks']))
    for key,path in [('train_source',BASE/'code/train_mutual.py'),('module_source',BASE/'code/module.py'),
                     ('label_source',Q/'supervision_alignment_control_20261012/code/gaussian_labels.py'),
                     ('top1_loss_source',Q/'top1_background_control_20261011/code/losses.py'),
                     ('prior_train_source',Q/'supervision_alignment_control_20261012/code/train_supervision.py'),
                     ('model_source',Path(prediction['model_source']))]:
        actual=tracked_hash(path)
        check(prefix+' '+key,checkpoint[key+'_sha256']==run[key+'_sha256']==prediction[key+'_sha256']==actual)
    check(prefix+' inference source',prediction['script_sha256']==tracked_hash(BASE/'code/infer_mutual.py'))
    # This is a source/record check of sigmoid placement, not a new model forward.
    source=(BASE/'code/infer_mutual.py').read_text()
    check(prefix+' sigmoid applied after logits and before fusion',
          'probabilities = logits.sigmoid()[0].numpy()' in source and
          'np.maximum(fused[start:start+40], probabilities)' in source)
    return prediction,evaluation


def audit_pair(condition,sequence,annotation_cache):
    prefix=condition+'/'+sequence
    prediction,receipt=metadata_audit(condition,sequence)
    allgt,meta,offsets=annotation_cache[sequence]
    check(prefix+' independent GT metadata',receipt['objects']==meta)
    eroot=BASE/'evaluation'/condition/sequence
    for name in ('scores.npz','component_areas.npz','FRAME_COUNTS.csv'):tracked_hash(eroot/name)
    with np.load(eroot/'scores.npz',allow_pickle=False) as z:scores={k:z[k] for k in z.files}
    with np.load(eroot/'component_areas.npz',allow_pickle=False) as z:areas={k:z[k] for k in z.files}
    check(prefix+' saved score threshold',float(scores['threshold'])==.5 and scores['plans'].tolist()==list(PLANS))
    check(prefix+' object offsets',np.array_equal(scores['offsets'],offsets))
    check(prefix+' strictsmall membership',np.array_equal(scores['strict_small'],[m['strict_small'] for m in meta]))
    infroot=BASE/'predictions'/condition/sequence
    fused={p:{t:np.zeros((512,640),np.float32) for t in FRAMES} for p in PLANS}
    coverage={p:np.zeros(160,np.int16) for p in PLANS}
    for start in sorted(set(PLANS['O']+PLANS['S'])):
        path=infroot/'windows'/f'window_{start:03d}.npz'
        check(prefix+f' window{start} hash',tracked_hash(path)==prediction['window_sha256'][path.name])
        with np.load(path,allow_pickle=False) as z:
            probabilities=z['probabilities']
            check(prefix+f' window{start} format',int(z['start'])==start and int(z['end'])==start+40 and
                  probabilities.shape==(40,512,640) and probabilities.dtype==np.float32 and
                  np.isfinite(probabilities).all() and probabilities.min()>=0 and probabilities.max()<=1)
            for plan,starts in PLANS.items():
                if start not in starts:continue
                coverage[plan][start:start+40]+=1
                for t in FRAMES:
                    if start<=t<start+40:np.maximum(fused[plan][t],probabilities[t-start],out=fused[plan][t])
        del probabilities
    officialpath=infroot/'official_fused.npz'
    check(prefix+' official fusion hash',tracked_hash(officialpath)==prediction['fused_sha256'])
    with np.load(officialpath,allow_pickle=False) as z:
        official=z['probabilities']
        check(prefix+' official starts and coverage',z['starts'].tolist()==list(PLANS['O']) and np.array_equal(z['coverage'],coverage['O']))
        for t in FRAMES:check(prefix+f' O frame{t} fusionbitexact',np.array_equal(official[t],fused['O'][t]))
    del official
    with (eroot/'FRAME_COUNTS.csv').open() as stream:
        csvrows={(r['arrangement'],int(r['frame'])):r for r in csv.DictReader(stream)}
    frame_records=[]
    for pi,plan in enumerate(PLANS):
        check(prefix+' '+plan+' coverage',np.array_equal(coverage[plan],scores['coverage'][pi]) and int(coverage[plan].sum())==200 and np.all(coverage[plan]>0))
        for t in FRAMES:
            name=f'{prefix}/{plan}/frame{t}'
            rows,counts,aa,uu=independent_frame(fused[plan][t],allgt[t])
            check(name+' six framecounts',np.array_equal(counts,scores['frame_counts'][pi,t]),actual=counts.tolist(),expected=scores['frame_counts'][pi,t].tolist())
            check(name+' CSVcounts',all(int(csvrows[plan,t][field])==int(counts[j]) for j,field in enumerate(FIELDS)))
            start,stop=map(int,offsets[t:t+2])
            for key in ('matched','box_max','peak_x','peak_y','component_area'):
                actual=np.asarray([r[key] for r in rows],dtype=scores[key].dtype)
                check(name+' objects '+key,np.array_equal(actual,scores[key][pi,start:stop],equal_nan=True))
            left,right=map(int,areas[plan+'_offsets'][t:t+2])
            check(name+' all component areas',np.array_equal(aa,areas[plan+'_areas'][left:right]))
            check(name+' all component assignmentflags',np.array_equal(uu,areas[plan+'_unmatched'][left:right]))
            check(name+' unmatched area total',int(aa[uu.astype(bool)].sum())==int(csvrows[plan,t]['unmatched_component_area']))
            frame_records.append(dict(condition=condition,sequence=sequence,arrangement=plan,frame=t,
                objects=len(rows),components=len(aa),matched=int(counts[1]),unmatched=int(counts[3]),
                foreground=int(counts[4]),outside=int(counts[5]),status='PASS'))
    return frame_records,scores,areas


def audit_summary(records):
    path=BASE/'evaluation/SUMMARY.csv'
    tracked_hash(path)
    with path.open() as stream:summary={(r['condition'],r['arrangement']):r for r in csv.DictReader(stream)}
    for condition in CONDITIONS:
        for pi,plan in enumerate(PLANS):
            counts=sum((records[condition,s][0]['frame_counts'][pi].sum(0) for s in SEQUENCES),np.zeros(6,np.int64))
            total_M=sum(int(r[0]['matched'][pi,r[0]['strict_small']].sum()) for key,r in records.items() if key[0]==condition)
            area_sum=sum(int(records[condition,s][1][plan+'_areas'][records[condition,s][1][plan+'_unmatched'].astype(bool)].sum()) for s in SEQUENCES)
            row=summary[condition,plan]
            check(condition+'/'+plan+' whole CSVtotals',int(row['M'])==total_M and all(int(row[f])==int(counts[j]) for j,f in enumerate(FIELDS)) and int(row['unmatched_component_area_sum'])==area_sum)
            check(condition+'/'+plan+' budget indicator',int(row['passes_pooled_U_budget'])==int(counts[3]<=4015) and int(row['U_budget'])==4015)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--self-test-only',action='store_true')
    args=parser.parse_args()
    started=time.perf_counter();self_test()
    if args.self_test_only:
        print(json.dumps(dict(status='PASS',checks=len(CHECKS),scope='independent auditor artificial cases')));return
    result=dict(status='RUNNING',seed=46,conditions=list(CONDITIONS),sequences=list(SEQUENCES),frames=list(FRAMES),plans=PLANS,
                sample_rule='frozen before observing new results',scope='100 frame-plan-condition checks; not a full independent 3200frame recomputation',
                no_scorer_imports=True,no_new_training_or_inference=True)
    try:
        freeze=read_json(BASE/'PROTOCOL_FREEZE.json')
        check('frozen protocol unchanged',tracked_hash(BASE/'PROTOCOL.md')==freeze['protocol_sha256'])
        for name,value in freeze['code_sha256'].items():check('frozen '+name,tracked_hash(BASE/name)==value)
        annotation_cache={s:annotations_from_json(Q/'stage1_goal/data'/s/'annotations.json') for s in SEQUENCES}
        frames=[];records={}
        for condition in CONDITIONS:
            for sequence in SEQUENCES:
                fr,sc,ar=audit_pair(condition,sequence,annotation_cache)
                frames.extend(fr);records[condition,sequence]=(sc,ar)
                print(json.dumps(dict(audited=condition+'/'+sequence,checks=len(CHECKS))),flush=True)
        audit_summary(records)
        check('exact frozen frame count',len(frames)==100)
        result.update(status='PASS',frame_records=frames,independently_recomputed_frames=len(frames))
    except BaseException as error:
        result.update(status='FAIL',error=repr(error))
        raise
    finally:
        result.update(checks_passed=sum(x['passed'] for x in CHECKS),checks=CHECKS,
                      source_sha256=SOURCE_HASHES,script_sha256=digest(__file__),
                      elapsed_seconds=time.perf_counter()-started,
                      limitations=['Selected frames independently recomputed; full totals checked for internal CSV/cache consistency.',
                      'Probability/sigmoid/input ordering verified from source and recorded provenance; no new GPU forward.',
                      'Single seed46, two development clips; no cross-seed or unseen-source generalization claim.'])
        (BASE/'RESULTS_AUDIT.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
        text='# 独立结果审计\n\n'+result['status']+'。固定seed46；E0/E1/E2/E2_mass/E2_current，两个片段，O/S，固定帧0、39、80、120、159。\n\n独立实现8连通洪泛、峰值行优先打破平局、迭代增广匹配；未导入正式匹配函数。检查逐对象匹配、峰坐标、框最大值、组件面积、背景量、实际概率max融合和检查点/输入来源。\n\n通过检查数：'+str(result['checks_passed'])+'；独立复算帧数：'+str(result.get('independently_recomputed_frames',0))+'。\n\n范围：抽取100个帧/安排/条件组合独立重算；全表总量只做缓存一致性核对，不冒称全量独立复算。无新训练、无GPU推理、无跨seed声明。\n\n[完整记录](RESULTS_AUDIT.json)\n'
        if result['status']=='FAIL':text+='\n错误：'+result['error']+'\n'
        (BASE/'RESULTS_AUDIT.md').write_text(text)
    print(json.dumps(dict(status=result['status'],checks=result['checks_passed'],frames=result.get('independently_recomputed_frames'))))

if __name__=='__main__':main()
