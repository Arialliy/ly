#!/usr/bin/env python3
"""Independent CPU-only geometry, quantization, domain and full-plan audit.

Reads only frozen training annotations and TRAIN_PLAN; no images, model, GPU,
development response, or label-generation helper is used in the scalar oracle.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import time
import traceback

import numpy as np
import gaussian_labels as tested


HERE = Path(__file__).resolve().parent
OUT = HERE.parent
Q = OUT.parent
CHECKS = []


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(name, condition, detail=None):
    CHECKS.append({'name': name, 'pass': bool(condition), 'detail': detail})
    if not condition:
        raise AssertionError(name)


def pixel_membership(box, crop, h=128, w=128, margin=0):
    """Independent center comparisons, never calls pixel_rect."""
    x,y,bw,bh = map(float, box)
    yy = np.arange(h, dtype=np.float64)+float(crop[0])+0.5
    xx = np.arange(w, dtype=np.float64)+float(crop[1])+0.5
    return ((yy[:,None] >= y-margin) & (yy[:,None] < y+bh+margin)
            & (xx[None,:] >= x-margin) & (xx[None,:] < x+bw+margin))


def scalar_gaussian(box, crop=(0,0), h=128, w=128):
    """Slow scalar mathematical oracle using math.exp and comparisons."""
    x,y,bw,bh = map(float, box)
    out=np.zeros((h,w),dtype=np.uint8)
    rows,cols=np.nonzero(pixel_membership(box,crop,h,w))
    for i,j in zip(rows,cols):
        px=float(j)+float(crop[1])+0.5
        py=float(i)+float(crop[0])+0.5
        e=-0.5*((px-(x+bw/2))/(bw/6))**2-0.5*((py-(y+bh/2))/(bh/6))**2
        out[i,j]=math.floor(255*math.exp(e))
    return out


def tested_gaussian(box,crop=(0,0),h=128,w=128):
    rect,g=tested.gaussian_region(box,crop,h,w)
    x0,y0,x1,y1=rect
    out=np.zeros((h,w),dtype=np.uint8)
    out[y0:y1,x0:x1]=g
    return out


def valid_box(ann):
    b=ann.get('bbox',[])
    return len(b)==4 and all(math.isfinite(float(x)) for x in b) and b[2]>0 and b[3]>0


def eligible(ann,crop):
    if not valid_box(ann) or ann.get('ignore',0) or ann.get('iscrowd',0):
        return False
    x,y,w,h=ann['bbox'];cy,cx=crop
    return x>=cx+9 and y>=cy+9 and x+w<=cx+119 and y+h<=cy+119


def oracle_background(entry,sequences):
    mask=np.zeros((4,40,128,128),dtype=bool)
    mask[:,:,9:119,9:119]=True
    wi=0
    for pair in entry['pairs']:
        seq=sequences[pair['sequence']]
        for start in pair['starts']:
            for ti,anns in enumerate(seq.annotations[start:start+40]):
                for ann in anns:
                    if valid_box(ann):
                        mask[wi,ti,pixel_membership(ann['bbox'],pair['crop_yx'],margin=2)]=False
            wi+=1
    return mask


def oracle_targets(entry,sequences,old_bg):
    valid=old_bg.copy()
    target=np.zeros(valid.shape,dtype=bool)
    object_stats=[]
    wi=0
    for pair in entry['pairs']:
        seq=sequences[pair['sequence']]
        crop=pair['crop_yx']
        for start in pair['starts']:
            for ti,anns in enumerate(seq.annotations[start:start+40]):
                excluded=[];objects=[]
                for ann in anns:
                    if not valid_box(ann):continue
                    if eligible(ann,crop):
                        rect=pixel_membership(ann['bbox'],crop)
                        support=scalar_gaussian(ann['bbox'],crop)>0
                        valid[wi,ti] |= rect
                        target[wi,ti] |= support
                        objects.append((support,int(rect.sum())))
                    else:excluded.append(ann)
                for ann in excluded:
                    valid[wi,ti,pixel_membership(ann['bbox'],crop,margin=2)]=False
                for support,rect_size in objects:
                    object_stats.append({'raw':int(support.sum()),
                        'effective':int((support & valid[wi,ti]).sum()),'rect':rect_size})
            wi+=1
    target &= valid
    return target,valid,object_stats


def fixture(anns):
    seq=SimpleNamespace(annotations=[anns]+[[] for _ in range(159)])
    entry={'pairs':[{'sequence':'toy','crop_yx':[0,0],'starts':[0,0]},
                    {'sequence':'toy','crop_yx':[0,0],'starts':[0,0]}]}
    return entry,{'toy':seq}


def unit_tests():
    cases=[('even',[20,30,8,6]),('odd',[20,30,7,9]),
           ('float',[20.2,30.7,7.4,5.8]),('noninteger_edges',[20.5,30.5,8,6]),
           ('quantized_empty',[20.499,20.499,0.5,0.5]),
           ('off_image',[-2.1,-1.3,8.7,5.6])]
    for label,box in cases:
        check('scalar_gaussian_'+label,np.array_equal(tested_gaussian(box),scalar_gaussian(box)))
    box=[25,20,20,20]
    g=tested_gaussian(box)
    check('quantization_nonzero_and_nonrectangular',0<int((g>0).sum())<400)
    check('quantized_zero_corners_positive_center',g[20,25]==0 and g[29,34]>0)
    check('subpixel_no_forced_center',not np.any(tested_gaussian([20.499,20.499,0.5,0.5])))
    box=[14.2,12.7,18.3,16.6]
    full=tested_gaussian(box,h=80,w=80)
    sub=tested_gaussian(box,crop=(14,17),h=20,w=20)
    check('crop_does_not_move_center',np.array_equal(sub,full[14:34,17:37]))
    check('even_box_center_not_integer_snapped',g[29,34]==g[29,35]==g[30,34]==g[30,35])
    for label,anns in [
        ('overlap',[{'bbox':[20,20,16,16]},{'bbox':[28,23,12,14]}]),
        ('ignore_priority',[{'bbox':[20,20,20,20]},{'bbox':[28,28,2,2],'ignore':1}]),
        ('crowd_priority',[{'bbox':[20,20,20,20]},{'bbox':[28,28,2,2],'iscrowd':1}]),
        ('partial_core_exclusion',[{'bbox':[10,20,10,10]},{'bbox':[7,20,4,4]}]),
        ('empty_no_peak',[{'bbox':[20.499,20.499,0.5,0.5]}]),
        ('core9_boundaries',[{'bbox':[9,9,10,10]},{'bbox':[109,109,10,10]},
                             {'bbox':[8.999,50,5,5]},{'bbox':[115,50,4.001,5]}]),
        ('no_size_class_filter',[{'bbox':[30,30,23,23],'category_id':999}]),
    ]:
        entry,seqs=fixture(anns)
        bg=oracle_background(entry,seqs)
        truth,domain,ostats=oracle_targets(entry,seqs,bg)
        y,d,stats=tested.build_targets(entry,seqs,bg)
        check(label+'_target',np.array_equal(y,truth.astype(np.float32)))
        check(label+'_domain',np.array_equal(d,domain))
        check(label+'_does_not_mutate_old_mask',np.array_equal(bg,oracle_background(entry,seqs)))
        if label=='ignore_priority':
            check('ignore_expanded2_overrides_positive',not np.any(d[:,0,26:32,26:32]))
        if label=='empty_no_peak':
            check('empty_window_and_object_counts',stats['pure_negative_windows']==4 and
                  sum(r['positive_box_frames_without_quantized_support'] for r in stats['per_window'])==4)
        if label=='overlap':
            expected=(np.maximum(scalar_gaussian(anns[0]['bbox']),scalar_gaussian(anns[1]['bbox']))>0)
            check('overlap_is_max_support_not_addition',np.array_equal(y[0,0],expected))


def full_plan_audit():
    planpath=Q/'stage1_goal/pilot/TRAIN_PLAN.json'
    selectionpath=Q/'stage1_goal/DATA_SELECTION.json'
    plan=json.loads(planpath.read_text())
    check('plan_identity',sha(planpath)=='6f86d929ad1275bbb120456dd23a25c7ea9571517f60b4c64a9d5ce6413c975c')
    check('selection_identity',sha(selectionpath)=='1f5e617764c0fe27a63440cc5b26d0e27a05db7dbc8c997b0bc5fd74a4b39c92')
    check('plan_512_updates',len(plan['entries'])==512)
    sequences={};annotation_sha={}
    for name in plan['train_sequences']:
        p=Q/'stage1_goal/data'/name/'annotations.json'
        data=json.loads(p.read_text());ind={im['id']:i for i,im in enumerate(data['images'])}
        anns=[[] for _ in data['images']]
        for a in data['annotations']:anns[ind[a['image_id']]].append(a)
        for aa in anns:aa.sort(key=lambda a:a['id'])
        sequences[name]=SimpleNamespace(annotations=anns)
        annotation_sha[name]=sha(p)
        check('annotation_identity_'+name,annotation_sha[name]==plan['data_fingerprints'][name]['annotations_sha256'])
    digest=hashlib.sha256();rows=[];object_counts={'exposures':0,'raw_empty':0,'effective_empty':0,
            'removed_entirely_by_exclusion':0,'sum_raw_support':0,'sum_effective_support':0,'sum_rect_pixels':0}
    support_ratios=[];total_empty=0
    for step,entry in enumerate(plan['entries'],1):
        bg=oracle_background(entry,sequences)
        truth,domain,objects=oracle_targets(entry,sequences,bg)
        y,d,stats=tested.build_targets(entry,sequences,bg)
        check('step_%03d_binary_target_matches_independent_scalar'%step,np.array_equal(y,truth.astype(np.float32)))
        check('step_%03d_domain_matches_independent_geometry'%step,np.array_equal(d,domain))
        digest.update(step.to_bytes(4,'little'));digest.update(y.tobytes());digest.update(d.tobytes())
        row={'step':step,'target_sha256':hashlib.sha256(y.tobytes()).hexdigest(),
             'valid_sha256':hashlib.sha256(d.tobytes()).hexdigest(),'stats':stats}
        rows.append(row);total_empty+=stats['pure_negative_windows']
        for obj in objects:
            object_counts['exposures']+=1;object_counts['raw_empty']+=obj['raw']==0
            object_counts['effective_empty']+=obj['effective']==0
            object_counts['removed_entirely_by_exclusion']+=obj['raw']>0 and obj['effective']==0
            object_counts['sum_raw_support']+=obj['raw'];object_counts['sum_effective_support']+=obj['effective']
            object_counts['sum_rect_pixels']+=obj['rect']
            if obj['rect']:support_ratios.append(obj['raw']/obj['rect'])
        if step%64==0:print(json.dumps({'completed_steps':step,'empty_windows':total_empty}),flush=True)
    ratio=np.array(support_ratios,dtype=np.float64)
    return {'updates':512,'window_exposures':2048,'frame_exposures':81920,'training_sequences':list(sequences),
            'training_annotation_sha256':annotation_sha,'plan_sha256':sha(planpath),'selection_sha256':sha(selectionpath),
            'target_and_domain_chain_sha256':digest.hexdigest(),'pure_negative_windows':total_empty,
            'object_exposure_statistics':object_counts,'raw_support_to_rect_quantiles':dict(zip(
                ['min','q25','median','q75','max'],map(float,np.quantile(ratio,[0,.25,.5,.75,1])))),
            'per_update':rows,'scope':'training annotations only; counts are repeated plan exposures, not distinct objects'}


def main():
    start=time.time();result={'status':'RUNNING','gpu_used':False,'development_outputs_read':False,
        'images_read':False,'test_script':str(Path(__file__).resolve()),'label_config':tested.LABEL_CONFIG,
        'tested_source_sha256':{str(HERE/'gaussian_labels.py'):sha(HERE/'gaussian_labels.py'),
                                str(Path(__file__).resolve()):sha(Path(__file__).resolve())}}
    try:
        unit_tests()
        result['full_plan']=full_plan_audit()
        check('generator_source_unchanged_during_test',sha(HERE/'gaussian_labels.py')==
              result['tested_source_sha256'][str(HERE/'gaussian_labels.py')])
        result['status']='PASS'
    except Exception:
        result['status']='FAIL';result['error']=traceback.format_exc()
    result['checks']=CHECKS;result['check_count']=len(CHECKS);result['elapsed_seconds']=time.time()-start
    (OUT/'LABEL_PREFLIGHT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','check_count','elapsed_seconds')},ensure_ascii=False),flush=True)
    if result['status']!='PASS':raise RuntimeError(result['error'])


if __name__=='__main__':main()
