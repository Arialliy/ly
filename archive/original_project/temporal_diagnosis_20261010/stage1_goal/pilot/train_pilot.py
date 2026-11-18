#!/usr/bin/env python3
"""Stage-1 weak-box fine-tuning. --prepare-plan is CPU-only; training requires --plan.

The original mask protocol is unavailable. These are explicitly weak-box controls,
not a reproduction of the official SoftIoU training recipe.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

PATCH = 128
CORE = 9
LENGTH = 40
CONFIG = {'version':1, 'length':40, 'patch':128, 'core_margin':9,
          'negative_box_margin':2, 'pairs_per_batch':2, 'extra_lambda':1.,
          'target_crop_fraction':.75, 'interior_fraction':.5,
          'interior_deltas':[4,8,12], 'boundary_deltas':[12,24,36],
          'mean':85.0799, 'std':47.4845, 'max_sampling_attempts':1000,
          'bn':'eval_running_stats_affine_trainable', 'dtype':'float32'}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temp.replace(path)


def pixel_rect(box, height=PATCH, width=PATCH):
    x,y,w,h = box
    return (max(0,min(width,math.ceil(x-.5))),
            max(0,min(height,math.ceil(y-.5))),
            max(0,min(width,math.ceil(x+w-.5))),
            max(0,min(height,math.ceil(y+h-.5))))


def valid_box(a):
    b = a.get('bbox', [])
    return len(b)==4 and all(math.isfinite(float(v)) for v in b) and b[2]>0 and b[3]>0


def positive_annotation(a):
    return valid_box(a) and not a.get('ignore',0) and not a.get('iscrowd',0)


def overlaps(a,b):
    x,y,w,h = a['bbox']; xx,yy,ww,hh = b['bbox']
    return max(x,xx)<min(x+w,xx+ww) and max(y,yy)<min(y+h,yy+hh)


def full_core(a, crop):
    cy,cx = crop
    x,y,w,h = a['bbox']
    return x>=cx+CORE and y>=cy+CORE and x+w<=cx+PATCH-CORE and y+h<=cy+PATCH-CORE


def local_rect(a,crop):
    cy,cx=crop; x,y,w,h=a['bbox']
    return pixel_rect((x-cx,y-cy,w,h))


def sequence_names(selection):
    splits=selection.get('splits',selection)
    names=splits.get('train')
    if not isinstance(names,list):
        raise ValueError('selection must contain splits.train (or train) list')
    names=[x if isinstance(x,str) else x['sequence'] for x in names]
    if len(names)!=len(set(names)) or len(names)<2:
        raise ValueError('At least two distinct training sequences are required')
    return names


class SequenceData:
    def __init__(self,root,name):
        self.name=name
        self.directory=Path(root)/name
        self.raw_path=self.directory/'raw_frames.npy'
        self.ann_path=self.directory/'annotations.json'
        self.frames=np.load(self.raw_path,mmap_mode='r',allow_pickle=False)
        if self.frames.dtype!=np.uint8 or self.frames.shape!=(160,512,640):
            raise ValueError(f'{name}: expected uint8[160,512,640], got {self.frames.shape}/{self.frames.dtype}')
        data=json.loads(self.ann_path.read_text())
        images=data['images']
        numbers=[int(Path(x['file_name']).stem) for x in images]
        if len(images)!=len(self.frames) or numbers!=sorted(numbers) or len(set(numbers))!=len(numbers):
            raise ValueError(f'{name}: image order is not unique numeric ascending with one image per frame')
        if any(numbers[i+1]!=numbers[i]+1 for i in range(len(numbers)-1)):
            raise ValueError(f'{name}: raw temporal sequence is not consecutive')
        if any(x['height']!=512 or x['width']!=640 for x in images):
            raise ValueError(f'{name}: annotation image dimensions disagree')
        index={x['id']:i for i,x in enumerate(images)}
        if len(index)!=len(images): raise ValueError('Duplicate image IDs')
        self.annotations=[[] for _ in images]
        seen=set()
        for a in data['annotations']:
            if a['image_id'] not in index: raise ValueError('Annotation references absent image')
            if a['id'] in seen: raise ValueError('Duplicate annotation ID')
            seen.add(a['id']); self.annotations[index[a['image_id']]].append(a)
        self.strict=[]
        self.track_samples=defaultdict(list)
        self.excluded_crop_anchor=0
        for t,anns in enumerate(self.annotations):
            anns.sort(key=lambda a:a['id'])
            strict=[]
            for a in anns:
                if not positive_annotation(a): continue
                x,y,w,h=a['bbox']
                if max(w,h)>16 or x<0 or y<0 or x+w>640 or y+h>512: continue
                if any(overlaps(a,b) for b in anns if b['id']!=a['id'] and valid_box(b)): continue
                rect=pixel_rect(a['bbox'],512,640)
                if rect[0]>=rect[2] or rect[1]>=rect[3]: continue
                strict.append(a)
                # A target-centric crop must contain the entire bbox inside its 9px core.
                if self.crop_limits(a) is None:
                    self.excluded_crop_anchor+=1
                    continue
                if 'track_id' not in a: raise ValueError('Strict target lacks track_id')
                self.track_samples[str(a['track_id'])].append((t,a))
            self.strict.append(strict)
        self.fingerprint={'raw_sha256':sha256(self.raw_path),'annotations_sha256':sha256(self.ann_path),
                          'frames':list(self.frames.shape),'strict_target_frames':sum(map(len,self.strict)),
                          'eligible_anchor_tracks':len(self.track_samples),
                          'strict_without_core_crop':self.excluded_crop_anchor}

    @staticmethod
    def crop_limits(a):
        x,y,w,h=a['bbox']
        lo_x=max(0,math.ceil(x+w-(PATCH-CORE)))
        hi_x=min(640-PATCH,math.floor(x-CORE))
        lo_y=max(0,math.ceil(y+h-(PATCH-CORE)))
        hi_y=min(512-PATCH,math.floor(y-CORE))
        return None if lo_x>hi_x or lo_y>hi_y else (lo_y,hi_y,lo_x,hi_x)


def temporal_options(t,interior):
    options={}
    for delta in (CONFIG['interior_deltas'] if interior else CONFIG['boundary_deltas']):
        starts=[]
        for a in range(0,160-LENGTH-delta+1):
            b=a+delta
            if b<=t<a+LENGTH and (not interior or (8<=t-a<=31 and 8<=t-b<=31)):
                starts.append(a)
        if starts: options[delta]=starts
    return options


def extra_specs_for_pairs(pairs,sequences):
    result=[]
    for pi,p in enumerate(pairs):
        seq=sequences[p['sequence']]
        for a in seq.strict[p['anchor_frame']]:
            if full_core(a,p['crop_yx']):
                result.append({'pair_index':pi,'annotation_id':a['id'], 'track_id':a['track_id'],
                               'slots':[p['anchor_frame']-s for s in p['starts']],
                               'rect':list(local_rect(a,p['crop_yx']))})
    return result


def random_matching(n,rng):
    if n<2:return []
    for attempt in range(10000):
        indices=rng.permutation(2*n).reshape(n,2)
        if all(a//2!=b//2 for a,b in indices):
            return indices.tolist()
    raise RuntimeError('Could not draw valid random matching within 10000 attempts')


def prepare_plan(args,sequences):
    if args.plan.exists(): raise FileExistsError(f'Will not replace existing frozen plan: {args.plan}')
    rng=np.random.default_rng(args.seed)
    # Separate RNG ensures random group control cannot change the image sample stream.
    match_rng=np.random.default_rng(args.seed+1000003)
    count=args.steps*2
    internal=np.array([True]*(count//2)+[False]*(count-count//2));rng.shuffle(internal)
    focused=np.array([True]*round(.75*count)+[False]*(count-round(.75*count)));rng.shuffle(focused)
    names=list(sequences)
    pools={}
    for name,seq in sequences.items():
        for kind in [True,False]:
            pools[name,kind]={track:[(t,a) for t,a in samples if temporal_options(t,kind)]
                              for track,samples in seq.track_samples.items()}
            pools[name,kind]={k:v for k,v in pools[name,kind].items() if v}
            if not pools[name,kind]: raise ValueError(f'{name}: no eligible {kind=} target anchors')
    entries=[];counts=Counter();retries=Counter()
    for step in range(args.steps):
        selected=rng.choice(names,size=2,replace=False).tolist();pairs=[]
        for pi,name in enumerate(selected):
            seq=sequences[name]; idx=2*step+pi;kind=bool(internal[idx]);focus=bool(focused[idx])
            for attempt in range(CONFIG['max_sampling_attempts']):
                tracks=sorted(pools[name,kind]);track=tracks[int(rng.integers(len(tracks)))]
                items=pools[name,kind][track]; t,a=items[int(rng.integers(len(items)))]
                opts=temporal_options(t,kind); deltas=sorted(opts)
                delta=deltas[int(rng.integers(len(deltas)))];starts=opts[delta]
                sa=int(starts[int(rng.integers(len(starts)))]);sb=sa+delta
                if focus:
                    lo_y,hi_y,lo_x,hi_x=seq.crop_limits(a);x,y,w,h=a['bbox']
                    cy=int(np.clip(round(y+h/2-PATCH/2+rng.uniform(-PATCH/6,PATCH/6)),lo_y,hi_y))
                    cx=int(np.clip(round(x+w/2-PATCH/2+rng.uniform(-PATCH/6,PATCH/6)),lo_x,hi_x))
                else:
                    cy=int(rng.integers(513-PATCH));cx=int(rng.integers(641-PATCH))
                if focus and not full_core(a,[cy,cx]):
                    retries['selected_target_outside_core']+=1;continue
                break
            else:raise RuntimeError('Sampling retry bound reached')
            p={'sequence':name,'anchor_frame':int(t),'selected_annotation_id':a['id'],
               'selected_track_id':a['track_id'],'starts':[sa,sb], 'crop_yx':[cy,cx],
               'stratum':'interior' if kind else 'boundary','target_centered_crop':focus,
               'sampling_attempts':attempt+1}
            pairs.append(p);counts[f'sequence/{name}']+=1;counts[p['stratum']]+=1
            counts['target_centered' if focus else 'random_crop']+=1
        specs=extra_specs_for_pairs(pairs,sequences);n=len(specs)
        entries.append({'step':step,'pairs':pairs,'extra_specs':specs,'random_pairs':random_matching(n,match_rng)})
        counts['extra_objects']+=n;counts['steps_extra_disabled_n_lt_2']+=int(n<2)
    plan={'config':CONFIG,'seed':args.seed,'steps':args.steps,'selection_sha256':sha256(args.selection),
          'train_sequences':names,'data_fingerprints':{k:v.fingerprint for k,v in sequences.items()},
          'counts':dict(counts),'retries':dict(retries),'entries':entries,
          'sampling_note':'Sequence uniform without replacement within each batch; track then eligible frame uniform. No model predictions read.'}
    save_json(args.plan,plan)
    args.plan.with_suffix(args.plan.suffix+'.sha256').write_text(sha256(args.plan)+'\n')
    print(json.dumps({'plan':str(args.plan),'sha256':sha256(args.plan),'counts':dict(counts)}),flush=True)


def build_batch(entry,sequences):
    arrays=[];positive=[];masks=[]
    for p in entry['pairs']:
        seq=sequences[p['sequence']];cy,cx=p['crop_yx']
        for start in p['starts']:
            raw=seq.frames[start:start+LENGTH,cy:cy+PATCH,cx:cx+PATCH]
            arrays.append((raw.astype(np.float32)-CONFIG['mean'])/CONFIG['std'])
            boxes=[];mask=np.zeros((LENGTH,PATCH,PATCH),dtype=bool)
            mask[:,CORE:PATCH-CORE,CORE:PATCH-CORE]=True
            for slot,anns in enumerate(seq.annotations[start:start+LENGTH]):
                for a in anns:
                    if not valid_box(a):continue
                    x,y,w,h=a['bbox'];margin=CONFIG['negative_box_margin']
                    x0,y0,x1,y1=pixel_rect((x-cx-margin,y-cy-margin,w+2*margin,h+2*margin))
                    mask[slot,y0:y1,x0:x1]=False
                    if positive_annotation(a) and full_core(a,[cy,cx]):
                        rect=local_rect(a,[cy,cx])
                        if rect[0]<rect[2] and rect[1]<rect[3]:boxes.append((slot,rect))
            positive.append(boxes);masks.append(mask)
    specs=extra_specs_for_pairs(entry['pairs'],sequences)
    if specs!=entry['extra_specs']:raise ValueError('Frozen extra-object list no longer matches labels')
    return np.stack(arrays)[:,None],positive,np.stack(masks),specs


def load_model(source,checkpoint_path):
    import torch
    sys.path.insert(0,str(source.resolve()))
    model_file=checkpoint_path.parent.parent/'DeepPro-Plus.py'
    if not model_file.is_file():raise FileNotFoundError('Checkpoint-adjacent original model source is required')
    spec=importlib.util.spec_from_file_location('stage1_official_checkpoint_model',model_file)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    numpy_core=np._core if hasattr(np,'_core') else np.core
    numeric_globals=[(numpy_core.multiarray.scalar,'numpy.core.multiarray.scalar'),np.dtype,type(np.dtype(np.float64))]
    with torch.serialization.safe_globals(numeric_globals):
        checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    model=module.detector(1,40,40)
    model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    return model,model_file


def bn_snapshot(model):
    import torch
    result={}
    for name,module in model.named_modules():
        if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):
            for key in ['running_mean','running_var','num_batches_tracked']:
                value=getattr(module,key)
                if value is not None:result[f'{name}.{key}']=value.detach().clone()
    return result


def buffer_hash(buffers):
    h=hashlib.sha256()
    for name,value in sorted(buffers.items()):
        h.update(name.encode());h.update(str(value.dtype).encode());h.update(value.cpu().numpy().tobytes())
    return h.hexdigest()


def train(args,sequences):
    import torch
    from losses import pilot_loss
    if not args.plan or not args.plan.is_file():raise ValueError('Training requires an existing shared --plan')
    plan=json.loads(args.plan.read_text())
    if plan['config']!=CONFIG or plan['steps']!=args.steps or plan['seed']!=args.seed:
        raise ValueError('Frozen plan config/steps/seed mismatch')
    if plan['selection_sha256']!=sha256(args.selection):raise ValueError('Selection SHA changed')
    if plan['train_sequences']!=list(sequences):raise ValueError('Train sequence selection changed')
    if plan['data_fingerprints']!={k:v.fingerprint for k,v in sequences.items()}:
        raise ValueError('Training data fingerprint changed')
    if len(plan['entries'])!=args.steps:raise ValueError('Wrong number of frozen steps')
    plan_hash=sha256(args.plan)
    hash_file=args.plan.with_suffix(args.plan.suffix+'.sha256')
    if not hash_file.is_file() or hash_file.read_text().strip()!=plan_hash:raise ValueError('Plan hash file mismatch')
    if args.output.exists() and any(args.output.iterdir()):raise FileExistsError('Output directory must be empty')
    args.output.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    model,model_file=load_model(args.source,args.checkpoint)
    device=torch.device(args.device);model=model.to(device=device,dtype=torch.float32)
    for parameter in model.parameters():parameter.requires_grad_(True)
    model.train()
    for module in model.modules():
        if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.eval()
    original_bn=bn_snapshot(model);bn_hash=buffer_hash(original_bn)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=1e-4)
    metadata={'strategy':args.strategy,'steps':args.steps,'lr':args.lr,'seed':args.seed,
              'optimizer':'fresh Adam','weight_decay':1e-4,'plan_sha256':plan_hash,
              'selection_sha256':sha256(args.selection),'checkpoint_sha256':sha256(args.checkpoint),
              'model_source_sha256':sha256(model_file),'train_source_sha256':sha256(Path(__file__)),
              'loss_source_sha256':sha256(Path(__file__).with_name('losses.py')),
              'bn_initial_sha256':bn_hash,'config':CONFIG,'torch_version':str(torch.__version__),
              'source_path':str(args.source.resolve()),'checkpoint_path':str(args.checkpoint.resolve()),
              'status':'running','protocol':'weak_box_MIL_not_official_SoftIoU'}
    save_json(args.output/'RUN.json',metadata)
    counts=Counter();started=time.perf_counter()
    try:
        with (args.output/'TRAIN.jsonl').open('w',buffering=1) as log:
            for step,entry in enumerate(plan['entries']):
                if entry['step']!=step:raise ValueError('Plan step order changed')
                for module in model.modules():
                    if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.eval()
                batch,pos,masks,specs=build_batch(entry,sequences)
                x=torch.from_numpy(batch).to(device)
                background=torch.from_numpy(masks).to(device)
                optimizer.zero_grad(set_to_none=True)
                logits=model(x)[1]
                if logits.shape!=(4,40,PATCH,PATCH):raise ValueError('Unexpected logits shape')
                loss,parts=pilot_loss(logits,pos,background,specs,args.strategy,entry['random_pairs'])
                if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss at {step}')
                loss.backward()
                parameters=[p for p in model.parameters() if p.grad is not None]
                if not all(torch.isfinite(p.grad).all() for p in parameters):raise FloatingPointError(f'Nonfinite gradient at {step}')
                grad_norm=torch.linalg.vector_norm(torch.stack([p.grad.detach().norm() for p in parameters]))
                optimizer.step()
                current_bn=bn_snapshot(model)
                if current_bn.keys()!=original_bn.keys() or any(not torch.equal(current_bn[k],v) for k,v in original_bn.items()):
                    raise AssertionError('BN running buffers changed')
                counts['extra_disabled_n_lt_2']+=parts['extra_disabled_n_lt_2'];counts['extra_objects']+=parts['extra_objects']
                if step%16==0 or step==args.steps-1:
                    row={'step':step+1,'loss':float(loss.detach()),'positive':float(parts['positive'].detach()),
                         'negative':float(parts['negative'].detach()),'extra':float(parts['extra'].detach()),
                         'grad_norm':float(grad_norm),'finite':True,'extra_objects':parts['extra_objects'],
                         'extra_disabled_n_lt_2':parts['extra_disabled_n_lt_2'],
                         'base_objects_per_window':parts['base_objects_per_window'],
                         'background_pixels_per_window':parts['background_pixels_per_window'],
                         'elapsed_seconds':time.perf_counter()-started}
                    line=json.dumps(row);log.write(line+'\n');print(line,flush=True)
                del logits,loss,parts,x,background
        if device.type=='cuda':torch.cuda.synchronize(device)
        duration=time.perf_counter()-started
        final_state={k:v.detach().cpu() for k,v in model.state_dict().items()}
        if not all(torch.isfinite(v).all() for v in final_state.values()):raise FloatingPointError('Nonfinite final parameters')
        checkpoint={'model_state_dict':final_state,'step':args.steps,'strategy':args.strategy,
                    'plan_sha256':plan_hash,'initial_checkpoint_sha256':metadata['checkpoint_sha256'],
                    'protocol':'weak_box_MIL_not_official_SoftIoU'}
        torch.save(checkpoint,args.output/'final_model.pth')
        metadata.update(status='complete',duration_seconds=duration,counts=dict(counts),
                        bn_final_sha256=buffer_hash(bn_snapshot(model)),
                        final_checkpoint_sha256=sha256(args.output/'final_model.pth'),
                        train_log_sha256=sha256(args.output/'TRAIN.jsonl'),
                        final_step=args.steps,checkpoint_selection='fixed final step; no best checkpoint')
        save_json(args.output/'RUN.json',metadata)
    except BaseException as error:
        metadata.update(status='failed',error=repr(error),duration_seconds=time.perf_counter()-started)
        save_json(args.output/'RUN.json',metadata)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['dataset-root','selection','source','checkpoint','output','plan']:
        parser.add_argument('--'+key,type=Path,required=(key in ['dataset-root','selection','plan']))
    parser.add_argument('--strategy',choices=['paired_mean','paired_max','ohem','random_pair_max'])
    parser.add_argument('--steps',type=int,default=512);parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--seed',type=int,default=46);parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--prepare-plan',action='store_true')
    args=parser.parse_args()
    if args.steps<1 or args.lr<=0:parser.error('steps and lr must be positive')
    if not args.prepare_plan and any(getattr(args,k) is None for k in ['source','checkpoint','output','strategy']):
        parser.error('training requires --source --checkpoint --output --strategy')
    selection=json.loads(args.selection.read_text())
    sequences={name:SequenceData(args.dataset_root,name) for name in sequence_names(selection)}
    if args.prepare_plan:prepare_plan(args,sequences)
    else:train(args,sequences)


if __name__=='__main__':main()
