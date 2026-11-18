"""Original F tensors/loss caches with a frozen single-window replay adapter.

Primary and auxiliary windows remain in their original eight positions. Only
source-matched, time-matched, role-matched existing crop exposures can change.
The original C algorithm is reused with two mechanical iteration changes so
four explicit one-start windows replace two shared-crop two-start pairs.
"""
from __future__ import annotations
import argparse
import copy
from functools import lru_cache
import inspect
import json
from pathlib import Path
import sys
import time
import numpy as np

E=Path(__file__).resolve().parents[1];ROOT=E.parents[1]
V1=ROOT/'experiments/track_contrast_v1';V2=ROOT/'experiments/track_contrast_v2'
import data as original_data
from runtime import load_module, sha, save_json
from build_schedules import read

inputs=original_data.inputs
RISK_KEYS=('target_regions','background_regions','negative_indices','segment_indices','segment_windows')


@lru_cache(maxsize=1)
def schedule():
    value=read(E/'SCHEDULE.json')
    for path,expected in value['sources'].items():assert sha(path)==expected,path
    for path,expected in value['csv_sha256'].items():assert sha(E/path)==expected,path
    return value


def normalize(condition):
    value=condition.upper()
    if value not in ('ORIGINAL','REPLAY'):raise ValueError(condition)
    return value


def row(index):return schedule()['entries'][int(index)]
def changed(index):return any(x['replaced']for x in row(index)['windows'])


def windows(index,condition):
    r=row(index);pool=schedule()['window_pool']
    return [pool[(x['donor_index']*8+x['donor_window']) if normalize(condition)=='REPLAY'
                 else r['original_index']*8+x['window']] for x in r['windows']]


def explicit_entries(index,condition,state):
    base,_,sequences,_=state;r=row(index);ws=windows(index,condition)
    entries=[]
    for offset in (0,4):
        entry={'step':r['original_index'],'pairs':[copy.deepcopy(w['pair'])for w in ws[offset:offset+4]],'random_pairs':[]}
        entry['extra_specs']=base.extra_specs_for_pairs(entry['pairs'],sequences);entries.append(entry)
    return entries


def batch_data(index,condition,state):
    condition=normalize(condition);r=row(index)
    if condition=='ORIGINAL'or not changed(index):
        batch=original_data.batch_data(r['original_index'],state)
    else:
        base,_,sequences,_=state;entry,aux=explicit_entries(index,condition,state)
        x,pos,bg,specs=base.build_batch(entry,sequences)
        targets,valid,stats=original_data.historical.build_targets(entry,sequences,background_masks=bg)
        ax,apos,abg,aspecs=base.build_batch(aux,sequences)
        batch={'x':np.concatenate([x,ax]),'targets':targets,'valid':valid,'det_background':bg,
            'background':np.concatenate([bg,abg]),'entry':entry,'aux_entry':aux,
            'input_sha256':original_data.historical.data_hash(x,pos,bg,specs),
            'label_sha256':original_data.historical.label_hash(targets,valid),
            'auxiliary_input_sha256':original_data.historical.data_hash(ax,apos,abg,aspecs),'label_stats':stats}
    expected=[not w['nonempty']for w in windows(index,condition)[:4]]
    assert [s['pure_negative_window']for s in batch['label_stats']['per_window']]==expected
    batch['additional_index']=int(index);batch['condition']=condition;batch['original_plan_index']=r['original_index']
    return batch


@lru_cache(maxsize=1)
def samplers():
    module=load_module('replay_original_track_sampler',V1/'train_modules/track_pair_sampler.py')
    segment=load_module('replay_original_segments',V2/'train_modules/segments.py')
    src=inspect.getsource(module.build_pair_plan)
    changes={"assert seed == 46 and len(entry[\"pairs\"]) == 2":"assert seed == 46 and len(entry[\"pairs\"]) == 4",
             "assert len(pair[\"starts\"]) == 2":"assert len(pair[\"starts\"]) == 1",
             "window = pair_index * 2 + within_pair":"window = pair_index"}
    for before,after in changes.items():assert src.count(before)==1;src=src.replace(before,after)
    scope=dict(vars(module));exec(compile(src,'<original_C_single_window_iteration_adapter>','exec'),scope)
    return module.TrackPairSampler(),segment.SegmentSampler(),scope['build_pair_plan'],segment,changes


def cache_paths(index,condition):
    condition=normalize(condition);pi=row(index)['original_index']
    if condition=='ORIGINAL'or not changed(index):
        return V1/'cache'/f'entry_{pi:03d}.npz', (V2/'cache'/f'entry_{pi:03d}.npz' if pi<256 else E/'cache'/f'original_frame_{pi:03d}.npz')
    return E/'cache'/f'replay_contrast_{index:03d}.npz',E/'cache'/f'replay_frame_{index:03d}.npz'


def load_caches(index,condition,batch=None):
    result=[]
    for path in cache_paths(index,condition):
        with np.load(path,allow_pickle=False)as f:c={k:f[k]for k in f.files}
        if batch is not None:
            meta=json.loads(str(c['metadata_json']))
            for k in ('input_sha256','label_sha256','auxiliary_input_sha256'):assert meta[k]==batch[k],(str(path),k)
        result.append(c)
    return tuple(result)


def donor_keep(index):
    """Reuse exact per-window original keep labels, groups, and pixel sampling."""
    coords=[];values=[];groups=[];provenance=[]
    for wi,w in enumerate(windows(index,'REPLAY')):
        path=V1/'cache'/f"entry_{w['original_index']:03d}.npz"
        with np.load(path,allow_pickle=False)as f:
            take=f['keep_indices'][:,0]==w['window'];c=f['keep_indices'][take].copy();c[:,0]=wi
            coords.append(c);values.append(f['keep_probabilities'][take]);groups.append(f['keep_groups'][take])
        provenance.append({'new_window':wi,'donor_index':w['original_index'],'donor_window':w['window'],'cache_sha256':sha(path)})
    return {'keep_indices':np.concatenate(coords),'keep_probabilities':np.concatenate(values),'keep_groups':np.concatenate(groups)},provenance


def source_hashes():
    paths=[Path(__file__),E/'SCHEDULE.json',E/'code/data.py',E/'code/runtime.py',
        V1/'train_modules/track_pair_sampler.py',V1/'train_modules/region_contrast_loss.py',
        V1/'train_modules/reliable_keep_loss.py',V2/'train_modules/segments.py',V2/'train_modules/temporal_risk.py']
    return {str(p):sha(p)for p in paths}


def write_cache(path,arrays,meta):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.npz.tmp')
    with temp.open('wb')as f:np.savez_compressed(f,**arrays,metadata_json=np.array(json.dumps(meta,ensure_ascii=False)))
    temp.replace(path);meta=dict(meta,sha256=sha(path));save_json(path.with_suffix('.json'),meta)


def jobs():
    values=[('ORIGINAL',i)for i in range(512)if row(i)['original_index']>=256]
    values +=[('REPLAY',i)for i in range(512)if changed(i)]
    return values


def prepare(shard,shards):
    import torch
    from runtime import setup,load_model,CP,MODEL_SOURCE
    setup();state=inputs();model,_=load_model(CP)
    sampler,segments,adapted,segment_mod,_=samplers()
    region=load_module('replay_original_background_mining',V1/'train_modules/region_contrast_loss.py')
    sources=source_hashes();sources.update({str(CP):sha(CP),str(MODEL_SOURCE):sha(MODEL_SOURCE)})
    begun=time.perf_counter();done=[]
    for ordinal,(condition,index)in enumerate(jobs()):
        if ordinal%shards!=shard:continue
        paths=cache_paths(index,condition);write_paths=paths if condition=='REPLAY'else paths[1:]
        if all(p.exists()and p.with_suffix('.json').exists()for p in write_paths):
            for p in write_paths:
                m=read(p.with_suffix('.json'));assert m['sha256']==sha(p)and m['sources']==sources
            done.append([condition,index]);continue
        if any(p.exists()for p in write_paths):raise RuntimeError('Partial cache job must be inspected before resuming')
        batch=batch_data(index,condition,state);pi=row(index)['original_index']
        with torch.inference_mode():
            probabilities=torch.cat([model(torch.from_numpy(batch['x'][i:i+2]).cuda())[1].sigmoid().cpu()for i in range(0,8,2)]).numpy()
        meta={'condition':condition,'additional_index':index,'original_index':pi,'sources':sources,
            **{k:batch[k]for k in ('input_sha256','label_sha256','auxiliary_input_sha256')},
            'teacher':'unchanged_release_model; ordinary training cache, no new weak qualification'}
        if condition=='REPLAY':
            entry,aux=explicit_entries(index,condition,state)
            cp=adapted(entry,sampler.sequences,46)
            regions=np.asarray(cp['target_regions'],np.int64).reshape(-1,6)
            pc=np.asarray(cp['positive_pairs_C'],np.int64).reshape(-1,2);pb=np.asarray(cp['positive_pairs_B'],np.int64).reshape(-1,2)
            anchors=regions[pc[:,0]]
            backgrounds,negative=region.mine_background_regions(probabilities,batch['background'],anchors,4+anchors[:,0],local_radius=32)
            invalid_units={cp['pair_metadata'][i]['swap_unit_id']for i in range(len(pc))if not bool((negative[i]>=0).all())}
            for i,m in enumerate(cp['pair_metadata']):
                if m['swap_unit_id']in invalid_units:negative[i,:]=-1
            keep,keep_sources=donor_keep(index)
            arrays=dict(target_regions=regions,positive_pairs_C=pc,positive_pairs_B=pb,background_regions=backgrounds,negative_indices=negative,**keep)
            contrast_meta=dict(meta,donor_keep=keep_sources,annotation_plan=cp,pairs=len(pc),valid_pairs=int((negative>=0).all(1).sum()))
            write_cache(paths[0],arrays,contrast_meta)
        sp=segments.sample(batch['entry'],batch['aux_entry'])
        targets=np.asarray(sp['target_regions'],np.int64).reshape(-1,6)
        globals_=np.where(targets[:,0]<4,targets[:,0]+4,targets[:,0])
        backgrounds,negative=region.mine_background_regions(probabilities,batch['background'],targets,globals_,local_radius=32)
        effective=segment_mod.build_segments(sp['descriptors'],(negative>=0).any(1),length=8,stride=4,tail_alignment=True)
        arrays=dict(target_regions=targets,background_regions=backgrounds,negative_indices=negative,
            segment_indices=np.asarray(effective['segment_indices'],np.int64).reshape(-1,8),segment_windows=np.asarray(effective['segment_windows'],np.int64))
        for wi,ti,x0,y0,x1,y1 in backgrounds:assert batch['background'][wi,ti,y0:y1,x0:x1].all()
        write_cache(paths[1],arrays,dict(meta,annotation_plan=sp,effective_coverage=effective['coverage']))
        done.append([condition,index])
        if len(done)==1 or len(done)%8==0:print(json.dumps({'shard':shard,'done':len(done),'condition':condition,'index':index,'elapsed_seconds':time.perf_counter()-begun}),flush=True)
    assert sources=={p:sha(p)for p in sources}
    save_json(E/'cache'/f'SHARD_{shard}.json',{'status':'COMPLETE','shard':shard,'shards':shards,'jobs':done,'sources':sources,'elapsed_seconds':time.perf_counter()-begun})


def merge():
    records=[]
    for condition,index in jobs():
        paths=cache_paths(index,condition);paths=paths if condition=='REPLAY'else paths[1:]
        for p in paths:
            m=read(p.with_suffix('.json'));assert m['sha256']==sha(p)
            records.append({'condition':condition,'additional_index':index,'path':str(p),'sha256':sha(p)})
    save_json(E/'CACHE_MANIFEST.json',{'status':'COMPLETE','jobs':len(jobs()),'files':records,'source_hashes':source_hashes()})
    print(json.dumps({'status':'COMPLETE','jobs':len(jobs()),'cache_files':len(records)}))


def smoke():
    state=inputs();sampler,segments,adapted,_,changes=samplers();checks=[]
    # Mechanical adapter must exactly reproduce the original C plan when only
    # the representation, and not the windows, is changed.
    old_plans=read(V1/'PAIR_PLAN.json')['entries']
    for index in (0,1,256,257):
        pi=row(index)['original_index'];entry,aux=explicit_entries(index,'ORIGINAL',state)
        actual=adapted(entry,sampler.sequences,46)
        assert actual==old_plans[pi],pi
        oldbatch=original_data.batch_data(pi,state);base,_,seqs,_=state
        x,pos,bg,specs=base.build_batch(entry,seqs);y,v,stats=original_data.historical.build_targets(entry,seqs,background_masks=bg)
        ax,apos,abg,aspecs=base.build_batch(aux,seqs)
        for key,a,b in [('x',np.concatenate([x,ax]),oldbatch['x']),('targets',y,oldbatch['targets']),('valid',v,oldbatch['valid']),('background',np.concatenate([bg,abg]),oldbatch['background'])]:assert np.array_equal(a,b),(pi,key)
        assert stats==oldbatch['label_stats']
        if pi<256:assert segments.sample(entry,aux)==read(V2/'SEGMENT_PLAN.json')['entries'][pi] or segments.sample(entry,aux)['target_regions']==read(V2/'SEGMENT_PLAN.json')['entries'][pi]['target_regions']
        checks.append({'index':index,'original_index':pi,'tensor_and_C_plan_exact':True})
    # Exercise the changed-input path, including an auxiliary replacement.
    selected=next(i for i in range(512)if any(x['replaced']and x['window']>=4 for x in row(i)['windows']))
    batch=batch_data(selected,'REPLAY',state);c,provenance=donor_keep(selected)
    assert c['keep_indices'].shape[1]==4 and len(provenance)==8
    value={'status':'PASS','checks':checks,'changed_aux_batch':selected,'single_window_adapter_changes':changes,'schedule_sha256':sha(E/'SCHEDULE.json'),'code_sha256':sha(Path(__file__))}
    save_json(E/'evaluation/SAMPLER_SMOKE.json',value);print(json.dumps(value))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--smoke',action='store_true');parser.add_argument('--merge',action='store_true');parser.add_argument('--shard',type=int);parser.add_argument('--shards',type=int,default=3);args=parser.parse_args()
    if args.smoke:smoke()
    elif args.merge:merge()
    elif args.prepare and args.shard is not None:prepare(args.shard,args.shards)
    else:parser.error('Choose --smoke, --merge, or --prepare --shard N')
