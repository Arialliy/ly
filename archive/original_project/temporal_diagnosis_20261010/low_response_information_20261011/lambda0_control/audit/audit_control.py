#!/usr/bin/env python3
"""One-page independent audit of lambda=0 provenance and fixed-0.5 outcomes."""
import csv,datetime,hashlib,json,math,time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage

OUT=Path(__file__).resolve().parent;L=OUT.parent;Q=L.parent.parent;P=Q/'stage1_goal'
PLANS={'O':[0,36,72,108,120],'S':[0,24,60,96,120]}
SEQS=['DJI_0026_2','DJI_0067_2']
FIELDS=['gt_count','matched_count','component_count','unmatched_components','positive_pixels','positive_pixels_outside_boxes']

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):
    with Path(p).open() as f:return list(csv.DictReader(f))
def crop(b):
    x,y,w,h=b
    return max(0,min(640,math.ceil(x-.5))),max(0,min(512,math.ceil(y-.5))),max(0,min(640,math.ceil(x+w-.5))),max(0,min(512,math.ceil(y+h-.5)))

def measure(prob,anns):
    binary=prob>=.5;labels,n=ndimage.label(binary,structure=np.ones((3,3),np.uint8))
    loc=np.flatnonzero(binary);lab=labels.ravel()[loc];maxscores=np.full(n+1,-np.inf,np.float32)
    np.maximum.at(maxscores,lab,prob.ravel()[loc]);eligible=loc[prob.ravel()[loc]==maxscores[lab]]
    first=np.full(n+1,prob.size,np.int64);np.minimum.at(first,labels.ravel()[eligible],eligible)
    ids=np.arange(1,n+1);peaks=first[1:];py,px=np.divmod(peaks,640);px=px+.5;py=py+.5
    areas=np.bincount(lab,minlength=n+1);edges=[];gtmask=np.zeros(prob.shape,bool)
    for a in anns:
        x,y,w,h=a['bbox'];xx,yy,xxx,yyy=crop(a['bbox']);gtmask[yy:yyy,xx:xxx]=True
        cs=np.flatnonzero((px>=x)&(px<x+w)&(py>=y)&(py<y+h)).tolist()
        cs.sort(key=lambda j:(((px[j]-x-w/2)/w)**2+((py[j]-y-h/2)/h)**2,-float(maxscores[j+1]),j+1))
        edges.append(cs)
    owner={}
    def augment(i,seen):
        for c in edges[i]:
            if c in seen:continue
            seen.add(c)
            if c not in owner or augment(owner[c],seen):owner[c]=i;return True
        return False
    for i in range(len(anns)):augment(i,set())
    matched=np.zeros(len(anns),np.uint8)
    for i in owner.values():matched[i]=1
    unmatched_area=int(sum(areas[j+1] for j in range(n) if j not in owner))
    frame=np.array([len(anns),len(owner),n,n-len(owner),int(binary.sum()),int((binary&~gtmask).sum())],np.int64)
    return matched,frame,unmatched_area

def main():
    start=time.time();checks={};protocol=json.loads((L/'PROTOCOL.json').read_text());run=json.loads((L/'run/RUN.json').read_text())
    assert (L/'SUMMARY.json').is_file(),'Wait for completed scoring'
    def check(name,condition):assert condition,name;checks[name]=True
    check('frozen_trainer_byte_identical',(L/'code/train_pilot.py').read_bytes()==(P/'pilot/frozen_v1/train_pilot.py').read_bytes())
    old=(P/'pilot/frozen_v1/losses.py').read_text();new=(L/'code/losses.py').read_text()
    before='    total = pos + neg + extra  # Frozen lambda=1; no score-dependent sample inclusion.'
    after='    total = pos + neg  # Isolated lambda=0 control; extra is logged but excluded from optimization.'
    check('only_loss_change_is_remove_extra',old.count(before)==1 and old.replace(before,after)==new)
    check('effective_lambda_zero_sidecar',protocol['effective_extra_lambda']==0 and protocol['unchanged_legacy_plan_config_extra_lambda']==1)
    check('frozen_sources_unchanged',all(sha(p)==h for p,h in protocol['source_sha256'].items()))
    check('copied_sources_match_declared_hashes',all(sha(p)==h for p,h in protocol['copied_source_sha256'].items()))
    plan=json.loads((P/'pilot/TRAIN_PLAN.json').read_text());check('same_512_step_plan',len(plan['entries'])==512 and run['plan_sha256']==sha(P/'pilot/TRAIN_PLAN.json'))
    check('same_seed_update_budget_optimizer',run['status']=='complete' and run['steps']==run['final_step']==512 and run['seed']==46 and run['lr']==run['weight_decay']==1e-4 and run['optimizer']=='fresh Adam')
    check('same_released_initial_checkpoint',run['checkpoint_sha256']=='4fd4aa2736d9a33f2bf1459099068d8b4a7e202ff80aecdb4a3277fd13c7abcc' and sha(run['checkpoint_path'])==run['checkpoint_sha256'])
    check('unchanged_train_and_loss_sources',run['train_source_sha256']==sha(L/'code/train_pilot.py') and run['loss_source_sha256']==sha(L/'code/losses.py'))
    check('fixed_bn_hashes_equal',run['bn_initial_sha256']==run['bn_final_sha256'])
    logs=[json.loads(s) for s in (L/'run/TRAIN.jsonl').read_text().splitlines()]
    check('logged_steps_finite_and_total_is_base_loss',[r['step'] for r in logs]==list(range(1,512,16))+[512] and all(r['finite'] and math.isfinite(r['grad_norm']) and abs(r['loss']-r['positive']-r['negative'])<=1e-5 for r in logs))
    check('final_checkpoint_hash',sha(L/'run/final_model.pth')==run['final_checkpoint_sha256'])
    numpy_core=np._core if hasattr(np,'_core') else np.core
    with torch.serialization.safe_globals([(numpy_core.multiarray.scalar,'numpy.core.multiarray.scalar'),np.dtype,type(np.dtype(np.float64))]):initial=torch.load(run['checkpoint_path'],map_location='cpu',weights_only=True)
    final=torch.load(L/'run/final_model.pth',map_location='cpu',weights_only=True)
    bufnames=[k for k in initial['model_state_dict'] if k.endswith(('running_mean','running_var','num_batches_tracked'))]
    check('checkpoint_bn_buffers_equal',all(torch.equal(initial['model_state_dict'][k],final['model_state_dict'][k]) for k in bufnames))
    check('all_final_weights_finite',all(torch.isfinite(v).all() for v in final['model_state_dict'].values()))
    del initial,final
    cohorts=read(L.parent/'COHORTS.csv');cohortkeys={(r['sequence'],int(r['frame']),int(r['annotation_id'])) for r in cohorts}
    verified={};frame_count=0
    for seq in SEQS:
        data=json.loads((P/'data'/seq/'annotations.json').read_text());mapping={a['id']:i for i,a in enumerate(data['images'])};grouped=defaultdict(list)
        for a in data['annotations']:
            x,y,w,h=a['bbox']
            if a.get('ignore') or w<=0 or h<=0 or min(640,x+w)<=max(0,x) or min(512,y+h)<=max(0,y):continue
            grouped[mapping[a['image_id']]].append(a)
        for aa in grouped.values():aa.sort(key=lambda a:(a['track_id'],a['id']))
        cache=np.load(L/f'{seq}_scores.npz');small=np.array([(seq,t,a['id']) in cohortkeys for t in range(160) for a in grouped[t]])
        check(seq+'_small_geometry_rejoins_frozen_cohort',np.array_equal(small,cache['small']))
        complete=json.loads((L/'predictions'/seq/'COMPLETE.json').read_text())
        check(seq+'_inference_uses_final_checkpoint',complete['checkpoint_sha256']==sha(L/'run/final_model.pth') and complete['reference_checkpoint_sha256']==run['checkpoint_sha256'])
        for pi,(planname,starts) in enumerate(PLANS.items()):
            fused=np.zeros((160,512,640),np.float32);coverage=np.zeros(160,np.int16)
            for s in starts:
                path=L/'predictions'/seq/'windows'/f'window_{s:03d}.npz';z=np.load(path)
                assert z['probabilities'].shape==(40,512,640) and int(z['start'])==s and int(z['end'])==s+40
                assert sha(path)==complete['window_sha256'][path.name]
                fused[s:s+40]=np.maximum(fused[s:s+40],z['probabilities']);coverage[s:s+40]+=1
            check(seq+'_'+planname+'_five_window_full_coverage',np.array_equal(coverage,cache['coverage'][pi]) and coverage.sum()==200 and coverage.min()>0)
            totals=np.zeros(6,np.int64);matches=[];ua=0
            for t in range(160):
                mm,ff,area=measure(fused[t],grouped[t]);i0,i1=cache['offsets'][t:t+2]
                assert np.array_equal(mm,cache['matched'][pi,i0:i1]),(seq,planname,t,'matches')
                assert np.array_equal(ff,cache['frame_counts'][pi,t]),(seq,planname,t,'frame counts')
                matches.extend(mm);totals+=ff;ua+=area;frame_count+=1
            check(seq+'_'+planname+'_unmatched_component_area',ua==int(cache['unmatched_component_area'][pi]))
            matches=np.array(matches,dtype=bool);base=cache['reference_matched'][pi].astype(bool)
            verified[seq,planname]={'small_gt':int(small.sum()),'M':int((matches&small).sum()),
              'G':int((matches&~base&small).sum()),'L':int((~matches&base&small).sum()),'reference_small_matched':int((base&small).sum()),
              **{k:int(v) for k,v in zip(FIELDS,totals)},'unmatched_component_area':ua,
              'matches':matches,'small':small,'reference_O':cache['reference_matched'][0].astype(bool)}
            del fused
    metrics=read(L/'METRICS.csv');public=[]
    for row in metrics:
        seqs=SEQS if row['group']=='pooled' else [row['group']];planname=row['arrangement']
        expected={k:sum(verified[s,planname][k] for s in seqs) for k in ['small_gt','M','G','L','reference_small_matched']+FIELDS+['unmatched_component_area']}
        if row['reference']=='original_O':
            for k in ['G','L','reference_small_matched']:expected[k]=0
            for s in seqs:
                r=verified[s,planname];bm=r['reference_O'];am=r['matches'];sm=r['small']
                expected['G']+=int((am&~bm&sm).sum());expected['L']+=int((~am&bm&sm).sum());expected['reference_small_matched']+=int((bm&sm).sum())
        assert all(int(row[k])==v for k,v in expected.items()),row
        assert expected['M']==expected['reference_small_matched']+expected['G']-expected['L']
        assert float(row['threshold'])==.5 and int(row['effective_extra_lambda'])==0
        if row['group']=='pooled':public.append({k:row[k] for k in ['arrangement','reference','reference_small_matched','M','G','L','unmatched_components','positive_pixels_outside_boxes','passes_pooled_U_budget']})
    check('reference_baselines_700_720',sum(verified[s,'O']['reference_small_matched'] for s in SEQS)==700 and sum(verified[s,'S']['reference_small_matched'] for s in SEQS)==720)
    check('all_metric_rows_M_equals_reference_plus_G_minus_L',True)
    result={'status':'PASS','effective_extra_lambda':0,'threshold':.5,'metadata_resolution':'Copied legacy CONFIG.extra_lambda=1 and RUN.strategy=paired_mean are historical; exact one-line optimized loss and frozen effective-lambda sidecar establish lambda0_base.',
      'checks':checks,'bn_buffer_tensors_compared':len(bufnames),'independently_rescored_frames':frame_count,'metric_rows_checked':len(metrics),
      'pooled_results':public,'scope':'Weak-box base-loss control; not official Gaussian supervision reproduction or selected intervention. No threshold retuning.',
      'elapsed_seconds':time.time()-start,'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'source_sha256':{str(p.relative_to(L)):sha(p) for p in [L/'PROTOCOL.json',L/'run/RUN.json',L/'run/TRAIN.jsonl',L/'METRICS.csv',L/'SUMMARY.json',Path(__file__)]}}
    (OUT/'CONTROL_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'PASS','frames':frame_count,'metrics':public,'seconds':result['elapsed_seconds']}),flush=True)

if __name__=='__main__':main()
