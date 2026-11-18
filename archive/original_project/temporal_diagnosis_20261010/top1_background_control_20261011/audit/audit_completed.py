#!/usr/bin/env python3
"""Independent training provenance and raw-probability result audit.

Does not import the author's inference/scoring or loss functions. Fresh CPU
component maxima use reduction (not scorer lexsort), then frozen matching rules.
"""
import csv, datetime, hashlib, json, math, time
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage
OUT=Path(__file__).resolve().parent;N=OUT.parent;Q=N.parent;P=Q/'stage1_goal'
OLD=Q/'low_response_information_20261011/lambda0_control';E=N/'evaluation'
PLANS={'O':[0,36,72,108,120],'S':[0,24,60,96,120]};SEQS=['DJI_0026_2','DJI_0067_2']
FIELDS=['gt_count','matched_count','component_count','unmatched_components','positive_pixels','positive_pixels_outside_boxes']
checks={}
def check(name,condition):
    assert condition,name
    checks[name]=True
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
    return h.hexdigest()
def rows(path):
    with Path(path).open() as f:return list(csv.DictReader(f))
def jload(p):return json.loads(Path(p).read_text())
def crop(box):
    x,y,w,h=box
    return max(0,min(640,math.ceil(x-.5))),max(0,min(512,math.ceil(y-.5))),max(0,min(640,math.ceil(x+w-.5))),max(0,min(512,math.ceil(y+h-.5)))
def annotations(seq):
    data=jload(P/'data'/seq/'annotations.json');images=data['images'];mapping={im['id']:t for t,im in enumerate(images)}
    frames=[[] for im in images];meta=[];offsets=[0]
    for a in data['annotations']:
        x,y,w,h=a['bbox']
        if a.get('ignore') or w<=0 or h<=0 or min(640,x+w)<=max(0,x) or min(512,y+h)<=max(0,y):continue
        frames[mapping[a['image_id']]].append(a)
    for t,anns in enumerate(frames):
        anns.sort(key=lambda a:(a['track_id'],a['id']))
        for a in anns:
            x,y,w,h=a['bbox'];overlap=any(b['id']!=a['id'] and max(x,b['bbox'][0])<min(x+w,b['bbox'][0]+b['bbox'][2]) and max(y,b['bbox'][1])<min(y+h,b['bbox'][1]+b['bbox'][3]) for b in anns)
            small=max(w,h)<=16 and x>=0 and y>=0 and x+w<=640 and y+h<=512 and not overlap
            meta.append(dict(frame=t,source_frame=int(Path(images[t]['file_name']).stem),annotation_id=a['id'],track_id=a['track_id'],small=small))
        offsets.append(len(meta))
    return frames,meta,np.array(offsets)

def measure(prob,anns):
    binary=prob>=.5;labels,n=ndimage.label(binary,structure=np.ones((3,3),np.uint8))
    loc=np.flatnonzero(binary);lab=labels.ravel()[loc]
    maxscores=np.full(n+1,-np.inf,np.float32);np.maximum.at(maxscores,lab,prob.ravel()[loc])
    eligible=loc[prob.ravel()[loc]==maxscores[lab]]
    first=np.full(n+1,prob.size,np.int64);np.minimum.at(first,labels.ravel()[eligible],eligible)
    py,px=np.divmod(first[1:],640);px=px+.5;py=py+.5
    areas=np.bincount(lab,minlength=n+1)[1:];edges=[];gtmask=np.zeros(prob.shape,bool);boxmax=[]
    for a in anns:
        x,y,w,h=a['bbox'];x0,y0,x1,y1=crop(a['bbox']);gtmask[y0:y1,x0:x1]=True
        b=prob[y0:y1,x0:x1];boxmax.append(float(b.max()) if b.size else np.nan)
        cs=np.flatnonzero((px>=x)&(px<x+w)&(py>=y)&(py<y+h)).tolist()
        cs.sort(key=lambda c:(((px[c]-x-w/2)/w)**2+((py[c]-y-h/2)/h)**2,-float(maxscores[c+1]),c+1))
        edges.append(cs)
    owner={}
    def augment(i,seen):
        for c in edges[i]:
            if c in seen:continue
            seen.add(c)
            if c not in owner or augment(owner[c],seen):owner[c]=i;return True
        return False
    for i in range(len(anns)):augment(i,set())
    match=np.zeros(len(anns),np.uint8);mx=np.full(len(anns),np.nan,np.float32);my=mx.copy();ma=np.zeros(len(anns),np.int32);um=np.ones(n,np.uint8)
    for c,i in owner.items():match[i]=1;mx[i]=px[c];my[i]=py[c];ma[i]=areas[c];um[c]=0
    stats=np.array([len(anns),len(owner),n,n-len(owner),binary.sum(),(binary&~gtmask).sum()],np.int64)
    return dict(matched=match,peak_x=mx,peak_y=my,component_area=ma,box_max=np.array(boxmax,np.float32),frame_counts=stats),areas,um

def training():
    run=jload(N/'training/run/RUN.json');oldrun=jload(OLD/'run/RUN.json');p=jload(N/'training/TRAINING_PROTOCOL.json')
    check('loss_mask_prelaunch_audit_pass',jload(OUT/'LOSS_MASK_AUDIT.json')['status']=='PASS')
    check('frozen_sources_and_codes_unchanged',all(sha(path)==h for path,h in {**p['sources_sha256'],**p['code_sha256']}.items()))
    common=['steps','lr','seed','optimizer','weight_decay','plan_sha256','selection_sha256','checkpoint_sha256','model_source_sha256','bn_initial_sha256','torch_version','checkpoint_path','source_path']
    check('initialization_and_every_shared_training_setting_matches_mean',all(run[k]==oldrun[k] for k in common))
    check('fresh_released_initialization_not_failed_mean_checkpoint',sha(run['checkpoint_path'])==run['checkpoint_sha256']=='4fd4aa2736d9a33f2bf1459099068d8b4a7e202ff80aecdb4a3277fd13c7abcc' and run['checkpoint_sha256']!=oldrun['final_checkpoint_sha256'])
    check('frozen_config_unchanged',run['frozen_plan_config']==oldrun['config'])
    check('complete_512_step_final_only',run['status']=='complete' and run['final_step']==run['steps']==512 and run['checkpoint_selection']==oldrun['checkpoint_selection'])
    check('zero_extra_unit_positive_negative_coefficients',run['effective_extra_lambda']==0 and run['positive_coefficient']==run['negative_coefficient']==1)
    check('final_checkpoint_digest_matches',sha(N/'training/run/final_model.pth')==run['final_checkpoint_sha256'])
    check('bn_metadata_equal',run['bn_initial_sha256']==run['bn_final_sha256'])
    numpy_core=np._core if hasattr(np,'_core') else np.core
    with torch.serialization.safe_globals([(numpy_core.multiarray.scalar,'numpy.core.multiarray.scalar'),np.dtype,type(np.dtype(np.float64))]):initial=torch.load(run['checkpoint_path'],map_location='cpu',weights_only=True)
    final=torch.load(N/'training/run/final_model.pth',map_location='cpu',weights_only=True)
    bufs=[k for k in initial['model_state_dict'] if k.endswith(('running_mean','running_var','num_batches_tracked'))]
    check('all_actual_bn_buffer_tensors_unchanged',all(torch.equal(initial['model_state_dict'][k],final['model_state_dict'][k]) for k in bufs))
    check('all_final_parameter_tensors_finite',all(torch.isfinite(v).all() for v in final['model_state_dict'].values()))
    check('checkpoint_training_identity',final['step']==512 and final['initial_checkpoint_sha256']==run['checkpoint_sha256'] and final['effective_extra_lambda']==0 and final['background_top_fraction']==.01)
    del initial,final
    logs=[json.loads(s) for s in (N/'training/run/TRAIN.jsonl').read_text().splitlines()];oldlogs={r['step']:r for r in [json.loads(s) for s in (OLD/'run/TRAIN.jsonl').read_text().splitlines()]}
    check('all_512_updates_logged_in_order',[r['step'] for r in logs]==list(range(1,513)))
    fracs=[]
    for r in logs:
        assert r['finite'] and all(math.isfinite(r[k]) for k in ['loss','positive','negative','grad_norm','background_full_mean','background_top1pct_mean'])
        assert math.isclose(r['loss'],r['positive']+r['negative'],abs_tol=2e-6,rel_tol=2e-6)
        assert r['weighted_positive']==r['positive'] and r['weighted_negative']==r['negative']
        assert r['positive_coefficient']==r['negative_coefficient']==1 and r['effective_extra_lambda']==0
        assert r['negative']==r['background_top1pct_mean']
        assert r['background_top1pct_mean']>=r['background_full_mean']-1e-7
        Ns=r['background_N_per_window'];ks=r['background_k_per_window'];fs=r['background_selected_fraction_per_window']
        assert len(Ns)==len(ks)==len(fs)==4 and Ns==r['background_pixels_per_window']
        assert ks==[max(1,math.ceil(n/100)) if n else 0 for n in Ns]
        assert fs==[k/n if n else None for n,k in zip(Ns,ks)]
        assert r['background_selected_fraction_pooled']==sum(ks)/sum(Ns)
        for f in fs:
            if f is not None:fracs.append(f)
        if r['step'] in oldlogs:
            old=oldlogs[r['step']]
            assert all(r[k]==old[k] for k in ['background_pixels_per_window','base_objects_per_window','extra_objects','extra_disabled_n_lt_2'])
    check('every_update_finite_complete_losses_and_correct_selection_fraction',True)
    check('same_background_and_positive_object_counts_as_old_logged_batches',True)
    check('first_update_identical_positive_and_full_background_mean',logs[0]['positive']==oldlogs[1]['positive'] and logs[0]['background_full_mean']==oldlogs[1]['negative'])
    return dict(bn_tensors=len(bufs),logged_updates=len(logs),fraction_min=min(fracs),fraction_max=max(fracs),first_update=logs[0],last_update=logs[-1],run_sha256=sha(N/'training/run/RUN.json'))

def evaluate():
    ecomp=jload(E/'COMPLETE.json');check('evaluation_complete',ecomp['status']=='COMPLETE')
    original_infer=(P/'pilot/infer_pilot.py').read_text();new_infer=(N/'evaluation_code/infer_top1.py').read_text()
    check('inference_only_change_is_eight_O_S_union_windows',original_infer.replace('starts = list(range(0,121,12))','starts = [0,24,36,60,72,96,108,120]')==new_infer)
    cohortrows=rows(Q/'low_response_information_20261011/COHORTS.csv');cohorts={(r['sequence'],int(r['frame']),int(r['annotation_id'])):r for r in cohortrows}
    check('frozen_cohort5200_low4342_extreme4170',len(cohorts)==5200 and sum(int(r['low_all_05']) for r in cohortrows)==4342 and sum(int(r['extreme_all_001']) for r in cohortrows)==4170)
    verified={};objects={};frame_count=0
    for seq in SEQS:
        anns,meta,offsets=annotations(seq);out=E/'lambda0_top1'/seq;cache=np.load(out/'scores.npz');areas_saved=np.load(out/'component_areas.npz');ec=jload(out/'COMPLETE.json')
        small=np.array([m['small'] for m in meta]);check(seq+'_independent_GT_order_and_small',np.array_equal(small,cache['strict_small']) and np.array_equal(offsets,cache['offsets']) and all((m['frame'],m['annotation_id'],m['track_id'])==(o['frame'],o['annotation_id'],o['track_id']) for m,o in zip(meta,ec['objects'])))
        check(seq+'_frozen_cohort_rejoin',all(((seq,m['frame'],m['annotation_id']) in cohorts)==bool(m['small']) for m in meta))
        complete=jload(N/'predictions'/seq/'COMPLETE.json');checkpoint=sha(N/'training/run/final_model.pth')
        check(seq+'_inference_checkpoint',complete['checkpoint_sha256']==checkpoint and complete['reference_checkpoint_sha256']=='4fd4aa2736d9a33f2bf1459099068d8b4a7e202ff80aecdb4a3277fd13c7abcc')
        check(seq+'_inference_same_model_source',complete['model_source_sha256']==jload(N/'training/run/RUN.json')['model_source_sha256'])
        refs={condition:np.load(E/condition/seq/'scores.npz') for condition in ['release','lambda0_mean']}
        for pi,(plan,starts) in enumerate(PLANS.items()):
            fused=np.zeros((160,512,640),np.float32);cover=np.zeros(160,np.int16)
            for s in starts:
                path=N/'predictions'/seq/'windows'/f'window_{s:03d}.npz';z=np.load(path)
                assert z['probabilities'].shape==(40,512,640) and z['probabilities'].dtype==np.float32 and int(z['start'])==s and int(z['end'])==s+40
                assert sha(path)==complete['window_sha256'][path.name]
                fused[s:s+40]=np.maximum(fused[s:s+40],z['probabilities']);cover[s:s+40]+=1
            assert np.array_equal(cover,cache['coverage'][pi]) and cover.min()>0 and cover.sum()==200
            if plan=='O':
                with np.load(N/'predictions'/seq/'official_fused.npz') as z:assert np.array_equal(fused,z['probabilities'])
            check(seq+'_'+plan+'_raw_probability_max_fullcoverage',True)
            match=[];frame_stats=[];allareas=[];allunmatched=[]
            for t in range(160):
                got,areas,um=measure(fused[t],anns[t]);i,j=offsets[t:t+2]
                for field in ['matched','peak_x','peak_y','component_area','box_max']:assert np.array_equal(got[field],cache[field][pi,i:j],equal_nan=True),(seq,plan,t,field)
                assert np.array_equal(got['frame_counts'],cache['frame_counts'][pi,t]),(seq,plan,t,'frame_counts')
                ao=areas_saved[plan+'_offsets'];aa,bb=ao[t:t+2]
                assert np.array_equal(areas,areas_saved[plan+'_areas'][aa:bb]) and np.array_equal(um,areas_saved[plan+'_unmatched'][aa:bb])
                match.extend(got['matched']);frame_stats.append(got['frame_counts']);allareas.extend(areas);allunmatched.extend(um);frame_count+=1
            match=np.array(match,bool);allareas=np.array(allareas);allunmatched=np.array(allunmatched,bool);counts=np.sum(frame_stats,axis=0);base=refs['release']['matched'][pi].astype(bool);mean=refs['lambda0_mean']['matched'][pi].astype(bool)
            verified[seq,plan]=dict(M=int((match&small).sum()),G=int((match&~base&small).sum()),L=int((~match&base&small).sum()),**{k:int(v) for k,v in zip(FIELDS,counts)},all_areas=allareas,unmatched_areas=allareas[allunmatched])
            for idx,m in enumerate(meta):
                if not small[idx]:continue
                key=(seq,m['frame'],m['annotation_id'],plan);c=cohorts[key[:3]]
                objects[key]=dict(sequence=seq,track_id=m['track_id'],frame=m['frame'],annotation_id=m['annotation_id'],arrangement=plan,release_matched=int(base[idx]),lambda0_mean_matched=int(mean[idx]),lambda0_top1_matched=int(match[idx]),low_all_05=int(c['low_all_05']),extreme_all_001=int(c['extreme_all_001']))
            check(seq+'_'+plan+'_160frames_actual_match_peak_area_boxscore_and_components',True)
            del fused
    table=rows(E/'OBJECT_COMPARISON.csv');check('10400_object_rows',len(table)==10400 and len(objects)==10400)
    for r in table:
        key=(r['sequence'],int(r['frame']),int(r['annotation_id']),r['arrangement']);o=objects[key]
        for f in ['track_id','release_matched','lambda0_mean_matched','lambda0_top1_matched','low_all_05','extreme_all_001']:assert int(r[f])==o[f],(key,f)
        for c in ['lambda0_mean','lambda0_top1']:
            assert int(r[c+'_gain_vs_release_same'])==int(o[c+'_matched'] and not o['release_matched'])
            assert int(r[c+'_loss_vs_release_same'])==int(not o[c+'_matched'] and o['release_matched'])
    check('all_objects_and_gain_loss_flags_verified',True)
    pooled=rows(E/'SUMMARY.csv');separate=rows(E/'SEQUENCE_SUMMARY.csv')
    check('six_pooled_and_twelve_sequence_reference_rows',len(pooled)==6 and len(separate)==12)
    baseline_expected={('release','O'):(700,4015),('release','S'):(720,3967),('lambda0_mean','O'):(3251,168161),('lambda0_mean','S'):(3279,168559)}
    for r in pooled:
        if (r['condition'],r['arrangement']) in baseline_expected:assert (int(r['M']),int(r['unmatched_components']))==baseline_expected[r['condition'],r['arrangement']]
    check('all_four_release_and_failed_mean_baselines_present_and_exact',True)
    summaries=pooled+separate;public=[]
    for r in summaries:
        seqs=SEQS if r['group']=='pooled' else [r['group']];plan=r['arrangement'];oo=[o for o in objects.values() if o['sequence'] in seqs and o['arrangement']==plan]
        ck=r['condition']+'_matched'
        assert int(r['M'])==sum(o[ck] for o in oo)
        assert int(r['G_vs_release_same'])==sum(o[ck] and not o['release_matched'] for o in oo)
        assert int(r['L_vs_release_same'])==sum(not o[ck] and o['release_matched'] for o in oo)
        assert int(r['G_vs_mean_same'])==sum(o[ck] and not o['lambda0_mean_matched'] for o in oo)
        assert int(r['L_vs_mean_same'])==sum(not o[ck] and o['lambda0_mean_matched'] for o in oo)
        original_o=[objects[o['sequence'],o['frame'],o['annotation_id'],'O']['release_matched'] for o in oo]
        assert int(r['release_M_O'])==sum(original_o)
        assert int(r['G_vs_release_O'])==sum(o[ck] and not b for o,b in zip(oo,original_o))
        assert int(r['L_vs_release_O'])==sum(not o[ck] and b for o,b in zip(oo,original_o))
        for flag in ['low_all_05','extreme_all_001']:
            rec=[o for o in oo if o[flag] and o[ck]]
            assert int(r[flag+'_gt'])==sum(o[flag] for o in oo)
            assert int(r[flag+'_matched'])==len(rec)
            assert int(r[flag+'_matched_tracks'])==len({(o['sequence'],o['track_id']) for o in rec})
            assert int(r[flag+'_matched_clips'])==len({o['sequence'] for o in rec})
        if r['condition']!='lambda0_top1':continue
        for col,vkey in [('M','M'),('G_vs_release_same','G'),('L_vs_release_same','L')]+[(f,f) for f in FIELDS]:assert int(r[col])==sum(verified[s,plan][vkey] for s in seqs),(plan,col)
        assert int(r['M'])==int(r['release_M_same'])+int(r['G_vs_release_same'])-int(r['L_vs_release_same'])
        for flag in ['low_all_05','extreme_all_001']:
            recovered=[o for o in oo if o[flag] and o['lambda0_top1_matched']]
            assert int(r[flag+'_gt'])==sum(o[flag] for o in oo)
            assert int(r[flag+'_matched'])==len(recovered)
            assert int(r[flag+'_matched_tracks'])==len({(o['sequence'],o['track_id']) for o in recovered})
            assert int(r[flag+'_matched_clips'])==len({o['sequence'] for o in recovered})
        for prefix,key in [('component','all_areas'),('unmatched_component','unmatched_areas')]:
            aa=np.concatenate([verified[s,plan][key] for s in seqs])
            for suffix,want in [('sum',aa.sum()),('max',aa.max() if len(aa) else 0),('mean',aa.mean() if len(aa) else 0),('q50',np.quantile(aa,.5) if len(aa) else 0),('q95',np.quantile(aa,.95) if len(aa) else 0),('q99',np.quantile(aa,.99) if len(aa) else 0)]:assert math.isclose(float(r[prefix+'_area_'+suffix]),float(want),rel_tol=1e-12,abs_tol=1e-12)
        if r['group']=='pooled':public.append(r)
    check('all_eighteen_summary_detection_and_recovery_counts_verified',True)
    check('all_top1_summary_metrics_and_cohort_tracks_and_component_distributions',True)
    tr=rows(E/'TRACK_SUMMARY.csv');checked_tracks=0
    for r in tr:
        oo=[o for o in objects.values() if o['sequence']==r['sequence'] and o['arrangement']==r['arrangement'] and o['track_id']==int(r['track_id'])]
        ck=r['condition']+'_matched'
        expect=dict(target_frames=len(oo),M=sum(o[ck] for o in oo),release_M_same=sum(o['release_matched'] for o in oo),G_vs_release_same=sum(o[ck] and not o['release_matched'] for o in oo),L_vs_release_same=sum(not o[ck] and o['release_matched'] for o in oo),low_all_05_gt=sum(o['low_all_05'] for o in oo),low_all_05_matched=sum(o['low_all_05'] and o[ck] for o in oo),extreme_all_001_gt=sum(o['extreme_all_001'] for o in oo),extreme_all_001_matched=sum(o['extreme_all_001'] and o[ck] for o in oo))
        assert all(int(r[k])==v for k,v in expect.items()),(r,expect);checked_tracks+=1
    check('780_track_rows_all_three_conditions_independently_aggregated',checked_tracks==780)
    for condition in ['release','lambda0_mean','lambda0_top1']:
        for seq in SEQS:
            cc=jload(E/condition/seq/'COMPLETE.json')
            assert all(sha(p)==h for p,h in cc['source_sha256'].items())
    check('all_evaluation_source_hashes_unchanged',True)
    check('evaluation_script_unchanged',sha(N/'evaluation_code/evaluate.py')==ecomp['script_sha256'])
    return dict(independent_frames=frame_count,object_rows=len(table),track_rows=checked_tracks,pooled_top1=public)

def outcome():
    out=jload(N/'OUTCOME.json');ss={(r['condition'],r['arrangement']):r for r in rows(E/'SUMMARY.csv')}
    check('outcome_scope_and_fixed_settings',out['status']=='COMPLETE' and out['one_seed']==46 and out['updates']==512 and out['fixed_top_fraction']==.01 and out['threshold']==.5 and out['plans']==PLANS)
    for o in out['outcomes']:
        plan=o['arrangement'];new=ss['lambda0_top1',plan];mean=ss['lambda0_mean',plan];ref=ss['release',plan]
        ni=lambda k:int(new[k]);mi=lambda k:int(mean[k]);ri=lambda k:int(ref[k])
        costs=['unmatched_components','positive_pixels_outside_boxes','unmatched_component_area_sum']
        dr={k:ni(k)-ri(k) for k in costs};dm={k:ni(k)-mi(k) for k in costs}
        assert (o['M'],o['U'],o['A'])==(ni('M'),ni('unmatched_components'),ni('positive_pixels_outside_boxes'))
        assert o['budget_pass']==(ni('unmatched_components')<=4015)
        assert o['recorded_background_deltas_vs_release']==dr and o['background_deltas_vs_mean']==dm
        assert o['recorded_background_costs_not_increased_vs_release']==all(v<=0 for v in dr.values())
        assert o['recorded_background_costs_reduced_vs_mean']==(all(v<=0 for v in dm.values()) and any(v<0 for v in dm.values()))
        assert o['delta_M_vs_release']==ni('M')-ri('M')
        assert o['gain_vs_release_same']==ni('G_vs_release_same') and o['loss_vs_release_same']==ni('L_vs_release_same')
        assert o['low_recovered']==ni('low_all_05_matched') and o['extreme_recovered']==ni('extreme_all_001_matched')
        assert o['exceeds_shared_720_within_U_budget']==(ni('unmatched_components')<=4015 and ni('M')>720)
        assert o['preserves_more_original_O_successes_than_release_S']==(plan=='S' and ni('L_vs_release_O')<54)
        assert o['auxiliary_old_S_joint_pass']==((ni('unmatched_components')<=4015 and ni('M')>=720 and ni('L_vs_release_O')<=54 and ni('positive_pixels_outside_boxes')<=38641) if plan=='S' else None)
        assert o['no_official_Pd_Fa_claim'] and o['no_temporal_causal_claim'] and o['no_novelty_claim']
    check('outcomes_separate_budget_from_same_arrangement_costs_and720_and_retention',True)
    check('auxiliary_joint_gate_not_substituted_for_separate_flags',True)
    return {'outcome_sha256':sha(N/'OUTCOME.json'),'summarizer_sha256':sha(N/'summarize_trial.py'),'review':'M/U budget flags, same-arrangement three recorded cost deltas, gain/loss, frozen cohort recovery, original O retention, shared720 and auxiliary old joint gate map correctly. Three cost aggregate flags do not claim all component area quantiles decreased.'}

def main():
    start=time.perf_counter();tr=training();ev=evaluate();dec=outcome()
    result=dict(status='PASS',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-start,checks=checks,training=tr,evaluation=ev,decision_mapping=dec,scope='Two existing development clips, frozen0.5 diagnostic matching; no official Pd/Fa, novelty or temporal causality inferred.',auditor_source_sha256=sha(__file__))
    (OUT/'COMPLETED_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',checks=len(checks),frames=ev['independent_frames'],pooled=[{k:r[k] for k in ['arrangement','M','G_vs_release_same','L_vs_release_same','unmatched_components','positive_pixels_outside_boxes','low_all_05_matched','extreme_all_001_matched']} for r in ev['pooled_top1']])),flush=True)
if __name__=='__main__':main()
