#!/usr/bin/env python3
"""Frozen .5 point matching; same existing helper and GT order, three weight states."""
import argparse, collections, csv, hashlib, importlib.util, json, time
from pathlib import Path
import numpy as np
L=Path(__file__).resolve().parents[1];Q=L.parent;P=Q/'stage1_goal';V=Q/'problem_validity_20261011';OLD=Q/'low_response_information_20261011/lambda0_control';E=L/'evaluation'
PLANS={'O':[0,36,72,108,120],'S':[0,24,60,96,120]};SEQS=['DJI_0026_2','DJI_0067_2'];CONDITIONS=['release','lambda0_mean','lambda0_top1']
FIELDS=['gt_count','matched_count','component_count','unmatched_components','positive_pixels','positive_pixels_outside_boxes']
ROOTS={'release':P/'predictions/official','lambda0_mean':OLD/'predictions','lambda0_top1':L/'predictions'}

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
    return h.hexdigest()
def module(p,name):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def csvwrite(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def jsonwrite(p,obj):p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def scorer():
    s=module(P/'pilot/score_pilot.py','base_scorer');h,hp=s.helper();return s,h,hp

def score(condition,seq):
    output=E/condition/seq;output.mkdir(parents=True,exist_ok=True)
    if (output/'COMPLETE.json').exists():raise FileExistsError(output)
    began=time.perf_counter();s,h,hp=scorer();ap=P/'data'/seq/'annotations.json';gt,meta,offsets,exc=s.gt_data(ap,h)
    small=np.array([bool(m['strict_small']) for m in meta]);n=len(meta);sources={str(ap):digest(ap),str(hp):digest(hp)}
    completion_path=ROOTS[condition]/seq/'COMPLETE.json';complete=json.loads(completion_path.read_text());assert complete['status']=='COMPLETE'
    assert complete['raw_sha256']==digest(P/'data'/seq/'raw_frames.npy')
    sources[str(completion_path)]=digest(completion_path);cache={}
    for start in sorted(set(sum(PLANS.values(),[]))):
        path=ROOTS[condition]/seq/'windows'/f'window_{start:03d}.npz';dh=digest(path)
        assert complete['window_sha256'][path.name]==dh
        with np.load(path,allow_pickle=False) as z:
            cache[start]=z['probabilities'];assert int(z['start'])==start and int(z['end'])==start+40
        assert cache[start].shape==(40,512,640) and cache[start].dtype==np.float32 and np.isfinite(cache[start]).all()
        sources[str(path)]=dh
    arrays=dict(matched=np.zeros((2,n),np.uint8),box_max=np.zeros((2,n),np.float32),peak_x=np.full((2,n),np.nan,np.float32),peak_y=np.full((2,n),np.nan,np.float32),component_area=np.zeros((2,n),np.int32),frame_counts=np.zeros((2,160,6),np.int64),coverage=np.zeros((2,160),np.int16),strict_small=small,offsets=offsets)
    frame_rows=[];area_arrays={};checks=[]
    for pi,(plan,starts) in enumerate(PLANS.items()):
        fused=np.zeros((160,512,640),np.float32)
        for start in starts:fused[start:start+40]=np.maximum(fused[start:start+40],cache[start]);arrays['coverage'][pi,start:start+40]+=1
        assert np.all(arrays['coverage'][pi]>0) and arrays['coverage'][pi].sum()==200
        if plan=='O':
            fp=ROOTS[condition]/seq/'official_fused.npz';sources[str(fp)]=digest(fp)
            assert complete['fused_sha256']==sources[str(fp)]
            with np.load(fp,allow_pickle=False) as z:assert np.array_equal(z['probabilities'],fused)
        all_areas=[];all_unmatched=[];area_offsets=[0]
        for t in range(160):
            rows,stats,comps=h.match_frame(fused[t],gt[t],.5)
            for idx,r in enumerate(rows,int(offsets[t])):
                assert (r['annotation_id'],r['track_id'])==(meta[idx]['annotation_id'],meta[idx]['track_id'])
                arrays['matched'][pi,idx]=r['matched'];arrays['box_max'][pi,idx]=r['box_max']
                if r['matched']:
                    for key in ['peak_x','peak_y','component_area']:arrays[key][pi,idx]=r[key]
            arrays['frame_counts'][pi,t]=[stats[k] for k in FIELDS]
            aa=[int(c['area']) for c in comps];uu=[int(c['unmatched_component']) for c in comps]
            all_areas.extend(aa);all_unmatched.extend(uu);area_offsets.append(len(all_areas));uareas=[a for a,u in zip(aa,uu) if u]
            assert sum(aa)==stats['positive_pixels'] and sum(uu)==stats['unmatched_components']
            frame_rows.append(dict(condition=condition,sequence=seq,arrangement=plan,frame=t,threshold=.5,coverage_windows=int(arrays['coverage'][pi,t]),**{k:int(stats[k]) for k in FIELDS},unmatched_component_area=sum(uareas),unmatched_component_max_area=max(uareas,default=0),component_max_area=max(aa,default=0)))
        area_arrays[plan+'_areas']=np.array(all_areas,np.int32);area_arrays[plan+'_unmatched']=np.array(all_unmatched,np.uint8);area_arrays[plan+'_offsets']=np.array(area_offsets,np.int64)
        if condition=='release':
            rp=(P/'evaluation/official'/seq/'official_fused_matches.npz') if plan=='O' else (V/'simple_controls/symmetric5_probmax'/seq/'fused_matches.npz')
            with np.load(rp,allow_pickle=False) as z:
                ti=z['thresholds'].tolist().index(.5);assert np.array_equal(z['strict_small'],small)
                for key in ['matched','frame_counts','component_area']:
                    assert np.array_equal(z[key][ti],arrays[key][pi]),(condition,seq,plan,key)
                for key in ['peak_x','peak_y']:assert np.array_equal(z[key][ti],arrays[key][pi],equal_nan=True),(condition,seq,plan,key)
            sources[str(rp)]=digest(rp);checks.append(plan+' existing release matches/frame counts/peaks/areas exact')
        elif condition=='lambda0_mean':
            rp=OLD/(seq+'_scores.npz')
            with np.load(rp,allow_pickle=False) as z:
                assert np.array_equal(z['small'],small)
                for key in ['matched','frame_counts','component_area','box_max']:assert np.array_equal(z[key][pi],arrays[key][pi]),(condition,seq,plan,key)
                for key in ['peak_x','peak_y']:assert np.array_equal(z[key][pi],arrays[key][pi],equal_nan=True)
                assert int(np.sum(area_arrays[plan+'_areas'][area_arrays[plan+'_unmatched'].astype(bool)]))==int(z['unmatched_component_area'][pi])
            sources[str(rp)]=digest(rp);checks.append(plan+' existing lambda0 matches/frame counts/peaks/areas/boxmax exact')
        print(json.dumps(dict(condition=condition,sequence=seq,arrangement=plan,M=int(arrays['matched'][pi,small].sum()),U=int(arrays['frame_counts'][pi,:,3].sum()),A=int(arrays['frame_counts'][pi,:,5].sum()),elapsed_seconds=time.perf_counter()-began)),flush=True)
        del fused
    np.savez_compressed(output/'scores.npz',**arrays,plans=np.array(list(PLANS)),threshold=np.array(.5))
    np.savez_compressed(output/'component_areas.npz',**area_arrays)
    csvwrite(output/'FRAME_COUNTS.csv',frame_rows)
    jsonwrite(output/'COMPLETE.json',dict(status='COMPLETE',condition=condition,sequence=seq,threshold=.5,objects=meta,excluded_annotations=exc,checks=checks,source_sha256=sources,script_sha256=digest(__file__),elapsed_seconds=time.perf_counter()-began,diagnostic_scope='Existing one-to-one component-peak-in-box matching; not official Pd/Fa.'))


def summary():
    records={};meta={};area={};frame_rows=[];input_hashes={}
    for condition in CONDITIONS:
        for seq in SEQS:
            out=E/condition/seq;cp=out/'COMPLETE.json';c=json.loads(cp.read_text());assert c['status']=='COMPLETE';meta[seq]=c['objects'];input_hashes[str(cp)]=digest(cp)
            with np.load(out/'scores.npz',allow_pickle=False) as z:records[condition,seq]={k:z[k] for k in z.files}
            with np.load(out/'component_areas.npz',allow_pickle=False) as z:area[condition,seq]={k:z[k] for k in z.files}
            with (out/'FRAME_COUNTS.csv').open() as f:frame_rows.extend(list(csv.DictReader(f)))
            for path,h in c['source_sha256'].items():assert digest(path)==h,path
    cohort_path=Q/'low_response_information_20261011/COHORTS.csv'
    with cohort_path.open() as f:cohort_rows=list(csv.DictReader(f))
    cohorts={(r['sequence'],int(r['frame']),int(r['annotation_id'])):r for r in cohort_rows};assert len(cohorts)==5200
    objects=[]
    for seq in SEQS:
        for idx,m in enumerate(meta[seq]):
            if not m['strict_small']:continue
            key=(seq,m['frame'],m['annotation_id']);c=cohorts[key];assert int(c['track_id'])==m['track_id']
            for pi,plan in enumerate(PLANS):
                r=dict(sequence=seq,track_id=m['track_id'],frame=m['frame'],source_frame=m['source_frame'],annotation_id=m['annotation_id'],arrangement=plan,threshold=.5,cohort=c['cohort'],**{key:int(c[key]) for key in ['low_all_05','extreme_all_001','low_nonextreme','both_fixed_success','extra_only','high_unmatched']})
                for condition in CONDITIONS:
                    a=records[condition,seq];assert a['strict_small'][idx]
                    for k in ['matched','box_max','peak_x','peak_y','component_area']:
                        x=a[k][pi,idx].item();r[condition+'_'+k]=x if np.isfinite(x) else ''
                assert r['release_matched']==int(c['original_matched' if plan=='O' else 'symmetric_matched'])
                for condition in ['lambda0_mean','lambda0_top1']:
                    r[condition+'_gain_vs_release_same']=int(r[condition+'_matched'] and not r['release_matched'])
                    r[condition+'_loss_vs_release_same']=int(not r[condition+'_matched'] and r['release_matched'])
                r['top1_gain_vs_mean_same']=int(r['lambda0_top1_matched'] and not r['lambda0_mean_matched']);r['top1_loss_vs_mean_same']=int(not r['lambda0_top1_matched'] and r['lambda0_mean_matched'])
                objects.append(r)
    assert len(objects)==10400
    for plan in PLANS:
        oo=[r for r in objects if r['arrangement']==plan]
        assert sum(r['low_all_05'] for r in oo)==4342 and sum(r['extreme_all_001'] for r in oo)==4170
    csvwrite(E/'OBJECT_COMPARISON.csv',objects);csvwrite(E/'FRAME_COUNTS.csv',frame_rows)
    summaries=[];tracks=[];distribution=[]
    for condition in CONDITIONS:
        for pi,plan in enumerate(PLANS):
            for group,seqs in [('pooled',SEQS)]+[(s,[s]) for s in SEQS]:
                oo=[r for r in objects if r['sequence'] in seqs and r['arrangement']==plan]
                totals=sum((records[condition,s]['frame_counts'][pi].sum(0) for s in seqs),np.zeros(6,np.int64));ref_totals=sum((records['release',s]['frame_counts'][pi].sum(0) for s in seqs),np.zeros(6,np.int64));mean_totals=sum((records['lambda0_mean',s]['frame_counts'][pi].sum(0) for s in seqs),np.zeros(6,np.int64))
                aa=np.concatenate([area[condition,s][plan+'_areas'] for s in seqs]);um=np.concatenate([area[condition,s][plan+'_unmatched'].astype(bool) for s in seqs]);ua=aa[um]
                after=np.array([r[condition+'_matched'] for r in oo],bool);ref=np.array([r['release_matched'] for r in oo],bool);mean=np.array([r['lambda0_mean_matched'] for r in oo],bool)
                g=after&~ref;l=~after&ref;assert after.sum()==ref.sum()+g.sum()-l.sum()
                low=np.array([r['low_all_05'] for r in oo],bool);ext=np.array([r['extreme_all_001'] for r in oo],bool)
                r=dict(condition=condition,arrangement=plan,group=group,threshold=.5,small_gt=len(oo),small_tracks=len({(o['sequence'],o['track_id']) for o in oo}),clips=len(seqs),M=int(after.sum()),release_M_same=int(ref.sum()),G_vs_release_same=int(g.sum()),L_vs_release_same=int(l.sum()),net_vs_release_same=int(g.sum()-l.sum()),mean_M_same=int(mean.sum()),G_vs_mean_same=int((after&~mean).sum()),L_vs_mean_same=int((~after&mean).sum()),**{k:int(x) for k,x in zip(FIELDS,totals)},U_budget=4015 if group=='pooled' else '',passes_pooled_U_budget=int(totals[3]<=4015) if group=='pooled' else '',release_U_same=int(ref_totals[3]),delta_U_vs_release_same=int(totals[3]-ref_totals[3]),release_A_same=int(ref_totals[5]),delta_A_vs_release_same=int(totals[5]-ref_totals[5]),mean_U_same=int(mean_totals[3]),U_ratio_vs_mean_same=float(totals[3]/mean_totals[3]),mean_A_same=int(mean_totals[5]),A_ratio_vs_mean_same=float(totals[5]/mean_totals[5]),low_all_05_gt=int(low.sum()),low_all_05_matched=int((low&after).sum()),low_all_05_matched_tracks=len({(o['sequence'],o['track_id']) for o,b in zip(oo,low&after) if b}),low_all_05_matched_clips=len({o['sequence'] for o,b in zip(oo,low&after) if b}),extreme_all_001_gt=int(ext.sum()),extreme_all_001_matched=int((ext&after).sum()),extreme_all_001_matched_tracks=len({(o['sequence'],o['track_id']) for o,b in zip(oo,ext&after) if b}),extreme_all_001_matched_clips=len({o['sequence'] for o,b in zip(oo,ext&after) if b}))
                original_o=np.concatenate([records['release',s]['matched'][0,records['release',s]['strict_small']].astype(bool) for s in seqs]);assert len(original_o)==len(after)
                r.update(release_M_O=int(original_o.sum()),G_vs_release_O=int((after&~original_o).sum()),L_vs_release_O=int((~after&original_o).sum()))
                for prefix,arr in [('component',aa),('unmatched_component',ua)]:
                    r[prefix+'_area_sum']=int(arr.sum());r[prefix+'_area_mean']=float(arr.mean()) if len(arr) else 0.;r[prefix+'_area_max']=int(arr.max()) if len(arr) else 0
                    for label,q in [('q50',.5),('q95',.95),('q99',.99)]:r[prefix+'_area_'+label]=float(np.quantile(arr,q)) if len(arr) else 0.
                release_aa=np.concatenate([area['release',s][plan+'_areas'] for s in seqs]);release_um=np.concatenate([area['release',s][plan+'_unmatched'].astype(bool) for s in seqs]);release_ua=int(release_aa[release_um].sum())
                r.update(release_unmatched_component_area_sum=release_ua,delta_unmatched_component_area_vs_release=int(ua.sum())-release_ua,recorded_background_costs_not_increased=int(totals[3]<=ref_totals[3] and totals[5]<=ref_totals[5] and int(ua.sum())<=release_ua),auxiliary_old_S_joint_pass=int(r['M']>=720 and r['L_vs_release_O']<=54 and totals[3]<=4015 and totals[5]<=38641) if group=='pooled' and plan=='S' else '')
                summaries.append(r)
                if group=='pooled':
                    for label,mask in [('gain_vs_release',g),('loss_vs_release',l),('low_recovered',low&after),('extreme_recovered',ext&after)]:
                        counts=collections.Counter((o['sequence'],o['track_id']) for o,b in zip(oo,mask) if b);ranked=sorted(counts.items(),key=lambda x:(-x[1],x[0]))
                        distribution.append(dict(condition=condition,arrangement=plan,outcome=label,frames=int(mask.sum()),tracks=len(counts),clips=len({k[0] for k in counts}),top1_frames=ranked[0][1] if ranked else 0,top5_frames=sum(v for k,v in ranked[:5]),top1_fraction=(ranked[0][1]/mask.sum()) if ranked else '',top5_fraction=(sum(v for k,v in ranked[:5])/mask.sum()) if ranked else '',ranked_tracks=json.dumps([dict(sequence=k[0],track_id=k[1],frames=v) for k,v in ranked],separators=(',',':'))))
            for seq,tid in sorted({(r['sequence'],r['track_id']) for r in objects}):
                oo=[r for r in objects if r['arrangement']==plan and r['sequence']==seq and r['track_id']==tid]
                counts=dict(condition=condition,arrangement=plan,sequence=seq,track_id=tid,target_frames=len(oo),M=sum(r[condition+'_matched'] for r in oo),release_M_same=sum(r['release_matched'] for r in oo),G_vs_release_same=sum(r[condition+'_matched'] and not r['release_matched'] for r in oo),L_vs_release_same=sum(not r[condition+'_matched'] and r['release_matched'] for r in oo),mean_M_same=sum(r['lambda0_mean_matched'] for r in oo),G_vs_mean_same=sum(r[condition+'_matched'] and not r['lambda0_mean_matched'] for r in oo),L_vs_mean_same=sum(not r[condition+'_matched'] and r['lambda0_mean_matched'] for r in oo),low_all_05_gt=sum(r['low_all_05'] for r in oo),low_all_05_matched=sum(r['low_all_05'] and r[condition+'_matched'] for r in oo),extreme_all_001_gt=sum(r['extreme_all_001'] for r in oo),extreme_all_001_matched=sum(r['extreme_all_001'] and r[condition+'_matched'] for r in oo))
                tracks.append(counts)
    csvwrite(E/'SUMMARY.csv',[r for r in summaries if r['group']=='pooled']);csvwrite(E/'SEQUENCE_SUMMARY.csv',[r for r in summaries if r['group']!='pooled']);csvwrite(E/'TRACK_SUMMARY.csv',tracks);csvwrite(E/'RECOVERY_CONCENTRATION.csv',distribution)
    assert [(r['M'],r['unmatched_components']) for r in summaries if r['group']=='pooled' and r['condition']=='release']==[(700,4015),(720,3967)]
    assert [(r['M'],r['unmatched_components']) for r in summaries if r['group']=='pooled' and r['condition']=='lambda0_mean']==[(3251,168161),(3279,168559)]
    input_hashes[str(cohort_path)]=digest(cohort_path)
    jsonwrite(E/'COMPLETE.json',dict(status='COMPLETE',threshold=.5,plans=PLANS,conditions=CONDITIONS,small_target_frames=5200,object_comparison_rows=len(objects),cohort_counts=dict(low_all_05=4342,extreme_all_001=4170),reference_recomputations_exact=True,main_rows=[r for r in summaries if r['group']=='pooled'],source_sha256=input_hashes,script_sha256=digest(__file__),scope='Frozen diagnostic point matching on two development clips; not official Pd/Fa, method novelty or temporal mechanism claim. Background A excludes original matching-valid GT boxes without adding a new mask rule. U counts unmatched components, not confirmed false positives.'))
    print(json.dumps({'status':'COMPLETE','rows':[r for r in summaries if r['group']=='pooled']}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--condition',choices=CONDITIONS);parser.add_argument('--sequence',choices=SEQS);parser.add_argument('--summary',action='store_true');args=parser.parse_args()
    if args.summary:summary()
    else:
        assert args.condition
        for seq in ([args.sequence] if args.sequence else SEQS):score(args.condition,seq)
