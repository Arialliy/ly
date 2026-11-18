import csv,json,hashlib,importlib.util,time
from pathlib import Path
import numpy as np
L=Path(__file__).resolve().parent;Q=L.parent.parent;P=Q/'stage1_goal';V=Q/'problem_validity_20261011'
PLANS={'O':[0,36,72,108,120],'S':[0,24,60,96,120]};SEQS=['DJI_0026_2','DJI_0067_2']
FIELDS=['gt_count','matched_count','component_count','unmatched_components','positive_pixels','positive_pixels_outside_boxes']
def module(p,name):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def csvwrite(name,rows):
    with (L/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    began=time.perf_counter();scorer=module(P/'pilot/score_pilot.py','scorer');helper,hp=scorer.helper();metrics=[];objects=[];records={};sources={}
    for seq in SEQS:
        ap=P/'data'/seq/'annotations.json';gt,meta,offsets,exc=scorer.gt_data(ap,helper);small=np.array([bool(m['strict_small']) for m in meta]);n=len(meta)
        cache={}
        for start in sorted(set(sum(PLANS.values(),[]))):
            path=L/'predictions'/seq/'windows'/f'window_{start:03d}.npz'
            with np.load(path) as z:cache[start]=z['probabilities'];assert int(z['start'])==start
            sources[str(path)]=sha(path)
        arrays=dict(matched=np.zeros((2,n),np.uint8),box_max=np.zeros((2,n),np.float32),peak_x=np.full((2,n),np.nan,np.float32),peak_y=np.full((2,n),np.nan,np.float32),component_area=np.zeros((2,n),np.int32),frame_counts=np.zeros((2,160,6),np.int64),coverage=np.zeros((2,160),np.int16),small=small,offsets=offsets,reference_matched=np.zeros((2,n),np.uint8),reference_frame_counts=np.zeros((2,160,6),np.int64),component_area_sum=np.zeros(2,np.int64),unmatched_component_area=np.zeros(2,np.int64))
        for pi,(plan,starts) in enumerate(PLANS.items()):
            fused=np.zeros((160,512,640),np.float32)
            for s in starts:fused[s:s+40]=np.maximum(fused[s:s+40],cache[s]);arrays['coverage'][pi,s:s+40]+=1
            assert (arrays['coverage'][pi]>0).all() and arrays['coverage'][pi].sum()==200
            if plan=='O':
                with np.load(L/'predictions'/seq/'official_fused.npz') as z:assert np.array_equal(z['probabilities'],fused)
            for t in range(160):
                rr,stats,comps=helper.match_frame(fused[t],gt[t],.5)
                for index,r in enumerate(rr,int(offsets[t])):
                    arrays['matched'][pi,index]=r['matched'];arrays['box_max'][pi,index]=r['box_max']
                    if r['matched']:
                        for key in ['peak_x','peak_y','component_area']:arrays[key][pi,index]=r[key]
                arrays['frame_counts'][pi,t]=[stats[k] for k in FIELDS]
                arrays['component_area_sum'][pi]+=sum(c['area'] for c in comps)
                arrays['unmatched_component_area'][pi]+=sum(c['area'] for c in comps if c['unmatched_component'])
            ref=(P/'evaluation/official'/seq/'official_fused_matches.npz') if plan=='O' else (V/'simple_controls/symmetric5_probmax'/seq/'fused_matches.npz')
            with np.load(ref) as z:
                ti=z['thresholds'].tolist().index(.5);assert np.array_equal(z['strict_small'],small)
                arrays['reference_matched'][pi]=z['matched'][ti];arrays['reference_frame_counts'][pi]=z['frame_counts'][ti]
            sources[str(ref)]=sha(ref)
            print(json.dumps({'seq':seq,'plan':plan,'M':int(arrays['matched'][pi,small].sum()),'U':int(arrays['frame_counts'][pi,:,3].sum()),'A':int(arrays['frame_counts'][pi,:,5].sum()),'seconds':time.perf_counter()-began}),flush=True)
            for idx,m in enumerate(meta):
                objects.append(dict(sequence=seq,arrangement=plan,frame=m['frame'],source_frame=m['source_frame'],annotation_id=m['annotation_id'],track_id=m['track_id'],strict_small=int(small[idx]),original_weight_matched=int(arrays['reference_matched'][pi,idx]),lambda0_matched=int(arrays['matched'][pi,idx]),lambda0_box_max=float(arrays['box_max'][pi,idx])))
        np.savez_compressed(L/(seq+'_scores.npz'),**arrays,plans=np.array(list(PLANS)),threshold=np.array(.5));records[seq]=arrays
    for group,seqs in [('pooled',SEQS)]+[(s,[s]) for s in SEQS]:
        for pi,plan in enumerate(PLANS):
            totals=sum((records[s]['frame_counts'][pi].sum(0) for s in seqs),np.zeros(6,np.int64));refs=sum((records[s]['reference_frame_counts'][pi].sum(0) for s in seqs),np.zeros(6,np.int64))
            for reference,ri in [('same_arrangement',pi)]+([('original_O',0)] if pi==1 else []):
                M=G=Lost=N=before=0
                for s in seqs:
                    a=records[s];mask=a['small'];after=a['matched'][pi].astype(bool);bm=a['reference_matched'][ri].astype(bool)
                    M+=int((mask&after).sum());G+=int((mask&after&~bm).sum());Lost+=int((mask&~after&bm).sum());N+=int(mask.sum());before+=int((mask&bm).sum())
                assert M==before+G-Lost
                metrics.append(dict(group=group,arrangement=plan,reference=reference,threshold=.5,effective_extra_lambda=0,small_gt=N,reference_small_matched=before,M=M,G=G,L=Lost,net_gain=G-Lost,**{k:int(v) for k,v in zip(FIELDS,totals)},U_budget=4015 if group=='pooled' else '',passes_pooled_U_budget=int(totals[3]<=4015) if group=='pooled' else '',reference_U_same_arrangement=int(refs[3]),reference_A_same_arrangement=int(refs[5]),unmatched_component_area=int(sum(records[s]['unmatched_component_area'][pi] for s in seqs))))
    csvwrite('METRICS.csv',metrics);csvwrite('OBJECTS.csv',objects)
    summary=dict(status='COMPLETE',effective_extra_lambda=0,threshold=.5,metrics=metrics,small_target_frames=5200,diagnostic_gt_frames=len(objects)//2,elapsed_seconds=time.perf_counter()-began,source_sha256=sources,old_source_unchanged=True,scope='Same weak-box base loss with extra lambda=0; not official Gaussian training reproduction. Fixed .5 original point matching; no threshold tuning.')
    protocol=json.loads((L/'PROTOCOL.json').read_text())
    for path,h in protocol['source_sha256'].items():assert sha(Path(path))==h,path
    run=json.loads((L/'run/RUN.json').read_text());assert run['status']=='complete' and run['final_step']==512 and run['bn_initial_sha256']==run['bn_final_sha256']
    logs=[json.loads(x) for x in (L/'run/TRAIN.jsonl').read_text().splitlines()];assert all(x['finite'] and abs(x['loss']-x['positive']-x['negative'])<=1e-5 for x in logs)
    summary['training_checks']=dict(final_step=512,finite_logged_steps=True,bn_buffers_equal=True,logged_total_equals_pos_plus_neg=True,unchanged_original_sources=True,train_seconds=run['duration_seconds'])
    (L/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({'complete':True,'pooled_metrics':[r for r in metrics if r['group']=='pooled']}),flush=True)
if __name__=='__main__':main()
