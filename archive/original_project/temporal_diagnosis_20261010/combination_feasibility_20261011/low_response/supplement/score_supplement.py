#!/usr/bin/env python3
"""Frozen post-coarse-grid .41-.49 exploratory supplement; fixed O/S arrangements."""
import csv,hashlib,importlib.util,json,time
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent;L=OUT.parent;C=L.parent;Q=C.parent;P=Q/'stage1_goal';V=Q/'problem_validity_20261011'
SEQS=['DJI_0026_2','DJI_0067_2'];THRESHOLDS=[x/100 for x in range(40,51)]
FIELDS=['gt_count','matched_count','component_count','unmatched_components','positive_pixels','positive_pixels_outside_boxes']
SOURCES={}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 SOURCES[str(p)]=h.hexdigest();return h.hexdigest()
def module(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def csvwrite(name,rows):
 with (OUT/name).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 began=time.perf_counter();sha(Path(__file__));sha(C/'LOW_RESPONSE_GRID_SUPPLEMENT.md');sha(C/'LOW_RESPONSE_GRID_SUPPLEMENT_FREEZE.json');sha(L/'LOW_COHORT.csv')
 ids={(r['sequence'],int(r['frame']),int(r['annotation_id'])) for r in csv.DictReader((L/'LOW_COHORT.csv').open())}
 scorer=module(P/'pilot/score_pilot.py','supplement_scorer');helper,hp=scorer.helper();sha(hp);sha(P/'pilot/score_pilot.py')
 results={};endpoint=[];recovered=[]
 for seq in SEQS:
  path=P/'data'/seq/'annotations.json';sha(path);gt,meta,offsets,_=scorer.gt_data(path,helper)
  small=np.array([bool(r['strict_small']) for r in meta]);low=np.array([(seq,r['frame'],r['annotation_id']) in ids for r in meta]);n=len(meta)
  original=P/'evaluation/official'/seq/'official_fused_matches.npz';sha(original)
  with np.load(original) as z:baseline=z['matched'][z['thresholds'].tolist().index(.5)].astype(bool)
  for mode in ['O','S']:
   pmap=P/'predictions/official'/seq/'official_fused.npz' if mode=='O' else V/'simple_controls/symmetric5_probmax'/seq/'fused.npz'
   oldpath=original if mode=='O' else V/'simple_controls/symmetric5_probmax'/seq/'fused_matches.npz'
   sha(pmap);sha(oldpath)
   with np.load(pmap,allow_pickle=False) as z:prob=z['probabilities'];coverage=z['coverage'];starts=z['starts']
   assert prob.shape==(160,512,640) and (coverage>0).all() and coverage.sum()==200
   assert starts.tolist()==([0,36,72,108,120] if mode=='O' else [0,24,60,96,120])
   arrays=dict(matched=np.zeros((11,n),np.uint8),peak_x=np.full((11,n),np.nan,np.float32),peak_y=np.full((11,n),np.nan,np.float32),
    component_area=np.zeros((11,n),np.int32),box_max=np.zeros(n,np.float32),frame_counts=np.zeros((11,160,6),np.int64),
    small=small,low=low,offsets=offsets,baseline=baseline,thresholds=np.array(THRESHOLDS),coverage=coverage,starts=starts)
   for ti,threshold in enumerate(THRESHOLDS):
    for t in range(160):
     rr,stats,_=helper.match_frame(prob[t],gt[t],threshold)
     for idx,r in enumerate(rr,int(offsets[t])):
      arrays['matched'][ti,idx]=r['matched']
      if ti==0:arrays['box_max'][idx]=r['box_max']
      if r['matched']:
       for key in ['peak_x','peak_y','component_area']:arrays[key][ti,idx]=r[key]
     arrays['frame_counts'][ti,t]=[stats[k] for k in FIELDS]
    print(json.dumps(dict(sequence=seq,mode=mode,threshold=threshold,small_matched=int(arrays['matched'][ti,small].sum()),low_recovered=int(arrays['matched'][ti,low].sum()),U=int(arrays['frame_counts'][ti,:,3].sum()),seconds=time.perf_counter()-began)),flush=True)
    for idx in np.flatnonzero(low&arrays['matched'][ti].astype(bool)):
     r=meta[idx];recovered.append(dict(sequence=seq,mode=mode,threshold=threshold,frame=r['frame'],source_frame=r['source_frame'],annotation_id=r['annotation_id'],track_id=r['track_id'],box_max=float(arrays['box_max'][idx]),peak_x=float(arrays['peak_x'][ti,idx]),peak_y=float(arrays['peak_y'][ti,idx]),component_area=int(arrays['component_area'][ti,idx])))
   with np.load(oldpath,allow_pickle=False) as z:
    for ti in [0,10]:
     oi=z['thresholds'].tolist().index(THRESHOLDS[ti])
     for key in ['matched','peak_x','peak_y','component_area','frame_counts']:assert np.array_equal(arrays[key][ti],z[key][oi],equal_nan=True),(seq,mode,ti,key)
     assert np.array_equal(arrays['box_max'],z['box_max'],equal_nan=True)
     assert np.array_equal(small,z['strict_small']) and np.array_equal(offsets,z['offsets'])
     endpoint.append(dict(sequence=seq,mode=mode,threshold=THRESHOLDS[ti],all_arrays_equal=True))
   np.savez_compressed(OUT/(mode+'_'+seq+'_scores.npz'),**arrays);results[mode,seq]=arrays;del prob
 rows=[]
 for group,seqs in [('pooled',SEQS)]+[(s,[s]) for s in SEQS]:
  for mode in ['O','S']:
   for ti,threshold in enumerate(THRESHOLDS):
    counts=sum((results[mode,s]['frame_counts'][ti].sum(0) for s in seqs),np.zeros(6,np.int64));matched=low_recovered=gain=loss=smalln=lown=0
    for seq in seqs:
     d=results[mode,seq];small=d['small'];low=d['low'];m=d['matched'][ti].astype(bool);baseline=d['baseline']
     smalln+=int(small.sum());lown+=int(low.sum());matched+=int(m[small].sum());low_recovered+=int(m[low].sum());gain+=int((small&m&~baseline).sum());loss+=int((small&~m&baseline).sum())
    budget=4015 if group=='pooled' else {'DJI_0026_2':741,'DJI_0067_2':3274}[group]
    rows.append(dict(group=group,mode=mode,threshold=threshold,new_supplement_point=int(threshold not in [.4,.5]),small_gt=smalln,small_matched=matched,low_cohort_gt=lown,low_cohort_recovered=low_recovered,gained_vs_original05=gain,lost_vs_original05=loss,net_vs_original05=gain-loss,**{k:int(v) for k,v in zip(FIELDS,counts)},U_budget_reference=budget,U_within_budget=int(counts[3]<=budget),budget_role='frozen_pooled_primary' if group=='pooled' else 'per_sequence_secondary_flag'))
 csvwrite('THRESHOLD_CURVES.csv',rows);csvwrite('LOW_RECOVERY_OBJECTS.csv',recovered)
 best={}
 for mode in ['O','S']:
  r=[x for x in rows if x['group']=='pooled' and x['mode']==mode and x['U_within_budget']]
  best[mode]=dict(max_small_matched=max(x['small_matched'] for x in r),max_low_recovered=max(x['low_cohort_recovered'] for x in r),feasible_rows=r)
 summary=dict(status='COMPLETE',exploratory_after_viewing_coarse_grid=True,thresholds=THRESHOLDS,new_thresholds=THRESHOLDS[1:-1],fixed_pool_and_threshold05_combination_analysis_unchanged=True,
  endpoint_exact_checks=endpoint,small_gt=5200,low_cohort_gt=4342,pooled_U_budget=4015,selections=best,elapsed_seconds=time.perf_counter()-began,source_sha256=SOURCES,
  limitations=['Existing reused development clips; no independent confirmation','Only fixed nine-point supplementary grid, not exhaustive continuous threshold search','No change to primary .5 combination feasibility or background budget','Diagnostic actual component-peak matching, not official Pd/Fa'])
 (OUT/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({'pooled':rows[:22],'best':best},indent=2))
if __name__=='__main__':main()
