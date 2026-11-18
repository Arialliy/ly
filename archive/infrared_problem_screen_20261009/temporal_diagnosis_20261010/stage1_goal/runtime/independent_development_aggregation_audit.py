from pathlib import Path
import json,csv,collections,hashlib,math
import numpy as np
B=Path('/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/temporal_diagnosis_20261010/stage1_goal');Q=B.parent
V=['paired_mean','paired_max'];SEQS=['DJI_0028_2','DJI_0028_4','DJI_0032_5','DJI_0047_3']
read=lambda p:list(csv.DictReader(p.open()));sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
key=lambda r:(int(r['local_frame'] if 'local_frame' in r else r['frame']),int(r['annotation_id']))
fullkey=lambda r:(r['sequence'],*key(r),int(r['track_id']))
cohorts=[r for r in read(Q/'result_stratification/context/same_gt_frame_choices.csv') if float(r['threshold'])==.5 and int(r['strict']) and int(r['multi']) and float(r['max_side'])<=16]
support={fullkey(r):r for r in read(Q/'mechanism_followup/SUPPORT_PAIRED.csv')}
assert len(cohorts)==len({fullkey(r) for r in cohorts})==1417
rows=[];frame_totals=[];hashes={};issues=[]
def getnew(v,seq,name):
 d=B/'diagnostic_evaluation'/v/seq;mp=d/(name+'_COMPLETE.json');zp=d/(name+'_matches.npz');meta=json.loads(mp.read_text());z=np.load(zp,allow_pickle=False)
 assert meta['status']=='COMPLETE' and meta['name']==name;ti=z['thresholds'].tolist().index(.5);objects=meta['objects'];assert len(objects)==z['matched'].shape[1]
 hashes[str(mp)]=sha(mp);hashes[str(zp)]=sha(zp)
 ans={}
 for i,o in enumerate(objects):
  k=key(o);assert k not in ans
  ans[k]={**o,'matched':int(z['matched'][ti,i]),'box_max':float(z['box_max'][i]),'peak_x':float(z['peak_x'][ti,i]),'peak_y':float(z['peak_y'][ti,i]),'center_error':float(z['center_error'][ti,i])}
 counts=z['frame_counts'][ti].sum(0)
 assert int(z['matched'][ti].sum())==int(counts[1])
 return ans,dict(zip(['gt','matched','components','unmatched_components','positive_pixels','positive_pixels_outside_boxes'],map(int,counts)))

for seq in SEQS:
 fp=Q/'analysis'/seq/'per_track_frame.csv';cp=Q/'window_offset_experiment/analysis'/seq/'per_context_all_tracks.csv';cntp=Q/'analysis'/seq/'per_frame_counts.csv'
 for p in [fp,cp,cntp]:hashes[str(p)]=sha(p)
 old={key(r):r for r in read(fp) if float(r['threshold'])==.5};ctx={}
 for r in read(cp):
  if float(r['threshold'])==.5:ctx[int(r['window_start']),*key(r)]=r
 crows=[r for r in read(cntp) if float(r['threshold'])==.5];assert len(crows)==160
 frame_totals.append({'variant':'official','sequence':seq,'frames':160,**{k:sum(int(r[f]) for r in crows) for k,f in [('matched','matched_count'),('components','component_count'),('unmatched_components','unmatched_components'),('positive_pixels_outside_boxes','positive_pixels_outside_boxes')]}})
 for v in V:
  fused,counts=getnew(v,seq,'official_fused');windows={s:getnew(v,seq,f'window_{s:03d}')[0] for s in range(0,121,12)}
  assert set(fused)==set(old)
  frame_totals.append({'variant':v,'sequence':seq,'frames':160,**{k:counts[k] for k in ['matched','components','unmatched_components','positive_pixels_outside_boxes']}})
  for c in (r for r in cohorts if r['sequence']==seq):
   k=key(c);starts=sorted(s for s,f,a in ctx if (f,a)==k);assert len(starts)==int(c['context_count'])>=2
   original=[ctx[s,*k] for s in starts];new=[windows[s][k] for s in starts]
   for o,n in zip(original,new):
    assert [float(o['bbox_'+x]) for x in ['x','y','w','h']]==n['bbox'] and int(o['source_frame'])==n['source_frame'] and int(o['track_id'])==n['track_id']
   of=old[k];nf=fused[k];om=np.asarray([int(r['matched']) for r in original],bool);nm=np.asarray([r['matched'] for r in new],bool);ob=np.asarray([float(r['box_max']) for r in original]);nb=np.asarray([r['box_max'] for r in new]);o=bool(int(of['matched']));n=bool(nf['matched'])
   assert [float(of['bbox_'+x]) for x in ['x','y','w','h']]==nf['bbox'] and int(of['track_id'])==nf['track_id'] and int(of['source_frame'])==nf['source_frame']
   order=sorted(range(len(starts)),key=lambda i:(ob[i],starts[i]));a,z=order[0],order[-1];strong=ob[a]<.1 and ob[z]>=.9;s=support[fullkey(c)]
   flag={'strong97':strong,'suppression24':strong and bool(int(s['a_suppressed_common_strong'])),'support42':strong and bool(int(s['b_requires_exclusive_strong'])),'stable_high502':ob.min()>=.5,'stable_low778':ob.max()<.5}
   fields={'original_fused_match':o,'candidate_fused_match':n,'fused_gain':not o and n,'fused_loss':o and not n,'window_count':len(starts),'original_window_matches':int(om.sum()),'candidate_window_matches':int(nm.sum()),'window_gain':int(np.count_nonzero(~om&nm)),'window_loss':int(np.count_nonzero(om&~nm)),'any_to_none':om.any() and not nm.any(),'none_to_any':not om.any() and nm.any(),'original_all_match':om.all(),'all_to_not_all':om.all() and not nm.all(),'protected_original':om.all() and o,'protected_lost':om.all() and o and not n,'a_original_match':om[a],'b_original_match':om[z],'a_candidate_match':nm[a],'b_candidate_match':nm[z],'a_candidate_ge05':nb[a]>=.5,'b_candidate_ge05':nb[z]>=.5,'a_candidate_ge09':nb[a]>=.9,'b_candidate_ge09':nb[z]>=.9,'candidate_all_box_low':nb.max()<.5,'candidate_all_box_high':nb.min()>=.5,'candidate_box_flip':nb.min()<.5<=nb.max(),'response_flip_to_both_low':ob.min()<.5<=ob.max() and nb.max()<.5}
   row={'variant':v,'sequence':seq,'local_frame':k[0],'annotation_id':k[1],'source_frame':int(of['source_frame']),'track_id':int(of['track_id']),**{k:int(val) for k,val in flag.items()},**{k:int(val) for k,val in fields.items()},'a_start':starts[a],'b_start':starts[z],'fused_joint_peak_shift':math.hypot(nf['peak_x']-float(of['peak_x']),nf['peak_y']-float(of['peak_y'])) if o and n else None,'candidate_none_window_match':int(not nm.any()),'both_fused_missed':int(not o and not n),'original_center_error':float(of['normalized_center_error']) if o else None,'candidate_center_error':nf['center_error'] if n else None,'joint_center_error_delta':nf['center_error']-float(of['normalized_center_error']) if o and n else None}
   rows.append(row)
def stats(rr):
 fields=['original_fused_match','candidate_fused_match','fused_gain','fused_loss','window_gain','window_loss','window_count','original_window_matches','candidate_window_matches','any_to_none','none_to_any','original_all_match','all_to_not_all','protected_original','protected_lost','a_original_match','b_original_match','a_candidate_match','b_candidate_match','a_candidate_ge05','b_candidate_ge05','a_candidate_ge09','b_candidate_ge09','candidate_all_box_low','candidate_all_box_high','candidate_box_flip','response_flip_to_both_low']
 out={k:sum(r[k] for r in rr) for k in fields};out.update(n=len(rr),tracks=len({(r['sequence'],r['track_id']) for r in rr}),fused_net=out['fused_gain']-out['fused_loss'],protected_loss_rate=out['protected_lost']/out['protected_original'] if out['protected_original'] else None,protection_satisfied=bool(out['protected_original']) and 100*out['protected_lost']<=out['protected_original'])
 for field in ['fused_joint_peak_shift','original_center_error','candidate_center_error','joint_center_error_delta']:
  a=np.asarray([r[field] for r in rr if r[field] is not None]);out[field]={'n':len(a),'mean':float(a.mean()) if len(a) else None,'median':float(np.median(a)) if len(a) else None,'p95':float(np.quantile(a,.95)) if len(a) else None}
 out['joint_fused_peak_displacement']=out.pop('fused_joint_peak_shift')
 out['candidate_no_matched_window']=sum(r['candidate_none_window_match'] for r in rr);out['both_fused_missed']=sum(r['both_fused_missed'] for r in rr)
 return out
cohort_stats={v:{'all1417':stats([r for r in rows if r['variant']==v]),**{g:stats([r for r in rows if r['variant']==v and r[g]]) for g in ['strong97','suppression24','support42','stable_high502','stable_low778']}} for v in V}
trackstats=[]
for v in V:
 rr=[r for r in rows if r['variant']==v];groups=collections.defaultdict(list)
 for r in rr:groups[r['sequence'],r['track_id']].append(r)
 for (seq,track),rs in sorted(groups.items()):
  x={'variant':v,'sequence':seq,'track_id':track,'n':len(rs),'fused_gain':sum(z['fused_gain'] for z in rs),'fused_loss':sum(z['fused_loss'] for z in rs)}
  for name,f in [('original_longest_miss','original_fused_match'),('candidate_longest_miss','candidate_fused_match')]:
   misses=sorted({r['source_frame'] for r in rs if not r[f]});runs=[]
   for index,frame in enumerate(misses):
    if index==0 or frame!=misses[index-1]+1:runs.append(1)
    else:runs[-1]+=1
   x[name]=max(runs,default=0)
  trackstats.append(x)
aggregate={v:{k:sum(r[k] for r in frame_totals if r['variant']==v) for k in ['frames','matched','components','unmatched_components','positive_pixels_outside_boxes']} for v in ['official']+V}
payload={'mode':'CPU-only aggregation of existing scored NPZ and original CSV. No inference, no component rescoring, no confirmation access.','cohort_stats':cohort_stats,'tracks':trackstats,'object_rows':rows,'all640_frames':aggregate,'source_hashes':hashes,'audit_script_sha256':sha('/audit.py')}
print(json.dumps(payload,ensure_ascii=False,allow_nan=False))
