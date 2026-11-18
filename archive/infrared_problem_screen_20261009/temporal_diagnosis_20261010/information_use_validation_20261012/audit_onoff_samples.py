#!/usr/bin/env python3
"""Fixed-frame independent on/off numeric audit, from saved tensors only."""
import csv,hashlib,importlib.util,json,math,time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
N=Path(__file__).resolve().parent;Q=N.parent;OLD=Q/'mutual_temporal_pilot_20261012'
FRAMES=(0,39,80,120,159);SEQS=('DJI_0026_2','DJI_0067_2');PLANS={'O':(0,36,72,108,120),'S':(0,24,60,96,120)}
CHECKS=[];SOURCES={}

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
 return h.hexdigest()
def track(path):
 v=digest(path);SOURCES[str(path)]=v;return v
def read_json(path):track(path);return json.loads(Path(path).read_text())
def read_csv(path):
 track(path)
 with Path(path).open() as f:return list(csv.DictReader(f))
def ck(name,yes,**details):
 CHECKS.append(dict(name=name,passed=bool(yes),**details))
 if not yes:raise AssertionError(name+': '+str(details))
def numeric(name,actual,expected):
 ck(name,math.isclose(float(actual),float(expected),rel_tol=1e-9,abs_tol=1e-11),actual=float(actual),expected=float(expected))

def main():
 torch.set_num_threads(2)
 spec=importlib.util.spec_from_file_location('previous_independent_audit_not_official_scorer',OLD/'audit_final_results.py');oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
 ck('independent_matcher_unchanged',track(OLD/'audit_final_results.py')=='31a8805cd94de7fc94a7924acce3eef5dc52963167b4a0487092a3944be44b2a')
 regionrows=read_csv(N/'A_REGIONS.csv');outputrows={r['region_id']:r for r in read_csv(N/'A_LOGIT_REGIONS.csv')}
 pixelrows={(r['sequence'],r['arrangement'],r['frame']):r for r in read_csv(N/'A_PIXEL_CHANGES.csv')}
 freeze=read_json(N/'A_FREEZE.json');score_complete=read_json(N/'A_SCORE_COMPLETE.json');chain=read_json(N/'A_BRANCH_CHAIN_AUDIT.json')
 ck('A_score_and_chain_complete',score_complete['status']=='COMPLETE' and chain['status']=='PASS')
 frame_count=0;region_count=0
 for seq in SEQS:
  root=N/'onoff'/seq;run=read_json(root/'COMPLETE.json');oldrun=read_json(OLD/'predictions/E2'/seq/'COMPLETE.json');offrun=read_json(N/'predictions/E2_off'/seq/'COMPLETE.json')
  training=read_json(OLD/'training/E2/run/RUN.json')
  cp=OLD/'training/E2/run/snapshot_512.pth'
  ck(seq+'same_actual_checkpoint',track(cp)==run['checkpoint_sha256']==oldrun['checkpoint_sha256']==offrun['checkpoint_sha256']==training['final_checkpoint_sha256'])
  ck(seq+'unchanged_model_state',run['model_state_sha256']==run['unchanged_final_model_state_sha256']==training['final_model_state_sha256'])
  ck(seq+'off_semantics',"same trained E2 state" in run['off_operation'] and "set_mode('off')" in run['off_operation'] and run['new_training'] is False)
  ck(seq+'frozen_preprocessing',run['seed']==46 and run['normalization']==[85.0799,47.4845] and run['tile_core']==128 and run['tile_halo']==20)
  ck(seq+'raw_fingerprint',run['raw_sha256']==oldrun['raw_sha256']==track(Q/'stage1_goal/data'/seq/'raw_frames.npy'))
  ck(seq+'onoff_script_bound',run['script_sha256']==track(N/'code/onoff_infer.py'))
  arrays={};areas={};receipts={}
  for state,folder in [('on',OLD/'evaluation/E2'/seq),('off',N/'A_evaluation/E2_off'/seq)]:
   receipts[state]=read_json(folder/'COMPLETE.json')
   with np.load(folder/'scores.npz',allow_pickle=False) as z:arrays[state]={k:z[k] for k in z.files}
   with np.load(folder/'component_areas.npz',allow_pickle=False) as z:areas[state]={k:z[k] for k in z.files}
   track(folder/'scores.npz');track(folder/'component_areas.npz')
   ck(seq+state+'threshold',float(arrays[state]['threshold'])==.5 and receipts[state]['threshold']==.5)
  ck(seq+'GT unchanged',receipts['on']['objects']==receipts['off']['objects'])
  fused={state:{plan:{t:np.zeros((512,640),np.float32) for t in FRAMES} for plan in PLANS} for state in ['on','off']}
  logits={state:{plan:{t:np.full((512,640),-np.inf,np.float32) for t in FRAMES} for plan in PLANS} for state in ['on','off']}
  for start in sorted(set(PLANS['O']+PLANS['S'])):
   file=root/'windows'/f'window_{start:03d}.npz';ck(seq+f'window{start}rawhash',track(file)==run['window_sha256'][file.name])
   with np.load(file,allow_pickle=False) as z:
    ck(seq+f'window{start}slots',int(z['start'])==start and int(z['end'])==start+40)
    for state in ['on','off']:
     zz=z[state+'_logits'];ck(seq+f'window{start}{state}rawformat',zz.shape==(40,512,640) and zz.dtype==np.float32 and np.isfinite(zz).all())
     probability=torch.from_numpy(zz).sigmoid().numpy()
     proot=(OLD/'predictions/E2'/seq if state=='on' else N/'predictions/E2_off'/seq)
     pp=proot/'windows'/file.name
     receipt=oldrun if state=='on' else offrun
     ck(seq+f'window{start}{state}probcachehash',track(pp)==receipt['window_sha256'][file.name])
     with np.load(pp,allow_pickle=False) as saved:ck(seq+f'window{start}{state}raw_sigmoid_exact_cached',np.array_equal(probability,saved['probabilities']))
     for plan,starts in PLANS.items():
      if start not in starts:continue
      for t in FRAMES:
       if start<=t<start+40:
        np.maximum(fused[state][plan][t],probability[t-start],out=fused[state][plan][t])
        np.maximum(logits[state][plan][t],zz[t-start],out=logits[state][plan][t])
  norms=defaultdict(list)
  for r in read_csv(root/'RESIDUAL_REGION_WINDOWS.csv'):norms[r['region_id']].append(r)
  for pi,plan in enumerate(PLANS):
   for t in FRAMES:
    prefix=f'{seq}/{plan}/{t}';left,right=map(int,arrays['on']['offsets'][t:t+2]);meta=receipts['on']['objects'][left:right]
    gt=[dict(bbox=r['bbox'],crop=oracle.crop_indices(r['bbox'],512,640)) for r in meta]
    for state in ['on','off']:
     rows,counts,aa,uu=oracle.independent_frame(fused[state][plan][t],gt)
     ck(prefix+state+'independent_framecounts',np.array_equal(counts,arrays[state]['frame_counts'][pi,t]))
     for key in ['matched','box_max','peak_x','peak_y','component_area']:
      actual=np.asarray([r[key] for r in rows],dtype=arrays[state][key].dtype)
      ck(prefix+state+key,np.array_equal(actual,arrays[state][key][pi,left:right],equal_nan=True))
     c0,c1=map(int,areas[state][plan+'_offsets'][t:t+2])
     ck(prefix+state+'component_areas',np.array_equal(aa,areas[state][plan+'_areas'][c0:c1]))
     ck(prefix+state+'component_assignment',np.array_equal(uu,areas[state][plan+'_unmatched'][c0:c1]))
     frame_count+=1
    a,b=logits['on'][plan][t],logits['off'][plan][t];delta=a-b;flat=delta.ravel();absolute=np.abs(flat)
    row=pixelrows[seq,plan,str(t)]
    vals=dict(pixels=flat.size,delta_mean=flat.mean(dtype=np.float64),delta_min=flat.min(),delta_max=flat.max(),absolute_mean=absolute.mean(dtype=np.float64),exact_nonzero_pixels=np.count_nonzero(flat),positive_delta_pixels=np.count_nonzero(flat>0),negative_delta_pixels=np.count_nonzero(flat<0),logit_zero_decision_changes=np.count_nonzero((a>=0)!=(b>=0)))
    for label,value in zip(['delta_q01','delta_q05','delta_q50','delta_q95','delta_q99'],np.percentile(flat,[1,5,50,95,99])):vals[label]=value
    for label,value in zip(['absolute_q50','absolute_q90','absolute_q99'],np.percentile(absolute,[50,90,99])):vals[label]=value
    for key,value in vals.items():numeric(prefix+'pixels '+key,value,row[key])
    selected=[r for r in regionrows if r['sequence']==seq and r['arrangement']==plan and int(r['frame'])==t]
    for region in selected:
     out=outputrows[region['region_id']]
     if region['valid']!='1':ck(prefix+region['region_id']+'invalid_retained',out['valid']=='0' and out['on_max_logit']=='');continue
     x0,y0,x1,y1=(int(region[k]) for k in ['x0','y0','x1','y1']);av,bv=a[y0:y1,x0:x1],b[y0:y1,x0:x1];dv=av-bv
     ay,ax=divmod(int(np.argmax(av)),x1-x0);by,bx=divmod(int(np.argmax(bv)),x1-x0)
     px,py=int(region['release_peak_x_index']),int(region['release_peak_y_index'])
     rv=dict(on_max_logit=av.max(),off_max_logit=bv.max(),delta_max_logit=av.max()-bv.max(),on_mean_logit=av.mean(dtype=np.float64),off_mean_logit=bv.mean(dtype=np.float64),delta_mean_logit=dv.mean(dtype=np.float64),on_peak_x_index=ax+x0,on_peak_y_index=ay+y0,off_peak_x_index=bx+x0,off_peak_y_index=by+y0,delta_at_release_peak=delta[py,px],delta_at_off_peak=dv[by,bx],delta_at_on_peak=dv[ay,ax],mean_absolute_delta=np.abs(dv).mean(dtype=np.float64),positive_delta_fraction=np.mean(dv>0),negative_delta_fraction=np.mean(dv<0))
     nr=norms[region['region_id']];expected={s for s in PLANS[plan] if s<=t<s+40}
     ck(region['region_id']+'normcoverage',{int(r['window_start']) for r in nr}==expected and len(nr)==len(expected))
     fn=[];cn=[];ratios=[]
     for r in nr:
      ck(region['region_id']+'normpixels'+r['window_start'],int(r['pixels'])==(x1-x0)*(y1-y0))
      ff,cc=float(r['feature_squared_norm']),float(r['correction_squared_norm']);ck(region['region_id']+'normnonnegative'+r['window_start'],ff>=0 and cc>=0)
      f,c=math.sqrt(ff),math.sqrt(cc);ratio=c/max(f,1e-12);numeric(region['region_id']+'energyratio'+r['window_start'],ratio,r['relative_correction_norm'])
      fn.append(f);cn.append(c);ratios.append(ratio)
     rv.update(residual_covering_windows=len(nr),feature_norm_window_mean=np.mean(fn),correction_norm_window_mean=np.mean(cn),relative_correction_norm_window_mean=np.mean(ratios),relative_correction_norm_window_max=max(ratios))
     for key,value in rv.items():numeric(region['region_id']+' '+key,value,out[key])
     region_count+=1
  print(json.dumps(dict(sequence=seq,checked_frames=frame_count,checked_regions=region_count,checks=len(CHECKS))),flush=True)
 ck('fixed_frame_total',frame_count==40)
 return frame_count,region_count

if __name__=='__main__':
 started=time.perf_counter();result=dict(status='RUNNING',frames=list(FRAMES),seed=46,no_new_inference_or_training=True)
 try:
  f,r=main();result.update(status='PASS',independently_scored_state_plan_frames=f,independently_recomputed_region_rows=r)
 except BaseException as e:
  result.update(status='FAIL',error=repr(e));raise
 finally:
  result.update(checks_passed=sum(r['passed'] for r in CHECKS),checks=CHECKS,source_sha256=SOURCES,script_sha256=digest(__file__),elapsed_seconds=time.perf_counter()-started,
   limitations=['40fixedframe-state-plan samples independentlymatched; not all640on/offframeplanstates.','Regionalrawlogit statistics recomputed from savedwindowlogits.','Feature/correctiontensors not stored; norm audit independentlychecks recordedenergies-to-ratio/averaging only, not underlyingfeatureenergies.','A usestrainedE2checkpoint; Busesoriginalreleasedfeatures, so no directsame-representationcausalchain.'])
  (N/'A_INDEPENDENT_NUMERIC_AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
  (N/'A_INDEPENDENT_NUMERIC_AUDIT.md').write_text('# A数值独立抽检\n\n'+result['status']+'；检查数'+str(result['checks_passed'])+'，独立匹配40组固定帧/安排/on-off状态。逐窗真实logits经sigmoid与概率缓存逐位一致，再按原概率max融合，使用独立洪泛和迭代增广匹配核对。\n\n区域logit变化从真实缓存重新计算。特征范数只保存能量，故本审计复核能量到比例和均值的计算，不能从未保存的32通道特征独立重建原能量。没有新增推理、训练或阈值选择。\n\n[完整记录](A_INDEPENDENT_NUMERIC_AUDIT.json)\n')
 print(json.dumps({k:v for k,v in result.items() if k in ['status','checks_passed','independently_scored_state_plan_frames','independently_recomputed_region_rows','elapsed_seconds']}))
