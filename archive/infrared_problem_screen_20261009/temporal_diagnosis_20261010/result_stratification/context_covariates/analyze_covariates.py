from pathlib import Path
import csv,json,statistics,math,collections,hashlib,datetime
Q=Path(__file__).resolve().parents[2]
O=Path(__file__).resolve().parent
(O/'SPEC.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'type':'exploratory followup after fixed-choice gains/losses observed','comparison':'earliest versus latest naturalwindow for sameGT/frame; fixed strict threshold.5, maxside<=16; matched flips; no fitting','covariates':'annotation support count, median absolute standardized local contrast, temporal standard deviation of ringmean, median residual speed; qualityproxies notvisibility/causal mechanisms','independence':'frames correlated; no pvalues'},indent=2)+'\n')
def read(p):return list(csv.DictReader(p.open()))
def save(name,rows):
 with (O/name).open('w') as f:
  if rows:
   w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
def median(vals):return statistics.median(vals) if vals else None
pairs=[];contexts=[]
for seq in ['DJI_0028_2','DJI_0028_4','DJI_0032_5','DJI_0047_3']:
 motion=Q/'motion' if seq=='DJI_0028_2' else Q/'motion'/seq
 feat=collections.defaultdict(dict)
 for r in read(motion/'TRACK_FRAME_DIAGNOSTICS.csv'):feat[int(r['track_id'])][int(r['offset'])]=r
 speeds={(int(r['track_id']),int(r['offset_from'])):float(r['adjusted_speed_px_per_frame']) for r in read(motion/'LAG5_TRACK_MOTION.csv') if r['camera_adjustment_valid']=='True' and r['adjusted_speed_px_per_frame']}
 summaries=read(Q/'window_offset_experiment/analysis'/seq/'same_gt_frame_context_summary.csv')
 eligible={(int(r['track_id']),int(r['local_frame'])) for r in summaries if r['threshold']=='0.5' and r['conservative_eligible']=='1' and float(r['current_frame_max_box_side'])<=16}
 per=collections.defaultdict(list)
 for r in read(Q/'window_offset_experiment/analysis'/seq/'per_context_all_tracks.csv'):
  if r['threshold']=='0.5' and (int(r['track_id']),int(r['local_frame'])) in eligible:per[int(r['track_id']),int(r['local_frame'])].append(r)
 cache={}
 def cov(tid,start):
  if (tid,start) in cache:return cache[tid,start]
  ff=[feat[tid][t] for t in range(start,start+40) if t in feat[tid]]
  zz=[abs(float(r['mean_difference']))/max(float(r['ring_std']),1) for r in ff if r['contrast_valid']=='True']
  rr=[float(r['ring_mean']) for r in ff if r['contrast_valid']=='True']
  sp=[speeds[tid,t] for t in range(start,start+35) if (tid,t) in speeds]
  c={'annotation_support':len(ff),'valid_contrast_support':len(zz),'median_abs_z':median(zz),'ringmean_temporal_std':statistics.pstdev(rr) if len(rr)>1 else None,'valid_motion_support':len(sp),'median_residual_speed':median(sp)}
  cache[tid,start]=c;contexts.append(dict(sequence=seq,track_id=tid,window_start=start,**c));return c
 for (tid,t),rr in per.items():
  a=min(rr,key=lambda r:int(r['window_start']));b=max(rr,key=lambda r:int(r['window_start']))
  if a['matched']==b['matched']:continue
  win,loss=(a,b) if a['matched']=='1' else (b,a)
  wc=cov(tid,int(win['window_start']));lc=cov(tid,int(loss['window_start']))
  row={'sequence':seq,'track_id':tid,'local_frame':t,'winning_start':int(win['window_start']),'losing_start':int(loss['window_start']),'winning_slot':int(win['slot']),'losing_slot':int(loss['slot']),'winning_boxmax':float(win['box_max']),'losing_boxmax':float(loss['box_max']),'strong_response_flip':int(float(win['box_max'])>=.9 and float(loss['box_max'])<.1)}
  for k in wc:row[k+'_win']=wc[k];row[k+'_loss']=lc[k];row[k+'_delta_win_minus_loss']=wc[k]-lc[k] if wc[k] is not None and lc[k] is not None else None
  pairs.append(row)
save('matched_flip_context_pairs.csv',pairs);save('context_covariates.csv',contexts)
summary=[]
for seq in ['ALL','DJI_0028_2','DJI_0028_4','DJI_0032_5','DJI_0047_3']:
 for kind in ['all_point_flips','strong_response_flips']:
  pp=[r for r in pairs if (seq=='ALL' or r['sequence']==seq) and (kind=='all_point_flips' or r['strong_response_flip'])]
  for key in ['annotation_support','median_abs_z','ringmean_temporal_std','median_residual_speed']:
   vs=[r[key+'_delta_win_minus_loss'] for r in pp if r[key+'_delta_win_minus_loss'] is not None]
   summary.append({'sequence':seq,'subset':kind,'covariate':key,'pairs':len(pp),'tracks':len({(r['sequence'],r['track_id']) for r in pp}),'valid_pairs':len(vs),'win_higher':sum(v>1e-12 for v in vs),'win_lower':sum(v< -1e-12 for v in vs),'equal':sum(abs(v)<=1e-12 for v in vs),'median_win_minus_loss':median(vs)})
save('summary.csv',summary)
(O/'SUMMARY.json').write_text(json.dumps({'rows':summary,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'limits':['annotation-support is not visibility','contrast uses bbox andring, nottargetSNR','differences correlate withcontext, notcausalproof','only fixedearliest/latestpair, notoraclebestpair','adjacentframes repeat tracks; nnotindependent']},indent=2)+'\n')
print('saved',len(pairs),'matched flip pairs')
for r in summary:
 if r['sequence']=='ALL':print(r)
