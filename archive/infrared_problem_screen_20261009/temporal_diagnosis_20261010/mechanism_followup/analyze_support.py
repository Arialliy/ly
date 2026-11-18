#!/usr/bin/env python3
import json,csv,collections,statistics
from pathlib import Path
Q=Path(__file__).resolve().parent

def read(p):
 with p.open() as f:return list(csv.DictReader(f))
def f(r,k):return float(r[k])
def i(r,k):return int(float(r[k]))
def resp(r,tag,variant):return f(r,tag+'_'+variant+'_box_max_probability')
rows=[];meta=[]
for p in sorted(Q.glob('support_probe/shard_*/results.csv')):
 m=json.loads((p.parent/'metadata.json').read_text());assert m['status']=='COMPLETE';meta.append(m);rows+=read(p)
assert len(rows)==1417 and len({i(r,'pair_index') for r in rows})==1417
rows.sort(key=lambda r:i(r,'pair_index'))
context={}
for seq in sorted({r['sequence'] for r in rows}):
 for c in read(Q.parent/'window_offset_experiment/analysis'/seq/'per_context_all_tracks.csv'):
  if float(c['threshold'])==.5:context[(seq,int(c['local_frame']),int(c['track_id']),int(c['window_start']))]=c
motion={(r['sequence'],int(r['local_frame']),int(r['track_id'])):r for r in read(Q.parent/'result_stratification/response/JOINED_TRACK_FRAME.csv') if float(r['threshold'])==.5}
for r in rows:
 r['group']='flip' if i(r,'boxmax_crossing') else ('stable_low' if i(r,'all_contexts_boxmax_low') else 'stable_high')
 for tag in ['a','b']:
  c=context[(r['sequence'],i(r,'local_frame'),i(r,'track_id'),i(r,tag+'_start'))];r[tag+'_point_matched']=int(c['matched'])
 m=motion[(r['sequence'],i(r,'local_frame'),i(r,'track_id'))];r['adjusted_speed']=m['adjusted_speed_px_per_frame'];r['motion_valid']=m['motion_valid']
 r['a_suppressed_common_half']=int(resp(r,'a','shared')>=.5>resp(r,'a','full'))
 r['a_suppressed_common_strong']=int(resp(r,'a','shared')>=.9 and resp(r,'a','full')<.1)
 r['b_requires_exclusive_half']=int(resp(r,'b','shared')<.5<=resp(r,'b','full'))
 r['b_requires_exclusive_strong']=int(resp(r,'b','shared')<.1 and resp(r,'b','full')>=.9)
 r['both_shared_high']=int(min(resp(r,t,'shared') for t in ['a','b'])>=.5)
 r['both_global_shared_high']=int(min(resp(r,t,'global_shared') for t in ['a','b'])>=.5)
 r['both_safe_high']=int(i(r,'safe_shared_eligible') and min(resp(r,t,'safe_shared') for t in ['a','b'])>=.5)
 r['both_safe_strong']=int(i(r,'safe_shared_eligible') and min(resp(r,t,'safe_shared') for t in ['a','b'])>=.9)
 r['a_suppressed_at_b_peak']=int(f(r,'a_shared_b_peak_probability')>=.5>f(r,'a_full_b_peak_probability'))
 r['a_strong_suppressed_at_b_peak']=int(f(r,'a_shared_b_peak_probability')>=.9 and f(r,'a_full_b_peak_probability')<.1)
 vals=[resp(r,'a','lambda_'+v) for v in ['000','025','050','075','100']]
 r['a_lambda_monotone_decreasing']=int(all(x>=y-1e-6 for x,y in zip(vals,vals[1:])))
 r['suppression_id']=r['a_suppressed_common_strong']

def summary(rr):
 out={'n':len(rr),'ids':len({(r['sequence'],r['track_id']) for r in rr})}
 for k in ['a_suppressed_common_half','a_suppressed_common_strong','b_requires_exclusive_half','b_requires_exclusive_strong','both_shared_high','both_global_shared_high','safe_shared_eligible','current_frame_in_safe_shared','both_safe_high','both_safe_strong','a_suppressed_at_b_peak','a_strong_suppressed_at_b_peak','a_lambda_monotone_decreasing']:
  out[k]=sum(i(r,k) for r in rr)
 out['strong_suppression_ids']=len({(r['sequence'],r['track_id']) for r in rr if r['a_suppressed_common_strong']})
 out['strong_suppression_clips']=dict(collections.Counter(r['sequence'] for r in rr if r['a_suppressed_common_strong']))
 out['suppression_and_b_matched']=sum(r['a_suppressed_common_strong'] and r['b_point_matched'] for r in rr)
 out['suppression_and_safe_both_high']=sum(r['a_suppressed_common_strong'] and r['both_safe_high'] for r in rr)
 out['suppression_and_safe_both_high_current_safe']=sum(r['a_suppressed_common_strong'] and r['both_safe_high'] and i(r,'current_frame_in_safe_shared') for r in rr)
 out['suppression_and_both_global_high']=sum(r['a_suppressed_common_strong'] and r['both_global_shared_high'] for r in rr)
 out['four_patterns_half']=dict(collections.Counter(('suppression' if r['a_suppressed_common_half'] else 'no_suppression')+'+'+('extra_support' if r['b_requires_exclusive_half'] else 'no_extra_support') for r in rr))
 out['a_lambda_means']=[statistics.mean([resp(r,'a','lambda_'+v) for r in rr]) if rr else None for v in ['000','025','050','075','100']]
 return out
s={'all':summary(rows)}
for name,pred in [('strong_flips',lambda r:i(r,'strong_response_flip')),('flips',lambda r:r['group']=='flip'),('stable_low',lambda r:r['group']=='stable_low'),('stable_high',lambda r:r['group']=='stable_high'),('strong_safe_eligible',lambda r:i(r,'strong_response_flip') and i(r,'safe_shared_eligible')),('strong_high_window_matched',lambda r:i(r,'strong_response_flip') and r['b_point_matched'])]:s[name]=summary([r for r in rows if pred(r)])
s['strong_by_clip']={seq:summary([r for r in rows if r['sequence']==seq and i(r,'strong_response_flip')]) for seq in sorted({r['sequence'] for r in rows})}
s['maximum_errors']={k:max(m['maximum_errors'].get(k,0) for m in meta) for k in meta[0]['maximum_errors']}
s['threshold_disagreements']={k:sum(m['replay_threshold_disagreements'].get(k,0) for m in meta) for k in meta[0]['replay_threshold_disagreements']}
(Q/'SUPPORT_SUMMARY.json').write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n')
with (Q/'SUPPORT_PAIRED.csv').open('w',newline='') as f_:
 w=csv.DictWriter(f_,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps({k:s[k] for k in ['strong_flips','stable_low','stable_high','strong_high_window_matched','strong_by_clip']},ensure_ascii=False,indent=2))
