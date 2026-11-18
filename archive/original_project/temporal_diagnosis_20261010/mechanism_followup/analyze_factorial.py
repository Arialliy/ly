#!/usr/bin/env python3
import csv,json,math,statistics,collections,hashlib
from pathlib import Path
Q=Path(__file__).resolve().parent
FACTORS=['slot','feature','membership']
def read(p):
 with p.open() as f:return list(csv.DictReader(f))
def f(r,k):return float(r[k])
def i(r,k):return int(float(r[k]))
def md(v):return statistics.median(v) if v else None
def mean(v):return statistics.mean(v) if v else None
def shp(r,prefix,metric='box_max_logit'):
 v={m:f(r,prefix+''.join(str((m>>j)&1) for j in range(3))+'_'+metric) for m in range(8)}
 a=[]
 for j in range(3):
  z=0
  for m in range(8):
   if not (m&(1<<j)):
    n=m.bit_count();w=math.factorial(n)*math.factorial(2-n)/6
    z+=w*(v[m|(1<<j)]-v[m])
  a.append(z)
 assert abs(sum(a)-(v[7]-v[0]))<1e-8
 return a
rows=[];metas=[]
for p in sorted(Q.glob('shard_*/results.csv')):
 meta=json.loads((p.parent/'metadata.json').read_text());assert meta['status']=='COMPLETE';metas.append(meta);rows+=read(p)
assert len(rows)==1417 and len({i(r,'pair_index') for r in rows})==1417
rows.sort(key=lambda r:i(r,'pair_index'))
for r in rows:
 r['group']='flip' if i(r,'boxmax_crossing') else ('stable_low' if i(r,'all_contexts_boxmax_low') else 'stable_high')
 for anchor,prefix in [('symmetric','combo_'),('anchor_a','anchor_a_combo_'),('anchor_b','anchor_b_combo_')]:
  for metric in ['box_max_logit','a_peak_logit','b_peak_logit']:
   vals=shp(r,prefix,metric)
   for name,val in zip(FACTORS,vals):r['shapley_'+anchor+'_'+metric+'_'+name]=val
   r['dominant_'+anchor+'_'+metric]=FACTORS[max(range(3),key=lambda k:vals[k])]
 r['dominant_anchor_agreement']=int(len({r['dominant_'+a+'_box_max_logit'] for a in ['symmetric','anchor_a','anchor_b']})==1)
 r['original_logit_gap']=f(r,'b_replay_box_max_logit')-f(r,'a_replay_box_max_logit')
 r['halo_probability_range']=abs(f(r,'b_halo_box_max')-f(r,'a_halo_box_max'))
 r['halo_pair_crosses_half']=int((f(r,'a_halo_box_max')>=.5)!=(f(r,'b_halo_box_max')>=.5))
 r['halo_pair_strong_flip']=int(min(f(r,'a_halo_box_max'),f(r,'b_halo_box_max'))<.1 and max(f(r,'a_halo_box_max'),f(r,'b_halo_box_max'))>=.9)
 r['both_complete_halo']=int(all(i(r,t+'_halo_missing_'+side)==0 for t in ['a','b'] for side in ['before','after']))

def summary(rr):
 d={'n':len(rr),'ids':len({(r['sequence'],r['track_id']) for r in rr}),'clips':dict(collections.Counter(r['sequence'] for r in rr)),'group_counts':dict(collections.Counter(r['group'] for r in rr))}
 if not rr:return d
 d['original_probability_gap_median']=md([f(r,'b_replay_box_max')-f(r,'a_replay_box_max') for r in rr]);d['original_logit_gap_median']=md([r['original_logit_gap'] for r in rr])
 d['dominant_anchors']={a:dict(collections.Counter(r['dominant_'+a+'_box_max_logit'] for r in rr)) for a in ['symmetric','anchor_a','anchor_b']}
 d['dominant_fixed_peaks']={p:dict(collections.Counter(r['dominant_symmetric_'+p] for r in rr)) for p in ['a_peak_logit','b_peak_logit']}
 d['all_anchor_dominant_agree']=sum(r['dominant_anchor_agreement'] for r in rr)
 d['shapley_symmetric']={k:{'mean':mean([r['shapley_symmetric_box_max_logit_'+k] for r in rr]),'median':md([r['shapley_symmetric_box_max_logit_'+k] for r in rr]),'positive_count':sum(r['shapley_symmetric_box_max_logit_'+k]>0 for r in rr)} for k in FACTORS}
 d['single_term_low_to_high']={k:sum(f(r,'a_replay_box_max')<.5<=f(r,'combo_'+''.join('1' if j==idx else '0' for j in range(3))+'_box_max_probability') for r in rr) for idx,k in enumerate(FACTORS)}
 d['remove_term_high_to_low']={k:sum(f(r,'b_replay_box_max')>=.5>f(r,'combo_'+''.join('0' if j==idx else '1' for j in range(3))+'_box_max_probability') for r in rr) for idx,k in enumerate(FACTORS)}
 d['halo']={'both_complete':sum(r['both_complete_halo'] for r in rr),'crosses_half':sum(r['halo_pair_crosses_half'] for r in rr),'strong_flip':sum(r['halo_pair_strong_flip'] for r in rr),'both_high':sum(min(f(r,'a_halo_box_max'),f(r,'b_halo_box_max'))>=.5 for r in rr),'both_low':sum(max(f(r,'a_halo_box_max'),f(r,'b_halo_box_max'))<.5 for r in rr),'pair_probability_range_median':md([r['halo_probability_range'] for r in rr])}
 for tag in ['a','b']:
  d['halo'][tag+'_response_gain']=sum(f(r,tag+'_replay_box_max')<.5<=f(r,tag+'_halo_box_max') for r in rr)
  d['halo'][tag+'_response_loss']=sum(f(r,tag+'_replay_box_max')>=.5>f(r,tag+'_halo_box_max') for r in rr)
 return d
s={'all':summary(rows)}
for name,pred in [('flips',lambda r:r['group']=='flip'),('strong_flips',lambda r:i(r,'strong_response_flip')),('stable_low',lambda r:r['group']=='stable_low'),('stable_high',lambda r:r['group']=='stable_high'),('strong_both_complete_halo',lambda r:i(r,'strong_response_flip') and r['both_complete_halo']),('flips_both_complete_halo',lambda r:r['group']=='flip' and r['both_complete_halo'])]:s[name]=summary([r for r in rows if pred(r)])
s['strong_by_clip']={seq:summary([r for r in rows if r['sequence']==seq and i(r,'strong_response_flip')]) for seq in sorted({r['sequence'] for r in rows})}
s['maximum_errors']={k:max(m['maximum_errors'].get(k,0) for m in metas) for k in metas[0]['maximum_errors']}
s['threshold_disagreements']={k:sum(m['replay_threshold_disagreements'].get(k,0) for m in metas) for k in metas[0]['replay_threshold_disagreements']}
s['interpretation']='Internal counterfactual decomposition of previously selected extreme windows; Shapley not unique natural causal factorization. Halo changes frontend true neighbors; response gains/losses are not point detections or Pd/Fa.'
(Q/'SUMMARY.json').write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n')
with (Q/'PAIRED_MECHANISMS.csv').open('w',newline='') as f_:
 w=csv.DictWriter(f_,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
tracks=[]
for seq,tid in sorted({(r['sequence'],r['track_id']) for r in rows if i(r,'strong_response_flip')}):
 rr=[r for r in rows if (r['sequence'],r['track_id'])==(seq,tid) and i(r,'strong_response_flip')];d=summary(rr);tracks.append({'sequence':seq,'track_id':tid,**d})
(Q/'STRONG_BY_TRACK.json').write_text(json.dumps(tracks,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:s[k] for k in ['strong_flips','flips','stable_low','stable_high','maximum_errors','threshold_disagreements']},ensure_ascii=False,indent=2))
