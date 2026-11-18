from pathlib import Path
import json,hashlib,csv,numpy as np
q=Path('/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/temporal_diagnosis_20261010');p=q/'stage1_goal';v=q/'problem_validity_20261011';rtd=v/'runtime/window_scoring'
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
 return h.hexdigest()
script_hash=digest(p/'pilot/score_pilot.py');helper_hash=digest(q/'analysis/analyze.py');records=[]
for seq in ['DJI_0026_2','DJI_0067_2']:
 launch=json.loads((rtd/(seq+'_launch.json')).read_text());assert launch['scoring_script_sha256']==script_hash
 data=p/'data'/seq;pred=p/'predictions/official'/seq;score=v/'window_scores'/seq;completion=json.loads((pred/'COMPLETE.json').read_text());annhash=digest(data/'annotations.json')
 fused=p/'evaluation/official'/seq/'official_fused_COMPLETE.json';fm=json.loads(fused.read_text());assert fm['scoring_script_sha256']==script_hash and fm['matching_helper_sha256']==helper_hash and fm['annotations_sha256']==annhash and fm['prediction_sha256']==completion['fused_sha256']
 row={'sequence':seq,'status':'PASS','script_sha256':script_hash,'helper_sha256':helper_hash,'windows':[],'reused_original_fused_complete_sha256':digest(fused)}
 for start in range(0,121,12):
  name=f'window_{start:03d}';mp=score/(name+'_COMPLETE.json');meta=json.loads(mp.read_text());assert meta['status']=='COMPLETE' and meta['thresholds']==[0.5] and meta['start']==start and meta['end']==start+40
  assert meta['scoring_script_sha256']==script_hash and meta['matching_helper_sha256']==helper_hash and meta['annotations_sha256']==annhash
  assert meta['prediction_sha256']==completion['window_sha256'][name+'.npz']==digest(pred/'windows'/(name+'.npz'))
  with np.load(score/(name+'_matches.npz'),allow_pickle=False) as z:
   assert z['thresholds'].tolist()==[0.5] and z['matched'].shape==(1,len(meta['objects'])) and z['frame_counts'].shape==(1,40,6)
   assert np.array_equal(z['strict_small'],[bool(x['strict_small']) for x in meta['objects']]);assert z['offsets'][-1]==len(meta['objects'])
   assert z['matched'].sum()==z['frame_counts'][0,:,1].sum();assert np.all(z['frame_counts'][0,:,1]+z['frame_counts'][0,:,3]==z['frame_counts'][0,:,2])
   curve=list(csv.DictReader((score/(name+'_curves.csv')).open()));assert len(curve)==1 and float(curve[0]['threshold'])==0.5
   tot=z['frame_counts'][0].sum(0)
   for k,j in [('gt',0),('matched',1),('components',2),('unmatched_components',3),('positive_pixels',4),('positive_pixels_outside_boxes',5)]:assert int(curve[0][k])==int(tot[j])
   assert int(curve[0]['strict_small_matched'])==int(z['matched'][0,z['strict_small']].sum())
   counts={k:int(curve[0][k]) for k in ['gt','matched','strict_small_gt','strict_small_matched','unmatched_components']}
  files={suffix:digest(score/(name+suffix)) for suffix in ['_COMPLETE.json','_matches.npz','_curves.csv']}
  row['windows'].append({'start':start,'counts':counts,'output_sha256':files})
 assert len(list(score.glob('*_COMPLETE.json')))==11
 records.append(row)
result={'status':'PASS','sequences':records,'gpu_requested':False,'confirmation_predictions_read':False,'original_fusion_rescored':False}
(rtd/'VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'PASS','sequences':len(records),'windows':sum(len(x['windows']) for x in records),'scoring_script_sha256':script_hash}))
