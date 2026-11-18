from pathlib import Path
import json,csv,math,hashlib,time,collections,concurrent.futures
import numpy as np
import cv2
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from PIL import Image

BASE=Path('/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/temporal_diagnosis_20261010/stage1_goal')
CAL=json.loads((BASE/'summary/CALIBRATION.json').read_text())
SEQS=CAL['calibration_sequences']
assert SEQS==['DJI_0026_2','DJI_0067_2']
GRID=CAL['threshold_grid']
VARIANTS=['official','paired_mean','paired_max','ohem','random_pair_max']
FIELDS=['gt','matched','components','unmatched_components','positive_pixels','positive_pixels_outside_boxes']
cv2.setNumThreads(1)

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
 return h.hexdigest()

def ground_truth(seq):
 d=BASE/'data'/seq;j=json.loads((d/'annotations.json').read_text());manifest=json.loads((d/'FRAME_MANIFEST.json').read_text())
 raw=np.load(d/'raw_frames.npy',mmap_mode='r',allow_pickle=False)
 byid=collections.defaultdict(list)
 for a in j['annotations']:byid[a['image_id']].append(a)
 allmeta=[];frames=[];offsets=[0];registration=[]
 assert raw.shape==(160,512,640) and raw.dtype==np.uint8
 assert sha(d/'raw_frames.npy')==manifest['array_sha256']
 assert sha(d/'annotations.json')==manifest['annotations_sha256']
 assert len(j['images'])==len(manifest['frames'])==160
 for t,(im,entry) in enumerate(zip(j['images'],manifest['frames'])):
  path=d/entry['relative_path'];decoded=np.asarray(Image.open(path))
  checks=[entry['index']==t,entry['source_image_id']==im['id'],entry['source_file_name']==im['file_name'],int(Path(im['file_name']).stem)==entry['raw_filename_number'],sha(path)==entry['sha256'],np.array_equal(decoded,raw[t]),set(entry['annotation_ids'])=={a['id'] for a in byid[im['id']]}]
  if not all(checks):registration.append({'frame':t,'checks':checks})
  anns=[]
  for a in sorted(byid[im['id']],key=lambda a:(a['track_id'],a['id'])):
   x,y,w,h=map(float,a['bbox'])
   if a.get('ignore',0) or w<=0 or h<=0 or min(640.,x+w)<=max(0.,x) or min(512.,y+h)<=max(0.,y):continue
   anns.append(a)
  boxes=np.asarray([a['bbox'] for a in anns],dtype=float).reshape(-1,4);strict=[];mask=np.zeros((512,640),bool)
  for i,(a,(x,y,w,h)) in enumerate(zip(anns,boxes)):
   overlap=any(max(x,xx)<min(x+w,xx+ww) and max(y,yy)<min(y+h,yy+hh) for k,(xx,yy,ww,hh) in enumerate(boxes) if k!=i)
   ok=max(w,h)<=16 and x>=0 and y>=0 and x+w<=640 and y+h<=512 and not overlap;strict.append(ok)
   x0=max(0,min(640,math.ceil(x-.5)));x1=max(0,min(640,math.ceil(x+w-.5)));y0=max(0,min(512,math.ceil(y-.5)));y1=max(0,min(512,math.ceil(y+h-.5)));mask[y0:y1,x0:x1]=True
   allmeta.append({'frame':t,'source_frame':int(Path(im['file_name']).stem),'image_id':im['id'],'track_id':a['track_id'],'annotation_id':a['id'],'category_id':a['category_id'],'bbox':list(map(float,a['bbox'])),'strict_small':int(ok)})
  frames.append((boxes,np.asarray(strict,bool),mask));offsets.append(len(allmeta))
 return {'frames':frames,'objects':allmeta,'offsets':np.asarray(offsets),'raw_sha256':sha(d/'raw_frames.npy'),'annotations_sha256':sha(d/'annotations.json'),'registration_issues':registration,'raw_jpg_and_metadata_frames_verified':160,'source_frame_range':[int(Path(j['images'][0]['file_name']).stem),int(Path(j['images'][-1]['file_name']).stem)],'ignored':sum(bool(a.get('ignore',0)) for a in j['annotations']),'iscrowd':sum(bool(a.get('iscrowd',0)) for a in j['annotations'])}

GT={seq:ground_truth(seq) for seq in SEQS}

def independently_match(prob,threshold,boxes,strict,gtmask):
 positive=np.asarray(prob>=threshold,dtype=np.uint8)
 count,labels=cv2.connectedComponents(positive,connectivity=8,ltype=cv2.CV_32S)
 components=count-1
 pos=int(positive.sum());outside=int(np.count_nonzero(positive.astype(bool)&~gtmask))
 if not components:return [len(boxes),0,0,0,pos,outside],0
 flat=np.flatnonzero(positive.ravel());lab=labels.ravel()[flat];values=prob.ravel()[flat]
 maximum=np.full(count,-np.inf,np.float32);np.maximum.at(maximum,lab,values)
 atmax=values==maximum[lab]
 first=np.full(count,prob.size,np.int64);np.minimum.at(first,lab[atmax],flat[atmax])
 py,px=np.divmod(first[1:],640);px=px+.5;py=py+.5
 x,y,w,h=boxes.T
 adjacency=(x[:,None]<=px)&(px<x[:,None]+w[:,None])&(y[:,None]<=py)&(py<y[:,None]+h[:,None])
 assignment=maximum_bipartite_matching(csr_matrix(adjacency),perm_type='column')
 matched=assignment>=0;n=int(matched.sum())
 return [len(boxes),n,components,components-n,pos,outside],int(np.count_nonzero(matched&strict))

def audit_one(item):
 variant,seq=item;start=time.perf_counter();d=BASE/'evaluation'/variant/seq;pred=BASE/'predictions'/variant/seq;gt=GT[seq]
 meta=json.loads((d/'official_fused_COMPLETE.json').read_text());pm=json.loads((pred/'COMPLETE.json').read_text());fp=CAL['source_fingerprints'][variant][seq]
 issues=[]
 for suffix in ['curves.csv','matches.npz','COMPLETE.json']:
  assert sha(d/('official_fused_'+suffix))==fp[suffix]
 assert sha(pred/'official_fused.npz')==fp['prediction_sha256']==pm['fused_sha256']==meta['prediction_sha256']
 assert gt['annotations_sha256']==fp['annotations_sha256']==meta['annotations_sha256']
 assert gt['raw_sha256']==pm['raw_sha256']
 assert meta['objects']==gt['objects']
 assert meta['scoring_script_sha256']==sha(BASE/'pilot/score_pilot.py')
 assert meta['matching_helper_sha256']==sha(BASE.parent/'analysis/analyze.py')
 assert pm['script_sha256']==sha(BASE/'pilot/infer_pilot.py')
 with np.load(d/'official_fused_matches.npz',allow_pickle=False) as z:saved={k:z[k] for k in z.files}
 with (d/'official_fused_curves.csv').open() as f:curves=[{k:float(v) if k=='threshold' else int(v) for k,v in r.items()} for r in csv.DictReader(f)]
 assert saved['thresholds'].tolist()==GRID==[r['threshold'] for r in curves]
 assert np.array_equal(saved['offsets'],gt['offsets'])
 assert np.array_equal(saved['strict_small'],[o['strict_small'] for o in gt['objects']])
 for ti,row in enumerate(curves):
  for k,v in zip(FIELDS,saved['frame_counts'][ti].sum(0)):assert row[k]==int(v)
  assert row['matched']==int(saved['matched'][ti].sum())
  assert row['strict_small_matched']==int(saved['matched'][ti,saved['strict_small']].sum())
  assert row['frames']==160 and row['strict_small_gt']==int(saved['strict_small'].sum())
 with np.load(pred/'official_fused.npz',allow_pickle=False) as z:
  prob=z['probabilities'];coverage=z['coverage'];starts=z['starts']
 assert prob.shape==(160,512,640) and prob.dtype==np.float32 and np.isfinite(prob).all() and prob.min()>=0 and prob.max()<=1
 reconstructed=np.zeros_like(prob);rcov=np.zeros(160,np.int32)
 for s in [0,36,72,108,120]:
  path=pred/'windows'/f'window_{s:03d}.npz';assert sha(path)==pm['window_sha256'][path.name]
  with np.load(path,allow_pickle=False) as z:
   assert int(z['start'])==s and int(z['end'])==s+40
   reconstructed[s:s+40]=np.maximum(reconstructed[s:s+40],z['probabilities']);rcov[s:s+40]+=1
 assert np.array_equal(prob,reconstructed) and np.array_equal(coverage,rcov) and starts.tolist()==[0,36,72,108,120]
 del reconstructed
 calculated=np.zeros_like(saved['frame_counts']);strictcounts=np.zeros((len(GRID),160),np.int64)
 for ti,th in enumerate(GRID):
  for t,(boxes,strict,mask) in enumerate(gt['frames']):
   calculated[ti,t],strictcounts[ti,t]=independently_match(prob[t],th,boxes,strict,mask)
  if not np.array_equal(calculated[ti],saved['frame_counts'][ti]):
   where=np.argwhere(calculated[ti]!=saved['frame_counts'][ti]);issues.extend({'threshold':th,'frame':int(t),'field':FIELDS[int(k)],'independent':int(calculated[ti,t,k]),'saved':int(saved['frame_counts'][ti,t,k])} for t,k in where)
  for t in range(160):
   lo,hi=gt['offsets'][t:t+2];expected=int(saved['matched'][ti,lo:hi][saved['strict_small'][lo:hi]].sum())
   if strictcounts[ti,t]!=expected:issues.append({'threshold':th,'frame':t,'field':'strict_small_matched','independent':int(strictcounts[ti,t]),'saved':expected})
  if ti in [0,11,22]:print(json.dumps({'progress':variant+'/'+seq,'threshold':th,'seconds':time.perf_counter()-start}),flush=True)
 sat={'exact_one_pixels':int(np.count_nonzero(prob==1)),'greater_equal_one_pixels':int(np.count_nonzero(prob>=1)),'greater_than_one_pixels':int(np.count_nonzero(prob>1)),'frames_with_exact_one':int(np.count_nonzero((prob==1).reshape(160,-1).any(1))),'probability_min':float(prob.min()),'probability_max':float(prob.max()),'dtype':str(prob.dtype)}
 recomputed=[{'threshold':th,'frames':160,**{k:int(v) for k,v in zip(FIELDS,calculated[ti].sum(0))},'strict_small_gt':int(saved['strict_small'].sum()),'strict_small_matched':int(strictcounts[ti].sum())} for ti,th in enumerate(GRID)]
 result={'variant':variant,'sequence':seq,'status':'PASS' if not issues else 'MISMATCH','hashes_verified':True,'annotation_object_metadata_and_offsets_match':True,'fusion_exactly_reconstructed':True,'full_raw_map_thresholds_recomputed':len(GRID),'raw_frame_threshold_evaluations':160*len(GRID),'all_frame_counts_match':not issues,'issues':issues,'saturation':sat,'curves':recomputed,'seconds':time.perf_counter()-start}
 print(json.dumps({'complete':variant+'/'+seq,'mismatches':len(issues),'seconds':result['seconds']}),flush=True)
 return result

begin=time.perf_counter()
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 results=list(pool.map(audit_one,[(v,s) for v in VARIANTS for s in SEQS]))
aggregated={}
for variant in VARIANTS:
 rr=[r for r in results if r['variant']==variant];curves=[{'threshold':t,**{k:sum(r['curves'][i][k] for r in rr) for k in rr[0]['curves'][i] if k!='threshold'}} for i,t in enumerate(GRID)]
 feasible=[r for r in curves if r['unmatched_components']*CAL['budget']['frames']<=CAL['budget']['unmatched_components']*r['frames']]
 aggregated[variant]={'curves':curves,'feasible_thresholds':[r['threshold'] for r in feasible],'minimum_U_point':min(curves,key=lambda x:x['unmatched_components']),'exact_one_pixels':sum(r['saturation']['exact_one_pixels'] for r in rr),'at_threshold_one':curves[-1]}
assert sha(BASE/'summary/CALIBRATION.json')=='2329b4d59dc1eaf2a0cfe71b743af7e8c7e5d682d175c3a88e03b1fae96dde8e'
baseline=next(r for r in aggregated['official']['curves'] if r['threshold']==.5)
assert baseline==CAL['variants']['official']['selected_calibration_counts']
with (BASE/'summary/CALIBRATION_CURVES.csv').open() as f:
 for row in csv.DictReader(f):
  ours=next(r for r in aggregated[row['variant']]['curves'] if r['threshold']==float(row['threshold']))
  assert all(ours[k]==(float(v) if k=='threshold' else int(v)) for k,v in row.items() if k!='variant')
payload={'status':'PASS_NUMERIC_CALIBRATION_FAILURE' if all(r['status']=='PASS' for r in results) and all(not aggregated[v]['feasible_thresholds'] for v in VARIANTS[1:]) else 'REVIEW_REQUIRED','calibration_sha256':sha(BASE/'summary/CALIBRATION.json'),'read_scope':SEQS,'confirmation_accessed':False,'GPU_used':False,'algorithm':'Independent OpenCV connectedComponents 8-connectivity; numpy per-label maximum and minimum row-major index; SciPy sparse maximum bipartite matching. Original scoring/matching/summary functions not imported.','registration':{s:{k:v for k,v in gt.items() if k not in ['frames','objects','offsets']} for s,gt in GT.items()},'per_variant_sequence':results,'aggregated':aggregated,'baseline_budget_recomputed':baseline,'elapsed_seconds':time.perf_counter()-begin,'source_sha256':{'audit_script':sha('/audit.py'),'scorer':sha(BASE/'pilot/score_pilot.py'),'matcher':sha(BASE.parent/'analysis/analyze.py'),'inference':sha(BASE/'pilot/infer_pilot.py'),'summary':sha(BASE/'pilot/summarize_pilot.py')}}
print('AUDIT_RESULT_JSON='+json.dumps(payload,ensure_ascii=False),flush=True)
