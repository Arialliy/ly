#!/usr/bin/env python3
import os,json,hashlib,datetime,csv,math,argparse
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
import cv2
from PIL import Image,ImageDraw
D=Path('/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009')
parser=argparse.ArgumentParser();parser.add_argument('--input',required=True);parser.add_argument('--output',required=True);args=parser.parse_args();O=Path(args.output);S=Path(args.input)
P=json.loads((O/'PROTOCOL.json').read_text()); np.random.seed(P['seed']); cv2.setRNGSeed(P['seed']); cv2.setNumThreads(2)
manifest=json.loads((S/'FRAME_MANIFEST.json').read_text()); anno=json.loads((S/'annotations.json').read_text()); files=manifest['frames']
byimage=defaultdict(list); tracks=defaultdict(dict)
for a in anno['annotations']: byimage[a['image_id']].append(a)
frames=[]; masks=[]; annotation_errors=[]; frame_metrics=[]
for ti,fr in enumerate(files):
 raw=(S/fr['local_path']).read_bytes()
 if hashlib.sha256(raw).hexdigest()!=fr['sha256']: raise ValueError('SHA mismatch '+fr['local_path'])
 im=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_GRAYSCALE); h,w=im.shape; assert(h,w)==(512,640)
 frames.append(im); mask=np.ones((h,w),np.uint8)*255; mask[:4]=0;mask[-4:]=0;mask[:,:4]=0;mask[:,-4:]=0
 for a in byimage[fr['source_image_id']]:
  x,y,bw,bh=a['bbox']; track=a['track_id']
  if ti in tracks[track]: annotation_errors.append({'kind':'duplicate_track_frame','track':track,'raw_frame':fr['raw_frame']})
  tracks[track][ti]=a
  if bw<=0 or bh<=0 or x<0 or y<0 or x+bw>w or y+bh>h: annotation_errors.append({'kind':'invalid_bbox','annotation_id':a['id']})
  x0=max(0,int(math.floor(x-8)));x1=min(w,int(math.ceil(x+bw+8)));y0=max(0,int(math.floor(y-8)));y1=min(h,int(math.ceil(y+bh+8))); mask[y0:y1,x0:x1]=0
 masks.append(mask)
 for a in byimage[fr['source_image_id']]:
  x,y,bw,bh=a['bbox'];x0=max(0,int(math.floor(x)));x1=min(w,int(math.ceil(x+bw)));y0=max(0,int(math.floor(y)));y1=min(h,int(math.ceil(y+bh)))
  tp=im[y0:y1,x0:x1].astype(float); rm=np.zeros((h,w),np.uint8)
  rm[max(0,y0-12):min(h,y1+12),max(0,x0-12):min(w,x1+12)]=1
  for b in byimage[fr['source_image_id']]:
   bx,by,bww,bhh=b['bbox'];rm[max(0,int(math.floor(by-4))):min(h,int(math.ceil(by+bhh+4))),max(0,int(math.floor(bx-4))):min(w,int(math.ceil(bx+bww+4)))]=0
  rp=im[rm>0].astype(float)
  row={'offset':ti,'raw_frame':fr['raw_frame'],'track_id':a['track_id'],'annotation_id':a['id'],'width':bw,'height':bh,'area':bw*bh,'center_x':x+bw/2,'center_y':y+bh/2,'ring_pixels':len(rp),'contrast_valid':len(rp)>=20 and tp.size>0,'bbox_truncated_by_image':x<0 or y<0 or x+bw>w or y+bh>h,'sampled_box_pixels':int(tp.size)}
  if len(rp)>=20 and tp.size>0: row.update(box_mean=float(tp.mean()),ring_mean=float(rp.mean()),ring_std=float(rp.std()),mean_difference=float(tp.mean()-rp.mean()),p90_minus_ring_median=float(np.percentile(tp,90)-np.median(rp)),standardized_p90_contrast=float((np.percentile(tp,90)-np.median(rp))/max(rp.std(),1)))
  frame_metrics.append(row)

def summary(values):
 a=np.array([v for v in values if v is not None],float)
 if not len(a): return {'n':0}
 return {'n':int(len(a)),'min':float(a.min()),'p25':float(np.percentile(a,25)),'median':float(np.median(a)),'p75':float(np.percentile(a,75)),'p90':float(np.percentile(a,90)),'max':float(a.max()),'mean':float(a.mean())}
def bounds_ok(p,mask):
 q=np.round(p).astype(int); b=(q[:,0]>=0)&(q[:,0]<640)&(q[:,1]>=0)&(q[:,1]<512);r=np.zeros(len(p),bool);r[b]=mask[q[b,1],q[b,0]]>0;return r
pairrows=[]; matrices={}; hulls={}; supports={}; diagnostic_points={}
for i in range(len(frames)-5):
 j=i+5;a=frames[i];b=frames[j];row={'offset_from':i,'offset_to':j,'raw_frame_from':files[i]['raw_frame'],'raw_frame_to':files[j]['raw_frame'],'valid':False,'invalid_reasons':[]}
 p0=cv2.goodFeaturesToTrack(a,mask=masks[i],**P['corners']); row['corners']=0 if p0 is None else len(p0)
 if p0 is None or len(p0)<30: row['invalid_reasons'].append('fewer_than_30_corners');pairrows.append(row);continue
 lk=dict(winSize=tuple(P['lk']['winSize']),maxLevel=P['lk']['maxLevel'],criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
 p1,st,e=cv2.calcOpticalFlowPyrLK(a,b,p0,None,**lk);pback,sback,eback=cv2.calcOpticalFlowPyrLK(b,a,p1,None,**lk)
 p=p0[:,0];q=p1[:,0];back=pback[:,0];fb=np.linalg.norm(p-back,axis=1);valid=(st[:,0]>0)&(sback[:,0]>0)&(fb<=1)&bounds_ok(q,masks[j]);p=p[valid];q=q[valid];row['tracked_pairs']=len(p);row['forward_backward_error']=summary(fb[valid]);
 if len(p)<30: row['invalid_reasons'].append('fewer_than_30_tracked_pairs');pairrows.append(row);continue
 cv2.setRNGSeed(46)
 A,inliers=cv2.estimateAffine2D(p,q,method=cv2.RANSAC,ransacReprojThreshold=1.5,maxIters=2000,confidence=.99,refineIters=10)
 if A is None: row['invalid_reasons'].append('no_affine');pairrows.append(row);continue
 ins=inliers[:,0].astype(bool);pi=p[ins];qi=q[ins]; pred=np.column_stack([pi,np.ones(len(pi))])@A.T;res=np.linalg.norm(qi-pred,axis=1);hull=cv2.convexHull(pi);area=cv2.contourArea(hull)/(640*512);sp=np.ptp(pi,axis=0)/[640,512];cells=set((min(3,int(x//160)),min(3,int(y//128)))for x,y in pi);sv=np.linalg.svd(A[:,:2],compute_uv=False);shear=abs(np.dot(A[:,0],A[:,1])/(np.linalg.norm(A[:,0])*np.linalg.norm(A[:,1])));medres=float(np.median(res))
 row.update(affine=A.tolist(),inliers=int(ins.sum()),inlier_ratio=float(ins.mean()),inlier_residual=summary(res),inlier_hull_fraction=area,inlier_x_span_fraction=float(sp[0]),inlier_y_span_fraction=float(sp[1]),occupied_grid_cells=len(cells),singular_values=sv.tolist(),shear_abs_cosine=float(shear),fitted_residual_y_lt400=summary(res[pi[:,1]<400]),fitted_residual_y_ge400=summary(res[pi[:,1]>=400]),inlier_raw_displacement_px=summary(np.linalg.norm(qi-pi,axis=1)))
 checks=[(ins.sum()>=25,'inliers_lt25'),(ins.mean()>=.6,'inlier_ratio_lt0.6'),(medres<=.75,'median_residual_gt0.75'),(area>=.1,'hull_fraction_lt0.1'),(sp[0]>=.45,'x_span_lt0.45'),(sp[1]>=.3,'y_span_lt0.3'),(len(cells)>=6,'grid_cells_lt6'),(sv.min()>=.95 and sv.max()<=1.05,'singular_values_outside_range'),(shear<=.05,'shear_gt0.05')]
 row['invalid_reasons']=[s for test,s in checks if not test];row['valid']=not row['invalid_reasons'];matrices[i]=A;hulls[i]=hull;supports[i]=pi;pairrows.append(row)
 if i in [0,40,80,120]:diagnostic_points[i]=(p,q,ins,hull)
 if i%40==0:print('pair',i,'valid',row['valid'],'inliers',row['inliers'],'coverage',round(area,3),flush=True)

motionrows=[]
for tid,track in sorted(tracks.items()):
 for i in range(155):
  if i not in track or i+5 not in track: continue
  a=track[i]['bbox'];b=track[i+5]['bbox'];c0=np.array([a[0]+a[2]/2,a[1]+a[3]/2]);c1=np.array([b[0]+b[2]/2,b[1]+b[3]/2]);row={'track_id':tid,'offset_from':i,'raw_frame_from':files[i]['raw_frame'],'raw_frame_to':files[i+5]['raw_frame'],'raw_speed_px_per_frame':float(np.linalg.norm(c1-c0)/5),'camera_adjustment_valid':False,'global_registration_valid':pairrows[i]['valid']}
  if i in matrices:
   A=matrices[i];cam=A@np.r_[c0,1];inside=cv2.pointPolygonTest(hulls[i],tuple(c0),False)>=0;nearest=float(np.min(np.linalg.norm(supports[i]-c0,axis=1)));res=c1-cam;row.update(source_center_inside_inlier_hull=inside,nearest_inlier_distance_px=nearest,camera_displacement_at_center_px_per_frame=float(np.linalg.norm(cam-c0)/5),camera_adjustment_valid=bool(pairrows[i]['valid']and inside and nearest<=160))
   if row['camera_adjustment_valid']:row.update(adjusted_speed_px_per_frame=float(np.linalg.norm(res)/5),adjusted_vx_px_per_frame=float(res[0]/5),adjusted_vy_px_per_frame=float(res[1]/5))
  motionrows.append(row)
trackreports=[]
for tid,track in sorted(tracks.items()):
 times=sorted(track);anns=[track[t]for t in times];centers=np.array([[a['bbox'][0]+a['bbox'][2]/2,a['bbox'][1]+a['bbox'][3]/2]for a in anns]);step_pairs=[(k,float(np.linalg.norm(centers[k+1]-centers[k])))for k in range(len(times)-1)if times[k+1]==times[k]+1];steps=np.array([v for k,v in step_pairs]);sizes=np.array([a['bbox'][2:]for a in anns]);thr=max(8,.5*np.median(sizes.max(axis=1)));tm=[r for r in frame_metrics if r['track_id']==tid];blocks=[]
 for start in range(0,160,40):
  mr=[r for r in motionrows if r['track_id']==tid and start<=r['offset_from']and r['offset_from']+5<start+40];fm=[r for r in tm if start<=r['offset']<start+40];br={'raw_start':files[start]['raw_frame'],'raw_end':files[start+39]['raw_frame'],'lag5_pairs':len(mr),'camera_adjustment_valid_pairs':sum(r['camera_adjustment_valid']for r in mr),'raw_speed_px_per_frame':summary(r['raw_speed_px_per_frame']for r in mr),'adjusted_speed_px_per_frame':summary(r.get('adjusted_speed_px_per_frame')for r in mr),'background_displacement_at_center_px_per_frame':summary(r.get('camera_displacement_at_center_px_per_frame')for r in mr),'width_px':summary(r['width']for r in fm),'height_px':summary(r['height']for r in fm),'mean_difference':summary(r.get('mean_difference')for r in fm),'p90_minus_ring_median':summary(r.get('p90_minus_ring_median')for r in fm),'standardized_p90_contrast':summary(r.get('standardized_p90_contrast')for r in fm)};blocks.append(br)
 trackreports.append({'track_id':tid,'category_ids':sorted(set(a['category_id']for a in anns)),'present_annotation_frames':len(times),'missing_offsets':sorted(set(range(160))-set(times)),'continuous_annotations':times==list(range(160)),'visibility_verified':False,'one_step_center_displacement_px':summary(steps),'large_step_threshold_px':float(thr),'flagged_steps':[{'raw_from':files[times[i]]['raw_frame'],'raw_to':files[times[i+1]]['raw_frame'],'distance_px':float(v)}for i,v in step_pairs if v>thr],'width_px':summary(sizes[:,0]),'height_px':summary(sizes[:,1]),'area_px2':summary(np.prod(sizes,axis=1)),'blocks':blocks})
report={'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'seed':46,'image':'deeppro-official-eval:20261009','image_id':'sha256:eb956873870d737f2b68dff0724b8f6f56f778829d119b0938d29b7abeaa2f92','opencv':cv2.__version__,'numpy':np.__version__,'protocol_sha256':(O/'PROTOCOL.sha256').read_text().strip(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'input_manifest_sha256':hashlib.sha256((S/'FRAME_MANIFEST.json').read_bytes()).hexdigest(),'input_annotations_sha256':hashlib.sha256((S/'annotations.json').read_bytes()).hexdigest(),'frame_count':160,'annotation_count':len(anno['annotations']),'track_count':len(tracks),'candidate_id':manifest['candidate_id'],'selected_track_id':manifest['selected_original_track_id'],'raw_frames_consecutive':[f['raw_frame']for f in files]==list(range(files[0]['raw_frame'],files[0]['raw_frame']+160)),'annotation_errors':annotation_errors,'annotation_ignore_values':dict(Counter(str(a.get('ignore'))for a in anno['annotations'])),'annotation_iscrowd_values':dict(Counter(str(a.get('iscrowd'))for a in anno['annotations'])),'no_predictions_read':True,'registration':{'pair_count':len(pairrows),'valid_pairs':sum(r['valid']for r in pairrows),'invalid_reason_counts':dict(Counter(s for r in pairrows for s in r['invalid_reasons'])),'inlier_count':summary(r.get('inliers')for r in pairrows),'inlier_ratio':summary(r.get('inlier_ratio')for r in pairrows),'hull_fraction':summary(r.get('inlier_hull_fraction')for r in pairrows),'median_residual_px':summary(r.get('inlier_residual',{}).get('median')for r in pairrows)},'tracks':trackreports,'interpretation_limits':['Global affine explains shared image displacement, not guaranteed pure physical camera motion; scene depth, moving water and foreground vegetation may violate one-affine assumption.','Adjusted target displacement is image-coordinate motion relative to fitted background, not physical target speed.','Source bounding boxes contain background; local contrast is box-based and not true target-mask SNR.','All eight tracks having labels on 160 frames proves annotation continuity, not continuous visibility.','Single preselected development clip cannot establish broad dataset or method deficiency.','No detector predictions read; this report cannot establish learned model failure.']}
(O/'MOTION_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n');(O/'REGISTRATION_PAIRS.json').write_text(json.dumps(pairrows,indent=2)+'\n')
for name,rows in [('TRACK_FRAME_DIAGNOSTICS.csv',frame_metrics),('LAG5_TRACK_MOTION.csv',motionrows)]:
 keys=list(dict.fromkeys(k for row in rows for k in row))
 with (O/name).open('w')as f:writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
# Derived analytical contact sheets; preserve source bytes and nearest-neighbor pixels.
contact=Image.new('RGB',(5*256,len(tracks)*154),'white');draw=ImageDraw.Draw(contact)
for ri,tid in enumerate(sorted(tracks)):
 for ci,ti in enumerate([0,40,80,120,159]):
  if ti not in tracks[tid]:
   draw.text((ci*256+10,ri*154+14),f'{tid} | f={files[ti]["raw_frame"]} | no label',fill='black');continue
  box=tracks[tid][ti]['bbox'];cx=box[0]+box[2]/2;cy=box[1]+box[3]/2;x0=max(0,min(640-48,int(cx)-24));y0=max(0,min(512-48,int(cy)-24));crop=Image.fromarray(frames[ti][y0:y0+48,x0:x0+48]).convert('RGB').resize((144,144),Image.Resampling.NEAREST);contact.paste(crop,(ci*256,ri*154+10));draw.text((ci*256+148,ri*154+14),f'{tid}\nf={files[ti]["raw_frame"]}\n{box[2]}x{box[3]}',fill='black')
contact.save(O/'ALL_TRACK_CROP_CONTACT.png')
for i,(p,q,ins,hull)in diagnostic_points.items():
 vis=cv2.cvtColor(frames[i],cv2.COLOR_GRAY2BGR)
 for x,y in p[ins]:cv2.circle(vis,(round(float(x)),round(float(y))),2,(0,255,0),-1)
 cv2.polylines(vis,[hull.astype(np.int32)],True,(255,255,0),1)
 for a in byimage[files[i]['source_image_id']]:
  x,y,w,h=map(int,a['bbox']);cv2.rectangle(vis,(x,y),(x+w,y+h),(0,0,255),1)
 cv2.imwrite(str(O/f'BACKGROUND_SUPPORT_{files[i]["raw_frame"]:05d}.png'),vis)
print(json.dumps({'valid_camera_pairs':report['registration']['valid_pairs'],'total':len(pairrows),'track_summaries':[{'id':r['track_id'],'valid_pairs':[b['camera_adjustment_valid_pairs']for b in r['blocks']],'raw':[b['raw_speed_px_per_frame'].get('median')for b in r['blocks']],'adjusted':[b['adjusted_speed_px_per_frame'].get('median')for b in r['blocks']]}for r in trackreports]},indent=2),flush=True)
