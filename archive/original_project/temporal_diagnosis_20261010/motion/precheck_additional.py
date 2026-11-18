from pathlib import Path
import json,collections,math,datetime,hashlib
Q=Path('/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/temporal_diagnosis_20261010'); summaries=[]
for inp in sorted((Q/'additional_clips').glob('DJI*')):
 if not (inp/'FRAME_MANIFEST.json').exists(): continue
 m=json.loads((inp/'FRAME_MANIFEST.json').read_text());a=json.loads((inp/'annotations.json').read_text());sel=m['selected_original_track_id'];byframe=collections.defaultdict(list);frames={f['source_image_id']:f['raw_frame']for f in m['frames']}
 for z in a['annotations']:byframe[frames[z['image_id']]].append(z)
 counts=collections.Counter(z['track_id']for z in a['annotations']);rows=[];dup=[];trunc=[];step=[];nearids=collections.Counter();neighrows=[]
 for f,anns in sorted(byframe.items()):
  selected=[z for z in anns if z['track_id']==sel]
  if len(selected)!=1:dup.append({'frame':f,'selected_count':len(selected)});continue
  z=selected[0];x,y,w,h=z['bbox'];c=[x+w/2,y+h/2];rows.append({'frame':f,'center':c,'bbox':z['bbox'],'annotation_id':z['id']})
  if x<0 or y<0 or x+w>640 or y+h>512:trunc.append(f)
  nr=[]
  for b in anns:
   if b['track_id']==sel:continue
   xx,yy,ww,hh=b['bbox'];dist=math.hypot(xx+ww/2-c[0],yy+hh/2-c[1]);inter=max(0,min(x+w,xx+ww)-max(x,xx))*max(0,min(y+h,yy+hh)-max(y,yy));iou=inter/(w*h+ww*hh-inter)if w*h+ww*hh-inter>0 else None
   nr.append({'other_track_id':b['track_id'],'center_distance_px':dist,'intersection_px2':inter,'iou':iou})
   if inter>0:nearids[b['track_id']]+=1
  neighrows.append({'frame':f,'overlapping_neighbors':[n for n in nr if n['intersection_px2']>0],'nearest_neighbor':min(nr,key=lambda n:n['center_distance_px'])if nr else None})
 for p,n in zip(rows,rows[1:]):
  step.append({'from':p['frame'],'to':n['frame'],'distance':math.dist(p['center'],n['center'])})
 out=Q/'motion'/m['sequence'];out.mkdir(parents=True,exist_ok=True)
 r={'sequence':m['sequence'],'candidate_id':m['candidate_id'],'selected_track_id':sel,'selected_presence':len(rows),'selected_duplicate_or_missing_per_available_frame':dup,'selected_expected_frames_missing':sorted(set(f['raw_frame']for f in m['frames'])-set(z['frame']for z in rows)),'selected_truncated_frame_ids':trunc,'unique_tracks':len(counts),'source_track_presence':dict(counts),'selected_min_width':min(z['bbox'][2]for z in rows),'selected_max_width':max(z['bbox'][2]for z in rows),'selected_min_height':min(z['bbox'][3]for z in rows),'selected_max_height':max(z['bbox'][3]for z in rows),'selected_max_center_step_px':max(z['distance']for z in step),'selected_overlapping_frame_count':sum(bool(z['overlapping_neighbors'])for z in neighrows),'selected_identical_box_frames':[z['frame']for z in neighrows if any(n['iou']==1 for n in z['overlapping_neighbors'])],'selected_overlap_tracks':dict(nearids),'selected_min_neighbor_distance_px':min(z['nearest_neighbor']['center_distance_px']for z in neighrows if z['nearest_neighbor']),'neighbor_per_frame':neighrows,'no_model_outputs_read':True,'annotation_presence_does_not_verify_visibility':True}
 (out/'ANNOTATION_PRECHECK.json').write_text(json.dumps(r,indent=2)+'\n')
 protocol=json.loads((Q/'motion/PROTOCOL.json').read_text());protocol['recorded_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();protocol['input']=f"fixed {m['candidate_id']}, 160 native640x512 grayscales; all source annotations; no model predictions";protocol['original_protocol_sha256']=(Q/'motion/PROTOCOL.sha256').read_text().strip();protocol['inheritance']='All algorithm parameters and validity thresholds unchanged from first clip; applies source-boundary amendment to intensity sampling.'
 ps=json.dumps(protocol,indent=2)+'\n';(out/'PROTOCOL.json').write_text(ps);(out/'PROTOCOL.sha256').write_text(hashlib.sha256(ps.encode()).hexdigest()+'\n');summaries.append({k:v for k,v in r.items()if k not in ['neighbor_per_frame','source_track_presence']})
(Q/'motion/ADDITIONAL_ANNOTATION_PRECHECK.json').write_text(json.dumps(summaries,indent=2)+'\n');print(json.dumps(summaries,indent=2))
