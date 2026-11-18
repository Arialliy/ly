#!/usr/bin/env python3
"""Training-only identity/density audit; no images, predictions, or dev labels read."""
from pathlib import Path
from collections import defaultdict, Counter
import csv, gzip, hashlib, json, math

N=Path(__file__).resolve().parents[1]
Q=N.parent
R=Q.parent
P=Q/'stage1_goal'
OUT=N/'data_audit'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,obj):
 (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def table(name,rows):
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def valid(a):return len(a['bbox'])==4 and all(math.isfinite(float(x)) for x in a['bbox']) and min(a['bbox'][2:])>0
def pos(a):return valid(a) and not a.get('ignore',0) and not a.get('iscrowd',0)
def core(a,cy,cx):
 x,y,w,h=a['bbox'];return cx+9<=x and cy+9<=y and x+w<=cx+119 and y+h<=cy+119
def intersects(a,cy,cx):
 x,y,w,h=a['bbox'];return max(x,cx)<min(x+w,cx+128) and max(y,cy)<min(y+h,cy+128)
def original(a):return (a['category_id'],str(a['tracking_id']))
def stat(xs):
 s=sorted(xs)
 return dict(n=len(s),min=s[0],q50=s[int((len(s)-1)*.5)],q90=s[int((len(s)-1)*.9)],q99=s[int((len(s)-1)*.99)],max=s[-1],mean=sum(s)/len(s))
def runs(frames):
 out=[]
 for t in sorted(frames):
  if not out or t!=out[-1][-1]+1:out.append([t])
  else:out[-1].append(t)
 return out

planpath=P/'pilot/TRAIN_PLAN.json'
plan=json.loads(planpath.read_text())
sel=json.loads((P/'DATA_SELECTION.json').read_text())
sourcepath=R/'runtime/VT_TRAIN_01_ANNOTATIONS.json.gz'
sourcebytes=gzip.decompress(sourcepath.read_bytes())
source=json.loads(sourcebytes)
sourceby={a['id']:a for a in source['annotations']}
sourceimages={a['id']:a for a in source['images']}
seqrows=[];winrows=[];crop=[];trackrows=[];alias=[];transitions=[];canonicalrows=[];allmotion=[]
allseq={};missing_fields=Counter();allkeys=Counter();source_mismatch=[];frame_errors=[];duplicates=[];key_reuse=[]
for s in plan['train_sequences']:
 path=P/'data'/s/'annotations.json';d=json.loads(path.read_text())
 images=d['images'];index={im['id']:i for i,im in enumerate(images)};by=defaultdict(list);tr=defaultdict(list);local=defaultdict(list)
 for i,im in enumerate(images):
  if im!=sourceimages[im['id']]:source_mismatch.append([s,'image',im['id']])
  if im['frame_id']!=images[0]['frame_id']+i or im['mot_frame_id']!=images[0]['mot_frame_id']+i or int(Path(im['file_name']).stem)!=int(Path(images[0]['file_name']).stem)+i or im['mot_frame_id']!=int(Path(im['file_name']).stem):frame_errors.append([s,i,im])
 for a in d['annotations']:
  if a!=sourceby[a['id']]:source_mismatch.append([s,'annotation',a['id']])
  for k in ('id','track_id','tracking_id','image_id','bbox','category_id','ignore','iscrowd'):
   if k not in a:missing_fields[k]+=1
  allkeys.update(a.keys())
  t=index[a['image_id']];by[t].append(a);tr[a['track_id']].append((t,a));local[original(a)].append((t,a))
 for t,anns in by.items():
  for typ,ids in [('track_id',[a['track_id'] for a in anns]),('category_tracking_id',[original(a) for a in anns])]:
   duplicates.extend([dict(sequence=s,frame=t,key_type=typ,key=str(k),count=n) for k,n in Counter(ids).items() if n>1])
 for tid,v in tr.items():
  fr=[t for t,a in v];cats={a['category_id'] for t,a in v};names={original(a) for t,a in v}
  if len(names)>1:key_reuse.append([s,tid,sorted(names)])
  rr=runs(fr)
  trackrows.append(dict(sequence=s,track_id=tid,source_keys=str(sorted(names)),frames=len(fr),first=min(fr),last=max(fr),continuous_runs=len(rr),longest_continuous_run=max(map(len,rr)),positive_frames=sum(pos(a) for t,a in v)))
 for key,v in local.items():
  v=sorted(v,key=lambda ta:ta[0]);tids={a['track_id'] for t,a in v}
  canonicalrows.append(dict(sequence=s,category_id=key[0],tracking_id=key[1],numeric_track_ids=';'.join(map(str,sorted(tids))),object_frames=len(v),first=v[0][0],last=v[-1][0],continuous_runs=len(runs(t for t,a in v))))
  if len(tids)>1:
   alias.append(dict(sequence=s,category_id=key[0],tracking_id=key[1],numeric_track_ids=';'.join(map(str,sorted(tids))),object_frames=len(v),same_frame_collisions=len(v)-len({t for t,a in v}),first=v[0][0],last=v[-1][0]))
  for (t,a),(u,b) in zip(v,v[1:]):
   if u!=t+1:continue
   ax,ay,aw,ah=a['bbox'];bx,by_,bw,bh=b['bbox']
   distance=math.hypot(ax+aw/2-bx-bw/2,ay+ah/2-by_-bh/2)
   allmotion.append(dict(sequence=s,category_id=key[0],tracking_id=key[1],from_frame=t,to_frame=u,center_distance=distance,distance_per_larger_box_diagonal=distance/max(math.hypot(aw,ah),math.hypot(bw,bh)),from_bbox=str(a['bbox']),to_bbox=str(b['bbox'])))
   if a['track_id']==b['track_id']:continue
   transitions.append(dict(sequence=s,category_id=key[0],tracking_id=key[1],from_frame=t,to_frame=u,from_numeric_id=a['track_id'],to_numeric_id=b['track_id'],center_distance=distance,from_bbox=str(a['bbox']),to_bbox=str(b['bbox'])))
 seqfull=[]
 for st in range(121):
  aa=[a for t in range(st,st+40) for a in by[t] if pos(a)]
  row=dict(sequence=s,start=st,positive_box_frames=len(aa),numeric_track_union=len({a['track_id'] for a in aa}),source_key_union=len({original(a) for a in aa}),concurrent_positive_max=max(sum(pos(a) for a in by[t]) for t in range(st,st+40)))
  seqfull.append(row);winrows.append(row)
 seqrows.append(dict(sequence=s,annotation_frames=len(images),first_source_frame=int(Path(images[0]['file_name']).stem),last_source_frame=int(Path(images[-1]['file_name']).stem),annotation_count=len(d['annotations']),positive_box_frames=sum(pos(a) for a in d['annotations']),ignored=sum(bool(a.get('ignore')) for a in d['annotations']),crowd=sum(bool(a.get('iscrowd')) for a in d['annotations']),numeric_track_ids=len(tr),source_category_tracking_keys=len(local),full40_numeric_union_max=max(r['numeric_track_union'] for r in seqfull),full40_source_union_max=max(r['source_key_union'] for r in seqfull),concurrent_positive_max=max(r['concurrent_positive_max'] for r in seqfull),annotations_sha256=sha(path)))
 allseq[s]=by
for entry in plan['entries']:
 for pi,pair in enumerate(entry['pairs']):
  s=pair['sequence'];by=allseq[s];cy,cx=pair['crop_yx']
  for wi,st in enumerate(pair['starts']):
   eligible=[a for t in range(st,st+40) for a in by[t] if pos(a) and core(a,cy,cx)]
   intersect=[a for t in range(st,st+40) for a in by[t] if valid(a) and intersects(a,cy,cx)]
   excluded=[a for a in intersect if not(pos(a) and core(a,cy,cx))]
   crop.append(dict(step=entry['step'],pair=pi,window=wi,sequence=s,start=st,crop_y=cy,crop_x=cx,numeric_track_union=len({a['track_id'] for a in eligible}),source_key_union=len({original(a) for a in eligible}),positive_box_frames=len(eligible),concurrent_positive_max=max(sum(pos(a) and core(a,cy,cx) for a in by[t]) for t in range(st,st+40)),intersecting_box_frames=len(intersect),excluded_intersecting_box_frames=len(excluded),partly_outside_core_positive_box_frames=sum(pos(a) and not core(a,cy,cx) for a in intersect),ignored_or_crowd_intersecting_box_frames=sum(not pos(a) for a in intersect)))
summary=dict(scope='Only 8 frozen training clips; no dev annotations, pixels, predictions, or evaluation results read',selection_sha256=sha(P/'DATA_SELECTION.json'),plan_sha256=sha(planpath),plan_steps=plan['steps'],exposures=len(crop),source_gzip_sha256=sha(sourcepath),source_uncompressed_sha256=hashlib.sha256(sourcebytes).hexdigest(),source_annotation_exact_matches=sum(r['annotation_count'] for r in seqrows),source_image_exact_matches=sum(r['annotation_frames'] for r in seqrows),source_mismatches=source_mismatch,frame_errors=frame_errors,missing_fields=dict(missing_fields),annotation_field_counts=dict(allkeys),duplicate_keys_sameframe=duplicates,numeric_key_reused_across_category_tracking_keys=key_reuse,ambiguous_source_alias_groups=len(alias),alias_numeric_id_adjacent_switches=len(transitions),alias_switch_center_distance=stat([r['center_distance'] for r in transitions]),training_sequences=seqrows,full40_numeric_union=stat([r['numeric_track_union'] for r in winrows]),full40_source_union=stat([r['source_key_union'] for r in winrows]),crop_numeric_union=stat([r['numeric_track_union'] for r in crop]),crop_source_union=stat([r['source_key_union'] for r in crop]),crop_concurrent=stat([r['concurrent_positive_max'] for r in crop]),empty_crop_windows=sum(r['positive_box_frames']==0 for r in crop),crop_positive_box_frame_exposures=sum(r['positive_box_frames'] for r in crop),numeric_tracks=sum(r['numeric_track_ids'] for r in seqrows),source_category_tracking_keys=sum(r['source_category_tracking_keys'] for r in seqrows),query_count_recommendation=dict(global_K=128,reason='Global 40-frame numeric ID union max91, concurrent max78; 128 predetermined grid capacity above training max. Do not derive global capacity from crop max19.',crop_only_K=32,crop_only_caveat='Only if per-tile independent query topology is explicitly intended; not equivalent to global query decoder.'),limitations=['Dataset-provided identity, not physical identity verified at every frame. Numeric alias switches fragment original labels. No silently merging numeric IDs.', 'No visibility, occlusion, absent-state fields occur. Gaps and excluded annotations must be unknown, not absent.', 'Integer uniqueness and smooth box coordinates are integrity checks, not proof of semantic identity.', 'K is hypothesis capacity, not a cap on connected components: each query spatial map can have multiple components.'])
save('TRAIN_ID_DENSITY_SUMMARY.json',summary)
summary['canonical_track_internal_gaps']=sum(r['continuous_runs']-1 for r in canonicalrows)
summary['canonical_adjacent_motion']=stat([r['center_distance'] for r in allmotion])
summary['canonical_adjacent_motion_per_box_diagonal']=stat([r['distance_per_larger_box_diagonal'] for r in allmotion])
summary['query_count_recommendation']={'global_K':100,'grid':[10,10],'reason':'Parent selected 10x10 global queries before dev; this exceeds numeric 40-window union max91, author-category-key union max79, concurrent max78. 128 was an earlier equally viable engineering suggestion, not tested. No developed results consulted.','crop_only_K':32,'crop_only_caveat':'A per-tile topology would be a different model; not used. Existing crop core maximum is19 but full-frame K is not derived from it.'}
summary['supervision_identity_recommendation']='Use explicit (sequence, category_id, str(tracking_id)) author-field key, retaining numeric track_id for provenance and evaluation. This is a documented annotation-ID choice, not inferred physical identity reconstruction. Missing/ignored/clipped observation states are unknown.'
save('TRAIN_ID_DENSITY_SUMMARY.json',summary)
table('TRAIN_SEQUENCE_IDENTITY.csv',seqrows);table('TRAIN_FULL40_DENSITY.csv',winrows);table('TRAIN_CROP_DENSITY.csv',crop);table('TRAIN_NUMERIC_TRACK_RUNS.csv',trackrows);table('SOURCE_ID_ALIAS_GROUPS.csv',alias);table('SOURCE_ID_ALIAS_ADJACENT_SWITCHES.csv',transitions)
table('TRAIN_CANONICAL_IDENTITY.csv',canonicalrows);table('TRAIN_CANONICAL_ADJACENT_MOTION.csv',allmotion)
print(json.dumps({k:summary[k] for k in ['source_annotation_exact_matches','source_image_exact_matches','source_uncompressed_sha256','numeric_tracks','source_category_tracking_keys','ambiguous_source_alias_groups','alias_numeric_id_adjacent_switches','alias_switch_center_distance','full40_numeric_union','full40_source_union','crop_numeric_union','crop_source_union','crop_concurrent','empty_crop_windows','crop_positive_box_frame_exposures']},indent=2))
print(json.dumps({k:summary[k] for k in ['source_mismatches','frame_errors','missing_fields','duplicate_keys_sameframe','numeric_key_reused_across_category_tracking_keys']},indent=2))
assert not source_mismatch and not frame_errors and not missing_fields and not duplicates and not key_reuse
