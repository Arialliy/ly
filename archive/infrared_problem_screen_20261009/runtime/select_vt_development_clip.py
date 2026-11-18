from pathlib import Path
import json,collections,hashlib,gzip
r=Path(__file__).resolve().parents[1];d=json.loads(gzip.decompress((r/'runtime/VT_TRAIN_01_ANNOTATIONS.json.gz').read_bytes()))
known=set((r/'official_deeppro/datasets/RGB-T/train.txt').read_text().split())|set((r/'official_deeppro/datasets/RGB-T/test.txt').read_text().split())
images={x['id']:x for x in d['images']};tracks=collections.defaultdict(list)
for a in d['annotations']:
 im=images[a['image_id']];seq=im['file_name'].split('/')[0];b=a['bbox']
 if seq in known or a.get('ignore',0) or a.get('iscrowd',0) or len(b)!=4 or not (0<b[2]<=12 and 0<b[3]<=12):continue
 if 'track_id' not in a:continue
 frame=int(Path(im['file_name']).stem);tracks[(seq,str(a['track_id']))].append((frame,im['id'],a['id']))
candidates=[]
for (seq,track),frames in sorted(tracks.items()):
 frames=sorted(set(frames));run=[]
 for item in frames:
  if run and item[0]!=run[-1][0]+1:run=[]
  run.append(item)
  if len(run)==160:
   candidates.append({'sequence':seq,'track_id':track,'start':run[0][0],'end':run[-1][0],'frame_count':160});break
candidates.sort(key=lambda x:(x['sequence'],int(x['track_id']),x['start']))
print('qualified_candidates',len(candidates),'eligible_sequences',len(set(x['sequence'] for x in candidates)),'first5',candidates[:5],flush=True)
report={'rule':'Use VT train2017 scenes absent from both official DeepPro RGB-T train/test manifests; require one original track_id with ignore=iscrowd=0, bbox w/h in (0,12], in 160 consecutive numeric raw filenames; choose lexicographic sequence, numeric track_id, earliest start. Retain ALL annotations in selected frames.','candidates':candidates,'limits':['Not an official validation split.','Not listed in DeepPro train/test does not establish independent scene identity.','Annotated presence and image-coordinate movement do not prove physical stop/restart.','COCO segmentation polygons here are rectangles around boxes, not verified pixel masks.']}
(r/'VT_SMALL_TARGET_PRESCREEN.json').write_text(json.dumps(report,indent=2))
if not candidates:raise SystemExit('No qualifying original-ID clip')
c=candidates[0];ims=sorted([im for im in d['images'] if im['file_name'].split('/')[0]==c['sequence'] and c['start']<=int(Path(im['file_name']).stem)<=c['end']],key=lambda im:int(Path(im['file_name']).stem));ids={x['id'] for x in ims};anns=[a for a in d['annotations'] if a['image_id'] in ids]
if len(ims)!=160:raise RuntimeError('Incomplete annotation image sequence')
out=r/'runtime/public_subset_vt';out.mkdir(exist_ok=True);subset={k:d[k] for k in ['categories']};subset.update({'images':ims,'annotations':anns,'selection':c,'selection_rule':report['rule'],'limitations':report['limits']})
for k in ['videos','tracks','num_instances','num_track']:
 if k in d:print('extra_key',k,'type',type(d[k]).__name__,'sample',str(d[k])[:250],flush=True)
# Preserve relevant original video/track metadata without inventing identities.
if 'videos' in d:subset['videos']=[v for v in d['videos'] if v.get('id') in {x.get('video_id') for x in ims}]
if 'tracks' in d:subset['tracks_source_metadata']=d['tracks'] if not isinstance(d['tracks'],list) else [t for t in d['tracks'] if not isinstance(t,dict) or t.get('name')==c['sequence']+'/01']
(out/'annotations.json').write_text(json.dumps(subset,ensure_ascii=False,indent=2));(out/'FRAME_MANIFEST.json').write_text(json.dumps({'subset':'extra development subset of VT-Tiny-MOT train2017','selection':c,'frames':[{'index':i,'original':im,'archive_member':'VT-Tiny-MOT/train2017/'+im['file_name'],'annotation_count':sum(a['image_id']==im['id'] for a in anns)} for i,im in enumerate(ims)]},indent=2))
print('CHOSEN',c,'total_annotations',len(anns),'bbox_categories',d['categories'],flush=True)
