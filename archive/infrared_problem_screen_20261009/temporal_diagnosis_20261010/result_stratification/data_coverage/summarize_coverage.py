from pathlib import Path
import gzip,json,collections,csv,hashlib
D=Path(__file__).resolve().parents[3]
Q=D/'temporal_diagnosis_20261010'
O=Path(__file__).resolve().parent
source=D/'runtime/VT_TRAIN_01_ANNOTATIONS.json.gz'
data=json.loads(gzip.decompress(source.read_bytes()))
images={i['id']:i for i in data['images']}
cats={x['id']:x['name'] for x in data['categories']}
train=set((D/'official_deeppro/datasets/RGB-T/train.txt').read_text().split())
counts=collections.Counter((a['image_id'],a['track_id']) for a in data['annotations'])
groups=collections.defaultdict(list)
duplicates=0
for a in data['annotations']:
 i=images[a['image_id']];seq=i['file_name'].split('/')[0]
 if seq not in train:continue
 if counts[a['image_id'],a['track_id']]>1:duplicates+=1;continue
 x,y,w,h=a['bbox']
 if a.get('ignore',0) or w<=0 or h<=0 or min(i['width'],x+w)<=max(0,x) or min(i['height'],y+h)<=max(0,y):continue
 b=dict(a,sequence=seq)
 groups['official45_sequence_annotation_metadata'].append(b)
paths=[D/'real_motion_stage2/selected_raw/annotations.json']+list((Q/'additional_clips').glob('DJI*/annotations.json'))
for p in paths:
 j=json.loads(p.read_text()); name='DJI_0028_2' if 'selected_raw' in str(p) else p.parent.name.split('__')[0]
 for a in j['annotations']:groups[name].append(dict(a,sequence=name))
rows=[]
for cohort, anns in groups.items():
 for cid,name in cats.items():
  aa=[a for a in anns if a['category_id']==cid]
  rows.append({'cohort':cohort,'category':name,'gt_frames':len(aa),'tracks':len({(a['sequence'],a['track_id']) for a in aa}),'sequences':len({a['sequence'] for a in aa}),'maxside_le8':sum(max(a['bbox'][2:])<=8 for a in aa),'maxside_le12':sum(max(a['bbox'][2:])<=12 for a in aa),'maxside_le16':sum(max(a['bbox'][2:])<=16 for a in aa)})
with (O/'category_scale_counts.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
summary={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'official_train_sequence_count':len(train),'matched_nonempty_sequences':len({a['sequence'] for a in groups['official45_sequence_annotation_metadata']}),'duplicate_rows_removed':duplicates,'rows':rows,'limitation':'Underlying VT annotation metadata in sequences matching official training list, not verified Gaussian masks or actual checkpoint exposure. No raw training contrast/visibility/event statistics, no evidence of label completeness or scene-independent splits.'}
(O/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
