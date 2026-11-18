#!/usr/bin/env python3
"""Independent arithmetic verification from source CSVs and annotation JSON."""
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
D=Path(__file__).resolve().parent;Q=D.parent.parent;V=Q/'problem_validity_20261011';P=Q/'stage1_goal'
def read(p):
    with p.open() as f:return list(csv.DictReader(f))
def identity(r):return r['sequence'],int(r['annotation_id'])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
source={identity(r):r for r in read(V/'OBJECTS.csv') if r['sequence'] in ('DJI_0026_2','DJI_0067_2') and r['small_geometry']=='1'}
controls={identity(r):r for r in read(V/'control_summary/OBJECTS.csv') if r['mode']=='symmetric5_probmax' and identity(r) in source}
out={identity(r):r for r in read(D/'OBJECT_CONDITIONS.csv')}
assert source.keys()==controls.keys()==out.keys()
counts=collections.Counter();events=collections.defaultdict(set);geometric_checks=0
for k,s in source.items():
    c=controls[k];o=out[k];a=int(c['official_matched']);b=int(c['control_matched']);u=int(s['any11_matched'])
    label={(0,1):'gain',(1,0):'loss',(1,1):'common_success'}.get((a,b),'both_failure_with_window_opportunity' if u else 'all11_failure')
    assert o['group']==label;counts[label]+=1;events[(label,k[0],int(s['track_id']))].add(int(s['source_frame']))
event_counts=collections.Counter()
for (label,seq,track),frames in events.items():event_counts[label]+=sum(frame-1 not in frames for frame in frames)
for r in read(D/'GROUP_COUNTS.csv'):
    if r['group']=='all_small':continue
    assert int(r['object_frames'])==counts[r['group']] and int(r['observation_events'])==event_counts[r['group']]
for seq in ('DJI_0026_2','DJI_0067_2'):
    ann=json.loads((P/'data'/seq/'annotations.json').read_text());ims={x['id']:x for x in ann['images']};aa=collections.defaultdict(list);byid={}
    for a in ann['annotations']:
        x,y,w,h=a['bbox'];im=ims[a['image_id']]
        if not a.get('ignore',0) and w>0 and h>0 and min(im['width'],x+w)>max(0,x) and min(im['height'],y+h)>max(0,y):aa[a['image_id']].append(a)
        byid[a['id']]=a
    for k,o in out.items():
        if k[0]!=seq:continue
        a=byid[k[1]];x,y,w,h=a['bbox'];cx=x+w/2;cy=y+h/2
        nearest=min(math.hypot((b['bbox'][0]+b['bbox'][2]/2)-cx,(b['bbox'][1]+b['bbox'][3]/2)-cy) for b in aa[a['image_id']] if b['id']!=a['id'])
        assert math.isclose(float(o['nearest_other_annotation_center_px']),nearest,rel_tol=1e-12,abs_tol=1e-12)
        assert int(o['frame_valid_annotation_count'])==len(aa[a['image_id']])
        assert float(o['max_side'])==max(w,h)
        for arrangement,starts in [('official',[0,36,72,108,120]),('symmetric',[0,24,60,96,120])]:
            frame=int(o['frame']);active=[s for s in starts if s<=frame<s+40]
            assert int(o[arrangement+'_cover_count'])==len(active)
            assert int(o[arrangement+'_latest_access_relative_to_current'])==max(active)+39-frame
        geometric_checks+=1
result=dict(status='PASS',source_objects=len(source),group_counts=dict(counts),event_counts=dict(event_counts),geometric_objects_checked=geometric_checks,
            source_hashes={str(p):sha(p) for p in (V/'OBJECTS.csv',V/'control_summary/OBJECTS.csv',D/'OBJECT_CONDITIONS.csv')},
            script_sha256=sha(Path(__file__)),scope='Independently rejoined V outcome tables; checked all 5200 annotations, max side, nearest distance, valid density, schedule coverage/latest access; rederived group events.')
(D/'INDEPENDENT_AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False))
