"""Independent validation of chosen frozen probe regions, no model/image reads."""
import json,hashlib,math
from pathlib import Path
from collections import defaultdict
B=Path(__file__).resolve().parent;Q=B.parent
P=B/'geometry/SAMPLE_PROPOSAL.json';D=json.loads(P.read_text());F=json.loads((B/'B_FREEZE.json').read_text())
checks=[]
def ck(n,v):
 checks.append(dict(name=n,passed=bool(v)))
 if not v:raise AssertionError(n)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def valid(a):
 b=a.get('bbox',[]);return len(b)==4 and all(math.isfinite(float(v)) for v in b) and b[2]>0 and b[3]>0
def pos(a):return valid(a) and not a.get('ignore',0) and not a.get('iscrowd',0)
def pix(r):return {(x,y) for y in range(r[1],r[3]) for x in range(r[0],r[2])}
def crop(a):
 x,y,w,h=map(float,a['bbox']);return [math.ceil(x-.5),math.ceil(y-.5),math.ceil(x+w-.5),math.ceil(y+h-.5)]
def over(a,b):
 x,y,w,h=a['bbox'];u,v,s,t=b['bbox'];return min(x+w,u+s)>max(x,u) and min(y+h,v+t)>max(y,v)
def contains(a,x,y,guard=0):
 bx,by,w,h=map(float,a['bbox']);return bx-guard<=x+.5<bx+w+guard and by-guard<=y+.5<by+h+guard
def bg(r,aa):return 0<=r[0]<r[2]<=640 and 0<=r[1]<r[3]<=512 and not any(contains(a,x,y,2) for x,y in pix(r) for a in aa if valid(a))
ck('frozen_sample_hash',sha(P)==F['samples_sha256']==F['sample_proposal_sha256'])
ck('frozen_geometry_source',sha(B/'code/probe_geometry.py')==F['code_sha256']['code/probe_geometry.py'])
planpath=Q/'stage1_goal/pilot/TRAIN_PLAN.json';train=set(json.loads(planpath.read_text())['train_sequences']);dev={'DJI_0026_2','DJI_0067_2'}
ck('source_isolation',len(train)==8 and not train&dev);ck('O_radius_guard',D['starts']==[0,36,72,108,120] and D['radius']==D['background_margin']==2)
cache={};sourcehash={str(P):sha(P),str(B/'B_FREEZE.json'):sha(B/'B_FREEZE.json'),str(planpath):sha(planpath)}
for seq in {r['sequence'] for r in D['samples']}:
 p=Q/'stage1_goal/data'/seq/'annotations.json';raw=json.loads(p.read_text());by=defaultdict(list)
 for a in raw['annotations']:
  if valid(a):by[a['image_id']].append(a)
 cache[seq]=(raw['images'],[by[i['id']] for i in raw['images']]);sourcehash[str(p)]=sha(p)
 ck(seq+'annotation_hash',D['source_sha256'][str(p)]==sourcehash[str(p)])
seen=set();Ds={}
for k,r in enumerate(D['samples']):
 tag=str(k)+':'+r['pair_id'];seq=r['sequence'];t=r['frame'];s=r['reference_frame'];ims,aa=cache[seq]
 ck(tag+'unique',r['pair_id'] not in seen);seen.add(r['pair_id'])
 ck(tag+'split',seq in train|dev and r['split']==('train' if seq in train else 'dev'))
 starts=[v for v in D['starts'] if v<=t<v+40];start=min(starts,key=lambda v:(abs(t-v-19.5),v))
 ck(tag+'window',r['start']==start and abs(t-s)==1 and start<=s<start+40 and r['reference_offset']==s-t)
 ck(tag+'sourceframe',int(Path(ims[s]['file_name']).stem)-int(Path(ims[t]['file_name']).stem)==s-t and r['source_frame']==int(Path(ims[t]['file_name']).stem))
 query=[a for a in aa[t] if a['id']==r['annotation_id']];refs=[a for a in aa[s] if a.get('track_id')==r['track_id'] and pos(a)]
 ck(tag+'identity',len(query)==len(refs)==1 and pos(query[0]) and query[0]['track_id']==r['track_id'])
 a,z=query[0],refs[0];x,y,w,h=map(float,a['bbox']);cx,cy=math.floor(x+w/2),math.floor(y+h/2)
 ck(tag+'strict_query',max(w,h)<=16 and 0<=x<x+w<=640 and 0<=y<y+h<=512 and not any(over(a,b) for b in aa[t] if b['id']!=a['id']))
 ck(tag+'reference_nonoverlap',not any(over(z,b) for b in aa[s] if b['id']!=z['id']))
 hood={(xx,yy) for yy in range(cy-2,cy+3) for xx in range(cx-2,cx+3)};q,ref,wrong=(pix(r[n]) for n in ['q_rect_xyxy','r1_rect_xyxy','r2_rect_xyxy'])
 ck(tag+'Qexact',bool(q) and q=={v for v in hood if contains(a,*v)})
 ck(tag+'Rexact',bool(ref) and ref=={v for v in hood if contains(z,*v)})
 fx,fy=r['p2_reflection_xy'];reflected={(2*cx-u if fx else u,2*cy-v if fy else v) for u,v in ref}
 ck(tag+'Breflection',wrong==reflected and wrong!=ref and wrong<=hood)
 ck(tag+'Bsize_distance',len(wrong)==len(ref) and sorted((u-cx)**2+(v-cy)**2 for u,v in wrong)==sorted((u-cx)**2+(v-cy)**2 for u,v in ref))
 ck(tag+'BGTplus2',bg(r['r2_rect_xyxy'],aa[s]))
 key=(seq,r['track_id'])
 if key not in Ds:
  sides=[]
  for frame in aa:
   for g in frame:
    if g.get('track_id')!=r['track_id'] or not pos(g):continue
    gx,gy,gw,gh=map(float,g['bbox'])
    if max(gw,gh)>16 or not(0<=gx<gx+gw<=640 and 0<=gy<gy+gh<=512):continue
    if any(over(g,b) for b in frame if b['id']!=g['id']):continue
    gr=crop(g);sides.append(max(gr[2]-gr[0],gr[3]-gr[1]))
  Ds[key]=4*max(sides)
 d=Ds[key];options=[(dx*m,dy*m) for m in [1,2] for dx,dy in [(d,0),(-d,0),(0,d),(0,-d),(d,d),(d,-d),(-d,d),(-d,-d)]]
 ck(tag+'negative_count',0<len(r['bg_candidates'])<=16);last=-1
 for i,n in enumerate(r['bg_candidates']):
  dx,dy=n['offset_xy'];idx=options.index((dx,dy));ck(tag+f'N{i}ordered',idx>last);last=idx
  for name,frame in [('q_rect_xyxy',t),('r1_rect_xyxy',s),('r2_rect_xyxy',s)]:
   cr=r[name];translated=[cr[0]+dx,cr[1]+dy,cr[2]+dx,cr[3]+dy]
   ck(tag+f'N{i}{name}translation',n[name]==translated);ck(tag+f'N{i}{name}background',bg(n[name],aa[frame]))
ck('edge_counts',len(D['samples'])==64 and sum(r['split']=='train' for r in D['samples'])==45 and sum(r['split']=='dev' for r in D['samples'])==19)
out=dict(status='PASS',checks_passed=len(checks),checks=checks,source_sha256=sourcehash,script_sha256=sha(Path(__file__)),frozen_samples_sha256=sha(P),previous_proposal_review_retained=str(B/'PROBE_IMPLEMENTATION_REVIEW.json'),scope='64selected frozenedges only; no image/model/feature/probe output reads',limitations=['P1 oracle labels are not deployment input.','GT+2 outside-region features may still contain target information through layer1 receptive field.','Only18devtargetframes; cannot conclude about entire4342failurecohort.'])
(B/'PROBE_IMPLEMENTATION_REVIEW_FROZEN.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n');print(json.dumps({'status':'PASS','checks':len(checks),'samples_sha256':sha(P)}))
