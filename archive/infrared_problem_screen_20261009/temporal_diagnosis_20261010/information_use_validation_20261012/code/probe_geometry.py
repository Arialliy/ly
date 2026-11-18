"""Geometry-only common-support audit. No image scores, features, or probe fitting."""
import collections,csv,hashlib,json,math
from pathlib import Path
BASE=Path(__file__).resolve().parents[1];Q=BASE.parent;DATA=Q/'stage1_goal/data'
STARTS=[0,36,72,108,120];R=2;MARGIN=2

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rect(box):
 x,y,w,h=map(float,box);return [math.ceil(x-.5),math.ceil(y-.5),math.ceil(x+w-.5),math.ceil(y+h-.5)]
def positive(a):return not a.get('ignore',0) and not a.get('iscrowd',0) and valid(a)
def valid(a):return len(a.get('bbox',[]))==4 and all(math.isfinite(float(v)) for v in a['bbox']) and a['bbox'][2]>0 and a['bbox'][3]>0
def overlap(a,b):return max(a[0],b[0])<min(a[2],b[2]) and max(a[1],b[1])<min(a[3],b[3])
def f_rect(a):
 x,y,w,h=a['bbox'];return[x,y,x+w,y+h]
def in_image(b):return 0<=b[0]<b[2]<=640 and 0<=b[1]<b[3]<=512
def intersect(a,b):return[max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3])]
def nonempty(b):return b[0]<b[2] and b[1]<b[3]
def shift(b,dx,dy):return[b[0]+dx,b[1]+dy,b[2]+dx,b[3]+dy]
def background(b,exclusion):return in_image(b) and not any(overlap(b,e) for e in exclusion)
def reflected(b,cx,cy,fx,fy):
 return [2*cx-b[2]+1 if fx else b[0],2*cy-b[3]+1 if fy else b[1],2*cx-b[0]+1 if fx else b[2],2*cy-b[1]+1 if fy else b[3]]
def writecsv(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
 planpath=Q/'stage1_goal/pilot/TRAIN_PLAN.json';plan=json.loads(planpath.read_text());train=plan['train_sequences'];dev=['DJI_0026_2','DJI_0067_2']
 cp=Q/'low_response_information_20261011/COHORTS.csv';cohorts={(r['sequence'],int(r['annotation_id'])):r for r in csv.DictReader(cp.open())}
 samples=[];counts=[];inputs={str(planpath):sha(planpath),str(cp):sha(cp)}
 for seq in train+dev:
  path=DATA/seq/'annotations.json';inputs[str(path)]=sha(path);a=json.loads(path.read_text());images=a['images'];index={im['id']:i for i,im in enumerate(images)};anns=[[]for _ in images]
  for ann in a['annotations']:
   if valid(ann):anns[index[ann['image_id']]].append(ann)
  strict=[];excludes=[];track_sides=collections.defaultdict(list)
  for frame,rr in enumerate(anns):
   exclusions=[];ss=[]
   for an in rr:
    b=rect(an['bbox']);exclusions.append([max(0,b[0]-MARGIN),max(0,b[1]-MARGIN),min(640,b[2]+MARGIN),min(512,b[3]+MARGIN)])
    if not positive(an):continue
    x,y,w,h=an['bbox']
    if max(w,h)>16 or x<0 or y<0 or x+w>640 or y+h>512 or not nonempty(b):continue
    if any(overlap(f_rect(an),f_rect(other)) for other in rr if other['id']!=an['id']):continue
    ss.append(an);track_sides[int(an['track_id'])].append(max(b[2]-b[0],b[3]-b[1]))
   strict.append(ss);excludes.append(exclusions)
  stage=collections.Counter();seen=collections.defaultdict(set)
  for t,rr in enumerate(strict):
   for an in rr:
    aid=an['id'];tid=int(an['track_id']);stage['strict_target_frames']+=1
    cx=math.floor(an['bbox'][0]+an['bbox'][2]/2);cy=math.floor(an['bbox'][1]+an['bbox'][3]/2);hood=[cx-R,cy-R,cx+R+1,cy+R+1];query=intersect(rect(an['bbox']),hood)
    start=min((s for s in STARTS if s<=t<s+40),key=lambda s:(abs((t-s)-19.5),s))
    for direction in [-1,1]:
     s=t+direction
     if not(start<=s<start+40 and 0<=s<160):continue
     if int(Path(images[s]['file_name']).stem)-int(Path(images[t]['file_name']).stem)!=direction:continue
     stage['legal_reference_edges']+=1
     same=[other for other in anns[s] if int(other.get('track_id',-1))==tid and positive(other)]
     if len(same)!=1:continue
     other=same[0]
     if any(overlap(f_rect(other),f_rect(z)) for z in anns[s] if z['id']!=other['id']):continue
     stage['unique_nonoverlap_reference_edges']+=1
     reference=intersect(rect(other['bbox']),hood)
     if not nonempty(reference):continue
     stage['P1_nonempty_edges']+=1;seen['P1_target_frames'].add(aid)
     candidates=[]
     for fx,fy in [(True,False),(False,True),(True,True)]:
      wrong=reflected(reference,cx,cy,fx,fy)
      if wrong in [p[2] for p in candidates] or wrong==reference:continue
      if background(wrong,excludes[s]):candidates.append([fx,fy,wrong])
     if not candidates:continue
     stage['P2_distance_size_matched_edges']+=1;seen['P2_target_frames'].add(aid)
     D=4*max(track_sides[tid]);offsets=[(dx*k,dy*k) for k in [1,2] for dx,dy in [(D,0),(-D,0),(0,D),(0,-D),(D,D),(D,-D),(-D,D),(-D,-D)]]
     for fx,fy,wrong in candidates:
      negatives=[]
      for dx,dy in offsets:
       nq=shift(query,dx,dy);nr=shift(reference,dx,dy);nw=shift(wrong,dx,dy)
       if background(nq,excludes[t]) and background(nr,excludes[s]) and background(nw,excludes[s]):
        negatives.append(dict(offset_xy=[dx,dy],q_rect_xyxy=nq,r1_rect_xyxy=nr,r2_rect_xyxy=nw))
      if not negatives:continue
      stage['paired_common_support_edges']+=1;seen['paired_target_frames'].add(aid);seen['paired_tracks'].add(tid)
      cc=cohorts.get((seq,aid),{})
      row=dict(pair_id=f'{seq}:{aid}:{direction:+d}',split='train' if seq in train else 'dev',sequence=seq,frame=t,source_frame=int(Path(images[t]['file_name']).stem),reference_frame=s,reference_offset=direction,start=start,annotation_id=aid,track_id=tid,q_rect_xyxy=query,r1_rect_xyxy=reference,r2_rect_xyxy=wrong,p2_reflection_xy=[fx,fy],bg_candidates=negatives,
       cohort=cc.get('cohort','train'),low_all_05=int(cc.get('low_all_05',0)),extreme_all_001=int(cc.get('extreme_all_001',0)),both_fixed_success=int(cc.get('both_fixed_success',0)))
      samples.append(row);break
  row=dict(sequence=seq,split='train' if seq in train else 'dev',**{k:stage[k] for k in ['strict_target_frames','legal_reference_edges','unique_nonoverlap_reference_edges','P1_nonempty_edges','P2_distance_size_matched_edges','paired_common_support_edges']},**{k:len(seen[k]) for k in ['P1_target_frames','P2_target_frames','paired_target_frames','paired_tracks']})
  counts.append(row);print(json.dumps(row),flush=True)
 out=BASE/'geometry';out.mkdir(exist_ok=True);writecsv(out/'COVERAGE.csv',counts)
 result=dict(status='GEOMETRY_FIXED_LIMITED_PROBE',radius=R,background_margin=MARGIN,starts=STARTS,source_sha256=inputs,samples=samples,counts=counts,
  feature_selection='No raw images, model scores, features or dev probe predictions accessed. Rectangular means; exact reflected candidate support preserves size and per-point query distance.',
  scope='Fixed common-support limited descriptive probe; root freezes fit authorization. No radius/guard relaxation. One observation per eligible target-frame/reference direction; both classes share exact geometry.')
 (out/'SAMPLE_PROPOSAL.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print('TOTAL',len(samples),'TRAIN',sum(s['split']=='train' for s in samples),'DEV',sum(s['split']=='dev' for s in samples),'DEVLOW',sum(s['split']=='dev' and s['low_all_05'] for s in samples))
if __name__=='__main__':main()
