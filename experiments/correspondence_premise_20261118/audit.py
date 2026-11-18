import csv, hashlib, json, math, time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
TRAIN = ['DJI_0030_9','DJI_0043_2','DJI_0057_1','DJI_0113_3','DJI_0115_2','DJI_0217_2','DJI_0309_1','DJI_0323_2']
DEV = ['DJI_0026_2','DJI_0067_2']
FORBIDDEN = {'DJI_0051_3','DJI_0075_3','DJI_0139_2','DJI_0343_3'}
RAD, STEP, PENALTY, T = 24, 6, .01, 8
GRID = 2*RAD+1
NEG = -1e8

def dump(name, value):
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def eligible(a):
    x,y,w,h=a['bbox']
    return not a.get('ignore',0) and not a.get('iscrowd',0) and 0<w*h<=144 and max(w,h)<=16

def center(a):
    x,y,w,h=a['bbox'];return [int(round(x+w/2)),int(round(y+h/2))]

def inside(p, bb, tol):
    x,y,w,h=bb
    return x-tol <= p[0] <= x+w+tol and y-tol <= p[1] <= y+h+tol

def gtmask(bb, anchor, tol):
    ys,xs=np.mgrid[-RAD:RAD+1,-RAD:RAD+1]
    x,y,w,h=bb
    return ((xs+anchor[0]>=x-tol)&(xs+anchor[0]<=x+w+tol)&
            (ys+anchor[1]>=y-tol)&(ys+anchor[1]<=y+h+tol))

def transition(a):
    """Exact box-bounded quadratic max convolution, with predecessor indices."""
    h,w=a.shape; xgrid=np.broadcast_to(np.arange(w),(h,w))
    ygrid=np.broadcast_to(np.arange(h)[:,None],(h,w))
    horizontal=np.full_like(a,NEG); argx=xgrid.copy()
    for dx in range(-STEP,STEP+1):
        x0=max(0,-dx);x1=min(w,w-dx)
        if x0>=x1:continue
        src=a[:,x0+dx:x1+dx]-PENALTY*dx*dx
        dest=horizontal[:,x0:x1]; take=src>dest
        dest[take]=src[take]
        argx[:,x0:x1][take]=xgrid[:,x0+dx:x1+dx][take]
    result=np.full_like(a,NEG); py=ygrid.copy();px=xgrid.copy()
    for dy in range(-STEP,STEP+1):
        y0=max(0,-dy);y1=min(h,h-dy)
        if y0>=y1:continue
        src=horizontal[y0+dy:y1+dy]-PENALTY*dy*dy
        dest=result[y0:y1];take=src>dest;dest[take]=src[take]
        py[y0:y1][take]=ygrid[y0+dy:y1+dy][take]
        px[y0:y1][take]=argx[y0+dy:y1+dy][take]
    return result, np.stack([py,px],axis=-1).astype(np.int16)

def forward(unary):
    scores=[unary[0].copy()];ptr=[None]
    for z in unary[1:]:
        m,p=transition(scores[-1]);scores.append(z+m);ptr.append(p)
    return scores,ptr

def decode(ptr, end):
    positions=[tuple(end)]
    for t in range(len(ptr)-1,0,-1):positions.append(tuple(ptr[t][positions[-1]]))
    return list(reversed(positions))

def peaks(a, sep, count=6):
    work=a.copy();ans=[];yy,xx=np.mgrid[:a.shape[0],:a.shape[1]]
    for _ in range(count):
        y,x=np.unravel_index(np.argmax(work),work.shape)
        if work[y,x]<NEG/2:break
        ans.append((int(y),int(x)))
        work[(xx-x)**2+(yy-y)**2<=sep**2]=NEG
    return ans

def descriptors(im):
    f=im.astype(np.float32)
    hp=f-cv2.GaussianBlur(f,(0,0),3)
    gx=cv2.Sobel(f,cv2.CV_32F,1,0,ksize=3)*.25
    gy=cv2.Sobel(f,cv2.CV_32F,0,1,ksize=3)*.25
    return np.stack([hp,gx,gy],-1)

def scoremaps(frames, desc, anchor, patch, rich):
    x,y=anchor;r=patch//2;src=desc if rich else frames
    templ=np.ascontiguousarray(src[-1][y-r:y+r+1,x-r:x+r+1])
    ans=[]
    for im in src[:-1]:
        crop=np.ascontiguousarray(im[y-RAD-r:y+RAD+r+1,x-RAD-r:x+RAD+r+1])
        a=cv2.matchTemplate(crop,templ,cv2.TM_CCOEFF_NORMED)
        assert a.shape==(GRID,GRID) and np.isfinite(a).all()
        ans.append(a.astype(np.float32))
    terminal=np.full((GRID,GRID),NEG,np.float32);terminal[RAD,RAD]=0
    ans.append(terminal)
    return ans

def absolute(path, anchor):
    return [[int(x-RAD+anchor[0]),int(y-RAD+anchor[1])] for y,x in path]

def contrast(im, a):
    x,y,w,h=a['bbox'];x0=max(0,int(math.floor(x)));x1=min(im.shape[1],int(math.ceil(x+w)))
    y0=max(0,int(math.floor(y)));y1=min(im.shape[0],int(math.ceil(y+h)))
    r=5;crop=im[max(0,y0-r):min(im.shape[0],y1+r),max(0,x0-r):min(im.shape[1],x1+r)].astype(float)
    mask=np.ones(crop.shape,bool)
    mask[y0-max(0,y0-r):y1-max(0,y0-r),x0-max(0,x0-r):x1-max(0,x0-r)]=False
    ring=crop[mask];signal=im[y0:y1,x0:x1].astype(float)
    if signal.size==0 or ring.size==0:return 0.,0.
    return float((signal.mean()-ring.mean())/max(1.,ring.std())),float(ring.std())

def object_type(p, anns, target_id, tol=2):
    if any(a['track_id']==target_id and inside(p,a['bbox'],tol) for a in anns):return 'same_target'
    if any(a['track_id']!=target_id and inside(p,a['bbox'],tol) for a in anns):return 'other_annotated_object'
    return 'unannotated_region'

def selfcheck():
    rng=np.random.default_rng(46);a=rng.normal(size=(5,6)).astype(np.float32)
    val,ptr=transition(a)
    for y in range(5):
        for x in range(6):
            b=max(float(a[j,i])-PENALTY*((j-y)**2+(i-x)**2) for j in range(5) for i in range(6) if abs(j-y)<=STEP and abs(i-x)<=STEP)
            assert abs(b-val[y,x])<2e-6
            j,i=ptr[y,x]; assert abs(float(a[j,i])-PENALTY*((j-y)**2+(i-x)**2)-val[y,x])<2e-6
    us=[rng.normal(size=(3,3)).astype(np.float32) for _ in range(3)]
    fs,ps=forward(us);end=(1,1);path=decode(ps,end)
    score=sum(float(us[t][p]) for t,p in enumerate(path))-sum(PENALTY*sum((path[t][k]-path[t-1][k])**2 for k in [0,1]) for t in [1,2])
    assert abs(score-fs[-1][end])<2e-6
    # Template peak coordinate witness.
    im=rng.normal(size=(70,70)).astype(np.float32);q=im[30:39,28:37].copy()
    match=cv2.matchTemplate(im[20:49,18:47],q,cv2.TM_CCOEFF_NORMED)
    assert np.unravel_index(match.argmax(),match.shape)==(10,10)
    return {'quadratic_transition_bruteforce':True,'backtracking_score':True,'ncc_coordinates':True}

def load_detection(seq):
    if seq not in DEV:return {}
    p=ROOT/'results/references/release_evaluation'/seq
    meta=json.loads((p/'COMPLETE.json').read_text())
    z=np.load(p/'scores.npz',allow_pickle=False)
    assert list(z['plans'])==['O','S']
    matched=z['matched'];boxmax=z['box_max']
    return {int(a['annotation_id']):{'release_matched':bool(matched[0,i]),'release_box_max':float(boxmax[0,i])} for i,a in enumerate(meta['objects'])}

def inspect_one(frames,desc,anchor,hist,anns,lk):
    boxes=[a['bbox'] for a in hist];x,y=anchor;sep=max(3,int(math.ceil(max(boxes[-1][2:])/2))+2)
    raw=scoremaps(frames,desc,anchor,9,False);rich=scoremaps(frames,desc,anchor,17,True)
    template_std=float(frames[-1][y-4:y+5,x-4:x+5].std())
    response_stds=[float(np.std(z)) for z in rich[:-1]]
    valid=template_std>=1 and min(response_stds)>1e-4
    independent={}
    for name,maps in [('raw9',raw),('rich17',rich)]:
        coords=[absolute([np.unravel_index(z.argmax(),z.shape)],anchor)[0] for z in maps[:-1]]
        hits1=[inside(p,b,1) for p,b in zip(coords,boxes)];hits2=[inside(p,b,2) for p,b in zip(coords,boxes)]
        z=maps[-2];pp=peaks(z,sep)
        positions=absolute(pp,anchor);labels=[inside(p,boxes[-2],2) for p in positions]
        scores=[float(z[p]) for p in pp]
        bestgt=float(z[gtmask(boxes[-2],anchor,2)].max()) if gtmask(boxes[-2],anchor,2).any() else None
        wrong=[float(s) for s,p in zip(scores,positions) if not inside(p,boxes[-2],2)]
        pairgap=None if bestgt is None or not wrong else float(max(scores)-min(bestgt,max(wrong)))
        independent[name]={'history_hits_tol1':int(sum(hits1)),'history_hits_tol2':int(sum(hits2)),
            'previous_hit_tol1':bool(hits1[-1]),'previous_hit_tol2':bool(hits2[-1]),'previous_top6_contains_gt':bool(any(labels)),
            'previous_gt_wrong_gap':pairgap,'positions':coords+[anchor]}
    masks=[gtmask(b,anchor,1) for b in boxes]
    possible=masks[0].astype(np.uint8)
    for mask in masks[1:]:possible=cv2.dilate(possible,np.ones((2*STEP+1,2*STEP+1),np.uint8))*mask.astype(np.uint8)
    support=bool(possible[RAD,RAD]);roi=all(m.any() for m in masks)
    fs,fp=forward(rich);best=float(fs[-1][RAD,RAD]);bestpath=decode(fp,(RAD,RAD))
    bs=[None]*T;bp=[None]*T;bs[-1]=rich[-1].copy()
    for k in range(T-2,-1,-1):
        val,ptr=transition(bs[k+1]);bs[k]=rich[k]+val;bp[k]=ptr
    candidates=[];seen=set()
    for mid in [1,3,5]:
        total=fs[mid]+bs[mid]-rich[mid]
        for pos in peaks(total,sep):
            before=decode(fp[:mid+1],pos);after=[];node=pos
            for k in range(mid,T-1):node=tuple(bp[k][node]);after.append(node)
            path=before+after;key=tuple(path)
            if key in seen:continue
            seen.add(key);coords=absolute(path,anchor)
            hit1=sum(inside(p,b,1) for p,b in zip(coords[:-1],boxes[:-1]))
            hit2=sum(inside(p,b,2) for p,b in zip(coords[:-1],boxes[:-1]))
            candidates.append({'score':float(total[pos]),'gap_per_frame':max(0.,float((best-total[pos])/(T-1))),
                'hits_tol1':hit1,'hits_tol2':hit2,'positions':coords,'branch_frame':mid,
                'branch_type':object_type(coords[mid],anns[mid],hist[-1]['track_id'])})
    candidates.sort(key=lambda c:c['score'],reverse=True)
    bestabs=absolute(bestpath,anchor)
    besthits1=sum(inside(p,b,1) for p,b in zip(bestabs[:-1],boxes[:-1]))
    besthits2=sum(inside(p,b,2) for p,b in zip(bestabs[:-1],boxes[:-1]))
    gtscore=None;gtpath=None;gtgap=None
    if support:
        restricted=[np.where(mask,z,NEG).astype(np.float32) for mask,z in zip(masks,rich)]
        gf,gp=forward(restricted);gtscore=float(gf[-1][RAD,RAD]);gtpath=absolute(decode(gp,(RAD,RAD)),anchor)
        gtgap=max(0.,(best-gtscore)/(T-1))
    wrong=[c for c in candidates if c['hits_tol2']<=3]
    wronggap=min([c['gap_per_frame'] for c in wrong],default=None)
    alt=wrong[0] if wrong else (candidates[1] if len(candidates)>1 else None)
    competition={str(delta):bool(valid and support and gtgap<=delta and wronggap is not None and wronggap<=delta) for delta in [.02,.05,.1]}
    explicit_competition=bool(competition['0.05'] and any(c['hits_tol1']==7 and c['gap_per_frame']<=.05 for c in candidates))
    scr,ringstd=contrast(frames[-1],hist[-1]);lkcoords,lkvalid=lk
    lkhits=sum(bool(v) and inside(p,b,2) for p,v,b in zip(lkcoords[:-1],lkvalid[:-1],boxes[:-1]))
    lkhits1=sum(bool(v) and inside(p,b,1) for p,v,b in zip(lkcoords[:-1],lkvalid[:-1],boxes[:-1]))
    rec={'anchor':anchor,'template_std':template_std,'response_stds':response_stds,'numerically_valid':valid,'separation_radius':sep,
        'absolute_scr':abs(scr),'signed_scr':scr,'ring_std':ringstd,'weak_proxy':abs(scr)<1.5,
        'roi_gt_support':roi,'graph_gt_support':support,'best_path_hits_tol1':besthits1,'best_path_hits_tol2':besthits2,
        'best_path_good_tol1':besthits1>=6,'best_path_good_tol2':besthits2>=6,'gt_path_gap_per_frame':gtgap,'wrong_path_gap_per_frame':wronggap,
        'competition':competition,'lk_history_hits_tol2':int(lkhits),'lk_all_valid':bool(all(lkvalid)),
        'explicit_candidate_competition':explicit_competition,'lk_history_hits_tol1':int(lkhits1),'lk_path_good_tol1':lkhits1>=6,
        'lk_path_good_tol2':lkhits>=6,'independent':independent,'best_path':bestabs,'gt_compatible_oracle_path':gtpath,
        'gt_boxes':boxes,'lk_path':[[float(v) for v in q] for q in lkcoords],'lk_valid':[bool(v) for v in lkvalid],
        'alternative_path':alt,'candidate_paths':candidates,'best_score':best,'gt_compatible_score':gtscore}
    return rec,rich[3]

def summarize(rows):
    out={}
    for group in TRAIN+DEV+['train','dev','all']:
        rr=[r for r in rows if r['sequence']==group or r['split']==group or group=='all']
        valid=[r for r in rr if r['numerically_valid'] and r['graph_gt_support']]
        if not rr:continue
        q={'n':len(rr),'distinct_tracks':len({(r['sequence'],r['track_id']) for r in rr}),
            'valid_supported':len(valid),'roi_support':sum(r['roi_gt_support'] for r in rr),
            'graph_support':sum(r['graph_gt_support'] for r in rr),'template_invalid':sum(not r['numerically_valid'] for r in rr)}
        for subset,ss in [('all_valid',valid),('weak_valid',[r for r in valid if r['weak_proxy']])]:
            v={'n':len(ss)}
            if ss:
                v.update({'raw9_previous_accuracy':float(np.mean([r['independent']['raw9']['previous_hit_tol2'] for r in ss])),
                    'primary_tol1':{'raw9_previous_accuracy':float(np.mean([r['independent']['raw9']['previous_hit_tol1'] for r in ss])),
                        'rich17_previous_accuracy':float(np.mean([r['independent']['rich17']['previous_hit_tol1'] for r in ss])),
                        'raw9_history_accuracy':float(np.mean([r['independent']['raw9']['history_hits_tol1']/7 for r in ss])),
                        'rich17_history_accuracy':float(np.mean([r['independent']['rich17']['history_hits_tol1']/7 for r in ss])),
                        'dp_history_accuracy':float(np.mean([r['best_path_hits_tol1']/7 for r in ss])),
                        'dp_path_good_rate':float(np.mean([r['best_path_good_tol1'] for r in ss])),
                        'klt_path_good_rate':float(np.mean([r['lk_path_good_tol1'] for r in ss]))},
                    'rich17_previous_accuracy':float(np.mean([r['independent']['rich17']['previous_hit_tol2'] for r in ss])),
                    'raw9_history_accuracy':float(np.mean([r['independent']['raw9']['history_hits_tol2']/7 for r in ss])),
                    'rich17_history_accuracy':float(np.mean([r['independent']['rich17']['history_hits_tol2']/7 for r in ss])),
                    'dp_history_accuracy':float(np.mean([r['best_path_hits_tol2']/7 for r in ss])),
                    'dp_path_good_rate':float(np.mean([r['best_path_good_tol2'] for r in ss])),
                    'klt_path_good_rate':float(np.mean([r['lk_path_good_tol2'] for r in ss])),
                    'competition_counts':{str(vv):sum(r['competition'][str(vv)] for r in ss) for vv in [.02,.05,.1]},
                    'competition_rate':float(np.mean([r['competition']['0.05'] for r in ss])),
                    'explicit_candidate_competition_count':sum(r['explicit_candidate_competition'] for r in ss),
                    'competition_klt_also_fails':sum(r['competition']['0.05'] and not r['lk_path_good_tol2'] for r in ss),
                    'competition_best_correct':sum(r['competition']['0.05'] and r['best_path_good_tol2'] for r in ss),
                    'competition_best_wrong':sum(r['competition']['0.05'] and not r['best_path_good_tol2'] for r in ss)})
                dd=[r for r in ss if 'release_matched' in r]
                v['release_association']={}
                for flag in [True,False]:
                    gg=[r for r in dd if r['competition']['0.05']==flag]
                    v['release_association'][str(flag)]={'n':len(gg),'not_matched':sum(not r['release_matched'] for r in gg),'not_matched_rate':float(np.mean([not r['release_matched'] for r in gg])) if gg else None}
            q[subset]=v
        out[group]=q
    return out

def main():
    start=time.time();cv2.setNumThreads(2);rng=np.random.default_rng(46)
    assert not (OUT/'RESULTS.json').exists()
    freeze=json.loads((OUT/'FREEZE.json').read_text())
    for name,h in freeze['files'].items():assert sha(OUT/name)==h,name
    checks=selfcheck();dump('SELF_CHECK.json',checks)
    rows=[];hashes={};screening={};(OUT/'heatmaps').mkdir(exist_ok=True)
    for seq in TRAIN+DEV:
        assert seq not in FORBIDDEN
        base=ROOT/'data/stage1'/seq; ap=base/'annotations.json';rp=base/'raw_frames.npy'
        hashes[str(ap.relative_to(ROOT))]=sha(ap);hashes[str(rp.relative_to(ROOT))]=sha(rp)
        meta=json.loads(ap.read_text());ims=sorted(meta['images'],key=lambda z:z['frame_id'])
        assert len(ims)==160 and all(ims[j]['frame_id']==j for j in range(160))
        byimage=defaultdict(list)
        for a in meta['annotations']:byimage[a['image_id']].append(a)
        anns=[byimage[i['id']] for i in ims];tracks=[{a['track_id']:a for a in aa} for aa in anns]
        raw=np.load(rp,mmap_mode='r');assert raw.shape==(160,512,640) and raw.dtype==np.uint8
        cached=load_detection(seq);counts=defaultdict(int);n0=len(rows)
        if seq in DEV:
            for fn in ['COMPLETE.json','scores.npz']:
                pp=ROOT/'results/references/release_evaluation'/seq/fn
                hashes[str(pp.relative_to(ROOT))]=sha(pp)
        for end in range(7,160,8):
            slots=list(range(end-7,end+1));counts['windows']+=1
            counts['current_annotations']+=len(anns[end]);good=[]
            for a in sorted(anns[end],key=lambda q:q['id']):
                if not eligible(a):continue
                counts['eligible_small_current']+=1
                if any(a['track_id'] not in tracks[t] for t in slots):counts['missing_track_history']+=1;continue
                hist=[tracks[t][a['track_id']] for t in slots]
                if any(b.get('ignore',0) or b.get('iscrowd',0) or b['bbox'][2]<=0 or b['bbox'][3]<=0 for b in hist):counts['invalid_history']+=1;continue
                anchor=center(a);x,y=anchor
                if not (RAD+9<=x<640-RAD-9 and RAD+9<=y<512-RAD-9):counts['anchor_near_border']+=1;continue
                good.append((a,hist,anchor))
            counts['eligible_before_cap']+=len(good)
            if len(good)>8:good=[good[i] for i in sorted(rng.choice(len(good),8,replace=False))]
            if not good:continue
            frames=[raw[t] for t in slots];desc=[descriptors(im) for im in frames]
            points=np.array([g[2] for g in good],np.float32).reshape(-1,1,2)
            lk_positions=[None]*T;lk_status=[None]*T;lk_positions[-1]=points[:,0].copy();active=np.ones(len(good),bool);lk_status[-1]=active.copy()
            for k in range(T-2,-1,-1):
                points,status,err=cv2.calcOpticalFlowPyrLK(frames[k+1],frames[k],points,None,winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01),minEigThreshold=1e-4)
                active &= status[:,0].astype(bool)&np.isfinite(points[:,0]).all(1)
                points=np.nan_to_num(points,nan=0.,posinf=639.,neginf=0.);points[:,:,0]=np.clip(points[:,:,0],0,639);points[:,:,1]=np.clip(points[:,:,1],0,511)
                lk_positions[k]=points[:,0].copy();lk_status[k]=active.copy()
            for i,(a,hist,anchor) in enumerate(good):
                lk=([p[i] for p in lk_positions],[s[i] for s in lk_status])
                rec,heat=inspect_one(frames,desc,anchor,hist,[anns[t] for t in slots],lk)
                rec.update({'sample_id':len(rows),'sequence':seq,'split':'train' if seq in TRAIN else 'dev','frame':end,'slots':slots,'track_id':int(a['track_id']),'annotation_id':int(a['id']),'category_id':int(a['category_id'])})
                rec.update(cached.get(int(a['id']),{}));rows.append(rec)
                np.save(OUT/'heatmaps'/f'{rec["sample_id"]:04d}.npy',heat)
            counts['sampled']+=len(good)
        screening[seq]=dict(counts)
        with (OUT/'SAMPLES.jsonl').open('a') as f:
            for r in rows[n0:]:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
        dump('SCREENING.json',screening)
        print(json.dumps({'sequence':seq,'sampled':len(rows)-n0,'elapsed_seconds':round(time.time()-start,2)}),flush=True)
    summary=summarize(rows);dump('INPUT_HASHES.json',hashes)
    result={'status':'COMPLETE','elapsed_seconds':time.time()-start,'samples':len(rows),'screening':screening,'summary':summary,
        'device':'cpu','gpu_hours':0,'training':False,'confirmation_loaded':False,'sampled_sequences':TRAIN+DEV,
        'interpretation':'Oracle-current-anchor correspondence diagnostics; competition proxies require visual review; not detector performance or irreducible ambiguity.'}
    dump('RESULTS.json',result)
    print(json.dumps({'complete':True,'samples':len(rows),'dev':summary.get('dev'),'seconds':time.time()-start}),flush=True)

if __name__=='__main__':main()
