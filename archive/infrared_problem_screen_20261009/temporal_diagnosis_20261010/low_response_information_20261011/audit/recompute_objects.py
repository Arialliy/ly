#!/usr/bin/env python3
"""Independent scalar/block-loop audit; imports no original analysis helpers."""
import csv, datetime, hashlib, json, math, time
from collections import defaultdict
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parent
RUN=OUT.parent
Q=RUN.parent
P=Q/'stage1_goal'
PLANS={'O':[0,36,72,108,120],'S':[0,24,60,96,120]}
SEQS=['DJI_0026_2','DJI_0067_2']

def read(p):
    with Path(p).open() as f:return list(csv.DictReader(f))

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def key(r):return r['sequence'],int(r['frame']),int(r['annotation_id'])

def rect(b):
    x,y,w,h=map(float,b)
    return (min(640,max(0,math.ceil(x-.5))),min(512,max(0,math.ceil(y-.5))),
            min(640,max(0,math.ceil(x+w-.5))),min(512,max(0,math.ceil(y+h-.5))))

def peak(a,x,y):
    # Explicit equality/first-flat selection, distinct from author's np.argmax.
    v=float(np.max(a)); flat=int(np.flatnonzero(a.ravel()==v)[0]);yy,xx=divmod(flat,a.shape[1])
    return v,xx+x+.5,yy+y+.5

def mask(annotations):
    m=np.zeros((512,640),bool); count=0
    for a in annotations:
        if a['bbox'][2]<=0 or a['bbox'][3]<=0:continue
        x,y,xx,yy=rect(a['bbox'])
        if xx<=x or yy<=y:continue
        count+=1;m[max(0,y-2):min(512,yy+2),max(0,x-2):min(640,xx+2)]=True
    return m,count

def nrect(x,y,w,h):
    d=2*max(w,h)
    return max(0,x-d),max(0,y-d),min(640,x+w+d),min(512,y+h+d)

def tiles(z,ex,w,h,anchor=None,region=None,skip=None):
    # Literal rectangle loops, not reshape reductions or author's index formulas.
    if anchor is None:
        xs=range(0,640-w+1,w);ys=range(0,512-h+1,h)
    else:
        ax,ay=anchor
        xs=range(ax%w,640-w+1,w);ys=range(ay%h,512-h+1,h)
    values=[]
    for y in ys:
        if region and (y<region[1] or y+h>region[3]):continue
        for x in xs:
            if region and (x<region[0] or x+w>region[2]):continue
            if skip and x<skip[2] and x+w>skip[0] and y<skip[3] and y+h>skip[1]:continue
            if np.any(ex[y:y+h,x:x+w]):continue
            values.append(float(np.max(z[y:y+h,x:x+w])))
    return values

def percentile(values,q):
    a=sorted(values);p=(len(a)-1)*q;i=math.floor(p);f=p-i
    return a[i]*(1-f)+a[min(i+1,len(a)-1)]*f

def percentile32(values,q):
    # NumPy 2.2.5 preserves the float32 input dtype for scalar quantiles.
    # Independently emulate rounding at the rank, interpolation and subtraction
    # operations; do not relax tolerance on any downstream score column.
    a=sorted(map(np.float32,values));p=np.float32((len(a)-1)*np.float32(q))
    i=math.floor(float(p));gamma=np.float32(p-np.float32(i));left=a[i];right=a[min(i+1,len(a)-1)]
    difference=np.float32(right-left)
    if gamma>=np.float32(.5):return np.float32(right-np.float32(difference*np.float32(1-gamma)))
    return np.float32(left+np.float32(difference*gamma))

def contrast(prefix,vals,target):
    n=len(vals);out={prefix+'_valid_blocks':n,prefix+'_background_insufficient':int(n<20)}
    names=['bg_median_logit','bg_q95_logit','bg_q99_logit','bg_max_logit','margin_q95','margin_q99',
           'target_midrank_percentile','blocks_stronger_than_target','blocks_not_lower_than_target',
           'blocks_tied_with_target','advantage_q95','advantage_q99']
    if not n:out.update({prefix+'_'+k:'' for k in names});return out
    q95=percentile(vals,.95);q99=percentile(vals,.99)
    smaller=sum(v<target for v in vals);equal=sum(v==target for v in vals)
    results=[percentile(vals,.5),q95,q99,max(vals),target-q95,target-q99,
             (smaller+.5*equal)/n,sum(v>target for v in vals),sum(v>=target for v in vals),equal,
             int(target>q95),int(target>q99)]
    out.update({prefix+'_'+k:v for k,v in zip(names,results)})
    return out

def regional_background_peaks(z,ex):
    # No scipy maximum_filter or labeling: compare eight shifted arrays, then
    # traverse equal-valued candidate components explicitly with a Python stack.
    padded=np.pad(z,1,constant_values=-np.inf)
    maxima=np.ones(z.shape,bool)
    for dy in [-1,0,1]:
        for dx in [-1,0,1]:
            if dx or dy:maxima &= z>=padded[1+dy:513+dy,1+dx:641+dx]
    candidates=set(map(int,np.flatnonzero(maxima)))
    values=[];excluded=0;multi=0
    while candidates:
        seed=candidates.pop();stack=[seed];members=[seed];value=float(z.ravel()[seed]);touch=bool(ex.ravel()[seed])
        while stack:
            current=stack.pop();y,x=divmod(current,640)
            for dy in [-1,0,1]:
                for dx in [-1,0,1]:
                    if not(dx or dy):continue
                    yy,xx=y+dy,x+dx
                    if 0<=yy<512 and 0<=xx<640:
                        k=yy*640+xx
                        if k in candidates and float(z[yy,xx])==value:
                            candidates.remove(k);stack.append(k);members.append(k);touch |= bool(ex[yy,xx])
        multi+=len(members)>1
        if touch:excluded+=1
        else:values.append(float(z.ravel()[min(members)]))
    return values,excluded,multi

def selected_rows(cohorts):
    selection=[]
    for seq in SEQS:
        pool=[r for r in cohorts if r['sequence']==seq]
        ordered=sorted(pool,key=lambda r:hashlib.sha256(('|'.join(map(str,key(r)))+'|independent-audit-v1').encode()).hexdigest())
        picks=[];frames=set();sizes=set()
        # Hash order, no new logit/background/peak values used for selection.
        for r in ordered:
            size=(r['bbox_w'],r['bbox_h'])
            if int(r['frame']) in frames or size in sizes:continue
            picks.append(r);frames.add(int(r['frame']));sizes.add(size)
            if len(picks)==8:break
        assert len(picks)==8
        selection.extend(picks)
    with (OUT/'SAMPLE_SELECTION.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(selection[0]));w.writeheader();w.writerows(selection)
    return selection

def refs(cohorts,excluded):
    grouped=defaultdict(list)
    for r in cohorts:grouped[int(r['track_id'])].append(r)
    out={}
    for tid,rr in grouped.items():
        D=4*max(max(rect([r['bbox_'+x] for x in ['x','y','w','h']])[2+i]-rect([r['bbox_'+x] for x in ['x','y','w','h']])[i] for i in [0,1]) for r in rr)
        directions=[(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
        candidates=[(D*f*x,D*f*y) for f in [1,2] for x,y in directions]
        counts=[]
        for dx,dy in candidates:
            count=0
            for r in rr:
                x,y,xx,yy=rect([r['bbox_'+a] for a in ['x','y','w','h']]);x+=dx;xx+=dx;y+=dy;yy+=dy
                count+=int(0<=x<xx<=640 and 0<=y<yy<=512 and not np.any(excluded[int(r['frame'])][y:yy,x:xx]))
            counts.append(count)
        best=max(counts);idx=counts.index(best);dx,dy=candidates[idx]
        out[tid]={'D':D,'dx':dx,'dy':dy,'candidate_index':idx,'valid_frames':best,
                  'total_frames':len(rr),'all_candidate_valid_counts':';'.join(map(str,counts)),'geometry_only_selection':'True'}
    return out

def check_values(actual,expected,context,errors,stats):
    absent=set(actual)-set(expected)
    assert not absent,(context,'unaudited columns',sorted(absent))
    for name,a in actual.items():
        e=expected[name];stats['cells_checked']+=1
        if e=='':ok=a==''
        elif isinstance(e,(int,float,np.number)):
            try:
                diff=abs(float(a)-float(e))
                tol=1e-8
                ok=math.isfinite(float(a)) and diff<=tol
                stats['largest_numeric_difference']=max(stats['largest_numeric_difference'],diff)
            except ValueError:ok=False
        else:ok=str(a)==str(e)
        if not ok:errors.append({'context':context,'column':name,'actual':a,'expected':e})

def main():
    begin=time.time();cohorts=read(RUN/'COHORTS.csv');samples=selected_rows(cohorts)
    actual_rows=read(RUN/'analysis/LOW_RESPONSE_OBJECTS.csv');lookup={(key(r),r['arrangement']):r for r in actual_rows}
    assert len(actual_rows)==len(lookup)==10400
    errors=[];stats={'cells_checked':0,'largest_numeric_difference':0};sample_checks=[];refchecks=[]
    for seq in SEQS:
        ann=json.loads((P/'data'/seq/'annotations.json').read_text());frames=defaultdict(list)
        imageidx={a['id']:i for i,a in enumerate(ann['images'])}
        for a in ann['annotations']:frames[imageidx[a['image_id']]].append(a)
        exclusions={t:mask(frames[t])[0] for t in range(160)}
        reference=refs([r for r in cohorts if r['sequence']==seq],exclusions)
        for r in read(RUN/'analysis/REFERENCE_SELECTION.csv'):
            if r['sequence']==seq:
                expected={'sequence':seq,'track_id':int(r['track_id']),**reference[int(r['track_id'])]}
                check_values(r,expected,('reference',seq,r['track_id']),errors,stats);refchecks.append(expected)
        cacheindex={};i=0
        for t in range(160):
            for a in sorted(frames[t],key=lambda a:(a['track_id'],a['id'])):
                x,y,w,h=a['bbox']
                if a.get('ignore') or w<=0 or h<=0 or min(640,x+w)<=max(0,x) or min(512,y+h)<=max(0,y):continue
                cacheindex[t,a['id']]=i;i+=1
        for plan,starts in PLANS.items():
            zall=np.load(RUN/'logits'/seq/f'fused_{plan}_logits.npy',mmap_mode='r')
            base=P/'predictions/official'/seq if plan=='O' else Q/'problem_validity_20261011/simple_controls/symmetric5_probmax'/seq
            probs=np.load(base/('official_fused.npz' if plan=='O' else 'fused.npz'))['probabilities']
            scorefile=P/'evaluation/official'/seq/'official_fused_matches.npz' if plan=='O' else base/'fused_matches.npz'
            cache=np.load(scorefile);threshold=list(cache['thresholds']).index(.5)
            for c in [r for r in samples if r['sequence']==seq]:
                t=int(c['frame']);tid=int(c['track_id']);a_id=int(c['annotation_id']);actual=lookup[(key(c),plan)]
                z=zall[t];p=probs[t];ex,count=mask(frames[t]);b=list(map(float,[c['bbox_'+a] for a in ['x','y','w','h']]))
                x,y,xx,yy=rect(b);w=xx-x;h=yy-y;v,px,py=peak(z[y:yy,x:xx],x,y);pv,ppx,ppy=peak(p[y:yy,x:xx],x,y)
                nb=nrect(x,y,w,h);nx,ny,nxx,nyy=nb;nz=z[ny:nyy,nx:nxx];nv,npx,npy=peak(nz,nx,ny)
                npv,nppx,nppy=peak(p[ny:nyy,nx:nxx],nx,ny)
                bgp,bgexcluded,multi=regional_background_peaks(z,ex)
                inside=lambda bb,px,py:bb[0]<=px<bb[0]+bb[2] and bb[1]<=py<bb[1]+bb[3]
                dist=math.hypot(max(b[0]-npx,0,npx-b[0]-b[2]),max(b[1]-npy,0,npy-b[1]-b[3]))
                ci=cacheindex[t,a_id];matched=int(cache['matched'][threshold,ci])
                e={'sequence':seq,'track_id':tid,'frame':t,'source_frame':int(Path(ann['images'][t]['file_name']).stem),
                   'annotation_id':a_id,'category':c['category'],'arrangement':plan,'arrangement_starts':';'.join(map(str,starts)),
                   'original_cohort':c['cohort'],'original_system_case':c['original_system_case'],
                   'all_windows_lt_05':int(c['low_all_05']),'all_windows_lt_001':int(c['extreme_all_001']),
                   'both_fixed_matched':int(c['both_fixed_success']),'extra_window_only':int(c['extra_only']),
                   'max_candidate_box_response':float(c['max_candidate_box_response']),
                   'bbox_x':b[0],'bbox_y':b[1],'bbox_w':b[2],'bbox_h':b[3],'crop_x0':x,'crop_y0':y,'crop_w':w,'crop_h':h,
                   'target_box_max_logit':v,'target_logit_peak_x':px,'target_logit_peak_y':py,
                   'target_box_max_probability':pv,'target_probability_peak_x':ppx,'target_probability_peak_y':ppy,
                   'target_logit_and_probability_peak_same':int((px,py)==(ppx,ppy)),'target_logit_peak_probability':float(p[int(py-.5),int(px-.5)]),
                   'coverage_windows':len([a for a in starts if a<=t<a+40]),'covering_starts':';'.join(str(a) for a in starts if a<=t<a+40),
                   'all_gt_excluded_annotations':count,'all_gt_excluded_pixels':int(ex.sum()),
                   'global_background_peaks':len(bgp),'global_peak_plateaus_excluded_for_gt_overlap':bgexcluded,
                   'global_background_peaks_stronger_than_target':sum(bv>v for bv in bgp),
                   'global_background_peaks_not_lower_than_target':sum(bv>=v for bv in bgp),
                   'global_background_peaks_tied_with_target':sum(bv==v for bv in bgp),
                   'neighborhood_x0':nx,'neighborhood_y0':ny,'neighborhood_x1':nxx,'neighborhood_y1':nyy,
                   'neighborhood_border_clipped':int(nx!=x-2*max(w,h) or ny!=y-2*max(w,h) or nxx!=xx+2*max(w,h) or nyy!=yy+2*max(w,h)),
                   'neighborhood_peak_x':npx,'neighborhood_peak_y':npy,'neighborhood_peak_logit':nv,'neighborhood_area':nz.size,
                   'neighborhood_max_minus_median':float(np.float32(np.float32(nv)-np.float32(percentile(nz.ravel().tolist(),.5)))),
                   'neighborhood_max_minus_q95':float(np.float32(np.float32(nv)-percentile32(nz.ravel(),.95))),
                   'neighborhood_dynamic_range':float(np.float32(np.float32(nv)-nz.min())),'neighborhood_nearly_flat':int(nv-float(nz.min())<=1e-6),
                   'neighborhood_peak_probability':float(p[int(npy-.5),int(npx-.5)]),'neighborhood_peak_inside_target':int(inside(b,npx,npy)),
                   'neighborhood_peak_other_gt_ids':';'.join(str(a['id']) for a in frames[t] if a['id']!=a_id and inside(a['bbox'],npx,npy)),
                   'neighborhood_peak_distance_to_box':dist,'neighborhood_peak_distance_normalized':dist/max(b[2:]),
                   'neighborhood_probability_max':npv,'neighborhood_probability_peak_x':nppx,'neighborhood_probability_peak_y':nppy,
                   'neighborhood_logit_and_probability_peak_same':int((npx,npy)==(nppx,nppy)),
                   'matched_at_05':matched,'matched_peak_x':float(cache['peak_x'][threshold,ci]) if matched else '',
                   'matched_peak_y':float(cache['peak_y'][threshold,ci]) if matched else '',
                   'target_partly_outside':0,'target_overlap_gt_count':0,'target_empty_crop':0}
                for name in ['low_all_05','extreme_all_001','low_nonextreme','both_fixed_success','extra_only','high_unmatched']:e[name]=int(c[name])
                e.update(contrast('local',tiles(z,ex,w,h,(x,y),nb),v));e.update(contrast('global',tiles(z,ex,w,h),v))
                rr=reference[tid];dx=rr['dx'];dy=rr['dy'];rx=x+dx;ry=y+dy;rxx=xx+dx;ryy=yy+dy
                inf=0<=rx<rxx<=640 and 0<=ry<ryy<=512;valid=inf and not np.any(ex[ry:ryy,rx:rxx])
                e.update({'reference_offset_dx':dx,'reference_offset_dy':dy,'reference_offset_D':rr['D'],
                          'reference_candidate_index':rr['candidate_index'],'reference_track_valid_frames':rr['valid_frames'],
                          'reference_valid':int(valid),'reference_x0':rx,'reference_y0':ry,'reference_w':w,'reference_h':h,
                          'reference_invalid_reason':'' if valid else ('outside_frame' if not inf else 'annotated_gt_buffer_overlap')})
                refcols=['max_logit','peak_x','peak_y','peak_probability','box_max_probability',
                         'local_valid_blocks','local_q95','local_q99','local_margin_q95','local_margin_q99','local_advantage_q95','local_background_insufficient',
                         'global_valid_blocks','global_q95','global_q99','global_margin_q95','global_margin_q99','global_advantage_q95','global_background_insufficient']
                e.update({'reference_'+s:'' for s in refcols})
                if valid:
                    rv,rpx,rpy=peak(z[ry:ryy,rx:rxx],rx,ry)
                    e.update({'reference_max_logit':rv,'reference_peak_x':rpx,'reference_peak_y':rpy,
                              'reference_peak_probability':float(p[int(rpy-.5),int(rpx-.5)]),'reference_box_max_probability':float(p[ry:ryy,rx:rxx].max())})
                    rex=ex.copy();rex[max(0,ry-2):min(512,ryy+2),max(0,rx-2):min(640,rxx+2)]=True
                    for name,vv in [('local',tiles(z,rex,w,h,(rx,ry),nrect(rx,ry,w,h))),('global',tiles(z,ex,w,h,skip=(rx,ry,rxx,ryy)))]:
                        e['reference_'+name+'_valid_blocks']=len(vv);e['reference_'+name+'_background_insufficient']=int(len(vv)<20)
                        if vv:
                            q95=percentile(vv,.95);q99=percentile(vv,.99)
                            e.update({'reference_'+name+'_q95':q95,'reference_'+name+'_q99':q99,'reference_'+name+'_margin_q95':rv-q95,
                                      'reference_'+name+'_margin_q99':rv-q99,'reference_'+name+'_advantage_q95':int(rv>q95)})
                check_values(actual,e,(seq,plan,t,a_id),errors,stats)
                sample_checks.append({'sequence':seq,'arrangement':plan,'frame':t,'annotation_id':a_id,'track_id':tid,
                                      'width':w,'height':h,'audited_columns':len(actual),'multi_pixel_peak_plateaus':multi})
            del probs,zall,cache
    result={'status':'PASS' if not errors else 'FAIL','rows_in_detail':len(actual_rows),'sample_rows':len(sample_checks),
            'sampled_target_frames':len(samples),'reference_tracks_all_recomputed':len(refchecks),**stats,
            'sample_checks':sample_checks,'mismatches':errors,'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'elapsed_seconds':time.time()-begin,'independence':'No import of author scoring/statistics functions. Direct block loops, manual quantiles, shifted comparison + stack plateau traversal.',
            'precision_resolution':'Initial float64 mathematical neighborhood quantile reference differed in 30 cells (max 3.8624e-6). Native NumPy 2.2.5 scalar quantile and subtraction are float32. Final audit emulates each float32 rounding operation independently, with no relaxed tolerance; initial findings preserved separately.',
            'source_sha256':{str(p):digest(p) for p in [Path(__file__),RUN/'PROTOCOL.md',RUN/'COHORTS.csv',RUN/'analysis/build_low_response_objects.py',RUN/'analysis/LOW_RESPONSE_OBJECTS.csv',RUN/'analysis/REFERENCE_SELECTION.csv']}}
    (OUT/'OBJECT_RECOMPUTATION.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'sample_rows':len(sample_checks),'checked_cells':stats['cells_checked'],'mismatches':len(errors),'seconds':result['elapsed_seconds']}),flush=True)
    if errors:raise AssertionError(errors[:5])

if __name__=='__main__':main()
