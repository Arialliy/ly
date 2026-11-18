#!/usr/bin/env python3
"""Fixed-cohort low-response diagnostics; no training, threshold or matching change."""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, math, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from scipy import ndimage

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
RUN=OUT.parent
P=ROOT/'stage1_goal'
PLANS={'O':[0,36,72,108,120], 'S':[0,24,60,96,120]}
SEQS=['DJI_0026_2','DJI_0067_2']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()


def readcsv(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def csvout(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def jsonout(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def at(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


def peak(arr,x0=0,y0=0):
    k=int(np.argmax(arr));y,x=divmod(k,arr.shape[1]);return float(arr[y,x]),x+x0+.5,y+y0+.5


def contains(box,px,py):
    x,y,w,h=box;return x<=px<x+w and y<=py<y+h


def bbox_distance(box,px,py):
    x,y,w,h=box;return math.hypot(max(x-px,0.,px-x-w),max(y-py,0.,py-y-h))


def global_tiles(logit,excluded,bw,bh):
    H,W=logit.shape;nr,nc=H//bh,W//bw
    shape=(nr,bh,nc,bw)
    scores=logit[:nr*bh,:nc*bw].reshape(shape).max(axis=(1,3))
    valid=~excluded[:nr*bh,:nc*bw].reshape(shape).any(axis=(1,3))
    vals=np.sort(scores[valid].astype(np.float64))
    return vals,scores,valid


def local_tiles(logit,excluded,bw,bh,x0,y0,neighborhood):
    nx0,ny0,nx1,ny1=neighborhood
    blocks=[]
    for j in range(math.ceil((ny0-y0)/bh),math.floor((ny1-y0-bh)/bh)+1):
        yy=y0+j*bh
        for i in range(math.ceil((nx0-x0)/bw),math.floor((nx1-x0-bw)/bw)+1):
            xx=x0+i*bw
            if excluded[yy:yy+bh,xx:xx+bw].any():continue
            v,px,py=peak(logit[yy:yy+bh,xx:xx+bw],xx,yy)
            blocks.append((v,i,j,xx,yy,px,py))
    # First background reference depends only on geometry, never model score.
    blocks.sort(key=lambda z:(abs(z[1])+abs(z[2]),z[2],z[1]))
    return np.sort(np.array([r[0] for r in blocks],np.float64)),blocks


def contrast(prefix,vals,target):
    n=len(vals)
    row={prefix+'_valid_blocks':n,prefix+'_background_insufficient':int(n<20)}
    if n:
        lo=int(np.searchsorted(vals,target,'left'));hi=int(np.searchsorted(vals,target,'right'))
        q50,q95,q99=np.quantile(vals,[.5,.95,.99])
        row.update({prefix+'_bg_median_logit':float(q50),prefix+'_bg_q95_logit':float(q95),
            prefix+'_bg_q99_logit':float(q99),prefix+'_bg_max_logit':float(vals[-1]),
            prefix+'_margin_q95':float(target-q95),prefix+'_margin_q99':float(target-q99),
            prefix+'_target_midrank_percentile':(lo+(hi-lo)/2)/n,
            prefix+'_blocks_stronger_than_target':n-hi,prefix+'_blocks_not_lower_than_target':n-lo,prefix+'_blocks_tied_with_target':hi-lo,
            prefix+'_advantage_q95':int(target>q95),prefix+'_advantage_q99':int(target>q99)})
    else:
        row.update({prefix+'_'+k:'' for k in ['bg_median_logit','bg_q95_logit','bg_q99_logit','bg_max_logit',
            'margin_q95','margin_q99','target_midrank_percentile','blocks_stronger_than_target','blocks_not_lower_than_target','blocks_tied_with_target','advantage_q95','advantage_q99']})
    return row


def background_peaks(logit,excluded):
    maximum=ndimage.maximum_filter(logit,size=3,mode='constant',cval=-np.inf)
    ispeak=logit==maximum
    labels,n=ndimage.label(ispeak,structure=np.ones((3,3),np.uint8))
    if not n:return np.empty(0,np.float64),0
    ids=np.arange(1,n+1)
    # An 8-connected equal-valued local-max plateau contributes one row-major point.
    flats=np.flatnonzero(ispeak);labs=labels.ravel()[flats]
    first=np.full(n+1,logit.size,np.int64);np.minimum.at(first,labs,flats)
    intersects=np.bincount(labels[excluded],minlength=n+1)>0
    chosen=first[ids[~intersects[1:]]]
    return np.sort(logit.ravel()[chosen].astype(np.float64)),int(intersects[1:].sum())


def summary(rows):
    groups={'all':lambda r:True,'low_all_05':lambda r:r['low_all_05']==1,
        'extreme_all_001':lambda r:r['extreme_all_001']==1,
        'low_nonextreme':lambda r:r['low_nonextreme']==1,
        'both_fixed_success':lambda r:r['both_fixed_success']==1,
        'extra_only':lambda r:r['extra_only']==1,
        'high_unmatched':lambda r:r['high_unmatched']==1}
    out=[]
    for plan in PLANS:
      for seq in ['pooled']+SEQS:
       for group,pred in groups.items():
        rr=[r for r in rows if r['arrangement']==plan and (seq=='pooled' or r['sequence']==seq) and pred(r)]
        valid=[r for r in rr if r['local_valid_blocks']>=20 and r['global_valid_blocks']>=20]
        nonempty=[r for r in rr if r['local_valid_blocks'] and r['global_valid_blocks']]
        n=len(rr);nv=len(valid)
        base={'arrangement':plan,'sequence':seq,'group':group,'target_frames':n,
            'tracks':len({(r['sequence'],r['track_id']) for r in rr}), 'clips':len({r['sequence'] for r in rr}),
            'valid_both_backgrounds':nv,'nonempty_both_backgrounds':len(nonempty),'insufficient_background_rows':n-nv,
            'matched_at_05':sum(r['matched_at_05'] for r in rr)}
        for prefix in ['local','global']:
            pv=[r for r in rr if r[prefix+'_valid_blocks']>=20]
            pn=[r for r in rr if r[prefix+'_valid_blocks']>0]
            base[prefix+'_valid_ge20_rows']=len(pv)
            base[prefix+'_nonempty_rows']=len(pn)
            base[prefix+'_insufficient_1_to_19_rows']=len(pn)-len(pv)
            base[prefix+'_missing_0_rows']=n-len(pn)
            for col in ['margin_q95','margin_q99','target_midrank_percentile','blocks_stronger_than_target']:
                vv=np.array([r[prefix+'_'+col] for r in pv],np.float64)
                for label,q in [('q25',.25),('median',.5),('q75',.75)]:
                    base[prefix+'_'+col+'_'+label]=float(np.quantile(vv,q)) if len(vv) else ''
            for q in ['q95','q99']:
                for label,subset in [('valid_ge20',pv),('all_nonempty',pn)]:
                    wins=sum(r[prefix+'_advantage_'+q] for r in subset)
                    base[prefix+'_advantage_'+q+'_'+label+'_count']=wins
                    base[prefix+'_advantage_'+q+'_'+label+'_rate']=wins/len(subset) if subset else ''
        for label,fn in [('both_advantage_q95',lambda r:r['local_advantage_q95'] and r['global_advantage_q95']),
            ('local_only_advantage_q95',lambda r:r['local_advantage_q95'] and not r['global_advantage_q95']),
            ('global_only_advantage_q95',lambda r:not r['local_advantage_q95'] and r['global_advantage_q95']),
            ('neither_advantage_q95',lambda r:not r['local_advantage_q95'] and not r['global_advantage_q95'])]:
            for slabel,subset in [('valid_ge20',valid),('all_nonempty',nonempty)]:
                ss=[r for r in subset if fn(r)]
                base[label+'_'+slabel+'_frames']=len(ss)
                base[label+'_'+slabel+'_tracks']=len({(r['sequence'],r['track_id']) for r in ss})
                base[label+'_'+slabel+'_clips']=len({r['sequence'] for r in ss})
        for col in ['target_box_max_logit','global_background_peaks_stronger_than_target','global_background_peaks_not_lower_than_target']:
            vv=[r[col] for r in rr]
            for label,q in [('q25',.25),('median',.5),('q75',.75)]:
                base[col+'_'+label]=float(np.quantile(vv,q)) if vv else ''
        out.append(base)
    return out


def expanded_neighborhood(x0,y0,bw,bh):
    m=2*max(bw,bh)
    return max(0,x0-m),max(0,y0-m),min(640,x0+bw+m),min(512,y0+bh+m)


def exclusion_masks(allann,helper):
    masks=[]
    for frame in range(160):
        excluded=np.zeros((512,640),bool)
        for aa in allann[frame]:
            box=list(map(float,aa['bbox']))
            if box[2]<=0 or box[3]<=0:continue
            ax0,ay0,ax1,ay1=helper.crop_indices(box,512,640)
            if ax1<=ax0 or ay1<=ay0:continue
            excluded[max(0,ay0-2):min(512,ay1+2),max(0,ax0-2):min(640,ax1+2)]=True
        masks.append(excluded)
    return masks


def select_reference_offsets(seqrows,excluded,helper):
    tracks=defaultdict(list)
    for r in seqrows:tracks[int(r['track_id'])].append(r)
    references={}
    info=[]
    for tid,rows in tracks.items():
        crops=[helper.crop_indices([float(r['bbox_'+k]) for k in ['x','y','w','h']],512,640) for r in rows]
        d=4*max(max(c[2]-c[0],c[3]-c[1]) for c in crops)
        dirs=[(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
        candidates=[(dx*factor*d,dy*factor*d) for factor in [1,2] for dx,dy in dirs]
        counts=[]
        for dx,dy in candidates:
            count=0
            for r,(x0,y0,x1,y1) in zip(rows,crops):
                rx0,ry0,rx1,ry1=x0+dx,y0+dy,x1+dx,y1+dy
                if 0<=rx0<rx1<=640 and 0<=ry0<ry1<=512 and not excluded[int(r['frame'])][ry0:ry1,rx0:rx1].any():count+=1
            counts.append(count)
        chosen=int(np.argmax(counts));dx,dy=candidates[chosen]
        references[tid]=(dx,dy,chosen,counts[chosen],d)
        info.append({'sequence':rows[0]['sequence'],'track_id':tid,'D':d,'dx':dx,'dy':dy,'candidate_index':chosen,'valid_frames':counts[chosen],
            'total_frames':len(rows),'all_candidate_valid_counts':';'.join(map(str,counts)),'geometry_only_selection':True})
    return references,info

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');a=parser.parse_args()
    if not a.run:raise SystemExit('Prepared only. Require explicit --run after frozen PROTOCOL exists.')
    if not (RUN/'PROTOCOL.md').exists() or not (RUN/'PROTOCOL_FREEZE.json').exists():raise RuntimeError('Protocol not frozen')
    target=OUT/'LOW_RESPONSE_OBJECTS.csv'
    if target.exists():raise FileExistsError('Refuse silent overwrite of completed detailed table')
    scorer=at('low_response_existing_scorer',P/'pilot/score_pilot.py');helper,_=scorer.helper()
    support=readcsv(ROOT/'combination_feasibility_20261011/support/OBJECT_SUPPORT.csv')
    old=readcsv(ROOT/'problem_validity_20261011/OBJECTS.csv')
    key=lambda r:(r['sequence'],int(r['frame']),int(r['annotation_id']))
    old={key(r):r for r in old if r['sequence'] in SEQS and r['small_geometry']=='1'}
    assert len(support)==len(old)==5200
    assert sum(float(r['max_candidate_box_response'])<.5 for r in support)==4342
    assert sum(float(r['max_candidate_box_response'])<.001 for r in support)==4170
    assert sum(r['cohort']=='both_success' for r in support)==646
    assert sum(r['system_case']=='extra_window_only_opportunity' for r in old.values())==104
    allrows=[];validation=[];reference_info=[];started=time.time()
    for seq in SEQS:
        annpath=P/'data'/seq/'annotations.json'
        if not annpath.exists():annpath=P/'data_diagnostics'/seq/'annotations.json'
        gt,meta,offsets,exc=scorer.gt_data(annpath,helper)
        ann=json.loads(annpath.read_text());images=ann['images'];imageframe={im['id']:i for i,im in enumerate(images)}
        allann=defaultdict(list)
        for aa in ann['annotations']:allann[imageframe[aa['image_id']]].append(aa)
        byframe=defaultdict(list)
        for r in support:
            if r['sequence']==seq:byframe[int(r['frame'])].append(r)
        lookup={(m['frame'],m['annotation_id']):i for i,m in enumerate(meta)}
        excluded_byframe=exclusion_masks(allann,helper)
        references,refinfo=select_reference_offsets([r for r in support if r['sequence']==seq],excluded_byframe,helper)
        reference_info.extend(refinfo)
        for plan,starts in PLANS.items():
            logpath=RUN/'logits'/seq/f'fused_{plan}_logits.npy'
            complete=RUN/'logits'/'COMPLETE.json'
            if not complete.exists():raise RuntimeError('Logit export incomplete: '+seq)
            logits=np.load(logpath,mmap_mode='r');assert logits.shape==(160,512,640)
            if plan=='O':
                probpath=P/'predictions/official'/seq/'official_fused.npz'
                scorepath=P/'evaluation/official'/seq/'official_fused_matches.npz'
            else:
                probpath=ROOT/'problem_validity_20261011/simple_controls/symmetric5_probmax'/seq/'fused.npz'
                scorepath=ROOT/'problem_validity_20261011/simple_controls/symmetric5_probmax'/seq/'fused_matches.npz'
            with np.load(probpath) as z:probs=z['probabilities']
            with np.load(scorepath) as z:
                th=np.flatnonzero(z['thresholds']==.5);assert len(th)==1
                matched=z['matched'][th[0]].copy();cachedbox=z['box_max'].copy()
                matchx=z['peak_x'][th[0]].copy();matchy=z['peak_y'][th[0]].copy()
            for frame in range(160):
                logit=np.asarray(logits[frame]);prob=probs[frame]
                assert np.isfinite(logit).all()
                excluded=excluded_byframe[frame]
                excluded_ann_count=sum(float(aa['bbox'][2])>0 and float(aa['bbox'][3])>0 and helper.crop_indices(aa['bbox'],512,640)[2]>helper.crop_indices(aa['bbox'],512,640)[0] and helper.crop_indices(aa['bbox'],512,640)[3]>helper.crop_indices(aa['bbox'],512,640)[1] for aa in allann[frame])
                globalpeaks,peaks_excluded=background_peaks(logit,excluded)
                globalcache={}
                for sr in byframe[frame]:
                    ii=lookup[(frame,int(sr['annotation_id']))];m=meta[ii]
                    box=m['bbox'];x,y,w,h=box;x0,y0,x1,y1=helper.crop_indices(box,512,640);bw,bh=x1-x0,y1-y0
                    assert bw>0 and bh>0
                    score,tx,ty=peak(logit[y0:y1,x0:x1],x0,y0)
                    probscore,px,py=peak(prob[y0:y1,x0:x1],x0,y0)
                    assert probscore==float(cachedbox[ii])
                    assert int(matched[ii])==int(sr['original_matched'] if plan=='O' else sr['symmetric_matched'])
                    margin=2*max(bw,bh)
                    nb=expanded_neighborhood(x0,y0,bw,bh)
                    nx0,ny0,nx1,ny1=nb
                    nscore,nx,ny=peak(logit[ny0:ny1,nx0:nx1],nx0,ny0)
                    nprob,npx,npy=peak(prob[ny0:ny1,nx0:nx1],nx0,ny0)
                    if (bw,bh) not in globalcache:globalcache[(bw,bh)]=global_tiles(logit,excluded,bw,bh)
                    gvals,gscores,gvalid=globalcache[(bw,bh)]
                    lvals,lblocks=local_tiles(logit,excluded,bw,bh,x0,y0,nb)
                    o=old[key(sr)]
                    row={'sequence':seq,'track_id':int(sr['track_id']),'frame':frame,'source_frame':int(sr['source_frame']),
                        'annotation_id':int(sr['annotation_id']),'category':sr['category'],'arrangement':plan,
                        'arrangement_starts':';'.join(map(str,starts)),'original_cohort':sr['cohort'],'original_system_case':o['system_case'],
                        'all_windows_lt_05':int(float(sr['max_candidate_box_response'])<.5),
                        'all_windows_lt_001':int(float(sr['max_candidate_box_response'])<.001),
                        'both_fixed_matched':int(sr['cohort']=='both_success'),
                        'extra_window_only':int(o['system_case']=='extra_window_only_opportunity'),
                        'low_all_05':int(float(sr['max_candidate_box_response'])<.5),
                        'extreme_all_001':int(float(sr['max_candidate_box_response'])<.001),
                        'low_nonextreme':int(.001<=float(sr['max_candidate_box_response'])<.5),
                        'both_fixed_success':int(sr['cohort']=='both_success'),
                        'extra_only':int(o['system_case']=='extra_window_only_opportunity'),
                        'high_unmatched':int(o['system_case']=='no_match_but_box_response'),
                        'max_candidate_box_response':float(sr['max_candidate_box_response']),
                        'bbox_x':x,'bbox_y':y,'bbox_w':w,'bbox_h':h,'crop_x0':x0,'crop_y0':y0,'crop_w':bw,'crop_h':bh,
                        'target_box_max_logit':score,'target_logit_peak_x':tx,'target_logit_peak_y':ty,
                        'target_box_max_probability':probscore,'target_probability_peak_x':px,'target_probability_peak_y':py,
                        'target_logit_and_probability_peak_same':int(tx==px and ty==py),
                        'target_logit_peak_probability':float(prob[int(ty-.5),int(tx-.5)]),
                        'coverage_windows':sum(s<=frame<s+40 for s in starts),
                        'covering_starts':';'.join(str(s) for s in starts if s<=frame<s+40),
                        'all_gt_excluded_annotations':excluded_ann_count,'all_gt_excluded_pixels':int(excluded.sum()),
                        'global_background_peaks':len(globalpeaks),'global_peak_plateaus_excluded_for_gt_overlap':peaks_excluded,
                        'global_background_peaks_stronger_than_target':len(globalpeaks)-int(np.searchsorted(globalpeaks,score,'right')),
                        'global_background_peaks_not_lower_than_target':len(globalpeaks)-int(np.searchsorted(globalpeaks,score,'left')),
                        'global_background_peaks_tied_with_target':int(np.searchsorted(globalpeaks,score,'right')-np.searchsorted(globalpeaks,score,'left'))}
                    row.update(contrast('local',lvals,score));row.update(contrast('global',gvals,score))
                    row.update({'neighborhood_x0':nx0,'neighborhood_y0':ny0,'neighborhood_x1':nx1,'neighborhood_y1':ny1,
                        'neighborhood_border_clipped':int(x-margin<0 or y-margin<0 or x+w+margin>640 or y+h+margin>512),
                        'neighborhood_peak_x':nx,'neighborhood_peak_y':ny,'neighborhood_peak_logit':nscore,
                        'neighborhood_area':int((nx1-nx0)*(ny1-ny0)),
                        'neighborhood_max_minus_median':float(nscore-np.median(logit[ny0:ny1,nx0:nx1])),
                        'neighborhood_max_minus_q95':float(nscore-np.quantile(logit[ny0:ny1,nx0:nx1],.95)),
                        'neighborhood_dynamic_range':float(nscore-logit[ny0:ny1,nx0:nx1].min()),
                        'neighborhood_nearly_flat':int(nscore-logit[ny0:ny1,nx0:nx1].min()<=1e-6),
                        'neighborhood_peak_probability':float(prob[int(ny-.5),int(nx-.5)]),
                        'neighborhood_peak_inside_target':int(contains(box,nx,ny)),
                        'neighborhood_peak_other_gt_ids':';'.join(str(g['id']) for g in allann[frame] if int(g['id'])!=m['annotation_id'] and contains(g['bbox'],nx,ny)),
                        'neighborhood_peak_distance_to_box':bbox_distance(box,nx,ny),
                        'neighborhood_peak_distance_normalized':bbox_distance(box,nx,ny)/max(w,h),
                        'neighborhood_probability_max':nprob,'neighborhood_probability_peak_x':npx,'neighborhood_probability_peak_y':npy,
                        'neighborhood_logit_and_probability_peak_same':int(nx==npx and ny==npy),
                        'matched_at_05':int(matched[ii]),'matched_peak_x':float(matchx[ii]) if matched[ii] else '',
                        'matched_peak_y':float(matchy[ii]) if matched[ii] else '',
                        'target_partly_outside':0,'target_overlap_gt_count':0,'target_empty_crop':0})
                    dx,dy,refchoice,nref,d=references[int(sr['track_id'])]
                    rx0,ry0,rx1,ry1=x0+dx,y0+dy,x1+dx,y1+dy
                    inframe=0<=rx0<rx1<=640 and 0<=ry0<ry1<=512
                    refvalid=inframe and not excluded[ry0:ry1,rx0:rx1].any()
                    row.update({'reference_offset_dx':dx,'reference_offset_dy':dy,'reference_offset_D':d,
                        'reference_candidate_index':refchoice,'reference_track_valid_frames':nref,
                        'reference_valid':int(refvalid),'reference_x0':rx0,'reference_y0':ry0,
                        'reference_w':bw,'reference_h':bh,'reference_invalid_reason':'' if refvalid else ('outside_frame' if not inframe else 'annotated_gt_buffer_overlap')})
                    for k in ['max_logit','peak_x','peak_y','peak_probability','box_max_probability',
                              'local_valid_blocks','local_q95','local_q99','local_margin_q95','local_margin_q99','local_advantage_q95','local_background_insufficient',
                              'global_valid_blocks','global_q95','global_q99','global_margin_q95','global_margin_q99','global_advantage_q95','global_background_insufficient']:
                        row['reference_'+k]=''
                    if refvalid:
                        rv,rpx,rpy=peak(logit[ry0:ry1,rx0:rx1],rx0,ry0)
                        refmask=excluded.copy()
                        refmask[max(0,ry0-2):min(512,ry1+2),max(0,rx0-2):min(640,rx1+2)]=True
                        rlvals,_=local_tiles(logit,refmask,bw,bh,rx0,ry0,expanded_neighborhood(rx0,ry0,bw,bh))
                        rgvalid=gvalid.copy()
                        # Full-grid quantiles exclude every tile intersecting the reference rectangle.
                        cr0=max(0,ry0//bh);cr1=min(rgvalid.shape[0],math.ceil(ry1/bh))
                        cc0=max(0,rx0//bw);cc1=min(rgvalid.shape[1],math.ceil(rx1/bw))
                        rgvalid[cr0:cr1,cc0:cc1]=False
                        rgvals=np.sort(gscores[rgvalid].astype(np.float64))
                        row.update({'reference_max_logit':rv,'reference_peak_x':rpx,'reference_peak_y':rpy,
                            'reference_peak_probability':float(prob[int(rpy-.5),int(rpx-.5)]),
                            'reference_box_max_probability':float(prob[ry0:ry1,rx0:rx1].max())})
                        for pref,vals in [('local',rlvals),('global',rgvals)]:
                            row['reference_'+pref+'_valid_blocks']=len(vals)
                            row['reference_'+pref+'_background_insufficient']=int(len(vals)<20)
                            if len(vals):
                                q95,q99=np.quantile(vals,[.95,.99])
                                row.update({'reference_'+pref+'_q95':float(q95),'reference_'+pref+'_q99':float(q99),
                                    'reference_'+pref+'_margin_q95':float(rv-q95),'reference_'+pref+'_margin_q99':float(rv-q99),
                                    'reference_'+pref+'_advantage_q95':int(rv>q95)})
                    allrows.append(row)
                if frame%40==39:print(json.dumps({'sequence':seq,'plan':plan,'frame':frame,'rows':len(allrows),'seconds':time.time()-started}),flush=True)
            validation.append({'sequence':seq,'arrangement':plan,'logits_sha256':sha(logpath),'probabilities_sha256':sha(probpath),
                'score_npz_sha256':sha(scorepath),'annotations_sha256':sha(annpath),'all_box_probabilities_identical_to_original_scoring':True,
                'all_point_matches_identical_to_frozen_support_table':True})
            del probs,logits
    allrows.sort(key=lambda r:(r['sequence'],r['frame'],r['track_id'],r['arrangement']))
    assert len(allrows)==10400
    assert len({(r['sequence'],r['frame'],r['annotation_id'],r['arrangement']) for r in allrows})==10400
    for plan in PLANS:
        rr=[r for r in allrows if r['arrangement']==plan]
        assert sum(r['matched_at_05'] for r in rr)==(700 if plan=='O' else 720)
        assert sum(r['all_windows_lt_05'] for r in rr)==4342
        assert sum(r['all_windows_lt_001'] for r in rr)==4170
        assert sum(r['both_fixed_matched'] for r in rr)==646
        assert sum(r['extra_window_only'] for r in rr)==104
    csvout(target,allrows);csvout(OUT/'SCORE_SEPARATION.csv',summary(allrows));csvout(OUT/'REFERENCE_SELECTION.csv',reference_info)
    jsonout(OUT/'COLUMN_SCHEMA.json',{'columns':list(allrows[0]),'table1_columns':list(summary(allrows)[0])})
    jsonout(OUT/'VALIDATION.json',{'status':'PASS','rows':len(allrows),'row_keys_unique':True,'comparisons':validation,
        'protocol_sha256':sha(RUN/'PROTOCOL.md'),'script_sha256':sha(__file__),'elapsed_seconds':time.time()-started})
    print(json.dumps({'complete':True,'rows':len(allrows),'seconds':time.time()-started}),flush=True)

if __name__=='__main__':main()
