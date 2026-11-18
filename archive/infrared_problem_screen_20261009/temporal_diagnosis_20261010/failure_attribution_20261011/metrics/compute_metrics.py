#!/usr/bin/env python3
"""CPU-only diagnostics, frozen bbox proxies; not paper Pd/Fa."""
from pathlib import Path
import collections,csv,hashlib,importlib.util,json,math,time
import numpy as np
from scipy import ndimage

OUT=Path(__file__).resolve().parent
Q=OUT.parents[1]
V=Q/'problem_validity_20261011'
P=Q/'stage1_goal'
R=Q.parent
SEQS=['DJI_0026_2','DJI_0067_2']
STARTS=[0,36,72,108,120]

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def mod(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def write(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def read(p):
    with Path(p).open() as f:return list(csv.DictReader(f))
def save(name,o):
    (OUT/name).write_text(json.dumps(o,indent=2,allow_nan=False)+'\n')
def center(g,h,w):
    x,y,bw,bh=g['bbox'];xx=math.floor(x+bw/2);yy=math.floor(y+bh/2)
    cx=max(0,min(w-1,xx));cy=max(0,min(h-1,yy))
    return cx,cy,int(cx!=xx or cy!=yy)
def region(cx,cy,r,h,w):return np.s_[max(0,cy-r):min(h,cy+r+1),max(0,cx-r):min(w,cx+r+1)]
def obj_values(prob,gg):
    h,w=prob.shape;result=[]
    for g in gg:
        cx,cy,clamped=center(g,h,w);box=prob[g['crop'][1]:g['crop'][3],g['crop'][0]:g['crop'][2]]
        result.append((int(np.any(prob[region(cx,cy,1,h,w)]>=.5)),int(np.any(box>=.5)),float(prob[region(cx,cy,1,h,w)].max()),float(box.max()) if box.size else None,cx,cy,clamped))
    return result
def area_stats(aa):
    a=np.asarray(aa,dtype=np.int64)
    return dict(count=len(a),sum=int(a.sum()),mean=float(a.mean()) if len(a) else None,
                p25=float(np.quantile(a,.25)) if len(a) else None,median=float(np.median(a)) if len(a) else None,
                p75=float(np.quantile(a,.75)) if len(a) else None,p90=float(np.quantile(a,.9)) if len(a) else None,
                p95=float(np.quantile(a,.95)) if len(a) else None,p99=float(np.quantile(a,.99)) if len(a) else None,
                maximum=int(a.max()) if len(a) else None)

def main():
    begin=time.time();freeze=json.loads((OUT/'PROTOCOL_FREEZE.json').read_text())
    assert freeze['protocol_sha256']==sha(OUT/'ADAPTED_METRIC_PROTOCOL.md')
    if (OUT/'SUMMARY.json').exists():raise FileExistsError('completed output')
    scorer=mod('prior_score',P/'pilot/score_pilot.py');matcher,_=scorer.helper()
    official=mod('official_shooting',R/'official_deeppro/ShootingRules.py').ShootingRules()
    controls={(r['sequence'],int(r['frame']),int(r['annotation_id'])):r for r in read(V/'control_summary/OBJECTS.csv') if r['mode']=='symmetric5_probmax'}
    lineage={(r['sequence'],int(r['frame']),int(r['annotation_id'])):r for r in read(V/'OBJECTS.csv') if r['sequence'] in SEQS}
    objs=[];frames=[];compsall=[];windows=[];monotonic=[];inputs={};adapter_checks=[]
    for seq in SEQS:
        gt,meta,offsets,_=scorer.gt_data(P/'data'/seq/'annotations.json',matcher)
        inputpaths={'official':P/'predictions/official'/seq/'official_fused.npz',
                    'symmetric':V/'simple_controls/symmetric5_probmax'/seq/'fused.npz'}
        loaded={}
        objlookup={}
        for arrangement,path in inputpaths.items():
            inputs[str(path)]=sha(path)
            with np.load(path,allow_pickle=False) as z:prob=z['probabilities']
            assert prob.shape==(160,512,640) and np.isfinite(prob).all()
            loaded[arrangement]=prob
            for t,gg in enumerate(gt):
                rows,stats,comps=matcher.match_frame(prob[t],gg,.5)
                _,labels=matcher.components(prob[t],.5)
                positive=prob[t]>=.5;h,w=positive.shape
                outside=np.ones((h,w),bool);center_bg=np.ones((h,w),bool);raw_bg=np.ones((h,w),bool);synthetic=np.zeros((h,w),np.uint8)
                vals=obj_values(prob[t],gg)
                for g,val in zip(gg,vals):
                    cx,cy=val[4:6];synthetic[cy,cx]=1
                    center_bg[region(cx,cy,4,h,w)]=False
                    raw_bg[cy-4:cy+5,cx-4:cx+5]=False
                    x0,y0,x1,y1=g['crop'];outside[y0:y1,x0:x1]=False
                synlabels,syntgt=ndimage.label(synthetic,structure=np.ones((3,3),np.uint8))
                synhits=set()
                for yy,xx in np.argwhere(synthetic):
                    if positive[yy-1:yy+2,xx-1:xx+2].any():synhits.add(int(synlabels[yy,xx]))
                raw_expected=(int((positive&raw_bg).sum()),len(synhits),int(syntgt))
                raw_actual=tuple(map(int,official(prob[t][None],synthetic[None],.5)))
                assert raw_expected==raw_actual,(seq,arrangement,t,raw_expected,raw_actual)
                adapter_checks.append(dict(sequence=seq,arrangement=arrangement,frame=t,official_call_equals_independent=True))
                row=dict(sequence=seq,arrangement=arrangement,frame=t,source_frame=meta[int(offsets[t])]['source_frame'],**stats,
                         all_pixels=h*w,center_proxy_hits=sum(v[0] for v in vals),box_pixel_hits=sum(v[1] for v in vals),
                         center_proxy_background_pixels=int((positive&center_bg).sum()),
                         synthetic_official_background_pixels=raw_actual[0],synthetic_official_true_regions=raw_actual[1],synthetic_official_target_regions=raw_actual[2],
                         synthetic_center_unique_pixels=int(synthetic.sum()),synthetic_center_collisions=len(gg)-int(synthetic.sum()),
                         synthetic_center_region_merges=int(synthetic.sum())-syntgt,
                         clipped_vs_raw_background_difference=int((positive&center_bg).sum())-raw_actual[0])
                frames.append(row)
                for g,mr,val in zip(gg,rows,vals):
                    key=(seq,t,g['id']);prior=controls[key];line=lineage[key]
                    assert mr['matched']==int(prior['official_matched' if arrangement=='official' else 'control_matched'])
                    r=dict(sequence=seq,arrangement=arrangement,frame=t,source_frame=line['source_frame'],annotation_id=g['id'],track_id=g['track_id'],category_id=g['category_id'],
                           small_geometry=int(prior['small_geometry']),point_group=('gain' if int(prior['gain']) else 'loss' if int(prior['loss']) else 'both_success' if int(prior['official_matched']) else 'both_failure'),
                           original_system_case=line['system_case'],point_matched=mr['matched'],center_proxy_hit=val[0],box_pixel_hit=val[1],center_neighborhood_max=val[2],box_max=val[3],center_x=val[4],center_y=val[5],center_clamped=val[6])
                    objs.append(r);objlookup[(arrangement,t,g['id'])]=r
                outside_counts=np.bincount(labels[positive&outside],minlength=len(comps)+1)
                for c in comps:
                    compsall.append(dict(sequence=seq,arrangement=arrangement,frame=t,component_id=c['component_id'],area=c['area'],outside_box_pixels=int(outside_counts[c['component_id']]),unmatched_component=c['unmatched_component'],assigned_track_id=c['assigned_track_id'],peak_x=c['peak_x'],peak_y=c['peak_y']))
            print(json.dumps({'sequence':seq,'arrangement':arrangement,'frames':160,'seconds':time.time()-begin}),flush=True)
        anycenter=np.zeros(len(meta),bool);anybox=np.zeros(len(meta),bool)
        for start in STARTS:
            path=P/'predictions/official'/seq/'windows'/f'window_{start:03d}.npz'
            inputs[str(path)]=sha(path)
            with np.load(path,allow_pickle=False) as z:wp=z['probabilities']
            fused=loaded['official'][start:start+40]
            violations=int(np.count_nonzero(fused<wp));assert violations==0
            monotonic.append(dict(sequence=seq,start=start,pixels_checked=int(wp.size),fused_less_than_window_pixels=violations))
            for t in range(start,start+40):
                vals=obj_values(wp[t-start],gt[t]);a,b=map(int,offsets[t:t+2])
                anycenter[a:b]|=np.array([v[0] for v in vals],bool);anybox[a:b]|=np.array([v[1] for v in vals],bool)
                for g,val in zip(gt[t],vals):
                    line=lineage[(seq,t,g['id'])]
                    if line['small_geometry']=='1' and line['system_case']=='original_window_match_lost_by_fusion':
                        windows.append(dict(sequence=seq,frame=t,source_frame=line['source_frame'],annotation_id=g['id'],track_id=g['track_id'],window_start=start,slot=t-start,center_proxy_hit=val[0],box_pixel_hit=val[1],center_neighborhood_max=val[2],box_max=val[3]))
            del wp,fused
        for j,m in enumerate(meta):
            ro=objlookup[('official',m['frame'],m['annotation_id'])]
            ro['any_original5_center_proxy_hit']=int(anycenter[j]);ro['any_original5_box_pixel_hit']=int(anybox[j])
            assert not anycenter[j] or ro['center_proxy_hit']
            assert not anybox[j] or ro['box_pixel_hit']
            # Max and any commute for a fixed per-object region.
            assert int(anycenter[j])==ro['center_proxy_hit'] and int(anybox[j])==ro['box_pixel_hit']
        del loaded
    # Symmetric rows have no original5 any fields; the value belongs to baseline.
    for o in objs:
        o.setdefault('any_original5_center_proxy_hit','');o.setdefault('any_original5_box_pixel_hit','')
    totals=[];areas=[];exchanges=[]
    for seq in SEQS+['pooled']:
        for arrangement in ['official','symmetric']:
            ff=[r for r in frames if r['arrangement']==arrangement and (seq=='pooled' or r['sequence']==seq)]
            oo=[r for r in objs if r['arrangement']==arrangement and (seq=='pooled' or r['sequence']==seq)]
            cc=[r for r in compsall if r['arrangement']==arrangement and (seq=='pooled' or r['sequence']==seq)]
            summed={k:sum(r[k] for r in ff) for k in ff[0] if k not in ['sequence','arrangement','frame','source_frame']}
            totals.append(dict(sequence=seq,arrangement=arrangement,frames=len(ff),**summed,small_objects=sum(r['small_geometry'] for r in oo),small_point_matches=sum(r['point_matched'] for r in oo if r['small_geometry']),small_center_proxy_hits=sum(r['center_proxy_hit'] for r in oo if r['small_geometry']),small_box_pixel_hits=sum(r['box_pixel_hit'] for r in oo if r['small_geometry']),center_proxy_background_fraction=summed['center_proxy_background_pixels']/summed['all_pixels']))
            for group in ['all','matched','unmatched']:
                selected=[c for c in cc if group=='all' or (c['unmatched_component']==(group=='unmatched'))]
                areas.append(dict(sequence=seq,arrangement=arrangement,group=group,**area_stats([r['area'] for r in selected]),outside_box_pixels=sum(r['outside_box_pixels'] for r in selected)))
        for point_group in ['gain','loss','both_success','both_failure','original5_point_fusion_loss','all_small']:
            old=[o for o in objs if o['arrangement']=='official' and o['small_geometry'] and (seq=='pooled' or o['sequence']==seq) and (point_group=='all_small' or o['point_group']==point_group or (point_group=='original5_point_fusion_loss' and o['original_system_case']=='original_window_match_lost_by_fusion'))]
            bykey={(o['sequence'],o['frame'],o['annotation_id']):o for o in objs if o['arrangement']=='symmetric'}
            for metric in ['point_matched','center_proxy_hit','box_pixel_hit']:
                c=collections.Counter((o[metric],bykey[(o['sequence'],o['frame'],o['annotation_id'])][metric]) for o in old)
                exchanges.append(dict(sequence=seq,point_group=point_group,metric=metric,objects=len(old),both_hit=c[(1,1)],gain=c[(0,1)],loss=c[(1,0)],both_miss=c[(0,0)]))
    pooled={r['arrangement']:r for r in totals if r['sequence']=='pooled'}
    assert pooled['official']['unmatched_components']==4015 and pooled['symmetric']['unmatched_components']==3967
    assert pooled['official']['positive_pixels_outside_boxes']==38368 and pooled['symmetric']['positive_pixels_outside_boxes']==38641
    assert pooled['official']['small_point_matches']==700 and pooled['symmetric']['small_point_matches']==720
    write('FRAME_METRICS.csv',frames);write('OBJECT_METRICS.csv',objs);write('COMPONENTS.csv',compsall)
    write('TOTALS.csv',totals);write('COMPONENT_AREA_SUMMARY.csv',areas);write('POINT_GROUP_METRIC_EXCHANGES.csv',exchanges)
    write('FIVE_CASE_WINDOW_PROXY.csv',windows);write('MAX_MONOTONICITY_CHECK.csv',monotonic)
    five=[r for r in objs if r['small_geometry'] and r['original_system_case']=='original_window_match_lost_by_fusion'];assert len(five)==10
    write('FIVE_CASE_FUSION_PROXY.csv',five)
    save('SUMMARY.json',dict(status='COMPLETE',protocol_sha256=freeze['protocol_sha256'],script_sha256=sha(__file__),inputs_sha256=inputs,official_synthetic_adapter_frames_checked=len(adapter_checks),adapter_disagreements=0,original5_max_pixel_checks=sum(r['pixels_checked'] for r in monotonic),max_pixel_violations=0,fixed_region_any_window_vs_fusion_hit_disagreements=0,totals=totals,component_areas=areas,exchanges=exchanges,elapsed_seconds=time.time()-begin,official_Pd_Fa_reproduced=False,reason='Author evaluation segmentation images unavailable; all label adaptations explicitly synthetic bbox proxies.'))
    print(json.dumps({'status':'COMPLETE','elapsed_seconds':time.time()-begin}),flush=True)

if __name__=='__main__':main()
