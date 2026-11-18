#!/usr/bin/env python3
"""Cross-agent audit from preceding window/object tables; only five raw traces."""
from pathlib import Path
import collections,csv,hashlib,json,time
import cv2
import numpy as np

OUT=Path(__file__).resolve().parent;F=OUT.parent;Q=F.parent;V=Q/'problem_validity_20261011';P=Q/'stage1_goal'
OFF={0,36,72,108,120};SYM={0,24,60,96,120};SEQS={'DJI_0026_2','DJI_0067_2'}
def read(p):
    with p.open() as f:return list(csv.DictReader(f))
def key(r):return r['sequence'],int(r['frame']),int(r['annotation_id'])
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def events(rows):
    tracks=collections.defaultdict(list)
    for r in rows:tracks[(r['sequence'],int(r['track_id']))].append(r)
    ev=[]
    for (seq,tid),rr in sorted(tracks.items()):
        current=[]
        for r in sorted(rr,key=lambda r:int(r['source_frame'])):
            if current and int(r['source_frame'])!=int(current[-1]['source_frame'])+1:
                ev.append((seq,tid,int(current[0]['source_frame']),int(current[-1]['source_frame']),tuple(int(v['annotation_id']) for v in current)));current=[]
            current.append(r)
        if current:ev.append((seq,tid,int(current[0]['source_frame']),int(current[-1]['source_frame']),tuple(int(v['annotation_id']) for v in current)))
    return sorted(ev)
def main():
    begin=time.time()
    base={key(r):r for r in read(V/'OBJECTS.csv') if r['sequence'] in SEQS and r['small_geometry']=='1'};assert len(base)==5200
    ctrl={key(r):r for r in read(V/'control_summary/OBJECTS.csv') if r['mode']=='symmetric5_probmax' and key(r) in base}
    stages={key(r):r for r in read(F/'stages/ERROR_STAGE_TABLE.csv')};assert set(stages)==set(base)
    cond={key(r):r for r in read(F/'conditions/OBJECT_CONDITIONS.csv')};assert set(cond)==set(base)
    windows=collections.defaultdict(list)
    for w in read(V/'WINDOWS.csv'):
        if key(w) in base:windows[key(w)].append(w)
    groups=collections.defaultdict(list);join=[]
    for k,r in base.items():
        cc=ctrl[k];ss=stages[k];ww=windows[k]
        assert sorted(int(w['window_start']) for w in ww)==[s for s in range(0,121,12) if s<=k[1]<s+40]
        any11=any(int(w['matched']) for w in ww)
        assert any11==bool(int(r['any11_matched']))
        maxresp=max(float(w['box_max']) for w in ww);assert maxresp==float(r['context_box_max'])==float(cond[k]['diagnostic_box_max'])
        origsuccess={int(w['window_start']) for w in ww if int(w['matched']) and int(w['window_start']) in OFF}
        symsuccess={int(w['window_start']) for w in ww if int(w['matched']) and int(w['window_start']) in SYM}
        assert sorted(origsuccess)==json.loads(ss['official_success_starts'])
        assert sorted(symsuccess)==json.loads(ss['symmetric_success_starts'])
        om,sm=map(int,(cc['official_matched'],cc['control_matched']))
        assert om==int(r['fused_matched'])==int(ss['official_matched']) and sm==int(ss['symmetric_matched'])
        group='common_success' if om and sm else 'gain' if sm else 'loss' if om else 'all11_failure' if not any11 else 'both_failure_with_window_opportunity'
        assert cond[k]['group']==group
        groups[group].append(r)
        if not om and not sm:groups['both_failure'].append(r)
        if r['system_case']=='original_window_match_lost_by_fusion':groups['original5_fusion_loss'].append(r)
        if group=='loss':
            assert origsuccess and not symsuccess and not (origsuccess&SYM)
            assert int(ss['official_success_all_removed_by_symmetric'])==1
        if group=='gain':
            assert symsuccess
            if not (symsuccess-OFF):assert r['system_case']=='original_window_match_lost_by_fusion'
        join.append(dict(sequence=k[0],frame=k[1],source_frame=r['source_frame'],annotation_id=k[2],track_id=r['track_id'],category=r['category'],group=group,window_count=len(ww),original_success_starts=json.dumps(sorted(origsuccess)),symmetric_success_starts=json.dumps(sorted(symsuccess)),max_all_available_diagnostic_box_response=maxresp,official_fused_box_max=float(ss['official_fused_box_max']),symmetric_fused_box_max=float(ss['symmetric_fused_box_max']),all_original_success_windows_removed=bool(origsuccess) and not bool(origsuccess&SYM),symmetric_has_added_success_window=bool(symsuccess-OFF)))
    jr={key(r):r for r in join}
    loss=[jr[key(r)] for r in groups['loss']];gain=[jr[key(r)] for r in groups['gain']]
    assert len(loss)==54 and sum(r['symmetric_fused_box_max']<.5 for r in loss)==47
    assert len(gain)==74 and sum(r['symmetric_has_added_success_window'] for r in gain)==73
    assert len(groups['all11_failure'])==4391 and len(groups['both_failure'])==4426
    counts=[];indevents=[]
    ce=read(F/'conditions/OBSERVATION_EVENTS.csv');se=read(F/'stages/OBSERVATION_EVENTS.csv')
    group_counts={r['group']:r for r in read(F/'conditions/GROUP_COUNTS.csv')}
    for group,rr in groups.items():
        ev=events(rr)
        counts.append(dict(group=group,object_frames=len(rr),tracks=len({(r['sequence'],r['track_id']) for r in rr}),sequences=len({r['sequence'] for r in rr}),events=len(ev)))
        if group in group_counts:
            cr=group_counts[group]
            assert len(rr)==int(cr['object_frames']) and len(ev)==int(cr['observation_events'])
            assert len({(r['sequence'],r['track_id']) for r in rr})==int(cr['sequence_tracks'])
            actual=sorted((r['sequence'],int(r['track_id']),int(r['first_source_frame']),int(r['last_source_frame']),tuple(map(int,r['annotation_ids'].split(';')))) for r in ce if r['group']==group)
            assert actual==ev,group
        stagegroup={'common_success':'both_success','original5_fusion_loss':'original_fusion_loss'}.get(group,group)
        stageev=[r for r in se if r['cohort']==stagegroup]
        if stageev:
            actual=sorted((r['sequence'],int(r['track_id']),int(r['source_start']),int(r['source_end']),tuple(json.loads(r['annotation_ids']))) for r in stageev)
            assert actual==ev,stagegroup
        for seq,tid,start,end,ids in ev:indevents.append(dict(group=group,sequence=seq,track_id=tid,start_source_frame=start,end_source_frame=end,objects=len(ids),annotation_ids=json.dumps(ids)))
    gainloss_tracks={(r['sequence'],r['track_id']) for r in groups['gain']+groups['loss']};assert len(gainloss_tracks)==29
    responses=[];bins=[];allfailure=[jr[key(r)] for r in groups['all11_failure']]
    strata=[('pooled',allfailure)]+[(seq,[r for r in allfailure if r['sequence']==seq]) for seq in sorted(SEQS)]+[('category:'+c,[r for r in allfailure if r['category']==c]) for c in sorted({r['category'] for r in allfailure})]
    for name,rr in strata:
        vals=np.array([r['max_all_available_diagnostic_box_response'] for r in rr])
        q=np.quantile(vals,[0,.25,.5,.75,1])
        for threshold in [.001,.01,.1,.5]:
            below=[r for r in rr if r['max_all_available_diagnostic_box_response']<threshold]
            above=[r for r in rr if r['max_all_available_diagnostic_box_response']>=threshold]
            responses.append(dict(stratum=name,objects=len(rr),tracks=len({(r['sequence'],r['track_id']) for r in rr}),threshold=threshold,strictly_below=len(below),below_fraction=len(below)/len(rr),below_tracks=len({(r['sequence'],r['track_id']) for r in below}),at_or_above=len(above),at_or_above_tracks=len({(r['sequence'],r['track_id']) for r in above}),minimum=q[0],q1=q[1],median=q[2],q3=q[3],maximum=q[4]))
        for lo,hi in [(0,.001),(.001,.01),(.01,.1),(.1,.5),(.5,1.00000001)]:
            rows=[r for r in rr if lo<=r['max_all_available_diagnostic_box_response']<hi]
            bins.append(dict(stratum=name,lower_inclusive=lo,upper_exclusive=hi if hi<=1 else '1 inclusive',objects=len(rows),fraction=len(rows)/len(rr),tracks=len({(r['sequence'],r['track_id']) for r in rows})))
    # Five full-frame raw cases, OpenCV labels and row-major numpy argmax.
    fusionrows=read(F/'stages/FUSION_LOSSES.csv');raw=[]
    for seq in sorted(SEQS):
        pred=P/'predictions/official'/seq
        with np.load(pred/'official_fused.npz',allow_pickle=False) as z:fp=z['probabilities']
        wp={}
        for fr in [r for r in fusionrows if r['sequence']==seq]:
            traces=json.loads(fr['source_trace']);assert len(traces)==1
            tr=traces[0];t=int(tr['frame']);start=int(tr['window_start']);r=base[(seq,t,int(fr['annotation_id']))]
            if start not in wp:
                with np.load(pred/'windows'/f'window_{start:03d}.npz',allow_pickle=False) as z:wp[start]=z['probabilities']
            source=wp[start][t-start];fused=fp[t]
            ns,ls,sa,_=cv2.connectedComponentsWithStats((source>=.5).astype(np.uint8),connectivity=8)
            nf,lf,fa,_=cv2.connectedComponentsWithStats((fused>=.5).astype(np.uint8),connectivity=8)
            sy,sx=int(tr['source_peak_y']),int(tr['source_peak_x']);sid=int(ls[sy,sx]);assert sid>0
            smask=ls==sid;dests=np.unique(lf[smask]);assert len(dests)==1 and dests[0]>0
            fid=int(dests[0]);fmask=lf==fid;oldlabels=np.unique(ls[fmask]);oldlabels=oldlabels[oldlabels>0]
            spi=np.flatnonzero(smask)[np.argmax(source[smask])];fpi=np.flatnonzero(fmask)[np.argmax(fused[fmask])]
            spy,spx=np.unravel_index(spi,source.shape);fpy,fpx=np.unravel_index(fpi,fused.shape)
            assert (spx+.5,spy+.5)==(float(tr['source_peak_x']),float(tr['source_peak_y']))
            assert (fpx+.5,fpy+.5)==(float(tr['fused_peak_x']),float(tr['fused_peak_y']))
            assert int(smask.sum())==int(tr['source_area']) and int(fmask.sum())==int(tr['fused_area'])
            assert len(oldlabels)==int(tr['source_components_in_fused_component'])
            xx,yy,bw,bh=[float(r[k]) for k in ['bbox_x','bbox_y','bbox_w','bbox_h']]
            eligible=0
            for cid in range(1,nf):
                pix=np.flatnonzero(lf==cid);ind=pix[np.argmax(fused.ravel()[pix])];py,px=np.unravel_index(ind,fused.shape)
                eligible+=int(xx<=px+.5<xx+bw and yy<=py+.5<yy+bh)
            assert eligible==0
            raw.append(dict(sequence=seq,frame=t,track_id=r['track_id'],annotation_id=r['annotation_id'],successful_window=start,source_area=int(smask.sum()),fused_area=int(fmask.sum()),old_source_components_in_fused_component=len(oldlabels),direct_merge=int(len(oldlabels)>1),foreground_preserved=1,exact_same_binary_component=int(np.array_equal(smask,fmask)),eligible_fused_peaks=eligible,source_peak_x=spx+.5,source_peak_y=spy+.5,fused_peak_x=fpx+.5,fused_peak_y=fpy+.5,fused_score_minus_preserved_original_peak=float(fused[fpy,fpx]-fused[spy,spx])))
    assert len(raw)==5 and sum(r['direct_merge'] for r in raw)==2
    write('INDEPENDENT_OBJECT_JOIN.csv',join);write('GROUP_EVENT_COUNTS.csv',counts);write('INDEPENDENT_EVENTS.csv',indevents)
    write('ALL_WINDOW_FAILURE_RESPONSE_CDF.csv',responses);write('ALL_WINDOW_FAILURE_RESPONSE_BINS.csv',bins);write('FIVE_RAW_TRACE_CHECK.csv',raw)
    result=dict(status='PASS',all_small_objects=5200,available_object_windows=sum(len(x) for x in windows.values()),group_counts=counts,gain_has_added_success_window=73,gain_original_fusion_repair=1,loss_original_success_windows_all_removed=54,loss_no_symmetric_success_windows=54,loss_new_fused_below_05=47,loss_new_fused_at_or_above_05_no_peak=7,original5_fusion_loss_objects=5,original5_direct_component_merges=2,original5_valid_fused_peak_missing=5,gainloss_union_tracks=29,all_failure_response_cdf=responses,raw_case_checks=raw,elapsed_seconds=time.time()-begin,script_sha256=digest(Path(__file__)),source_hashes={str(p):digest(p) for p in [V/'OBJECTS.csv',V/'WINDOWS.csv',V/'control_summary/OBJECTS.csv',F/'stages/ERROR_STAGE_TABLE.csv',F/'stages/FUSION_LOSSES.csv',F/'conditions/OBJECT_CONDITIONS.csv',F/'conditions/OBSERVATION_EVENTS.csv',F/'conditions/GROUP_COUNTS.csv']})
    (OUT/'AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'PASS','group_counts':counts,'pooled_response':[r for r in responses if r['stratum']=='pooled'],'elapsed_seconds':time.time()-begin}),flush=True)
if __name__=='__main__':main()
