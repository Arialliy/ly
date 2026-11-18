#!/usr/bin/env python3
"""Frozen one-window-at-a-time path, fixed .5, no inference or threshold search."""
import csv,json,hashlib,importlib.util,time
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
OUT=Path(__file__).resolve().parent;F=OUT.parent;Q=F.parent;P=Q/'stage1_goal';V=Q/'problem_validity_20261011'
PLANS={'O':[0,36,72,108,120],'I1':[0,36,72,96,120],'I2':[0,36,60,96,120],'S':[0,24,60,96,120]}
SEQS=['DJI_0026_2','DJI_0067_2'];SOURCES={}
FIELDS=['gt_count','matched_count','component_count','unmatched_components','positive_pixels','positive_pixels_outside_boxes']

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    SOURCES[str(p)]=h.hexdigest();return h.hexdigest()
def readcsv(p):sha(p);return list(csv.DictReader(p.open()))
def module(p,name):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def dumpcsv(name,rows):
    with (OUT/name).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def j(v):return json.dumps(v,separators=(',',':'))

def main():
    if (OUT/'SUMMARY.json').exists():raise FileExistsError('Complete output already exists')
    began=time.perf_counter();sha(Path(__file__));sha(F/'ONE_REPLACEMENT_PROTOCOL.md');sha(F/'ONE_REPLACEMENT_FREEZE.json')
    scorer=module(P/'pilot/score_pilot.py','scorer');helper,hp=scorer.helper();sha(hp)
    windows=defaultdict(dict)
    for w in readcsv(V/'WINDOWS.csv'):
        if w['sequence'] in SEQS:windows[(w['sequence'],int(w['frame']),int(w['annotation_id']))][int(w['window_start'])]=w
    result={};objects=[];observations=[];integrity=[]
    for seq in SEQS:
        path=P/'data'/seq/'annotations.json';sha(path);gt,meta,offsets,exc=scorer.gt_data(path,helper)
        small=np.array([bool(m['strict_small']) for m in meta]);n=len(meta)
        frames=[int(Path(m['file_name']).stem) for m in json.loads(path.read_text())['images']]
        cache={}
        for s in sorted(set(sum(PLANS.values(),[]))):
            path=P/'predictions/official'/seq/'windows'/f'window_{s:03d}.npz';sha(path)
            with np.load(path,allow_pickle=False) as z:cache[s]=z['probabilities'];assert int(z['start'])==s
        arrays=dict(matched=np.zeros((4,n),np.uint8),box_max=np.zeros((4,n),np.float32),peak_x=np.full((4,n),np.nan,np.float32),
            peak_y=np.full((4,n),np.nan,np.float32),component_area=np.zeros((4,n),np.int32),
            frame_counts=np.zeros((4,160,6),np.int64),coverage=np.zeros((4,160),np.int16),small=small,offsets=offsets)
        for pi,(plan,starts) in enumerate(PLANS.items()):
            prob=np.zeros((160,512,640),np.float32)
            for s in starts:prob[s:s+40]=np.maximum(prob[s:s+40],cache[s]);arrays['coverage'][pi,s:s+40]+=1
            assert (arrays['coverage'][pi]>0).all() and arrays['coverage'][pi].sum()==200
            for t in range(160):
                rr,stats,_=helper.match_frame(prob[t],gt[t],.5)
                for index,r in enumerate(rr,int(offsets[t])):
                    arrays['matched'][pi,index]=r['matched'];arrays['box_max'][pi,index]=r['box_max']
                    if r['matched']:
                        for k in ['peak_x','peak_y','component_area']:arrays[k][pi,index]=r[k]
                arrays['frame_counts'][pi,t]=[stats[k] for k in FIELDS]
            if plan in ['O','S']:
                d=P/'evaluation/official'/seq if plan=='O' else V/'simple_controls/symmetric5_probmax'/seq
                prefix='official_fused' if plan=='O' else 'fused'
                p=d/(prefix+'_matches.npz');sha(p)
                with np.load(p,allow_pickle=False) as z:
                    ti=z['thresholds'].tolist().index(.5)
                    for k in ['matched','peak_x','peak_y','component_area','frame_counts']:
                        assert np.array_equal(arrays[k][pi],z[k][ti],equal_nan=True),(seq,plan,k)
                    assert np.array_equal(arrays['box_max'][pi],z['box_max'],equal_nan=True)
                    assert np.array_equal(small,z['strict_small'])
                integrity.append(dict(sequence=seq,plan=plan,all_matching_and_peak_arrays_equal_frozen_endpoint=True))
            print(j(dict(sequence=seq,plan=plan,small_matched=int(arrays['matched'][pi,small].sum()),counts=arrays['frame_counts'][pi].sum(0).tolist(),seconds=time.perf_counter()-began)),flush=True)
            del prob
        np.savez_compressed(OUT/(seq+'_scores.npz'),**arrays,plans=np.array(list(PLANS)),threshold=np.array(.5))
        for index,m in enumerate(meta):
            ww=windows[(seq,m['frame'],m['annotation_id'])]
            row=dict(sequence=seq,frame=m['frame'],source_frame=m['source_frame'],annotation_id=m['annotation_id'],track_id=m['track_id'],
                small_geometry=m['strict_small'],four_states=''.join(map(str,arrays['matched'][:,index].tolist())))
            for pi,(plan,starts) in enumerate(PLANS.items()):
                used=[s for s in starts if s in ww];success=[s for s in used if int(ww[s]['matched'])]
                row.update({plan+'_'+k:v for k,v in dict(matched=int(arrays['matched'][pi,index]),box_max=float(arrays['box_max'][pi,index]),
                    selected_starts=j(used),successful_starts=j(success),slots=j([m['frame']-s for s in used]),coverage=len(used),
                    earliest_access_source=frames[min(used)],latest_access_source=frames[max(used)+39]).items()})
            objects.append(row)
            if m['strict_small']:
                for ai,bi in [(0,1),(1,2),(2,3)]:
                    am,bm=int(arrays['matched'][ai,index]),int(arrays['matched'][bi,index]);a,b=list(PLANS)[ai],list(PLANS)[bi]
                    if am==bm:continue
                    before=json.loads(row[a+'_successful_starts']);after=json.loads(row[b+'_successful_starts'])
                    observations.append(dict(sequence=seq,frame=m['frame'],source_frame=m['source_frame'],annotation_id=m['annotation_id'],track_id=m['track_id'],
                        step=a+'->'+b,change='gain' if bm else 'loss',before_success_starts=j(before),after_success_starts=j(after),
                        removed_success_starts=j(sorted(set(before)-set(after))),added_success_starts=j(sorted(set(after)-set(before))),
                        before_any_window_match=int(bool(before)),after_any_window_match=int(bool(after)),
                        before_box_max=float(arrays['box_max'][ai,index]),after_box_max=float(arrays['box_max'][bi,index])))
        result[seq]=arrays
    metrics=[]
    for group,seqs in [('pooled',SEQS)]+[(s,[s]) for s in SEQS]:
        for pi,plan in enumerate(PLANS):
            totals=sum((result[s]['frame_counts'][pi].sum(0) for s in seqs),np.zeros(6,np.int64))
            for reference,ri in [('O',0)]+([(list(PLANS)[pi-1],pi-1)] if pi>1 else []):
                gain=loss=matched=smalln=0
                for s in seqs:
                    z=result[s];mm=z['matched'][pi].astype(bool);before=z['matched'][ri].astype(bool);eligible=z['small']
                    gain+=int((eligible&mm&~before).sum());loss+=int((eligible&~mm&before).sum());matched+=int(mm[eligible].sum());smalln+=int(eligible.sum())
                metrics.append(dict(group=group,plan=plan,reference=reference,frames=len(seqs)*160,small_gt=smalln,small_matched=matched,gained=gain,lost=loss,net=gain-loss,
                    **{k:int(v) for k,v in zip(FIELDS,totals)}))
    dumpcsv('METRICS.csv',metrics);dumpcsv('OBJECTS.csv',objects);dumpcsv('STEP_CHANGES.csv',observations)
    summary=dict(status='COMPLETE',threshold=.5,plans=PLANS,path=['O','I1','I2','S'],all_steps_have_full_160_frame_coverage=True,
        per_sequence_window_frame_budget=200,all_valid_objects=len(objects),small_objects=sum(r['small_geometry'] for r in objects),
        metrics=metrics,endpoint_checks=integrity,small_state_patterns=dict(Counter(r['four_states'] for r in objects if r['small_geometry'])),
        elapsed_seconds=time.perf_counter()-began,sources=SOURCES,
        limitations=['Exploratory fixed path on previously used development material','Marginal effects depend on other four windows and path order',
                    'Changing start changes both context and current-frame slot','Diagnostic point matching and pixel/component counts, not paper Pd/Fa'])
    (OUT/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))

if __name__=='__main__':main()
