#!/usr/bin/env python3
"""Join original-checkpoint window detections and official system detections.

All annotated observations are retained; original score-selected cohorts are
secondary strata only. No per-object oracle is a deployable method.
"""
import csv
import hashlib
import importlib.util
import json
from collections import Counter,defaultdict
from pathlib import Path

import numpy as np

V=Path(__file__).resolve().parent
Q=V.parent
P=Q/'stage1_goal'
OLD=['DJI_0028_2','DJI_0028_4','DJI_0032_5','DJI_0047_3']
NEW=['DJI_0026_2','DJI_0067_2']
ORIGINAL={0,36,72,108,120}
CASES=['fused_and_window_match','fused_only_match','original_window_match_lost_by_fusion',
       'extra_window_only_opportunity','no_match_but_box_response','no_match_all_boxes_low']
SOURCES={}


def fingerprint(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    SOURCES[str(p)]=h.hexdigest()


def read_csv(p):
    fingerprint(p)
    with p.open() as f:return list(csv.DictReader(f))


def read_json(p):
    fingerprint(p);return json.loads(p.read_text())


def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def k(row):return int(row.get('local_frame',row.get('frame'))),int(row['annotation_id'])


def old_record(r):
    return dict(frame=int(r['local_frame']),source_frame=int(r['source_frame']),
        annotation_id=int(r['annotation_id']),track_id=int(r['track_id']),
        bbox=[float(r['bbox_'+a]) for a in ['x','y','w','h']],matched=int(r['matched']),
        box_max=float(r['box_max']) if r['box_max'] else 0.,
        peak_x=float(r['peak_x']) if r['peak_x'] else None,
        peak_y=float(r['peak_y']) if r['peak_y'] else None,
        association_ambiguous=int(r['association_ambiguous']))


def score_records(directory,name):
    meta=read_json(directory/(name+'_COMPLETE.json'))
    path=directory/(name+'_matches.npz');fingerprint(path)
    assert meta['status']=='COMPLETE' and meta['name']==name
    with np.load(path,allow_pickle=False) as z:
        assert meta['thresholds']==z['thresholds'].tolist()
        t=z['thresholds'].tolist().index(.5)
        arrays={name:z[name] for name in ['matched','box_max','peak_x','peak_y']}
        result={}
        for i,o in enumerate(meta['objects']):
            key=k(o);assert key not in result
            result[key]=dict(o,matched=int(arrays['matched'][t,i]),box_max=float(arrays['box_max'][i]),
                peak_x=float(arrays['peak_x'][t,i]) if arrays['matched'][t,i] else None,
                peak_y=float(arrays['peak_y'][t,i]) if arrays['matched'][t,i] else None)
    return result


def check(meta,row):
    for a in ['frame','source_frame','annotation_id','track_id','bbox']:
        assert meta[a]==row[a],(a,meta,row)


def summarize(rows):
    counts={c:sum(r['system_case']==c for r in rows) for c in CASES}
    n=len(rows);miss=sum(not r['fused_matched'] for r in rows)
    assert sum(counts.values())==n
    out=dict(n=n,tracks=len({(r['sequence'],r['track_id']) for r in rows}),
        fused_matched=n-miss,fused_unmatched=miss,
        any11_matched=sum(r['any11_matched'] for r in rows),
        any5_matched=sum(r['any5_matched'] for r in rows),**counts)
    out.update(any11_yes_fused_no=counts[CASES[2]]+counts[CASES[3]],
        any11_no_fused_no=counts[CASES[4]]+counts[CASES[5]],
        final_detected_despite_response_cross=sum(r['fused_matched'] and r['response_cross_05'] for r in rows),
        final_detected_despite_window_match_disagreement=sum(r['fused_matched'] and r['match_disagreement'] for r in rows))
    out['candidate_opportunity_fraction_all']=out['any11_yes_fused_no']/n if n else None
    out['candidate_opportunity_fraction_final_miss']=out['any11_yes_fused_no']/miss if miss else None
    out['fusion_loss_tracks']=len({(r['sequence'],r['track_id']) for r in rows if r['system_case']==CASES[2]})
    out['extra_opportunity_tracks']=len({(r['sequence'],r['track_id']) for r in rows if r['system_case']==CASES[3]})
    return out


def write_csv(path,rows):
    assert rows
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    if (V/'SUMMARY.json').exists():raise FileExistsError('Do not silently replace complete join')
    scorer=load_module(P/'pilot/score_pilot.py','existing_scorer');helper,_=scorer.helper()
    selected=read_csv(P/'local_branch_probe/full/selected_pairs.csv')
    historic={(r['sequence'],*k(r)):r for r in selected};assert len(historic)==1417
    output=[];windows=[]
    for seq in OLD+NEW:
        group='prior_motion_selected_development' if seq in OLD else 'additional_annotation_selected_development'
        ad=P/('data_diagnostics' if seq in OLD else 'data')/seq/'annotations.json'
        data=read_json(ad);cats={c['id']:c['name'] for c in data['categories']}
        _,objects,_,_=scorer.gt_data(ad,helper)
        if seq in OLD:
            fr=[r for r in read_csv(Q/'analysis'/seq/'per_track_frame.csv') if float(r['threshold'])==.5]
            cr=[r for r in read_csv(Q/'window_offset_experiment/analysis'/seq/'per_context_all_tracks.csv') if float(r['threshold'])==.5]
            fused={k(r):old_record(r) for r in fr};assert len(fused)==len(fr)
            context={}
            for r in cr:
                key=(int(r['window_start']),*k(r));assert key not in context
                context[key]=old_record(r)
        else:
            fused=score_records(P/'evaluation/official'/seq,'official_fused')
            context={}
            for start in range(0,121,12):
                for key,r in score_records(V/'window_scores'/seq,f'window_{start:03d}').items():
                    context[(start,*key)]=r
        assert len(objects)==len(fused) and {k(o) for o in objects}==set(fused)
        for obj in objects:
            key=k(obj);f=fused[key];check(obj,f)
            starts=[s for s in range(0,121,12) if s<=obj['frame']<s+40]
            rr=[context[(s,*key)] for s in starts]
            for r in rr:check(obj,r)
            matching=[s for s,r in zip(starts,rr) if r['matched']]
            official_matching=[s for s in matching if s in ORIGINAL]
            extra_matching=[s for s in matching if s not in ORIGINAL]
            values=[r['box_max'] for r in rr]
            assert all(np.isfinite(values))
            fm=bool(f['matched']);a=bool(matching);b=bool(official_matching)
            if fm:case=CASES[0] if a else CASES[1]
            elif b:case=CASES[2]
            elif a:case=CASES[3]
            else:case=CASES[4] if max(values)>=.5 else CASES[5]
            if b:assert f['box_max']>=.5 # Pixelwise max cannot suppress an existing above-threshold pixel.
            h=historic.get((seq,*key));strong=bool(h and int(h['in_strong_97']))
            rec=dict(group=group,sequence=seq,frame=obj['frame'],source_frame=obj['source_frame'],
                annotation_id=obj['annotation_id'],track_id=obj['track_id'],category_id=obj['category_id'],
                category=cats[obj['category_id']],bbox_x=obj['bbox'][0],bbox_y=obj['bbox'][1],
                bbox_w=obj['bbox'][2],bbox_h=obj['bbox'][3],max_side=max(obj['bbox'][2:]),
                small_geometry=int(obj['strict_small']),multi_context=int(len(starts)>=2),
                historic1417=int(h is not None),historic97=int(strong),
                window_count=len(starts),official_window_count=sum(s in ORIGINAL for s in starts),
                fused_matched=int(fm),any11_matched=int(a),any5_matched=int(b),
                fused_box_max=f['box_max'],context_box_min=min(values),context_box_max=max(values),
                response_cross_05=int(min(values)<.5<=max(values)),strong_response_flip=int(min(values)<.1 and max(values)>=.9),
                match_disagreement=int(0<len(matching)<len(starts)),system_case=case,
                successful_official_starts='|'.join(map(str,official_matching)),
                successful_extra_starts='|'.join(map(str,extra_matching)),
                successful_slots='|'.join(str(obj['frame']-s) for s in matching),
                any_success_slot_in_8_to_31=int(any(8<=obj['frame']-s<=31 for s in matching)),
                any_success_slot_outside_8_to_31=int(any(not 8<=obj['frame']-s<=31 for s in matching)))
            output.append(rec)
            for s,r in zip(starts,rr):
                windows.append(dict(group=group,sequence=seq,frame=obj['frame'],source_frame=obj['source_frame'],
                    annotation_id=obj['annotation_id'],track_id=obj['track_id'],window_start=s,slot=obj['frame']-s,
                    original5=int(s in ORIGINAL),matched=r['matched'],box_max=r['box_max'],
                    peak_x=r['peak_x'],peak_y=r['peak_y'],fused_matched=int(fm),system_case=case))
    assert len({(r['sequence'],r['frame'],r['annotation_id']) for r in output})==len(output)
    assert sum(r['historic1417'] for r in output)==1417 and sum(r['historic97'] for r in output)==97
    strata={'all_valid':lambda r:True,'small_geometry':lambda r:r['small_geometry'],
            'all_valid_multi':lambda r:r['multi_context'],
            'small_geometry_multi':lambda r:r['small_geometry'] and r['multi_context'],
            'historic1417':lambda r:r['historic1417'],'historic97':lambda r:r['historic97']}
    tables=[];bytrack=[];conditions=[]
    for group in sorted({r['group'] for r in output}):
        for scope,test in strata.items():
            rr=[r for r in output if r['group']==group and test(r)]
            if not rr:continue
            tables.append(dict(group=group,sequence='ALL',scope=scope,**summarize(rr)))
            for seq in sorted({r['sequence'] for r in rr}):
                sr=[r for r in rr if r['sequence']==seq]
                tables.append(dict(group=group,sequence=seq,scope=scope,**summarize(sr)))
                for tid in sorted({r['track_id'] for r in sr}):
                    bytrack.append(dict(group=group,sequence=seq,scope=scope,track_id=tid,
                        **summarize([r for r in sr if r['track_id']==tid])))
            if scope in ['all_valid','small_geometry']:
                for field in ['category','window_count','official_window_count']:
                    for value in sorted({r[field] for r in rr}):
                        conditions.append(dict(group=group,scope=scope,condition_field=field,condition_value=value,
                            **summarize([r for r in rr if r[field]==value])))
    write_csv(V/'OBJECTS.csv',output);write_csv(V/'WINDOWS.csv',windows)
    write_csv(V/'TABLE.csv',tables);write_csv(V/'TRACKS.csv',bytrack);write_csv(V/'CONDITIONS.csv',conditions)
    write_csv(V/'REMAINING_OPPORTUNITIES.csv',[r for r in output if r['system_case'] in CASES[2:4]])
    fingerprint(Path(__file__));fingerprint(V/'PROTOCOL.md')
    result=dict(status='COMPLETE',original_checkpoint_only=True,threshold=.5,cases=CASES,
        observations=len(output),window_observations=len(windows),tables=tables,
        interpretation='Additional-window any-match is GT-assisted opportunity, not equal-budget deployable gain; old and additional annotation-selected development groups reported separately',
        sources=SOURCES)
    with (V/'SUMMARY.json').open('x') as f:json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    print(json.dumps([r for r in tables if r['sequence']=='ALL'],ensure_ascii=False))


if __name__=='__main__':main()
