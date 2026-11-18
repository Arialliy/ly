#!/usr/bin/env python3
"""Reuse frozen .5 component matching for E2 residual-off only; on is cached."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
Q=ROOT.parent
OLD=Q/'mutual_temporal_pilot_20261012'
SCORER=Q/'top1_background_control_20261011/evaluation_code/evaluate.py'
SEQUENCES=['DJI_0026_2','DJI_0067_2']
ARRANGEMENTS=['O','S']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def jsonout(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def csvout(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def score(sequence):
    freeze=json.loads((ROOT/'A_FREEZE.json').read_text())
    for name,expected in freeze['external_sha256'].items():
        if sha(name)!=expected:raise AssertionError('Frozen A source changed: '+name)
    receipt_path=ROOT/'predictions/E2_off'/sequence/'COMPLETE.json'
    receipt=json.loads(receipt_path.read_text())
    if receipt['status']!='COMPLETE' or receipt['condition']!='E2_off' or receipt['model_state_sha256']!=receipt['unchanged_final_model_state_sha256']:
        raise AssertionError('Off inference identity or unchanged-state check failed')
    old=json.loads((OLD/'predictions/E2'/sequence/'COMPLETE.json').read_text())
    if receipt['checkpoint_sha256']!=old['checkpoint_sha256'] or not all(r['old_on_probabilities_bitwise_identical'] for r in receipt['on_cache_checks']):
        raise AssertionError('On/off do not share the old E2 trained state and verified on outputs')
    spec=importlib.util.spec_from_file_location('unchanged_point_match_scorer',SCORER);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.E=ROOT/'A_evaluation';module.ROOTS['E2_off']=ROOT/'predictions/E2_off'
    module.score('E2_off',sequence)
    destination=ROOT/'A_evaluation/E2_off'/sequence/'COMPLETE.json'
    scored=json.loads(destination.read_text());scored.update(onoff_wrapper_sha256=sha(__file__),
        onoff_source_freeze_sha256=sha(ROOT/'A_FREEZE.json'),same_E2_checkpoint_sha256=receipt['checkpoint_sha256'])
    jsonout(destination,scored)


def summary():
    cohorts={}
    with (Q/'low_response_information_20261011/COHORTS.csv').open() as stream:
        for r in csv.DictReader(stream):cohorts[r['sequence'],int(r['frame']),int(r['annotation_id'])]=r
    arrays={};areas={};metadata={};objects=[];sources={}
    for sequence in SEQUENCES:
        for name,folder in [('on',OLD/'evaluation/E2'/sequence),('off',ROOT/'A_evaluation/E2_off'/sequence)]:
            completion=json.loads((folder/'COMPLETE.json').read_text())
            if sequence in metadata and completion['objects']!=metadata[sequence]:raise AssertionError('GT ordering changed')
            metadata[sequence]=completion['objects']
            with np.load(folder/'scores.npz') as z:arrays[name,sequence]={k:z[k] for k in z.files}
            with np.load(folder/'component_areas.npz') as z:areas[name,sequence]={k:z[k] for k in z.files}
            sources[str(folder/'COMPLETE.json')]=sha(folder/'COMPLETE.json')
            sources[str(folder/'scores.npz')]=sha(folder/'scores.npz')
        for index,meta in enumerate(metadata[sequence]):
            if not meta['strict_small']:continue
            cohort=cohorts[sequence,meta['frame'],meta['annotation_id']]
            for pi,arrangement in enumerate(ARRANGEMENTS):
                on=int(arrays['on',sequence]['matched'][pi,index]);off=int(arrays['off',sequence]['matched'][pi,index])
                r={k:meta[k] for k in ['frame','source_frame','track_id','annotation_id']}
                r.update(sequence=sequence,arrangement=arrangement,on_matched=on,off_matched=off,on_gain_vs_off=int(on and not off),on_loss_vs_off=int(off and not on))
                r.update({k:int(cohort[k]) for k in ['low_all_05','extreme_all_001','both_fixed_success','extra_only']})
                objects.append(r)
    if len(objects)!=10400:raise AssertionError('Strict-small universe changed')
    csvout(ROOT/'A_DETECTION_OBJECTS.csv',objects)
    rows=[]
    for group,sequences in [('pooled',SEQUENCES)]+[(x,[x]) for x in SEQUENCES]:
        for pi,arrangement in enumerate(ARRANGEMENTS):
            rr=[r for r in objects if r['sequence'] in sequences and r['arrangement']==arrangement]
            row={'group':group,'arrangement':arrangement,'target_frames':len(rr),'tracks':len({(r['sequence'],r['track_id']) for r in rr}),'clips':len(sequences),
                 'M_on':sum(r['on_matched'] for r in rr),'M_off':sum(r['off_matched'] for r in rr),
                 'G_on_vs_off':sum(r['on_gain_vs_off'] for r in rr),'L_on_vs_off':sum(r['on_loss_vs_off'] for r in rr),
                 'net_on_vs_off':sum(r['on_matched']-r['off_matched'] for r in rr),'U_budget':4015 if group=='pooled' else ''}
            for name in ['on','off']:
                counts=sum((arrays[name,s]['frame_counts'][pi].sum(0) for s in sequences),np.zeros(6,np.int64))
                row.update({f'{name}_{k}':int(counts[j]) for j,k in enumerate(['all_gt','all_gt_matched','components','U','foreground_pixels','A'])})
                row[name+'_unmatched_component_area']=sum(int(areas[name,s][arrangement+'_areas'][areas[name,s][arrangement+'_unmatched'].astype(bool)].sum()) for s in sequences)
                for flag in ['low_all_05','extreme_all_001','both_fixed_success']:
                    selected=[r for r in rr if r[flag] and r[name+'_matched']]
                    row[name+'_'+flag+'_matched']=len(selected)
                    row[name+'_'+flag+'_matched_tracks']=len({(r['sequence'],r['track_id']) for r in selected})
            row.update(delta_U_on_minus_off=row['on_U']-row['off_U'],delta_A_on_minus_off=row['on_A']-row['off_A'])
            rows.append(row)
    csvout(ROOT/'A_DETECTION_SUMMARY.csv',rows)
    lookup={(r['sequence'],r['annotation_id'],r['arrangement']):r for r in objects}
    protected=[r for r in objects if r['arrangement']=='O' and r['both_fixed_success']]
    protection=[]
    for name in ['on','off']:
        counts={'both_success':0,'O_only':0,'S_only':0,'neither':0}
        for r in protected:
            o=r[name+'_matched'];s=lookup[r['sequence'],r['annotation_id'],'S'][name+'_matched']
            key='both_success' if o and s else 'O_only' if o else 'S_only' if s else 'neither';counts[key]+=1
        protection.append(dict(condition='E2_'+name,protected_target_frames=len(protected),**counts))
    csvout(ROOT/'A_PROTECTION.csv',protection)
    jsonout(ROOT/'A_SCORE_COMPLETE.json',{'status':'COMPLETE','script_sha256':sha(__file__),'object_rows':len(objects),
        'sources_sha256':sources,'tables_sha256':{p.name:sha(p) for p in [ROOT/'A_DETECTION_OBJECTS.csv',ROOT/'A_DETECTION_SUMMARY.csv',ROOT/'A_PROTECTION.csv']},
        'scorer_unchanged_sha256':sha(SCORER),'scope':'Existing .5 one-to-one component peak matching; U not official Fa; on scores reused after exact probability agreement'})
    print(json.dumps({'status':'COMPLETE','summary_rows':len(rows),'pooled':[r for r in rows if r['group']=='pooled']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--score',choices=SEQUENCES);parser.add_argument('--summary',action='store_true');args=parser.parse_args()
    if args.score:score(args.score)
    elif args.summary:summary()
    else:parser.error('Choose --score sequence or --summary')
