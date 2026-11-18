#!/usr/bin/env python3
"""Predefined sequence-conditioned descriptive summaries; no predictor fitting."""
import collections
import csv
import hashlib
import json
from pathlib import Path
from build_conditions import qstats, csvread, csvwrite, js, SEQS, PRIMARY

D=Path(__file__).resolve().parent
rows=csvread(D/'OBJECT_CONDITIONS.csv')
groups=(*PRIMARY,'both_failure_with_window_opportunity')
continuous_fields=['max_side','nearest_other_annotation_center_px','frame_valid_annotation_count',
                   'official_support_center_step_median','official_support_center_step_p90',
                   'official_window_annotation_presence_fraction_min','symmetric_window_annotation_presence_fraction_min',
                   'symmetric_minus_official_latest_access_frames']
levels={'category':('car','cyclist','pedestrian','ship'),'scale_bin':('(0,4]','(4,8]','(8,16]'),
        'frame_position_bin':('0..39','40..119','120..159')}
continuous=[]; strata=[]; access=[]
for seq in ('ALL',*SEQS):
    population=[r for r in rows if seq=='ALL' or r['sequence']==seq]
    for group in ('all_small',*groups):
        rr=population if group=='all_small' else [r for r in population if r['group']==group]
        for field in continuous_fields:
            continuous.append(dict(sequence=seq,group=group,condition=field,**qstats([float(r[field]) for r in rr if r[field]!=''])))
    for field,levs in levels.items():
        for level in levs:
            eligible=[r for r in population if r[field]==level]
            for group in groups:
                rr=[r for r in eligible if r['group']==group]
                strata.append(dict(sequence=seq,condition=field,level=level,group=group,object_frames=len(rr),
                                   condition_denominator=len(eligible),fraction_of_condition=len(rr)/len(eligible) if eligible else '',
                                   sequence_tracks=len({(r['sequence'],r['track_id']) for r in rr})))
    for group in groups:
        rr=[r for r in population if r['group']==group]
        for change in ('earlier','same','later'):
            cc=[r for r in rr if ('earlier' if int(r['symmetric_minus_official_latest_access_frames'])<0 else 'later' if int(r['symmetric_minus_official_latest_access_frames'])>0 else 'same')==change]
            access.append(dict(sequence=seq,group=group,latest_access_change=change,object_frames=len(cc),group_denominator=len(rr)))
csvwrite(D/'SEQUENCE_CONTINUOUS_CONDITIONS.csv',continuous)
csvwrite(D/'CONDITION_STRATA.csv',strata)
csvwrite(D/'LATEST_ACCESS_CHANGE.csv',access)
paired=[]; controls=csvread(D/'STABLE_CONTROLS.csv')
fields=('max_side','official_support_center_step_median','official_support_center_step_p90','nearest_other_annotation_center_px')
for seq in ('ALL',*SEQS):
    for group in ('gain','loss'):
        cc=[r for r in controls if r['group']==group and (seq=='ALL' or r['sequence']==seq)]
        matched=[r for r in cc if r['matched']=='1']
        for field in fields:
            values=[float(r['case_'+field])-float(r['control_'+field]) for r in matched]
            paired.append(dict(sequence=seq,group=group,condition=field,cases=len(cc),matched_cases=len(matched),
                               difference_definition='case minus reused response-defined stable control',**qstats(values)))
csvwrite(D/'STABLE_PAIRED_DIFFERENCES.csv',paired)
tracks=csvread(D/'TRACK_CONDITIONS.csv')
concentrations=[]
for group in groups:
    rr=sorted([r for r in tracks if r['group']==group],key=lambda r:(-int(r['object_frames']),r['sequence'],int(r['track_id'])))
    total=sum(int(r['object_frames']) for r in rr)
    concentrations.append(dict(group=group,object_frames=total,tracks=len(rr),top1_frames=int(rr[0]['object_frames']),
                               top1_fraction=int(rr[0]['object_frames'])/total,top3_frames=sum(int(r['object_frames']) for r in rr[:3]),
                               top3_fraction=sum(int(r['object_frames']) for r in rr[:3])/total,
                               top3_tracks=[dict(sequence=r['sequence'],track_id=int(r['track_id']),object_frames=int(r['object_frames'])) for r in rr[:3]]))
js(D/'TRACK_CONCENTRATION.json',concentrations)
print(json.dumps(concentrations,ensure_ascii=False))
