#!/usr/bin/env python3
"""Read-only reaggregation of already scored combinations; no inference or fusion."""
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
SOURCE = OUT.parent.parent / 'combination_feasibility_20261011'
SEQS = ['DJI_0026_2', 'DJI_0067_2']
FILES = ['PER_CLIP_COMBINATIONS.csv', 'SELECTED_TRACK_RESULTS.csv',
         'SELECTED_OBJECT_RESULTS.csv', 'scoring/METRICS.csv',
         'scoring/COMPONENT_AREA_DISTRIBUTIONS.csv', 'scoring/OBJECTS.csv',
         'scoring/TRACK_METRICS.csv'] + [f'scoring/{s}_scores.npz' for s in SEQS]

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def read(name):
    with (SOURCE / name).open(newline='') as f:
        return list(csv.DictReader(f))

def write(name, rows):
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)

hash_before = {n: sha(SOURCE/n) for n in FILES}
combos = read('PER_CLIP_COMBINATIONS.csv')
metrics = {(r['group'], r['plan']): r for r in read('scoring/METRICS.csv')}
dists = defaultdict(Counter)
for r in read('scoring/COMPONENT_AREA_DISTRIBUTIONS.csv'):
    dists[r['group'], r['plan'], r['subset']][int(r['area'])] += int(r['component_count'])

def quantile(values, q):
    position = (len(values)-1)*q
    a, b = math.floor(position), math.ceil(position)
    return values[a]*(b-position)+values[b]*(position-a) if a != b else float(values[a])

def stats(counter):
    values = sorted(counter.elements())
    return dict(count=len(values), area_sum=sum(values), mean=sum(values)/len(values),
                **{f'p{p}': quantile(values, p/100) for p in [50,90,95,99,100]})

distribution_checks = 0
for (seq, plan, subset), counter in dists.items():
    s = stats(counter)
    m = metrics[seq, plan]
    pref = {'all':'all','assigned':'assigned','unmatched':'unmatched'}[subset] + '_component_area_'
    for suffix, key in [('sum','area_sum'),('mean','mean')]+[(f'p{p}',f'p{p}') for p in [50,90,95,99,100]]:
        assert math.isclose(s[key], float(m[pref+suffix]), abs_tol=1e-9)
        distribution_checks += 1

high = [r for r in combos if int(r['small_matched']) >= 720 and int(r['unmatched_components']) <= 4015]
zero = [r for r in combos if int(r['lost']) == 0 and int(r['unmatched_components']) <= 4015]
assert len(high) == 3
assert max(int(r['small_matched']) for r in zero) == 707
selected = [('C09','C09'),('C03','C03'),('C06','C03'),('C08','C03'),('C08','C09')]
combo_by_choice = {(r['choice_0026'],r['choice_0067']): r for r in combos}
background, per_clip = [], []
for c0,c1 in selected:
    r = combo_by_choice[c0,c1]
    row = {k:r[k] for k in ['choice_0026','choice_0067','small_matched','gained','lost','matched_count',
                           'component_count','unmatched_components','positive_pixels','positive_pixels_outside_boxes']}
    for subset in ['all','assigned','unmatched']:
        s = stats(dists[SEQS[0],c0,subset] + dists[SEQS[1],c1,subset])
        for k,v in s.items():
            row[f'{subset}_{k}'] = v
    assert row['unmatched_count'] == int(row['unmatched_components'])
    assert row['all_area_sum'] == int(row['positive_pixels'])
    background.append(row)
    for seq, choice in zip(SEQS,[c0,c1]):
        m=metrics[seq,choice]
        q={'choice_0026':c0,'choice_0067':c1,'sequence':seq,'plan':choice}
        for k in ['small_matched','gained','lost','matched_count','component_count','unmatched_components',
                  'positive_pixels','positive_pixels_outside_boxes','unmatched_component_area_sum',
                  'unmatched_component_area_mean','unmatched_component_area_p95']:
            q[k] = m[k]
        per_clip.append(q)
write('BACKGROUND_COSTS.csv', background)
write('PER_CLIP_COSTS.csv', per_clip)

meta = defaultdict(list)
for r in read('scoring/OBJECTS.csv'):
    meta[r['sequence']].append(r)
raw, diffs = {}, []
for seq in SEQS:
    arr = np.load(SOURCE/f'scoring/{seq}_scores.npz',allow_pickle=False)
    plan_names = [str(p) for p in arr['plans']]
    raw[seq] = {p:arr['matched'][i].astype(bool) for i,p in enumerate(plan_names)}
    assert len(meta[seq]) == len(raw[seq]['C03'])
    assert all(int(r['index'])==i for i,r in enumerate(meta[seq]))
    for c0,c1 in selected:
        choice = c0 if seq==SEQS[0] else c1
        a,b,o=raw[seq][choice],raw[seq]['C03'],raw[seq]['C09']
        for i in np.flatnonzero(a != b):
            r = meta[seq][i]
            if int(r['small_geometry']) != 1:
                continue
            diffs.append(dict(choice_0026=c0,choice_0067=c1,sequence=seq,frame=int(r['frame']),
                source_frame=int(r['source_frame']),annotation_id=r['annotation_id'],track_id=r['track_id'],
                original_matched=int(o[i]),symmetric_matched=int(b[i]),choice_matched=int(a[i]),
                relation='recovers_S_loss' if a[i] and o[i] else 'new_vs_S' if a[i]
                else 'sacrifices_S_gain' if not o[i] else 'loses_original_success'))
write('OBJECT_DIFFERENCES_VS_S.csv', diffs)

selected_object_checks=0
# Metadata indices are joined by the same unique annotation IDs, not CSV row order.
index_map={s:{r['annotation_id']:i for i,r in enumerate(meta[s])} for s in SEQS}
for r in read('SELECTED_OBJECT_RESULTS.csv'):
    seq=r['sequence'];choice=r['choice_0026'] if seq==SEQS[0] else r['choice_0067']
    assert int(raw[seq][choice][index_map[seq][r['annotation_id']]]) == int(r['matched'])
    selected_object_checks+=1

zero_gains=[]
for seq,choice in zip(SEQS,['C08','C09']):
    a,o=raw[seq][choice],raw[seq]['C09']
    for i in np.flatnonzero(a & ~o):
        r=meta[seq][i]
        if int(r['small_geometry'])==1:
            zero_gains.append({k:r[k] for k in ['sequence','frame','source_frame','annotation_id','track_id']})
write('ZERO_LOSS_GAINS.csv',zero_gains)
zero_tracks=Counter((r['sequence'],r['track_id']) for r in zero_gains)
high_diffs=[r for r in diffs if (r['choice_0026'],r['choice_0067'])==('C08','C03')]
assert len(high_diffs)==3
assert {r['track_id'] for r in high_diffs} == {'400008'}
small26=np.array([int(r['small_geometry'])==1 for r in meta[SEQS[0]]])
c06_same = bool(np.array_equal(raw[SEQS[0]]['C06'][small26],raw[SEQS[0]]['C03'][small26]))
assert c06_same
assert sorted(zero_tracks.values()) == [1,6]

tracks=defaultdict(lambda:Counter())
for r in diffs:
    k=r['choice_0026'],r['choice_0067'],r['sequence'],r['track_id']
    tracks[k][r['relation']]+=1
track_rows=[]
for k,v in sorted(tracks.items()):
    track_rows.append(dict(zip(['choice_0026','choice_0067','sequence','track_id'],k)) |
                      {n:v[n] for n in ['recovers_S_loss','new_vs_S','sacrifices_S_gain','loses_original_success']})
write('TRACK_DIFFERENCES_VS_S.csv',track_rows)

hash_after = {n:sha(SOURCE/n) for n in FILES}
assert hash_before==hash_after
(OUT/'SOURCE_SHA256.json').write_text(json.dumps({'source_root':str(SOURCE),'sha256':hash_before},indent=2)+'\n')
summary={'scope':'existing-output reaggregation only; threshold 0.5; no inference, fusion, or threshold search',
 'high_feasible_combinations':high,'zero_loss_combinations':zero,
 'C06_C03_same_small_match_set_as_S':c06_same,
 'best_high_differences_vs_S':high_diffs,
 'zero_loss_gain_tracks':[{"sequence":s,"track_id":t,"count":n} for (s,t),n in zero_tracks.items()],
 'background_costs':background,'distribution_metrics_crosschecks':distribution_checks,
 'selected_object_npz_checks':selected_object_checks,'sources_unchanged':hash_before==hash_after}
(OUT/'SUMMARY.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ['high_feasible_combinations','zero_loss_combinations','background_costs']},indent=2,ensure_ascii=False))
