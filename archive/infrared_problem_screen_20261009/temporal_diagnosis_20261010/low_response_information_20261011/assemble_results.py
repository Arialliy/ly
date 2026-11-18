#!/usr/bin/env python3
"""Assemble immutable numeric diagnostics and finite manual annotations.

No threshold, cohort, fusion, matching or numerical score is changed here.
"""
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent
GROUPS = ['all', 'low_all_05', 'extreme_all_001', 'low_nonextreme',
          'both_fixed_success', 'extra_only', 'high_unmatched']


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def write(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def key(r):
    return r['sequence'], int(r['track_id']), int(r['frame']), r['arrangement']


def flag(r, k):
    return r[k] == '1'


def quantile(vals, q):
    vals = sorted(vals)
    if not vals:
        return ''
    pos = (len(vals) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def members(rows):
    return len(rows), len({(r['sequence'], r['track_id']) for r in rows}), len({r['sequence'] for r in rows})


def summarize(rows):
    out = []
    for arrangement in ['O', 'S']:
        for seq in ['pooled', 'DJI_0026_2', 'DJI_0067_2']:
            for group in GROUPS:
                rr = [r for r in rows if r['arrangement'] == arrangement and
                      (seq == 'pooled' or r['sequence'] == seq) and
                      (group == 'all' or flag(r, group))]
                n, nt, nc = members(rr)
                row = dict(arrangement=arrangement, sequence=seq, group=group,
                           target_frames=n, tracks=nt, clips=nc)
                conditions = {
                    'matched_at_05': lambda r: flag(r, 'matched_at_05'),
                    'peak_inside_own_box': lambda r: flag(r, 'neighborhood_peak_inside_target'),
                    'peak_in_other_annotated_box': lambda r: bool(r['neighborhood_peak_other_gt_ids']),
                    'peak_outside_all_annotated_boxes': lambda r: not flag(r, 'neighborhood_peak_inside_target') and not r['neighborhood_peak_other_gt_ids'],
                    'neighborhood_probability_ge05': lambda r: float(r['neighborhood_probability_max']) >= .5,
                    'neighborhood_logit_ge0': lambda r: float(r['neighborhood_peak_logit']) >= 0,
                    'inside_peak_ge05': lambda r: flag(r, 'neighborhood_peak_inside_target') and float(r['neighborhood_peak_probability']) >= .5,
                    'outside_peak_ge05': lambda r: not flag(r, 'neighborhood_peak_inside_target') and float(r['neighborhood_peak_probability']) >= .5,
                    'nearly_flat': lambda r: flag(r, 'neighborhood_nearly_flat'),
                    'border_clipped': lambda r: flag(r, 'neighborhood_border_clipped'),
                    'target_logit_probability_argmax_differs': lambda r: not flag(r, 'target_logit_and_probability_peak_same'),
                    'neighborhood_logit_probability_argmax_differs': lambda r: not flag(r, 'neighborhood_logit_and_probability_peak_same'),
                    'visual_reviewed': lambda r: flag(r, 'visual_reviewed'),
                    'visual_suspected_target': lambda r: r['visual_peak_label'] == 'suspected_target',
                    'visual_background_or_other': lambda r: r['visual_peak_label'] == 'background_or_other',
                    'visual_uncertain': lambda r: r['visual_peak_label'] == 'uncertain',
                    'visual_suspected_target_outside': lambda r: r['visual_peak_label'] == 'suspected_target' and not flag(r, 'neighborhood_peak_inside_target'),
                    'visual_suspected_target_ge05_unmatched': lambda r: r['visual_peak_label'] == 'suspected_target' and float(r['neighborhood_peak_probability']) >= .5 and not flag(r, 'matched_at_05'),
                }
                for name, pred in conditions.items():
                    subset = [r for r in rr if pred(r)]
                    count, tracks, clips = members(subset)
                    row[name + '_frames'] = count
                    row[name + '_tracks'] = tracks
                    row[name + '_clips'] = clips
                for col in ['target_box_max_logit', 'neighborhood_peak_logit',
                            'neighborhood_peak_distance_to_box', 'neighborhood_peak_distance_normalized',
                            'neighborhood_max_minus_median', 'neighborhood_max_minus_q95']:
                    vals = [float(r[col]) for r in rr]
                    for label, q in [('q25', .25), ('median', .5), ('q75', .75), ('q95', .95)]:
                        row[col + '_' + label] = quantile(vals, q)
                assert row['peak_inside_own_box_frames'] + row['peak_in_other_annotated_box_frames'] + row['peak_outside_all_annotated_boxes_frames'] == n
                assert row['visual_suspected_target_frames'] + row['visual_background_or_other_frames'] + row['visual_uncertain_frames'] == row['visual_reviewed_frames']
                out.append(row)
    return out


def main():
    rows = read(BASE / 'analysis/LOW_RESPONSE_OBJECTS.csv')
    assert len(rows) == 10400 and len({key(r) for r in rows}) == 10400
    # A normalized, unique-object table is written from the manual reviewer table;
    # neighboring frames retain their own frozen cohort, never the anchor's cohort.
    labels_path = BASE / 'visual/PEAK_REVIEW.csv'
    assert labels_path.exists(), 'Manual peak review must be complete before final assembly'
    labels = read(labels_path)
    assert len({key(r) for r in labels}) == len(labels)
    review = {key(r): r for r in labels}
    assert set(review) <= {key(r) for r in rows}
    for row in rows:
        label = review.get(key(row))
        row['visual_reviewed'] = str(int(label is not None))
        row['visual_peak_label'] = {'T': 'suspected_target', 'B': 'background_or_other', 'U': 'uncertain'}[label['review_label_code']] if label else ''
        row['visual_case_ids'] = label['case_ids'] if label else ''
        row['visual_peak_evidence'] = label['peak_observation'] if label else ''
        row['visual_raw_readability'] = label.get('raw_readability', '') if label else ''
        if label:
            assert row['visual_peak_label'] in ['suspected_target', 'background_or_other', 'uncertain']
    write(BASE / 'LOW_RESPONSE_OBJECTS.csv', rows)
    shutil.copyfile(BASE / 'analysis/SCORE_SEPARATION.csv', BASE / 'SCORE_SEPARATION.csv')
    peaks = summarize(rows)
    write(BASE / 'PEAK_LOCATION.csv', peaks)
    counts = {}
    for arrangement in ['O', 'S']:
        rr = [r for r in rows if r['arrangement'] == arrangement]
        counts[arrangement] = {g: sum(g == 'all' or flag(r, g) for r in rr) for g in GROUPS}
        counts[arrangement]['matched_at_05'] = sum(flag(r, 'matched_at_05') for r in rr)
        assert [counts[arrangement][g] for g in GROUPS] == [5200, 4342, 4170, 172, 646, 104, 49]
    assert counts['O']['matched_at_05'] == 700 and counts['S']['matched_at_05'] == 720
    validation = dict(status='PASS', detail_rows=len(rows), numeric_columns_preserved=True,
                      peak_summary_rows=len(peaks), manual_rows=len(labels), counts=counts)
    validation['source_sha256'] = {str(p.relative_to(BASE)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [BASE/'analysis/LOW_RESPONSE_OBJECTS.csv', labels_path, Path(__file__)]}
    (BASE/'ASSEMBLY_VALIDATION.json').write_text(json.dumps(validation, indent=2)+'\n')
    (BASE/'PEAK_COLUMNS.md').write_text('''# Table 2 and manual columns

Numeric table uses all frozen objects. Peak inside/outside columns describe coordinates only; an unannotated location is not a proven background object. Neighborhood max is not the actual component output peak. No foreground is added and detection remains at 0.5.

Visual labels apply only to the 72 presampled unique target frames (144 arrangement rows); all other rows stay unreviewed. Labels: suspected_target / background_or_other / uncertain. Numerators and frame/track/clip denominators are separate. Grouping uses each frame's true frozen flags, not the sample anchor group. No population rate is inferred from this finite visual set.

Distances are to the original half-open box; center is not a true target center. The logit and probability argmax difference is a numerical tie/saturation diagnostic; it does not by itself explain a detection failure.
''')
    print(json.dumps(validation, indent=2))


if __name__ == '__main__':
    main()
