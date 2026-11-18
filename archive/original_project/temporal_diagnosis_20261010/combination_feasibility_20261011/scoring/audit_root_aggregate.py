#!/usr/bin/env python3
"""Independently reconstruct root aggregate tables from scores and per-clip CSV."""
import csv
import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent
C = OUT.parent
SEQS = ['DJI_0026_2', 'DJI_0067_2']
FIELDS = ['gt_count', 'matched_count', 'component_count', 'unmatched_components', 'positive_pixels', 'positive_pixels_outside_boxes']
checks = []


def read(name):
    return list(csv.DictReader((C / name).open()))


def pair(row):
    return row['choice_0026'], row['choice_0067']


def flag(value):
    assert value in ['True', 'False']
    return value == 'True'


with (OUT / 'METRICS.csv').open() as handle:
    metrics = {(r['group'], r['plan']): r for r in csv.DictReader(handle)}
arrays = {}
for seq in SEQS:
    with np.load(OUT / (seq + '_scores.npz'), allow_pickle=False) as z:
        arrays[seq] = {k: z[k] for k in ['matched', 'small', 'plans']}
legal = json.loads((C / 'LEGAL_COMBINATIONS.json').read_text())['legal_full_coverage_5']
shared, pairs = read('SHARED_COMBINATIONS.csv'), read('PER_CLIP_COMBINATIONS.csv')
assert len(shared) == 10 and len(pairs) == 100
assert {pair(r) for r in pairs} == set(itertools.product([f'C{i:02d}' for i in range(10)], repeat=2))
assert {pair(r) for r in shared} == {(f'C{i:02d}', f'C{i:02d}') for i in range(10)}
for row in shared + pairs + read('ALL11_REFERENCE.csv'):
    choices = pair(row)
    for key in ['small_gt', 'small_matched', 'gained', 'lost', 'net'] + FIELDS:
        expected = sum(int(metrics[seq, plan][key]) for seq, plan in zip(SEQS, choices))
        assert int(row[key]) == expected, (choices, key)
    assert flag(row['U_budget_feasible']) == (int(row['unmatched_components']) <= 4015)
    assert flag(row['both_sequence_U_within_original']) == all(int(metrics[s, p]['unmatched_components']) <= b for s, p, b in zip(SEQS, choices, [741, 3274]))
    assert abs(float(row['match_percent']) - int(row['small_matched']) / 52) < 1e-12
    assert abs(float(row['delta_percentage_points']) - int(row['net']) / 52) < 1e-12
    union_retained = sym_gain_retained = beyond = 0
    for seq, plan, starts_key in zip(SEQS, choices, ['starts_0026', 'starts_0067']):
        a = arrays[seq]
        idx = a['plans'].tolist().index(plan)
        expected_starts = legal[idx] if idx < 10 else list(range(0, 121, 12))
        assert json.loads(row[starts_key]) == expected_starts
        small = a['small']
        original = a['matched'][9, small].astype(bool)
        symmetric = a['matched'][3, small].astype(bool)
        current = a['matched'][idx, small].astype(bool)
        union_retained += np.count_nonzero(current & (original | symmetric))
        sym_gain_retained += np.count_nonzero(current & symmetric & ~original)
        beyond += np.count_nonzero(current & ~(original | symmetric))
    assert int(row['original_sym_union_retained']) == union_retained
    assert int(row['symmetric_gains_retained']) == sym_gain_retained
    assert int(row['beyond_original_sym_union']) == beyond
checks.append('All 10 shared and 100 per-clip assignments uniquely exhaustive; ALL11 remains separate')
checks.append('All counts, gain/loss, union support, percentages, starts, and budget flags independently reproduced')


def rank(row):
    return (-int(row['small_matched']), int(row['lost']), int(row['unmatched_components']),
            int(row['positive_pixels_outside_boxes']), tuple(json.loads(row['starts_0026'])), tuple(json.loads(row['starts_0067'])))


selected = read('TRADEOFF_SELECTIONS.csv')
assert len(selected) == 16
assert {(r['scope'], r['require_U_budget'], r['loss_limit']) for r in selected} == set(itertools.product(['shared', 'per_clip_hindsight'], ['True', 'False'], ['0', '7', '35', 'unlimited']))
for selection in selected:
    pool = shared if selection['scope'] == 'shared' else pairs
    loss_limit = None if selection['loss_limit'] == 'unlimited' else int(selection['loss_limit'])
    eligible = [r for r in pool if (not flag(selection['require_U_budget']) or int(r['unmatched_components']) <= 4015)
                and (loss_limit is None or int(r['lost']) <= loss_limit)]
    expected = sorted(eligible, key=rank)[0]
    assert int(selection['feasible_count']) == len(eligible)
    for key, value in expected.items():
        assert selection[key] == value, (key, selection[key], value)
checks.append('All 16 frozen budget/loss-limit selections and feasible counts reproduced using actual-start lexicographic ties')
front = read('PARETO_GAIN_LOSS_U.csv')
for scope, rows in [('shared', shared), ('per_clip_hindsight', pairs)]:
    expected = set()
    for row in rows:
        value = (-int(row['gained']), int(row['lost']), int(row['unmatched_components']))
        alternatives = [(-int(other['gained']), int(other['lost']), int(other['unmatched_components'])) for other in rows]
        dominated = any(all(a <= b for a, b in zip(other, value)) and other != value for other in alternatives)
        if not dominated:
            expected.add(pair(row))
    assert {pair(r) for r in front if r['scope'] == scope} == expected
checks.append('Gain/loss/U Pareto fronts exactly reproduced; budget is not replaced by outside-box pixel cost')
expected_by_pair = {pair(r): r for r in pairs + read('ALL11_REFERENCE.csv')}
for name, field in [('SELECTED_OBJECT_RESULTS.csv', 'original_matched'), ('SELECTED_TRACK_RESULTS.csv', 'original_matched')]:
    sums = defaultdict(lambda: defaultdict(int))
    count = defaultdict(int)
    for row in read(name):
        key = pair(row)
        count[key] += 1
        for source, target in [('matched', 'small_matched'), ('gained', 'gained'), ('lost', 'lost')]:
            sums[key][target] += int(row[source])
        sums[key]['original_matched'] += int(row[field])
    for key, values in sums.items():
        for metric in ['small_matched', 'gained', 'lost']:
            assert values[metric] == int(expected_by_pair[key][metric])
        assert values['original_matched'] == 700
        if name == 'SELECTED_OBJECT_RESULTS.csv':
            assert count[key] == 5200
checks.append('Selected object and track tables reproduce aggregate matches/gains/losses and original 700 successes')
audited_files = ['aggregate_combinations.py', 'SHARED_COMBINATIONS.csv', 'PER_CLIP_COMBINATIONS.csv', 'ALL11_REFERENCE.csv',
                 'TRADEOFF_SELECTIONS.csv', 'PARETO_GAIN_LOSS_U.csv', 'SELECTED_OBJECT_RESULTS.csv', 'SELECTED_TRACK_RESULTS.csv']
result = dict(status='PASS', checks=checks, issues=[],
              source_code_review='Budget remains pooled U<=4015, each selected plan is whole-clip and costs five windows, annotations only used for hindsight scoring/selection, ranking matches preregistered rules.',
              input_probabilities_finite=dict(checked_in='score_combinations.py per-window loading assertions', arrays=22,
                                             values_per_array=40*512*640, dtype='float32', all_passed=True),
              audited_sha256={name: hashlib.sha256((C / name).read_bytes()).hexdigest() for name in audited_files})
(OUT / 'ROOT_AGGREGATE_AUDIT.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
