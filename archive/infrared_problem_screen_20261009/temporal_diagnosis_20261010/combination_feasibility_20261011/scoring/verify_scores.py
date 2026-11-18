#!/usr/bin/env python3
"""Independent output accounting, geometry, and array invariant checks."""
import csv
import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent
summary = json.loads((OUT / 'SUMMARY.json').read_text())
legal = json.loads((OUT.parent / 'LEGAL_COMBINATIONS.json').read_text())
checks = []
geometric = []
for starts in itertools.combinations(range(0, 121, 12), 5):
    covered = set().union(*(set(range(s, s + 40)) for s in starts))
    if covered == set(range(160)):
        geometric.append(list(starts))
assert geometric == legal['legal_full_coverage_5']
checks.append('Independently enumerated all 462 subsets; exactly the registered 10 cover all 160 frames')
metrics = {(r['group'], r['plan']): r for r in csv.DictReader((OUT / 'METRICS.csv').open())}
components = defaultdict(lambda: [0, 0, 0, 0])
for r in csv.DictReader((OUT / 'COMPONENTS.csv').open()):
    value = components[r['sequence'], r['plan']]
    area = int(r['area'])
    unmatched = int(r['unmatched_component'])
    value[0] += 1
    value[1] += unmatched
    value[2] += area
    value[3] += area * unmatched
objects = list(csv.DictReader((OUT / 'OBJECTS.csv').open()))
assert len(objects) == 16297
assert sum(int(r['small_geometry']) for r in objects) == 5200
assert len({(r['sequence'], r['frame'], r['annotation_id']) for r in objects}) == len(objects)
checks.append('All 16297 valid object-frame identities unique; exactly 5200 geometric small object-frames')
small_peak_equivalence = {}
for seq in summary['sequences']:
    with np.load(OUT / (seq + '_scores.npz'), allow_pickle=False) as z:
        assert z['matched'].shape == (11, int(z['offsets'][-1]))
        assert z['frame_counts'].shape == (11, 160, 6)
        assert np.all(z['box_max'][10] >= z['box_max'][:10])
        assert np.all(z['frame_counts'][10, :, 4] >= z['frame_counts'][:10, :, 4])
        assert np.all(z['matched'].astype(bool) <= (z['eligible_components'] > 0))
        assert np.array_equal(z['matched'][:, z['small']].astype(bool), z['eligible_components'][:, z['small']] > 0)
        small_peak_equivalence[seq] = True
        original = z['matched'][9].astype(bool)
        for pi, plan in enumerate(z['plans'].tolist()):
            current = z['matched'][pi].astype(bool)
            mask = z['small']
            row = metrics[seq, plan]
            assert int(current[mask].sum()) == int(row['small_matched'])
            assert int((mask & current & ~original).sum()) == int(row['gained'])
            assert int((mask & ~current & original).sum()) == int(row['lost'])
            for fi, key in enumerate(summary['frame_count_fields']):
                assert int(z['frame_counts'][pi, :, fi].sum()) == int(row[key])
            assert components[seq, plan] == [int(row[key]) for key in ['component_count', 'unmatched_components', 'positive_pixels', 'unmatched_component_area_sum']]
checks.append('Every per-sequence CSV metric independently reproduced from NPZ arrays and component rows')
checks.append('All small-object matching bits exactly equivalent to existence of at least one eligible component peak in every scored combination')
checks.append('All11 box maxima and foreground pixel counts dominate every five-window combination; match monotonicity not assumed')
for plan in summary['plans']:
    pooled = metrics['pooled', plan]
    for key in ['small_gt', 'small_matched', 'gained', 'lost', 'net'] + summary['frame_count_fields']:
        assert int(pooled[key]) == sum(int(metrics[seq, plan][key]) for seq in summary['sequences'])
checks.append('Pooled counts equal the sum of the two complete per-clip plans')
for path, expected in summary['source_sha256'].items():
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8388608), b''):
            digest.update(block)
    assert digest.hexdigest() == expected, path
checks.append('Every registered source hash unchanged after scoring')
result = dict(status='PASS', checks=checks, small_peak_equivalence=small_peak_equivalence,
              exact_historical_plan_array_checks=len(summary['previous_results_exact_checks']),
              total_components=sum(c[0] for c in components.values()))
(OUT / 'VALIDATION.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
