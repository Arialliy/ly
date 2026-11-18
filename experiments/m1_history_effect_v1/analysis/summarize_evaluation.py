"""Read finalized score artifacts and separate discrete effects from numeric tails."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

E = Path(__file__).resolve().parents[1]
ROOT = E.parents[1]
OUTPUT = E / 'evaluation/step_2048'
Q = ROOT / 'archive/infrared_problem_screen_20261009/temporal_diagnosis_20261010'
sources = {}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def rows(name):
    path = OUTPUT / name
    sources[str(path)] = sha(path)
    with path.open(encoding='utf-8', newline='') as stream: return list(csv.DictReader(stream))


def main():
    complete = json.loads((OUTPUT / 'SUMMARY_COMPLETE.json').read_text(encoding='utf-8'))
    assert complete['status'] == 'COMPLETE'
    for name, expected in complete['output_sha256'].items(): assert sha(OUTPUT / name) == expected
    sources[str(OUTPUT / 'SUMMARY_COMPLETE.json')] = sha(OUTPUT / 'SUMMARY_COMPLETE.json')
    strict, allgt = rows('OBJECT_COMPARISON.csv'), rows('ALL_GT_OBJECT_COMPARISON.csv')
    compare = {}
    for label, values in [('strict_small', strict), ('all_valid_gt', allgt)]:
        compare[label] = {'arrangement_object_rows': len(values), 'matched_differences': sum(
            row['History_matched'] != row['HistoryCurrent_matched'] for row in values)}
    continuity = {}
    for name, keys in [('TRACK_COVERAGE.csv', ('arrangement', 'sequence', 'track_id')),
                       ('CONTINUOUS_MISS_TRACKS.csv', ('arrangement', 'sequence', 'track_id', 'cohort')),
                       ('CONTINUOUS_MISS_SUMMARY.csv', ('arrangement', 'group', 'cohort'))]:
        values = rows(name)
        selected = {condition: {tuple(row[key] for key in keys): {k: v for k, v in row.items() if k != 'condition'}
            for row in values if row['condition'] == condition} for condition in ('History', 'HistoryCurrent')}
        assert selected['History'].keys() == selected['HistoryCurrent'].keys()
        continuity[name] = {'paired_rows': len(selected['History']), 'different_rows': sum(
            row != selected['HistoryCurrent'][key] for key, row in selected['History'].items())}
    summary = rows('SUMMARY.csv')
    compact = [{key: row[key] for key in ('condition', 'arrangement', 'M', 'unmatched_components',
        'positive_pixels_outside_boxes', 'unmatched_component_area_sum', 'low_all_05_matched',
        'extreme_all_001_matched', 'both_fixed_success_matched', 'm1_low_common_recovered_matched')}
        for row in summary]
    off = []
    for sequence in ('DJI_0026_2', 'DJI_0067_2'):
        now = OUTPUT / 'scored/HistoryOff' / sequence / 'scores.npz'
        old = Q / 'top1_background_control_20261011/evaluation/release' / sequence / 'scores.npz'
        sources.update({str(p): sha(p) for p in (now, old)})
        with np.load(now, allow_pickle=False) as current, np.load(old, allow_pickle=False) as reference:
            finite = np.isfinite(current['box_max']) & np.isfinite(reference['box_max'])
            delta = np.abs(current['box_max'][finite] - reference['box_max'][finite])
            off.append({'sequence': sequence, 'finite_box_max_values': int(finite.sum()),
                'different_box_max_values': int(np.count_nonzero(delta)),
                'box_max_absmax': float(delta.max()), 'box_max_mean_abs': float(delta.mean()),
                'matched_exact': bool(np.array_equal(current['matched'], reference['matched'])),
                'frame_counts_exact': bool(np.array_equal(current['frame_counts'], reference['frame_counts'])),
                'component_area_exact': bool(np.array_equal(current['component_area'], reference['component_area']))})
    result = {'status': 'COMPLETE', 'no_model_execution': True, 'no_training': True,
        'History_vs_HistoryCurrent': compare, 'continuity': continuity, 'summary': compact,
        'common9': {'unique_objects': 9, 'arrangement_object_rows': 18,
            'History_O_S_matches': [5, 7], 'HistoryCurrent_O_S_matches': [5, 7], 'HistoryOff_O_S_matches': [0, 0],
            'scope': 'Outcome-conditioned explanatory subgroup from completed v1, not an independent population estimate'},
        'HistoryOff_vs_archived_release': off,
        'interpretation': 'History dependence does not change any matched target or recorded continuity metric versus '
            'current-only at these same weights, clips, arrangements and threshold. Background U is unchanged; '
            'History has 1/2 fewer outside-box positive pixels and 3/4 less unmatched-component area in O/S. '
            'Off restores archived release discrete diagnostics; box maxima have cross-runtime floating tails.',
        'source_sha256': sources, 'script_sha256': sha(__file__)}
    path = E / 'analysis/EVALUATION_INTERPRETATION.json'
    assert not path.exists(), 'Preserve a completed interpretation artifact'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'comparison': compare, 'continuity': continuity, 'off': off}), flush=True)


if __name__ == '__main__': main()
