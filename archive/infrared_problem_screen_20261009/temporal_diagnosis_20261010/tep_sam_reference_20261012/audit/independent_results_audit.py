"""Independent std-lib table audit; no experiment helpers or model execution."""
import collections
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics

N = Path(__file__).resolve().parents[1]
E = N / 'evaluation'
CONDITIONS = ['release_O', 'release_S', 'tep_sam', 'tep_auxiliary']
HEADS = ['tep_sam', 'tep_auxiliary']
COHORTS = ['all', 'low_all_05', 'extreme_all_001', 'both_fixed_success']
SEQS = ['DJI_0026_2', 'DJI_0067_2']
hashes, discrepancies, checks = {}, [], collections.Counter()

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def read(path):
    path = Path(path)
    hashes[str(path)] = sha(path)
    with path.open() as f:
        return list(csv.DictReader(f))

def check(group, label, actual, expected):
    checks[group] += 1
    ok = math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-10) if isinstance(expected, float) else actual == expected
    if not ok:
        discrepancies.append({'group': group, 'label': label, 'actual': actual, 'expected': expected})

def fields(group, label, row, expected):
    for key, value in expected.items():
        actual = json.loads(row[key]) if isinstance(value, list) else float(row[key]) if isinstance(value, float) else int(row[key]) if isinstance(value, int) else row[key]
        check(group, label + ':' + key, actual, value)

def selected(rows, cohort):
    return rows if cohort == 'all' else [r for r in rows if r[cohort]]

def matched(row, condition):
    return row[condition + '_matched']

def runs(rows, condition=None):
    """New implementation: scan observations, resetting on either frame gap or match."""
    result, current, previous = [], [], None
    for row in rows:
        if previous is not None and (row['frame'] != previous['frame'] + 1 or row['source_frame'] != previous['source_frame'] + 1):
            if current:
                result.append(current)
                current = []
        if condition is None or not matched(row, condition):
            current.append(row)
        elif current:
            result.append(current)
            current = []
        previous = row
    if current:
        result.append(current)
    return result

def longest(rows, condition):
    return max((len(r) for r in runs(rows, condition)), default=0)

def quantile(values, q):
    v = sorted(values)
    pos = (len(v) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)

objects = read(E / 'OBJECT_COMPARISON.csv')
for row in objects:
    for k in ['frame', 'source_frame', 'track_id', 'annotation_id'] + COHORTS[1:] + [c + '_matched' for c in CONDITIONS]:
        row[k] = int(row[k])
check('object_identity', 'object_count', len(objects), 5200)
check('object_identity', 'unique_object_keys', len({(r['sequence'], r['frame'], r['annotation_id']) for r in objects}), 5200)
check('object_identity', 'tracks', len({(r['sequence'], r['track_id']) for r in objects}), 130)
check('object_identity', 'clips', sorted({r['sequence'] for r in objects}), SEQS)
frozen = read(N.parent / 'low_response_information_20261011/COHORTS.csv')
frozen = {(r['sequence'], int(r['frame']), int(r['annotation_id'])): r for r in frozen}
check('object_identity', 'cohort_key_set', sorted((r['sequence'], r['frame'], r['annotation_id']) for r in objects), sorted(frozen))
for row in objects:
    key = row['sequence'], row['frame'], row['annotation_id']
    for cohort in COHORTS[1:]:
        check('frozen_cohorts', str(key) + ':' + cohort, row[cohort], int(frozen[key][cohort]))
    for c in CONDITIONS:
        check('object_identity', str(key) + ':' + c + '_binary', matched(row, c) in (0, 1), True)
    for c in HEADS:
        for p in ['O', 'S']:
            ref = matched(row, 'release_' + p)
            fields('object_gain_loss', str(key), row, {c + '_gain_vs_' + p: int(bool(matched(row, c) and not ref)), c + '_loss_vs_' + p: int(bool(not matched(row, c) and ref))})

summary_rows = read(E / 'SUMMARY.csv') + read(E / 'SEQUENCE_SUMMARY.csv')
summaries = {(r['condition'], r['group']): r for r in summary_rows}
check('summaries', 'summary_keys', sorted(summaries), sorted((c, g) for c in CONDITIONS for g in ['pooled'] + SEQS))
recomputed = {}
for (c, group), row in summaries.items():
    rr = [r for r in objects if group == 'pooled' or r['sequence'] == group]
    expected = {'target_frames': len(rr), 'tracks': len({(r['sequence'], r['track_id']) for r in rr}), 'clips': len({r['sequence'] for r in rr}), 'M': sum(matched(r, c) for r in rr)}
    for cohort in COHORTS[1:]:
        ss = selected(rr, cohort)
        mm = [r for r in ss if matched(r, c)]
        expected.update({cohort + '_frames': len(ss), cohort + '_still_matched': len(mm), cohort + '_matched_tracks': len({(r['sequence'], r['track_id']) for r in mm}), cohort + '_matched_clips': len({r['sequence'] for r in mm})})
    for p in ['O', 'S']:
        gain = sum(bool(matched(r, c) and not matched(r, 'release_' + p)) for r in rr)
        loss = sum(bool(not matched(r, c) and matched(r, 'release_' + p)) for r in rr)
        expected.update({'G_vs_' + p: gain, 'L_vs_' + p: loss, 'net_M_vs_' + p: gain - loss})
    fields('summaries', c + ':' + group, row, expected)
    recomputed[c + ':' + group] = expected

partitions = collections.defaultdict(list)
for row in objects:
    partitions[row['sequence'], row['track_id']].append(row)
for rows in partitions.values():
    rows.sort(key=lambda r: (r['frame'], r['source_frame']))
expected_tracks = {}
for (seq, track), raw in sorted(partitions.items()):
    for cohort in COHORTS:
        rr = selected(raw, cohort)
        if not rr:
            continue
        for c in CONDITIONS:
            lengths = [len(x) for x in runs(rr, c)]
            expected = {'target_frames': len(rr), 'observation_segments': len(runs(rr)), 'observation_gap_pairs': len(runs(rr)) - 1, 'M': sum(matched(r, c) for r in rr), 'missed_frames': sum(lengths), 'miss_run_count': len(lengths), 'longest_miss_run': max(lengths, default=0), 'miss_run_lengths': lengths}
            for p in ['O', 'S']:
                baseline = 'release_' + p
                expected.update({'longest_miss_delta_vs_' + p: expected['longest_miss_run'] - longest(rr, baseline), 'missed_frames_delta_vs_' + p: expected['missed_frames'] - sum(not matched(r, baseline) for r in rr), 'G_vs_' + p: sum(bool(matched(r, c) and not matched(r, baseline)) for r in rr), 'L_vs_' + p: sum(bool(not matched(r, c) and matched(r, baseline)) for r in rr)})
            expected_tracks[c, seq, track, cohort] = expected
track_rows = read(E / 'TRACK_SUMMARY.csv')
actual_track_keys = [(r['condition'], r['sequence'], int(r['track_id']), r['cohort']) for r in track_rows]
check('tracks', 'track_key_set', sorted(actual_track_keys), sorted(expected_tracks))
for row, key in zip(track_rows, actual_track_keys):
    fields('tracks', str(key), row, expected_tracks[key])

continuous = read(E / 'CONTINUOUS_MISS_SUMMARY.csv')
check('continuous', 'continuous_keys', sorted((r['condition'], r['group'], r['cohort']) for r in continuous), sorted((c, g, h) for c in CONDITIONS for g in ['pooled'] + SEQS for h in COHORTS))
continuous_recomputed = []
for row in continuous:
    c, group, cohort = row['condition'], row['group'], row['cohort']
    entries = [(k, r) for k, r in expected_tracks.items() if k[0] == c and k[3] == cohort and (group == 'pooled' or k[1] == group)]
    rr = [v for k, v in entries]
    lengths = [v['longest_miss_run'] for v in rr]
    expected = {'target_frames': sum(v['target_frames'] for v in rr), 'tracks': len(rr), 'clips': len({k[1] for k, v in entries}), 'missed_frames': sum(v['missed_frames'] for v in rr), 'miss_run_count': sum(v['miss_run_count'] for v in rr), 'longest_miss_track_mean': float(statistics.mean(lengths)), 'longest_miss_track_median': float(statistics.median(lengths)), 'longest_miss_track_q90': quantile(lengths, .9), 'longest_miss_max': max(lengths), 'track_equal_miss_fraction': statistics.mean(v['missed_frames'] / v['target_frames'] for v in rr)}
    for p in ['O', 'S']:
        dd = [v['longest_miss_delta_vs_' + p] for v in rr]
        expected.update({'tracks_shorter_vs_' + p: sum(d < 0 for d in dd), 'tracks_same_vs_' + p: sum(d == 0 for d in dd), 'tracks_longer_vs_' + p: sum(d > 0 for d in dd)})
    fields('continuous', str((c, group, cohort)), row, expected)
    if group == 'pooled':
        continuous_recomputed.append(dict(condition=c, cohort=cohort, **expected))

overlap_rows = read(E / 'AUXILIARY_SUPPORT.csv')
check('overlap', 'overlap_keys', sorted((r['group'], r['cohort']) for r in overlap_rows), sorted((g, h) for g in ['pooled'] + SEQS for h in COHORTS))
overlap = []
for row in overlap_rows:
    rr = selected([r for r in objects if row['group'] == 'pooled' or r['sequence'] == row['group']], row['cohort'])
    count = collections.Counter((matched(r, 'tep_sam'), matched(r, 'tep_auxiliary')) for r in rr)
    expected = dict(frames=len(rr), both_matched=count[1, 1], main_only=count[1, 0], auxiliary_only=count[0, 1], neither=count[0, 0])
    fields('overlap', row['group'] + ':' + row['cohort'], row, expected)
    overlap.append(dict(group=row['group'], cohort=row['cohort'], **expected))

concentration = read(E / 'RECOVERY_CONCENTRATION.csv')
for row in concentration:
    c, outcome = row['condition'], row['outcome']
    if outcome == 'low_recovered':
        rr = [r for r in objects if r['low_all_05'] and matched(r, c)]
    else:
        kind, _, plan = outcome.split('_')
        rr = [r for r in objects if (matched(r, c) and not matched(r, 'release_' + plan))] if kind == 'gain' else [r for r in objects if not matched(r, c) and matched(r, 'release_' + plan)]
    counts = collections.Counter((r['sequence'], r['track_id']) for r in rr)
    ranked = [dict(sequence=s, track_id=t, frames=v) for (s, t), v in sorted(counts.items(), key=lambda x: (-x[1], x[0]))]
    fields('concentration', c + ':' + outcome, row, dict(frames=len(rr), tracks=len(counts), clips=len({r['sequence'] for r in rr}), top1_frames=ranked[0]['frames'] if ranked else 0, top5_frames=sum(r['frames'] for r in ranked[:5]), ranked_tracks=ranked))

release_rows = read(E / 'RELEASE_MISS_SEGMENTS.csv')
expected_release = {}
for (seq, track), raw in sorted(partitions.items()):
    for p in ['O', 'S']:
        reference = 'release_' + p
        for i, rr in enumerate(runs(raw, reference)):
            expected = dict(frame_start=rr[0]['frame'], frame_end=rr[-1]['frame'], source_frame_start=rr[0]['source_frame'], source_frame_end=rr[-1]['source_frame'], target_frames=len(rr), low_all_05_frames=sum(r['low_all_05'] for r in rr))
            for c in HEADS:
                expected.update({c + '_recovered_frames': sum(matched(r, c) for r in rr), c + '_remaining_miss_longest': longest(rr, c), c + '_remaining_miss_runs': len(runs(rr, c))})
            expected_release[reference, seq, track, i] = expected
actual_release_keys = [(r['reference'], r['sequence'], int(r['track_id']), int(r['release_miss_segment'])) for r in release_rows]
check('release_miss_segments', 'release_segment_keys', sorted(actual_release_keys), sorted(expected_release))
for row, key in zip(release_rows, actual_release_keys):
    fields('release_miss_segments', str(key), row, expected_release[key])

background_path = N / 'audit/BACKGROUND_AUDIT.json'
background = json.loads(background_path.read_text())
hashes[str(background_path)] = sha(background_path)
for p, h in background['source_sha256'].items():
    p = str(Path(p) if Path(p).is_absolute() else Path(background['source_root']) / p)
    check('background_input_identity', p, sha(p), h)
    hashes[p] = h
discrepancies.extend(background['discrepancies'])
check('background_status', 'background_audit_status', background['status'], 'PASS')
for p, h in list(hashes.items()):
    check('inputs_unchanged', p, sha(p), h)
hashes[str(Path(__file__))] = sha(__file__)
result = {'status': 'PASS' if not discrepancies else 'FAIL', 'audit_kind': 'Independent CPU/std-lib reaggregation and consecutive-frame scan', 'scope': 'Recomputed from 5200 object rows and per-frame/component tables; no model execution and no imported scoring/segment helpers. This is arithmetic/identity consistency, not re-evaluation of predicted pixels against boxes.', 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'check_count': sum(checks.values()) + background['check_count'], 'object_and_trajectory_check_count': sum(checks.values()), 'checks_by_group': dict(checks), 'background_check_count': background['check_count'], 'discrepancies': discrepancies, 'recomputed_object_summaries': recomputed, 'recomputed_continuity_pooled': continuous_recomputed, 'recomputed_main_auxiliary_overlap': overlap, 'background_audit': background, 'source_sha256': hashes}
output = N / 'audit/INDEPENDENT_RESULTS_AUDIT.json'
with output.open('x') as f:
    json.dump(result, f, indent=2, allow_nan=False)
    f.write('\n')
print(json.dumps({'status': result['status'], 'check_count': result['check_count'], 'discrepancy_count': len(discrepancies), 'discrepancies': discrepancies[:20], 'artifact': str(output)}))
