#!/usr/bin/env python3
"""Exhaustive legal fixed-pool five-window actual max-fusion scores, plus all11."""
import csv
import hashlib
import importlib.util
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
C = OUT.parent
Q = C.parent
P = Q / 'stage1_goal'
V = Q / 'problem_validity_20261011'
F = Q / 'failure_attribution_20261011'
SEQS = ['DJI_0026_2', 'DJI_0067_2']
FIELDS = ['gt_count', 'matched_count', 'component_count', 'unmatched_components',
          'positive_pixels', 'positive_pixels_outside_boxes']
SOURCES = {}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8388608), b''):
            h.update(block)
    SOURCES[str(path)] = h.hexdigest()
    return h.hexdigest()


def module(path, name):
    sha(path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def dumpcsv(name, rows):
    with (OUT / name).open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    if (OUT / 'SUMMARY.json').exists():
        raise FileExistsError('Completed scoring is immutable')
    began = time.perf_counter()
    for path in [Path(__file__), C / 'PROTOCOL.md', C / 'PROTOCOL_FREEZE.json', C / 'LEGAL_COMBINATIONS.json']:
        sha(path)
    legal = json.loads((C / 'LEGAL_COMBINATIONS.json').read_text())
    plans = {f'C{i:02d}': starts for i, starts in enumerate(legal['legal_full_coverage_5'])}
    plans['ALL11'] = legal['candidate_starts']
    assert len(plans) == 11
    aliases = {'O': 'C09', 'S': 'C03', 'I1': 'C08', 'I2': 'C06'}
    scorer = module(P / 'pilot/score_pilot.py', 'scorer')
    helper, helper_path = scorer.helper()
    sha(helper_path)
    results, objects, checks, frame_rows, area_hist, track_rows = {}, [], [], [], [], []
    component_rows = []
    for seq in SEQS:
        annpath = P / 'data' / seq / 'annotations.json'
        sha(annpath)
        gt, meta, offsets, excluded = scorer.gt_data(annpath, helper)
        small = np.array([bool(m['strict_small']) for m in meta])
        count = len(meta)
        cache = {}
        for start in legal['candidate_starts']:
            path = P / 'predictions/official' / seq / 'windows' / f'window_{start:03d}.npz'
            sha(path)
            with np.load(path, allow_pickle=False) as saved:
                cache[start] = saved['probabilities']
                assert int(saved['start']) == start
                assert cache[start].shape == (40, 512, 640)
                assert cache[start].dtype == np.float32
                assert np.isfinite(cache[start]).all()
        arrays = dict(
            matched=np.zeros((11, count), np.uint8),
            box_max=np.zeros((11, count), np.float32),
            peak_x=np.full((11, count), np.nan, np.float32),
            peak_y=np.full((11, count), np.nan, np.float32),
            component_area=np.zeros((11, count), np.int32),
            eligible_components=np.zeros((11, count), np.int32),
            frame_counts=np.zeros((11, 160, 6), np.int64),
            coverage=np.zeros((11, 160), np.int16),
            small=small, offsets=offsets,
            frame=np.array([m['frame'] for m in meta], np.int32),
            source_frame=np.array([m['source_frame'] for m in meta], np.int64),
            annotation_id=np.array([m['annotation_id'] for m in meta], np.int64),
            track_id=np.array([m['track_id'] for m in meta], np.int64),
        )
        for index, m in enumerate(meta):
            objects.append(dict(sequence=seq, index=index, frame=m['frame'], source_frame=m['source_frame'],
                                annotation_id=m['annotation_id'], track_id=m['track_id'],
                                category_id=m['category_id'], small_geometry=int(m['strict_small']),
                                bbox=json.dumps(m['bbox'], separators=(',', ':'))))
        previous_path = F / 'one_replacement' / (seq + '_scores.npz')
        sha(previous_path)
        with np.load(previous_path, allow_pickle=False) as saved:
            previous = {key: saved[key] for key in saved.files}
        for pi, (plan, starts) in enumerate(plans.items()):
            prob = np.zeros((160, 512, 640), np.float32)
            for start in starts:
                np.maximum(prob[start:start + 40], cache[start], out=prob[start:start + 40])
                arrays['coverage'][pi, start:start + 40] += 1
            assert np.all(arrays['coverage'][pi] > 0)
            assert arrays['coverage'][pi].sum() == len(starts) * 40
            for frame in range(160):
                rows, stats, components = helper.match_frame(prob[frame], gt[frame], .5)
                assert len(rows) == offsets[frame + 1] - offsets[frame]
                for index, row in enumerate(rows, int(offsets[frame])):
                    arrays['matched'][pi, index] = row['matched']
                    arrays['box_max'][pi, index] = row['box_max']
                    arrays['eligible_components'][pi, index] = row['eligible_components']
                    assert row['annotation_id'] == meta[index]['annotation_id']
                    if row['matched']:
                        for key in ['peak_x', 'peak_y', 'component_area']:
                            arrays[key][pi, index] = row[key]
                arrays['frame_counts'][pi, frame] = [stats[key] for key in FIELDS]
                frame_rows.append(dict(sequence=seq, plan=plan, frame=frame,
                                       coverage=int(arrays['coverage'][pi, frame]), **stats))
                assert sum(c['area'] for c in components) == stats['positive_pixels']
                assert sum(c['unmatched_component'] for c in components) == stats['unmatched_components']
                for component in components:
                    component_rows.append(dict(sequence=seq, plan=plan, frame=frame, **component))
            old_alias = next((name for name, code in aliases.items() if code == plan), None)
            if old_alias is not None:
                old_pi = previous['plans'].tolist().index(old_alias)
                keys = ['matched', 'box_max', 'peak_x', 'peak_y', 'component_area', 'frame_counts', 'coverage']
                for key in keys:
                    assert np.array_equal(arrays[key][pi], previous[key][old_pi], equal_nan=True), (seq, old_alias, key)
                assert np.array_equal(small, previous['small'])
                checks.append(dict(sequence=seq, plan=plan, alias=old_alias,
                                   exact_previous_replacement_array_checks=keys, all_equal=True))
                if old_alias in ['O', 'S']:
                    directory = P / 'evaluation/official' / seq if old_alias == 'O' else V / 'simple_controls/symmetric5_probmax' / seq
                    prefix = 'official_fused' if old_alias == 'O' else 'fused'
                    frozen = directory / (prefix + '_matches.npz')
                    sha(frozen)
                    with np.load(frozen, allow_pickle=False) as saved:
                        ti = saved['thresholds'].tolist().index(.5)
                        for key in ['matched', 'peak_x', 'peak_y', 'component_area', 'frame_counts']:
                            assert np.array_equal(arrays[key][pi], saved[key][ti], equal_nan=True), (seq, old_alias, key)
                        assert np.array_equal(arrays['box_max'][pi], saved['box_max'], equal_nan=True)
                        assert np.array_equal(small, saved['strict_small'])
                    checks[-1]['exact_original_evaluation_arrays_equal'] = True
            print(json.dumps(dict(sequence=seq, plan=plan, starts=starts,
                                  small_matched=int(arrays['matched'][pi, small].sum()),
                                  counts=arrays['frame_counts'][pi].sum(0).tolist(),
                                  elapsed_seconds=round(time.perf_counter() - began, 3))), flush=True)
            del prob
        np.savez_compressed(OUT / (seq + '_scores.npz'), **arrays,
                            plans=np.array(list(plans)), threshold=np.array(.5),
                            frame_count_fields=np.array(FIELDS),
                            plans_json=np.array(json.dumps(plans, separators=(',', ':'))))
        results[seq] = arrays
        del cache
    metrics = []
    for group, seqs in [('pooled', SEQS)] + [(s, [s]) for s in SEQS]:
        for pi, (plan, starts) in enumerate(plans.items()):
            counts = sum((results[s]['frame_counts'][pi].sum(0) for s in seqs), np.zeros(6, np.int64))
            gain = loss = matched = small_n = 0
            for seq in seqs:
                z = results[seq]
                mask = z['small']
                current = z['matched'][pi].astype(bool)
                original = z['matched'][9].astype(bool)
                gain += int((mask & current & ~original).sum())
                loss += int((mask & ~current & original).sum())
                matched += int(current[mask].sum())
                small_n += int(mask.sum())
                for track in np.unique(z['track_id'][mask]):
                    subset = mask & (z['track_id'] == track)
                    if group == seq:
                        track_rows.append(dict(sequence=seq, plan=plan, track_id=int(track), small_gt=int(subset.sum()),
                                               small_matched=int(current[subset].sum()), original_matched=int(original[subset].sum()),
                                               gained=int((subset & current & ~original).sum()),
                                               lost=int((subset & ~current & original).sum())))
            components = [c for c in component_rows if c['sequence'] in seqs and c['plan'] == plan]
            distributions = {}
            for subset_name, subset in [('all', components),
                                        ('assigned', [c for c in components if not c['unmatched_component']]),
                                        ('unmatched', [c for c in components if c['unmatched_component']])]:
                values = np.array([c['area'] for c in subset], np.int64)
                distributions[subset_name + '_component_area_sum'] = int(values.sum())
                for percentile in [50, 90, 95, 99, 100]:
                    distributions[subset_name + f'_component_area_p{percentile}'] = float(np.percentile(values, percentile)) if len(values) else None
                distributions[subset_name + '_component_area_mean'] = float(values.mean()) if len(values) else None
                for area, n in sorted(Counter(values.tolist()).items()):
                    area_hist.append(dict(group=group, plan=plan, subset=subset_name, area=area, component_count=n))
            assert len(components) == counts[2]
            assert distributions['all_component_area_sum'] == counts[4]
            metrics.append(dict(group=group, plan=plan, starts=json.dumps(starts, separators=(',', ':')),
                                windows_per_clip=len(starts), window_frames_per_clip=40 * len(starts),
                                frames=len(seqs) * 160, small_gt=small_n, small_matched=matched,
                                gained=gain, lost=loss, net=gain-loss,
                                **{key: int(value) for key, value in zip(FIELDS, counts)}, **distributions))
    dumpcsv('METRICS.csv', metrics)
    dumpcsv('OBJECTS.csv', objects)
    dumpcsv('FRAME_COUNTS.csv', frame_rows)
    dumpcsv('COMPONENTS.csv', component_rows)
    dumpcsv('COMPONENT_AREA_DISTRIBUTIONS.csv', area_hist)
    dumpcsv('TRACK_METRICS.csv', track_rows)
    summary = dict(status='COMPLETE', threshold=.5, plans=plans, aliases=aliases, sequences=SEQS,
                   all_valid_objects=len(objects), small_objects=sum(o['small_geometry'] for o in objects),
                   legal_5_subsets_scored=10, all11_scored=True, frame_count_fields=FIELDS,
                   all_coverage_checks_passed=True, previous_results_exact_checks=checks,
                   pooled_metrics=[r for r in metrics if r['group'] == 'pooled'],
                   elapsed_seconds=time.perf_counter()-began, source_sha256=SOURCES,
                   limitations=['Cached released-model predictions on two previously used development clips',
                                'Fixed-pool hindsight feasibility; no deployable selector',
                                'Point matching and background diagnostics, not official Pd/Fa',
                                'ALL11 uses 2.2 times the five-window inference budget'])
    (OUT / 'SUMMARY.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(dict(status='COMPLETE', elapsed_seconds=summary['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
