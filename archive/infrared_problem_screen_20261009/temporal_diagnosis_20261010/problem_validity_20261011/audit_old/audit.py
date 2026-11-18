#!/usr/bin/env python3
"""Independent CPU-only reaggregation of original saved point matches."""
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
Q = OUT.parent.parent
SEQS = ['DJI_0028_2', 'DJI_0028_4', 'DJI_0032_5', 'DJI_0047_3']
OFFICIAL = {0, 36, 72, 108, 120}
INPUTS = {}


def read_csv(path):
    INPUTS[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open() as f:
        return list(csv.DictReader(f))


def read_json(path):
    INPUTS[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())


def key(r):
    return r['sequence'], int(r['local_frame']), int(r['annotation_id'])


def bbox(r):
    return [float(r['bbox_'+s]) for s in ('x', 'y', 'w', 'h')]


def write_csv(name, rows):
    with (OUT/name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def aggregate(rows):
    d = {'object_frames': len(rows),
         'processed_track_ids': len({(r['sequence'], r['track_id']) for r in rows}),
         'source_category_tracking_groups': len({(r['sequence'], r['category_id'], r['source_tracking_id']) for r in rows})}
    for any_window in (0, 1):
        for fused in (0, 1):
            d[f'any{any_window}_fused{fused}'] = sum(r['any_window_matched'] == any_window and r['fused_matched'] == fused for r in rows)
    for col in ['fused_matched', 'any_window_matched', 'any_official_window_matched',
                'loss_despite_official_window_match', 'opportunity_only_extra_window',
                'unmatched_all_windows_but_boxmax_ge_half', 'unmatched_all_windows_but_boxmax_ge_nine',
                'strong_response_flip', 'fused_ambiguous']:
        d[col] = sum(r[col] for r in rows)
    assert d['any1_fused0'] == d['loss_despite_official_window_match'] + d['opportunity_only_extra_window']
    assert sum(d[f'any{a}_fused{b}'] for a in (0, 1) for b in (0, 1)) == len(rows)
    for col in ['loss_despite_official_window_match', 'opportunity_only_extra_window']:
        d[col+'_track_ids'] = len({(r['sequence'], r['track_id']) for r in rows if r[col]})
    return d


def main():
    candidates = read_json(Q.parent/'real_motion_stage2/INSPECTION_QUEUE.json')['candidates'][:4]
    candidates = {r['sequence']: r for r in candidates}
    records = []
    for seq in SEQS:
        orig = [r for r in read_csv(Q/'analysis'/seq/'per_track_frame.csv') if float(r['threshold']) == .5]
        contexts = [r for r in read_csv(Q/'window_offset_experiment/analysis'/seq/'per_context_all_tracks.csv') if float(r['threshold']) == .5]
        annpath = (Q.parent/'real_motion_stage2/selected_raw/annotations.json' if seq == 'DJI_0028_2'
                   else Q/'additional_clips'/candidates[seq]['candidate_id']/'annotations.json')
        annotations = read_json(annpath)
        byid = {r['id']: r for r in annotations['annotations']}
        images = {r['id']: r for r in annotations['images']}
        sorted_images = sorted(images.values(), key=lambda r: int(Path(r['file_name']).stem))
        cats = {r['id']: r['name'] for r in annotations['categories']}
        baseline = {key(r): r for r in orig}
        assert len(baseline) == len(orig)
        groups = defaultdict(list)
        for r in contexts:
            groups[key(r)].append(r)
        assert set(groups) == set(baseline)
        assert {int(r['window_start']) for r in contexts} == set(range(0, 121, 12))
        for k, rr in sorted(groups.items()):
            b = baseline[k]
            annotation = byid[k[2]]
            im = images[annotation['image_id']]
            assert sorted_images[k[1]]['id'] == im['id']
            assert int(Path(im['file_name']).stem) == int(b['source_frame'])
            assert int(annotation['track_id']) == int(b['track_id'])
            assert [float(x) for x in annotation['bbox']] == bbox(b)
            assert not annotation.get('ignore', 0) and annotation['bbox'][2] > 0 and annotation['bbox'][3] > 0
            assert len({int(r['window_start']) for r in rr}) == len(rr)
            expected = [s for s in range(0, 121, 12) if s <= k[1] < s+40]
            assert sorted(int(r['window_start']) for r in rr) == expected
            for r in rr:
                assert key(r) == k and r['sequence'] == seq
                assert int(r['track_id']) == int(b['track_id'])
                assert int(r['source_frame']) == int(b['source_frame'])
                assert bbox(r) == bbox(b)
                assert r['partly_outside'] == b['partly_outside']
                assert r['label_overlap_count'] == b['label_overlap_count']
                assert int(r['slot']) == k[1] - int(r['window_start'])
                assert int(r['window_end']) == int(r['window_start']) + 40
                assert (r['window_source'] == 'original') == (int(r['window_start']) in OFFICIAL)
            any_match = any(int(r['matched']) for r in rr)
            official_match = any(int(r['matched']) for r in rr if int(r['window_start']) in OFFICIAL)
            fused = int(b['matched'])
            small_clean = max(bbox(b)[2:]) <= 16 and not int(b['partly_outside']) and not int(b['label_overlap_count'])
            multi_unambiguous = len(rr) >= 2 and not any(int(r['association_ambiguous']) for r in rr)
            values = [float(r['box_max']) for r in rr]
            strong = min(values) < .1 and max(values) >= .9
            records.append(dict(sequence=seq, local_frame=k[1], source_frame=int(b['source_frame']),
                annotation_id=k[2], image_id=annotation['image_id'], track_id=int(b['track_id']),
                source_tracking_id=annotation.get('tracking_id', ''), category_id=annotation['category_id'],
                category=cats[annotation['category_id']], bbox_x=bbox(b)[0], bbox_y=bbox(b)[1],
                bbox_w=bbox(b)[2], bbox_h=bbox(b)[3], context_count=len(rr),
                strict_small=int(small_clean), legacy_1417=int(small_clean and multi_unambiguous),
                legacy_97=int(small_clean and multi_unambiguous and strong),
                strong_response_flip=int(strong), any_window_matched=int(any_match),
                any_official_window_matched=int(official_match), fused_matched=fused,
                loss_despite_official_window_match=int(official_match and not fused),
                opportunity_only_extra_window=int(any_match and not official_match and not fused),
                any_ambiguous=int(any(int(r['association_ambiguous']) for r in rr)),
                fused_ambiguous=int(b['association_ambiguous']),
                context_boxmax_min=min(values), context_boxmax_max=max(values),
                fused_boxmax=float(b['box_max']),
                unmatched_all_windows_but_boxmax_ge_half=int(not any_match and max(values) >= .5),
                unmatched_all_windows_but_boxmax_ge_nine=int(not any_match and max(values) >= .9),
                matched_window_starts=';'.join(str(int(r['window_start'])) for r in rr if int(r['matched'])),
                matched_window_peaks=';'.join(f"{r['window_start']}:{r['peak_x']}:{r['peak_y']}:{r['component_area']}" for r in rr if int(r['matched']))))

    # Reference cohort keys only: no statistics are imported from existing summary.
    prior = [r for r in read_csv(Q/'result_stratification/context/same_gt_frame_choices.csv') if float(r['threshold']) == .5]
    prior_1417 = {key(r) for r in prior if int(r['multi']) and int(r['strict']) and float(r['max_side']) <= 16}
    prior_97 = {key(r) for r in read_csv(Q/'result_stratification/context/strong_flips_strict_half_le16.csv')}
    assert {key(r) for r in records if r['legacy_1417']} == prior_1417
    assert {key(r) for r in records if r['legacy_97']} == prior_97
    assert len(prior_1417) == 1417 and len(prior_97) == 97
    cohorts = {'all_valid': records, 'strict_small': [r for r in records if r['strict_small']],
               'legacy_1417': [r for r in records if r['legacy_1417']],
               'legacy_97': [r for r in records if r['legacy_97']]}
    grouped = []
    tracks = []
    for name, rows in cohorts.items():
        for seq in ['ALL']+SEQS:
            rr = rows if seq == 'ALL' else [r for r in rows if r['sequence'] == seq]
            grouped.append(dict(cohort=name, sequence=seq, **aggregate(rr)))
        bytrack = defaultdict(list)
        for r in rows:
            bytrack[(r['sequence'], r['track_id'])].append(r)
        for (seq, track), rr in sorted(bytrack.items()):
            tracks.append(dict(cohort=name, sequence=seq, track_id=track,
                               source_tracking_id=rr[0]['source_tracking_id'], category=rr[0]['category'], **aggregate(rr)))
    for p, digest in INPUTS.items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest, p
    write_csv('objects.csv', records)
    write_csv('by_cohort_sequence.csv', grouped)
    write_csv('by_cohort_track.csv', tracks)
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), threshold=.5,
        method='Independent aggregation of saved original-checkpoint connected-component point matches; no new inference or GT-assisted deployment.',
        official_starts=sorted(OFFICIAL), all_starts=list(range(0, 121, 12)),
        cohort_definitions={'all_valid': 'All valid ground-truth rows from original analyzer; retains partly-outside and overlapping labels.',
        'strict_small': 'max bbox side <=16, wholly in-frame, no overlap with other valid GT; all available context counts; no matching-ambiguity exclusion.',
        'legacy_1417': 'strict_small, >=2 contexts, no association ambiguity in any individual context at .5.',
        'legacy_97': 'legacy_1417 and min box maximum <.1 and max box maximum >=.9.'},
        integrity_checks={'sequence_frame_annotation_key': 'PASS', 'annotation_source_image_track_bbox_alignment': 'PASS',
        'context_exact_starts_slots_source_types': 'PASS', 'legacy_cohort_keys_match': 'PASS',
        'input_hashes_unchanged': 'PASS'}, inputs_sha256=INPUTS,
        summary=[r for r in grouped if r['sequence']=='ALL'], by_sequence=grouped)
    (OUT/'SUMMARY.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result['summary'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
