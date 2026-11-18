#!/usr/bin/env python3
"""Independent CPU-only localization of existing temporal response instability."""
import collections
import csv
import hashlib
import itertools
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
INPUTS = {}

def read(p):
    INPUTS[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return list(csv.DictReader(p.open()))

def write(name, rows):
    if not rows:
        return
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def key(r):
    return r['sequence'], int(r['local_frame']), int(r['track_id'])

def nt(rows):
    return len({(r['sequence'], r['track_id']) for r in rows})

choices = read(ROOT / 'result_stratification/context/same_gt_frame_choices.csv')
strict = [r for r in choices if float(r['threshold']) == .5 and
          int(r['strict']) and int(r['multi']) and float(r['max_side']) <= 16]
strong = [r for r in strict if int(r['strong_response_flip'])]
assert len(strict) == 1417 and len(strong) == 97 and nt(strong) == 18
contexts = collections.defaultdict(list)
for s in sorted({r['sequence'] for r in strict}):
    for r in read(ROOT / f'window_offset_experiment/analysis/{s}/per_context_all_tracks.csv'):
        if float(r['threshold']) == .5:
            contexts[key(r)].append(r)
covariates = {key(r): r for r in read(ROOT / 'result_stratification/response/JOINED_TRACK_FRAME.csv')
              if float(r['threshold']) == .5}

interior = []
for margin in [0, 4, 8, 12]:
    comparable = []
    retained = []
    for r in strict:
        rr = [v for v in contexts[key(r)] if margin <= int(v['slot']) < 40 - margin]
        if len(rr) < 2:
            continue
        comparable.append(r)
        vals = [float(v['box_max']) for v in rr]
        if min(vals) < .1 and max(vals) >= .9:
            retained.append(r)
    for s in ['ALL'] + sorted({r['sequence'] for r in strict}):
        dd = [r for r in comparable if s == 'ALL' or r['sequence'] == s]
        zz = [r for r in retained if s == 'ALL' or r['sequence'] == s]
        interior.append(dict(margin=margin, slot_min=margin, slot_max=39-margin,
                             sequence=s, comparable_target_frames=len(dd),
                             strong_flips=len(zz), strong_flip_tracks=nt(zz)))
write('interior_restriction.csv', interior)

by_track = collections.defaultdict(list)
for r in strong:
    by_track[(r['sequence'], r['track_id'])].append(r)
episodes = []
track_summary = []
for (s, tid), rr in sorted(by_track.items()):
    fs = sorted(int(r['local_frame']) for r in rr)
    starts = [0] + [i for i in range(1, len(fs)) if fs[i] != fs[i-1] + 1]
    ends = starts[1:] + [len(fs)]
    for a, b in zip(starts, ends):
        episodes.append(dict(sequence=s, track_id=tid, category=rr[0]['category'],
                             frame_first=fs[a], frame_last=fs[b-1], length=b-a))
    track_summary.append(dict(sequence=s, track_id=tid,
                              source_tracking_id=rr[0]['source_tracking_id'],
                              category=rr[0]['category'], strong_flip_frames=len(rr),
                              contiguous_runs=len(starts), longest_run=max(b-a for a,b in zip(starts,ends))))
write('strong_flip_tracks.csv', track_summary)
write('strong_flip_runs.csv', episodes)

strong_pairs = []
for r in strong:
    for a, b in itertools.combinations(sorted(contexts[key(r)], key=lambda x:int(x['window_start'])), 2):
        pa, pb = float(a['box_max']), float(b['box_max'])
        if not ((pa < .1 and pb >= .9) or (pb < .1 and pa >= .9)):
            continue
        win, lose = (a,b) if pa > pb else (b,a)
        cv = covariates[key(r)]
        strong_pairs.append(dict(sequence=r['sequence'], local_frame=r['local_frame'],
            source_frame=r['source_frame'], track_id=r['track_id'], category=r['category'],
            a_start=int(a['window_start']), b_start=int(b['window_start']),
            a_slot=int(a['slot']), b_slot=int(b['slot']), a_boxmax=pa, b_boxmax=pb,
            a_matched=int(a['matched']), b_matched=int(b['matched']),
            winning_start=int(win['window_start']), losing_start=int(lose['window_start']),
            winning_slot=int(win['slot']), losing_slot=int(lose['slot']),
            winning_more_central=int(abs(int(win['slot'])-19.5) < abs(int(lose['slot'])-19.5)),
            losing_more_central=int(abs(int(win['slot'])-19.5) > abs(int(lose['slot'])-19.5)),
            both_interior4=int(4<=int(a['slot'])<=35 and 4<=int(b['slot'])<=35),
            original_point=int(r['original_point']), original_boxmax=float(r['original_boxmax']),
            max_side=float(r['max_side']), bbox_x=float(a['bbox_x']), bbox_y=float(a['bbox_y']),
            bbox_w=float(a['bbox_w']), bbox_h=float(a['bbox_h']),
            motion_valid=cv['motion_valid'], adjusted_speed=cv['adjusted_speed_px_per_frame'],
            signed_contrast=cv['signed_mean_contrast'], abs_standardized_contrast=cv['absolute_standardized_mean_contrast']))
write('all_strong_window_pairs.csv', strong_pairs)

# Simultaneous opposite directions refute a uniform slot/window benefit.
by_pair = collections.defaultdict(list)
for r in strong_pairs:
    by_pair[(r['sequence'],r['local_frame'],r['a_start'],r['b_start'])].append(r)
opposite = []
for k, rr in sorted(by_pair.items()):
    ups = [r for r in rr if r['a_boxmax'] < .1]
    downs = [r for r in rr if r['b_boxmax'] < .1]
    if ups and downs:
        for r in rr:
            opposite.append(dict(pair_key=':'.join(map(str,k)), direction='gain' if r['a_boxmax'] < .1 else 'loss', **r))
write('same_frame_opposite_direction.csv', opposite)

# Latest/earliest point-match flips are a separate population from arbitrary-window strong response flips.
pointflips = [r for r in strict if r['earliest_point'] != r['latest_point']]
overlap = set(map(key, pointflips)) & set(map(key, strong))
gap = collections.Counter((r['sequence'], r['category']) for r in strict if int(r['all_contexts_boxmax_low']))
write('persistent_low_by_category.csv', [dict(sequence=s,category=c,target_frames=n) for (s,c),n in sorted(gap.items())])
summary = dict(
    scope='Existing saved predictions only; diagnostic box maxima and point matches, not paper Pd/Fa.',
    strict_small_target_frames=len(strict), strong_flip_target_frames=len(strong), strong_flip_track_ids=nt(strong),
    strong_flip_contiguous_runs=len(episodes), longest_strong_flip_run=max(r['length'] for r in episodes),
    largest_track_strong_flips=max(r['strong_flip_frames'] for r in track_summary),
    top_three_track_strong_flips=sum(sorted([r['strong_flip_frames'] for r in track_summary],reverse=True)[:3]),
    arbitrary_window_strong_pairs=len(strong_pairs),
    strong_pairs_success_more_central=sum(r['winning_more_central'] for r in strong_pairs),
    strong_pairs_failure_more_central=sum(r['losing_more_central'] for r in strong_pairs),
    strong_pairs_success_later_start=sum(r['winning_start']>r['losing_start'] for r in strong_pairs),
    strong_pairs_success_earlier_start=sum(r['winning_start']<r['losing_start'] for r in strong_pairs),
    same_image_window_pairs_opposite_strong_directions=len({r['pair_key'] for r in opposite}),
    same_image_window_pairs_opposite_strong_directions_both_interior4=len({r['pair_key'] for r in opposite if r['both_interior4']}),
    earliest_latest_point_flip_frames=len(pointflips),
    earliest_latest_point_flip_overlap_with_any_window_strong_response=len(overlap),
    strong_flip_frames_already_original_point_matched=sum(int(r['original_point']) for r in strong),
    strong_flip_frames_original_point_unmatched=sum(not int(r['original_point']) for r in strong),
    strong_flip_frames_original_unmatched_any_context_point_matched=sum(
        not int(r['original_point']) and any(int(v['matched']) for v in contexts[key(r)]) for r in strong),
    window_start_stride=12,
    temporal_phase_limitation='All starts differ by multiples of 12. Comparisons cannot test phase effects whose period divides 12.',
    uncertainty='Exploratory associations. Window content and current-frame position are jointly changed. Correlated target-frames and selected clips preclude population estimates or causal claims.')
(OUT/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
(OUT/'INPUT_SHA256.json').write_text(json.dumps(INPUTS,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2))
